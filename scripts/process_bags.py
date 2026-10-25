#!/usr/bin/env python3
"""Read ROS2 MCAP without ROS; preserve raw clocks and compute actual link FK."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import shutil

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image, ImageDraw
from rosbags.highlevel import AnyReader

from urdf_fk import ForwardKinematics, quaternion


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def clock_stats(rows):
    if not rows:
        return {}
    h = np.array([r['stamp_ns'] for r in rows], dtype=np.int64)
    b = np.array([r['bag_ns'] for r in rows], dtype=np.int64)
    return {'count': len(rows), 'zero_header_stamps': int((h <= 0).sum()),
            'nonincreasing_header_stamps': int((np.diff(h) <= 0).sum()),
            'max_header_gap_ms': float(np.diff(h).max()/1e6) if len(h)>1 else None,
            'bag_minus_header_ms_quantiles': np.quantile((b-h)/1e6, [0, .5, .95, 1]).tolist()}


def process(bag, output, fk):
    dest = output/bag.name
    dest.mkdir()
    joints, images, wrist_images, suction, errors, names_seen = [], [], [], [], [], set()
    previews = []
    previous_q = None
    with AnyReader([bag]) as reader:
        available = {c.topic for c in reader.connections}
        image_topic = next((t for t in ('/global_camera/image_raw/compressed',
                                      '/camera/image_raw/compressed') if t in available), None)
        if image_topic is None:
            raise ValueError(f'No supported front camera topic in {bag}: {sorted(available)}')
        wrist_topic = '/wrist_camera/image_raw/compressed'
        if wrist_topic not in available:
            raise ValueError(f'Missing wrist camera topic in {bag}')
        wrist_size = None
        start, end = int(reader.start_time), int(reader.end_time)
        targets = [start+(end-start)*i//5 for i in range(6)]
        topics = [{'topic': c.topic, 'type': c.msgtype, 'count': c.msgcount} for c in reader.connections]
        for connection, timestamp, raw in reader.messages():
            msg = reader.deserialize(raw, connection.msgtype)
            if connection.topic == '/joint_states':
                names_seen.add(tuple(msg.name))
                if len(msg.name) != len(msg.position) or len(set(msg.name)) != len(msg.name):
                    errors.append({'bag_ns': timestamp, 'reason': 'invalid joint names/positions'})
                    continue
                positions = dict(zip(msg.name, msg.position))
                if set(fk.joint_names)-positions.keys() or not all(np.isfinite(positions[n]) for n in fk.joint_names):
                    errors.append({'bag_ns': timestamp, 'reason': 'missing/nonfinite position'})
                    continue
                transform = fk.compute(positions)
                q = quaternion(transform)
                if previous_q is not None and np.dot(previous_q, q) < 0:
                    q = [-v for v in q]
                previous_q = q
                joints.append({'bag_ns': timestamp, 'stamp_ns': msg.header.stamp.sec*10**9+msg.header.stamp.nanosec,
                               'position': [float(positions[n]) for n in fk.joint_names],
                               'ee_position': [transform[i][3] for i in range(3)], 'ee_quaternion_xyzw': q,
                               'velocity_present': len(msg.velocity) == len(msg.name),
                               'effort_finite': bool(len(msg.effort) == len(msg.name) and np.isfinite(msg.effort).all())})
            elif connection.topic == image_topic:
                images.append({'bag_ns': timestamp, 'stamp_ns': msg.header.stamp.sec*10**9+msg.header.stamp.nanosec,
                               'format': msg.format, 'bytes': len(msg.data)})
                if len(previews) < 6 and timestamp >= targets[len(previews)]:
                    frame = Image.open(io.BytesIO(bytes(msg.data))).convert('RGB')
                    size = frame.size
                    frame.thumbnail((320, 240))
                    previews.append((frame.copy(), (timestamp-start)/1e9))
            elif connection.topic == wrist_topic:
                wrist_images.append({'bag_ns': timestamp, 'stamp_ns': msg.header.stamp.sec*10**9+msg.header.stamp.nanosec,
                                     'format': msg.format, 'bytes': len(msg.data)})
                if wrist_size is None:
                    with Image.open(io.BytesIO(bytes(msg.data))) as frame:
                        wrist_size = list(frame.size)
            elif connection.topic == '/suction/commanded_enabled':
                suction.append({'bag_ns': timestamp, 'enabled': bool(msg.data)})
    if not joints or not images or not wrist_images:
        raise ValueError(f'Missing joint/image data: {bag}')
    for name, rows in [('joints', joints), ('images', images), ('images_wrist', wrist_images), ('suction', suction)]:
        if rows:
            pq.write_table(pa.Table.from_pylist(rows), dest/f'{name}.parquet', compression='zstd')
    sheet = Image.new('RGB', (960, 540), 'white')
    draw = ImageDraw.Draw(sheet)
    for i, (frame, sec) in enumerate(previews):
        x, y = i%3*320, i//3*270
        sheet.paste(frame, (x, y))
        draw.text((x+5, y+242), f'{bag.name} +{sec:.1f}s', fill='black')
    sheet.save(dest/'preview.jpg')
    jp = np.asarray([j['position'] for j in joints])
    ep = np.asarray([j['ee_position'] for j in joints])
    report = {'bag': str(bag.resolve()), 'start_bag_ns': start, 'end_bag_ns': end,
              'duration_seconds': (end-start)/1e9, 'topics': topics,
              'joint_clock': clock_stats(joints), 'image_clock': clock_stats(images),
              'image_topic': image_topic,
              'cameras': {
                  'front': {'topic': image_topic, 'table': 'images.parquet', 'clock': clock_stats(images),
                            'size_width_height': list(size)},
                  'wrist': {'topic': wrist_topic, 'table': 'images_wrist.parquet', 'clock': clock_stats(wrist_images),
                            'size_width_height': wrist_size}},
              'image_size_width_height': list(size), 'joint_names_seen': [list(x) for x in sorted(names_seen)],
              'joint_min_rad': jp.min(axis=0).tolist(), 'joint_max_rad': jp.max(axis=0).tolist(),
              'ee_min_m': ep.min(axis=0).tolist(), 'ee_max_m': ep.max(axis=0).tolist(),
              'invalid_joint_samples': errors, 'suction_events': suction,
              'suction_initial_state_known': bool(suction and suction[0]['bag_ns'] <= images[0]['bag_ns']),
              'first_suction_event_seconds_from_start': (suction[0]['bag_ns']-start)/1e9 if suction else None,
              'pose_reference': 'base_link', 'pose_tip': 'link6',
              'label_semantics': 'measured joint feedback and FK estimate; not controller target',
              'episode_boundaries_confirmed': False}
    write_json(dest/'report.json', report)
    print(json.dumps({'bag': bag.name, 'joints': len(joints), 'images': len(images), 'wrist_images': len(wrist_images),
                      'suction': len(suction), 'initial_suction_known': report['suction_initial_state_known']}, ensure_ascii=False), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, default=Path('datas'))
    parser.add_argument('--output', type=Path, default=Path('processed_dual'))
    parser.add_argument('--urdf', type=Path, default=Path(__file__).resolve().parents[1]/'episode1_urdf_1113.urdf')
    args = parser.parse_args()
    bags = sorted(p.parent for p in args.input.glob('*/metadata.yaml'))
    if not bags:
        raise ValueError('No ROS bags found')
    if not args.urdf.is_file():
        parser.error(f'URDF not found: {args.urdf}. Pass --urdf /path/to/robot.urdf')
    fk = ForwardKinematics(args.urdf, 'base_link', 'link6')
    if args.output.exists() and any(args.output.iterdir()):
        parser.error(f'Output directory is not empty: {args.output}. Choose a new --output path.')
    args.output.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.urdf, args.output/'robot.urdf')
    reports = [process(bag, args.output, fk) for bag in bags]
    write_json(args.output/'summary.json', {'urdf_sha256': hashlib.sha256(args.urdf.read_bytes()).hexdigest(),
               'joint_order': fk.joint_names, 'camera_names': ['front', 'wrist'], 'reports': reports})


if __name__ == '__main__':
    main()
