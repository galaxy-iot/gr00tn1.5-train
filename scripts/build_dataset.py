#!/usr/bin/env python3
"""Build pinned GR00T/LeRobot v2 data, with explicit draft/label provenance."""
import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
import io
import json
from pathlib import Path
import shutil

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image
from rosbags.highlevel import AnyReader

from process_bags import write_json
from task_labels import task_text, validate_tasks
from urdf_fk import ForwardKinematics

CAMERAS = ('front', 'wrist')


def causal_indices(times, targets):
    return np.searchsorted(times, targets, side='right')-1


def stats(array):
    # Accumulate in float64: long float32 trajectories otherwise lose enough
    # precision for normalization metadata to disagree with the stored data.
    a = np.asarray(array, dtype=np.float64)
    return {'mean': a.mean(0).tolist(), 'std': a.std(0).tolist(),
            'min': a.min(0).tolist(), 'max': a.max(0).tolist(),
            'q01': np.quantile(a, .01, axis=0).tolist(), 'q99': np.quantile(a, .99, axis=0).tolist()}


def encode_video(bag, indices, path, fps, size, image_topic='/camera/image_raw/compressed', encoder_threads=2):
    path.parent.mkdir(parents=True, exist_ok=True)
    desired = sorted(set(indices))
    cache = {}
    # One source bag at a time; at most one selected JPEG per source frame.
    with AnyReader([bag]) as reader:
        connections = [c for c in reader.connections if c.topic == image_topic]
        wanted = set(desired)
        for i, (connection, _, raw) in enumerate(reader.messages(connections=connections)):
            if i in wanted:
                msg = reader.deserialize(raw, connection.msgtype)
                cache[i] = bytes(msg.data)
    with av.open(str(path), 'w') as container:
        stream = container.add_stream('libx264', rate=Fraction(fps))
        stream.width, stream.height = size
        stream.pix_fmt = 'yuv420p'
        stream.thread_count = encoder_threads
        stream.options = {'crf': '18', 'preset': 'fast'}
        for i, source_index in enumerate(indices):
            img = Image.open(io.BytesIO(cache[source_index])).convert('RGB').resize(size, Image.Resampling.LANCZOS)
            frame = av.VideoFrame.from_image(img)
            frame.pts = i
            frame.time_base = Fraction(1, fps)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def draft_manifest(processed, task=None):
    """Create an explicitly unconfirmed full-range manifest from processed reports."""
    reports = sorted(processed.glob('bag_*/report.json'))
    if not reports:
        raise ValueError(f'No processed bag reports found in {processed}')
    episodes = []
    for index, report_path in enumerate(reports):
        report = json.loads(report_path.read_text())
        episodes.append({
            'bag': report_path.parent.name,
            'split': 'val' if index % 10 == 0 else 'train',
            'start_seconds': 0.0,
            'end_seconds': float(report['duration_seconds']),
            'task': task if task is not None else '__UNLABELED_DO_NOT_TRAIN__',
            'boundaries_confirmed': False,
            'initial_suction': False,
            'image_time_offset_seconds': 0.0,
        })
    return {
        'boundaries_confirmed': False,
        'synchronization_confirmed': False,
        'feedback_as_action_confirmed': False,
        'initial_suction_provenance': 'User confirmed suction is OFF at recording start; automatic drafts default to false.',
        'notes': 'Auto-generated draft from processed bag reports. Review labels, boundaries, clocks, action semantics, and initial suction before training.',
        'episodes': episodes,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--processed', type=Path, default=Path('processed_dual'))
    labels = p.add_mutually_exclusive_group()
    labels.add_argument('--task', help='Language instruction shared by all recorded bags')
    labels.add_argument('--manifest', type=Path,
                   help='Optional reviewed annotations; otherwise use all processed bags as a draft')
    p.add_argument('--output', type=Path, default=Path('datasets/episode_joint_dual'))
    p.add_argument('--allow-draft', action='store_true')
    p.add_argument('--generate-draft-manifest', type=Path,
                   help='Write a full-range, unlabeled manifest from processed reports and exit')
    p.add_argument('--fps', type=int, default=20)
    p.add_argument('--video-workers', type=int, default=2, help='Concurrent video encoding jobs')
    p.add_argument('--encoder-threads', type=int, default=2, help='H.264 threads per video job')
    args = p.parse_args()
    if args.task is not None:
        args.task = task_text(args.task)
    if not args.generate_draft_manifest and not (args.task or args.manifest):
        p.error('Dataset construction requires --task "instruction" or --manifest PATH')
    if args.video_workers < 1 or args.encoder_threads < 1:
        p.error('Video workers and encoder threads must be positive')
    if args.fps <= 0 or 10**9 % args.fps:
        raise ValueError('FPS must divide 1e9 exactly')
    if args.generate_draft_manifest:
        destination = args.generate_draft_manifest
        if destination.exists():
            raise ValueError(f'Refusing to overwrite existing manifest: {destination}')
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(draft_manifest(args.processed, args.task), indent=2, ensure_ascii=False)+'\n')
        print(f'Wrote unconfirmed draft manifest: {destination}', flush=True)
        return
    if args.manifest:
        manifest_bytes = args.manifest.read_bytes()
        manifest = json.loads(manifest_bytes)
    else:
        manifest = draft_manifest(args.processed, args.task)
        manifest_bytes = (json.dumps(manifest, indent=2, ensure_ascii=False)+'\n').encode()
        print(f'Using all {len(manifest["episodes"])} processed bags as an unconfirmed draft', flush=True)
    validate_tasks(manifest)
    if not manifest.get('episodes'):
        raise ValueError('Manifest contains no episodes')
    seen_splits = {}
    for entry in manifest['episodes']:
        previous_split = seen_splits.setdefault(entry['bag'], entry['split'])
        if previous_split != entry['split']:
            raise ValueError('The same source bag cannot appear in both train and val')
        if not (np.isfinite(entry['start_seconds']) and np.isfinite(entry['end_seconds'])
                and 0 <= entry['start_seconds'] < entry['end_seconds']):
            raise ValueError('Invalid episode time range')
        if entry.get('initial_suction') is not None and type(entry['initial_suction']) is not bool:
            raise ValueError('initial_suction must be true, false, or null')
    ready = bool(manifest.get('boundaries_confirmed') and manifest.get('synchronization_confirmed')
                 and manifest.get('feedback_as_action_confirmed')
                 and all(isinstance(e.get('task'), str) and e['task'].strip()
                         and '__UNLABELED' not in e['task'] and e.get('boundaries_confirmed')
                         for e in manifest['episodes']))
    if not ready and not args.allow_draft:
        raise ValueError('Unconfirmed labels/boundaries/clocks. Review manifest, or use --allow-draft for inspection only.')
    processed_bags = {p.name for p in args.processed.glob('bag_*') if p.is_dir() and (p/'report.json').is_file()}
    manifest_bags = {e['bag'] for e in manifest['episodes']}
    missing = sorted(manifest_bags-processed_bags)
    if missing:
        preview = ', '.join(missing[:8])
        suffix = f' (and {len(missing)-8} more)' if len(missing) > 8 else ''
        raise ValueError(f'Manifest references {len(missing)} bags absent from {args.processed}: {preview}{suffix}. Generate a matching draft with --generate-draft-manifest PATH, then review it.')
    if not processed_bags:
        raise ValueError(f'No processed bag reports found in {args.processed}')
    joint_names = ForwardKinematics(args.processed/'robot.urdf', 'base_link', 'link6').joint_names
    summary = json.loads((args.processed/'summary.json').read_text())
    if summary.get('joint_order') != joint_names or not joint_names or len(joint_names) > 31:
        raise ValueError('Processed joint order does not match URDF or exceeds model capacity')
    dimension = len(joint_names)+1
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(args.processed/'robot.urdf', args.output/'robot.urdf')
    (args.output/'episodes.source.json').write_bytes(manifest_bytes)
    split_meta = {s: [] for s in ('train', 'val')}
    arrays = {'train': {'observation.state': [], 'action': []}, 'val': {'observation.state': [], 'action': []}}
    tasks = list(dict.fromkeys(e.get('task') or '__UNLABELED_DO_NOT_TRAIN__' for e in manifest['episodes']))
    provenance = []
    counters = {'train': 0, 'val': 0}
    pending = deque()
    with ThreadPoolExecutor(max_workers=args.video_workers) as encoder:
        for entry in manifest['episodes']:
            name, split = entry['bag'], entry['split']
            if split not in split_meta:
                raise ValueError('Unknown split')
            source = args.processed/name
            report_path = source/'report.json'
            required_paths = (report_path, source/'joints.parquet')
            if not all(path.is_file() for path in required_paths):
                missing = [str(path) for path in required_paths if not path.is_file()]
                print(f'Skipping {name}: missing processed files: {", ".join(missing)}', flush=True)
                continue
            report = json.loads((source/'report.json').read_text())
            joints = pq.read_table(source/'joints.parquet').to_pylist()
            if set(report.get('cameras', {})) != set(CAMERAS):
                raise ValueError(f'Dual camera metadata missing in {source}; rerun process_bags.py')
            camera_times = {camera: np.asarray(pq.read_table(source/report['cameras'][camera]['table'])['stamp_ns'],
                                              dtype=np.int64) for camera in CAMERAS}
            suction = pq.read_table(source/'suction.parquet').to_pylist() if (source/'suction.parquet').exists() else []
            if any(len(j['position']) != len(joint_names) or not np.isfinite(j['position']).all() for j in joints):
                raise ValueError(f'Invalid motor angles in {name}')
            jt = np.array([j['stamp_ns'] for j in joints], dtype=np.int64)
            for t in (jt, *camera_times.values()):
                if not len(t):
                    raise ValueError(f'Empty joint/camera stream in {name}')
                if (t <= 0).any() or (np.diff(t) <= 0).any():
                    raise ValueError(f'Invalid header timestamps in {name}')
            offsets = {camera: int(round(entry.get('camera_time_offsets_seconds', {}).get(
                camera, entry.get('image_time_offset_seconds', 0))*1e9)) for camera in CAMERAS}
            camera_times = {camera: t+offsets[camera] for camera, t in camera_times.items()}
            st = np.array([x['bag_ns'] for x in suction], dtype=np.int64)
            sv = np.array([x['enabled'] for x in suction], dtype=np.float32)
            if entry.get('initial_suction') is not None:
                st = np.insert(st, 0, min(jt[0], *(t[0] for t in camera_times.values()), report['start_bag_ns'])-1)
                sv = np.insert(sv, 0, float(entry['initial_suction']))
            if not len(st):
                raise ValueError(f'No known suction values: {name}')
            start = max(jt[0], *(t[0] for t in camera_times.values()), st[0], report['start_bag_ns']+int(entry['start_seconds']*1e9))
            end = min(jt[-1], *(t[-1] for t in camera_times.values()), report['start_bag_ns']+int(entry['end_seconds']*1e9))
            step = 10**9//args.fps
            grid = np.arange(start, end, step, dtype=np.int64)
            ji, si = causal_indices(jt, grid), causal_indices(st, grid)
            camera_indices = {camera: causal_indices(t, grid) for camera, t in camera_times.items()}
            valid = (ji >= 0) & (si >= 0)
            valid &= (grid-jt[np.maximum(ji, 0)] <= 50_000_000)
            for camera, t in camera_times.items():
                indices = camera_indices[camera]
                valid &= (indices >= 0) & (grid-t[np.maximum(indices, 0)] <= 75_000_000)
            # Split at gaps; future feedback labels cannot cross discontinuities.
            runs = np.split(np.flatnonzero(valid), np.flatnonzero(np.diff(np.flatnonzero(valid)) != 1)+1)
            for run in runs:
                if len(run) < 18:
                    continue
                values = np.asarray([joints[ji[k]]['position']+[float(sv[si[k]])] for k in run], dtype=np.float32)
                # Action at t is observed absolute motor angles + commanded suction at t+1/fps.
                observation, action = values[:-1], values[1:]
                selected = run[:-1]
                ep = len(split_meta[split]); n = len(selected)
                task = entry.get('task') or '__UNLABELED_DO_NOT_TRAIN__'
                rows = []
                for k, idx in enumerate(selected):
                    rows.append({'observation.state': observation[k].tolist(), 'action': action[k].tolist(),
                                 'timestamp': k/args.fps, 'frame_index': k, 'episode_index': ep,
                                 'index': counters[split]+k, 'task_index': tasks.index(task),
                                 'annotation.human.task_description': tasks.index(task),
                                 'source_timestamp_ns': int(grid[idx]),
                                 'action_source_timestamp_ns': int(grid[run[k+1]]),
                                 'joint_timestamp_ns': int(jt[ji[idx]]),
                                 **{f'{camera}_image_timestamp_ns': int(camera_times[camera][camera_indices[camera][idx]])
                                    for camera in CAMERAS},
                                 **{f'{camera}_image_original_timestamp_ns': int(camera_times[camera][camera_indices[camera][idx]]-offsets[camera])
                                    for camera in CAMERAS},
                                 'joint_positions': joints[ji[idx]]['position'],
                                 'ee_quaternion_xyzw': joints[ji[idx]]['ee_quaternion_xyzw']})
                root = args.output/split
                data = root/'data/chunk-000'/f'episode_{ep:06d}.parquet'
                data.parent.mkdir(parents=True, exist_ok=True)
                table = pa.Table.from_pylist(rows)
                for column in ('observation.state', 'action'):
                    table = table.set_column(table.schema.get_field_index(column), column,
                                             pa.array(table[column].to_pylist(), type=pa.list_(pa.float32())))
                pq.write_table(table, data, compression='zstd')
                for camera in CAMERAS:
                    video = root/f'videos/chunk-000/observation.images.{camera}'/f'episode_{ep:06d}.mp4'
                    pending.append(encoder.submit(
                        encode_video, Path(report['bag']), camera_indices[camera][selected].tolist(),
                        video, args.fps, (640, 480), report['cameras'][camera]['topic'], args.encoder_threads))
                    # Bound queued work and JPEG caches; propagate failures before publishing metadata.
                    if len(pending) >= args.video_workers:
                        pending.popleft().result()
                split_meta[split].append({'episode_index': ep, 'tasks': [task], 'length': n})
                counters[split] += n
                arrays[split]['observation.state'].append(observation)
                arrays[split]['action'].append(action)
                provenance.append({'bag': name, 'split': split, 'episode_index': ep, 'frames': n,
                                   'first_seconds_from_bag_start': (int(grid[selected[0]])-report['start_bag_ns'])/1e9,
                                   'last_seconds_from_bag_start': (int(grid[selected[-1]])-report['start_bag_ns'])/1e9,
                                   'camera_time_offsets_seconds': {c: offsets[c]/1e9 for c in CAMERAS},
                                   'max_joint_age_ms': float((grid[selected]-jt[ji[selected]]).max()/1e6),
                                   'max_image_age_ms': {c: float((grid[selected]-camera_times[c][camera_indices[c][selected]]).max()/1e6)
                                                        for c in CAMERAS}})
                print(f'{name} -> {split}/episode_{ep:06d}: {n} frames (video jobs submitted)', flush=True)
        while pending:
            pending.popleft().result()
    print('All video encoding jobs completed', flush=True)
    if not all(arrays[s]['action'] for s in arrays):
        raise ValueError('Both train and val require usable sequences')
    training_stats = {key: stats(np.concatenate(parts)) for key, parts in arrays['train'].items()}
    names = joint_names+['suction']
    modalities = {s: {'joint_position': {'start': 0, 'end': len(joint_names), 'absolute': True},
                       'suction': {'start': len(joint_names), 'end': dimension, 'absolute': True}}
                  for s in ('state', 'action')}
    modalities.update(video={c: {'original_key': f'observation.images.{c}'} for c in CAMERAS},
                      annotation={'human.task_description': {}})
    for group in ('state', 'action'):
        for field in modalities[group].values():
            field['dtype'] = 'float32'
    for split in split_meta:
        meta = args.output/split/'meta'; meta.mkdir()
        (meta/'episodes.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in split_meta[split]))
        (meta/'tasks.jsonl').write_text(''.join(json.dumps({'task_index': i, 'task': t}, ensure_ascii=False)+'\n' for i, t in enumerate(tasks)))
        write_json(meta/'modality.json', modalities)
        write_json(meta/'stats.json', training_stats)  # Never fit normalization on validation data.
        features = {k: {'dtype': 'float32', 'shape': [dimension], 'names': names} for k in ('observation.state', 'action')}
        for camera in CAMERAS:
            features[f'observation.images.{camera}'] = {'dtype': 'video', 'shape': [480, 640, 3], 'names': ['height', 'width', 'channel'],
                'video_info': {'video.fps': args.fps, 'video.codec': 'h264', 'video.pix_fmt': 'yuv420p', 'video.is_depth_map': False, 'has_audio': False}}
        for k in ('frame_index', 'episode_index', 'index', 'task_index', 'annotation.human.task_description'):
            features[k] = {'dtype': 'int64', 'shape': [1]}
        features['timestamp'] = {'dtype': 'float64', 'shape': [1]}
        write_json(meta/'info.json', {'codebase_version': 'v2.0', 'robot_type': 'episode_arm_joint',
            'total_episodes': len(split_meta[split]), 'total_frames': counters[split], 'total_tasks': len(tasks),
            'total_videos': len(split_meta[split])*len(CAMERAS), 'total_chunks': 1, 'chunks_size': 1000,
            'fps': args.fps, 'splits': {split: f'0:{len(split_meta[split])}'},
            'data_path': 'data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet',
            'video_path': 'videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4', 'features': features})
    write_json(args.output/'build_report.json', {'ready_for_training': ready, 'fps': args.fps, 'camera_names': list(CAMERAS),
        'split_frames': counters, 'episodes': provenance, 'action_representation': 'joint_position_suction', 'joint_names': joint_names, 'joint_unit': 'radian',
        'action_semantics': f'future measured absolute motor joint angles and commanded suction, +{1/args.fps}s; NOT recorded control target',
        'clock_policy': 'image and joint header stamps, Bool bag receive time; image offset only if explicitly configured',
        'statistics': 'fit on train only; copied unchanged to val',
        'source_manifest_sha256': __import__('hashlib').sha256(manifest_bytes).hexdigest()})


if __name__ == '__main__':
    main()
