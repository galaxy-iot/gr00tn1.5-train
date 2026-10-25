#!/usr/bin/env python3
"""Validate generated clocks, future-label alignment, videos, and normalization."""
import argparse
import json
from pathlib import Path

import av
import numpy as np
import pyarrow.parquet as pq


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('dataset', type=Path)
    p.add_argument('--decode-video', action='store_true', help='Decode every frame, not just check container metadata')
    args = p.parse_args()
    report = json.loads((args.dataset/'build_report.json').read_text())
    cameras = report.get('camera_names', ['front'])
    period = 10**9//report['fps']
    arrays = {'observation.state': [], 'action': []}
    result = {}
    for split in ('train', 'val'):
        meta = args.dataset/split/'meta'
        info = json.loads((meta/'info.json').read_text())
        modalities = json.loads((meta/'modality.json').read_text())
        require(set(modalities['video']) == set(cameras), 'Camera modality mismatch')
        require({k for k, v in info['features'].items() if v['dtype'] == 'video'} ==
                {f'observation.images.{c}' for c in cameras}, 'Camera feature mismatch')
        episodes = [json.loads(x) for x in (meta/'episodes.jsonl').read_text().splitlines()]
        count = 0
        for e in episodes:
            ep = e['episode_index']
            table = pq.read_table(args.dataset/split/'data/chunk-000'/f'episode_{ep:06d}.parquet').to_pydict()
            obs = np.asarray(table['observation.state']); action = np.asarray(table['action'])
            n = len(obs)
            require(n == e['length'], 'Episode length mismatch')
            require(obs.shape == (n, 10) and action.shape == (n, 10), 'Invalid dimensions')
            require(np.isfinite(obs).all() and np.isfinite(action).all(), 'Nonfinite values')
            require(np.allclose(action[:-1], obs[1:]), 'Future feedback labels are misaligned')
            times = np.array(table['source_timestamp_ns'], dtype=np.int64)
            require(np.all(np.diff(times) == period), 'Nonuniform sample timeline')
            require(np.all(np.array(table['action_source_timestamp_ns'])-times == period), 'Wrong action shift')
            image_columns = ([f'{c}_image_timestamp_ns' for c in cameras]
                             if 'camera_names' in report else ['image_timestamp_ns'])
            for name, limit in [('joint_timestamp_ns', 50_000_000)]+[(c, 75_000_000) for c in image_columns]:
                age = times-np.array(table[name], dtype=np.int64)
                require(np.all((age >= 0) & (age <= limit)), 'Stale/future observation')
            for values in (obs, action):
                require(np.isin(values[:, 9], [0, 1]).all(), 'Invalid suction values')
                rows = values[:, 3:9].reshape(-1, 2, 3)
                require(np.allclose(np.linalg.norm(rows, axis=2), 1, atol=1e-5), 'Invalid 6D rotation rows')
                require(np.allclose(np.sum(rows[:, 0]*rows[:, 1], axis=1), 0, atol=1e-5), 'Nonorthogonal rotation rows')
            for camera in cameras:
                video = args.dataset/split/f'videos/chunk-000/observation.images.{camera}'/f'episode_{ep:06d}.mp4'
                with av.open(str(video)) as container:
                    stream = container.streams.video[0]
                    require((stream.width, stream.height) == (640, 480), f'{camera}: Video size mismatch')
                    require(stream.average_rate == report['fps'], f'{camera}: Video FPS mismatch')
                    require(stream.frames == n, f'{camera}: Video frame count mismatch')
                    if args.decode_video:
                        require(sum(1 for _ in container.decode(video=0)) == n, f'{camera}: Video decode/count failure')
            if split == 'train':
                arrays['observation.state'].append(obs)
                arrays['action'].append(action)
            count += n
        require(count == info['total_frames'] == report['split_frames'][split], 'Total frame mismatch')
        require(info['total_videos'] == len(episodes)*len(cameras), 'Total video mismatch')
        result[split] = {'episodes': len(episodes), 'frames': count}
    train_stats = json.loads((args.dataset/'train/meta/stats.json').read_text())
    require(train_stats == json.loads((args.dataset/'val/meta/stats.json').read_text()), 'Validation normalization differs')
    for name, chunks in arrays.items():
        values = np.concatenate(chunks)
        require(np.allclose(values.mean(0), train_stats[name]['mean'], atol=2e-6), 'Training mean mismatch')
        require(np.allclose(values.min(0), train_stats[name]['min']), 'Training minimum mismatch')
        require(np.allclose(values.max(0), train_stats[name]['max']), 'Training maximum mismatch')
    result['ready_for_training'] = report['ready_for_training']
    result['camera_names'] = cameras
    result['all_video_frames_decoded'] = args.decode_video
    (args.dataset/'validation_report.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
