#!/usr/bin/env python3
"""Exercise GR00T's real dual-camera loader/transforms on CPU, without model weights."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=ROOT/'third_party/Isaac-GR00T')
    parser.add_argument('--dataset', type=Path, default=ROOT/'datasets/episode_joint_dual')
    args = parser.parse_args()
    os.environ.setdefault('NO_ALBUMENTATIONS_UPDATE', '1')
    os.environ.setdefault('USE_TF', '0')
    sys.path[:0] = [str(ROOT), str(args.repo.resolve())]
    import numpy as np
    from episode_config import EpisodeJointConfig
    from gr00t.data.dataset import LeRobotSingleDataset
    from gr00t.model.transforms import collate

    report = json.loads((args.dataset/'build_report.json').read_text())
    if report.get('camera_names') != ['front', 'wrist']:
        raise ValueError('Expected a front + wrist dataset')
    if report.get('action_representation') != 'joint_position_suction':
        raise ValueError('Expected motor angles + suction dataset')
    dimension = len(report['joint_names'])+1
    config = EpisodeJointConfig()
    results = {}
    for split in ('train', 'val'):
        transforms = config.transform()
        dataset = LeRobotSingleDataset(args.dataset/split, config.modality_config(),
                                       'new_embodiment', video_backend='decord', transforms=transforms)
        episodes = [json.loads(line) for line in
                    (args.dataset/split/'meta/episodes.jsonl').read_text().splitlines()]
        checked = 0
        example = None
        for episode in episodes:
            for index in (0, episode['length']-1):
                raw = dataset.get_step_data(episode['episode_index'], index)
                for camera in config.video_keys:
                    if raw[camera].shape != (1, 480, 640, 3):
                        raise ValueError(f'{camera}: unexpected image shape {raw[camera].shape}')
                # Check the view axis immediately before the model transform.
                data = raw
                for transform in transforms.transforms[:-1]:
                    data = transform(data)
                if data['video'].shape != (1, 2, 224, 224, 3):
                    raise ValueError(f'Dual camera concatenation failed: {data["video"].shape}')
                sample = transforms.transforms[-1](data)
                if len(sample['eagle_content']['image_inputs']) != 2:
                    raise ValueError('The vision-language processor did not receive both cameras')
                for key, shape, active in [('state', (1, 64), dimension), ('action', (16, 32), 16*dimension)]:
                    if sample[key].shape != shape or not np.isfinite(sample[key]).all():
                        raise ValueError(f'Invalid transformed {key}')
                    if int(sample[key+'_mask'].sum()) != active:
                        raise ValueError(f'Invalid {key} mask')
                example = sample
                checked += 1
        batch = collate([example, example], transforms.transforms[-1].eagle_processor)
        if tuple(batch['action'].shape) != (2, 16, 32):
            raise ValueError('Training batch collation failed')
        results[split] = {'episodes': len(episodes), 'checked_samples': checked,
                          'batch_action_shape': list(batch['action'].shape),
                          'video_shape_before_model': [1, 2, 224, 224, 3]}
    results.update(camera_names=report['camera_names'], ready_for_training=report['ready_for_training'],
                   model_forward_backward_tested=False)
    (args.dataset/'training_data_check.json').write_text(json.dumps(results, indent=2)+'\n')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
