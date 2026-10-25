#!/usr/bin/env python3
"""Run Qualcomm's calibration recipe with our registered robot config."""
import argparse
import json
from pathlib import Path
import sys

from export_onnx import register_config


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--num-samples', type=int, default=100)
    p.add_argument('--component', choices=['all', 'vit', 'llm', 'vlm_proj', 'dit'], default='all')
    args = p.parse_args()
    if not json.loads((args.dataset/'build_report.json').read_text())['ready_for_training']:
        print('Dataset annotations are unconfirmed; continuing quantization.', flush=True)
    if not (args.checkpoint/'experiment_cfg/metadata.json').is_file():
        raise ValueError('Missing trained checkpoint experiment_cfg/metadata.json')
    if args.output.exists():
        raise ValueError('Use a fresh quantization output directory')
    register_config()
    from qai_hub_models.models.grootn15 import quantize
    # Upstream counts image crops/views as batch items when creating action
    # noise. Robot batch size comes from state, independently of camera count.
    import torch
    original_preprocess = quantize.GrootApp._preprocess

    def preprocess_robot_batch(self, step_data):
        pixels, ids, mask, state, actions = original_preprocess(self, step_data)
        batch_size = state.shape[0]
        if ids.shape[0] != batch_size or mask.shape[0] != batch_size:
            raise ValueError('Calibration text and robot state batch sizes differ')
        if actions.shape[0] != batch_size:
            actions = torch.randn(
                (batch_size, self.config.action_horizon, self.config.action_dim),
                dtype=actions.dtype, device=actions.device,
            )
        return pixels, ids, mask, state, actions

    quantize.GrootApp._preprocess = preprocess_robot_batch
    # Upstream computes its default dataset eagerly, even with --dataset-path.
    # Avoid downloading the unrelated GR1 demo dataset.
    quantize.get_default_dataset_path = lambda: str((args.dataset/'train').resolve())
    sys.argv = ['grootn15-quantize', '--checkpoint', str(args.checkpoint.resolve()),
                '--dataset-path', str((args.dataset/'train').resolve()),
                '--data-config', 'episode_pose', '--embodiment-tag', 'new_embodiment',
                '--host-device', 'cuda:0', '--component', args.component,
                '--num-samples', str(args.num_samples), '--output', str(args.output.resolve())]
    quantize.main()


if __name__ == '__main__':
    main()
