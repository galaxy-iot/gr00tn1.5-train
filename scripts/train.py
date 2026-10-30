#!/usr/bin/env python3
"""Validated launcher; execute with the NVIDIA training environment's Python."""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import shlex

ROOT = Path(__file__).resolve().parents[1]
REVISION = '4af2b622892f7dcb5aae5a3fb70bcb02dc217b96'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--repo', type=Path, required=True)
    p.add_argument('--dataset', type=Path, required=True, help='Dataset root containing train/, val/, build_report.json')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--max-steps', type=int, default=10000)
    p.add_argument('--smoke', action='store_true', help='Ten training steps and checkpoint verification')
    p.add_argument('--allow-draft-smoke', action='store_true', help='Allow unconfirmed data ONLY for the ten-step flow test')
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--lora-rank', type=int, default=0)
    p.add_argument('--base-model', default='nvidia/GR00T-N1.5-3B')
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    repo, data, out = args.repo.resolve(), args.dataset.resolve(), args.output.resolve()
    if subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip() != REVISION:
        raise ValueError('NVIDIA checkout does not match Qualcomm recipe revision')
    report = json.loads((data/'build_report.json').read_text())
    if report.get('action_representation') != 'joint_position_suction':
        raise ValueError('Expected motor angles + suction dataset. Rebuild with convert into a new --dataset path.')
    if report.get('camera_names') != ['front', 'wrist']:
        raise ValueError('Training requires both front and wrist cameras. Rerun convert with the dual-camera defaults.')
    if args.allow_draft_smoke and not args.smoke:
        p.error('--allow-draft-smoke is only valid with --smoke')
    if args.batch_size < 1 or args.workers < 1 or args.max_steps < 1 or args.lora_rank < 0:
        p.error('Batch size, workers and steps must be positive; LoRA rank must be nonnegative')
    if out.exists():
        raise ValueError('Output directory already exists; choose a new directory')
    if not report['ready_for_training']:
        print('Dataset annotations are unconfirmed; continuing training with the generated dataset.', flush=True)
    steps = 10 if args.smoke else args.max_steps
    cmd = [sys.executable, str(repo/'scripts/gr00t_finetune.py'),
           '--dataset-path', str(data/'train'), '--output-dir', str(out),
           '--data-config', 'episode_config:EpisodeJointConfig',
           '--embodiment-tag', 'new_embodiment', '--base-model-path', args.base_model,
           '--num-gpus', '1', '--batch-size', str(args.batch_size), '--max-steps', str(steps),
           '--save-steps', str(min(1000, steps)), '--learning-rate', '1e-4',
           '--no-tune-llm', '--no-tune-visual', '--tune-projector', '--tune-diffusion-model',
           '--dataloader-num-workers', str(args.workers), '--dataloader-prefetch-factor', '2',
           '--lora-rank', str(args.lora_rank),
           '--video-backend', 'decord', '--report-to', 'tensorboard']
    env = os.environ.copy()
    env['PYTHONPATH'] = os.pathsep.join([str(ROOT), str(repo), env.get('PYTHONPATH', '')])
    env.pop('EPISODE_GR00T_PREFIX', None)
    print('Launching:', shlex.join(cmd), flush=True)
    if args.dry_run:
        return
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable in this Python environment')
    print(f'GPU: {torch.cuda.get_device_name(0)}; smoke={args.smoke}; confirmed_data={report["ready_for_training"]}', flush=True)
    subprocess.run(cmd, env=env, cwd=repo, check=True)
    if args.smoke:
        checkpoint = out/'checkpoint-10'
        state = json.loads((checkpoint/'trainer_state.json').read_text())
        losses = [row['loss'] for row in state['log_history'] if 'loss' in row]
        gradients = [row['grad_norm'] for row in state['log_history'] if 'grad_norm' in row]
        if state['global_step'] != 10 or not losses or not gradients or not all(math.isfinite(x) for x in losses+gradients):
            raise RuntimeError('Smoke run did not record ten steps with finite loss/gradients')
        if not list(checkpoint.glob('*.safetensors')) and not list(checkpoint.glob('*.bin')):
            raise RuntimeError('Smoke run did not save model/adapter weights')
        if not (checkpoint/'experiment_cfg/metadata.json').is_file():
            raise RuntimeError('Smoke checkpoint is missing experiment metadata')
        receipt = {'steps': 10, 'losses': losses, 'gradient_norms': gradients,
                   'checkpoint': str(checkpoint), 'gpu': torch.cuda.get_device_name(0),
                   'confirmed_training_data': report['ready_for_training'],
                   'purpose': 'Training flow test only; not task performance validation', 'lora_rank': args.lora_rank}
        (out/'smoke_result.json').write_text(json.dumps(receipt, indent=2)+'\n')
        print('Ten-step smoke test passed; see', out/'smoke_result.json', flush=True)


if __name__ == '__main__':
    main()
