"""Arguments and validation for the run.sh command line."""
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Prepare environments, convert data, train and export GR00T models.")
    p.add_argument('stage', choices=['download', 'setup-data', 'setup-train', 'setup-quant', 'setup', 'convert', 'check-data', 'check', 'smoke', 'train', 'all', 'deploy'])
    p.add_argument('--dry-run', action='store_true', help='Print commands without downloading, installing, converting or training')
    deployment = p.add_argument_group('Deployment and stage reuse')
    deployment.add_argument('--work-dir', type=Path, default=ROOT/'exports/workflow', help='Persistent deploy stage records and outputs')
    deployment.add_argument('--checkpoint', type=Path, help='Deploy: reuse an existing trained checkpoint instead of training')
    target = deployment.add_mutually_exclusive_group()
    target.add_argument('--device', help='Deploy: exact AI Hub device name')
    target.add_argument('--chipset', help='Deploy: AI Hub chipset identifier')
    deployment.add_argument('--num-samples', type=int, default=100)
    environment = p.add_argument_group('Environment setup')
    environment.add_argument('--conda')
    environment.add_argument('--data-env', default='gr00t-data')
    environment.add_argument('--train-env', default='gr00t-train')
    environment.add_argument('--repo', type=Path, default=ROOT/'third_party/Isaac-GR00T')
    data = p.add_argument_group('Data conversion')
    data.add_argument('--input', type=Path, default=ROOT/'datas')
    data.add_argument('--processed', type=Path, default=ROOT/'processed_dual')
    data.add_argument('--manifest', type=Path, help='Optional annotations; defaults to all bags in --input')
    snapshot = ROOT/'processed_dual/robot.urdf'
    data.add_argument('--urdf', type=Path, default=snapshot if snapshot.is_file() else
                   ROOT/'episode1_urdf_1113.urdf')
    data.add_argument('--dataset', type=Path, default=ROOT/'datasets/episode_pose_dual')
    training = p.add_argument_group('Training')
    training.add_argument('--output', type=Path)
    training.add_argument('--batch-size', type=int, default=1)
    training.add_argument('--workers', type=int, default=2)
    data.add_argument('--video-workers', type=int, default=2, help='Concurrent conversion video jobs')
    data.add_argument('--encoder-threads', type=int, default=2, help='H.264 threads per video job')
    training.add_argument('--max-steps', type=int, default=10000)
    training.add_argument('--lora-rank', type=int, default=0)
    training.add_argument('--base-model', default='nvidia/GR00T-N1.5-3B')
    environment.add_argument('--flash-attn-version', default='2.7.1.post4')
    environment.add_argument('--flash-attn-wheel', type=Path, help='Install a downloaded FlashAttention wheel instead of accessing GitHub')
    args = p.parse_args(argv)
    if args.video_workers < 1 or args.encoder_threads < 1:
        p.error('Video workers and encoder threads must be positive')
    if args.batch_size < 1 or args.workers < 1 or args.max_steps < 1 or args.lora_rank < 0:
        p.error('Batch size, workers and steps must be positive; LoRA rank must be nonnegative')
    if args.data_env == args.train_env or any(not name or name == 'base' or '/' in name or '\\' in name
                                             or name.startswith('-') for name in (args.data_env, args.train_env)):
        p.error('Choose two distinct non-base Conda environment names')
    for key in ('repo', 'input', 'processed', 'manifest', 'urdf', 'dataset', 'output', 'work_dir', 'checkpoint'):
        value = getattr(args, key)
        if value is not None:
            setattr(args, key, value.resolve())
    if args.stage == 'deploy' and (not (args.device or args.chipset) or args.num_samples < 1):
        p.error('deploy requires --device or --chipset, and positive --num-samples')
    return args
