#!/usr/bin/env python3
"""Execute the environment, data and training stages behind run.sh."""
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

from pipeline_cli import parse_args
from task_labels import validate_dataset_tasks

ROOT = Path(__file__).resolve().parents[1]


def find_conda(explicit, dry_run):
    candidates = [explicit, os.environ.get('CONDA_EXE'), shutil.which('conda'),
                  str(Path.home()/'.miniconda3/bin/conda'), str(Path.home()/'miniconda3/bin/conda')]
    for item in candidates:
        if item and (Path(item).is_file() or shutil.which(item)):
            return item
    if dry_run:
        return explicit or 'conda'
    raise RuntimeError('Conda not found. Pass --conda /absolute/path/to/conda')


class Pipeline:
    def __init__(self, args):
        self.args = args
        self.conda = 'conda' if args.stage in ('download', 'setup-quant') else find_conda(args.conda, args.dry_run)
        self.data_python = [self.conda, 'run', '--no-capture-output', '-n', args.data_env, 'python']
        self.train_python = [self.conda, 'run', '--no-capture-output', '-n', args.train_env, 'python']

    def run(self, cmd):
        cmd = list(map(str, cmd))
        print('+ '+shlex.join(cmd), flush=True)
        if not self.args.dry_run:
            env = os.environ.copy()
            env.setdefault('MAX_JOBS', '2')
            subprocess.run(cmd, cwd=ROOT, env=env, check=True)

    def ensure_env(self, name, version):
        if self.args.dry_run:
            print(f'# Create {name} (Python {version}) only if missing; reuse otherwise.')
        else:
            result = json.loads(subprocess.check_output([self.conda, 'env', 'list', '--json'], text=True))
            if any(Path(p).name == name for p in result['envs']):
                self.run([self.conda, 'run', '-n', name, 'python', '-c',
                          f'import sys; assert sys.version_info[:2] == {tuple(map(int, version.split(".")))}, "Wrong environment Python version"'])
                return
        self.run([self.conda, 'create', '-y', '-n', name, f'python={version}', 'pip'])

    def download(self):
        self.run(['bash', ROOT/'scripts/download_gr00t.sh', self.args.repo])

    def setup_data(self):
        self.ensure_env(self.args.data_env, '3.11')
        self.run(self.data_python+['-m', 'pip', 'install', '-r', ROOT/'requirements-data.txt'])

    def setup_train(self):
        self.download()
        self.ensure_env(self.args.train_env, '3.10')
        self.run(self.train_python+['-m', 'pip', 'install', 'setuptools<81', 'wheel', 'packaging', 'ninja', 'psutil'])
        self.run(self.train_python+['-m', 'pip', 'install', '-e', str(self.args.repo)+'[base]', 'tensorboard'])
        cmd = self.train_python+[ROOT/'scripts/install_flash_attn.py',
                                 '--version', self.args.flash_attn_version]
        if self.args.flash_attn_wheel:
            cmd += ['--wheel', self.args.flash_attn_wheel]
        self.run(cmd)
        self.check()

    def setup(self):
        self.setup_data()
        self.setup_train()
        self.setup_quant()

    def setup_quant(self):
        self.run(['bash', ROOT/'quantize.sh', 'setup'])

    def check(self):
        # Actual CUDA tensor allocation catches runtime errors that nvidia-smi alone cannot.
        code = ('import torch, flash_attn, decord; '
                'print("Torch:",torch.__version__,"CUDA runtime:",torch.version.cuda); '
                'assert torch.cuda.is_available(), "CUDA unavailable in training environment"; '
                'print("GPU:",torch.cuda.get_device_name(0)); '
                'assert torch.cuda.get_device_capability(0)[0] >= 8, "This FlashAttention 2 recipe requires an Ampere or newer GPU"; '
                'assert torch.cuda.is_bf16_supported(), "BF16 required by this training recipe"; '
                'x=torch.randn(1,16,2,64,device="cuda",dtype=torch.bfloat16); '
                'y=flash_attn.flash_attn_func(x,x,x); torch.cuda.synchronize(); '
                'assert torch.isfinite(y).all(), "FlashAttention returned nonfinite values"; '
                'print("CUDA and FlashAttention kernel OK; flash-attn:",flash_attn.__version__)')
        self.run(self.train_python+['-c', code])

    def convert(self):
        a = self.args
        if not (a.task or a.manifest):
            raise ValueError('Conversion requires --task "instruction" or --manifest PATH')
        if not a.urdf.is_file():
            raise ValueError(f'URDF not found: {a.urdf}. Pass --urdf /path/to/robot.urdf')
        if a.processed.exists() and any(a.processed.iterdir()):
            if not (a.processed/'summary.json').is_file():
                raise ValueError('Incomplete processed directory. Choose a new --processed path.')
            summary = json.loads((a.processed/'summary.json').read_text())
            if summary.get('camera_names') != ['front', 'wrist']:
                raise ValueError('Decoded data lacks dual cameras. Choose a new --processed path and rerun convert.')
            if hashlib.sha256(a.urdf.read_bytes()).hexdigest() != summary['urdf_sha256']:
                raise ValueError('URDF changed. Choose new --processed and --dataset paths.')
            expected = {str(p.parent.resolve()) for p in a.input.glob('*/metadata.yaml')}
            if expected != {r['bag'] for r in summary['reports']}:
                raise ValueError('Bag set differs. Choose new --processed and --dataset paths.')
            print(f'Reusing decoded data: {a.processed}', flush=True)
        else:
            self.run(self.data_python+[ROOT/'scripts/process_bags.py', '--input', a.input,
                     '--output', a.processed, '--urdf', a.urdf])
        if a.dataset.exists():
            report_path = a.dataset/'build_report.json'
            if not report_path.is_file():
                raise ValueError(f'Incomplete dataset directory: {a.dataset}. Existing outputs were kept; choose a new --dataset path to rebuild.')
            report = json.loads(report_path.read_text())
            if report.get('action_representation') != 'joint_position_suction':
                raise ValueError('Expected motor angles + suction dataset. Rebuild with convert into a new --dataset path.')
            if report.get('camera_names') != ['front', 'wrist']:
                raise ValueError('Dataset lacks dual cameras. Choose a new --dataset path and rerun convert.')
            if a.manifest and report['source_manifest_sha256'] != hashlib.sha256(a.manifest.read_bytes()).hexdigest():
                raise ValueError('Manifest changed. Choose a new --dataset path to rebuild without overwriting.')
            if not a.manifest:
                source_manifest = json.loads((a.dataset/'episodes.source.json').read_text())
                expected_bags = {p.parent.name for p in a.input.glob('*/metadata.yaml')}
                if {e['bag'] for e in source_manifest['episodes']} != expected_bags:
                    raise ValueError('Dataset bag set differs from input. Choose a new --dataset path.')
            if (a.dataset/'robot.urdf').read_bytes() != a.urdf.read_bytes():
                raise ValueError('Dataset URDF differs. Choose a new --dataset path.')
            validate_dataset_tasks(a.dataset, a.task)
            print(f'Reusing dataset: {a.dataset}', flush=True)
        else:
            cmd = self.data_python+[ROOT/'scripts/build_dataset.py', '--processed', a.processed,
                                   '--output', a.dataset, '--video-workers', str(a.video_workers),
                                   '--encoder-threads', str(a.encoder_threads)]
            if a.task:
                cmd += ['--task', a.task]
            if a.manifest:
                cmd += ['--manifest', a.manifest]
            # Generated datasets can be used directly for training.
            self.run(cmd+['--allow-draft'])
        self.run(self.data_python+[ROOT/'scripts/validate_dataset.py', a.dataset, '--decode-video'])

    def train(self, smoke=False, output=None):
        a = self.args
        output = output or a.output or ROOT/'checkpoints'/datetime.now().strftime(('smoke_' if smoke else 'train_')+'%Y%m%d_%H%M%S')
        cmd = self.train_python+[ROOT/'scripts/train.py', '--repo', a.repo, '--dataset', a.dataset,
               '--output', output, '--batch-size', str(a.batch_size), '--workers', str(a.workers),
               '--max-steps', str(a.max_steps), '--lora-rank', str(a.lora_rank), '--base-model', a.base_model]
        if smoke:
            cmd += ['--smoke', '--allow-draft-smoke']
        self.run(cmd)

    def check_data(self):
        self.run(self.data_python+[ROOT/'scripts/validate_dataset.py', self.args.dataset, '--decode-video'])
        self.run(self.train_python+[ROOT/'scripts/check_training_data.py', '--repo', self.args.repo,
                                  '--dataset', self.args.dataset])


def main():
    args = parse_args()
    try:
        pipeline = Pipeline(args)
        if args.stage == 'deploy':
            from deployment_workflow import deploy
            deploy(pipeline)
        elif args.stage == 'all':
            pipeline.setup()
            pipeline.convert()
            pipeline.train(smoke=True)
        elif args.stage in ('smoke', 'train'):
            pipeline.check()
            pipeline.train(smoke=args.stage == 'smoke')
        else:
            getattr(pipeline, args.stage.replace('-', '_'))()
    except (subprocess.CalledProcessError, ValueError, RuntimeError, OSError) as exc:
        print(f'\nStopped: {exc}\nFix the error above and rerun the same stage. Existing outputs are retained.', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
