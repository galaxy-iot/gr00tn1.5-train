#!/usr/bin/env python3
"""Install an official binary wheel without silently falling back to compilation."""
import argparse
import platform
from pathlib import Path
import subprocess
import sys

import torch
from packaging.version import Version


def wheel_url(version):
    if sys.platform != 'linux' or not torch.version.cuda:
        raise RuntimeError('This training setup requires Linux and CUDA-enabled PyTorch')
    cuda = Version(torch.version.cuda)
    if cuda.major not in (11, 12):
        raise RuntimeError(f'Unsupported CUDA wheel family: {cuda}')
    # FlashAttention 2.7 releases group CUDA builds by major version.
    torch_version = Version(torch.__version__)
    python_tag = f'cp{sys.version_info.major}{sys.version_info.minor}'
    abi = str(torch._C._GLIBCXX_USE_CXX11_ABI).upper()
    name = (f'flash_attn-{version}+cu{cuda.major}'
            f'torch{torch_version.major}.{torch_version.minor}cxx11abi{abi}'
            f'-{python_tag}-{python_tag}-linux_{platform.machine()}.whl')
    original = f'https://github.com/Dao-AILab/flash-attention/releases/download/v{version}/{name}'
    return f'https://gh-proxy.org/{original}'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--version', required=True, type=Version)
    parser.add_argument('--wheel', type=Path)
    parser.add_argument('--print-url', action='store_true')
    args = parser.parse_args()
    target = str(args.wheel.resolve()) if args.wheel else wheel_url(args.version)
    if args.print_url:
        print(target)
        return 0
    if args.wheel and not args.wheel.is_file():
        parser.error(f'Wheel does not exist: {args.wheel}')
    print(f'Installing FlashAttention binary: {target}', flush=True)
    result = subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-deps',
                             '--retries', '5', '--timeout', '60', target])
    if result.returncode:
        print('Binary installation failed; source compilation was not attempted. '
              'Check gh-proxy.org connectivity, or download the wheel above on another '
              'machine and rerun ./run.sh setup-train --flash-attn-wheel /path/to/file.whl.',
              file=sys.stderr)
        return result.returncode
    import flash_attn
    if Version(flash_attn.__version__).public != args.version.public:
        raise RuntimeError(f'Installed FlashAttention {flash_attn.__version__}, expected {args.version}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
