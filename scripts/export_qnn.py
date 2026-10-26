#!/usr/bin/env python3
"""Compile an already calibrated collection with an explicitly chosen Hub device.

This submits jobs to Qualcomm AI Hub. Run --help to inspect this wrapper first.
"""
import argparse
from pathlib import Path

from export_onnx import register_config


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=Path, required=True, help='Quantized collection directory')
    target = p.add_mutually_exclusive_group(required=True)
    target.add_argument('--device', help='Exact supported AI Hub device name')
    target.add_argument('--chipset', help='AI Hub chipset identifier, e.g. qualcomm-qcs8275')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--runtime', choices=['precompiled_qnn_onnx', 'qnn_context_binary'], default='precompiled_qnn_onnx')
    args = p.parse_args()
    if not args.checkpoint.is_dir():
        raise ValueError('Missing calibrated checkpoint directory')
    if args.output.exists():
        raise ValueError('Use a fresh export directory')
    register_config()
    from qai_hub_models_cli.cli import main as cli_main
    # Generic GrootDataset currently loads DEFAULT GR1 metadata. Do not run
    # that example-based inferencing for our robot; validate with task inputs.
    target_args = ['--device', args.device] if args.device else ['--chipset', args.chipset]
    cli_main(['export', 'grootn15', '--checkpoint', str(args.checkpoint.resolve()),
              '--data-config', 'episode_pose', '--embodiment-tag', 'new_embodiment',
              '--precision', 'mixed', '--target-runtime', args.runtime,
              *target_args, '--output-dir', str(args.output.resolve()), '--skip-inferencing'])


if __name__ == '__main__':
    main()
