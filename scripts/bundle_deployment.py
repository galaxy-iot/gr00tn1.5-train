#!/usr/bin/env python3
"""Package existing compiled assets using the shared deployment bundle format."""
import argparse
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

from deployment_bundle import export_preprocessing, finalize_bundle, validate_bundle


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--compiled', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True, help='Original full trained checkpoint')
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--processor', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--runtime', choices=['precompiled_qnn_onnx', 'qnn_context_binary'], default='precompiled_qnn_onnx')
    a = p.parse_args()
    compiled, checkpoint, dataset, processor, output = (
        v.resolve() for v in (a.compiled, a.checkpoint, a.dataset, a.processor, a.output))
    if output.exists():
        p.error('Use a new output directory')
    for source in (compiled, checkpoint, dataset, processor):
        if not source.is_dir():
            p.error(f'Missing input directory: {source}')
        if output.is_relative_to(source) or source.is_relative_to(output):
            p.error('Output and input directories must not contain each other')
    if not (dataset/'robot.urdf').is_file():
        p.error('Dataset must contain robot.urdf')
    if not any(path.is_file() for path in compiled.rglob('*')
               if path.suffix in ('.onnx', '.bin', '.zip', '.data')
               and 'preprocessing' not in path.relative_to(compiled).parts):
        p.error('No compiled model artifacts found')
    # Export and validate the matching processor before copying large compiled assets.
    with TemporaryDirectory(prefix='gr00t-bundle-') as temporary:
        prep = Path(temporary)/'preprocessing'
        export_preprocessing(checkpoint, prep, dataset=dataset, processor_dir=processor)
        shutil.copytree(compiled, output,
                        ignore=lambda folder, names: [name for name in names
                            if (Path(folder) == compiled and name in ('preprocessing', 'bundle.json'))
                            or name in ('.hf_modules', '__pycache__') or name.endswith('.pyc')])
        finalize_bundle(output, a.runtime, preprocessing=prep)
        try:
            validate_bundle(output)
        except Exception:
            (output/'bundle.json').unlink(missing_ok=True)
            raise
    print(f'Complete deployment bundle: {output}')


if __name__ == '__main__':
    main()
