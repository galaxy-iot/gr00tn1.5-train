"""Attach offline preprocessing assets to portable model output directories."""
from contextlib import contextmanager
import json
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]


def validate_preprocessing(directory):
    directory = Path(directory)
    report = json.loads((directory/'export_report.json').read_text())
    if not report.get('complete') or not report.get('offline_processor_reload_checked'):
        raise ValueError(f'Incomplete preprocessing resources: {directory}')
    for name in ('embedding.npy', 'prepare_language.py', 'preprocess_config.json',
                 'model_config.json', 'requirements-preprocessing.txt',
                 'experiment_cfg/metadata.json', 'processor/config.json',
                 'processor/tokenizer_config.json', 'processor/processor_config.json',
                 'processor/preprocessor_config.json', 'processor/vocab.json',
                 'processor/merges.txt', 'processor/chat_template.json',
                 'processor/processing_eagle2_5_vl.py', 'processor/image_processing_eagle2_5_vl_fast.py'):
        path = directory/name
        if not path.is_file() or not path.stat().st_size:
            raise ValueError(f'Missing preprocessing asset: {path}')


def export_preprocessing(checkpoint, directory, dataset=None, processor_dir=None):
    command = [sys.executable, str(ROOT/'scripts/export_preprocessing.py'),
               '--checkpoint', str(Path(checkpoint).resolve()), '--output', str(directory)]
    if processor_dir:
        command += ['--processor-dir', str(processor_dir)]
    # A separate process keeps offline Hugging Face settings out of cloud compilation.
    subprocess.run(command, check=True, cwd=ROOT)
    if dataset is not None:
        shutil.copy2(Path(dataset)/'robot.urdf', Path(directory)/'robot.urdf')
    validate_preprocessing(directory)


@contextmanager
def preprocessing_source(quantized, source_checkpoint=None, processor_dir=None):
    if source_checkpoint:
        with TemporaryDirectory(prefix='gr00t-preprocessing-') as temporary:
            source = Path(temporary)/'preprocessing'
            export_preprocessing(source_checkpoint, source, processor_dir=processor_dir)
            yield source
    else:
        source = Path(quantized)/'preprocessing'
        if not source.is_dir():
            raise ValueError('Quantized checkpoint lacks preprocessing resources. '
                             'Pass --source-checkpoint /path/to/original/trained/checkpoint')
        validate_preprocessing(source)
        yield source


def finalize_bundle(output, runtime, preprocessing=None):
    output = Path(output)
    if preprocessing is not None:
        shutil.copytree(preprocessing, output/'preprocessing',
                        ignore=shutil.ignore_patterns('.hf_modules', '__pycache__', '*.pyc'))
    validate_preprocessing(output/'preprocessing')
    artifacts = sorted(str(path.relative_to(output)) for path in output.rglob('*')
                       if path.is_file() and 'preprocessing' not in path.relative_to(output).parts
                       and path.suffix in ('.onnx', '.bin', '.zip', '.data'))
    if not artifacts:
        raise ValueError(f'No exported model artifacts found in {output}')
    shutil.copy2(ROOT/'ort_qnn_standalone.py', output/'ort_qnn_standalone.py')
    (output/'README_BUNDLE.md').write_text(
        '# GR00T portable model bundle\n\n'
        'Copy this entire directory, including model external weights/context files and preprocessing/.\n'
        'Model artifact paths are recorded relative to this directory in bundle.json.\n'
        'Install preprocessing/requirements-preprocessing.txt in a compatible Python environment.\n'
        'Prepare language inputs from this directory:\n\n'
        '    python preprocessing/prepare_language.py --bundle preprocessing --instruction "Pick up the object." --output instruction.npz\n\n'
        'Replace image-token slots with current ViT outputs before LLM execution.\n'
        'ort_qnn_standalone.py accepts prepared tensors and benchmarks model components; '
        'it is not a full raw-image robot controller.\n'
        'Image/state preprocessing, projector/DiT masks, diffusion scheduling and action inverse transforms '
        'must be implemented by the inference application.\n'
        'Install a compatible ONNX Runtime/QNN SDK on the target device separately.\n')
    (output/'bundle.json').write_text(json.dumps({
        'complete': True, 'runtime': runtime, 'models': artifacts,
        'preprocessing': 'preprocessing', 'language_script': 'preprocessing/prepare_language.py',
        'inference_tool': 'ort_qnn_standalone.py',
        'full_robot_inference_validated': False,
    }, indent=2)+'\n')


def validate_bundle(output):
    output = Path(output)
    manifest = json.loads((output/'bundle.json').read_text())
    if not manifest.get('complete') or not manifest.get('models'):
        raise ValueError(f'Incomplete model bundle: {output}')
    validate_preprocessing(output/manifest['preprocessing'])
    for name in manifest['models']:
        if not (output/name).is_file() or not (output/name).stat().st_size:
            raise ValueError(f'Missing bundled model: {output/name}')
