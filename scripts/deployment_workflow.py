"""Repeatable conversion -> training -> calibration -> AI Hub compilation."""
import hashlib
import json
import os
from pathlib import Path

from workflow_state import StageRunner, lock
from task_labels import validate_dataset_tasks

ROOT = Path(__file__).resolve().parents[1]


def file_id(path):
    path = Path(path)
    return [str(path.resolve()), hashlib.sha256(path.read_bytes()).hexdigest()]


def checkpoint_id(path):
    metadata = path/'experiment_cfg/metadata.json'
    if not metadata.is_file():
        raise ValueError(f'Missing checkpoint metadata: {metadata}')
    weights = sorted(list(path.glob('*.safetensors')) + list(path.glob('*.bin')))
    if not weights:
        raise ValueError(f'Missing checkpoint weights: {path}')
    return [file_id(metadata), [[str(p.resolve()), p.stat().st_size, p.stat().st_mtime_ns]
                                for p in weights]]


def deploy(pipeline):
    a = pipeline.args
    with lock(a.work_dir, a.dry_run):
        _deploy(pipeline)


def _deploy(pipeline):
    a = pipeline.args
    # Existing datasets can be used even when raw bags/decoded intermediates were archived.
    if (a.dataset/'build_report.json').is_file():
        report = json.loads((a.dataset/'build_report.json').read_text())
        if report.get('camera_names') != ['front', 'wrist']:
            raise ValueError('Dataset requires front and wrist cameras')
        if a.manifest and report.get('source_manifest_sha256') != file_id(a.manifest)[1]:
            raise ValueError('Manifest changed; choose a new dataset path')
        validate_dataset_tasks(a.dataset, a.task)
        print(f'Reusing dataset: {a.dataset}', flush=True)
        pipeline.run(pipeline.data_python+[ROOT/'scripts/validate_dataset.py', a.dataset, '--decode-video'])
    else:
        pipeline.convert()

    stages = StageRunner(a.work_dir, a.dry_run)

    sources = [file_id(ROOT/p) for p in ('episode_config.py', 'scripts/train.py',
               'scripts/quantize_model.py', 'scripts/quantization_samples.py',
               'scripts/quantization_config.py', 'scripts/export_qnn.py')]
    dataset_id = file_id(a.dataset/'build_report.json') if (a.dataset/'build_report.json').is_file() else str(a.dataset)
    if a.checkpoint:
        checkpoint = a.checkpoint
        identity = checkpoint_id(checkpoint)
        print(f'Reusing trained checkpoint: {checkpoint}', flush=True)
    else:
        if a.output:
            raise ValueError('deploy uses --work-dir for training outputs; use --checkpoint to reuse weights')
        def train(output):
            pipeline.check()
            pipeline.train(output=output)
        def validate_train(output):
            checkpoint = output/f'checkpoint-{a.max_steps}'
            checkpoint_id(checkpoint)
            state = json.loads((checkpoint/'trainer_state.json').read_text())
            if state.get('global_step') != a.max_steps:
                raise ValueError(f'Training has not reached {a.max_steps} steps')
        output = stages.run('train', [dataset_id, sources, a.base_model, a.max_steps,
                       a.batch_size, a.lora_rank, str(a.repo)], train, validate_train)
        checkpoint = output/f'checkpoint-{a.max_steps}'
        identity = str(checkpoint) if a.dry_run else checkpoint_id(checkpoint)

    quant_env = Path(os.environ.get('GR00T_QUANT_ENV', ROOT/'.venv-quant')).resolve()
    if not (quant_env/'bin/python').is_file():
        pipeline.run(['bash', ROOT/'quantize.sh', 'setup'])

    def validate_files(output):
        if not output.is_dir() or not any(p.is_file() and p.stat().st_size for p in output.rglob('*')):
            raise ValueError(f'Missing or empty stage output: {output}')

    quantized = stages.run('quantize', [identity, dataset_id, sources, a.num_samples, str(quant_env)],
        lambda output: pipeline.run(['bash', ROOT/'quantize.sh', 'quantize',
            '--checkpoint', checkpoint, '--dataset', a.dataset, '--output', output,
            '--num-samples', a.num_samples]), validate_files)
    target = ['--device', a.device] if a.device else ['--chipset', a.chipset]
    exported = stages.run('qnn', [str(quantized), sources, target, 'precompiled_qnn_onnx'],
        lambda output: pipeline.run(['bash', ROOT/'quantize.sh', 'qnn',
            '--checkpoint', quantized, *target, '--runtime', 'precompiled_qnn_onnx',
            '--output', output]), validate_files)
    print(f'Precompiled QNN ONNX output: {exported}', flush=True)
