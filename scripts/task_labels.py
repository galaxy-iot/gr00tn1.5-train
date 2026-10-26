"""Validate explicit language instructions and dataset reuse."""
import json


def task_text(value):
    if not isinstance(value, str) or not value.strip() or '__UNLABELED' in value:
        raise ValueError('Provide a nonempty language instruction with --task, or labeled --manifest')
    return value.strip()


def validate_tasks(manifest, expected=None):
    episodes = manifest.get('episodes', [])
    if not episodes:
        raise ValueError('Manifest contains no episodes')
    for entry in episodes:
        actual = task_text(entry.get('task'))
        if expected is not None and actual != expected:
            raise ValueError('Dataset task differs from --task; choose a new --dataset path')


def validate_dataset_tasks(dataset, expected=None):
    manifest = json.loads((dataset/'episodes.source.json').read_text())
    validate_tasks(manifest, expected)
    for split in ('train', 'val'):
        tasks = [json.loads(line)['task'] for line in
                 (dataset/split/'meta/tasks.jsonl').read_text().splitlines() if line.strip()]
        if not tasks:
            raise ValueError(f'Dataset {split} contains no language instructions')
        for value in tasks:
            actual = task_text(value)
            if expected is not None and actual != expected:
                raise ValueError('Dataset task differs from --task; choose a new --dataset path')
