"""Persistent successful-stage records and exclusive workflow locking."""
from contextlib import contextmanager
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


@contextmanager
def lock(work, dry_run):
    if dry_run:
        yield
        return
    work.mkdir(parents=True, exist_ok=True)
    with (work/'.lock').open('w') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f'Another deploy process is using {work}') from None
        yield


class StageRunner:
    def __init__(self, work_dir, dry_run=False):
        self.work_dir = work_dir
        self.dry_run = dry_run

    def run(self, name, signature, command, validate):
        key = fingerprint(signature)
        receipt = self.work_dir/f'{name}.json'
        if receipt.is_file():
            saved = json.loads(receipt.read_text())
            output = Path(saved['output'])
            if saved['signature'] != key:
                raise ValueError(f'{name} inputs/configuration changed; choose a new --work-dir')
            validate(output)
            print(f'Skipping completed {name}: {output}', flush=True)
            return output
        # Keep failed attempts intact; retry in a fresh directory.
        suffix = 'preview' if self.dry_run else datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        output = self.work_dir/f'{name}_{suffix}'
        command(output)
        if not self.dry_run:
            validate(output)
            temporary = receipt.with_suffix('.tmp')
            temporary.write_text(json.dumps({'signature': key, 'output': str(output)}, indent=2)+'\n')
            os.replace(temporary, receipt)
        return output
