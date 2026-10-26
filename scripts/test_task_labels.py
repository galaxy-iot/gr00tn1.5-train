"""Language labels must survive conversion and prevent stale dataset reuse."""
import ast
from contextlib import redirect_stderr
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from pipeline import Pipeline
from pipeline_cli import parse_args
from task_labels import task_text, validate_dataset_tasks


class TaskLabelsTest(unittest.TestCase):
    def test_missing_empty_placeholder_and_conflicting_options_fail(self):
        for argv in (['convert'], ['convert', '--task', '  '],
                     ['convert', '--task', '__UNLABELED_DO_NOT_TRAIN__'],
                     ['convert', '--task', 'pick', '--manifest', 'labels.json']):
            with self.subTest(argv=argv), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    parse_args(argv)
        self.assertEqual(task_text('  吸取物体  '), '吸取物体')

    def test_conversion_forwards_task(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            args = parse_args(['convert', '--task', '吸取物体', '--dry-run',
                               '--processed', str(root/'processed'), '--dataset', str(root/'dataset')])
            pipeline = Pipeline(args)
            with patch.object(pipeline, 'run') as run:
                pipeline.convert()
            build = next(call.args[0] for call in run.call_args_list
                         if any(str(value).endswith('build_dataset.py') for value in call.args[0]))
            self.assertEqual(build[build.index('--task')+1], '吸取物体')

    def test_all_bags_receive_instruction(self):
        # Exercise the real manifest function without importing video/ROS dependencies.
        source = Path(__file__).with_name('build_dataset.py').read_text()
        function = next(node for node in ast.parse(source).body
                        if isinstance(node, ast.FunctionDef) and node.name == 'draft_manifest')
        namespace = {'json': json}
        exec(compile(ast.Module(body=[function], type_ignores=[]), 'build_dataset.py', 'exec'), namespace)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            for index in range(3):
                bag = root/f'bag_{index}'
                bag.mkdir()
                (bag/'report.json').write_text('{"duration_seconds": 12}')
            manifest = namespace['draft_manifest'](root, '吸取物体')
            self.assertEqual([e['task'] for e in manifest['episodes']], ['吸取物体']*3)
            self.assertFalse(manifest['boundaries_confirmed'])

    def test_reuse_rejects_old_or_inconsistent_labels(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'episodes.source.json').write_text('{"episodes": [{"task": "吸取物体"}]}')
            for split in ('train', 'val'):
                meta = root/split/'meta'
                meta.mkdir(parents=True)
                (meta/'tasks.jsonl').write_text('{"task_index": 0, "task": "吸取物体"}\n')
            validate_dataset_tasks(root, '吸取物体')
            with self.assertRaisesRegex(ValueError, 'differs'):
                validate_dataset_tasks(root, '放下物体')
            (root/'train/meta/tasks.jsonl').write_text('{"task_index": 0, "task": "__UNLABELED_DO_NOT_TRAIN__"}\n')
            with self.assertRaises(ValueError):
                validate_dataset_tasks(root)


if __name__ == '__main__':
    unittest.main()
