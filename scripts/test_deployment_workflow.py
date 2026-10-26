"""Exercise persistent stage reuse without GPU or AI Hub calls."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from deployment_workflow import deploy
from deployment_bundle import finalize_bundle
from test_deployment_bundle import make_preprocessing


class WorkflowTests(unittest.TestCase):
    def test_training_output_is_reused_without_mutating_arguments(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root/'data'
            dataset.mkdir()
            (dataset/'episodes.source.json').write_text('{"episodes": [{"task": "Pick up the object"}]}')
            for split in ('train', 'val'):
                meta = dataset/split/'meta'
                meta.mkdir(parents=True)
                (meta/'tasks.jsonl').write_text('{"task_index": 0, "task": "Pick up the object"}\n')
            (dataset/'build_report.json').write_text(json.dumps({'camera_names': ['front', 'wrist']}))
            args = SimpleNamespace(work_dir=root/'work', dry_run=False, dataset=dataset,
                manifest=None, task=None, checkpoint=None, output=None, device='test-device', chipset=None,
                num_samples=100, base_model='test-model', max_steps=10, batch_size=1,
                lora_rank=0, repo=root/'repo')
            trained = []

            def train(*, output):
                trained.append(output)
                checkpoint = output/'checkpoint-10'
                (checkpoint/'experiment_cfg').mkdir(parents=True)
                (checkpoint/'experiment_cfg/metadata.json').write_text('{}')
                (checkpoint/'model.safetensors').write_bytes(b'weights')
                (checkpoint/'trainer_state.json').write_text('{"global_step": 10}')

            def run(command):
                command = list(map(str, command))
                if '--output' in command:
                    output = Path(command[command.index('--output')+1])
                    output.mkdir(parents=True)
                    (output/'model.onnx').write_bytes(b'model')
                    make_preprocessing(output/'preprocessing')
                    if command[2] == 'qnn':
                        finalize_bundle(output, 'precompiled_qnn_onnx')

            pipeline = SimpleNamespace(args=args, data_python=['python'], run=run,
                                       check=lambda: None, train=train)
            with patch.dict('os.environ', {'GR00T_QUANT_ENV': str(root/'env')}):
                deploy(pipeline)
                deploy(pipeline)
            self.assertEqual(len(trained), 1)
            self.assertIsNone(args.output)

    def test_reuse_retry_and_changed_inputs(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root/'data'
            dataset.mkdir()
            (dataset/'episodes.source.json').write_text('{"episodes": [{"task": "Pick up the object"}]}')
            for split in ('train', 'val'):
                meta = dataset/split/'meta'
                meta.mkdir(parents=True)
                (meta/'tasks.jsonl').write_text('{"task_index": 0, "task": "Pick up the object"}\n')
            (dataset/'build_report.json').write_text(json.dumps({'camera_names': ['front', 'wrist']}))
            checkpoint = root/'checkpoint'
            (checkpoint/'experiment_cfg').mkdir(parents=True)
            (checkpoint/'experiment_cfg/metadata.json').write_text('{}')
            (checkpoint/'model.safetensors').write_bytes(b'weights')
            args = SimpleNamespace(work_dir=root/'work', dry_run=False, dataset=dataset,
                manifest=None, task=None, checkpoint=checkpoint, device=None, chipset='test-chip', num_samples=100)
            calls = []
            fail_qnn = True

            def run(command):
                nonlocal fail_qnn
                command = list(map(str, command))
                if '--output' not in command:
                    return
                kind = command[2]
                output = Path(command[command.index('--output')+1])
                output.mkdir(parents=True)
                (output/'model.onnx').write_bytes(b'model')
                make_preprocessing(output/'preprocessing')
                if kind == 'qnn':
                    finalize_bundle(output, 'precompiled_qnn_onnx')
                calls.append((kind, output))
                if kind == 'qnn' and fail_qnn:
                    fail_qnn = False
                    raise RuntimeError('simulated cloud failure')

            pipeline = SimpleNamespace(args=args, data_python=['python'], run=run)
            with patch.dict('os.environ', {'GR00T_QUANT_ENV': str(root/'env')}):
                with self.assertRaisesRegex(RuntimeError, 'cloud failure'):
                    deploy(pipeline)
                self.assertTrue((args.work_dir/'quantize.json').is_file())
                self.assertFalse((args.work_dir/'qnn.json').exists())
                deploy(pipeline)
                self.assertEqual([c[0] for c in calls], ['quantize', 'qnn', 'qnn'])
                self.assertNotEqual(calls[1][1], calls[2][1])
                deploy(pipeline)
                self.assertEqual(len(calls), 3)
                args.num_samples = 50
                with self.assertRaisesRegex(ValueError, 'configuration changed'):
                    deploy(pipeline)
                self.assertEqual(len(calls), 3)


if __name__ == '__main__':
    unittest.main()
