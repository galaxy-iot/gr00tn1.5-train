"""Check portable packaging and QNN wrapper integration without cloud jobs."""
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from deployment_bundle import finalize_bundle, preprocessing_source, validate_bundle
from export_qnn import main as export_qnn


def make_preprocessing(directory):
    directory.mkdir(parents=True)
    files = ('embedding.npy', 'prepare_language.py', 'preprocess_config.json',
             'model_config.json', 'requirements-preprocessing.txt', 'experiment_cfg/metadata.json',
             'processor/config.json', 'processor/tokenizer_config.json', 'processor/processor_config.json',
             'processor/preprocessor_config.json', 'processor/vocab.json', 'processor/merges.txt',
             'processor/chat_template.json', 'processor/processing_eagle2_5_vl.py',
             'processor/image_processing_eagle2_5_vl_fast.py')
    for name in files:
        path = directory/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('fixture')
    (directory/'export_report.json').write_text(json.dumps({
        'complete': True, 'offline_processor_reload_checked': True}))


class BundleTests(unittest.TestCase):
    def test_copy_is_self_contained_and_paths_are_relative(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root/'source'
            make_preprocessing(source)
            (source/'.hf_modules').mkdir()
            (source/'.hf_modules/cache.py').write_text('cache')
            output = root/'output'
            output.mkdir()
            (output/'vit.onnx').write_bytes(b'model')
            (output/'vit.onnx.data').write_bytes(b'weights')
            finalize_bundle(output, 'precompiled_qnn_onnx', source)
            validate_bundle(output)
            self.assertFalse((output/'preprocessing/.hf_modules').exists())
            self.assertFalse((output/'preprocessing/embedding.npy').is_symlink())
            manifest = json.loads((output/'bundle.json').read_text())
            self.assertEqual(manifest['models'], ['vit.onnx', 'vit.onnx.data'])
            self.assertTrue((output/'ort_qnn_standalone.py').is_file())
            (output/'vit.onnx.data').unlink()
            with self.assertRaisesRegex(ValueError, 'Missing bundled model'):
                validate_bundle(output)

    def test_old_quantized_model_requires_original_checkpoint_before_upload(self):
        with TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, '--source-checkpoint'):
                with preprocessing_source(Path(directory)):
                    self.fail('Missing resources accepted')

    def test_qnn_export_automatically_attaches_preprocessing(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            quantized = root/'quantized'
            make_preprocessing(quantized/'preprocessing')
            output = root/'qnn'

            def fake_cli(argv):
                target = Path(argv[argv.index('--output-dir')+1])
                target.mkdir()
                (target/'llm.onnx').write_bytes(b'compiled model')

            with patch('export_qnn.register_config'), patch.dict(sys.modules, {
                'qai_hub_models_cli.cli': SimpleNamespace(main=fake_cli)}), patch.object(sys, 'argv',
                ['export_qnn.py', '--checkpoint', str(quantized), '--chipset', 'test-chip', '--output', str(output)]):
                export_qnn()
            validate_bundle(output)
            self.assertTrue((output/'preprocessing/embedding.npy').exists())

    def test_quantization_automatically_exports_matching_checkpoint_resources(self):
        from quantize_model import main as quantize_model
        with TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root/'trained'
            (checkpoint/'experiment_cfg').mkdir(parents=True)
            (checkpoint/'experiment_cfg/metadata.json').write_text('{}')
            dataset = root/'dataset'
            dataset.mkdir()
            (dataset/'build_report.json').write_text('{"ready_for_training": true}')
            output = root/'quantized'
            recipe = SimpleNamespace(GrootApp=SimpleNamespace(_preprocess=lambda *args: None))

            def quantize(recipe, args):
                args.output.mkdir()
                (args.output/'llm.onnx').write_bytes(b'quantized model')

            def attach(source, destination, **kwargs):
                self.assertTrue((output/'llm.onnx').is_file())
                self.assertEqual(source, checkpoint)
                self.assertEqual(kwargs['dataset'], dataset)
                make_preprocessing(destination)

            with patch('quantize_model.register_config'), patch('quantize_model.run_quantization', side_effect=quantize), patch(
                'quantize_model.export_preprocessing', side_effect=attach) as export, patch.dict(sys.modules, {
                'qai_hub_models.models.grootn15': SimpleNamespace(quantize=recipe),
                'torch': SimpleNamespace()}), patch.object(sys, 'argv', ['quantize_model.py',
                '--checkpoint', str(checkpoint), '--dataset', str(dataset), '--output', str(output)]):
                quantize_model()
            export.assert_called_once()
            self.assertTrue((output/'preprocessing/embedding.npy').exists())

    def test_legacy_resources_are_generated_from_original_checkpoint(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root/'trained'
            processor = root/'processor'
            with patch('deployment_bundle.export_preprocessing') as export:
                export.side_effect = lambda checkpoint, source, **kwargs: make_preprocessing(source)
                with preprocessing_source(root/'old_quantized', checkpoint, processor) as source:
                    self.assertTrue((source/'embedding.npy').exists())
                    export.assert_called_once_with(checkpoint, source, processor_dir=processor)


if __name__ == '__main__':
    unittest.main()
