"""CPU regression tests for camera grouping; no model weights required."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from quantization_samples import load_samples


class CameraSamplesTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.base = Path(self.directory.name)
        self.sink = SimpleNamespace(_io_dir=lambda *args: self.base)

    def write_views(self, name, shape):
        for i in range(4):
            np.save(self.base / f'{name}_{i:04d}.npy', np.full(shape, i, dtype=np.float32))

    def test_two_steps_keep_both_cameras_in_order(self):
        self.write_views('pixel_values', (3, 4, 4))
        samples = load_samples(self.sink, 'vit',
                               {'pixel_values': ((2, 3, 4, 4), 'float32')},
                               'inputs', 2, num_cameras=2)['pixel_values']
        self.assertEqual(len(samples), 2)
        self.assertEqual(samples[0].shape, (2, 3, 4, 4))
        np.testing.assert_array_equal(samples[0][:, 0, 0, 0], [0, 1])
        np.testing.assert_array_equal(samples[1][:, 0, 0, 0], [2, 3])

    def test_reference_outputs_keep_same_camera_groups(self):
        self.write_views('vit_embeds', (5, 7))
        samples = load_samples(self.sink, 'vit', {'vit_embeds': None},
                               'outputs', 2, num_cameras=2)['vit_embeds']
        self.assertEqual(samples[1].shape, (2, 5, 7))
        np.testing.assert_array_equal(samples[1][:, 0, 0], [2, 3])

    def test_missing_camera_fails(self):
        np.save(self.base / 'pixel_values_0000.npy', np.zeros((3, 4, 4)))
        with self.assertRaises(FileNotFoundError):
            load_samples(self.sink, 'vit', {'pixel_values': ((2, 3, 4, 4), 'float32')},
                         'inputs', 1, num_cameras=2)

    def test_wrong_image_shape_fails(self):
        self.write_views('pixel_values', (3, 4, 4))
        with self.assertRaises(ValueError):
            load_samples(self.sink, 'vit', {'pixel_values': ((2, 3, 8, 8), 'float32')},
                         'inputs', 1, num_cameras=2)

    def test_other_components_preserve_batch_dimension(self):
        values = np.arange(12).reshape(3, 4)
        self.sink.load = lambda *args, **kwargs: {'state': values}
        samples = load_samples(self.sink, 'dit', {'state': None},
                               'inputs', 3, num_cameras=2)['state']
        self.assertEqual(samples[1].shape, (1, 4))
        np.testing.assert_array_equal(samples[1], values[1:2])


if __name__ == '__main__':
    unittest.main()
