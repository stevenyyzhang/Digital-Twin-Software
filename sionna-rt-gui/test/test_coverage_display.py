"""Display projection must preserve unknown regions and measured cell values."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

spec = importlib.util.spec_from_file_location(
    'coverage_display', Path(__file__).parents[1]/'src/sionna_rt_gui/coverage_display.py')
display = importlib.util.module_from_spec(spec)
spec.loader.exec_module(display)


class CoverageDisplayTests(unittest.TestCase):
    def test_smoothing_preserves_centers_unknowns_and_source(self):
        gain = np.arange(1., 17.).reshape(4, 4)
        gain[1, 1] = 0
        mask = np.ones_like(gain, dtype=bool)
        mask[2, 2] = False
        original = gain.copy()
        x, y = np.meshgrid(np.arange(4)+.5, np.arange(4)+.5)
        sampled, outdoor = display.sample_grid(gain, mask, x, y, True)
        valid = mask & (gain > 0)
        np.testing.assert_allclose(sampled[valid], gain[valid])
        self.assertTrue(np.isnan(sampled[~valid]).all())
        np.testing.assert_array_equal(outdoor, mask)
        np.testing.assert_array_equal(gain, original)

    def test_rotation_and_y_flip_preserve_map_registration(self):
        cfg = SimpleNamespace(smooth_display=False, color_map='turbo', vmin=0, vmax=20)
        gain = np.array([[1., 0.], [10., 100.]])
        mask = np.array([[True, True], [False, True]])
        # 90 degree rotation about the origin, translated into a 2 x 2 image.
        image = display.plan_overlay(gain, mask, [2., 0.], np.array([[0., -1.], [1., 0.]]),
                                     [0., 0.], 2., 2, 0., cfg)
        self.assertLess(image[1, 0, 3], .7)  # excluded: dark hatch
        np.testing.assert_allclose(image[0, 1], [.16, .20, .36, .88])  # unknown: blue-gray
        self.assertAlmostEqual(float(image[0, 0, 3]), .8)
        self.assertFalse(np.allclose(image[0, 0, :3], image[1, 1, :3]))

    def test_cell_edges_are_continuous_only_when_supported(self):
        gain = np.arange(1., 26.).reshape(5, 5)
        mask = np.ones_like(gain, dtype=bool)
        left, _ = display.sample_grid(gain, mask, np.array([2.-1e-8]), np.array([2.3]), True)
        right, _ = display.sample_grid(gain, mask, np.array([2.+1e-8]), np.array([2.3]), True)
        np.testing.assert_allclose(left, right, atol=1e-6)
        values, outdoor = display.sample_grid(gain, mask, np.array([-1., 5.]), np.array([0., 0.]), True)
        self.assertTrue(np.isnan(values).all())
        self.assertFalse(outdoor.any())


if __name__ == '__main__':
    unittest.main()
