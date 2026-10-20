import importlib.util
from pathlib import Path
import unittest
import numpy as np

spec=importlib.util.spec_from_file_location('coverage_accumulator',Path(__file__).parents[1]/'src/sionna_rt_gui/coverage_accumulator.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class AccumulatorTests(unittest.TestCase):
    def test_maximum_is_taken_after_weighted_averaging(self):
        a=module.CoverageAccumulator(3)
        a.add(0,np.array([[[100.,0.]]]),1)
        a.add(0,np.array([[[0.,0.]]]),9)
        a.add(1,np.array([[[20.,0.]]]),10)
        values,winners=a.result()
        np.testing.assert_allclose(values,[[[20.,0.]]])
        np.testing.assert_array_equal(winners,[[[1,-1]]])
        self.assertEqual(a.next_beam(10),2)
        a.add(2,np.array([[[5.,0.]]]),10)
        self.assertIsNone(a.next_beam(10))
        self.assertEqual(a.next_beam(20),0)


if __name__=='__main__':unittest.main()
