import importlib.util
from pathlib import Path
import unittest
import numpy as np

spec=importlib.util.spec_from_file_location('roadshow_results',Path(__file__).parents[1]/'src/sionna_rt_gui/roadshow_results.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

class ResultSemantics(unittest.TestCase):
    def test_unknown_and_excluded_do_not_count_as_weak_signal(self):
        summary,rss=module.summarize(np.array([[1e-10,0],[1e-9,1e-7]]),np.array([[1,1],[1,0]],bool),0,-95)
        self.assertEqual(summary['total'],3)
        self.assertEqual(summary['observed'],2)
        self.assertEqual(summary['unknown'],1)
        self.assertEqual(summary['above_threshold_fraction'],.5)
        self.assertTrue(np.isnan(rss[0,1]) and np.isnan(rss[1,1]))
    def test_comparison_requires_same_station_and_common_valid_cells(self):
        a=dict(settings=dict(scene='A',transmitters=[1],mode='current',beam_index=1),rss=np.array([-80.,np.nan,-100.]))
        b=dict(settings=dict(scene='A',transmitters=[1],mode='max',beam_index=-1),rss=np.array([-70.,-60.,np.nan]))
        result,_=module.compare(b,a)
        self.assertEqual(result,dict(cells=1,median_delta_db=10.))
        b['settings']['transmitters']=[2]
        self.assertIsNone(module.compare(b,a)[0])

if __name__=='__main__':unittest.main()
