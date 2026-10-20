import copy
import unittest
from sionna_rt_gui.link_jobs import same_environment


class LinkResultIsolation(unittest.TestCase):
    def test_only_receiver_translation_is_compatible(self):
        a=dict(scene='a.xml',panel=0,tx=[('tx-0',[0,0,5],[0,0,0])],
               rx=[('rx-0',[10,0,1.5],[0,0,0])],frequency=28e9,paths={'max_depth':5})
        moved=copy.deepcopy(a);moved['rx'][0][1][0]=12
        self.assertTrue(same_environment(a,moved))
        for key,value in [('scene','b.xml'),('panel',1),('frequency',3.5e9),('paths',{'max_depth':2})]:
            changed={**moved,key:value}
            self.assertFalse(same_environment(a,changed),key)
        moved['rx'][0][2][0]=.5
        self.assertFalse(same_environment(a,moved))
        self.assertFalse(same_environment(None,a))
