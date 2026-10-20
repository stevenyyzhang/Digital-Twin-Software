import importlib.util
from pathlib import Path
import unittest
import numpy as np

spec=importlib.util.spec_from_file_location('panel_codebook',
    Path(__file__).parents[1]/'src/sionna_rt_gui/panel_codebook.py')
panels=importlib.util.module_from_spec(spec)
spec.loader.exec_module(panels)


class PanelCodebookTests(unittest.TestCase):
    def test_every_direction_has_a_nearby_beam(self):
        _,_,local,world,_=panels.make_codebook()
        self.assertEqual(len(world),294)
        np.testing.assert_allclose(np.linalg.norm(world,axis=1),1.)
        np.testing.assert_allclose(np.linalg.norm(local,axis=1),1.)
        # Uniform sphere points include the full rear hemisphere and poles.
        rng=np.random.default_rng(42)
        targets=rng.normal(size=(20000,3))
        targets/=np.linalg.norm(targets,axis=1,keepdims=True)
        error=np.degrees(np.arccos(np.clip(np.max(targets@world.T,axis=1),-1,1)))
        self.assertLess(float(error.max()),11.)

    def test_six_panel_normals_are_exact_selectable_beams(self):
        _,_,_,world,_=panels.make_codebook()
        normals=np.array([[1,0,0],[0,1,0],[-1,0,0],[0,-1,0],[0,0,1],[0,0,-1]])
        centers=np.arange(6)*panels.PER_PANEL+panels.PER_PANEL//2
        np.testing.assert_allclose(world[centers],normals,atol=1e-14)
        np.testing.assert_array_equal(np.argmax(normals@world.T,axis=1),centers)

    def test_station_and_panel_rotations_compose_including_poles(self):
        for base in ([0,0,0],[.8,.3,-.2],[1.,np.pi/2,0]):
            for panel in panels.PANEL_ROTATIONS:
                expected=panels.rotation_matrix_numpy(base)@panel
                actual=panels.rotation_matrix_numpy(panels.matrix_to_orientation(expected))
                np.testing.assert_allclose(actual,expected,atol=1e-7)


if __name__=='__main__':unittest.main()
