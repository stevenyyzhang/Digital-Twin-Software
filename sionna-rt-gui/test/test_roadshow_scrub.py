"""Replay a fast native drag whose movement arrives on the release frame."""
from types import SimpleNamespace
from unittest.mock import Mock, patch
import unittest

from sionna_rt_gui import roadshow_ui


class FastSliderDrag(unittest.TestCase):
    def setUp(self):
        self.app = SimpleNamespace()
        self.io = SimpleNamespace(MousePos=(50., 10.), KeyCtrl=False,
                                  MouseDragThreshold=6.)
        self.ui = Mock()
        self.ui.GetIO.return_value = self.io
        self.ui.GetItemRectMin.return_value = (0., 0.)
        self.ui.GetItemRectMax.return_value = (204., 20.)
        self.ui.GetStyle.return_value = SimpleNamespace(GrabMinSize=20.)
        self.patcher = patch.object(roadshow_ui, 'ui', self.ui)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)

    def frame(self, value, x, pressed=False, released=False, down=False,
              active=False, native=None):
        self.io.MousePos = (x, 10.)
        self.ui.IsItemActive.return_value = active
        self.ui.IsMouseClicked.return_value = pressed
        self.ui.IsMouseReleased.return_value = released
        self.ui.IsMouseDown.return_value = down
        self.ui.SliderFloat.return_value = native or (False, value)
        return roadshow_ui.scrub_slider(self.app, 'route', value, 0., 12., '%.1f')

    def test_release_frame_commits_final_position_and_preserves_grab_offset(self):
        # Track is [12, 192]; at 6 m the handle center is 102 px.
        self.frame(6., 106., pressed=True, down=True, active=True)
        changed, value = self.frame(6., 166., released=True)
        self.assertTrue(changed)
        self.assertAlmostEqual(value, 10.)
        self.assertEqual(self.app._scrub_owners, {})

    def test_reverse_drag_clamps_to_start(self):
        self.frame(12., 192., pressed=True, down=True, active=True)
        self.assertEqual(self.frame(12., -20., released=True), (True, 0.))

    def test_drag_from_another_widget_cannot_change_value(self):
        self.frame(6., 102., pressed=True, down=True, active=False)
        self.assertEqual(self.frame(6., 190., released=True), (False, 6.))

    def test_ctrl_entry_keeps_native_value(self):
        self.io.KeyCtrl = True
        self.frame(6., 102., pressed=True, down=True, active=True)
        self.assertEqual(self.frame(6., 190., released=True, native=(True, 7.5)),
                         (True, 7.5))


if __name__ == '__main__':
    unittest.main()
