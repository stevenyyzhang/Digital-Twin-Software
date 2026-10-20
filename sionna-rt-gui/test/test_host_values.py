"""CPU mirrors must never conceal a changed Sionna device property."""
import unittest
import numpy as np
import sionna.rt as rt
from sionna.rt.utils.geometry import rotation_matrix
from sionna_rt_gui.host_values import host_value, set_host_position
from sionna_rt_gui.beamforming import rotation


class HostValueTests(unittest.TestCase):
    def setUp(self):
        self.tx = rt.Transmitter('test', [1, 2, 3], orientation=[.2, .3, .4])

    def test_unchanged_value_reuses_immutable_cpu_copy(self):
        value = host_value(self.tx, 'position')
        self.assertIs(value, host_value(self.tx, 'position'))
        self.assertFalse(value.flags.writeable)

    def test_assignment_and_inplace_component_change_invalidate(self):
        old = host_value(self.tx, 'position')
        self.tx.position = [4, 5, 6]
        np.testing.assert_allclose(host_value(self.tx, 'position').ravel(), [4, 5, 6])
        self.tx.position.x[0] = 9
        np.testing.assert_allclose(host_value(self.tx, 'position').ravel(), [9, 5, 6])
        np.testing.assert_allclose(old.ravel(), [1, 2, 3])

    def test_cpu_rotation_matches_sionna_after_orientation_edit(self):
        for angles in ([.2, .3, .4], [1.2, -np.pi / 2, -.7]):
            self.tx.orientation = angles
            np.testing.assert_allclose(rotation(self.tx),
                rotation_matrix(self.tx.orientation).numpy().reshape(3, 3), atol=1e-6)

    def test_seeded_position_exactly_matches_device_rounding_and_invalidates(self):
        set_host_position(self.tx, [1.123456789, -234.987654321, 9.87654321])
        value = host_value(self.tx, 'position')
        np.testing.assert_array_equal(value, self.tx.position.numpy())
        self.assertIs(value, self.tx._gui_host_values['position'][2])
        self.tx.position.y[0] = 17
        np.testing.assert_array_equal(host_value(self.tx, 'position'), self.tx.position.numpy())
        self.assertNotEqual(value[1, 0], 17)
