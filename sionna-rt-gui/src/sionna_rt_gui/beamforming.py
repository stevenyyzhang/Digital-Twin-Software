"""Shared unit-norm TX codebook for Sionna CIR and radio-map solvers."""
from dataclasses import asdict
from contextlib import contextmanager
from functools import wraps
import numpy as np
import mitsuba as mi
from sionna.rt.utils.geometry import rotation_matrix
from .host_values import host_value


def rotation(tx):
    from .panel_codebook import rotation_matrix_numpy
    return rotation_matrix_numpy(host_value(tx, "orientation").ravel())


def panel_orientation(tx, panel):
    from .panel_codebook import PANEL_ROTATIONS, matrix_to_orientation
    return matrix_to_orientation(rotation(tx) @ PANEL_ROTATIONS[panel])


@contextmanager
def active_panel(app):
    """Solve with the selected panel pose, then restore the station frame."""
    devices=list(app.scene.transmitters.values())
    original=[tx.orientation.numpy().ravel().copy() for tx in devices]
    try:
        for tx in devices:
            tx.orientation=mi.Point3f(panel_orientation(tx,app.beam_demo.panel))
        yield
    finally:
        for tx, orientation in zip(devices,original):
            tx.orientation=mi.Point3f(orientation)


def panel_solver(method):
    @wraps(method)
    def solve(app, *args, **kwargs):
        if not app.cfg.demo_mode:
            return method(app,*args,**kwargs)
        with active_panel(app):
            return method(app,*args,**kwargs)
    return solve


def weights(scene, directions_local, tx=None):
    """Sionna uses exp(+j k.r) array response, hence negative steering phase."""
    if scene.tx_array.num_ant != scene.tx_array.array_size:
        raise ValueError('Beam scan currently requires single-polarized TX antennas')
    positions=scene.tx_array.normalized_positions.numpy().T
    result=np.exp(-2j*np.pi*(np.atleast_2d(directions_local)@positions.T))
    return (result/np.sqrt(len(positions))).astype(np.complex64)


def precoding(scene, direction_local):
    w=weights(scene,direction_local)[0]
    w=np.tile(w,(len(scene.transmitters),1))
    return mi.TensorXf(w.real.copy()),mi.TensorXf(w.imag.copy())


def transmit_gain(scene, directions_local, steering_direction):
    """Selected-beam far-field gain, including Sionna's element pattern (dBi).

    This is a free-space antenna property, not RSS or a propagation solution.
    """
    import drjit as dr
    u = np.asarray(directions_local)
    positions = scene.tx_array.normalized_positions.numpy().T
    response = np.exp(2j*np.pi*(u@positions.T))
    array_gain = np.abs(response@weights(scene, steering_direction)[0])**2
    theta = mi.Float(np.arccos(np.clip(u[:,2], -1, 1)))
    phi = mi.Float(np.arctan2(u[:,1], u[:,0]))
    ct, cp = scene.tx_array.antenna_pattern.patterns[0](theta, phi)
    element_gain = (dr.abs(ct)**2+dr.abs(cp)**2).numpy()
    return 10*np.log10(np.maximum(array_gain*element_gain, 1e-12))


def path_powers(amplitudes, w):
    """Coherent across TX elements, incoherent across resolved paths.

    Result [beam, path], single RX element / first Tx-Rx link.
    Matches radio-map path-energy convention, not narrowband coherent fading.
    """
    a=np.asarray(amplitudes)
    if a.shape[0]!=1 or a.shape[1]!=1 or a.shape[2]!=1:
        raise ValueError('Beam scan requires one TX and one single-element RX')
    h=a[0,0,0,:,:,0]
    if h.shape[0]!=w.shape[1]:raise ValueError('CIR and codebook array sizes differ')
    return np.abs(w@h)**2


def link_signature(app):
    def devices(items):
        return [(d.name,host_value(d,"position").ravel().tolist(),host_value(d,"orientation").ravel().tolist()) for d in items]
    return dict(scene=app.cfg.scene_filename,panel=app.beam_demo.panel,tx=devices(app.scene.transmitters.values()),
        rx=devices(app.scene.receivers.values()),power=[float(host_value(t,"power_dbm")[0]) for t in app.scene.transmitters.values()],
        frequency=float(host_value(app.scene,"frequency")[0]),tx_array=asdict(app.cfg.tx_array),rx_array=asdict(app.cfg.rx_array),
        paths=asdict(app.cfg.paths),thickness=app.cfg.radio_material_thickness,scattering=app.cfg.radio_material_scattering_coefficient)
