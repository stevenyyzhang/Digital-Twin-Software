"""Six co-located switched panels; only one panel transmits at a time.

Each panel scans +/-45 degrees in local azimuth/elevation, including its
normal. The six sectors cover the sphere with overlap at sector boundaries.
"""
import numpy as np

PANEL_NAMES = ['Front +X', 'Left +Y', 'Back -X', 'Right -Y', 'Up +Z', 'Down -Z']
ANGLES = np.arange(-45., 46., 15.)
SIDE = len(ANGLES)
PER_PANEL = SIDE*SIDE


def rotation_matrix_numpy(angles):
    a, b, c = angles
    ca, sa, cb, sb, cc, sc = np.cos(a), np.sin(a), np.cos(b), np.sin(b), np.cos(c), np.sin(c)
    return np.array([[ca*cb, ca*sb*sc-sa*cc, ca*sb*cc+sa*sc],
                     [sa*cb, sa*sb*sc+ca*cc, sa*sb*cc-ca*sc],
                     [-sb, cb*sc, cb*cc]])


PANEL_ROTATIONS = np.array([rotation_matrix_numpy(np.radians(a)) for a in
                           [(0,0,0), (90,0,0), (180,0,0), (-90,0,0), (0,-90,0), (0,90,0)]])


def matrix_to_orientation(rot):
    """Sionna Z-Y-X Euler angles, including vertical panel gimbal positions."""
    beta = np.arcsin(np.clip(-rot[2,0], -1., 1.))
    if abs(np.cos(beta)) > 1e-7:
        return np.array([np.arctan2(rot[1,0], rot[0,0]), beta, np.arctan2(rot[2,1], rot[2,2])])
    return np.array([np.arctan2(-rot[0,1], rot[1,1]), beta, 0.])


def directions(az, el):
    a, e = np.radians(az), np.radians(el)
    return np.stack([np.cos(e)*np.cos(a), np.cos(e)*np.sin(a), np.sin(e)], axis=-1)


def make_codebook():
    az, el = np.meshgrid(ANGLES, ANGLES)
    az, el = az.ravel(), el.ravel()
    panel_local = directions(az, el)
    station_local = np.concatenate([panel_local @ r.T for r in PANEL_ROTATIONS])
    corners = np.stack([directions(np.clip(az+a,-45,45), np.clip(el+e,-45,45))
                        for a,e in [(-7.5,-7.5),(7.5,-7.5),(7.5,7.5),(-7.5,7.5)]], axis=1)
    vertices = np.concatenate([corners @ r.T for r in PANEL_ROTATIONS]).reshape(-1,3)
    return np.tile(az,6), np.tile(el,6), np.tile(panel_local,(6,1)), station_local, vertices
