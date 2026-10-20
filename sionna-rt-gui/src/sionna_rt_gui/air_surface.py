"""Independent horizontal measurement planes; no interpolation between heights."""
import numpy as np
import mitsuba as mi
from .beamforming import precoding
from .panel_codebook import PER_PANEL, PANEL_ROTATIONS, rotation_matrix_numpy, matrix_to_orientation
from .coverage_sampling import independent_launch_directions


def solve_air(solver, scene, settings, job, panel_dirs):
    beam=job['beam_index']
    for data in settings['transmitters']:
        matrix=rotation_matrix_numpy(data['orientation'])@PANEL_ROTATIONS[beam//PER_PANEL]
        scene.get(data['name']).orientation=mi.Point3f(matrix_to_orientation(matrix))
    options=dict(settings['solver'])
    options.pop('measurement_surface',None)
    options['center']=list(options['center'])
    options['center'][2]=float(job['height'])
    options['orientation']=[0.,0.,0.]
    with independent_launch_directions(job['seed']):
        rm=solver(scene,**options,precoding_vec=precoding(scene,panel_dirs[beam]),
                  samples_per_tx=job['samples'],seed=job['seed'])
    centers=rm.cell_centers.numpy().copy()
    flat=centers.reshape(-1,3)
    # Conservative open-sky mask: reject everything below the topmost surface.
    # This intentionally excludes underpasses; it is NOT a watertight inside test.
    # Buildings still obstruct and reflect rays in the full solver above.
    top=max(float(scene.mi_scene.bbox().max.z)+10.,float(job['height'])+10.)
    hit=scene.mi_scene.ray_intersect(mi.Ray3f(
        mi.Point3f(flat[:,0],flat[:,1],np.full(len(flat),top)),mi.Vector3f(0,0,-1)))
    valid=hit.is_valid().numpy()
    surface_z=hit.p.z.numpy()
    mask=(valid & np.isfinite(surface_z) & (flat[:,2]>surface_z+.25)).reshape(centers.shape[:2])
    values=rm.path_gain.numpy().copy()
    if not np.isfinite(values).all():raise ValueError('Non-finite air coverage result')
    values[:,~mask]=0.
    return values,centers,mask
