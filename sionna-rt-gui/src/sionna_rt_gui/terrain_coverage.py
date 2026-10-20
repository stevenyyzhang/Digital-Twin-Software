"""Outdoor measurement patches following semantic ground meshes at +1.5 m.

The solver uses a real Mitsuba triangle mesh, not a fixed-height map whose
pixels are merely moved for display. Each grid cell is a horizontal patch at
its sampled ground height; discontinuous elevations are not bridged.
"""
import re
import json
from pathlib import Path
import numpy as np
import mitsuba as mi
import sionna.rt as rt
from .coverage_sampling import independent_launch_directions

GROUND_TOKENS={'ground','terrain','road','plaza','lawn','grass','floor','pavement','sidewalk','walkway'}


def overlay_matches(surface_match,support_match,top_z,ground_z,normal_z,rule):
    """Only explicitly designated low finishes over their ground support qualify."""
    gap=top_z-ground_z
    return (surface_match & support_match & np.isfinite(gap)
            & (gap>=-.02) & (gap<=rule['max_vertical_gap_m'])
            & (np.abs(normal_z)>=rule['min_abs_normal_z']))


def build_surface(scene, settings):
    c=settings['solver']
    if not np.allclose(c['orientation'],0):
        raise ValueError('Terrain coverage requires an XY grid (orientation 0,0,0).')
    grid=rt.PlanarRadioMap(scene,mi.Point2f(c['cell_size']),c['center'],[0,0,0],c['size'])
    centers=grid.cell_centers.numpy().copy()
    flat=centers.reshape(-1,3)
    shapes=[s for s in scene.mi_scene.shapes()
            if GROUND_TOKENS.intersection(re.split(r'[^a-z]+',s.id().lower()))]
    if not shapes:
        raise ValueError('No named ground/terrain/road/plaza/lawn mesh found; designate ground meshes before terrain coverage.')
    ground=mi.load_dict(dict(type='scene',**{s.id():s for s in shapes}))
    top=float(scene.mi_scene.bbox().max.z)+10
    ray=mi.Ray3f(mi.Point3f(flat[:,0],flat[:,1],np.full(len(flat),top)),mi.Vector3f(0,0,-1))
    hit=ground.ray_intersect(ray)
    visible=scene.mi_scene.ray_intersect(ray)
    elevation=hit.p.z.numpy()
    # Explicitly exclude roofs, water, and ground covered by a building/bridge.
    valid=hit.is_valid().numpy() & visible.is_valid().numpy()
    visible_ground=np.zeros(len(flat),dtype=bool)
    for shape in shapes:visible_ground |= (visible.shape==shape).numpy()
    direct_ground=visible_ground & (np.abs(elevation-visible.p.z.numpy())<.25) & (np.abs(hit.n.z.numpy())>.5)
    overlays=np.zeros(len(flat),dtype=bool)
    scene_path=settings.get('scene')
    rules_path=Path(scene_path).parent/'coverage-surfaces.json' if scene_path else None
    rules_text=settings.get('surface_rules')
    if rules_text is None and rules_path is not None and rules_path.exists():
        rules_text=rules_path.read_text(encoding='utf-8')
    if rules_text:
        rules=json.loads(rules_text)
        by_id={shape.id():shape for shape in scene.mi_scene.shapes()}
        for rule in rules.get('ground_overlays',[]):
            surface=by_id[rule['surface']];support=by_id[rule['support']]
            overlays |= overlay_matches((visible.shape==surface).numpy(),(hit.shape==support).numpy(),
                                        visible.p.z.numpy(),elevation,visible.n.z.numpy(),rule)
    valid &= direct_ground | overlays
    # Use the actual exposed finish elevation, never the buried plaza elevation.
    elevation=np.where(overlays,visible.p.z.numpy(),elevation)
    flat[:,2]=np.where(valid,elevation+settings.get('height_above_ground',1.5),np.nan)
    mask=valid.reshape(centers.shape[:2])
    if not valid.any():raise ValueError('No outdoor ground cells in the selected extent.')
    half=np.asarray(c['cell_size'])/2
    offsets=np.array([[-half[0],-half[1],0],[half[0],-half[1],0],
                      [half[0],half[1],0],[-half[0],half[1],0]])
    vertices=(flat[valid,None,:]+offsets).reshape(-1,3).astype(np.float32)
    faces=(np.arange(valid.sum())[:,None,None]*4+np.array([[0,1,2],[0,2,3]])).reshape(-1,3).astype(np.uint32)
    mesh=mi.Mesh('ground_measurement_1p5m',len(vertices),len(faces))
    params=mi.traverse(mesh)
    params['vertex_positions']=vertices.ravel()
    params['faces']=faces.ravel()
    params.update()
    return mesh,centers,mask,[s.id() for s in shapes]


def solve_surface(solver,scene,surface,settings,precoding_vec,samples,seed):
    mesh,centers,mask,_=surface
    options={k:v for k,v in settings['solver'].items()
             if k not in ('center','size','orientation','cell_size','measurement_surface')}
    with independent_launch_directions(seed):
        solved=solver(scene,measurement_surface=mesh,**options,precoding_vec=precoding_vec,
                      samples_per_tx=samples,seed=seed)
    # The two flat triangles have equal areas: average their power estimates.
    values=np.zeros((len(scene.transmitters),*mask.shape),dtype=np.float32)
    values[:,mask]=solved.path_gain.numpy().reshape(len(scene.transmitters),-1,2).mean(axis=-1)
    return values


class TerrainDisplayMap(rt.PlanarRadioMap):
    """Grid view of mesh-solved values; the inherited plane is never solved."""
    @property
    def cell_centers(self):
        return self.terrain_centers


def display_map(scene,c,centers,mask,values):
    rm=TerrainDisplayMap(scene,mi.Point2f(c.cell_size),c.center,[0,0,0],c.size)
    rm.terrain_centers=mi.TensorXf(centers)
    rm.demo_outdoor_mask=mask
    rm._pathgain_map=mi.TensorXf(values)
    return rm
