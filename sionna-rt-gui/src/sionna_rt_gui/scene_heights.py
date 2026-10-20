"""Height presets measured from the loaded RF geometry, never guessed storeys."""
import re
import json
from pathlib import Path
import numpy as np
import mitsuba as mi
from .terrain_coverage import GROUND_TOKENS


def measure_scene_heights(scene):
    shapes=list(scene.mi_scene.shapes())
    ground=[];buildings=[]
    for shape in shapes:
        tokens=set(re.split(r'[^a-z]+',shape.id().lower()))
        if tokens & GROUND_TOKENS:ground.append(shape)
        elif not tokens & {'water','tree','trees','vegetation'}:buildings.append(shape)
    bbox=scene.mi_scene.bbox()
    lo=np.asarray(list(bbox.min),dtype=float);hi=np.asarray(list(bbox.max),dtype=float)
    step=max(3.,float(max(hi[:2]-lo[:2]))/160)
    x,y=np.meshgrid(np.arange(lo[0]+step/2,hi[0],step),np.arange(lo[1]+step/2,hi[1],step))
    hit=scene.mi_scene.ray_intersect(mi.Ray3f(
        mi.Point3f(x.ravel(),y.ravel(),np.full(x.size,hi[2]+10.)),mi.Vector3f(0,0,-1)))
    valid=hit.is_valid().numpy();z=hit.p.z.numpy()
    gm=np.zeros(x.size,bool);bm=gm.copy()
    for shape in ground:gm|=(hit.shape==shape).numpy()
    for shape in buildings:bm|=(hit.shape==shape).numpy()
    grounds=z[valid & gm & np.isfinite(z)]
    if not len(grounds):
        return dict(available=False,reason='No exposed named ground samples; set model-Z heights manually.')
    reference=float(np.median(grounds));upper_ground=float(np.percentile(grounds,90))
    roofs=z[valid & bm & np.isfinite(z) & (z>reference+2.) & (np.abs(hit.n.z.numpy())>.5)]
    if not buildings or not len(roofs):
        return dict(available=False,reason='No elevated building surfaces; set model-Z heights manually.')
    highest=max(float(shape.bbox().max.z) for shape in buildings)
    roof50=float(np.percentile(roofs,50));roof90=float(np.percentile(roofs,90))
    # Six uniform scene-Z levels; each increment is a multiple of 5 m.
    step=max(5.,float(np.ceil((highest+5.)/30.)*5.))
    layers=[dict(label='Uniform height',z_m=step*i) for i in range(1,7)]
    return dict(available=True,ground_reference_z_m=reference,upper_ground_z_m=upper_ground,
        roof_median_z_m=roof50,roof_p90_z_m=roof90,highest_building_z_m=highest,
        max_z_minus_reference_m_not_building_height=highest-reference,ground_samples=len(grounds),roof_samples=len(roofs),
        sample_spacing_m=step,layers=layers,
        method='Exposed horizontal surface heights sampled by XY area; highest is exact building-shape bbox Z.',
        reference='Model coordinates. Ground is a reference surface, not certified terrain or sea-level datum.')


def configure_scene_heights(app):
    profile=getattr(app.scene,'_air_height_profile',None)
    if profile is None:
        profile=measure_scene_heights(app.scene);app.scene._air_height_profile=profile
    calibration=Path(app.cfg.scene_filename).parent/'height-calibration.json'
    if calibration.exists():
        profile['source_verification']=json.loads(calibration.read_text(encoding='utf-8'))
    app.air_height_profile=profile
    if app.cfg.radio_map.air_height_mode=='buildings':
        if profile['available']:
            heights=[layer['z_m'] for layer in profile['layers']]
        else:
            step=max(5.,float(np.ceil((float(app.scene.mi_scene.bbox().max.z)+5.)/30.)*5.))
            heights=[step*i for i in range(1,7)]
        app.cfg.radio_map.air_heights=heights
        app.cfg.radio_map.air_selected_layer=0
        app.air_waypoint_z=heights[-1]


def height_labels(app):
    c=app.cfg.radio_map;profile=getattr(app,'air_height_profile',{})
    if c.air_height_mode=='buildings' and profile.get('available'):
        return [f"{layer['z_m']:.2f} m | {layer['label']}" for layer in profile['layers']]
    return [f'Model Z = {h:g} m' for h in c.air_heights]
