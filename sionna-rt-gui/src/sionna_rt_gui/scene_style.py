"""PolyU-style display lighting; never modifies the radio scene/materials."""
import re
import numpy as np
import mitsuba as mi

LABELS = {
    'simple_street_canyon': 'Street canyon',
    'simple_street_canyon_with_cars': 'Street with cars',
    'munich': 'Munich', 'florence': 'Florence',
    'etoile': 'Place de l Etoile', 'san_francisco': 'San Francisco',
    'floor_wall': 'Floor and wall', 'box': 'Box',
    'box_knife':'Box: knife-edge diffraction', 'box_one_screen':'Box: one screen',
    'box_two_screens':'Box: two screens', 'simple_wedge':'Wedge diffraction',
    'simple_reflector':'One reflector', 'double_reflector':'Two reflectors',
    'triple_reflector':'Three reflectors',
}

def scene_label(name, path):
    path=str(path).replace('\\','/')
    if 'polyu-stylized/' in path:return 'PolyU campus'
    if 'mong-kok/' in path:return 'Mong Kok | City'
    if 'kam-tin/' in path:return 'Kam Tin | Village'
    if 'polyu-official/' in path:return 'PolyU campus (original)'
    return LABELS.get(name,name)

def display_color(mesh):
    tokens=set(re.split(r'[^a-z]+',mesh.id().lower()))
    if tokens & {'terrain','ground','floor','road','pavement'}:return [.49,.57,.60]
    if tokens & {'roof'}:return [.66,.73,.75]
    if tokens & {'tree','vegetation','grass'}:return [.36,.53,.43]
    material=mesh.bsdf()
    kind=getattr(material,'itu_type','')
    return {'brick':[.76,.48,.34], 'glass':[.46,.66,.72],
            'metal':[.62,.69,.73], 'wood':[.65,.49,.36]}.get(kind,[.72,.68,.59])

def display_illumination(normals):
    from .presentation_style import key_light
    light=key_light()
    direct=np.maximum(normals@light,0)
    sky=.5+.5*np.maximum(normals[:,2],0)
    return (.40+sky[:,None]*.30)*np.array([.91,.95,1.12])+direct[:,None]*np.array([.40,.36,.29])


def bake_display(scene, mesh, base_color=None, smooth=False):
    vertices=mesh.vertex_positions_buffer().numpy().reshape(-1,3)
    faces=mesh.faces_buffer().numpy().reshape(-1,3)
    triangles=vertices[faces]
    shading=None
    if smooth and mesh.has_vertex_normals():
        shading=mesh.vertex_normals_buffer().numpy().reshape(-1,3)[faces]
    normals=np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0])
    normals/=np.maximum(np.linalg.norm(normals,axis=1,keepdims=True),1e-10)
    # Some source scenes wind a zero-thickness floor downwards. Its visible
    # upper side still receives daylight: offsetting rays below it otherwise
    # makes the floor shadow itself. Only orient horizontal sheet meshes;
    # keep closed building faces unchanged so walls retain their shading.
    extent=np.ptp(vertices,axis=0)
    horizontal_sheet=(extent[2]<=max(.001,float(extent.max())*1e-5)
                      and np.all(np.abs(normals[:,2])>.99))
    if horizontal_sheet:
        normals[normals[:,2]<0]*=-1
    vertices=triangles.reshape(-1,3)
    normals=np.repeat(normals,3,axis=0)
    shading=normals if shading is None else shading.reshape(-1,3)
    invalid=~np.isfinite(shading).all(axis=1)|(np.linalg.norm(shading,axis=1)<1e-10)
    shading[invalid]=normals[invalid]
    shading=shading/np.maximum(np.linalg.norm(shading,axis=1,keepdims=True),1e-10)
    if horizontal_sheet:shading[shading[:,2]<0]*=-1
    # Directional material shading only. Cast shadows belong in world-space
    # textures: interpolating sparse vertex visibility creates triangular blobs.
    illumination=display_illumination(shading)
    colors=np.clip(np.asarray(display_color(mesh) if base_color is None else base_color)*illumination,0,1).astype(np.float32)
    return vertices,np.arange(len(vertices),dtype=np.uint32).reshape(-1,3),colors


def baked_visual_assets(assets):
    """Cached shadow lighting on the exact display assets, independent of RF."""
    import hashlib
    import json
    import os
    from pathlib import Path
    from types import SimpleNamespace
    digest=hashlib.sha256(b'soft-display-directional-v3')
    for asset in assets:
        digest.update(Path(asset['mesh']).read_bytes())
        digest.update(json.dumps({'color':asset['color'],
                                  'face_normals':asset.get('face_normals',False)},
                                 sort_keys=True).encode())
    folder=Path(assets[0]['mesh']).resolve().parent.parent/'.display-cache'
    path=folder/(digest.hexdigest()+'.npz')
    if path.is_file():
        try:
            with np.load(path,allow_pickle=False) as data:
                return [(data[f'v{i}'],data[f'f{i}'],data[f'c{i}']) for i in range(len(assets))]
        except (OSError,ValueError,KeyError):
            pass  # An incomplete cache is disposable; source assets are intact.
    meshes=[mi.load_dict({'type':'ply','filename':asset['mesh'],
                         'face_normals':asset.get('face_normals',False)}) for asset in assets]
    visual_scene=mi.load_dict({'type':'scene',**{f'visual{i}':mesh for i,mesh in enumerate(meshes)}})
    context=SimpleNamespace(mi_scene=visual_scene)
    results=[bake_display(context,mesh,asset['color'],not asset.get('face_normals',False))
             for mesh,asset in zip(meshes,assets)]
    folder.mkdir(parents=True,exist_ok=True)
    temporary=folder/(path.stem+f'.{os.getpid()}.tmp.npz')
    np.savez(temporary,**{f'{name}{i}':array for i,result in enumerate(results)
                         for name,array in zip(('v','f','c'),result)})
    os.replace(temporary,path)
    return results
