"""World-space shadow textures on unchanged horizontal display triangles."""
import hashlib
import inspect
import json
import os
from pathlib import Path

import mitsuba as mi
import numpy as np

PIXEL_SIZE = .75
MAX_TEXTURE_SIZE = 2048
_VERSION = b'linear-shadow-split-normal-v6'


def planar_face_groups(vertices, faces):
    """Partition large horizontal levels; leave small and sloping faces alone."""
    triangles=vertices[faces]
    cross=np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0])
    areas=np.linalg.norm(cross,axis=1)*.5
    flat=(np.ptp(triangles[:,:,2],axis=1)<.001)&(areas>1e-7)
    levels=np.round(triangles[:,:,2].mean(axis=1),3)
    groups=[]
    for level in np.unique(levels[flat]):
        selected=np.flatnonzero(flat&(levels==level))
        area=float(areas[selected].sum())
        xy=triangles[selected,:,:2].reshape(-1,2)
        extent=np.ptp(xy,axis=0)
        # Avoid allocating a campus-sized texture for a few tiny roof details.
        if area<64 or extent.min()<1 or float(np.prod(extent))>area*20:continue
        groups.append((area,float(level),selected))
    groups.sort(key=lambda value:-value[0])
    return [(level,selected) for _,level,selected in groups[:16]]


def texture_coordinates(vertices, pixel_size=PIXEL_SIZE):
    lo=vertices[:,:2].min(axis=0);hi=vertices[:,:2].max(axis=0)
    extent=hi-lo
    spacing=max(pixel_size,float(extent.max())/(MAX_TEXTURE_SIZE-2))
    shape=np.maximum(2,np.ceil(extent/spacing).astype(int)+2)
    # One sample border keeps interpolation stable at mesh boundaries.
    origin=lo-spacing*.5
    uv=(vertices[:,:2]-origin)/(shape*spacing)
    return uv.astype(np.float32),origin,spacing,tuple(shape)


def _texture(scene, origin, spacing, shape, height, base_color):
    width,height_px=shape
    xx,yy=np.meshgrid(origin[0]+(np.arange(width)+.5)*spacing,
                      origin[1]+(np.arange(height_px)+.5)*spacing)
    positions=np.column_stack((xx.ravel(),yy.ravel(),np.full(xx.size,height+.025)))
    from .presentation_style import key_light
    light=key_light()
    tangent=np.cross(light,[0.,0.,1.]);tangent/=np.linalg.norm(tangent)
    bitangent=np.cross(light,tangent)
    sun=[light]
    for index in range(12):
        phase=index*np.pi*(3-np.sqrt(5))
        radius=.065*np.sqrt((index+.5)/12)
        direction=light+radius*(np.cos(phase)*tangent+np.sin(phase)*bitangent)
        sun.append(direction/np.linalg.norm(direction))
    # Cosine-weighted hemisphere: dense sky sampling removes four-ray bands.
    sky=[]
    for index in range(32):
        radius=np.sqrt((index+.5)/32);angle=index*np.pi*(3-np.sqrt(5))
        sky.append([radius*np.cos(angle),radius*np.sin(angle),np.sqrt(1-radius*radius)])
    colors=np.empty((len(positions),3),dtype=np.float32)
    for start in range(0,len(positions),262144):
        stop=min(start+262144,len(positions))
        origins=mi.Point3f(positions[start:stop].T)
        visible=np.zeros(stop-start,dtype=np.float32)
        for direction in sun:
            visible+=~scene.ray_test(mi.Ray3f(origins,mi.Vector3f(direction))).numpy()
        visible/=len(sun)
        sky_visible=np.zeros(stop-start,dtype=np.float32)
        for direction in sky:
            ray=mi.Ray3f(origins,mi.Vector3f(np.asarray(direction)));ray.maxt=20
            sky_visible+=~scene.ray_test(ray).numpy()
        sky_visible/=len(sky)
        from .presentation_style import lighting_components,linear,srgb
        fill,direct=lighting_components(np.array([[0.,0.,1.]]))
        illumination=fill*(.40+.60*sky_visible[:,None])+direct*visible[:,None]
        colors[start:stop]=srgb(linear(base_color)*illumination)
    return colors.reshape(height_px,width,3)


def mesh_parts(scene_or_factory, mesh, base_color, cache_key, cache_dir, smooth=False):
    """Return disjoint original-face parts, with textures or vertex lighting."""
    from .scene_style import bake_display, display_illumination
    from .facade_lighting import facade_visibility
    from .presentation_style import lighting_components,linear,srgb,key_light
    vertices=mesh.vertex_positions_buffer().numpy().reshape(-1,3)
    faces=mesh.faces_buffer().numpy().reshape(-1,3)
    digest=hashlib.sha256(_VERSION+cache_key)
    digest.update(str((PIXEL_SIZE,MAX_TEXTURE_SIZE)).encode())
    for function in (planar_face_groups,texture_coordinates,_texture):
        digest.update(inspect.getsource(function).encode())
    digest.update(vertices.tobytes());digest.update(faces.tobytes())
    digest.update(np.asarray(base_color,dtype=np.float32).tobytes())
    digest.update(str(smooth).encode());digest.update(inspect.getsource(display_illumination).encode())
    digest.update(inspect.getsource(facade_visibility).encode())
    digest.update(inspect.getsource(lighting_components).encode())
    digest.update(key_light().tobytes())
    path=Path(cache_dir)/('planar-'+digest.hexdigest()+'.npz')
    if path.is_file():
        try:
            with np.load(path,allow_pickle=False) as data:
                kinds=json.loads(str(data['kinds']))
                return [{name:data[f'{i}_{name}'] for name in names} for i,names in enumerate(kinds)]
        except (OSError,KeyError,ValueError):pass
    parts=[];remaining=np.ones(len(faces),dtype=bool)
    groups=planar_face_groups(vertices,faces)
    scene=scene_or_factory() if callable(scene_or_factory) else scene_or_factory
    for height,selected in groups:
        used,inverse=np.unique(faces[selected].ravel(),return_inverse=True)
        v=vertices[used].copy();f=inverse.reshape(-1,3).astype(np.uint32)
        uv,origin,spacing,shape=texture_coordinates(v)
        texture=_texture(scene,origin,spacing,shape,height,base_color)
        parts.append(dict(vertices=v,faces=f,uv=uv,texture=texture,source_faces=selected,baked_linear=np.array(True)))
        remaining[selected]=False
    if remaining.any():
        # bake_display preserves original face order and only computes shading.
        v,_,colors=bake_display(None,mesh,base_color,smooth)
        selected=np.flatnonzero(remaining)
        v=v.reshape(-1,3,3)[selected].reshape(-1,3)
        if smooth and mesh.has_vertex_normals():
            normals=mesh.vertex_normals_buffer().numpy().reshape(-1,3)[faces[selected]].reshape(-1,3)
        else:
            tri=v.reshape(-1,3,3);n=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
            n/=np.maximum(np.linalg.norm(n,axis=1,keepdims=True),1e-9);normals=np.repeat(n,3,axis=0)
        bad=~np.isfinite(normals).all(axis=1)|(np.linalg.norm(normals,axis=1)<1e-9)
        normals[bad]=[0,0,1]
        normals/=np.maximum(np.linalg.norm(normals,axis=1,keepdims=True),1e-9)
        visibility=facade_visibility(scene,v,return_components=True)
        fill,direct=lighting_components(normals)
        # The old 40% unoccluded-fill floor washed out dense street shadows
        # relative to Original. Retain a smaller bounce approximation while
        # leaving fully sky-visible surfaces and the direct key unchanged.
        colors=srgb(linear(base_color)*(fill*(.25+.75*visibility[:,0,None])+direct*visibility[:,1,None]))
        parts.append(dict(vertices=v,faces=np.arange(len(v),dtype=np.uint32).reshape(-1,3),
                          colors=colors,source_faces=selected,normals=normals,baked_linear=np.array(True)))
    Path(cache_dir).mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(f'.{os.getpid()}.tmp.npz')
    arrays={f'{i}_{name}':value for i,part in enumerate(parts) for name,value in part.items()}
    np.savez(temporary,kinds=np.array(json.dumps([list(part) for part in parts])),**arrays)
    os.replace(temporary,path)
    return parts


def visual_asset_parts(assets):
    """Use the complete visual scene as occluders, without altering RF geometry."""
    digest=hashlib.sha256()
    for asset in assets:digest.update(Path(asset['mesh']).read_bytes())
    key=digest.digest()
    meshes=[mi.load_dict({'type':'ply','filename':asset['mesh'],
                         'face_normals':asset.get('face_normals',False)}) for asset in assets]
    scenes=[]
    def occluders():
        if not scenes:
            scenes.append(mi.load_dict({'type':'scene',**{f'visual{i}':mesh for i,mesh in enumerate(meshes)}}))
        return scenes[0]
    folder=Path(assets[0]['mesh']).resolve().parent.parent/'.display-cache'
    from .facade_lighting import material_for_mesh
    from .district_style import is_district, terrain_texture_path, building_parts
    results=[]
    for mesh,asset in zip(meshes,assets):
        if asset.get('texture'):
            # Official terrain imagery retains roads, fields and water boundaries.
            # Reuse the supplied UVs; never add this imagery to the RF material.
            from PIL import Image
            path=terrain_texture_path(asset) if is_district(asset) else asset['texture']
            texture=np.asarray(Image.open(path).convert('RGB'),dtype=np.float32)/255.
            from .terrain_lighting import shade_terrain
            texture=shade_terrain(occluders,mesh,texture,key,folder)
            results.append([dict(vertices=mesh.vertex_positions_buffer().numpy().reshape(-1,3),
                faces=mesh.faces_buffer().numpy().reshape(-1,3),
                uv=mesh.vertex_texcoords_buffer().numpy().reshape(-1,2),
                texture=texture,material='studio-ground')])
            continue
        parts=mesh_parts(occluders,mesh,asset['color'],key,folder,not asset.get('face_normals',False))
        for part in parts:part['material']=material_for_mesh(asset['mesh'])
        parts=building_parts(parts,asset)
        results.append(parts)
    return results


def register_parts(name, parts, group, slice_plane):
    import polyscope as ps
    from .facade_lighting import load_materials
    from .sionna_utils import display_rgb_linear
    load_materials()
    for index,part in enumerate(parts):
        material=part.get('material','flat')
        colors=lambda data:display_rgb_linear(data) if material!='flat' else data
        vertices=part['vertices'];faces=part['faces'];values=part.get('colors')
        smooth='normals' in part
        if smooth:
            # Share continuous corners, but keep crease-normal discontinuities.
            key=np.column_stack((vertices,np.round(part['normals'],4)))
            _,first,inverse=np.unique(key,axis=0,return_index=True,return_inverse=True)
            vertices=vertices[first];faces=inverse[faces]
            if values is not None:
                merged=np.zeros((len(first),3),np.float32)
                np.add.at(merged,inverse,values)
                values=merged/np.bincount(inverse)[:,None]
        struct=ps.register_surface_mesh(f'{name}-{index}',vertices,faces,
                                        material=material,smooth_shade=smooth)
        if 'texture' in part:
            struct.add_parameterization_quantity('Lighting UV',part['uv'],enabled=False)
            struct.add_color_quantity('Baked visual lighting',colors(part['texture']),defined_on='texture',
                                      param_name='Lighting UV',image_origin='lower_left',
                                      filter_mode='linear',enabled=True)
        else:
            struct.add_color_quantity('Baked visual lighting',colors(values),defined_on='vertices',enabled=True)
        struct.add_to_group(group);struct.set_ignore_slice_plane(slice_plane,False)
