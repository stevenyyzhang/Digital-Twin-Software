"""Campus-style presentation shared by both renderers; no RF changes."""
from pathlib import Path
import hashlib
import numpy as np

def is_district(asset):
    return Path(asset['mesh']).resolve().parent.parent.name in ('mong-kok','kam-tin')

def terrain_colors(texture):
    # Illustrative treatment, not surveyed land-cover classification.
    luma=texture@np.array([.2126,.7152,.0722],dtype=np.float32)
    from PIL import Image,ImageFilter
    # Small photographic details become noise at an overview scale. Smooth only
    # the display image, retaining broad roads/fields/water boundaries.
    reduced=Image.fromarray(np.uint8(np.clip(texture,0,1)*255)).filter(ImageFilter.GaussianBlur(2.0))
    texture=np.asarray(reduced,dtype=np.float32)/255.
    luma=texture@np.array([.2126,.7152,.0722],dtype=np.float32)
    green=(texture[...,1]>texture[...,0]*1.04)&(texture[...,1]>texture[...,2]*1.10)
    base=np.empty_like(texture);base[:]=[.72,.79,.81]
    base[green]=[.58,.76,.61]
    return np.clip(base*(.94+.08*luma[...,None]),0,1).astype(np.float32)

def terrain_texture_path(asset):
    from PIL import Image
    source=Path(asset['texture'])
    key=hashlib.sha256(b'studio-terrain-v3'+source.read_bytes()).hexdigest()
    target=source.parent.parent/'.display-cache'/f'styled-ground-{key}.png'
    if not target.exists():
        target.parent.mkdir(parents=True,exist_ok=True)
        data=np.asarray(Image.open(source).convert('RGB'),dtype=np.float32)/255
        Image.fromarray(np.round(terrain_colors(data)*255).astype(np.uint8)).save(target)
    return str(target)

def building_parts(parts,asset):
    from .presentation_style import linear,srgb,lighting,surface_kind
    from .scene_style import display_illumination
    result=[]
    for source in parts:
        part=dict(source)
        if bool(part.get('baked_linear',False)):
            kind=surface_kind(asset['mesh'])
            part['material']='studio-'+{'glass':'glass','wall':'plaster','roof':'roof','ground':'ground'}[kind]
            result.append(part)
            continue
        channel='texture' if 'texture' in part else 'colors'
        if channel=='texture':normals=np.array([[0.,0.,1.]])
        else:
            tri=part['vertices'][part['faces']]
            normals=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
            normals/=np.maximum(np.linalg.norm(normals,axis=1,keepdims=True),1e-9)
            normals=np.repeat(normals,3,axis=0)
        old=display_illumination(normals)
        visibility=np.clip(part[channel]/np.maximum(np.asarray(asset['color'])*old,.001),0,1)
        # Decode the base colour before applying light, not after it.
        part[channel]=srgb(linear(asset['color'])*lighting(normals)*visibility)
        kind=surface_kind(asset['mesh'])
        part['material']='studio-'+{'glass':'glass','wall':'plaster','roof':'roof','ground':'ground'}[kind]
        result.append(part)
    return result
