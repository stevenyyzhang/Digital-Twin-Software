"""Shared display colour/material definitions; never used by the RF solver."""
from pathlib import Path
import numpy as np

def linear(rgb):
    rgb=np.clip(np.asarray(rgb,dtype=np.float32),0,1)
    return np.where(rgb<=.04045,rgb/12.92,((rgb+.055)/1.055)**2.4)

def srgb(rgb):
    rgb=np.maximum(np.asarray(rgb,dtype=np.float32),0)
    return np.clip(np.where(rgb<=.0031308,12.92*rgb,1.055*rgb**(1/2.4)-.055),0,1)

def surface_kind(path):
    name=Path(path).stem
    if 'glass' in name or 'water' in name:return 'glass'
    if 'roof' in name:return 'roof'
    if any(x in name for x in ('terrain','ground','road','plaza','lawn','infrastructure')):return 'ground'
    return 'wall'

def bsdf(asset,texture=None):
    """Opaque glazed model surfaces, including stylised glass, not RF materials."""
    kind=surface_kind(asset['mesh'])
    colour=({'type':'bitmap','filename':texture} if texture else
            {'type':'rgb','value':linear(asset['color']).tolist()})
    if kind=='ground':return {'type':'diffuse','reflectance':colour}
    # Match the Fast albedo for all three scenes, including campus. Roughplastic
    # attenuated its diffuse base under an extra dielectric layer. Keep a modest
    # highlight here, with broader plaster and a tighter glazed-surface response.
    return {'type':'principled','base_color':colour,'metallic':0.,
            'roughness':{'wall':.48,'roof':.38,'glass':.24}[kind],
            'specular':{'wall':.25,'roof':.28,'glass':.40}[kind]}

def key_light():
    v=np.array([-.65,.35,.68],dtype=np.float32)
    return v/np.linalg.norm(v)

def studio_environment():
    # Broad reflected light without inserting an occluding panel in the scene.
    u,v=np.meshgrid((np.arange(256)+.5)/256,(np.arange(128)+.5)/128)
    patch=np.exp(-.5*(((u-.63)/.065)**2+((v-.27)/.10)**2))
    # Near-neutral daylight preserves warm stone/brick in the shaded faces.
    # Keep this base identical to the calibrated Fast fill below.
    base=np.empty((*u.shape,3),np.float32);base[:]=[.50,.52,.56]
    return base+patch[...,None]*np.array([3.0,2.85,2.65],np.float32)

def lighting(normals):
    fill,direct=lighting_components(normals)
    return fill+direct

def lighting_components(normals):
    # Calibrated to diffuse swatches rendered by the Original rig at -60 deg.
    # Smooth positive lobe approximates the finite-width environment highlight.
    dot=normals@np.array([.26609490,.79662829,.54274937])
    patch=.5*(dot+np.sqrt(dot*dot+.22548452**2))
    fill=np.array([.50,.52,.56])+patch[...,None]*np.array([.48984794,.46521462,.43225730])
    direct=np.maximum(normals@key_light(),0)
    return fill,direct[...,None]*np.array([.663,.6205,.5525])
