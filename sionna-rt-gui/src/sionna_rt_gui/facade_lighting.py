"""Static facade occlusion baked once; no RF geometry or material mutations."""
import numpy as np
import mitsuba as mi


def facade_visibility(scene, vertices, return_components=False):
    tri=vertices.reshape(-1,3,3)
    normal=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
    normal/=np.maximum(np.linalg.norm(normal,axis=1,keepdims=True),1e-9)
    # Stay slightly inside the face at corners to avoid a neighbouring face
    # incorrectly shadowing a sample exactly on their shared edge.
    positions=(tri*.96+tri.mean(axis=1)[:,None,:]*.04).reshape(-1,3)
    normals=np.repeat(normal,3,axis=0)
    result=np.ones((len(positions),1),np.float32)
    components=np.ones((len(positions),2),np.float32) if return_components else None
    from .presentation_style import key_light
    light=key_light()
    for start in range(0,len(positions),131072):
        p=positions[start:start+131072];n=normals[start:start+131072]
        tangent=np.cross(n,np.where((abs(n[:,2])<.9)[:,None],[0.,0.,1.],[0.,1.,0.]))
        tangent/=np.maximum(np.linalg.norm(tangent,axis=1,keepdims=True),1e-9)
        bitangent=np.cross(n,tangent)
        origins=mi.Point3f((p+n*.06).T)
        sky=np.zeros(len(p));sun=np.zeros(len(p))
        for i in range(16):
            radius=np.sqrt((i+.5)/16);angle=i*np.pi*(3-np.sqrt(5))
            direction=tangent*(radius*np.cos(angle))+bitangent*(radius*np.sin(angle))+n*np.sqrt(1-radius*radius)
            ray=mi.Ray3f(origins,mi.Vector3f(direction.T));ray.maxt=12.
            sky+=~scene.ray_test(ray).numpy()
        for dx,dy in [(0,0),(.035,0),(-.035,0),(0,.035),(0,-.035)]:
            direction=light+[dx,dy,0];direction/=np.linalg.norm(direction)
            sun+=~scene.ray_test(mi.Ray3f(origins,mi.Vector3f(direction))).numpy()
        facing=np.maximum(n@light,0)
        # The directional shading is already baked in the base colours.
        # Remove its occluded fraction and deepen contact corners gently.
        result[start:start+len(p),0]=(.60+.40*sky/16)*(1-.28*facing*(1-sun/5))
        if return_components:
            components[start:start+len(p),0]=sky/16
            components[start:start+len(p),1]=sun/5
    return components if return_components else result


def material_for_mesh(path):
    name=str(path).lower()
    if 'glass' in name or 'water' in name:return 'studio-glass'
    if 'white' in name or 'roof' in name:return 'studio-plaster'
    if 'brick' in name:return 'studio-brick'
    return 'flat'


_loaded=False
def load_materials():
    """Small linear HDR lookup tables: bounded highlights, no runtime tracing.

    RGB + black bases follow Polyscope's documented blendable material format.
    Glass uses a stylised studio reflection, not a reflection of campus geometry.
    """
    global _loaded
    if _loaded:return
    from pathlib import Path
    import polyscope as ps
    folder=Path(__file__).resolve().parents[3]/'polyu-stylized/.display-cache/materials-v1'
    folder.mkdir(parents=True,exist_ok=True)
    axis=np.linspace(-1,1,256);x,y=np.meshgrid(axis,-axis)
    z=np.sqrt(np.maximum(0,1-x*x-y*y))
    diffuse=.96+.04*z
    highlight=np.maximum(0,-.35*x+.45*y+.82*z)
    for kind,strength,power in [('ground',0.,8),('brick',.08,12),('plaster',.18,20),('roof',.24,24),('glass',.40,40),
                               ('district-brick',.04,10),('district-plaster',.18,20),
                               ('district-glass',.55,44)]:
        district=kind.startswith('district-')
        base=kind.removeprefix('district-')
        material_diffuse=(.78+.16*z+.06*y) if district else diffuse
        if district and base=='glass':
            material_diffuse=.73+.18*z+.08*np.tanh((x+.15)*7)
        spec=strength*highlight**power
        if base=='glass':
            spec+=(.30 if district else .11)*np.exp(-((x+.4)**2/.035+(y-.15)**2/.7))
            spec+=(.12 if district else .06)*(1-z)**5
        paths=[]
        for i,basis in enumerate(np.vstack([np.eye(3),np.zeros(3)])):
            data=material_diffuse[...,None]*basis+spec[...,None]*np.array([1.,.97,.92])
            maximum=data.max(axis=-1);mantissa,exponent=np.frexp(maximum)
            rgb=data*(mantissa*256/np.maximum(maximum,1e-32))[...,None]
            rgbe=np.concatenate([np.clip(rgb,0,255).astype(np.uint8),
                 np.where(maximum>1e-32,exponent+128,0).astype(np.uint8)[...,None]],axis=-1)
            # Content-keyed district tables prevent stale highlights after edits.
            import hashlib
            version='-'+hashlib.sha256(rgbe.tobytes()).hexdigest()[:12]
            path=folder/f'{kind}{version}-{i}.hdr'
            if not path.exists():path.write_bytes(b'#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y 256 +X 256\n'+rgbe.tobytes())
            paths.append(str(path))
        ps.load_blendable_material(kind if district else 'studio-'+kind,filenames=paths)
    _loaded=True
