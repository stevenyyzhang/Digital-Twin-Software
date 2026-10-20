"""Bake visibility into existing terrain UVs, at actual source elevations."""
import hashlib,inspect
from pathlib import Path
import numpy as np
import mitsuba as mi
from .presentation_style import key_light,lighting_components,linear,srgb

def shade_terrain(scene_factory,mesh,texture,scene_key,folder):
    from PIL import Image
    # Bound bake cost and retain the existing UVs/terrain triangles.
    h,w=texture.shape[:2];scale=min(1.,1024/max(h,w))
    if scale<1:
        texture=np.asarray(Image.fromarray(np.uint8(texture*255)).resize(
            (round(w*scale),round(h*scale)),Image.Resampling.LANCZOS),np.float32)/255
    h,w=texture.shape[:2]
    vertices=mesh.vertex_positions_buffer().numpy().reshape(-1,3)
    faces=mesh.faces_buffer().numpy().reshape(-1,3)
    uv=mesh.vertex_texcoords_buffer().numpy().reshape(-1,2)
    digest=hashlib.sha256(b'uv-terrain-visibility-v1'+scene_key)
    for value in (vertices,faces,uv,texture,key_light()):digest.update(value.tobytes())
    digest.update(inspect.getsource(lighting_components).encode())
    path=Path(folder)/('terrain-shadow-'+digest.hexdigest()+'.npz')
    if path.exists():
        with np.load(path) as data:return data['texture']
    points=np.zeros((h,w,3),np.float32);normals=np.zeros_like(points)
    covered=np.zeros((h,w),bool)
    for face in faces:
        a,b,c=uv[face];ab=b-a;ac=c-a
        det=ab[0]*ac[1]-ab[1]*ac[0]
        if abs(det)<1e-12:continue
        lo=np.maximum(0,np.floor(np.min(uv[face],axis=0)*[w,h]-.5).astype(int))
        hi=np.minimum([w-1,h-1],np.ceil(np.max(uv[face],axis=0)*[w,h]-.5).astype(int))
        if np.any(hi<lo):continue
        xx,yy=np.meshgrid(np.arange(lo[0],hi[0]+1),np.arange(lo[1],hi[1]+1))
        x=(xx+.5)/w-a[0];y=(yy+.5)/h-a[1]
        s=(x*ac[1]-y*ac[0])/det;t=(ab[0]*y-ab[1]*x)/det
        inside=(s>=-1e-5)&(t>=-1e-5)&(s+t<=1+1e-5)
        if not inside.any():continue
        xyz=vertices[face]
        normal=np.cross(xyz[1]-xyz[0],xyz[2]-xyz[0])
        normal/=max(np.linalg.norm(normal),1e-10)
        if normal[2]<0:normal=-normal
        rows=yy[inside];cols=xx[inside]
        points[rows,cols]=xyz[0]+s[inside,None]*(xyz[1]-xyz[0])+t[inside,None]*(xyz[2]-xyz[0])
        normals[rows,cols]=normal;covered[rows,cols]=True
    scene=scene_factory()
    p=points[covered];n=normals[covered]
    lit=np.empty((len(p),3),np.float32);sun_all=np.empty(len(p),np.float32)
    light=key_light();tangent=np.cross(light,[0.,0.,1.]);tangent/=np.linalg.norm(tangent)
    bitangent=np.cross(light,tangent)
    for start in range(0,len(p),131072):
        end=min(start+131072,len(p));normal=n[start:end]
        origin=mi.Point3f((p[start:end]+normal*.06).T)
        sun=np.zeros(end-start,np.float32)
        for dx,dy in ((0,0),(.025,0),(-.025,0),(0,.025),(0,-.025)):
            direction=light+dx*tangent+dy*bitangent;direction/=np.linalg.norm(direction)
            sun+=~scene.ray_test(mi.Ray3f(origin,mi.Vector3f(direction))).numpy()
        sun/=5;sun_all[start:end]=sun
        sky=np.zeros(end-start,np.float32)
        for i in range(12):
            radius=np.sqrt((i+.5)/12);angle=i*np.pi*(3-np.sqrt(5))
            ray=mi.Ray3f(origin,mi.Vector3f(np.array([radius*np.cos(angle),radius*np.sin(angle),np.sqrt(1-radius*radius)],np.float32)))
            ray.maxt=20.;sky+=~scene.ray_test(ray).numpy()
        fill,direct=lighting_components(normal)
        lit[start:end]=fill*(.40+.60*sky[:,None]/12)+direct*sun[:,None]
    result=texture.copy()
    result[covered]=srgb(linear(texture[covered])*lit)
    assert np.isfinite(result).all(), 'Invalid terrain lighting'
    shadow=np.ones((h,w),np.float32);shadow[covered]=sun_all
    Path(folder).mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,texture=result,covered=covered,sun_visibility=shadow)
    return result
