"""Fit the model in the canvas remaining between the controls."""
import itertools
from pathlib import Path
import numpy as np
import polyscope as ps

def _fit_district(app, bounds, top_view):
    # Cache the convex envelope once per loaded scene, never during orbit/playback.
    cached=getattr(app,'_district_camera_envelope',None)
    if cached is None or cached[0] is not app.scene:
        from scipy.spatial import ConvexHull, QhullError
        points=np.concatenate([np.asarray(s.vertex_positions_buffer()).reshape(-1,3)
                               for s in app.scene.mi_scene.shapes()])
        points=points[np.isfinite(points).all(axis=1)]
        try:points=points[ConvexHull(points).vertices]
        except QhullError:pass
        app._district_camera_envelope=(app.scene,points)
    else:points=cached[1]
    width,height=ps.get_window_size()
    lo,hi,upper,lower=bounds
    # Four percent inset on each side leaves space around roofs and terrain edges.
    mx=(hi-lo)*.04;my=(lower-upper)*.04
    camera=ps.get_view_camera_parameters()
    forward=np.array([0.,0.,-1.]) if top_view else np.asarray(camera.get_look_dir())
    up=np.array([0.,1.,0.]) if top_view else np.asarray(camera.get_up_dir())
    r=np.array([1.,0.,0.]) if top_view else np.asarray(camera.get_right_dir())
    focal=height/(2*np.tan(np.radians(camera.get_fov_vertical_deg())/2))
    center=(points.min(axis=0)+points.max(axis=0))/2
    delta=points-center;x=delta@r;y=delta@up;z=delta@forward
    a,b=(lo+mx-width/2)/focal,(hi-mx-width/2)/focal
    c,e=(height/2-lower+my)/focal,(height/2-upper-my)/focal
    def intervals(d):
        depth=d+z
        return (np.max(x-b*depth),np.min(x-a*depth),
                np.max(y-e*depth),np.min(y-c*depth))
    near=max(1.,-z.min()+1.);far=max(near*2,np.linalg.norm(np.ptp(points,axis=0)))
    def feasible(d):
        xl,xh,yl,yh=intervals(d)
        return xl<=xh and yl<=yh
    while not feasible(far):far*=2
    for _ in range(40):
        mid=(near+far)/2
        if feasible(mid):far=mid
        else:near=mid
    xl,xh,yl,yh=intervals(far)
    eye_x,eye_y=(xl+xh)/2,(yl+yh)/2
    if Path(app.cfg.scene_filename).parent.name=='polyu-stylized':
        # Center the projected footprint, including perspective, in the canvas.
        def centered_offset(values,low,high,desired):
            for _ in range(40):
                mid=(low+high)/2
                projected=(values-mid)/(far+z)
                if (projected.min()+projected.max())/2>desired:low=mid
                else:high=mid
            return (low+high)/2
        eye_x=centered_offset(x,xl,xh,(a+b)/2)
        eye_y=centered_offset(y,yl,yh,(c+e)/2)
    target=center+r*eye_x+up*eye_y
    ps.look_at_dir(target-forward*far,target,up)
    ps.set_view_center_raw(center)

def fit_canvas(app,top_view=False):
    if app.demo_page!=0 or not hasattr(app,'roadshow_layout'):return
    pad,top,left,right,bottom=app.roadshow_layout
    width,height=ps.get_window_size()
    lo=left+2*pad;hi=width-right-pad;upper=top+pad;lower=height-bottom-pad
    if lower<=upper or hi<=lo:return
    scene_name=Path(app.cfg.scene_filename).parent.name
    if scene_name in ('mong-kok','kam-tin') or (scene_name=='polyu-stylized' and not top_view):
        _fit_district(app,(lo,hi,upper,lower),top_view)
        return
    bbox=app.scene.mi_scene.bbox()
    points=np.array(list(itertools.product(*zip(list(bbox.min),list(bbox.max)))))
    center=points.mean(axis=0)
    camera=ps.get_view_camera_parameters()
    if top_view:
        forward=np.array([0.,0.,-1.]);up=np.array([0.,1.,0.]);r=np.array([1.,0.,0.])
    else:
        forward=np.asarray(camera.get_look_dir());up=np.asarray(camera.get_up_dir());r=np.asarray(camera.get_right_dir())
    focal=height/(2*np.tan(np.radians(camera.get_fov_vertical_deg())/2))
    delta=points-center;z=delta@forward
    distance=1.12*max(np.max(abs(delta@r)*focal/((hi-lo)/2)-z),
                       np.max(abs(delta@up)*focal/((lower-upper)/2)-z))
    dx=(lo+hi-width)/2;dy=(upper+lower-height)/2
    shift=(-r*dx+up*dy)*distance/focal
    target=center+shift
    ps.look_at_dir(target-forward*distance,target,up)
    ps.set_view_center_raw(center)
