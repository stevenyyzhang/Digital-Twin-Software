"""Scene-Z flight waypoints; conservative open-sky segment validation."""
import numpy as np
import mitsuba as mi
import polyscope as ps
import polyscope.imgui as ui


def segment_clear(app,start,end):
    start=np.asarray(start);end=np.asarray(end)
    length=float(np.linalg.norm(end-start))
    if length<.05:return False
    ray=mi.Ray3f(mi.Point3f(start),mi.Vector3f((end-start)/length));ray.maxt=length
    if bool(app.scene.mi_scene.ray_test(ray)[0]):return False
    points=start+np.linspace(0,1,max(2,int(np.ceil(length/2))+1))[:,None]*(end-start)
    top=max(float(app.scene.mi_scene.bbox().max.z)+10.,float(points[:,2].max())+10.)
    hit=app.scene.mi_scene.ray_intersect(mi.Ray3f(
        mi.Point3f(points[:,0],points[:,1],np.full(len(points),top)),mi.Vector3f(0,0,-1)))
    return bool(np.all(hit.is_valid().numpy() & (points[:,2]>hit.p.z.numpy()+.5)))


def place(app,screen):
    eye=np.asarray(ps.get_view_camera_parameters().get_position())
    direction=ps.screen_coords_to_world_position(screen)-eye
    z=getattr(app,'air_waypoint_z',40.)
    if abs(direction[2])<1e-9:
        app.demo_status='Look down toward the selected height plane.';return False
    t=(z-eye[2])/direction[2]
    if t<=0:
        app.demo_status='Selected height is behind the camera. Change view or Z.';return False
    point=eye+t*direction
    traj=app.animation_config.trajectories['rx-0']
    start=traj.points[-1] if app.demo_edit_route and len(traj) else point+[0,0,1.]
    if not segment_clear(app,start,point):
        app.demo_status='Flight waypoint or segment is blocked / outside supported open-sky space.';return False
    app.move_demo_receiver(point)
    if app.demo_edit_route:traj.add_point(point)
    app.draw_demo_route()
    app.demo_status=f'Flight waypoint at scene Z={z:g} m.'
    if not app.demo_edit_route:app.demo_place_receiver=False
    return True


def controls(app):
    changed,enabled=ui.Checkbox('Air route / scene-Z placement',getattr(app,'air_route_mode',False))
    if changed:app.air_route_mode=enabled
    if not enabled:return
    changed,z=ui.InputFloat('Waypoint Z [m]',getattr(app,'air_waypoint_z',40.))
    if changed and np.isfinite(z):app.air_waypoint_z=float(z)
    ui.TextWrapped('Click XY on the selected height plane. Change Z before adding the next waypoint. Open-sky check; point RX, no aircraft safety clearance.')
    if ui.Button('Create rising flight from current route'):
        traj=app.animation_config.trajectories['rx-0']
        source=np.array(traj.points,copy=True)
        if len(source)<2:
            app.demo_status='Create at least two route points first.';return
        top=float(app.scene.mi_scene.bbox().max.z)
        low=max(top+10.,getattr(app,'air_waypoint_z',40.))
        source[:,2]=low+10*np.linspace(0,1,len(source))
        if not all(segment_clear(app,a,b) for a,b in zip(source[:-1],source[1:])):
            app.demo_status='Flight route failed the open-sky check.';return
        app.move_demo_receiver(source[0]);traj.clear()
        for p in source:traj.add_point(p)
        traj.distance=0.;app.draw_demo_route()
        app.demo_status=f'Flight ready: Z {source[0,2]:g} to {source[-1,2]:g} m.'
