"""Left/middle drag pans; right drag orbits the existing focus.

Wheel zoom follows the cursor. Double-click in browse mode (or Ctrl-click)
focuses a surface. A camera gesture must begin on the scene, not a UI control.
"""
import numpy as np
import time
import polyscope as ps
import polyscope.imgui as ui
from .spatial_beam import pick_surface


def surface_at(point):
    camera=ps.get_view_camera_parameters()
    origin=np.asarray(camera.get_position())
    delta=ps.screen_coords_to_world_position(point)-origin
    return origin,delta/max(np.linalg.norm(delta),1e-9)


def focus(app,target):
    eye=np.asarray(ps.get_view_camera_parameters().get_position())
    target=np.asarray(target)
    ps.look_at(target+(eye-target)*.45,target)
    ps.set_view_center_raw(target)


def pan(dx,dy):
    camera=ps.get_view_camera_parameters();eye=np.asarray(camera.get_position())
    center=ps.get_view_center();distance=max(np.linalg.norm(center-eye),1.)
    units=2*distance*np.tan(np.radians(camera.get_fov_vertical_deg())/2)/ps.get_window_size()[1]
    shift=units*(-dx*np.asarray(camera.get_right_dir())+dy*np.asarray(camera.get_up_dir()))
    ps.look_at_dir(eye+shift,eye+shift+np.asarray(camera.get_look_dir())*distance,camera.get_up_dir())
    ps.set_view_center_raw(center+shift)


def zoom(app,point,wheel):
    eye,direction=surface_at(point);target=pick_surface(app,eye,direction)
    if target is None:target=eye+direction*max(np.linalg.norm(ps.get_view_center()-eye),1.)
    camera=ps.get_view_camera_parameters();distance=np.linalg.norm(target-eye)
    factor=max(.5/max(distance,.001),np.exp(-np.clip(wheel,-5,5)*.18))
    moved=target+(eye-target)*factor
    ps.look_at_dir(moved,moved+np.asarray(camera.get_look_dir())*max(distance,1),camera.get_up_dir())
    ps.set_view_center_raw(target)


def orbit(dx,dy):
    camera=ps.get_view_camera_parameters();center=np.asarray(ps.get_view_center())
    v=np.asarray(camera.get_position())-center
    forward=np.asarray(camera.get_look_dir());up=np.asarray(camera.get_up_dir())
    def rotate(value,axis,angle):
        return value*np.cos(angle)+np.cross(axis,value)*np.sin(angle)+axis*np.dot(axis,value)*(1-np.cos(angle))
    vertical=np.array([0.,0.,1.]);yaw=-dx*.005
    v,forward,up=(rotate(value,vertical,yaw) for value in (v,forward,up))
    right=np.cross(forward,up);right/=max(np.linalg.norm(right),1e-9)
    pitched=rotate(forward,right,-dy*.005)
    tilt=abs(pitched[2])/max(np.linalg.norm(pitched),1e-9)
    current_tilt=abs(forward[2])/max(np.linalg.norm(forward),1e-9)
    if tilt<.98 or tilt<current_tilt:
        v,forward,up=(rotate(value,right,-dy*.005) for value in (v,forward,up))
    # Cursor zoom may leave the pivot off the optical axis. Preserve that
    # framing instead of snapping the view to look directly at the pivot.
    eye=center+v
    ps.look_at_dir(eye,eye+forward*max(np.linalg.norm(v),1.),up)
    ps.set_view_center_raw(center)


def process(app):
    ps.set_do_default_mouse_interaction(False)
    io=ui.GetIO()
    owners=getattr(app,'_navigation_buttons',None)
    if owners is None:owners={};app._navigation_buttons=owners
    for button in (0,1,2):
        if ui.IsMouseClicked(button):owners[button]=not io.WantCaptureMouse
        elif not ui.IsMouseDown(button):owners[button]=False
    dx,dy=io.MouseDelta
    # Suspend tracing at press time, before the drag-distance threshold, and
    # keep it suspended while holding still or crossing over a UI panel.
    held=any(owners.get(b) and ui.IsMouseDown(b) for b in (0,1,2))
    if held or (not io.WantCaptureMouse and io.MouseWheel):
        app._navigation_until=time.monotonic()+.9
    browse=not any(getattr(app,name,False) for name in
                   ('demo_place_transmitter','demo_place_receiver','demo_aim_beam'))
    if owners.get(0) and ui.IsMouseClicked(0) and (io.KeyCtrl or
            (browse and ui.IsMouseDoubleClicked(0))):
        eye,direction=surface_at(io.MousePos);point=pick_surface(app,eye,direction)
        if point is not None:focus(app,point)
    if owners.get(1) and ui.IsMouseDragging(1):orbit(dx,dy)
    elif ((owners.get(0) and ui.IsMouseDragging(0) and not io.KeyCtrl)
          or (owners.get(2) and ui.IsMouseDragging(2))):pan(dx,dy)
    if not io.WantCaptureMouse and io.MouseWheel:zoom(app,io.MousePos,io.MouseWheel)
