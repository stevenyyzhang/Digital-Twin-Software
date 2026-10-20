"""Small physical TX marker and geometry-clipped direction annotations."""
import numpy as np
from .host_values import host_value
import mitsuba as mi
import polyscope as ps
from .beamforming import rotation
from .config import DEFAULT_SLICE_PLANE_NAME
from .panel_codebook import PANEL_ROTATIONS


def clipped_distances(scene, origin, directions, length):
    u = np.atleast_2d(directions)
    hit = scene.mi_scene.ray_intersect(mi.Ray3f(mi.Point3f(origin), mi.Vector3f(u.T)))
    distances = hit.t.numpy()
    return np.maximum(0, np.minimum(length, distances-.2))


def station(app):
    tx = app.scene.get('tx-0')
    if tx is None:return
    center = host_value(tx,"position").ravel()
    rot = rotation(tx)
    key = tuple(center)+tuple(rot.ravel())+(app.beam_demo.panel,)
    if getattr(app, '_station_pose', None) == key and ps.has_surface_mesh('Base station'):return
    app._station_pose = key
    vertices=[];faces=[];colors=[]
    def box(offset, size, color, panel_rot=None, world_aligned=False):
        base=len(vertices)
        pts=np.array([[x,y,z] for x in [-1,1] for y in [-1,1] for z in [-1,1]],float)
        pts=pts*np.array(size)/2+offset
        if panel_rot is not None:pts=pts@panel_rot.T
        pts=(pts if world_aligned else pts@rot.T)+center
        vertices.extend(pts)
        quads=[[0,4,6,2],[1,3,7,5],[0,1,5,4],[2,6,7,3],[0,2,3,1],[4,5,7,6]]
        for a,b,c,d in quads:
            faces.extend([[base+a,base+b,base+c],[base+a,base+c,base+d]])
            colors.extend([color,color])
    # Six visual faces share an ideal RF phase center; no coupling model.
    from .demo_workflow import support_below
    support = support_below(app, center)
    height = float(center[2]-support[2]) if support is not None else 5.0
    box([0,0,-height/2],[.35,.35,height],[.22,.27,.30],world_aligned=True)
    box([0,0,-height+.13],[1.3,1.3,.26],[.16,.19,.22],world_aligned=True)
    for panel,panel_rot in enumerate(PANEL_ROTATIONS):
        color=[1.,.65,.08] if panel==app.beam_demo.panel else [.70,.08,.12]
        box([1.05,0,0],[.16,1.9,1.9],color,panel_rot)
    mesh=ps.register_surface_mesh('Base station',np.asarray(vertices),np.asarray(faces),smooth_shade=False)
    mesh.add_color_quantity('Station materials',np.asarray(colors),defined_on='faces',enabled=True)
    mesh.set_ignore_slice_plane(DEFAULT_SLICE_PLANE_NAME,True)
    if ps.has_point_cloud('Transmitters'):ps.get_point_cloud('Transmitters').set_enabled(False)


def direction_marker(app, center, direction):
    length=float(clipped_distances(app.scene,center,direction,18)[0])
    # The display mesh has more detail than the RF mesh: stop at either surface.
    if app.render_cache is not None:
        from .rendering import intersect_visual
        hit=intersect_visual(app.render_cache,center,direction)
        length=min(length,max(0,float(hit.t.numpy()[0])-.2))
    app.beam_demo.direction_blocked=length<17.99
    app.beam_demo.direction_length=length
    end=center+length*direction
    ray=ps.register_curve_network('Selected Sionna beam',np.array([center,end]),
        np.array([[0,1]]),color=(1.,.63,.10),enabled=length>.05)
    ray.set_radius(.13,relative=False)
    ray.set_ignore_slice_plane(DEFAULT_SLICE_PLANE_NAME,True)
    # Arrowhead stays before the collision, rather than extending into the wall.
    axis=np.array([0.,0.,1.]) if abs(direction[2])<.9 else np.array([0.,1.,0.])
    right=np.cross(direction,axis);right/=np.linalg.norm(right);up=np.cross(right,direction)
    head=min(1.8,length*.25);radius=min(.58,head*.32)
    angles=np.linspace(0,2*np.pi,16,endpoint=False)
    ring=end-head*direction+radius*(np.cos(angles)[:,None]*right+np.sin(angles)[:,None]*up)
    mesh=ps.register_surface_mesh('Beam arrow',np.vstack([end,ring]),
        np.array([[0,1+i,1+(i+1)%16] for i in range(16)]),color=(1.,.63,.10),enabled=length>.05)
    mesh.set_ignore_slice_plane(DEFAULT_SLICE_PLANE_NAME,True)
    if ps.has_point_cloud('Beam direction tip'):ps.remove_point_cloud('Beam direction tip')
