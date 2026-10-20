"""Select real codebook beams against the actual scene, without a receiver."""
import numpy as np
import mitsuba as mi
import polyscope as ps
import polyscope.imgui as ui
from PIL import Image, ImageDraw
from .beamforming import rotation
from .desktop_window import layout
from .panel_codebook import ANGLES, SIDE, PER_PANEL, PANEL_NAMES


def draw_target(draw, anchor, target, scale, label='Target'):
    """Cyan dashed guide is an aiming annotation, never a propagation path."""
    anchor=np.asarray(anchor);target=np.asarray(target)
    delta=target-anchor;distance=np.linalg.norm(delta)
    if distance>1:
        unit=delta/distance
        for t in np.arange(0,distance,18*scale):
            a=tuple(anchor+t*unit);b=tuple(anchor+min(t+9*scale,distance)*unit)
            draw.AddLine(a,b,0xFF151515,5*scale)
            draw.AddLine(a,b,0xFFFFFF35,2*scale)
    point=tuple(target)
    draw.AddCircle(point,10*scale,0xFF151515,0,6*scale)
    draw.AddCircle(point,10*scale,0xFFFFFF35,0,3*scale)
    draw.AddLine(tuple(target-[15*scale,0]),tuple(target+[15*scale,0]),0xFFFFFF35,2*scale)
    draw.AddLine(tuple(target-[0,15*scale]),tuple(target+[0,15*scale]),0xFFFFFF35,2*scale)
    at=target+np.array([15,-22])*scale
    draw.AddText(tuple(at+[1,1]),0xFF111111,label)
    draw.AddText(tuple(at),0xFFFFFF35,label)


def pick_surface(app,origin,direction):
    cache=getattr(app,'render_cache',None)
    if cache is not None and 'visual_scene' in cache:
        from .rendering import intersect_visual
        hit=intersect_visual(cache,np.asarray(origin),np.asarray(direction))
        return hit.p.numpy().ravel().copy() if bool(hit.is_valid()[0]) else None
    hit=app.scene.mi_scene.ray_intersect(mi.Ray3f(mi.Point3f(np.asarray(origin)),mi.Vector3f(np.asarray(direction))))
    return hit.p.numpy().ravel().copy() if bool(hit.is_valid()[0]) else None


def set_beam_target(app,index):
    """Show the selected beam axis, ending at a surface or a direction marker."""
    tx=app.scene.get('tx-0')
    if tx is None:return
    origin=tx.position.numpy().ravel()
    direction=app.beam_demo.local_dirs[index]@rotation(tx).T
    target=pick_surface(app,origin,direction)
    # An upward beam may never meet geometry; this endpoint denotes direction
    # only, not a receiver or a simulated propagation/reflection path.
    app.beam_target=origin+80.*direction if target is None else np.asarray(target).copy()
    app.beam_target_horizontal=False


def choose_plan_target(app,xy):
    """Pick the top visible surface and match the complete 3D direction."""
    top=float(app.scene.mi_scene.bbox().max.z)+10
    target=pick_surface(app,[float(xy[0]),float(xy[1]),top],[0,0,-1])
    if target is None:
        app.demo_status='No surface here. Click the ground or a building.'
        return None
    return choose_target(app,target)


def choose_target(app, target, horizontal=False):
    tx=app.scene.get('tx-0')
    if tx is None:return None
    b=app.beam_demo;delta=np.asarray(target)-tx.position.numpy().ravel()
    world=b.local_dirs@rotation(tx).T
    if horizontal:
        if np.linalg.norm(delta[:2])<.01:return None
        indices=np.arange(len(b.local_dirs))
        dirs=world[indices,:2]
        scores=(dirs/np.maximum(np.linalg.norm(dirs,axis=1,keepdims=True),1e-8))@(delta[:2]/np.linalg.norm(delta[:2]))
    else:
        if np.linalg.norm(delta)<.01:return None
        indices=np.arange(len(b.local_dirs));scores=world@(delta/np.linalg.norm(delta))
    index=int(indices[np.argmax(scores)])
    error=float(np.degrees(np.arccos(np.clip(np.max(scores),-1,1))))
    b.select(index)
    app.demo_status=f'Selected {b.label} | aim error {error:.1f}°'
    app.beam_target=np.asarray(target).copy()
    app.beam_target_horizontal=horizontal
    return index,error


def aim_at_screen(app, point):
    origin=np.asarray(ps.get_view_camera_parameters().get_position())
    destination=ps.screen_coords_to_world_position(point)
    delta=destination-origin
    if not np.all(np.isfinite(delta)) or np.linalg.norm(delta)==0:return False
    target=pick_surface(app,origin,delta/np.linalg.norm(delta))
    if target is None:
        app.demo_status='Click a ground or building surface to aim.'
        return False
    choose_target(app,target)
    return True


class SpatialBeam:
    def __init__(self):
        self.scene=None;self.texture=None;self.near=True
        self.show_coverage=True;self.coverage_key=None;self.coverage_texture=None
        self.coverage_note=''

    def prepare_coverage(self,app):
        rm=app.radio_map;c=app.cfg.radio_map
        key=(id(app.scene),id(rm),app.rm_accumulated_samples,c.color_map,c.vmin,c.vmax)
        if key==self.coverage_key:return
        self.coverage_key=key;self.coverage_texture=None;self.coverage_note=''
        if rm is None:return
        from sionna.rt.utils.geometry import rotation_matrix
        from .coverage_display import plan_overlay
        rot=rotation_matrix(rm.orientation).numpy().reshape(3,3)
        cell=np.asarray(rm.cell_size.numpy()).ravel()
        basis=rot[:2,:2]@np.diag(cell)
        if abs(np.linalg.det(basis))<1e-8:
            self.coverage_note='Vertical measurement planes cannot be shown as top-view coverage.'
            return
        self.coverage_gain=np.max(rm.path_gain.numpy(),axis=0)
        self.coverage_mask=getattr(rm,'demo_outdoor_mask',np.ones(self.coverage_gain.shape,dtype=bool))
        self.coverage_origin=rm.cell_centers.numpy()[0,0,:2]-basis@np.array([.5,.5])
        self.coverage_basis=basis
        self.coverage_power=getattr(rm,'beam_power_dbm',0.)
        pixels=plan_overlay(self.coverage_gain,self.coverage_mask,self.coverage_origin,basis,
                            self.low,self.side,768,self.coverage_power,c)
        ps.add_color_alpha_image_quantity('beam_plan_coverage',pixels,enabled=True,
                                         image_origin='upper_left',show_fullscreen=False)
        self.coverage_texture=ps.get_quantity_buffer('beam_plan_coverage','colors').get_texture_native_id()

    def prepare(self,app):
        if self.scene is app.scene:return
        bbox=app.scene.mi_scene.bbox()
        self.low=np.array([float(bbox.min.x),float(bbox.min.y)])
        self.high=np.array([float(bbox.max.x),float(bbox.max.y)])
        span=np.maximum(self.high-self.low,1)
        self.side=float(max(span)*1.04)
        self.low=(self.low+self.high)/2-self.side/2
        picture=Image.new('RGBA',(768,768),(30,42,53,255));draw=ImageDraw.Draw(picture)
        polygons=[]
        for mesh in app.scene.mi_scene.shapes():
            v=mesh.vertex_positions_buffer().numpy().reshape(-1,3)
            f=mesh.faces_buffer().numpy().reshape(-1,3)
            tri=v[f];cross=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
            # Horizontal/roof silhouettes, drawn back to front by elevation.
            valid=np.abs(cross[:,2])>.15*np.maximum(np.linalg.norm(cross,axis=1),1e-9)
            for t in tri[valid]:polygons.append((float(t[:,2].mean()),t[:,:2]))
        zmin=float(bbox.min.z);zmax=float(bbox.max.z)
        for z,xy in sorted(polygons,key=lambda p:p[0]):
            uv=(xy-self.low)/self.side;uv[:,1]=1-uv[:,1]
            shade=float(np.clip((z-zmin)/max(zmax-zmin,1),0,1))
            color=tuple(int(v) for v in np.array([62,77,86])+shade*np.array([80,77,68]))+(255,)
            draw.polygon([tuple(p) for p in uv*768],fill=color)
        pixels=np.asarray(picture,dtype=np.float32)/255
        ps.add_color_alpha_image_quantity('beam_campus_plan',pixels,enabled=True,image_origin='upper_left',show_fullscreen=False)
        self.texture=ps.get_quantity_buffer('beam_campus_plan','colors').get_texture_native_id()
        self.scene=app.scene

    def draw(self,app,compact=False):
        tx=app.scene.get('tx-0')
        if tx is None:
            if compact:
                self.prepare(app)
                size=ui.GetContentRegionAvail()[0]
                ui.Image(ui.ImTextureRef(self.texture),(size,size))
                ui.TextWrapped('Place a base station to see direction and coverage.')
            return
        self.prepare(app);self.prepare_coverage(app);b=app.beam_demo;s=app.ui_scale
        w,_,y,h=layout(app)
        w=min(800*s,ps.get_window_size()[0]*.66)
        if not compact:
            ui.SetNextWindowPos((10*s,y),ui.ImGuiCond_Always);ui.SetNextWindowSize((w,h),ui.ImGuiCond_Always)
            ui.Begin('Beam direction',open=True)
        else:
            _,self.near=ui.Checkbox('Focus near TX',self.near)
            if app.demo_page==0 and app.rm_colorbar_texture_id is not None:
                ui.Image(ui.ImTextureRef(app.rm_colorbar_texture_id),(ui.GetContentRegionAvail()[0],40))
            ui.TextColored((.57,.66,.75,1),'RSS [dBm] · Blue: weak / Red: strong' if app.demo_page==0 else 'TX / RX and route')
        p=tx.position.numpy().ravel()
        view_side=self.side/(2.5 if self.near else 1.)
        target=getattr(app,'beam_target',None) if app.demo_page==0 else None
        center=p[:2]
        rx=app.scene.get('rx-0') if app.demo_page==1 else None
        if rx is not None:
            route=app.animation_config.trajectories['rx-0'].points
            points=np.vstack([p,rx.position.numpy().ravel(),route])
            center=(points[:,:2].min(axis=0)+points[:,:2].max(axis=0))/2
            view_side=min(self.side,max(view_side,float(np.max(np.ptp(points[:,:2],axis=0)))*1.3))
        if target is not None:
            center=(p[:2]+target[:2])/2
            view_side=min(self.side,max(view_side,float(np.max(abs(target[:2]-p[:2])))*1.3))
        view_low=np.clip(center-view_side/2,self.low,self.low+self.side-view_side)
        uv0=((view_low[0]-self.low[0])/self.side,1-(view_low[1]+view_side-self.low[1])/self.side)
        uv1=((view_low[0]+view_side-self.low[0])/self.side,1-(view_low[1]-self.low[1])/self.side)
        size=min(ui.GetContentRegionAvail()[0],max(150,ui.GetContentRegionAvail()[1]-65)) if compact else max(90*s,min(w*.60,ui.GetContentRegionAvail()[1]-12*s))
        start=np.array(ui.GetCursorScreenPos());ui.Image(ui.ImTextureRef(self.texture),(size,size),uv0,uv1)
        hovered=ui.IsItemHovered()
        clicked=hovered and ui.IsMouseClicked(0)
        def project(xy):
            uv=(np.asarray(xy)-view_low)/view_side
            return start+np.array([uv[0],1-uv[1]])*size
        anchor=project(p[:2]);draw=ui.GetWindowDrawList()
        if app.demo_page==0 and self.show_coverage and self.coverage_texture is not None:
            draw.AddImage(ui.ImTextureRef(self.coverage_texture),tuple(start),tuple(start+size),uv0,uv1)
            if hovered:
                from .coverage_display import sample_grid
                uv=(np.array(ui.GetIO().MousePos)-start)/size
                xy=view_low+np.array([uv[0],1-uv[1]])*view_side
                cell=np.linalg.solve(self.coverage_basis,xy-self.coverage_origin)
                value,outdoor=sample_grid(self.coverage_gain,self.coverage_mask,cell[0:1],cell[1:2])
                ui.BeginTooltip()
                if np.isfinite(value[0]):
                    ui.Text(f'Cell RSS: {10*np.log10(value[0])+self.coverage_power:.1f} dBm')
                else:ui.Text('No valid sample (signal unknown)' if outdoor[0] else 'Outside measurement area')
                if outdoor[0]:
                    ix,iy=np.floor(cell).astype(int)
                    ui.Text(f'Sample Z: {app.radio_map.cell_centers.numpy()[iy,ix,2]:.2f} m(1.5 m above ground)')
                    if getattr(app.radio_map,'coverage_mode','current')=='max':
                        winner=int(app.radio_map.best_beam[iy,ix])
                        if winner>=0:ui.Text(f'Best beam: {PANEL_NAMES[winner//PER_PANEL]} / Beam {winner}')
                ui.EndTooltip()
        direction=(b.local_dirs@rotation(tx).T)[b.selected]
        d=np.array([direction[0],-direction[1]]);d/=max(np.linalg.norm(d),1e-8)
        length=min(size*.23,100*s)
        # Clip the UI heading to the map rectangle, not to radio geometry.
        for k in range(2):
            if abs(d[k])>1e-8:
                edge=start[k]+(size-12*s if d[k]>0 else 12*s)
                length=min(length,max(0,(edge-anchor[k])/d[k]))
        end=anchor+length*d;side=np.array([-d[1],d[0]])
        draw.AddTriangleFilled(tuple(anchor),tuple(end+side*length*.22),tuple(end-side*length*.22),0x5537CEFF)
        draw.AddLine(tuple(anchor),tuple(end),0x70302820,5*s)
        draw.AddLine(tuple(anchor),tuple(end),0xFF55DFFF,2.5*s)
        draw.AddTriangleFilled(tuple(end),tuple(end-d*14*s+side*7*s),tuple(end-d*14*s-side*7*s),0xFF00D2FF)
        draw.AddCircle(tuple(anchor),7*s,0xFFFFFFFF,0,2*s)
        draw.AddCircleFilled(tuple(anchor),6*s,0xFF5147ED)
        draw.AddText(tuple(anchor+np.array([10,5])*s),0xFFFFFFFF,'TX')
        draw.AddText(tuple(start+np.array([6,5])*s),0xFFFFFFFF,'↑ +Y     +X →')
        if clicked and app.demo_aim_beam:
            uv=(np.array(ui.GetIO().MousePos)-start)/size
            xy=view_low+np.array([uv[0],1-uv[1]])*view_side
            choose_plan_target(app,xy)
        if rx is not None:
            draw.PushClipRect(tuple(start),tuple(start+size),True)
            for a,c in zip(route[:-1],route[1:]):draw.AddLine(tuple(project(a[:2])),tuple(project(c[:2])),0xFFE4BE45,3.)
            r=project(rx.position.numpy().ravel()[:2])
            draw.AddCircleFilled(tuple(r),6.,0xFFFFC878);draw.AddCircle(tuple(r),9.,0xFFFFFFFF,0,2.)
            draw.AddText(tuple(r+[10,-12]),0xFFFFFFFF,'RX');draw.PopClipRect()
        target=getattr(app,'beam_target',None) if app.demo_page==0 else None
        if target is not None:
            draw.PushClipRect(tuple(start),tuple(start+size),True)
            draw_target(draw,anchor,project(target[:2]),s)
            draw.PopClipRect()
        if compact:
            ui.TextWrapped('Gray: no valid sample. Hatched: outside area.\nCyan dashed line: aim reference.' if app.demo_page==0 else 'Cyan: receiver route\nGold: beam direction')
            return
        ui.SameLine();ui.BeginGroup()
        ui.PushItemWidth(max(80*s,w-size-42*s))
        if ui.Button('Full-sphere beam selection'):app.spatial_beam_mode=1
        _,self.near=ui.Checkbox('Focus near TX',self.near)
        _,self.show_coverage=ui.Checkbox('Show RSS coverage',self.show_coverage)
        if app.radio_map is None:
            ui.TextWrapped('Compute coverage to see RSS. Switch between selected beam and best beam envelope.')
        elif self.coverage_texture is not None and self.show_coverage:
            ui.TextWrapped(getattr(app.radio_map,'coverage_label',f'Beam {app.radio_map.beam_index} RSS')+' [dBm]')
            ui.Image(ui.ImTextureRef(app.rm_colorbar_texture_id),(max(80*s,w-size-52*s),42*s))
            ui.TextWrapped('Gray: no valid sample. Hatched: outside area. Hover for cell details.')
        elif self.coverage_note:ui.TextWrapped(self.coverage_note)
        if target is not None:
            if ui.Button('Clear target'):app.beam_target=None
        ui.TextColored((.2,1.,1.,1.),'Cyan: aim target')
        ui.TextColored((1.,.82,0.,1.),'Gold: selected beam')
        ui.TextWrapped('Dashed line: aim reference, not a propagation path.')
        if target is not None:
            delta=target-p;delta/=max(np.linalg.norm(delta),1e-8)
            error=float(np.degrees(np.arccos(np.clip(np.dot(direction,delta),-1,1))))
            ui.Text(f'Target Z {target[2]:.1f} m | aim error {error:.1f}°')
            if error>15:ui.TextColored((1.,.65,.25,1.),'Large aim error. Click the target again to match a beam.')
        changed,panel=ui.Combo('Panel',b.panel,PANEL_NAMES)
        if changed:b.select(panel*PER_PANEL+b.selected%PER_PANEL)
        ui.Text('Azimuth')
        changed,az=ui.SliderFloat('##az',float(b.az[b.selected]),-45.,45.,format='%.1f deg')
        if changed:b.select((b.selected//SIDE)*SIDE+int(np.argmin(abs(ANGLES-az))))
        ui.Text('Elevation')
        changed,el=ui.SliderFloat('##el',float(b.el[b.selected]),-45.,45.,format='%.1f deg')
        if changed:b.select(b.panel*PER_PANEL+int(np.argmin(abs(ANGLES-el)))*SIDE+b.selected%SIDE)
        ui.Text(b.label)
        ui.TextWrapped('15° steps. The sector shows direction, not coverage.')
        ui.PopItemWidth();ui.EndGroup()
        ui.End()
