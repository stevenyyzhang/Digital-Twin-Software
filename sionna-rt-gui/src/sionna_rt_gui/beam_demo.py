"""Sionna channel scan with a shared, unit-power transmit beam codebook."""
import time
import numpy as np
from .host_values import host_value
import matplotlib
import polyscope as ps
import polyscope.imgui as ui
from .config import DEFAULT_SLICE_PLANE_NAME
from .beamforming import weights, precoding, path_powers, rotation, link_signature
from .panel_codebook import make_codebook, PER_PANEL, PANEL_NAMES


def directions(az, el):
    a,e=np.radians(az),np.radians(el)
    return np.stack([np.cos(e)*np.cos(a),np.cos(e)*np.sin(a),np.sin(e)],axis=-1)


class BeamDemo:
    def __init__(self):
        self.az,self.el,self.panel_dirs,self.local_dirs,self.vertices=make_codebook()
        self.dirs=self.local_dirs.copy()
        self.local_vertices=self.vertices.copy()
        self.faces=np.arange(len(self.vertices)).reshape(-1,4)
        self.values=np.full(len(self.dirs),-np.inf);self.colors=np.zeros((len(self.dirs),3))
        self.selected=PER_PANEL//2;self.visible=False;self.scan=False;self.cir=False
        self.yaw=-45.;self.pitch=-25.;self.drag=0.;self.last_scan=0.
        self.signature=None;self.last_update=0.;self.center=None
        self.app=None;self.ready=False;self.dirty=True;self.status="Waiting for Sionna CIR"
        self.map_due=None;self.map_signature=None;self.map_requested=False

    def precoding(self,scene):
        return precoding(scene,self.panel_dirs[self.selected])

    @property
    def panel(self):
        return self.selected//PER_PANEL

    @property
    def label(self):
        return f'{PANEL_NAMES[self.panel]} | Beam {self.selected} (local {self.selected%PER_PANEL})'

    def consume(self,app,amplitudes):
        self.app=app
        self.signature=link_signature(app)
        try:
            panel_powers=path_powers(amplitudes,weights(app.scene,self.panel_dirs[:PER_PANEL]))
            self.powers=np.zeros((len(self.local_dirs),panel_powers.shape[1]))
            self.powers[self.panel*PER_PANEL:(self.panel+1)*PER_PANEL]=panel_powers
            gain=self.powers.sum(axis=1)
            tx=next(iter(app.scene.transmitters.values()))
            self.values=np.full(len(gain),-np.inf)
            valid=gain>0
            self.values[valid]=10*np.log10(gain[valid])+float(tx.power_dbm.numpy()[0])
            # Never change a user-selected beam implicitly when entering link analysis.
            self.ready=True
            self.status=f'{PANEL_NAMES[self.panel]} | {PER_PANEL} beams in panel | RSS dBm' if valid.any() else 'No resolved paths (signal unknown)'
            self.colors=matplotlib.colormaps['turbo'](np.clip((self.values+120)/80,0,1))[:,:3]
            self.colors[~valid]=[.3,.34,.38]
            result=self.powers[self.selected]
        except ValueError as exc:
            self.status=str(exc);self.values[:]=-np.inf;self.colors[:]=[.3,.34,.38]
            result=np.zeros(np.asarray(amplitudes).shape[-2])
        self.dirty=True
        return result

    def invalidate(self,app):
        self.signature=link_signature(app);self.values[:]=-np.inf
        self.colors[:]=[.3,.34,.38];self.dirty=True;self.status='No valid Sionna channel'

    def select(self,index):
        self.auto_beam_ready=True
        self.map_due=time.monotonic()+.35
        index=int(index)%len(self.dirs)
        if self.app is not None:
            from .spatial_beam import set_beam_target
            set_beam_target(self.app,index)
        if index==self.selected:return
        old_panel=self.panel
        self.selected=index
        self.dirty=True
        self.update_ray()
        if self.app is not None:
            current_mode=self.app.cfg.radio_map.coverage_mode=='current'
            if current_mode:self.app.clear_radio_map()
            if self.app.demo_page == 1:
                if old_panel!=self.panel:
                    self.invalidate(self.app)
                    self.app.clear_paths()
                    self.app.update_paths(clear_first=True,show=True)
                else:self.app.update_demo_cir()
            self.map_requested=False
            # Keep the explicit selection scheduled, even if a quick A/B/A
            # selection leaves the next frame's snapshot unchanged.
            self.map_due=time.monotonic()+.35
            self.map_signature=None
            self.app.demo_status=f'{self.label} | '+('Recompute coverage for this beam.' if current_mode else 'Best beam envelope retained.')

    def update_ray(self):
        if self.center is None:return
        if self.app is None:return
        key=(id(self.app.scene),id(self.app.render_cache),tuple(self.center),tuple(self.dirs[self.selected]))
        if (key==getattr(self,'_ray_key',None) and ps.has_curve_network('Selected Sionna beam')
                and ps.has_surface_mesh('Beam arrow')):return
        from .beam_visuals import direction_marker
        direction_marker(self.app,self.center,self.dirs[self.selected])
        self._ray_key=key

    def sync(self,app):
        self.app=app
        tx=app.scene.get('tx-0');rx=app.scene.get('rx-0')
        if tx is None:return
        from .beam_visuals import station
        station(app)
        now=time.monotonic()
        signature=link_signature(app) if rx is not None else None
        if app.demo_page == 1 and rx is not None and signature!=self.signature:
            self.last_update=now
            app.update_paths(clear_first=True,show=True)
        from .coverage_jobs import snapshot
        map_signature=snapshot(app)
        if self.map_signature is not None and map_signature!=self.map_signature:
            had_map=app.radio_map is not None or app.rm_refining
            app.clear_radio_map()
            self.map_due=None
            self.map_requested=False
        self.map_signature=map_signature
        if (getattr(self,'auto_beam_ready',False) and app.demo_page==0
                and (map_signature!=getattr(self,'_auto_signature',None)
                     or getattr(self,'_auto_page',None)!=0)):
            self._auto_signature=map_signature
            self.map_due=now+.35
        self._auto_page=app.demo_page

        self.center=host_value(tx,"position").ravel().astype(float)
        rot=rotation(tx)
        self.dirs=self.local_dirs@rot.T;self.vertices=self.local_vertices@rot.T
        if self.visible and app.demo_page == 1 and (self.dirty or not ps.has_surface_mesh('Sionna angular scan')):
            from .beam_visuals import clipped_distances
            lengths=clipped_distances(app.scene,self.center,self.vertices,2.5)
            sphere=ps.register_surface_mesh('Sionna angular scan',
                self.center+lengths[:,None]*self.vertices,self.faces,material='flat',enabled=self.visible and app.demo_page == 1)
            sphere.add_color_quantity('Sionna RSS [-120,-40] dBm',self.colors,defined_on='faces',enabled=True)
            sphere.set_transparency(.7)
            sphere.set_ignore_slice_plane(DEFAULT_SLICE_PLANE_NAME,True)
            self.dirty=False
        self.update_ray()
        if self.map_due is not None and now>=self.map_due and app.demo_page==0:
            self.map_due=None
            app.request_coverage()
        if self.scan and app.demo_page == 1 and now-self.last_scan>3:
            self.select(self.selected+1);self.last_scan=now

    def pick(self,result):
        if not result.is_hit or result.structure_name!='Sionna angular scan':return False
        direction=np.asarray(result.position)-self.center
        if np.linalg.norm(direction)>0:self.select(np.argmax(self.dirs@direction))
        return True

    def set_visible(self,value):
        self.visible=value
        if ps.has_surface_mesh('Sionna angular scan'):
            ps.get_surface_mesh('Sionna angular scan').set_enabled(value and self.app.demo_page == 1)
        self.update_ray()

    def project(self,points):
        a,e=np.radians(self.yaw),np.radians(self.pitch)
        right=np.array([-np.sin(a),np.cos(a),0])
        up=np.array([-np.sin(e)*np.cos(a),-np.sin(e)*np.sin(a),np.cos(e)])
        eye=np.array([np.cos(e)*np.cos(a),np.cos(e)*np.sin(a),np.sin(e)])
        return np.stack([points@right,-points@up,points@eye],axis=-1)

    def draw_globe(self,scale):
        start=np.array(ui.GetCursorScreenPos());size=np.array([325.,165.])*scale
        ui.InvisibleButton('Angular globe canvas',tuple(size))
        hovered=ui.IsItemHovered();active=ui.IsItemActive();io=ui.GetIO()
        if hovered and ui.IsMouseClicked(0):self.drag=0.
        if active:
            dx,dy=io.MouseDelta;self.drag+=abs(dx)+abs(dy)
            if self.drag>3:
                self.yaw-=dx*.5;self.pitch=float(np.clip(self.pitch+dy*.5,-85,85))
        center=start+size/2;radius=68*scale
        draw=ui.GetWindowDrawList()
        draw.AddRectFilled(tuple(start),tuple(start+size),0xFF281D17)
        projected=self.project(self.vertices).reshape(-1,4,3)
        mids=self.project(self.dirs)
        for i in np.argsort(mids[:,2]):
            if mids[i,2]<-.15:continue
            xy=projected[i,:,:2]*radius+center
            rgb=(self.colors[i]*255).astype(int)
            color=int(rgb[0])|(int(rgb[1])<<8)|(int(rgb[2])<<16)|(255<<24)
            draw.AddTriangleFilled(tuple(xy[0]),tuple(xy[1]),tuple(xy[2]),color)
            draw.AddTriangleFilled(tuple(xy[0]),tuple(xy[2]),tuple(xy[3]),color)
        marker=mids[self.selected]
        end=center+marker[:2]*radius*1.13
        draw.AddLine(tuple(center),tuple(end),0xFF36D4FF,2*scale)
        draw.AddCircle(tuple(end),5*scale,0xFFFFFFFF,0,2*scale)
        for axis,label in zip(np.eye(3),['+X','+Y','+Z']):
            point=center+self.project(axis[None,:])[0,:2]*radius*1.15
            draw.AddText(tuple(point),0xFFE7DED4,label)
        draw.AddText(tuple(start+np.array([5,5])*scale),0xFFBEB4A8,'Drag: rotate | click: select')
        if hovered and ui.IsMouseReleased(0) and self.drag<=3:
            xy=(np.array(io.MousePos)-center)/radius
            if np.dot(xy,xy)<=1:
                # Pick only the visible hemisphere, not a direction behind the sphere.
                target=np.r_[xy,np.sqrt(max(0,1-np.dot(xy,xy)))]
                self.select(np.argmax(mids@target))

    def draw(self,app):
        s=app.ui_scale
        from .desktop_window import layout
        w,_,y,h=layout(app)
        ui.SetNextWindowPos((10*s,y),ui.ImGuiCond_Always)
        ui.SetNextWindowSize((w,h),ui.ImGuiCond_Always)
        ui.Begin('3D scan power / linked beam',open=True)
        ui.TextColored((.5,.9,.8,1),self.status)
        self.draw_globe(s)
        changed,index=ui.SliderInt('Beam index',self.selected,0,len(self.dirs)-1)
        if changed:self.select(index)
        ui.Text(self.label)
        ui.Text(f'Panel az {self.az[self.selected]:.1f} | el {self.el[self.selected]:.1f} deg')
        ui.Text(f'Rx RSS: {self.values[self.selected]:.1f} dBm | scale -120..-40')
        changed,self.scan=ui.Checkbox('Sweep beams',self.scan)
        if changed and self.scan:
            self.last_scan=time.monotonic()

        ui.SameLine()
        changed,value=ui.Checkbox('Direction sphere',self.visible)
        if changed:self.set_visible(value)
        ui.BeginDisabled(not np.isfinite(self.values).any())
        if ui.Button('Best beam at RX in this panel'):self.select(np.argmax(self.values))
        ui.EndDisabled()
        ui.SameLine()
        if ui.Button('Sionna CIR'):self.cir=True
        ui.TextWrapped('Six panels cover all directions; one panel transmits at a time. RX power is scanned for the selected panel only. Gray panels are unscanned; select one to update.')
        ui.End()

