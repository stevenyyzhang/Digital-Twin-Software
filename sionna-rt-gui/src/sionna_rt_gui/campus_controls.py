"""Compact demo controls and an absolute, per-path channel plot."""
import numpy as np
import polyscope as ps
import polyscope.imgui as ui
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from .host_values import host_value


def refresh_chart(app):
    # Keep one canvas and the same artists for the lifetime of the GUI. Each
    # accepted channel snapshot still gets a complete draw and texture upload.
    cache=getattr(app,'_cir_chart_cache',None)
    if cache is None:
        fig=Figure(figsize=(4.2,2.5),dpi=120,facecolor='#171d28')
        canvas=FigureCanvasAgg(fig)
        ax=fig.add_subplot(111,facecolor='#171d28')
        stems=ax.vlines([],[],[],color='#55cdd2',linewidth=1.4)
        points=ax.scatter([],[],s=13,color='#e9bb7e',zorder=3)
        empty=ax.text(.5,.5,'No valid paths',ha='center',va='center',
                      color='white',transform=ax.transAxes)
        ax.set_xlabel('Absolute delay [ns]',color='#ccd6e5',fontsize=9)
        ax.set_ylabel('',color='#ccd6e5',fontsize=9)
        ax.tick_params(colors='#aebdcc',labelsize=8)
        ax.yaxis.get_offset_text().set_color('#aebdcc')
        ax.yaxis.get_offset_text().set_fontsize(8)
        for spine in ax.spines.values():spine.set_color('#405064')
        # Fixed margins fit the labels and scientific-notation exponent, without
        # repeating text measurement and tight_layout on every moving-RX result.
        fig.subplots_adjust(left=.18,right=.97,bottom=.22,top=.92)
        cache=dict(figure=fig,canvas=canvas,axes=ax,stems=stems,points=points,empty=empty)
        app._cir_chart_cache=cache
    canvas,ax=cache['canvas'],cache['axes']
    delays=np.asarray(app.demo_path_delays,dtype=float).reshape(-1).copy()
    powers=np.asarray(app.demo_path_powers,dtype=float).reshape(-1).copy()
    valid=np.isfinite(delays)&np.isfinite(powers)&(powers>0)
    delays,powers=delays[valid],powers[valid]
    mode=app.demo_chart_mode
    label=('Path power gain [dB]' if mode==0 else
           'CIR magnitude |a|' if mode==1 else 'Relative path power')
    populated=bool(len(powers))
    if populated:
        if mode == 0:
            values = 10*np.log10(np.maximum(powers, np.finfo(float).tiny))
            bottom=min(-160.,float(np.floor((values.min()-5)/10)*10))
            ax.set_ylim(bottom, max(-40, float(values.max())+5))
        else:
            values=np.sqrt(powers) if mode==1 else powers/powers.max()
            bottom=0.
            ax.set_ylim(0.,float(values.max())*1.08)
        cache['stems'].set_segments(np.stack([
            np.column_stack([delays,np.full(len(delays),bottom)]),
            np.column_stack([delays,values])],axis=1))
        cache['points'].set_offsets(np.column_stack([delays,values]))
        ax.set_xlim(0, max(100, float(max(delays))*1.08))
    else:
        cache['stems'].set_segments([])
        cache['points'].set_offsets(np.empty((0,2)))
        ax.set_xlim(0.,1.);ax.set_ylim(0.,1.)
    cache['empty'].set_visible(not populated)
    cache['points'].set_visible(populated)
    cache['stems'].set_visible(populated)
    ax.set_ylabel(label)
    # Keep the locators intact, so ticks return after an empty-path snapshot.
    ax.tick_params(bottom=populated,left=populated,labelbottom=populated,labelleft=populated)
    ax.yaxis.get_offset_text().set_visible(populated)
    if populated:ax.grid(True,alpha=.15,color='#aebdcc')
    else:ax.grid(False)
    canvas.draw()
    pixels=np.asarray(canvas.buffer_rgba(),dtype=np.float32)/255
    ps.add_color_alpha_image_quantity('campus_cir',pixels,enabled=True,
        image_origin='upper_left',show_fullscreen=False)
    app.demo_chart_texture=ps.get_quantity_buffer('campus_cir','colors').get_texture_native_id()
    app.demo_chart_dirty=False


def draw_chart(app):
    if not hasattr(app,'beam_demo'):
        from .beam_demo import BeamDemo
        app.beam_demo=BeamDemo()
    app.beam_demo.sync(app)
    if app.demo_page == 0:
        from .spatial_beam import SpatialBeam
        from .tx_pattern import TxPattern
        if not hasattr(app,'spatial_beam'):app.spatial_beam=SpatialBeam()
        if not hasattr(app,'tx_pattern'):app.tx_pattern=TxPattern()
        if getattr(app,'spatial_beam_mode',0)==0:app.spatial_beam.draw(app)
        else:app.tx_pattern.draw(app)
        return
    if not app.scene.receivers:return
    if not app.beam_demo.cir:
        app.beam_demo.draw(app)
        return
    scale=app.ui_scale
    if app.demo_chart_dirty:refresh_chart(app)
    from .desktop_window import layout
    w,_,y,h=layout(app)
    ui.SetNextWindowPos((10*scale,y),ui.ImGuiCond_Always)
    ui.SetNextWindowSize((w,h),ui.ImGuiCond_Always)
    ui.Begin('CIR / Channel power',open=True)
    if ui.Button('3D scan power'):app.beam_demo.cir=False
    changed,app.demo_chart_mode=ui.Combo('Metric',app.demo_chart_mode,
        ['Power gain [dB]','CIR magnitude','Relative power'])
    if changed:app.demo_chart_dirty=True
    ui.Image(ui.ImTextureRef(app.demo_chart_texture),(w-24*scale,max(100*scale,h-150*scale)))
    ui.Text(f'Beam {app.beam_demo.selected} | {app.demo_cir_path_count} paths | absolute delay')
    rx=app.scene.get('rx-0')
    if rx is not None:
        p=host_value(rx,'position').squeeze();ui.Text(f'Rx: {p[0]:.1f}, {p[1]:.1f}, {p[2]:.1f} m')
    ui.End()


def draw_device_labels(app):
    camera=ps.get_view_camera_parameters()
    origin=np.asarray(camera.get_position())
    right=np.asarray(camera.get_right_dir());up=np.asarray(camera.get_up_dir())
    forward=np.asarray(camera.get_look_dir())
    width,height=ps.get_window_size()
    focal=height/(2*np.tan(np.radians(camera.get_fov_vertical_deg())/2))
    draw=ui.GetBackgroundDrawList()
    if app.demo_page==1 and getattr(app,'demo_link_snapshot_visible',False) and getattr(app,'demo_link_settings',None):
        point=np.asarray(app.demo_link_settings['rx'][0][1]);delta=point-origin;depth=float(delta@forward)
        if depth>0:
            xy=(width/2+focal*float(delta@right)/depth,height/2-focal*float(delta@up)/depth)
            draw.AddCircle(xy,10.,0xFF8DD9FF,0,2.)
            draw.AddText((xy[0]+14,xy[1]+12),0xFF8DD9FF,'Link sample')
    for name,label,color in [('tx-0','TX',0xFFBCD57C),('rx-0','RX',0xFFBCD57C)]:
        device=app.scene.get(name)
        if device is None:continue
        delta=host_value(device,'position').squeeze()-origin;depth=float(np.dot(delta,forward))
        if depth<=0:continue
        x=width/2+focal*float(np.dot(delta,right))/depth
        y=height/2-focal*float(np.dot(delta,up))/depth
        if 0<x<width and 0<y<height:
            if name=='tx-0':
                # Screen-space locator is intentionally visible even when the physical mast is occluded.
                scale=app.ui_scale
                draw.AddCircle((x,y),8*scale,0xDDF3F3FF,0,2*scale)
                draw.AddCircle((x,y),6*scale,0xFF5147ED,0,2*scale)
                bx=x+18*scale;by=y-25*scale
                draw.AddRectFilled((bx,by),(bx+115*scale,by+31*scale),0xEE251B10,5*scale)
                draw.AddText((bx+7*scale,by+5*scale),0xFFF4F0E8,f'TX · Beam {app.beam_demo.selected}')
                # Trace the same geometry-clipped endpoint as the physical marker, with a readable outline.
                world_dir=app.beam_demo.dirs[app.beam_demo.selected]
                end_world=host_value(device,'position').ravel()+getattr(app.beam_demo,'direction_length',0)*world_dir
                end_delta=end_world-origin;end_depth=float(np.dot(end_delta,forward))
                if end_depth>0:
                    end=np.array([width/2+focal*np.dot(end_delta,right)/end_depth,
                                  height/2-focal*np.dot(end_delta,up)/end_depth])
                    begin=np.array([x,y]);vec=end-begin;distance=np.linalg.norm(vec)
                    if distance>10*scale:
                        unit=vec/distance;side=np.array([-unit[1],unit[0]])
                        draw.AddLine(tuple(begin),tuple(end),0x70302820,5*scale)
                        draw.AddLine(tuple(begin),tuple(end),0xFF55DFFF,2.5*scale)
                        head=min(14*scale,distance*.35)
                        draw.AddTriangleFilled(tuple(end),tuple(end-unit*head+side*head*.32),tuple(end-unit*head-side*head*.32),0xFF00D2FF)
            else:
                draw.AddCircleFilled((x,y),7.,0xFFFFC878)
                draw.AddCircle((x,y),11.,0xFFFFFFFF,0,2.)
                draw.AddRectFilled((x+15,y-24),(x+65,y+7),0xEE251B10,4.)
                draw.AddText((x+22,y-20),0xFFF4F0E8,'RX')
    target=getattr(app,'beam_target',None)
    tx=app.scene.get('tx-0')
    if target is not None and tx is not None and app.demo_page==0:
        def project(point):
            delta=np.asarray(point)-origin;depth=float(np.dot(delta,forward))
            if depth<=0:return None
            return np.array([width/2+focal*np.dot(delta,right)/depth,
                             height/2-focal*np.dot(delta,up)/depth])
        end=project(target);begin=project(host_value(tx,'position').ravel())
        if end is not None and begin is not None:
            from .spatial_beam import draw_target
            draw.PushClipRect((0,0),(width,height),True)
            draw_target(draw,begin,end,app.ui_scale,'Aim target')
            draw.PopClipRect()


def draw_controls(app):
    ui.TextColored((.56,.85,.83,1),'POLYU  /  RADIO MAP STUDIO')
    from .demo_workflow import draw_workflow
    if draw_workflow(app):return
    ui.Text('Link analysis | movable RX')
    if not app.scene.receivers:
        ui.TextWrapped('Place a TX in Area coverage first.')
        return
    if ui.Button('Stop placing' if app.demo_place_receiver else 'Place Rx'):
        app.demo_place_receiver=not app.demo_place_receiver
        app.animation_config.playing=False
    ui.SameLine()
    ui.SetNextItemWidth(150*app.ui_scale)
    _,app.demo_placement_mode=ui.Combo('Surface',app.demo_placement_mode,['Ground','Rooftop'])
    _,app.demo_edit_route=ui.Checkbox('Click to append route points',app.demo_edit_route)
    traj=app.animation_config.trajectories['rx-0']
    ui.BeginDisabled(len(traj)<2)
    if ui.Button('Pause' if app.animation_config.playing else 'Play'):
        traj.enabled=True;app.demo_place_receiver=False
        app.animation_config.playing=not app.animation_config.playing
    ui.EndDisabled();ui.SameLine()
    if ui.Button('Reset route'):app.restore_demo_route()
    ui.SameLine()
    if ui.Button('Undo point') and len(traj):
        pts=traj.points[:-1].copy();traj.clear()
        for p in pts:traj.add_point(p)
        if len(pts):app.move_demo_receiver(pts[-1])
        app.animation_config.playing=False;app.draw_demo_route()
    changed,speed=ui.SliderFloat('Speed [m/s]',traj.velocity,.5,8.,format='%.1f')
    if changed:traj.velocity=speed
    changed,distance=ui.SliderFloat('Route [m]',traj.distance,0.,max(.01,traj.total_distance()),format='%.1f')
    if changed:
        traj.distance=distance;position=traj.current_position_and_direction()
        if position is not None:app.move_demo_receiver(position[0])
    ui.Text(f'{traj.total_distance():.0f} m | {len(traj)} waypoints')
    if getattr(app,'demo_scene_data',None) is not None and app.demo_scene_data.get('source')=='landsd_stylized':
        if ui.Button('Courtyard view'):ps.look_at([220,-205,195],[42,-42,15])
        ui.SameLine()
        if ui.Button('Campus view'):app.move_camera_home()
    if app.demo_status:ui.TextWrapped(app.demo_status)
    ui.Separator()
    _,app.demo_advanced=ui.Checkbox('Advanced settings / sources',app.demo_advanced)


def draw_coverage_controls(app):
    from .coverage_jobs import sample_budget,beam_budget
    mode=0 if app.cfg.radio_map.coverage_mode=='current' else 1
    changed,mode=ui.Combo('Map mode',mode,['Selected beam','Best beam envelope'])
    if changed:
        app.pause_coverage()
        app.cfg.radio_map.coverage_mode='current' if mode==0 else 'max'
        app.clear_radio_map()
        app.request_coverage()
    sizes=np.array([6.,3.,2.,1.])
    index=int(np.argmin(abs(sizes-app.cfg.radio_map.cell_size[0])))
    changed,index=ui.Combo('Grid',index,['6 m preview','3 m detail','2 m fine','1 m very fine'])
    if changed:
        app.pause_coverage();app.clear_radio_map()
        cell=float(sizes[index])
        app.cfg.radio_map.cell_size=(cell,cell)
        app.cfg.radio_map.accumulate_max_samples_per_tx=int(3000000*(6/cell)**2)
    ui.TextWrapped(f'Ground +{app.cfg.radio_map.height_above_ground:g} m | {beam_budget(app)/1e6:.2f}M samples/beam')
    if mode==1:
        ui.TextWrapped('Best single beam at each cell, across 294 beams. Pause and resume anytime.')
    if ui.Button('Compute coverage'):
        app.request_coverage()
    ui.SameLine()
    at_budget=app.rm_accumulated_samples>=sample_budget(app)
    if ui.Button('Pause coverage' if app.rm_refining else 'More samples' if at_budget else 'Refine coverage'):
        if app.rm_refining:app.pause_coverage()
        elif app.radio_map is None:app.request_coverage()
        else:
            if at_budget:
                cell=app.cfg.radio_map.cell_size[0]
                if mode==1:app.cfg.radio_map.max_samples_per_beam+=250000
                else:app.cfg.radio_map.accumulate_max_samples_per_tx+=int(3000000*(6/cell)**2)
            if app.coverage_jobs is None:
                from .coverage_jobs import CoverageJobs
                app.coverage_jobs=CoverageJobs(app)
            app.rm_refining=True
    if app.coverage_jobs is not None:
        ui.TextWrapped(app.coverage_jobs.status)
    if app.radio_map is not None:
        visible,mesh=app.has_visible_radio_map()
        changed,visible=ui.Checkbox('Show coverage',visible)
        if changed:
            mesh.set_enabled(visible)
            if not visible:app.pause_coverage()
        done=app.rm_accumulated_samples>=sample_budget(app)
        state='Budget reached' if done else 'Refining' if app.rm_refining else 'Paused'
        ui.Text(f'{state} | {app.rm_accumulated_samples/1e6:.1f}M / {sample_budget(app)/1e6:.0f}M rays')
        if hasattr(app.radio_map,'demo_outdoor_cells'):
            ui.Text(f'Positive samples: {app.radio_map.demo_observed_cells}/{app.radio_map.demo_outdoor_cells} outdoor cells')
            ui.TextWrapped(getattr(app.radio_map,'coverage_label','RSS [dBm]'))
            ui.TextWrapped('Gray: no valid sample. Hatched: outside measurement area.')
    _,app.demo_advanced=ui.Checkbox('Advanced settings / sources',app.demo_advanced)
    if app.demo_advanced:
        ui.TextWrapped('DEMO: no trained model connected. Buildings derived from Lands Department, HKSAR Government; 3d.map.gov.hk. Ground: OpenStreetMap contributors, ODbL; openstreetmap.org/copyright. Ground and RF materials are schematic.')


def merge_coverage(app,new_map,batch_samples):
    """Average independent estimates in linear power, weighted by ray count."""
    total=app.rm_accumulated_samples+batch_samples
    weight=batch_samples/total
    app.radio_map._pathgain_map=(1-weight)*app.radio_map.path_gain+weight*new_map.path_gain
    app.rm_accumulated_samples=total
