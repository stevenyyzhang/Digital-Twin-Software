"""Two presentation workspaces: regional coverage and receiver/channel analysis."""
from contextlib import contextmanager
import numpy as np
import mitsuba as mi
import polyscope as ps
import polyscope.imgui as ui
from .panel_codebook import PANEL_NAMES,PER_PANEL,ANGLES,SIDE
from .roadshow_results import result_data,compare,export_result,provenance
from .host_values import host_value

ACCENT=(.25,.82,.77,1);MUTED=(.57,.66,.75,1);WARN=(1.,.73,.38,1)


@contextmanager
def theme():
    colors={'WindowBg':(.045,.065,.095,.98),'ChildBg':(.06,.085,.12,1),
        'PopupBg':(.07,.10,.15,1),'Border':(.16,.22,.29,1),'Text':(.91,.95,.98,1),
        'TextDisabled':MUTED,'FrameBg':(.10,.15,.21,1),'FrameBgHovered':(.16,.24,.31,1),
        'FrameBgActive':(.17,.30,.36,1),'Button':(.12,.23,.28,1),
        'ButtonHovered':(.17,.36,.40,1),'ButtonActive':(.12,.45,.44,1),
        'Header':(.12,.23,.28,1),'HeaderHovered':(.17,.36,.40,1),
        'CheckMark':ACCENT,'SliderGrab':ACCENT,'PlotHistogram':ACCENT}
    for name,value in colors.items():ui.PushStyleColor(getattr(ui,'ImGuiCol_'+name),value)
    variables={'WindowRounding':10.,'FrameRounding':5.,'WindowBorderSize':1.,
        'WindowPadding':(16.,14.),'FramePadding':(9.,7.),'ItemSpacing':(8.,9.)}
    for name,value in variables.items():ui.PushStyleVar(getattr(ui,'ImGuiStyleVar_'+name),value)
    try:yield
    finally:ui.PopStyleVar(len(variables));ui.PopStyleColor(len(colors))


def panel(name,x,y,w,h):
    ui.SetNextWindowPos((x,y),ui.ImGuiCond_Always)
    ui.SetNextWindowSize((w,h),ui.ImGuiCond_Always)
    ui.Begin(name,flags=ui.ImGuiWindowFlags_NoTitleBar|ui.ImGuiWindowFlags_NoMove|ui.ImGuiWindowFlags_NoResize|ui.ImGuiWindowFlags_NoCollapse)
    ui.PushItemWidth(-1)


def end_panel():ui.PopItemWidth();ui.End()


def heading(number,text):
    ui.Spacing();ui.TextColored(ACCENT,f'{number}  /  {text}');ui.Separator()


def section(app,key,label):
    opened=app.roadshow_sections[key]
    if button(('−  ' if opened else '+  ')+label):
        opened=not opened;app.roadshow_sections[key]=opened
    return opened


def button(text,primary=False):
    if primary:ui.PushStyleColor(ui.ImGuiCol_Button,(.07,.38,.38,1))
    result=ui.Button(text,(ui.GetContentRegionAvail()[0],0))
    if primary:ui.PopStyleColor()
    return result


def scrub_slider(app,label,value,low,high,format):
    """Keep the final pointer position even when a fast drag ends between frames."""
    changed,result=ui.SliderFloat(label,value,low,high,format=format)
    io=ui.GetIO();x=float(io.MousePos[0])
    owners=getattr(app,'_scrub_owners',None)
    if owners is None:owners={};app._scrub_owners=owners
    lo,hi=ui.GetItemRectMin()[0],ui.GetItemRectMax()[0]
    half=min(ui.GetStyle().GrabMinSize,hi-lo-4)/2
    left,right=lo+2+half,hi-2-half
    if ui.IsItemActive() and ui.IsMouseClicked(0) and not io.KeyCtrl:
        center=left+(right-left)*(value-low)/max(high-low,1e-9)
        owners[label]=(x-center if abs(x-center)<=half else 0.,x)
    owner=owners.get(label)
    if owner is not None and ui.IsMouseReleased(0):
        offset,pressed=owner
        if abs(x-pressed)>=io.MouseDragThreshold and right>left and np.isfinite(x):
            result=float(low+np.clip((x-offset-left)/(right-left),0.,1.)*(high-low))
            changed=result!=value
    if not ui.IsMouseDown(0):owners.pop(label,None)
    return changed,result


def set_operation(app,operation):
    getattr(app,'_scrub_owners',{}).clear()
    app.demo_place_transmitter=operation=='tx'
    app.demo_aim_beam=operation=='aim'
    app.demo_place_receiver=operation in ('rx','route')
    app.demo_edit_route=operation=='route'
    app.animation_config.playing=False


def operation(app):
    if app.demo_place_transmitter:return 'Click a ground or roof surface to place TX, 5 m above it. Esc to finish.'
    if app.demo_place_receiver:return ('Click to add route points.' if app.demo_edit_route else 'Click to move RX; the route is kept.')+' Esc to finish.'
    if app.demo_aim_beam:return 'Click a target in the scene to aim the beam. Esc to finish.'
    return 'Drag: pan  |  Right drag: orbit  |  Scroll: zoom  |  R: overview  |  Space: play / pause'


def sample_station(app):
    from .presentation_presets import apply_preset
    apply_preset(app,getattr(app,'roadshow_site',0));set_operation(app,'browse')


def scene_controls(app):
    from .scene_style import scene_label
    cities=['polyu-stylized','mong-kok','kam-tin']
    choices=[i for i,p in enumerate(app.known_scene_paths) if p and (getattr(app,'roadshow_test_scenes',False) or app.known_scene_names[i] in cities or i==app.current_scene_idx)]
    choices.sort(key=lambda i:cities.index(app.known_scene_names[i]) if app.known_scene_names[i] in cities else 99)
    labels=[scene_label(app.known_scene_names[i],app.known_scene_paths[i]) for i in choices]
    current=choices.index(app.current_scene_idx) if app.current_scene_idx in choices else 0
    changed,index=ui.Combo('##roadshow_scene',current,labels)
    if changed:
        app.load_scene_requested=app.known_scene_paths[choices[index]]
        app.demo_status='Loading '+labels[index]+'...'


def station_controls(app):
    if button('Finish placement' if app.demo_place_transmitter else 'Place TX'):
        set_operation(app,'browse' if app.demo_place_transmitter else 'tx')
    tx=app.scene.get('tx-0')
    if tx is not None:
        p=host_value(tx,'position').ravel()
        ui.TextColored(MUTED,f'TX  Z {p[2]:.1f} m')


def beam_controls(app):
    if app.scene.get('tx-0') is None:return
    half=(ui.GetContentRegionAvail()[0]-8)/2
    if ui.Button('Aim',(half,0)):
        app.studio_beam_mode='aim';set_operation(app,'aim')
    ui.SameLine()
    if ui.Button('Select beam',(half,0)):
        app.studio_beam_mode='select';set_operation(app,'browse')
    if getattr(app,'studio_beam_mode','aim')=='select':
        from .tx_pattern import TxPattern
        if not hasattr(app,'tx_pattern'):app.tx_pattern=TxPattern()
        app.tx_pattern.draw(app,compact=True)
    else:ui.TextColored(MUTED,f'Beam {app.beam_demo.selected}')


def coverage_settings(app):
    pass  # The five heights are already labelled beside their top views.


def link_controls(app):
    rx=app.scene.get('rx-0');ui.BeginDisabled(rx is None)
    if rx is not None:
        traj=app.animation_config.trajectories['rx-0']
        ui.BeginDisabled(len(traj)<2)
        half=(ui.GetContentRegionAvail()[0]-8)/2
        if ui.Button('Restart',(half,0)):
            set_operation(app,'browse');traj.distance=0.;traj.backward=False
            app.move_demo_receiver(traj.points[0])
        ui.SameLine()
        play=ui.Button('Pause' if app.animation_config.playing else 'Play route',(half,0))
        if play:
            playing=not app.animation_config.playing;set_operation(app,'browse')
            if playing:app.move_demo_receiver(traj.current_position_and_direction()[0])
            traj.enabled=True;app.animation_config.playing=playing
        ui.EndDisabled()
        changed,distance=scrub_slider(app,'##route_distance',traj.distance,0.,max(.01,traj.total_distance()),format='Position %.1f m')
        if changed:
            traj.distance=distance;app.animation_config.playing=False
            position=traj.current_position_and_direction()
            if position is not None:app.move_demo_receiver(position[0])
        if ui.CollapsingHeader('Edit route & speed'):
            from .air_routes import controls as flight_controls
            flight_controls(app)
            changed,speed=scrub_slider(app,'##route_speed',traj.velocity,.5,8.,format='Speed %.1f m/s')
            if changed:traj.velocity=speed
            ui.TextColored(MUTED,f'{len(traj)} points / {traj.total_distance():.1f} m / back and forth')
            if button('Move RX only'):set_operation(app,'rx')
            if button('Add route points'):set_operation(app,'route')
            if app.demo_edit_route:
                ui.BeginDisabled(len(traj)<2)
                if button('Undo last point'):app.edit_demo_route('undo')
                ui.EndDisabled()
            if button('New route from RX'):app.edit_demo_route('new');set_operation(app,'route')
            if button('Restore example route'):app.restore_demo_route()
    ui.EndDisabled()


def compute_action(app):
    if app.rm_refining:ui.TextColored(ACCENT,'Computing RSS...')
    elif getattr(app.coverage_jobs,'last_error',None):
        ui.TextWrapped(app.coverage_jobs.last_error)
    else:ui.TextColored(MUTED,'Automatic | 8M samples / height')


def coverage_results(app):
    from .studio_layers import draw_layers
    draw_layers(app)


def link_results(app):
    from .link_jobs import current_result,same_environment
    from .beamforming import link_signature
    from time import monotonic
    ready=current_result(app);settings=getattr(app,'demo_link_settings',None)
    rx=app.scene.get('rx-0');tx=app.scene.get('tx-0')
    ui.TextColored(ACCENT,'How does the link change?')
    if rx is None:ui.TextWrapped('Place a TX to start.');return
    compatible=same_environment(settings,link_signature(app))
    snapshot=not ready and app.animation_config.playing and compatible
    if ps.has_curve_network('paths'):ps.get_curve_network('paths').set_enabled(ready or snapshot)
    if app.link_jobs and app.link_jobs.error:
        ui.TextWrapped(app.link_jobs.error)
        if button('Retry link calculation'):app.link_jobs.request(app,force=True)
    ui.TextColored(ACCENT if ready else WARN,'Current RX' if ready else 'Last sampled RX' if snapshot else 'Updating current RX...')
    if not ready and not snapshot:return
    if snapshot:
        distance=np.linalg.norm(np.asarray(settings['rx'][0][1])-host_value(rx,'position').ravel())
        ui.TextColored(MUTED,f'{distance:.1f} m behind / received {monotonic()-app.demo_link_received:.1f} s ago')
    powers=app.demo_path_powers;delays=app.demo_path_delays
    if len(powers):
        rss=float(host_value(tx,'power_dbm')[0])+10*np.log10(np.sum(powers))
        ui.Text(f'Received power  {rss:.1f} dBm')
        ui.Text(f'Propagation paths  {len(powers)}')
    else:ui.TextWrapped('No valid paths found with these settings.')
    from .live_chart import draw_path_chart
    draw_path_chart(app)
    ui.TextColored(MUTED,'Gold: radio paths / Cyan: RX route')
    if ui.CollapsingHeader('Link details'):
        ui.TextWrapped('Noncoherent sum of path powers. Chart: path gain vs absolute delay. A missing path is not proof of no signal.')
        p=host_value(rx,'position').ravel();ui.Text(f'RX: {p[0]:.1f}, {p[1]:.1f}, {p[2]:.1f} m')
        if settings:
            p=settings['rx'][0][1];ui.Text(f'Sample: {p[0]:.1f}, {p[1]:.1f}, {p[2]:.1f} m')


def evidence(app):
    if not ui.CollapsingHeader('Settings & model details'):return
    from .config import RenderingMode
    changed,mode=ui.Combo('##display_mode',app.cfg.rendering.mode.value,['Fast / baked lighting','Original / ray tracing'])
    if changed:app.set_rendering_mode(RenderingMode(mode))
    ui.TextWrapped('Fast uses cached studio lighting. Original traces the display materials and studio lights. Neither changes radio calculations.')
    sizes=[6.,3.,2.,1.];selected=int(np.argmin(abs(np.asarray(sizes)-app.cfg.radio_map.cell_size[0])))
    changed,index=ui.Combo('##grid_studio',selected,['6 m / preview','3 m / standard','2 m / fine','1 m / very fine'])
    if changed:
        app.pause_coverage();app.clear_radio_map();cell=sizes[index]
        app.cfg.radio_map.cell_size=(cell,cell)
        app.cfg.radio_map.accumulate_max_samples_per_tx=8000000
        app.beam_demo.map_signature=None
    ui.TextWrapped('Measurement: independent scene-Z layers.' if app.cfg.radio_map.coverage_domain=='air' else 'Measurement: 1.5 m above ground.')
    ui.TextWrapped('Finer grids and more height layers require more sampling.')
    ui.TextWrapped(provenance(app))
    profile=getattr(app,'air_height_profile',{})
    if profile.get('source_verification'):
        ui.TextWrapped('Building Z checked against original official FBX data. Synthetic legacy mast removed. Ground remains schematic; geometry max Z is not building height above ground.')
    ui.TextWrapped('Sionna RT simulation. Illustrative materials; not calibrated against measurements.')
    ui.Text(f'{float(app.scene.frequency.numpy()[0])/1e9:.2f} GHz / 6 panels x 49 beams')
    _,app.roadshow_test_scenes=ui.Checkbox('Show all scenes',getattr(app,'roadshow_test_scenes',False))


def draw(app):
    app.beam_demo.sync(app)
    width,height=ps.get_window_size();font=ui.GetFontSize()
    unit=max(.85,min(1.35,font/20));pad=12*unit;top=92*unit;bottom=70*unit
    side=min(380*unit,width*.28)
    panel_height=height-top-bottom-pad
    maps_available=getattr(app,'studio_maps_available',False) and getattr(app,'studio_maps_scene',None)==id(app.scene)
    maps_open=maps_available and not getattr(app,'studio_maps_collapsed',False)
    thumb=max(60.,(panel_height-28-40-95)/3-(font+10))
    right=min(width*.36,max(300*unit,2*thumb+42)) if app.demo_page==0 and maps_open else (0 if app.demo_page==0 else min(350*unit,width*.25))
    app.roadshow_layout=(pad,top,side,right,bottom)
    canvas_key=(id(app.scene),width,height,app.demo_page,right)
    if app.demo_page==0 and getattr(app,'_studio_canvas_key',None)!=canvas_key:
        app._studio_canvas_key=canvas_key;app.move_camera_home()
    with theme():
        panel('##studio_header',pad,pad,width-2*pad,top-2*pad)
        ui.TextColored(ACCENT,'Digital Twin Studio')
        ui.SameLine(250*unit)
        for page,label in enumerate(('Radio Map','Receiver Path')):
            if page:ui.SameLine()
            active=app.demo_page==page
            if active:ui.PushStyleColor(ui.ImGuiCol_Button,(.07,.38,.38,1))
            clicked=ui.Button(('> ' if active else '')+label)
            if active:ui.PopStyleColor()
            if clicked and not active:
                from .demo_workflow import set_page
                set_page(app,page);set_operation(app,'browse')
        end_panel()
        # Stable height based on mode, never feedback from the scrollable child.
        content_height=(810 if getattr(app,'studio_beam_mode','aim')=='select' and app.scene.get('tx-0') else 510)*unit
        workflow_height=min(panel_height,content_height) if app.demo_page==0 else panel_height
        panel('##studio_workflow',pad,top,side,workflow_height)
        # A width-sized sphere can cross the vertical overflow threshold when
        # the scrollbar takes width away. Reserve its gutter to prevent an
        # alternating scrollbar -> smaller sphere -> no scrollbar feedback loop.
        selecting=getattr(app,'studio_beam_mode','aim')=='select' and app.scene.get('tx-0') is not None
        ui.BeginChild('workflow_body',(0,max(100.,ui.GetContentRegionAvail()[1]-(48 if app.demo_page==0 else 175)*unit)),
                      window_flags=ui.ImGuiWindowFlags_AlwaysVerticalScrollbar if selecting else 0)
        ui.PushItemWidth(-1)
        heading('01','Scene');scene_controls(app)
        heading('02','Base Station');station_controls(app)
        heading('03','Beam');beam_controls(app)
        if app.demo_page==0:coverage_settings(app)
        if app.demo_page==0:result_actions(app)
        ui.PopItemWidth();ui.EndChild();ui.Separator()
        if app.demo_page==0:compute_action(app)
        else:link_controls(app)
        end_panel()
        if app.demo_page==0 and maps_open:
            panel('##studio_results',width-pad-right,top,right,panel_height)
            coverage_results(app)
        elif app.demo_page==1:
            panel('##studio_results',width-pad-right,top,right,panel_height)
            link_results(app)
            result_actions(app)
        if app.demo_page==1 or maps_open:
            app.studio_results_scroll_max=ui.GetScrollMaxY()
            end_panel()
        elif maps_available:
            panel('##studio_restore_maps',width-pad-180*unit,top,180*unit,62*unit)
            if ui.Button('Expand RSS maps'):app.studio_maps_collapsed=False
            end_panel()
        panel('##studio_footer',pad,height-bottom,width-2*pad,bottom-pad/2)
        if ui.Button('Overview'):app.move_camera_home()
        ui.SameLine()
        if ui.Button('Top'):camera_view(app,'top')
        ui.SameLine()
        if ui.Button('TX'):camera_view(app,'tx')
        message=operation(app)
        text_width=ui.CalcTextSize(message)[0]
        ui.SameLine(max(285*unit,width-2*pad-text_width-20))
        ui.TextColored(WARN if app.demo_place_transmitter or app.demo_place_receiver or app.demo_aim_beam else MUTED,message)
        end_panel()
        from .campus_controls import draw_device_labels
        draw_device_labels(app)


def result_actions(app):
    ui.Separator();evidence(app)
    from .link_jobs import current_result
    can_export=app.radio_map is not None if app.demo_page==0 else (current_result(app) and not app.animation_config.playing)
    ui.BeginDisabled(not can_export)
    if button('Export snapshot'):
        try:
            path=export_result(app);app.demo_status='Snapshot exported to exports/'+path.name
            app.roadshow_export=str(path)
        except Exception as exc:app.demo_status=f'Export failed: {exc}'
    ui.EndDisabled()


def camera_view(app,kind):
    if kind=='top' and app.demo_page==0:
        from .studio_camera import fit_canvas
        fit_canvas(app,top_view=True);return
    if kind=='route':
        rx=app.scene.get('rx-0')
        if rx is None:return
        points=app.animation_config.trajectories['rx-0'].points
        if len(points)==0:points=host_value(rx,'position').reshape(1,3)
        center=points.mean(axis=0)
        extent=max(25.,float(np.max(np.ptp(points,axis=0))))
        # The old fixed close-up could put the camera inside a street building.
        # Choose a viewpoint with a clear line to the route, using RF geometry
        # only for visibility; this never changes the receiver or its solver.
        targets=np.vstack([points[0],center,points[-1]])
        eye=None
        for offset in ([.9,-1.2,.9],[-.9,-1.2,1.3],[.9,1.2,1.6],
                       [-.9,1.2,1.8],[0.,-.01,2.2],[0.,-.001,4.]):
            candidate=center+np.asarray(offset)*extent
            delta=targets-candidate;length=np.linalg.norm(delta,axis=1)
            ray=mi.Ray3f(mi.Point3f(candidate),mi.Vector3f((delta/length[:,None]).T))
            hit=app.scene.mi_scene.ray_intersect(ray)
            if np.all(hit.t.numpy()>=length-.1):eye=candidate;break
        if eye is None:eye=center+[0.,-.01,max(extent*4,float(app.scene.mi_scene.bbox().max.z)-center[2]+25.)]
        ps.look_at(eye,center)
    elif kind=='top':
        bbox=app.scene.mi_scene.bbox();center=np.array([float(bbox.center().x),float(bbox.center().y),float(bbox.min.z)])
        width,height=ps.get_window_size()
        pad,top,left,right,bottom=app.roadshow_layout
        unit=max(.85,min(1.35,ui.GetFontSize()/20))
        # Fit above the ground AND tallest roof, inside the unobstructed canvas.
        half_h=max(50.,min(height/2-top-62*unit,height/2-bottom))
        half_w=max(50.,min(width/2-left-2*pad,width/2-right-2*pad))
        focal=height/(2*np.tan(np.radians(ps.get_view_camera_parameters().get_fov_vertical_deg())/2))
        altitude=float(bbox.extents().z)+1.08*focal*max(
            float(bbox.extents().x)/(2*half_w),float(bbox.extents().y)/(2*half_h))
        ps.look_at(center+np.array([0.,-.01,altitude]),center)
    else:
        device=app.scene.get('tx-0' if kind=='tx' else 'rx-0')
        if device is None:return
        center=host_value(device,'position').ravel();ps.look_at(center+[45.,-60.,50.],center)
    ps.set_view_center_raw(center)

