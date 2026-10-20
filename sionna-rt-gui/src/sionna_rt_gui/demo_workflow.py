"""Separate area coverage (no receiver) from point-link analysis."""
import numpy as np
import mitsuba as mi
import drjit as dr
import polyscope as ps
import polyscope.imgui as ui
from .sionna_utils import set_or_update_radio_devices_polyscope


def set_page(app, page):
    previous_page=app.demo_page
    views=getattr(app,'_page_cameras',None)
    if views is None:views={};app._page_cameras=views
    if previous_page!=page:views[previous_page]=ps.get_camera_view_matrix().copy()
    app.demo_page = page
    app.demo_aim_beam = False
    app.demo_place_receiver = app.demo_place_transmitter = False
    app.animation_config.playing = False
    if app.link_jobs is not None:app.link_jobs.reset()
    app.beam_demo.scan = False
    app.beam_demo.dirty = True
    app.clear_selection()
    if page == 0:
        rx = app.scene.get('rx-0')
        if rx is not None:
            app.demo_saved_rx = rx
            app.scene.remove(rx.name)
        app.clear_paths()
        set_or_update_radio_devices_polyscope(app.scene.receivers, False, app)
        if ps.has_curve_network('Rx route'):ps.get_curve_network('Rx route').set_enabled(False)
    elif app.scene.transmitters:
        app.pause_coverage()
        if not app.scene.receivers:
            if app.demo_saved_rx is not None:
                app.scene.add(app.demo_saved_rx)
                set_or_update_radio_devices_polyscope(app.scene.receivers, False, app)
            else:
                from .presentation_presets import prepare_route
                route=prepare_route(app)
                app.demo_default_rx=route[0].tolist();app.demo_default_route=route.tolist()
                app.add_radio_device(app.demo_default_rx, False, allow_auto_update=False)
                traj = app.animation_config.trajectories['rx-0']
                for point in app.demo_default_route:traj.add_point(point)
                traj.velocity = float(app.demo_scene_data.get('route_speed_mps',2.0)) if app.demo_scene_data else 2.0
                traj.distance = 0.0
                traj.enabled = False
            app.draw_demo_route()
        app._last_paths_update_time = 0
        app.update_paths(clear_first=True, show=True)
    if ps.has_surface_mesh('Sionna angular scan'):
        ps.get_surface_mesh('Sionna angular scan').set_enabled(page == 1 and app.beam_demo.visible)
    visible, mesh = app.has_visible_radio_map()
    if mesh is not None:mesh.set_enabled(page == 0)
    app.demo_status = 'Select a beam, then compute coverage.' if page == 0 else 'Move the receiver to explore paths and channel response.'
    if page==0 and app.radio_map is not None:app.demo_status='Coverage restored. Resume calculation or add samples.'
    if previous_page!=page:
        if page in views:ps.set_camera_view_matrix(views[page])
        elif page==1:
            from .roadshow_ui import camera_view
            camera_view(app,'route')


def support_below(app, position):
    origin = np.asarray(position, dtype=float)
    hit = app.scene.mi_scene.ray_intersect(mi.Ray3f(mi.Point3f(origin), mi.Vector3f(0,0,-1)))
    distances = [float(hit.t.numpy()[0])]
    if app.render_cache is not None:
        from .rendering import intersect_visual
        distances.append(float(intersect_visual(app.render_cache, origin, [0,0,-1]).t.numpy()[0]))
    distance = min(distances)
    return origin + [0,0,-distance] if np.isfinite(distance) else None


def place_tx(app, position):
    app.beam_target = None
    app.beam_demo.auto_beam_ready=False
    app.beam_demo._auto_signature=None
    app.pause_coverage()
    app.clear_radio_map()
    tx = app.scene.get('tx-0')
    if tx is None:
        tx = app.add_radio_device(position, True, allow_auto_update=False)
        tx.power_dbm = 30.0
    else:
        from .host_values import set_host_position
        set_host_position(tx, position)
        set_or_update_radio_devices_polyscope(app.scene.transmitters, True, app)
    app.beam_demo.dirty = True
    app.beam_demo.signature = None
    app.demo_place_transmitter = False
    app.demo_aim_beam = False
    app.demo_status = 'Base station placed. Aim a beam or compute coverage.'


def place_tx_at_screen(app, point):
    origin = np.asarray(ps.get_view_camera_parameters().get_position())
    destination = ps.screen_coords_to_world_position(point)
    direction = destination-origin
    if not np.all(np.isfinite(direction)) or np.linalg.norm(direction) == 0:return False
    hit = app.scene.mi_scene.ray_intersect(mi.Ray3f(origin, dr.normalize(mi.Vector3f(direction))))
    if app.render_cache is not None:
        from .rendering import intersect_visual
        visual_hit=intersect_visual(app.render_cache,origin,direction/np.linalg.norm(direction))
        if float(visual_hit.t.numpy()[0]) < float(hit.t.numpy()[0]):hit=visual_hit
    if not bool(hit.is_valid()[0]):
        app.demo_status = 'Click a ground or roof surface.'
        return False
    if abs(float(hit.n.z[0]))<.5:
        app.demo_status='Place the base station on a horizontal ground or roof surface.'
        return False
    # Upward placement clearance, independent of the coverage measurement plane.
    place_tx(app, hit.p.numpy().ravel()+[0, 0, 5.0])
    return True


def draw_workflow(app):
    changed, page = ui.Combo('Workspace', app.demo_page, ['Radio Map', 'Receiver Path'])
    if changed:set_page(app, page)
    ui.TextWrapped('Scene → Base station → Beam')
    ui.Text('F11: maximize / restore')
    # Exclude the empty scene; built-in scene assets ship with Sionna RT.
    choices = [i for i,p in enumerate(app.known_scene_paths) if p]
    current = choices.index(app.current_scene_idx) if app.current_scene_idx in choices else 0
    from .scene_style import scene_label
    names = [scene_label(app.known_scene_names[i],app.known_scene_paths[i]) for i in choices]
    changed, index = ui.Combo('Scene', current, names)
    if changed:app.load_scene_requested = app.known_scene_paths[choices[index]]
    ui.Text('PolyU visual style')
    if app.demo_page == 1:return False
    if ui.Button('Cancel placement' if app.demo_place_transmitter else 'Place base station'):
        app.demo_place_transmitter = not app.demo_place_transmitter
    ui.Text('Antenna: 5 m above the selected surface')
    tx = app.scene.get('tx-0')
    if tx is None:
        ui.TextWrapped('Choose Place base station, then click the ground or a roof.')
        return True
    p = tx.position.numpy().ravel()
    if ui.Button('Focus TX'):
        ps.look_at(p+np.array([42,-55,35]),p)
    ui.SameLine()
    if ui.Button('Fit scene'):app.move_camera_home()
    if getattr(app,'beam_target',None) is not None:
        ui.SameLine()
        if ui.Button('Focus target'):
            from .navigation import focus
            focus(app,app.beam_target)
    ui.TextWrapped('Drag to pan · Right-drag to orbit · Scroll to zoom')
    ui.Text(f'TX: {p[0]:.1f}, {p[1]:.1f}, {p[2]:.1f} m')
    angles = np.degrees(tx.orientation.numpy().ravel())
    changed, yaw = ui.SliderFloat('Base station yaw [deg]', float(angles[0]), -180., 180.)
    if changed:
        tx.orientation = mi.Point3f(np.radians([yaw, angles[1], angles[2]]))
        app.beam_demo.dirty = True
    b = app.beam_demo
    ui.Text(f'{b.label}')
    if ui.CollapsingHeader('Beam index'):
        changed, index = ui.SliderInt('Beam index', b.selected, 0, len(b.local_dirs)-1)
        if changed:b.select(index)
        ui.Text(f'Local az {b.az[b.selected]:.1f} | el {b.el[b.selected]:.1f} deg')
    ui.Text(f'TX power: {float(tx.power_dbm.numpy()[0]):.1f} dBm | {app.scene.tx_array.num_ant} elements / panel')
    ui.Text(f'6 panels | {len(b.local_dirs)} beams | one active panel at a time')
    ui.Text(f'Receiver height: ground +{app.cfg.radio_map.height_above_ground:g} m | RSS [dBm]')
    ui.TextWrapped('Click a target in the scene to select the closest beam.')
    if getattr(b,'direction_blocked',False):ui.TextColored((1.,.65,.25,1),'The beam direction is blocked by a building.')
    if app.demo_status:ui.TextWrapped(app.demo_status)
    from .campus_controls import draw_coverage_controls
    draw_coverage_controls(app)
    return True
