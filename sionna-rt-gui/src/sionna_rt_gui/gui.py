#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
import os
import json
import sys
import time

import drjit as dr
import mitsuba as mi
import numpy as np
import matplotlib.pyplot as plt
import polyscope as ps
import polyscope.imgui as psim
from sionna import rt
from sionna.rt.scene_utils import remove_objects_duplicate_vertices

from . import __version__ as GUI_VERSION
from .animation import AnimationConfig, animation_gui, animation_tick
from .demo_radio_map import LAYER_NAME, register_illustrative_map
from .campus_controls import draw_controls, draw_chart, draw_device_labels, merge_coverage
from .coverage_jobs import CoverageJobs, mark_outdoor_cells
from .beamforming import panel_solver
from .antenna_array import antenna_array_gui
from .config import (
    GuiConfig,
    RadioMapConfig,
    PathsConfig,
    GuiMode,
    RenderingMode,
    RENDERING_MODE_NAMES,
    NVIDIA_GREEN,
)
from .rendering import (
    render_scene,
    set_envmap_rotation,
    add_or_update_ray_traced_image_quantity,
)
from .rm_utils import radio_map_colorbar_to_image
from .ps_utils import (
    set_custom_imgui_style,
    set_polyscope_device_interop_funcs,
    supports_direct_update_from_device,
)
from .sionna_utils import (
    add_paths_to_polyscope,
    add_radio_map_to_polyscope,
    add_scene_to_polyscope,
    get_built_in_scenes,
    get_point_from_camera_center_ray,
    get_normal_for_path,
    set_or_update_radio_devices_polyscope,
)
from .selection import SelectionType, selection_gui

CTRL_OR_CMD = "Cmd" if sys.platform == "darwin" else "Ctrl"
HELP_WINDOW_TABLES = {
    "Camera controls": {
        "Left click + drag": "Camera rotation",
        "Right click + drag": "Camera panning",
        "Shift + left click + drag": "Camera panning",
        "Double left click": "Center camera rotation around clicked point",
        "Mouse scroll": "Camera zoom",
        f"{CTRL_OR_CMD} + shift + left click + drag": "Continuous camera zoom",
        "R": "Reset camera to initial position",
        "F": "Fit scene to camera",
    },
    "Mouse bindings": {
        "Left click": "Select a radio device",
        f"{CTRL_OR_CMD} + left click": "Add transmitter",
        f"{CTRL_OR_CMD} + right click": "Add receiver",
    },
    "Key bindings": {
        "H / ?": "Show this help window",
        "K": "Add transmitter at the current mouse position",
        "L": "Add receiver at the current mouse position",
        "M": "Show / hide radio map. Requires at least one transmitter.",
        "C": "Go to next rendering mode",
        "Tab": "Go to next GUI mode (can be used to hide the GUI)",
        "Esc": "Close help window or de-select current object",
        "Del (with item selected)": "Delete radio device",
        "Shift + R": "Reload application",
        f"{CTRL_OR_CMD} + Q": "Exit",
    },
    "Slice plane": {
        "S": "Toggle slice plane visibility (not supported in ray-traced rendering mode)",
        "Alt + left click drag": "Move slice plane along its normal",
    },
}


class SionnaRtGui:
    def __init__(self, cfg: GuiConfig):
        self.cfg = cfg

        # --- Sionna RT
        # Scene
        built_in_scenes = get_built_in_scenes()
        self.known_scene_names = ["None"] + list(built_in_scenes.keys())
        self.known_scene_paths = [None] + list(built_in_scenes.values())
        self.current_scene_idx: int = 0
        self.load_scene_requested: str | None = None
        self.scene: rt.Scene | None = None

        # Radio map results
        self.radio_map: rt.RadioMap | None = None
        self.rm_accumulated_samples: int = 0
        self.rm_color_map_options = [v for v in plt.colormaps() if not v.endswith("_r")]
        self.rm_color_map_index = self.rm_color_map_options.index(
            self.cfg.radio_map.color_map
        )
        self.rm_colorbar: np.ndarray | None = None
        self.rm_colorbar_texture_id: int | None = None
        self.demo_colorbar: np.ndarray | None = None
        self.demo_colorbar_texture_id: int | None = None
        self.rm_refining = False
        self.coverage_jobs = None
        self.link_jobs = None
        self.demo_link_cir = None
        self.demo_link_settings = None
        from .beam_demo import BeamDemo
        self.beam_demo = BeamDemo()
        self.demo_page = 0
        self.demo_aim_beam = False
        self.demo_place_transmitter = False
        self.demo_tx_offset = 5.0
        self.demo_saved_rx = None
        self._last_rm_batch_samples = 0
        self.demo_chart_dirty = True
        self.demo_chart_texture = None
        self.demo_chart_mode = 0
        self.demo_path_delays = np.array([])
        self.demo_path_powers = np.array([])
        self.demo_advanced = False
        self.demo_edit_route = False
        self.demo_placement_mode = 0
        self.demo_status = ""
        self.demo_place_receiver: bool = False
        self.demo_cir_values = np.zeros(64, dtype=np.float32)
        self.demo_cir_range_ns: float = 100.0
        self.demo_cir_path_count: int = 0
        # Due to a current limitation, we can't have alpha on radio maps when using
        # direct updates from the device.
        self.cfg.radio_map.use_alpha = not (
            self.cfg.radio_map.use_direct_update_from_device
            and supports_direct_update_from_device()
        )

        # Paths results
        self.paths: rt.Paths | None = None
        # Shape: [num_rx, num_rx_ant, num_tx, num_tx_ant, num_time_steps, l_max - l_min + 1]
        self.paths_taps: np.ndarray | None = None
        self.paths_cir: tuple[np.ndarray, np.ndarray] | None = None
        # Used to throttle path computations
        self._last_paths_update_time: float = 0.0

        # --- Rendering
        self.render_cache: dict = None
        self.ray_traced_img: mi.TensorXf | None = None
        self.ray_traced_depth: mi.TensorXf | None = None
        self.previous_camera_pose: np.ndarray = ps.get_camera_view_matrix()
        self.rendering_accumulated_samples: int = 0
        self.reset_accumulation_requested: bool = False
        self.denoiser: mi.OptixDenoiser | None = None
        self.slice_plane: ps.SlicePlane | None = None

        # --- Animation state
        self.animation_config: AnimationConfig = AnimationConfig()

        # --- Inputs state
        self.last_mouse_pos: mi.ScalarVector2f | None = None

        self.snapshot_load_requested: bool = False
        self.code_reload_requested: bool = False

        # --- Selections
        self.selected_object: rt.SceneObject | None = None
        self.selected_type: SelectionType | None = None
        self.prev_gizmo_to_world: np.ndarray | None = None

        # --- Polyscope setup
        # Can be used to derive e.g. random seeds.
        self.ps_groups: dict[str, ps.Group] = {}
        self.frame_i: int = 0
        self.was_mouse_dragging: bool = False
        self.home_camera_to_world: np.ndarray | None = None

        # Pre-init settings
        ps.set_program_name(self.cfg.title)
        # Window size and position will be loaded from the last run from `.polyscope.ini`
        ps.set_use_prefs_file(not self.cfg.demo_mode)
        ps.set_enable_vsync(cfg.use_vsync)
        # On some machines (especially with remote access), VSync doesn't do anything,
        # so we cap fps as well.
        ps.set_max_fps(60)
        ps.set_user_gui_is_on_right_side(False)
        ps.set_open_imgui_window_for_user_callback(False)
        ps.set_verbosity(1)
        ps.set_build_default_gui_panels(self.cfg.show_polyscope_gui)
        ps.set_background_color(self.cfg.background_color)
        ps.set_ground_plane_mode("none")
        ps.set_window_resizable(True)
        ps.set_give_focus_on_show(True)
        ps.set_transparency_mode("pretty")
        ps.set_files_dropped_callback(self.on_files_dropped)
        set_custom_imgui_style()

        was_initialized = ps.is_initialized()
        if not was_initialized:
            ps.set_up_dir("z_up")
            ps.set_front_dir("y_front")
            font = os.path.join(os.environ.get('WINDIR', 'C:/Windows'), 'Fonts', 'msyh.ttc')
            if self.cfg.demo_mode and os.path.isfile(font):
                def prepare_fonts(atlas):
                    default=atlas.AddFontFromFileTTF(font,22.0)
                    mono=atlas.AddFontFromFileTTF(font,22.0)
                    self.roadshow_fonts={size:atlas.AddFontFromFileTTF(font,float(size)) for size in (18,22,30)}
                    return default,mono
                ps.set_prepare_imgui_fonts_callback(prepare_fonts)
            ps.init()
            if supports_direct_update_from_device():
                set_polyscope_device_interop_funcs()

        # Set window size once UI scale is known (after init).
        if self.cfg.demo_mode:ps.set_do_default_mouse_interaction(False)
        self.ui_scale: float = ps.get_ui_scale()
        ps.set_window_size(
            self.cfg.rendering.default_resolution[0] * self.ui_scale,
            self.cfg.rendering.default_resolution[1] * self.ui_scale,
        )
        self.fullscreen_window = None
        if self.cfg.demo_mode:
            from .desktop_window import FullscreenWindow
            self.fullscreen_window = FullscreenWindow()
            self.fullscreen_window.toggle()
        # Used to track window size changes. Includes DPI scaling.
        self.previous_window_resolution: tuple[int, int] = ps.get_window_size()
        # Note that our `set_window_size()` call may not succeed, e.g. if the window is
        # maximized. We adopt the effective resolution to make sure that we rendering with
        # the right aspect ratio, etc.
        self.cfg.rendering.current_resolution = self.render_pixel_resolution(
            self.previous_window_resolution
        )
        # Enable denoiser if requested.
        if "cuda" in mi.variant():
            self.set_use_denoiser(self.cfg.rendering.use_denoiser)
        else:
            # If there is no CUDA-compatible GPU, switch to rasterization (cheaper).
            self.cfg.rendering.mode = RenderingMode.RASTERIZATION

        # Used to throttle window size change handling. Set to None if no change is pending.
        self.last_window_size_changed_time: float | None = None

        # --- Load scene
        # TODO: preserve currently-selected scene across reloads
        if not self.cfg.scene_filename.endswith(".xml"):
            if self.cfg.scene_filename not in built_in_scenes:
                raise ValueError(
                    f'Scene "{self.cfg.scene_filename}" not found. Built-in scenes: {list(built_in_scenes.keys())}'
                )
            self.cfg.scene_filename = built_in_scenes[self.cfg.scene_filename]
        self.load_scene(
            self.cfg.scene_filename,
            # If the program was just reloaded (live coding), don't move the camera
            recenter_camera=not was_initialized,
        )

        # --- Slice plane
        ps.remove_all_slice_planes()
        plane = ps.add_scene_slice_plane()
        plane.set_pose(
            (
                self.cfg.rendering.slice_plane_position
                if self.cfg.rendering.slice_plane_position is not None
                else (0, 0, self.scene.mi_scene.bbox().center().z)
            ),
            self.cfg.rendering.slice_plane_normal,
        )
        plane.set_active(
            self.cfg.rendering.default_slice_plane_enabled
            and (self.cfg.rendering.mode == RenderingMode.RASTERIZATION)
        )
        self.slice_plane = plane

        # --- Example scenario
        if self.cfg.demo_mode and not self.cfg.create_example_scenario:
            self.create_demo_scenario()
        elif self.cfg.create_example_scenario:
            self.create_example_scenario(
                set_camera=not was_initialized, add_radio_map=False
            )
            if self.cfg.demo_mode:
                self.animation_config.playing = False
                self.show_illustrative_map()

    def create_demo_scenario(self):
        """Place a fixed rooftop transmitter and a movable receiver."""
        self._page_cameras={}
        self.demo_scene_data = None
        if self.cfg.scene_filename.replace("\\", "/").endswith(("polyu-campus/scene.xml", "polyu-official/scene.xml", "polyu-stylized/scene.xml", "mong-kok/scene.xml", "kam-tin/scene.xml")):
            with open(os.path.join(os.path.dirname(self.cfg.scene_filename), "scenario.json"), encoding="utf-8") as source:
                self.demo_scene_data = json.load(source)
        is_blender_city = self.cfg.scene_filename.replace("\\", "/").endswith(
            "blender-city/scene.xml"
        )
        if self.demo_scene_data is not None:
            tx = self.demo_scene_data["tx"]
            rx = self.demo_scene_data["rx"]
            route = self.demo_scene_data["route"]
            target = self.demo_scene_data["target"]
            camera = self.demo_scene_data["camera"]
        elif is_blender_city:
            tx = [45.0, 54.0, 83.5]
            rx = [55.0, -69.0, 47.5]
            route = (rx, [60.0, -69.0, 47.5], [64.0, -69.0, 47.5])
            target = [0.0, 5.0, 10.0]
            camera = [185.0, -215.0, 155.0]
        else:
            tx = [-130.0, -95.0, 16.0]
            rx = [-30.0, -95.0, 18.36]
            route = (rx, [-30.0, -82.0, 18.36], [-30.0, -70.0, 18.36])
            target = [-95.0, -95.0, 8.0]
            camera = [target[0] + 115.0, target[1] - 150.0, 230.0]
        self.demo_default_rx = rx
        self.demo_default_route = route
        self.air_route_mode = bool(self.demo_scene_data and self.demo_scene_data.get('route_kind')=='air')
        self.demo_saved_rx = None
        self.demo_aim_beam = False
        if self.link_jobs is not None:self.link_jobs.reset()
        self.demo_link_settings=None;self.demo_link_cir=None
        self.spatial_beam_mode = 0
        self.beam_target = None
        self.demo_page = 0
        self.demo_place_receiver = False
        self.demo_place_transmitter = False
        self.beam_demo.map_due = None
        self.beam_demo.map_requested = False
        self.beam_demo.map_signature = None
        self.beam_demo.center = None
        self.beam_demo.signature = None
        self.beam_demo.dirty = True
        self.beam_demo.scan = False
        self.beam_demo.ready = False
        self.animation_config.playing = False
        if self.demo_scene_data is None:
            bbox = self.scene.mi_scene.bbox()
            self.cfg.radio_map.center = (float(bbox.center().x), float(bbox.center().y), float(bbox.min.z)+1.5)
            self.cfg.radio_map.size = (float(bbox.extents().x), float(bbox.extents().y))
            self.demo_default_rx = [float(bbox.center().x), float(bbox.center().y), float(bbox.min.z)+1.5]
            self.demo_default_route = [self.demo_default_rx]
            self.fit_camera_to_scene()
        else:
            bounds = self.demo_scene_data['coverage_bounds']
            self.cfg.radio_map.center = ((bounds[0]+bounds[1])/2, (bounds[2]+bounds[3])/2, float(self.demo_scene_data['rx'][2]))
            self.cfg.radio_map.size = (bounds[1]-bounds[0], bounds[3]-bounds[2])
            ps.look_at(camera, target)
        self.home_camera_to_world = ps.get_camera_view_matrix()
        self.roadshow_baseline = None
        self.demo_status = 'Choose a base station, select a beam, then compute coverage.'

    def show_illustrative_map(self):
        """Show synthetic coverage on the current scene for UI review."""
        if self.cfg.demo_mode:
            return  # Demo now uses only Sionna-computed coverage.
        if not self.scene._transmitters:
            return
        bbox = self.scene.mi_scene.bbox()
        visible = (ps.has_surface_mesh(LAYER_NAME)
                   and ps.get_surface_mesh(LAYER_NAME).is_enabled())
        tx = next(iter(self.scene._transmitters.values())).position.numpy().squeeze()
        is_blender_city = self.cfg.scene_filename.replace("\\", "/").endswith(
            "blender-city/scene.xml"
        )
        width = 260.0 if is_blender_city else min(370.0, bbox.max.x - bbox.min.x)
        height = 200.0 if is_blender_city else min(270.0, bbox.max.y - bbox.min.y)
        center = (0.0, 0.0) if is_blender_city else (float(tx[0]), float(tx[1]))
        bounds = (center[0] - width / 2, center[0] + width / 2,
                  center[1] - height / 2, center[1] + height / 2)
        elevation = 0.5 if is_blender_city else bbox.min.z + 1.7
        scene_data = getattr(self, "demo_scene_data", None)
        if scene_data is not None:
            bounds = scene_data["coverage_bounds"]
            elevation = scene_data["coverage_z"]
        vmin, vmax = register_illustrative_map(
            bounds,
            elevation,
            (float(tx[0]), float(tx[1])),
            self.ps_groups["radio_maps"],
            visible=visible,
        )
        if self.demo_colorbar_texture_id is None:
            self.demo_colorbar = radio_map_colorbar_to_image(
                "turbo", vmin, vmax, unit="dBm"
            )
            ps.add_color_alpha_image_quantity(
                "demo_colorbar", values=self.demo_colorbar, enabled=True,
                image_origin="upper_left", show_fullscreen=False,
            )
            self.demo_colorbar_texture_id = ps.get_quantity_buffer(
                "demo_colorbar", "colors"
            ).get_texture_native_id()

    def move_demo_receiver(self, position: np.ndarray | list[float]):
        """Move only the receiver, then refresh its physical channel results."""
        rx = self.scene.get("rx-0")
        from .host_values import set_host_position
        set_host_position(rx, position)
        self.animation_config.playing = False
        self.animation_config.trajectories[rx.name].enabled = False
        set_or_update_radio_devices_polyscope(
            self.scene.receivers, is_transmitter=False, gui=self
        )
        self.update_paths(clear_first=True, show=True)

    def draw_demo_route(self):
        traj = self.animation_config.trajectories["rx-0"]
        if ps.has_curve_network("Rx route"):
            ps.remove_curve_network("Rx route")
        if len(traj) > 1:
            route = ps.register_curve_network("Rx route", traj.points, edges="line",
                color=(0.18, 0.65, 0.68), enabled=True)
            route.set_radius(0.12, relative=False)

    def restore_demo_route(self):
        if not self.demo_default_route:
            return
        traj = self.animation_config.trajectories["rx-0"]
        traj.clear()
        for p in self.demo_default_route:
            traj.add_point(p)
        traj.distance = 0.0
        self.move_demo_receiver(traj.points[0])
        self.draw_demo_route()
        self.demo_status = 'Example route restored. Play or edit the waypoints.'

    def edit_demo_route(self, action):
        traj=self.animation_config.trajectories['rx-0']
        self.animation_config.playing=False
        points=traj.points[:-1].copy() if action=='undo' else self.scene.get('rx-0').position.numpy().reshape(1,3)
        traj.clear()
        for point in points:traj.add_point(point)
        if len(traj):self.move_demo_receiver(traj.points[-1])
        self.draw_demo_route()
        self.demo_status='Last waypoint removed.' if action=='undo' else 'New route starts at RX. Click to add waypoints.'

    def demo_segment_clear(self, start, end):
        delta = np.asarray(end)-np.asarray(start)
        distance = float(np.linalg.norm(delta))
        if distance < 0.05:
            return False
        ray = mi.Ray3f(mi.Point3f(start), mi.Vector3f(delta/distance))
        ray.maxt = distance
        if bool(self.scene.mi_scene.ray_test(ray)[0]):
            return False
        # Ground routes must have support without jumping onto roofs.
        if self.demo_placement_mode == 0:
            for t in np.linspace(0, 1, max(2, int(distance/2)+1)):
                p = np.asarray(start)+t*delta
                hit = self.scene.mi_scene.ray_intersect(mi.Ray3f([float(p[0]),float(p[1]),500],[0,0,-1]))
                if not bool(hit.is_valid()[0]) or abs(float(hit.p.z[0])-(p[2]-1.5)) > 0.8:
                    return False
        return True

    def place_demo_receiver_at_screen_point(self, screen_point) -> bool:
        if getattr(self,"air_route_mode",False):
            from .air_routes import place
            return place(self,screen_point)
        origin = np.asarray(ps.get_view_camera_parameters().get_position())
        destination = ps.screen_coords_to_world_position(screen_point)
        if not np.all(np.isfinite(destination)):
            return False
        direction = destination-origin
        if np.linalg.norm(direction) == 0:
            return False
        hit = self.scene.mi_scene.ray_intersect(mi.Ray3f(origin,dr.normalize(mi.Vector3f(direction))))
        if not bool(hit.is_valid()[0]):
            self.demo_status = 'No surface here. Click the ground or a roof.'
            return False
        if abs(float(hit.n.z[0])) < .5:
            self.demo_status = 'Choose a horizontal ground or roof surface.'
            return False
        position = hit.p.numpy().squeeze()+[0,0,1.5]
        traj = self.animation_config.trajectories["rx-0"]
        if self.demo_edit_route and len(traj) and not self.demo_segment_clear(traj.points[-1],position):
            self.demo_status = 'Route blocked or height changes sharply. Add an intermediate waypoint.'
            return False
        self.move_demo_receiver(position)
        if self.demo_edit_route:
            traj.add_point(position)
        self.draw_demo_route()
        self.demo_status = 'Waypoint added. Click to add more; Esc to finish.' if self.demo_edit_route else 'Receiver moved. Route retained; updating the link.'
        if not self.demo_edit_route:self.demo_place_receiver=False
        return True

    def update_demo_cir(self):
        """Bin Sionna CIR path powers for the live receiver plot."""
        self.demo_chart_dirty = True
        self.demo_path_delays = np.array([])
        self.demo_path_powers = np.array([])
        self.demo_cir_values = np.zeros(64, dtype=np.float32)
        self.demo_cir_range_ns = 100.0
        self.demo_cir_path_count = 0
        if self.demo_link_cir is not None:
            amplitudes, delays = self.demo_link_cir
        elif self.paths_cir is not None:
            amplitudes, delays = self.paths.cir(normalize_delays=False, out_type="numpy")
        else:
            self.beam_demo.invalidate(self)
            return
        power = self.beam_demo.consume(self, amplitudes)
        delays = np.asarray(delays)
        if power.size == 0 or delays.size == 0:
            return
        delays_ns = delays.reshape(-1, delays.shape[-1])[0] * 1e9
        valid = np.isfinite(delays_ns) & np.isfinite(power) & (power > 0)
        if not np.any(valid):
            return
        self.demo_path_delays = delays_ns[valid]
        self.demo_path_powers = power[valid]
        self.demo_cir_path_count = int(np.count_nonzero(valid))
        self.demo_cir_range_ns = max(100.0, float(np.ceil(np.max(delays_ns[valid]) / 100) * 100))
        bins, _ = np.histogram(
            delays_ns[valid], bins=64, range=(0, self.demo_cir_range_ns),
            weights=power[valid],
        )
        self.demo_cir_values = (bins / np.max(bins)).astype(np.float32)

    def create_example_scenario(
        self, set_camera: bool = True, add_radio_map: bool = True
    ):
        if set_camera:
            self.home_camera_to_world = np.array(
                [
                    [
                        2.0079615e-03,
                        -9.9999154e-01,
                        -3.9256822e-08,
                        4.4776478e00,
                    ],
                    [7.8317523e-01, 1.5742097e-03, 6.2179816e-01, 1.1707677e01],
                    [
                        -6.2179959e-01,
                        -1.2489425e-03,
                        7.8317869e-01,
                        -2.2836572e02,
                    ],
                    [0.0000000e00, 0.0000000e00, 0.0000000e00, 1.0000000e00],
                ]
            )
            self.move_camera_home()

        # Add some example transmitters
        for pos in [
            [-34.0, 13.0, 33.0],
        ]:
            self.add_radio_device(pos, is_transmitter=True, allow_auto_update=False)

            shifted = [pos[0] + 15, pos[1] - 14, pos[2] - 20]
            self.add_radio_device(
                shifted, is_transmitter=False, allow_auto_update=False
            )

        # Example animation
        p = self.scene.get("rx-0").position.numpy().squeeze()
        traj = self.animation_config.trajectories["rx-0"]
        traj.add_point(p - [0, 30, 0])
        traj.add_point(p)
        traj.add_point(p + [40, 0, 0])
        traj.enabled = True
        traj.distance = 0.0  # Start at the first point
        self.animation_config.playing = True
        self.animation_config.speed_multiplier = 10.0

        if add_radio_map:
            self.start_radio_map()
        if self.cfg.paths.auto_update:
            self.update_paths(show=True)

    def reset_and_setup_structures(self):
        if self.link_jobs is not None:self.link_jobs.reset()
        self._navigation_preview=False;self._navigation_until=0.
        # Clear Sionna state
        self.clear_radio_map()
        self.clear_selection()
        self.clear_paths()
        self.clear_ray_traced_image()
        self.animation_config.clear()

        # Clear Polyscope state
        ps.remove_all_structures()
        ps.remove_all_groups()

        self.ps_groups = {
            "scene": ps.create_group("Scene meshes"),
            "rd": ps.create_group("Radio devices"),
            "radio_maps": ps.create_group("Radio maps"),
            "paths": ps.create_group("Paths"),
        }

    def load_scene(self, scene_path: str, recenter_camera: bool = True):
        load_started=time.perf_counter()
        scene_path = os.path.abspath(scene_path)
        # Bound geometry reuse to two scenes; never reuse transmitters/receivers.
        from pathlib import Path
        root=Path(scene_path).parent
        stamps=tuple((str(p),p.stat().st_mtime_ns,p.stat().st_size)
                     for p in sorted(root.rglob('*'))
                     if p.is_file() and p.suffix.lower() in ('.xml','.ply','.obj','.json'))
        key=(scene_path,stamps,self.cfg.radio_material_thickness,
             self.cfg.radio_material_scattering_coefficient)
        cache=getattr(self,'_scene_cache',{})
        self._scene_cache=cache
        scene=cache.pop(key,None)
        cache_hit=scene is not None
        if scene is None:
            # A failed load leaves the previous view usable.
            scene=rt.load_scene(scene_path)
            remove_objects_duplicate_vertices(scene.mi_scene)
            scene._gui_initial_frequency=float(scene.frequency.numpy()[0])
        else:
            scene.frequency=scene._gui_initial_frequency
        self.reset_and_setup_structures()
        self.scene=scene
        for device in list(scene.transmitters)+list(scene.receivers):scene.remove(device)
        cache[key]=scene
        while len(cache)>2:cache.pop(next(iter(cache)))
        rf_loaded=time.perf_counter()
        self.scene.studio_style = self.cfg.demo_mode
        visuals_path = os.path.join(os.path.dirname(scene_path), "visuals.json")
        if os.path.isfile(visuals_path):
            with open(visuals_path, encoding="utf-8") as source:
                self.scene.demo_visuals = json.load(source)
            for asset in self.scene.demo_visuals:
                for key in ('mesh', 'texture'):
                    if asset.get(key) and not os.path.isabs(asset[key]):
                        asset[key] = os.path.abspath(os.path.join(os.path.dirname(visuals_path), asset[key]))

        self.scene.tx_array = self.cfg.tx_array.create()
        self.scene.rx_array = self.cfg.rx_array.create()

        thickness = self.cfg.radio_material_thickness
        scattering_coefficient = self.cfg.radio_material_scattering_coefficient
        for sh in self.scene.mi_scene.shapes():
            bsdf = sh.bsdf()
            if isinstance(bsdf, rt.RadioMaterialBase):
                if thickness is not None:
                    bsdf.thickness = thickness
                if scattering_coefficient is not None:
                    bsdf.scattering_coefficient = scattering_coefficient

        self.cfg.scene_filename = scene_path
        from .scene_heights import configure_scene_heights
        configure_scene_heights(self)

        # Add this scene to the list of known scene names, if missing
        matched=next((p for p in self.known_scene_paths if p and os.path.abspath(p)==os.path.abspath(scene_path)),None)
        if matched is not None:
            self.cfg.scene_filename=scene_path=matched
        if scene_path not in self.known_scene_paths:
            self.known_scene_names.append(scene_path)
            self.known_scene_paths.append(scene_path)

        try:
            self.current_scene_idx = self.known_scene_paths.index(
                self.cfg.scene_filename
            )
        except ValueError:
            self.current_scene_idx = 0

        add_scene_to_polyscope(self.scene, self.ps_groups)
        self.scene_load_timing=dict(scene=scene_path,rf_cache_hit=cache_hit,rf_seconds=rf_loaded-load_started,
            display_seconds=time.perf_counter()-rf_loaded,
            bake_cache_hit=getattr(self.scene,'gui_bake_cache_hit',False))
        print('[scene-load] '+json.dumps(self.scene_load_timing),flush=True)
        self.set_rendering_mode(self.cfg.rendering.mode)
        if recenter_camera:
            self.fit_camera_to_scene()

        if (self.cfg.rendering.slice_plane_position is None) and (
            self.slice_plane is not None
        ):
            self.slice_plane.set_pose(
                (0, 0, self.scene.mi_scene.bbox().center().z),
                self.cfg.rendering.slice_plane_normal,
            )

    def on_files_dropped(self, files: list[str]):
        for file in files:
            if file.endswith(".xml"):
                print(f"[i] Loading dropped XML file: {file}")
                try:
                    self.load_scene(file)
                    if self.cfg.demo_mode:self.create_demo_scenario()
                except Exception as e:
                    print(f'[!] Failed loading scene "{file}":\n{e}')
                    continue
                # Only handle one valid XML file
                break

    def move_camera_home(self):
        """Move camera to the home view, if any. Otherwise, fits the scene in the camera viewport."""
        if self.home_camera_to_world is not None:
            # We probably can't rely on ps.pick() since the buffers probably haven't been
            # updated yet with the new camera pose. So we trace our own ray to get the rotation
            # point of the turntable navigation.
            if ps.get_navigation_style() == "turntable":
                center = get_point_from_camera_center_ray(
                    self.scene, np.linalg.inv(self.home_camera_to_world)
                )
                if center is not None:
                    ps.set_view_center_raw(center)

            ps.set_camera_view_matrix(self.home_camera_to_world)

            ps.request_redraw()
        else:
            self.fit_camera_to_scene()

        if self.cfg.demo_mode and self.demo_page==0 and hasattr(self,'roadshow_layout'):
            from .studio_camera import fit_canvas
            fit_canvas(self)

    def fit_camera_to_scene(self):
        """Move the camera to a position where most of the scene is visible."""
        fov_vertical_deg = ps.get_view_camera_parameters().get_fov_vertical_deg()
        bbox = self.scene.mi_scene.bbox()

        center = bbox.center()
        extents = bbox.extents()

        # Calculate the required distance to fit the scene in the vertical FOV
        # We use the larger of height or diagonal extent for safety margin
        scene_diagonal = np.linalg.norm(extents)
        scene_size = max(extents.z, scene_diagonal * 0.8)  # 0.8 for some margin

        # Convert FOV to radians and calculate required distance
        fov_vertical_rad = np.radians(fov_vertical_deg)
        distance = scene_size / (2.0 * np.tan(0.5 * fov_vertical_rad))
        distance = distance * 0.95

        # Position camera at a reasonable viewing angle.
        # Angle above horizontal
        es, ec = dr.sincos(np.radians(45))
        # Angle from positive X axis
        zs, zc = dr.sincos(np.radians(0))
        origin = [
            center.x + distance * ec * zc,
            center.y + distance * ec * zs,
            center.z + distance * es,
        ]
        target = [center.x - 0.2 * (center.x - origin[0]), center.y, center.z]

        ps.set_view_center_raw(target)
        ps.look_at(origin, target)

    # ------------------------

    def tick(self):
        # Present the loading status for a complete frame before scene upload.
        request=self.load_scene_requested
        if request is not None and getattr(self,'_announced_scene',None)!=request:
            self._announced_scene=request
            self.animation_config.playing=False
            self.pause_coverage()
            self.demo_status='Loading scene...'
        elif request is not None:
            try:
                self.load_scene(self.load_scene_requested)
                if self.cfg.demo_mode:
                    self.create_demo_scenario()
                    elapsed=self.scene_load_timing['rf_seconds']+self.scene_load_timing['display_seconds']
                    self.demo_status=f'Scene ready ({elapsed:.1f} s). Place a base station, then aim or select a beam.'
            except Exception as e:
                print(f'[!] Failed loading scene "{self.load_scene_requested}":\n{e}')
                self.demo_status=f'Scene loading failed: {e}'
            self.load_scene_requested = None
            self._announced_scene = None

        self.process_inputs()

        # --- Resolution changes
        current_window_resolution = ps.get_window_size()
        now = time.time()
        if current_window_resolution != self.previous_window_resolution:
            self.last_window_size_changed_time = now
            self.previous_window_resolution = current_window_resolution
        if (self.last_window_size_changed_time is not None) and (
            now - self.last_window_size_changed_time > 0.3
        ):
            self.clear_ray_traced_image()
            self.set_rendering_resolution(current_window_resolution)
            self.last_window_size_changed_time = None

        # While navigating, use the same baked PolyU-style meshes. Ray tracing
        # resumes after the gesture settles, so a render cannot stall dragging.
        if self.cfg.demo_mode and not np.allclose(
                self.previous_camera_pose, ps.get_camera_view_matrix()):
            # Includes preset buttons and changes outside mouse navigation.
            self._navigation_until=max(getattr(self,'_navigation_until',0.),time.monotonic()+.9)
        interactive=(self.cfg.demo_mode and (self.animation_config.playing or time.monotonic()<getattr(self,'_navigation_until',0.)))
        preview=interactive and self.cfg.rendering.mode==RenderingMode.RAY_TRACING
        if preview!=getattr(self,'_navigation_preview',False):
            self.clear_ray_traced_image(preserve_cache=True)
            self.ps_groups['scene'].set_enabled(preview or self.cfg.rendering.mode!=RenderingMode.RAY_TRACING)
            self.reset_accumulation_requested=True
            self._navigation_preview=preview
        # --- Rendering
        if self.cfg.rendering.mode == RenderingMode.RAY_TRACING and not preview:
            camera_changed = self.reset_accumulation_requested or not np.allclose(
                self.previous_camera_pose, ps.get_camera_view_matrix()
            )
            if camera_changed:
                self.rendering_reset_accumulation()
                self.reset_accumulation_requested = False

            if self.cfg.demo_mode:
                from .responsive_render import tick as display_tick
                display_tick(self)
            elif (
                self.rendering_accumulated_samples
                < self.cfg.rendering.max_accumulated_spp
            ):
                # TODO: we could potentially skip rendering depth in subsequent frames,
                #       since we only accumulate RGB.
                new_img, aovs, self.render_cache = render_scene(
                    self.cfg.rendering,
                    self.scene,
                    seed=self.frame_i,
                    camera_changed=camera_changed,
                    cache=self.render_cache,
                    use_denoiser=self.denoiser is not None,
                )
                if self.ray_traced_img is None:
                    self.ray_traced_img = new_img
                    self.ray_traced_depth = aovs[0]
                    if self.denoiser is not None:
                        self._denoise_albedo = aovs[1]
                        self._denoise_normals = aovs[2]
                else:
                    t = self.cfg.rendering.spp_per_frame / (
                        self.rendering_accumulated_samples
                        + self.cfg.rendering.spp_per_frame
                    )
                    t = dr.opaque(mi.Float32, t)
                    self.ray_traced_img = (1 - t) * self.ray_traced_img + t * new_img
                    if self.denoiser is not None:
                        self._denoise_albedo = (1-t)*self._denoise_albedo+t*aovs[1]
                        self._denoise_normals = (1-t)*self._denoise_normals+t*aovs[2]
                    # Keep using 1spp depth, it looks better than accumulating.
                    self.ray_traced_depth = aovs[0]

                self.rendering_accumulated_samples += self.cfg.rendering.spp_per_frame

                # Keep raw accumulation separate from the denoised display.
                # Feeding denoised pixels back into averaging smears detail.
                display_img = self.ray_traced_img
                if self.denoiser is not None:
                    to_sensor = self.render_cache["sensor"].world_transform().inverse()
                    display_img = self.denoiser(
                        self.ray_traced_img,
                        albedo=self._denoise_albedo,
                        normals=self._denoise_normals,
                        to_sensor=to_sensor,
                    )

                add_or_update_ray_traced_image_quantity(
                    display_img, self.ray_traced_depth
                )

        if self.coverage_jobs is not None:
            self.coverage_jobs.tick(self)
        # Synchronous API remains available for non-demo callers and numerical tests.
        if (self.coverage_jobs is None and self.radio_map is not None and self.rm_refining
                and self.has_visible_radio_map()[0]
                and self.rm_accumulated_samples < self.cfg.radio_map.accumulate_max_samples_per_tx):
            rm_new = self.compute_radio_map()
            if rm_new is not None:
                merge_coverage(self, rm_new, self._last_rm_batch_samples)
                add_radio_map_to_polyscope(
                    "radio_map", self.radio_map, self.ps_groups, self.cfg.radio_map,
                    direct_update_from_device=self.cfg.radio_map.use_direct_update_from_device,
                    use_alpha=self.cfg.radio_map.use_alpha)
            if self.rm_accumulated_samples >= self.cfg.radio_map.accumulate_max_samples_per_tx:
                self.rm_refining = False

        # --- Radio device animations
        animation_tick(self, psim.GetIO().DeltaTime)
        if self.link_jobs is not None:
            self.link_jobs.tick(self)

        # Save the camera used for this rendered frame BEFORE GUI buttons may
        # change it. Otherwise their new camera is mistaken for an already
        # rendered pose and the image/depth remain at the previous viewpoint.
        self.previous_camera_pose = ps.get_camera_view_matrix()
        # --- GUI
        self.gui()
        self.frame_i += 1

    # ------------------------

    def set_rendering_mode(self, mode: RenderingMode):
        self.cfg.rendering.mode = mode
        is_ray_tracing = self.cfg.rendering.mode == RenderingMode.RAY_TRACING

        if is_ray_tracing and (self.slice_plane is not None):
            self.slice_plane.set_active(False)

        if not is_ray_tracing:
            self.clear_ray_traced_image()

        # Hide Polyscope-side meshes if we are ray tracing.
        self.ps_groups["scene"].set_enabled(not is_ray_tracing)

    def set_use_denoiser(self, use_denoiser: bool):
        use_denoiser = use_denoiser and "cuda" in mi.variant()
        self.cfg.rendering.use_denoiser = use_denoiser

        # The integrator will be re-created, so we reset the rendering cache.
        self.render_cache = None

        if self.cfg.rendering.use_denoiser:
            self.denoiser = mi.OptixDenoiser(
                input_size=self.cfg.rendering.rendering_resolution,
                albedo=True,
                normals=True,
                temporal=False,
            )
        else:
            self.denoiser = None
        self.rendering_reset_accumulation()

    def rendering_reset_accumulation(self):
        renderer=getattr(self,'_display_renderer',None)
        if renderer is not None:renderer.invalidate()
        self.rendering_accumulated_samples = 0
        if self.cfg.demo_mode:
            # A worker may still hold these tensors; never zero them in place.
            self.ray_traced_img=None
            self.ray_traced_depth=None
            return
        if self.ray_traced_img is not None:
            self.ray_traced_img[:] = 0
            self.ray_traced_depth[:] = 0

    def clear_ray_traced_image(self, preserve_cache=False):
        import polyscope_bindings as psb

        renderer=getattr(self,'_display_renderer',None)
        if renderer is not None:
            renderer.invalidate()
            if not preserve_cache:renderer.quantity=None
        if not preserve_cache:self.render_cache = None
        self.ray_traced_img = None
        self.ray_traced_depth = None
        self.rendering_accumulated_samples = 0
        # Reuse the disabled GL image during gestures instead of reallocating
        # a full-size image after every release.
        if not (preserve_cache and renderer is not None and renderer.quantity is not None):
            psb.get_global_floating_quantity_structure().remove_quantity(
                "ray_traced_img", errorIfAbsent=False
            )

    def render_pixel_resolution(self, window_resolution: tuple[int, int]):
        if self.cfg.demo_mode:
            # Polyscope's window dimensions already include DPI scaling. Dividing
            # by UI scale blurred Original when expanded over the native Fast
            # framebuffer (e.g. 1066x666 -> 1600x1000 at 150% scaling).
            # Bound static render memory on large monitors; interaction stays
            # on the native raster preview and never pays for these extra rays.
            w,h=window_resolution
            ratio=min(1.,2560/w,1440/h)
            return (max(1,int(w*ratio)),max(1,int(h*ratio)))
        return tuple(max(1,int(v/self.ui_scale)) for v in window_resolution)

    def set_rendering_resolution(self, window_resolution: tuple[int, int]):
        self.cfg.rendering.current_resolution = self.render_pixel_resolution(window_resolution)
        self.clear_ray_traced_image()
        # Re-create denoiser for the new resolution, if appropriate
        self.set_use_denoiser(self.cfg.rendering.use_denoiser)

    # ------------------------

    def set_radio_map(self, radio_map: rt.RadioMap, show: bool = False):
        if radio_map is not None and self.cfg.demo_mode:
            tx=next(iter(self.scene.transmitters.values()))
            radio_map.beam_power_dbm=float(tx.power_dbm.numpy()[0])
            radio_map.beam_index=-1 if getattr(radio_map,'coverage_mode','current')=='max' else self.beam_demo.selected
        if (radio_map is not None and self.cfg.demo_mode
                and getattr(self, 'demo_scene_data', None) is not None
                and self.demo_scene_data.get('source') == 'landsd_stylized'
                and not hasattr(radio_map,'demo_outdoor_mask')):
            mark_outdoor_cells(self.scene, radio_map)
        self.radio_map = radio_map
        self.rm_accumulated_samples = self._last_rm_batch_samples if radio_map is not None else 0
        self.rm_refining = self.cfg.radio_map.auto_update
        self.update_radio_map_colorbar()

        if show:
            add_radio_map_to_polyscope(
                "radio_map",
                self.radio_map,
                self.ps_groups,
                self.cfg.radio_map,
                direct_update_from_device=self.cfg.radio_map.use_direct_update_from_device,
                use_alpha=self.cfg.radio_map.use_alpha,
            )

    def update_radio_map_colorbar(self):
        key=(self.cfg.radio_map.color_map,self.cfg.radio_map.vmin,self.cfg.radio_map.vmax,self.cfg.demo_mode)
        if getattr(self,'_rm_colorbar_key',None)==key and self.rm_colorbar_texture_id is not None:return
        self._rm_colorbar_key=key
        # Draw the colorbar to an array.
        self.rm_colorbar = radio_map_colorbar_to_image(
            self.cfg.radio_map.color_map,
            self.cfg.radio_map.vmin,
            self.cfg.radio_map.vmax,
            unit="dBm" if self.cfg.demo_mode else "dB",
        )
        # Note: we upload the colorbar to a texture, even if it's not shown right now.
        ps.add_color_alpha_image_quantity(
            "rm_colorbar",
            values=self.rm_colorbar,
            enabled=True,
            image_origin="upper_left",
            show_fullscreen=False,
            # show_in_camera_billboard=True,
        )
        self.rm_colorbar_texture_id = ps.get_quantity_buffer(
            "rm_colorbar", "colors"
        ).get_texture_native_id()

    def start_radio_map(self):
        if self.cfg.demo_mode:
            self.request_coverage()
        else:
            self.set_radio_map(self.compute_radio_map(), show=True)

    def request_coverage(self):
        self.studio_maps_available=True
        self.studio_maps_scene=id(self.scene)
        self.studio_maps_collapsed=False
        if self.cfg.demo_mode:
            self.cfg.radio_map.accumulate_max_samples_per_tx=8000000
            self.cfg.radio_map.air_show_in_3d=True
            self.cfg.radio_map.air_show_stack=True
        if not self.scene.transmitters:
            self.demo_status = 'Add a transmitter first.'
            return
        self.clear_radio_map()
        self.beam_demo.map_requested = True
        self.beam_demo.map_due = None
        from .coverage_jobs import snapshot
        self.beam_demo.map_signature = snapshot(self)
        if self.coverage_jobs is None:
            self.coverage_jobs = CoverageJobs(self)
        self.rm_refining = True
        self.coverage_jobs.submit(self)

    def pause_coverage(self):
        self.rm_refining = False
        self.beam_demo.map_due = None
        self.beam_demo.map_requested = False
        if self.coverage_jobs is not None:
            self.coverage_jobs.cancel()

    @panel_solver
    def compute_radio_map(self) -> rt.RadioMap | None:
        if not self.scene._transmitters:
            return None

        solver = rt.RadioMapSolver()
        samples_per_tx = self.cfg.radio_map.samples_per_it // len(
            self.scene._transmitters
        )
        self._last_rm_batch_samples = samples_per_tx
        if self.cfg.demo_mode:
            from .terrain_coverage import build_surface,solve_surface,display_map
            from .coverage_jobs import snapshot
            settings=snapshot(self)
            surface=build_surface(self.scene,settings)
            values=solve_surface(solver,self.scene,surface,settings,self.beam_demo.precoding(self.scene),samples_per_tx,self.frame_i)
            return display_map(self.scene,self.cfg.radio_map,surface[1],surface[2],values)
        return solver(
            self.scene,
            seed=self.frame_i,
            center=self.cfg.radio_map.center,
            orientation=self.cfg.radio_map.orientation,
            size=self.cfg.radio_map.size,
            cell_size=self.cfg.radio_map.cell_size,
            measurement_surface=self.cfg.radio_map.measurement_surface,
            precoding_vec=self.beam_demo.precoding(self.scene) if self.cfg.demo_mode else None,
            samples_per_tx=samples_per_tx,
            max_depth=self.cfg.radio_map.max_depth,
            los=self.cfg.radio_map.los,
            specular_reflection=self.cfg.radio_map.specular_reflection,
            diffuse_reflection=self.cfg.radio_map.diffuse_reflection,
            refraction=self.cfg.radio_map.refraction,
            diffraction=self.cfg.radio_map.diffraction,
            edge_diffraction=self.cfg.radio_map.edge_diffraction,
            diffraction_lit_region=self.cfg.radio_map.diffraction_lit_region,
        )

    def has_visible_radio_map(self) -> tuple[bool, ps.SurfaceMesh]:
        rm_struct = None
        if ps.has_surface_mesh("radio_map"):
            rm_struct = ps.get_surface_mesh("radio_map")
            rm_visible = rm_struct.is_enabled()
        else:
            rm_visible = False

        return rm_visible, rm_struct

    # ------------------------

    def update_paths(self, clear_first: bool = False, show: bool = True):
        if self.cfg.demo_mode and self.demo_page == 0:
            return
        if self.cfg.demo_mode:
            from .link_jobs import LinkJobs
            if self.link_jobs is None:self.link_jobs=LinkJobs(self)
            self.link_jobs.request(self)
            return
        # Optionally throttle path computations to reduce load
        current_time = time.time()
        time_since_last_update = current_time - self._last_paths_update_time
        if time_since_last_update < self.cfg.paths.min_update_delay_s:
            # Skip this update
            return

        if clear_first:
            self.clear_paths()

        self.paths = self.compute_paths()
        self._last_paths_update_time = current_time
        if self.paths is None:
            if self.cfg.demo_mode:
                self.paths_cir = None
                self.update_demo_cir()
            return

        if show:
            add_paths_to_polyscope(self, self.paths, self.ps_groups)

        if self.cfg.paths.compute_cir or self.cfg.demo_mode:
            self.paths_taps = self.paths.taps(
                bandwidth=self.cfg.paths.bandwidth,
                l_min=self.cfg.paths.l_min,
                l_max=self.cfg.paths.l_max,
                sampling_frequency=self.cfg.paths.sampling_frequency,
                num_time_steps=self.cfg.paths.num_time_steps,
                normalize=self.cfg.paths.normalize,
                normalize_delays=self.cfg.paths.normalize_delays,
                out_type="numpy",
            )
            self.paths_cir = self.paths.cir(normalize_delays=True, out_type="numpy")

            # If self.paths_cir[0] has no elements, replace self.paths_taps with zeros, first element = 1e-10
            if np.sum(self.paths_cir[0]) == 0:
                # Set the first element to 1e-10
                self.paths_taps[0, 0, 0, 0, 0, 0] = 1e-10
            if self.cfg.demo_mode:
                self.update_demo_cir()

    @panel_solver
    def compute_paths(self) -> rt.Paths | None:
        if not self.scene._transmitters or not self.scene._receivers:
            return None

        solver = rt.PathSolver()
        return solver(
            self.scene,
            max_depth=self.cfg.paths.max_depth,
            max_num_paths_per_src=self.cfg.paths.max_num_paths_per_src,
            samples_per_src=self.cfg.paths.samples_per_src,
            synthetic_array=self.cfg.paths.synthetic_array,
            los=self.cfg.paths.los,
            specular_reflection=self.cfg.paths.specular_reflection,
            diffuse_reflection=self.cfg.paths.diffuse_reflection,
            refraction=self.cfg.paths.refraction,
            diffraction=self.cfg.paths.diffraction,
            edge_diffraction=self.cfg.paths.edge_diffraction,
            diffraction_lit_region=self.cfg.paths.diffraction_lit_region,
            seed=12345,
        )

    # ------------------------

    def add_radio_device(
        self,
        position: list[float],
        is_transmitter: bool,
        allow_auto_update: bool = True,
    ) -> rt.RadioDevice:
        # TODO: controllable offset to the clicked surface (along normal?)
        existing_rd = (
            self.scene._transmitters if is_transmitter else self.scene._receivers
        )

        # Add actual radio device to Sionna scene
        prefix = "tx" if is_transmitter else "rx"
        free_index = len(existing_rd)
        while f"{prefix}-{free_index}" in existing_rd:
            free_index += 1

        new_rd = (rt.Transmitter if is_transmitter else rt.Receiver)(
            name=f"{prefix}-{free_index}",
            position=position,
            orientation=[0, 0, 0],
        )
        self.scene.add(new_rd)

        set_or_update_radio_devices_polyscope(
            existing_rd,
            is_transmitter,
            self,
        )

        if (
            allow_auto_update
            and self.cfg.radio_map.auto_update
            and is_transmitter
            and (self.radio_map is not None)
        ):
            self.start_radio_map()
        if allow_auto_update and self.cfg.paths.auto_update:
            self.update_paths(show=True)

        return new_rd

    def remove_object(
        self, object: rt.SceneObject, selected_type: SelectionType
    ) -> None:
        match selected_type:
            case SelectionType.Transmitter:
                del self.scene._transmitters[object.name]
            case SelectionType.Receiver:
                del self.scene._receivers[object.name]
            case _:
                print(f"[!] Unexpected selection type: {selected_type}")
                pass

        if object.name in self.animation_config.trajectories:
            del self.animation_config.trajectories[object.name]

    def clear_radio_devices(self) -> None:
        for name in self.scene._transmitters.keys():
            if name in self.animation_config.trajectories:
                del self.animation_config.trajectories[name]
        for name in self.scene._receivers.keys():
            if name in self.animation_config.trajectories:
                del self.animation_config.trajectories[name]

        self.scene._transmitters.clear()
        self.scene._receivers.clear()
        for name in self.ps_groups["rd"].get_child_structure_names():
            ps.get_point_cloud(name).remove()
        if self.selected_type in (SelectionType.Transmitter, SelectionType.Receiver):
            self.clear_selection()

        if self.cfg.radio_map.auto_update:
            self.clear_radio_map()
        if self.cfg.paths.auto_update:
            self.clear_paths()

    def reset_radio_map(self):
        """
        Attempts to reset the radio map to zero, as well as the accumulation counter.
        However, if the number of transmitters has changed, the radio map will be
        completely removed (and re-computed if auto-updates are enabled).
        """
        self.rm_accumulated_samples = 0
        if self.radio_map is None:
            return

        if len(self.scene._transmitters) != self.radio_map.num_tx:
            self.clear_radio_map()
            if self.cfg.radio_map.auto_update:
                self.start_radio_map()
            return

        self.radio_map._pathgain_map *= 0.0
        self.rm_refining = self.cfg.radio_map.auto_update

    def clear_radio_map(self):
        from .air_coverage import clear_stack
        clear_stack()
        if self.coverage_jobs is not None:
            self.coverage_jobs.cancel()
        self.radio_map = None
        self.rm_refining = False
        self.rm_accumulated_samples = 0
        self.rm_colorbar = None
        self.rm_colorbar_texture_id = None
        if ps.has_surface_mesh("radio_map"):
            ps.get_surface_mesh("radio_map").remove()

    def clear_paths(self):
        self.demo_link_cir = None
        self.demo_link_settings = None
        self.paths = None
        self.paths_taps = None
        self.paths_cir = None
        self.demo_path_powers=np.array([]);self.demo_path_delays=np.array([])
        self.demo_chart_dirty=True
        if ps.has_curve_network("paths"):
            ps.get_curve_network("paths").remove()

    # ------------------------

    def set_slice_plane_active(self, active: bool):
        if self.slice_plane is None:
            return
        self.slice_plane.set_active(active)

        if active and self.cfg.rendering.mode == RenderingMode.RAY_TRACING:
            # Switch to rasterization mode when activating the slice plane.
            self.set_rendering_mode(RenderingMode.RASTERIZATION)

    # ------------------------

    def process_inputs(self):
        imgui_io = psim.GetIO()
        if self.cfg.demo_mode:
            if psim.IsKeyPressed(psim.ImGuiKey_Escape, repeat=False):
                from .roadshow_ui import set_operation
                set_operation(self, 'browse')
                self.demo_status = 'Explore: drag to pan, right-drag to orbit, double-click to focus.'
            if (self.demo_page==1 and not psim.IsAnyItemActive() and
                    psim.IsKeyPressed(psim.ImGuiKey_Space, repeat=False)):
                traj=self.animation_config.trajectories['rx-0']
                if len(traj)>1:
                    from .roadshow_ui import set_operation
                    playing=not self.animation_config.playing
                    set_operation(self,'browse')
                    if playing:self.move_demo_receiver(traj.current_position_and_direction()[0])
                    traj.enabled=True;self.animation_config.playing=playing
            from .navigation import process
            process(self)
        if imgui_io.WantCaptureKeyboard:
            # If the user is e.g. typing in a text field, don't process
            # keyboard or mouse inputs.
            return

        allow_click = not imgui_io.WantCaptureMouse
        has_left_click = allow_click and psim.IsMouseClicked(psim.ImGuiMouseButton_Left)
        has_mouse_drag = psim.IsMouseDragging(
            psim.ImGuiMouseButton_Left
        ) or psim.IsMouseDragging(psim.ImGuiMouseButton_Right)
        has_left_release = allow_click and psim.IsMouseReleased(
            psim.ImGuiMouseButton_Left
        )
        if self.cfg.demo_mode:
            if psim.IsMouseClicked(0):self._demo_click_owned=allow_click;self._demo_click_dragged=False
            if has_mouse_drag:self._demo_click_dragged=True
            has_left_release=has_left_release and getattr(self,'_demo_click_owned',False) and not getattr(self,'_demo_click_dragged',False)
        has_right_click = allow_click and psim.IsMouseClicked(
            psim.ImGuiMouseButton_Right
        )
        has_active_item = psim.IsAnyItemActive()

        # TODO: +/- to zoom in/out
        # TODO: keyboard shortcuts to move around (WASD + QE)
        # TODO: keyboard shortcuts to rotate the envmap?

        # K/L or (Ctrl + left/right click): add transmitter/receiver
        has_k = psim.IsKeyPressed(psim.ImGuiKey(ps.get_key_code("K")))
        has_l = psim.IsKeyPressed(psim.ImGuiKey(ps.get_key_code("L")))
        if not self.cfg.demo_mode and (
            has_k
            or has_l
            or (
                imgui_io.KeyCtrl
                and (has_left_click or has_right_click)
                and not imgui_io.KeyShift
            )
        ):
            is_transmitter = imgui_io.MouseClicked[0] or has_k

            origin = ps.get_camera_view_matrix()[:3, 3]
            rd_position = ps.screen_coords_to_world_position(imgui_io.MousePos)

            if np.all(np.isfinite(rd_position)):
                normal = get_normal_for_path(self.scene, origin, rd_position)
                if normal is None:
                    normal = np.array([0, 0, 1])
                # Make sure that normal is oriented towards the camera
                if np.dot(normal, rd_position - origin) < 0:
                    normal = -normal
                rd_position += 1.5 * normal

                self.add_radio_device(rd_position, is_transmitter)

        # Plain left click: object selection
        if has_left_release and not (
            imgui_io.KeyCtrl
            or imgui_io.KeyShift
            or imgui_io.KeyAlt
            or has_mouse_drag
            or self.was_mouse_dragging
        ):
            if self.cfg.demo_mode and self.demo_place_transmitter:
                from .demo_workflow import place_tx_at_screen
                place_tx_at_screen(self, imgui_io.MousePos)
            elif self.cfg.demo_mode and self.demo_aim_beam:
                from .spatial_beam import aim_at_screen
                aim_at_screen(self,imgui_io.MousePos)
            elif self.cfg.demo_mode and self.demo_place_receiver:
                self.place_demo_receiver_at_screen_point(imgui_io.MousePos)
            else:
                pick_result=ps.pick(screen_coords=imgui_io.MousePos)
                if not (self.cfg.demo_mode and hasattr(self,'beam_demo') and self.beam_demo.pick(pick_result)):
                    self.process_pick_result(pick_result)

        # Shift + R: reload code
        if imgui_io.KeyShift and psim.IsKeyPressed(
            psim.ImGuiKey(ps.get_key_code("R")), repeat=False
        ):
            self.code_reload_requested = True

        if self.fullscreen_window is not None and psim.IsKeyPressed(psim.ImGuiKey_F11, repeat=False):
            self.fullscreen_window.toggle()
        # R: reset camera to initial position
        if psim.IsKeyPressed(psim.ImGuiKey(ps.get_key_code("R")), repeat=False):
            self.move_camera_home()
        # F: fit scene in camera viewport
        if psim.IsKeyPressed(psim.ImGuiKey(ps.get_key_code("F")), repeat=False):
            self.fit_camera_to_scene()

        # C: go to next rendering mode.
        if psim.IsKeyPressed(psim.ImGuiKey(ps.get_key_code("C")), repeat=False):
            self.set_rendering_mode(
                RenderingMode((self.cfg.rendering.mode.value + 1) % len(RenderingMode))
            )

        # H / ?: show help window
        if psim.IsKeyPressed(psim.ImGuiKey(ps.get_key_code("H")), repeat=False) or (
            imgui_io.KeyShift
            and psim.IsKeyPressed(psim.ImGuiKey(ps.get_key_code("/")), repeat=False)
        ):
            self.cfg.show_help_window = not self.cfg.show_help_window

        # M: toggle radio map computation and display
        if (not self.cfg.demo_mode or self.demo_page == 0) and psim.IsKeyPressed(psim.ImGuiKey(ps.get_key_code("M")), repeat=False):
            rm_visible, rm_struct = self.has_visible_radio_map()

            if rm_visible:
                # Hide radio map (but don't delete it, we may want to show it later)
                self.cfg.radio_map.auto_update = False
                rm_struct.set_enabled(False)
                self.pause_coverage()
            else:
                self.cfg.radio_map.auto_update = True
                if self.radio_map is not None:
                    # Show the existing radio map
                    rm_struct.set_enabled(True)
                else:
                    # Compute and show the radio map
                    self.start_radio_map()

        if self.slice_plane is not None:
            # S: toggle slice plane
            if psim.IsKeyPressed(psim.ImGuiKey(ps.get_key_code("S")), repeat=False):
                self.set_slice_plane_active(not self.slice_plane.get_active())

            # Alt + left click drag: move slice plane along its normal.
            # TODO: avoid using middle click, since it's not easy to do on trackpads.
            if self.slice_plane.get_active() and imgui_io.KeyAlt and has_mouse_drag:
                ps.set_do_default_mouse_interaction(False)
                center = self.slice_plane.get_center()
                normal = self.slice_plane.get_normal()
                self.slice_plane.set_pose(
                    center + 0.3 * self.ui_scale * imgui_io.MouseDelta[1] * normal,
                    normal,
                )
            else:
                ps.set_do_default_mouse_interaction(not self.cfg.demo_mode)

        # Tab: toggle show GUI (ours)
        if not has_active_item and psim.IsKeyPressed(psim.ImGuiKey_Tab, repeat=False):
            self.cfg.gui_mode = GuiMode((self.cfg.gui_mode.value + 1) % len(GuiMode))

        # Esc: close help window or de-select
        if psim.IsKeyPressed(psim.ImGuiKey_Escape, repeat=False):
            if self.cfg.show_help_window:
                self.cfg.show_help_window = False
            elif self.selected_object is not None:
                self.clear_selection()

        # Ctrl + Q: exit
        if imgui_io.KeyCtrl and psim.IsKeyPressed(psim.ImGuiKey(ps.get_key_code("Q"))):
            ps.unshow()

        self.was_mouse_dragging = has_mouse_drag

    def process_pick_result(self, pick_result: ps.PickResult) -> bool:
        if not pick_result.is_hit or "index" not in pick_result.structure_data:
            self.clear_selection()
            return False

        picked_index = pick_result.structure_data["index"]
        if pick_result.structure_name in "Transmitters":
            if self.cfg.demo_mode:
                self.clear_selection()
                return False
            self.selected_object = list(self.scene._transmitters.values())[picked_index]
            self.selected_type = SelectionType.Transmitter
            return True
        elif pick_result.structure_name in "Receivers":
            self.selected_object = list(self.scene._receivers.values())[picked_index]
            self.selected_type = SelectionType.Receiver
            return True

        self.clear_selection()
        return False

    def clear_selection(self):
        self.selected_object = None
        self.selected_type = None
        if ps.has_point_cloud("Gizmo"):
            ps.get_point_cloud("Gizmo").remove()
        if ps.has_curve_network("Trajectory"):
            ps.get_curve_network("Trajectory").remove()

    def gui(self):
        if not self.cfg.demo_mode:
            return self._gui_contents()
        psim.GetStyle().FontScaleMain=1.
        font_size=18 if ps.get_window_size()[1]<900 else 30 if ps.get_window_size()[1]>1500 else 22
        psim.PushFont(getattr(self,'roadshow_fonts',{}).get(font_size),float(font_size))
        try:
            self.demo_font_pixels = psim.GetFontSize()
            return self._gui_contents()
        finally:
            psim.PopFont()

    def _gui_contents(self):
        if self.cfg.demo_mode:
            from .roadshow_ui import draw
            draw(self)
            return
        # TODO: change GUI accent color to a non-default color.

        if self.cfg.gui_mode == GuiMode.HIDDEN:
            return

        # --- Selection window
        if self.selected_object is not None:
            selection_gui(self, self.selected_object, self.selected_type)

        # --- Help window
        if self.cfg.show_help_window:
            self.gui_help_window()

        # --- Colorbar window
        if (
            self.has_visible_radio_map()[0]
            and self.cfg.radio_map.show_colorbar
            and (self.rm_colorbar is not None)
            and (self.rm_colorbar_texture_id is not None)
            and hasattr(psim, "Image")
        ):
            window_resolution = ps.get_window_size()
            h, w = self.rm_colorbar.shape[:2]
            psim.SetNextWindowSize((w * self.ui_scale, h * self.ui_scale))
            psim.SetNextWindowPos(
                (0.5 * (window_resolution[0] - w * self.ui_scale), 5 * self.ui_scale)
            )
            psim.Begin(
                "Colorbar",
                open=True,
                flags=(
                    psim.ImGuiWindowFlags_NoTitleBar
                    | psim.ImGuiWindowFlags_NoDecoration
                    | psim.ImGuiWindowFlags_NoBackground
                ),
            )
            psim.SetCursorPosX(0)
            psim.SetCursorPosY(0)
            psim.Image(
                psim.ImTextureRef(self.rm_colorbar_texture_id),
                (self.rm_colorbar.shape[1], self.rm_colorbar.shape[0]),
            )
            psim.End()

        if (
            self.cfg.demo_mode
            and ps.has_surface_mesh(LAYER_NAME)
            and ps.get_surface_mesh(LAYER_NAME).is_enabled()
            and self.demo_colorbar_texture_id is not None
            and hasattr(psim, "Image")
        ):
            w = 520 * self.ui_scale
            h = 45 * self.ui_scale
            window_resolution = ps.get_window_size()
            psim.SetNextWindowSize((w, h))
            psim.SetNextWindowPos((0.5 * (window_resolution[0] - w), 12 * self.ui_scale))
            psim.Begin(
                "Illustrative coverage legend##demo", open=True,
                flags=(psim.ImGuiWindowFlags_NoTitleBar
                       | psim.ImGuiWindowFlags_NoDecoration
                       | psim.ImGuiWindowFlags_NoBackground),
            )
            psim.Image(psim.ImTextureRef(self.demo_colorbar_texture_id), (w, h))
            psim.End()

        if self.cfg.demo_mode:
            draw_chart(self)
            draw_device_labels(self)

        # --- Main GUI window
        from .desktop_window import layout
        panel_width,panel_height,_,_=layout(self)
        psim.SetNextWindowSize(
            ((panel_width if self.cfg.demo_mode else 430*self.ui_scale),
             (panel_height if self.cfg.demo_mode else 800*self.ui_scale)),
            psim.ImGuiCond_Always if self.cfg.demo_mode else psim.ImGuiCond_FirstUseEver,
        )
        psim.SetNextWindowPos(
            (10 * self.ui_scale, 10 * self.ui_scale), psim.ImGuiCond_FirstUseEver
        )
        psim.Begin("Sionna RT##sionna", open=True)

        if self.cfg.demo_mode:
            draw_controls(self)
            if not self.demo_advanced:
                psim.End()
                return

        psim.Text(f"Frame time: {1000 * psim.GetIO().DeltaTime:.2f} ms")

        psim.SameLine()
        bw = 20
        psim.SetCursorPosX(
            (psim.GetCursorPosX() + psim.GetContentRegionAvail()[0] - bw)
            * self.ui_scale
        )
        if psim.Button("?", size=(bw * self.ui_scale, 0)):
            self.cfg.show_help_window = not self.cfg.show_help_window

        if psim.CollapsingHeader(
            "Scene", 0 if self.cfg.demo_mode else psim.ImGuiTreeNodeFlags_DefaultOpen
        ):
            psim.Spacing()

            # Quick pick from built-in or recently-loaded scenes.
            psim.Text("Scene selection:")
            changed, combo_i = psim.Combo(
                "##scene_picker",
                self.current_scene_idx,
                self.known_scene_names,
            )
            if changed:
                self.load_scene_requested = self.known_scene_paths[combo_i]

            psim.Spacing()

        if psim.CollapsingHeader(
            "Radio devices", 0 if self.cfg.demo_mode else psim.ImGuiTreeNodeFlags_DefaultOpen
        ):
            psim.Spacing()
            # TODO: button to place radio devices: at random; or samples from a radio map
            antenna_array_gui(self)

            psim.BeginDisabled(self.cfg.demo_mode)
            clicked = psim.Button("Clear all radio devices")
            psim.EndDisabled()
            if clicked:
                self.clear_radio_devices()

            psim.Spacing()

        if psim.CollapsingHeader(
            "Radio map", 0 if self.cfg.demo_mode else psim.ImGuiTreeNodeFlags_DefaultOpen
        ):
            psim.Spacing()
            needs_update = False
            needs_visual_update = False

            if psim.Button("Compute radio map"):
                self.start_radio_map()

            psim.SameLine()
            psim.BeginDisabled(self.radio_map is None)
            if psim.Button("Remove##radio_map"):
                self.clear_radio_map()
            psim.EndDisabled()

            psim.SameLine()
            changed_auto, self.cfg.radio_map.auto_update = psim.Checkbox(
                "Automatic update##rm", self.cfg.radio_map.auto_update
            )
            if changed_auto:
                if not self.cfg.radio_map.auto_update:
                    self.pause_coverage()
                elif self.cfg.demo_mode and self.coverage_jobs is None:
                    self.request_coverage()
                else:
                    self.rm_refining = True

            # -- Radio map computation options
            psim.Spacing()

            changed, self.cfg.radio_map.cell_size = psim.InputFloat2(
                "Cell size",
                self.cfg.radio_map.cell_size,
                # v_min=0.1,
                # v_max=100,
                format="%.2f",
            )
            if changed:
                self.cfg.radio_map.cell_size = tuple(
                    min(max(v, 0.01), 100) for v in self.cfg.radio_map.cell_size
                )
            needs_update |= changed

            changed, self.cfg.radio_map.log_samples_per_it = psim.SliderFloat(
                "Samples / it (log 10)",
                self.cfg.radio_map.log_samples_per_it,
                v_min=0,
                v_max=9,
                format="10^%.1f",
            )
            needs_update |= changed

            changed, self.cfg.radio_map.max_depth = psim.SliderInt(
                "Max depth##rm",
                self.cfg.radio_map.max_depth,
                v_min=1,
                v_max=10,
            )
            needs_update |= changed

            # -- Checkboxes table
            needs_update |= self._gui_features_checkboxes(self.cfg.radio_map, "##rm")

            # -- Radio map display options
            if self.radio_map is not None:
                psim.Spacing()

                psim.Text("Accumulating samples:")

                psim.PushStyleColor(psim.ImGuiCol_PlotHistogram, NVIDIA_GREEN)
                psim.ProgressBar(
                    min(
                        self.rm_accumulated_samples
                        / self.cfg.radio_map.accumulate_max_samples_per_tx,
                        1.0,
                    ),
                    (psim.CalcItemWidth(), 0),
                )
                psim.PopStyleColor()

                struct = ps.get_surface_mesh("radio_map")
                changed, show_rm = psim.Checkbox("Show radio map", struct.is_enabled())
                if changed:
                    struct.set_enabled(show_rm)

                psim.SameLine()
                _, self.cfg.radio_map.show_colorbar = psim.Checkbox(
                    "Show color bar", self.cfg.radio_map.show_colorbar
                )

                changed_cmap, self.rm_color_map_index = psim.Combo(
                    "Colormap",
                    self.rm_color_map_index,
                    self.rm_color_map_options,
                )
                if changed_cmap:
                    self.cfg.radio_map.color_map = self.rm_color_map_options[
                        self.rm_color_map_index
                    ]

                changed_vmin, self.cfg.radio_map.vmin = psim.SliderFloat(
                    "vmin",
                    self.cfg.radio_map.vmin,
                    v_min=-200,
                    v_max=0,
                )
                changed_vmax, self.cfg.radio_map.vmax = psim.SliderFloat(
                    "vmax",
                    self.cfg.radio_map.vmax,
                    v_min=-200,
                    v_max=0,
                )
                self.cfg.radio_map.vmin = min(
                    self.cfg.radio_map.vmin, self.cfg.radio_map.vmax
                )
                self.cfg.radio_map.vmax = max(
                    self.cfg.radio_map.vmin, self.cfg.radio_map.vmax
                )

                needs_visual_update = changed_cmap or changed_vmin or changed_vmax

            if self.cfg.radio_map.auto_update and needs_update:
                self.start_radio_map()
            if needs_update or needs_visual_update:
                self.update_radio_map_colorbar()
                add_radio_map_to_polyscope(
                    "radio_map",
                    self.radio_map,
                    self.ps_groups,
                    self.cfg.radio_map,
                    direct_update_from_device=self.cfg.radio_map.use_direct_update_from_device,
                    use_alpha=self.cfg.radio_map.use_alpha,
                )

            # TODO: plane / mesh picker (changes type of radio map)
            # TODO: some way to move the radio map (at least the plane z offset)

            psim.Spacing()

        if psim.CollapsingHeader("Paths", 0 if self.cfg.demo_mode else psim.ImGuiTreeNodeFlags_DefaultOpen):
            psim.Spacing()
            needs_update = False

            clicked = psim.Button("Compute paths")
            if clicked:
                self.update_paths(show=True)

            psim.SameLine()
            psim.BeginDisabled(self.paths is None)
            if psim.Button("Remove##paths"):
                self.clear_paths()
            psim.EndDisabled()

            psim.SameLine()
            _, self.cfg.paths.auto_update = psim.Checkbox(
                "Automatic update##paths", self.cfg.paths.auto_update
            )

            changed, self.cfg.paths.max_depth = psim.SliderInt(
                "Max depth##paths",
                self.cfg.paths.max_depth,
                v_min=1,
                v_max=10,
            )
            needs_update |= changed

            psim.SetNextItemWidth(200 * self.ui_scale)
            changed, self.cfg.paths.synthetic_array = psim.Checkbox(
                "Synthetic array##paths", self.cfg.paths.synthetic_array
            )
            needs_update |= changed

            needs_update |= self._gui_features_checkboxes(self.cfg.paths, "##paths")

            # TODO: rendering parameters (radius, transparency, etc)
            # TODO: show & configure segment color per type of interaction
            needs_visual_update = False

            if self.cfg.paths.auto_update and needs_update:
                self.update_paths(show=False)
            if needs_update or needs_visual_update:
                add_paths_to_polyscope(self, self.paths, self.ps_groups)

            psim.Spacing()

        if psim.CollapsingHeader("Animation"):
            animation_gui(self)
            psim.Spacing()

        if psim.CollapsingHeader("Rendering"):
            psim.Spacing()

            changed, combo_i = psim.Combo(
                "Rendering mode",
                self.cfg.rendering.mode.value,
                RENDERING_MODE_NAMES,
            )
            if changed:
                self.cfg.rendering.mode = RenderingMode(combo_i)
                self.set_rendering_mode(self.cfg.rendering.mode)

            if (self.cfg.rendering.mode == RenderingMode.RAY_TRACING) and (
                "cuda" in mi.variant()
            ):
                _, self.cfg.rendering.spp_per_frame = psim.SliderInt(
                    "SPP / frame",
                    self.cfg.rendering.spp_per_frame,
                    v_min=1,
                    v_max=1024,
                )
                self.cfg.rendering.spp_per_frame = max(
                    self.cfg.rendering.spp_per_frame, 1
                )
                _, self.cfg.rendering.max_accumulated_spp = psim.SliderInt(
                    "SPP max",
                    self.cfg.rendering.max_accumulated_spp,
                    v_min=1,
                    v_max=1024,
                )
                self.cfg.rendering.max_accumulated_spp = max(
                    self.cfg.rendering.max_accumulated_spp, 1
                )

                changed, self.cfg.rendering.relative_resolution = psim.SliderFloat(
                    "Rel. resolution",
                    self.cfg.rendering.relative_resolution,
                    v_min=0.1,
                    v_max=1.0,
                    format="%.2f",
                )
                self.cfg.rendering.relative_resolution = min(
                    max(self.cfg.rendering.relative_resolution, 0.1), 1.0
                )
                if changed:
                    self.clear_ray_traced_image()
                    # Need to re-create the denoiser for the new resolution
                    self.set_use_denoiser(self.cfg.rendering.use_denoiser)

                changed, self.cfg.rendering.envmap_rotation_deg = psim.SliderFloat(
                    "Lighting angle",
                    self.cfg.rendering.envmap_rotation_deg,
                    v_min=-180,
                    v_max=180,
                    format="%.0f",
                )
                self.cfg.rendering.envmap_rotation_deg = min(
                    max(self.cfg.rendering.envmap_rotation_deg, -180), 180
                )
                if changed:
                    set_envmap_rotation(
                        self.render_cache, self.cfg.rendering.envmap_rotation_deg
                    )
                    self.reset_accumulation_requested = True

                changed, self.cfg.rendering.use_denoiser = psim.Checkbox(
                    "Use OptiX denoiser", self.cfg.rendering.use_denoiser
                )
                if changed:
                    self.set_use_denoiser(self.cfg.rendering.use_denoiser)

            changed, self.cfg.use_vsync = psim.Checkbox("VSync", self.cfg.use_vsync)
            if changed:
                ps.set_enable_vsync(self.cfg.use_vsync)

            psim.SameLine()

            self.cfg.background_color = ps.get_background_color()
            changed, self.cfg.background_color = psim.ColorEdit4(
                "Background",
                self.cfg.background_color,
                psim.ImGuiColorEditFlags_NoInputs,
            )
            if changed:
                ps.set_background_color(self.cfg.background_color)

            changed, self.cfg.show_polyscope_gui = psim.Checkbox(
                "Show Polyscope UI", self.cfg.show_polyscope_gui
            )
            if changed:
                ps.set_build_default_gui_panels(self.cfg.show_polyscope_gui)

            psim.Spacing()

            if self.slice_plane is not None:
                psim.SeparatorText("Slice plane")
                changed, plane_active = psim.Checkbox(
                    "Active", self.slice_plane.get_active()
                )
                if changed:
                    self.set_slice_plane_active(plane_active)

                if plane_active:
                    psim.SameLine()
                    changed, draw_plane = psim.Checkbox(
                        "Show plane", self.slice_plane.get_draw_plane()
                    )
                    if changed:
                        self.slice_plane.set_draw_plane(draw_plane)

                    psim.SameLine()
                    changed, gizmo_active = psim.Checkbox(
                        "Show gizmo", self.slice_plane.get_draw_widget()
                    )
                    if changed:
                        self.slice_plane.set_draw_widget(gizmo_active)

        psim.End()  # End main Sionna RT window

    def _gui_features_checkboxes(
        self, cfg: RadioMapConfig | PathsConfig, suffix: str
    ) -> bool:
        any_changed = False

        psim.Columns(2, borders=False)
        psim.SetColumnWidth(0, 165 * self.ui_scale)

        changed, cfg.los = psim.Checkbox("Line of sight" + suffix, cfg.los)
        any_changed |= changed

        changed, cfg.diffuse_reflection = psim.Checkbox(
            "Diffuse reflection" + suffix, cfg.diffuse_reflection
        )
        any_changed |= changed

        changed, cfg.diffraction = psim.Checkbox(
            "Diffraction" + suffix, cfg.diffraction
        )
        any_changed |= changed

        psim.NextColumn()

        changed, cfg.specular_reflection = psim.Checkbox(
            "Specular reflection" + suffix, cfg.specular_reflection
        )
        any_changed |= changed

        changed, cfg.refraction = psim.Checkbox("Refraction" + suffix, cfg.refraction)
        any_changed |= changed

        if cfg.diffraction:
            changed, cfg.edge_diffraction = psim.Checkbox(
                "Edge" + suffix, cfg.edge_diffraction
            )
            any_changed |= changed

            psim.SameLine()
            changed, cfg.diffraction_lit_region = psim.Checkbox(
                "Lit region" + suffix, cfg.diffraction_lit_region
            )
            any_changed |= changed

        psim.Columns(1)

        return any_changed

    def gui_help_window(self):
        window_resolution = ps.get_window_size()
        w, h = 600, 600
        psim.SetNextWindowSize(
            (w * self.ui_scale, h * self.ui_scale), psim.ImGuiCond_FirstUseEver
        )
        psim.SetNextWindowPos(
            (
                0.5 * (window_resolution[0] - w * self.ui_scale),
                0.5 * (window_resolution[1] - h * self.ui_scale),
            ),
            psim.ImGuiCond_FirstUseEver,
        )

        _, self.cfg.show_help_window = psim.Begin(
            "Help",
            open=True,
            flags=psim.ImGuiWindowFlags_Modal
            | psim.ImGuiWindowFlags_NoFocusOnAppearing,
        )

        psim.Text(
            f"Sionna RT GUI v{GUI_VERSION[0]}.{GUI_VERSION[1]}.{GUI_VERSION[2]}.\n(c) NVIDIA Corporation 2025.\n\n"
            "Uses map data from OpenStreetMap (openstreetmap.org/copyright)."
        )

        psim.NewLine()

        for title, table in HELP_WINDOW_TABLES.items():
            # Centered text
            text_pos = (
                psim.GetCursorPosX()
                + psim.GetColumnWidth(0) * 0.5
                - psim.CalcTextSize(title)[0]
            )
            psim.SetCursorPosX(text_pos * self.ui_scale)
            psim.Text(title)

            psim.Separator()
            psim.Columns(2)
            psim.SetColumnWidth(0, 220 * self.ui_scale)
            psim.Text(os.linesep.join(table.keys()))
            psim.NextColumn()
            psim.Text(os.linesep.join(table.values()))
            psim.Columns(1)
            psim.Separator()

            psim.NewLine()

        psim.Columns(1)
        psim.End()



