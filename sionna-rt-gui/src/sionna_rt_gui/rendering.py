#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
from __future__ import annotations

from typing import Any

import drjit as dr
import mitsuba as mi
import numpy as np
import polyscope as ps
import polyscope_bindings as psb
from sionna import rt
from sionna.rt.renderer import visual_scene_from_wireless_scene

from .config import RenderingConfig, RenderingMode
from .ps_utils import supports_direct_update_from_device


def setup_scene_for_rendering(
    cfg: RenderingConfig,
    scene: rt.Scene,
    use_denoiser: bool = False,
) -> dict[str, Any]:
    from mitsuba.python.util import _RenderOp

    assert (
        "_rgb" in mi.variant()
    ), "This function is supposed to be called with an RGB variant."

    render_res = cfg.rendering_resolution
    fov_y_deg = ps.get_view_camera_parameters().get_fov_vertical_deg()
    sensor = mi.load_dict(
        {
            "type": "perspective",
            "fov": fov_y_deg,
            "fov_axis": "y",
            "near_clip": 1.0,
            "far_clip": max(10000, 10 * rt.utils.render.scene_scale(scene)),
            "film": {
                "type": "hdrfilm",
                "pixel_format": "rgb",
                "width": render_res[0],
                "height": render_res[1],
                "filter": {
                    "type": "box",
                },
            },
        }
    )

    # NOTE: this assumes that the scene does *not* change between renders!
    visual_scene_dict = visual_scene_from_wireless_scene(
        scene,
        sensor=sensor,
        max_depth=8,
        envmap=cfg.envmap,
        #  clip_at: float | None = None,
        #  clip_plane_orientation: tuple[float, float, float] = (0, 0, -1),
        #  envmap: str | None = None,
        lighting_scale=cfg.envmap_factor,
        #  exclude_mesh_ids: set[str] = None
    )
    # Textured source geometry is kept separate from radio materials. This also
    # preserves UV seams when Sionna merges duplicate radio-mesh vertices.
    visuals = getattr(scene, "demo_visuals", None)
    if visuals is None and getattr(scene,'studio_style',False):
        from .scene_style import display_color
        from sionna.rt.utils import clone_mesh
        for index,mesh in enumerate(scene.mi_scene.shapes()):
            key=f'shape-{index}-{mesh.id()}'
            props=mi.Properties()
            props['bsdf']=mi.load_dict({'type':'twosided','nested':{
                'type':'diffuse','reflectance':{'type':'rgb','value':display_color(mesh)}}})
            visual_scene_dict[key]=clone_mesh(mesh,name=key,props=props)
    if visuals is not None:
        for key in list(visual_scene_dict):
            if key.startswith("shape-"):
                del visual_scene_dict[key]
        for index, asset in enumerate(visuals):
            from .district_style import is_district, terrain_texture_path
            from .presentation_style import bsdf
            texture = (terrain_texture_path(asset) if asset.get("texture") and is_district(asset)
                       else asset.get("texture"))
            reflectance = ({"type": "bitmap", "filename": texture}
                           if asset.get("texture") else
                           {"type": "rgb", "value": asset["color"]})
            visual_scene_dict[f"textured-{index}"] = {
                "type": "ply", "filename": asset["mesh"],
                "face_normals": asset.get("face_normals", False),
                "bsdf": {"type": "twosided", "nested": bsdf(asset,texture)},
            }
        # A controlled studio fill/key replaces arbitrary photographic lighting.
        # Existing brightness control scales the complete rig together.
        from .presentation_style import key_light,studio_environment
        strength=cfg.envmap_factor/.55
        visual_scene_dict["emitter"]={"type":"envmap",
            "bitmap":mi.Bitmap(studio_environment()),"scale":strength}
        visual_scene_dict["studio-key"]={"type":"directional",
            "direction":(-key_light()).tolist(),"irradiance":{
                "type":"rgb","value":(np.array([.78,.73,.65])*np.pi*.85*strength).tolist()}}
    # Envmap rotation
    envmap_base_transform = None
    if "emitter" in visual_scene_dict:
        emitter = visual_scene_dict["emitter"]
        envmap_base_transform = emitter.get("to_world", mi.ScalarTransform4f())
        emitter["to_world"] = (
            mi.ScalarTransform4f.rotate(axis=(0, 0, 1), angle=cfg.envmap_rotation_deg)
            @ envmap_base_transform
        )

    visual_scene = mi.load_dict(visual_scene_dict)

    # Hack for better sampling: zero-out the bottom half or so of the envmap,
    # since our scenes always have a ground plane.
    params = mi.traverse(visual_scene)
    studio_transforms={k:params[k] for k in ("studio-key.to_world",) if k in params}
    for key,base in studio_transforms.items():
        params[key]=mi.Transform4f(mi.ScalarTransform4f.rotate(axis=(0,0,1),angle=cfg.envmap_rotation_deg+60))@base
    if studio_transforms:params.update()
    k = "emitter.data"
    if k in params and not studio_transforms:
        # Height, width, channels
        h, *_ = dr.shape(params[k])
        params[k][int(0.6 * h) :, :, :] = 0
        params.update()

    integrator = {
        "type": "path",
        "hide_emitters": True,
        "max_depth": 5,
    }
    if use_denoiser:
        integrator = {
            "type": "aov",
            "aovs": "albedo:albedo,normals:sh_normal",
            "nested": integrator,
        }
    integrator = mi.load_dict(integrator)

    depth_integrator = mi.load_dict({"type": "depth"})

    # Helper to convert between radial and perpendicular-style depth buffers.
    # Camera parameters from the sensor
    fy = render_res[1] / (2.0 * np.tan(np.radians(fov_y_deg) / 2.0))
    fx = fy  # Assume square pixels
    cx = 0.5 * render_res[0]
    cy = 0.5 * render_res[1]
    # Create pixel coordinate grids
    i_coords, j_coords = dr.meshgrid(
        dr.arange(mi.Float, render_res[0]),
        dr.arange(mi.Float, render_res[1]),
        indexing="xy",
    )
    x_norm = (i_coords - cx) / fx
    y_norm = (j_coords - cy) / fy
    # Calculate radial distance factor
    radial_factor = mi.TensorXf(
        dr.sqrt(1.0 + dr.square(x_norm) + dr.square(y_norm)),
        shape=(render_res[1], render_res[0]),
    )

    return {
        "sensor": sensor,
        "visual_scene": visual_scene,
        # Keep the RGB constructors with their scene. Switching the global
        # variant for each mouse pick reloads Sionna and stalls the UI.
        "ray_type": mi.Ray3f,
        "point_type": mi.Point3f,
        "vector_type": mi.Vector3f,
        "integrator": integrator,
        "depth_integrator": depth_integrator,
        "render_op": _RenderOp(),
        "radial_factor": radial_factor,
        "envmap_base_transform": envmap_base_transform,
        "studio_transforms":studio_transforms,
    }


def intersect_visual(cache, origin, direction):
    hit=cache['visual_scene'].ray_intersect(cache['ray_type'](
        cache['point_type'](origin), cache['vector_type'](direction)))
    hit.t=dr.select(hit.shape.is_emitter(),dr.inf,hit.t)
    return hit


def _render_scene(
    cfg: RenderingConfig,
    seed: int,
    camera_changed: bool,
    cache: dict[str, Any],
    use_denoiser: bool = False,
    camera_pose: np.ndarray | None = None,
) -> tuple[mi.TensorXf, list[mi.TensorXf], dict[str, Any]]:
    """
    Note that even though this function does RGB rendering, we do *not*
    need to switch variant, because all the relevant objects were already
    created using the RGB variant above.
    """

    sensor = cache["sensor"]
    visual_scene = cache["visual_scene"]
    integrator = cache["integrator"]
    depth_integrator = cache["depth_integrator"]
    render_op = cache["render_op"]
    radial_factor = cache["radial_factor"]

    if camera_changed:
        # Camera to world transform
        # Workers receive a snapshot; OpenGL/UI access stays on the main thread.
        view_pose = ps.get_camera_view_matrix() if camera_pose is None else camera_pose

        ps_to_world = np.linalg.inv(view_pose)
        del view_pose
        view_rotation = np.eye(4)
        view_rotation[:3, :3] = ps_to_world[:3, :3]
        view_translation = np.eye(4)
        view_translation[:3, -1] = ps_to_world[:3, -1]

        # Conversions:
        # - World space stays the same (Z up, Y forward), so the translation is left intact.
        # - However, the local camera space needs to be converted from Polyscope's
        #   left-handed (+y up, -z forward, +x right) to Mitsuba's right-handed
        #   system (y up, -z forward, +x left).
        # TODO: do this in a single go for efficiency.
        to_world = (
            mi.ScalarTransform4f(view_translation)
            @ mi.ScalarTransform4f(
                [
                    [-1, 0, 0, 0],
                    [0, -1, 0, 0],
                    [0, 0, -1, 0],
                    [0, 0, 0, 1],
                ]
            )
            @ mi.ScalarTransform4f(view_rotation)
            @ mi.ScalarTransform4f(
                [
                    [1, 0, 0, 0],
                    [0, -1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                ]
            )
        )

        params = mi.traverse(sensor)
        params["to_world"] = to_world
        params.update()

    # Note: we can't directly use `mi.render()` because of a couple of asserts on the
    # types of integrator, sensor, etc.
    # TODO: use kernel freezing, if it helps
    img = render_op.eval(
        scene=visual_scene,
        sensor=sensor,
        _=None,
        params=None,
        integrator=integrator,
        seed=(seed, None),
        spp=(cfg.spp_per_frame, None),
    )
    rgb = img[..., :3]

    # Depth: render separately because we really want 1spp. Otherwise, depth values
    # get averaged with 0.f at the edge of objects, which is problematic for compositing.
    # A fresh random 1-spp depth per frame made silhouette/interior gaps jump
    # while RGB was accumulating. Use stable pixel-centre visibility instead.
    if camera_changed or "stable_depth" not in cache:
        width,height=cfg.rendering_resolution
        ix,iy=dr.meshgrid(dr.arange(mi.Float,width),dr.arange(mi.Float,height),indexing="xy")
        ray,_=sensor.sample_ray(time=0.,sample1=0.,
                               sample2=mi.Point2f((ix+.5)/width,(iy+.5)/height),
                               sample3=mi.Point2f(.5,.5))
        hit=visual_scene.ray_intersect(ray)
        visible=hit.is_valid() & ~hit.shape.is_emitter()
        distance=dr.select(visible,dr.norm(hit.p-sensor.world_transform().translation()),dr.inf)
        cache["stable_depth"]=mi.TensorXf(distance,shape=(height,width))
    depth=cache["stable_depth"]

    aovs = [depth]
    if use_denoiser:
        # Albedo
        aovs.append(img[..., 3:6])
        # Normals
        aovs.append(img[..., 6:9])

    return rgb, aovs, cache


def render_scene(
    cfg: RenderingConfig,
    scene: rt.Scene,
    seed: int,
    camera_changed: bool,
    cache: dict[str, Any] = None,
    use_denoiser: bool = False,
) -> tuple[mi.TensorXf, mi.TensorXf, dict[str, Any]]:
    assert cfg.mode == RenderingMode.RAY_TRACING

    if cache is None:
        with mi.scoped_set_variant("cuda_ad_rgb", "llvm_ad_rgb"):
            cache = setup_scene_for_rendering(cfg, scene, use_denoiser=use_denoiser)
        camera_changed = True

    return _render_scene(
        cfg, seed, camera_changed, cache=cache, use_denoiser=use_denoiser
    )


def set_envmap_rotation(
    cache: dict[str, Any],
    angle_deg: float,
    axis: tuple[float, float, float] = (0, 0, 1),
) -> bool:
    if "visual_scene" not in cache:
        return False

    props = mi.traverse(cache["visual_scene"])
    if cache.get("studio_transforms"):
        for key,base in cache["studio_transforms"].items():
            props[key]=mi.Transform4f(mi.ScalarTransform4f.rotate(axis=axis,angle=angle_deg+60))@base
        if "emitter.to_world" in props:
            props["emitter.to_world"]=mi.ScalarTransform4f.rotate(axis=axis,angle=angle_deg)@cache["envmap_base_transform"]
        props.update()
        return True
    if "emitter.to_world" not in props:
        return False

    props["emitter.to_world"] = (
        mi.ScalarTransform4f.rotate(axis=axis, angle=angle_deg)
        @ cache["envmap_base_transform"]
    )
    props.update()
    return True


def encode_render_image(ray_traced_img):
    # Raw image quantities bypass Polyscope's mesh output transfer. Encode
    # linear radiance once here (after accumulation/denoising), not in materials.
    rgb=dr.maximum(ray_traced_img,0)
    ray_traced_img=dr.select(rgb<=.0031308,12.92*rgb,1.055*dr.power(rgb,1/2.4)-.055)
    ray_traced_img=dr.minimum(ray_traced_img,1)
    ray_traced_img = dr.concat(
        [ray_traced_img, dr.ones_like(ray_traced_img[..., :1])], axis=-1
    )
    return ray_traced_img


def add_or_update_ray_traced_image_quantity(
    ray_traced_img: mi.TensorXf, ray_traced_depth: mi.TensorXf
) -> bool:
    exists, _ = psb.get_global_floating_quantity_structure().has_quantity_buffer_type(
        "ray_traced_img", "colors"
    )
    ray_traced_img=encode_render_image(ray_traced_img)

    if supports_direct_update_from_device() and exists:
        ps.get_quantity_buffer("ray_traced_img", "colors").update_data_from_device(
            ray_traced_img
        )
        ps.get_quantity_buffer("ray_traced_img", "depths").update_data_from_device(
            ray_traced_depth
        )
    else:
        ps.add_raw_color_alpha_render_image_quantity(
            "ray_traced_img",
            depth_values=ray_traced_depth.numpy(),
            color_values=ray_traced_img.numpy(),
            enabled=True,
            image_origin="upper_left",
        )

    ps.request_redraw()
    return exists
