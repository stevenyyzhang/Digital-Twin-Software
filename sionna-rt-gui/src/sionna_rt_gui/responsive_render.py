"""Bounded background display rendering, independent of all radio solvers.

Only the UI thread creates visual scenes and presents images. A single worker
owns accumulation and denoising. Stale results are discarded after navigation,
resize or scene/mode changes. There is never a queue of outdated frames.
"""
from concurrent.futures import ThreadPoolExecutor
from copy import copy
import time
import drjit as dr
import mitsuba as mi
import polyscope as ps
import polyscope_bindings as psb
from .rendering import setup_scene_for_rendering, _render_scene, encode_render_image


def _step(cfg, cache, camera, denoiser, state, seed, needs_host):
    started=time.perf_counter()
    # Dr.Jit requires a fresh basic-block scope at each cross-thread handoff.
    dr.detail.new_scope(dr.JitBackend.CUDA)
    rgb,aovs,_=_render_scene(cfg,seed,state is None,cache,
                            use_denoiser=denoiser is not None,camera_pose=camera)
    n=cfg.spp_per_frame
    if state is None:
        raw,depth=rgb,aovs[0]
        albedo,normals=(aovs[1],aovs[2]) if denoiser is not None else (None,None)
    else:
        old_n,old_raw,depth,old_albedo,old_normals=state
        n+=old_n
        # Keep the weight opaque, or every accumulation count specializes and
        # recompiles a new GPU kernel.
        weight=dr.opaque(mi.Float32,cfg.spp_per_frame/n)
        raw=(1-weight)*old_raw+weight*rgb
        albedo=((1-weight)*old_albedo+weight*aovs[1]) if denoiser is not None else None
        normals=((1-weight)*old_normals+weight*aovs[2]) if denoiser is not None else None
    # Early previews, then every 32 spp: do not filter a full-size image every
    # UI frame. Final accumulated radiance and denoising quality are unchanged.
    publish=n>=cfg.max_accumulated_spp or n in (8,16) or n%32==0
    display=None
    if publish:
        display=raw if denoiser is None else denoiser(
            raw,albedo=albedo,normals=normals,
            to_sensor=cache["sensor"].world_transform().inverse())
        display=encode_render_image(display)
    for value in (raw,depth,albedo,normals,display):
        if value is not None:dr.eval(value)
    dr.sync_thread()
    host=(display.numpy(),depth.numpy()) if publish and needs_host else None
    return (n,raw,depth,albedo,normals),display,host,(time.perf_counter()-started)*1000


class DisplayRenderer:
    def __init__(self):
        self.pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix="studio-display")
        self.future=None
        self.generation=0
        self.state=None
        self.last_step_ms=0.
        self.submitted=0
        self.discarded=0
        self.closed=False
        self.quantity=None

    def invalidate(self):
        # Never wait for a GPU job while handling mouse input.
        self.generation+=1
        self.state=None
        if self.quantity is not None:self.quantity.set_enabled(False)

    def tick(self,app):
        if self.future is not None:
            if not self.future.done():return
            future,self.future=self.future,None
            state,display,host,self.last_step_ms=future.result()
            dr.detail.new_scope(dr.JitBackend.CUDA)
            if self.job_generation!=self.generation:
                self.discarded+=1
            else:
                self.state=state
                n,raw,depth,_,_=state
                app.rendering_accumulated_samples=n
                if display is not None:
                    app.ray_traced_img=raw
                    app.ray_traced_depth=depth
                    if self.quantity is None:
                        rgba,z=host
                        h,w=z.shape
                        self.quantity=psb.get_global_floating_quantity_structure().add_raw_color_alpha_render_image_quantity(
                            "ray_traced_img",w,h,z.reshape(-1),rgba.reshape(-1,4),psb.ImageOrigin.upper_left)
                    else:
                        ps.get_quantity_buffer("ray_traced_img","colors").update_data_from_device(display)
                        ps.get_quantity_buffer("ray_traced_img","depths").update_data_from_device(depth)
                    self.quantity.set_enabled(True)
                    ps.request_redraw()
                    app.ps_groups["scene"].set_enabled(False)
        cfg=copy(app.cfg.rendering)
        completed=0 if self.state is None else self.state[0]
        if completed>=cfg.max_accumulated_spp:return
        if app.render_cache is None:
            with mi.scoped_set_variant("cuda_ad_rgb","llvm_ad_rgb"):
                app.render_cache=setup_scene_for_rendering(
                    cfg,app.scene,use_denoiser=app.denoiser is not None)
        # Bound each job and never queue the next one during interaction.
        cfg.spp_per_frame=min(8,cfg.spp_per_frame,cfg.max_accumulated_spp-completed)
        camera=ps.get_camera_view_matrix().copy()
        self.job_generation=self.generation
        self.future=self.pool.submit(_step,cfg,app.render_cache,camera,app.denoiser,
                                     self.state,app.frame_i,self.quantity is None)
        self.submitted+=1

    def close(self):
        if not self.closed:
            self.closed=True
            self.pool.shutdown(wait=True,cancel_futures=True)
            self.future=None
            self.state=None


def tick(app):
    renderer=getattr(app,"_display_renderer",None)
    if renderer is None:
        renderer=app._display_renderer=DisplayRenderer()
    if app.ray_traced_img is None:app.ps_groups["scene"].set_enabled(True)
    try:
        renderer.tick(app)
    except Exception as error:
        # A display/driver failure must not disable mouse handling or RF work.
        from .config import RenderingMode
        app.set_rendering_mode(RenderingMode.RASTERIZATION)
        app.demo_status=f"Original display error; Fast preview retained: {error}"
        print(f"[display-render] {app.demo_status}",flush=True)


def close(app):
    renderer=getattr(app,"_display_renderer",None)
    if renderer is not None:
        renderer.close()
        app._display_renderer=None
