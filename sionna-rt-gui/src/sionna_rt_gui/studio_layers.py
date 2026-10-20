"""Six cached, aligned top views. Only upload a texture when its result changes."""
import numpy as np
import polyscope as ps
import polyscope.imgui as ui
from .spatial_beam import SpatialBeam
from .coverage_display import plan_overlay


def draw_layers(app):
    if not hasattr(app,'studio_plan'):app.studio_plan=SpatialBeam()
    plan=app.studio_plan;plan.prepare(app)
    c=app.cfg.radio_map
    ui.TextColored((.25,.82,.77,1),'RSS | six heights | Model Z [m]')
    ui.SameLine()
    if ui.SmallButton('Collapse'):app.studio_maps_collapsed=True
    air=getattr(app.coverage_jobs,'air',None)
    layers=air.layers if air is not None and air.settings is not None else []
    cache=getattr(app,'studio_layer_textures',None)
    if cache is None:cache={};app.studio_layer_textures=cache
    width,available=ui.GetContentRegionAvail()
    gap=10.;caption=ui.GetFontSize()+10
    footer=3*ui.GetFontSize()+14+4*ui.GetStyle().ItemSpacing[1]
    side=max(40.,min((width-gap)/2,(available-footer)/3-caption))
    origin=np.asarray(ui.GetCursorScreenPos())
    app.studio_thumbnail_side=side
    budget=c.accumulate_max_samples_per_tx

    for i,height in enumerate(c.air_heights):
        layer=layers[i] if i<len(layers) else None
        rm=layer['rm'] if layer else None
        key=(id(app.scene),id(rm),layer['revision'] if layer else 0,c.vmin,c.vmax,c.color_map)
        if cache.get(i,{}).get('key')!=key:
            texture=None
            if rm is not None:
                from sionna.rt.utils.geometry import rotation_matrix
                basis=rotation_matrix(rm.orientation).numpy().reshape(3,3)[:2,:2]@np.diag(np.asarray(rm.cell_size.numpy()).ravel())
                grid_origin=rm.cell_centers.numpy()[0,0,:2]-basis@np.array([.5,.5])
                gain=np.max(rm.path_gain.numpy(),axis=0)
                pixels=plan_overlay(gain,layer['data']['mask'],grid_origin,basis,plan.low,plan.side,512,rm.beam_power_dbm,c)
                name=f'studio-height-{i}'
                ps.add_color_alpha_image_quantity(name,pixels,enabled=True,image_origin='upper_left',show_fullscreen=False)
                texture=ps.get_quantity_buffer(name,'colors').get_texture_native_id()
            cache[i]=dict(key=key,texture=texture)
        ui.SetCursorScreenPos(tuple(origin+[(i%2)*(side+gap),(i//2)*(side+caption)]))
        start=np.asarray(ui.GetCursorScreenPos())
        ui.Image(ui.ImTextureRef(plan.texture),(side,side))
        if ui.IsItemHovered():
            ui.SetTooltip(f'Show only Z = {height:g} m in 3D')
            if ui.IsMouseClicked(0):
                c.air_selected_layer=i;c.air_show_in_3d=True;c.air_show_stack=False
                if air:air.restore(app)
        draw=ui.GetWindowDrawList()
        if c.air_show_in_3d and not c.air_show_stack and c.air_selected_layer==i:
            draw.AddRect(tuple(start),tuple(start+side),0xFFBBD84A,0.,0,2.)
        texture=cache[i]['texture']
        if texture is not None:draw.AddImage(ui.ImTextureRef(texture),tuple(start),tuple(start+side))
        tx=app.scene.get('tx-0')
        if tx is not None:
            from .host_values import host_value
            uv=(host_value(tx,'position').ravel()[:2]-plan.low)/plan.side
            uv[1]=1-uv[1]
            point=start+uv*side
            if np.all((uv>=0)&(uv<=1)):draw.AddCircleFilled(tuple(point),3.,0xFFFFFFFF)
        samples=int(layer['acc'].counts[0]) if layer else 0
        ui.SetCursorScreenPos(tuple(start+[0,side+4]))
        ready=rm is not None and samples>=budget
        label=f'{height:g} m | Ready' if ready else f'{height:g} m | {samples/1e6:.1f}/{budget/1e6:g}M'
        ui.TextColored((.25,.82,.77,1) if ready else (.57,.66,.75,1),label)
        if ready and ui.IsItemHovered():
            ui.SetTooltip(f'{samples/1e6:g}M samples completed at this height. Budget reached; not a convergence guarantee.')
    ui.SetCursorScreenPos(tuple(origin+[0,3*(side+caption)+4]))
    if ui.Button('Show all heights'):
        c.air_show_in_3d=True;c.air_show_stack=True
        if air:air.restore(app)
    ui.TextColored((.57,.66,.75,1),'RSS -120 to -40 dBm')
    ui.TextColored((.57,.66,.75,1),'Blue-gray: no valid estimate')
    error=getattr(app.coverage_jobs,'last_error',None)
    if error:ui.TextWrapped(error)
