"""Reproducible teaching scenarios, not automated optimal site selection."""
import numpy as np
import mitsuba as mi

SCENE_NOTES={
    'simple_street_canyon_with_cars':('Street with cars','Start here: see beam direction, blockage and reflections along a street.'),
    'simple_street_canyon':('Street canyon','A simple street for beam steering and blockage.'),
    'polyu-stylized':('PolyU campus','Campus coverage: compare one beam with the best beam envelope.'),
    'florence':('Florence','City-scale coverage. Start with a 6 m grid.'),
    'munich':('Munich','Explore another city. Start with a coarse grid.'),
}


def prepare_route(app):
    key=(id(app.scene),tuple(app.cfg.radio_map.cell_size),app.cfg.radio_map.height_above_ground)
    if getattr(app,'presentation_route_key',None)==key:return app.presentation_route.copy()
    if app.demo_scene_data and app.demo_scene_data.get('route_kind')=='air':
        from .air_routes import segment_clear
        route=np.asarray(app.demo_scene_data['route'],dtype=float)
        if len(route)<2 or not all(segment_clear(app,a,b) for a,b in zip(route[:-1],route[1:])):
            raise ValueError('Scene flight route failed the geometry clearance check')
        app.air_route_mode=True
        app.presentation_route_key=key;app.presentation_route=route
        return route.copy()
    from .coverage_jobs import snapshot
    from .terrain_coverage import build_surface
    _,centers,mask,_=build_surface(app.scene,snapshot(app))
    points=centers[mask]
    ref=np.asarray(app.demo_scene_data['rx']) if app.demo_scene_data else np.median(points,axis=0)
    # Choose nearby measured ground, never bbox center inside a building.
    candidates=points[np.argsort(np.linalg.norm(points[:,:2]-ref[:2],axis=1))[:64]]
    cell=app.cfg.radio_map.cell_size[0];stride=max(5.,2*cell)
    xy0=centers[0,0,:2]
    def at(xy):
        col,row=np.rint((xy-xy0)/cell).astype(int)
        if 0<=row<mask.shape[0] and 0<=col<mask.shape[1] and mask[row,col]:return centers[row,col].copy()
    def clear(a,b):
        delta=b-a;length=float(np.linalg.norm(delta))
        if length<.1 or abs(delta[2])>.8:return False
        ray=mi.Ray3f(mi.Point3f(a),mi.Vector3f(delta/length));ray.maxt=length
        if bool(app.scene.mi_scene.ray_test(ray)[0]):return False
        for t in np.linspace(0,1,max(3,int(length)+1)):
            p=at(a[:2]+t*delta[:2])
            if p is None or abs(p[2]-(a[2]+t*delta[2]))>.8:return False
        return True
    route=None
    for a in candidates:
        for direction in ([1,0],[-1,0],[0,1],[0,-1],[1,1],[-1,1],[1,-1],[-1,-1]):
            direction=np.asarray(direction,dtype=float);direction/=np.linalg.norm(direction)
            b=at(a[:2]+direction*stride);c=at(a[:2]+direction*stride*2)
            if b is not None and c is not None and clear(a,b) and clear(b,c):
                route=np.array([a,b,c]);break
        if route is not None:break
    if route is None:route=candidates[:1].copy()
    app.presentation_route_key=key;app.presentation_route=route
    return route.copy()


def apply_preset(app,site=0):
    from .demo_workflow import place_tx
    from .spatial_beam import choose_target
    route=prepare_route(app)
    data=app.demo_scene_data
    position=np.asarray(data['tx']) if data and site==0 else route[0]+[0,0,3.5]
    # Old presets stored absolute antenna heights. Resolve the actual support
    # beneath that point so the visible mast and the stated 5 m agree.
    from .demo_workflow import support_below
    support = support_below(app, position)
    if support is not None: position = support + [0, 0, 5.0]
    place_tx(app,position)
    app.demo_default_rx=route[0].tolist();app.demo_default_route=route.tolist()
    app.beam_demo.sync(app);choose_target(app,route[-1])
    app.pause_coverage()
    app.demo_status='Example site and ground-following route ready. Site placement is illustrative, not optimized.'
    if app.demo_page==1:
        from .demo_workflow import set_page
        set_page(app,1);app.restore_demo_route()


def best_in_current_panel(app):
    b=app.beam_demo;start=b.panel*49;values=b.values[start:start+49]
    if not np.isfinite(values).any():
        app.demo_status='No resolved paths at this receiver. Beam comparison is unavailable.';return
    b.select(start+int(np.argmax(values)))
    app.demo_status='Selected the strongest of 49 beams in this panel at the receiver.'
