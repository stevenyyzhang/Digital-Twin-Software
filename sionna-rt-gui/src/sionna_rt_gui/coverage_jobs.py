"""Nonblocking terrain coverage with cached current/max modes and beam means."""
import atexit,json,subprocess,sys,time,uuid
from dataclasses import asdict
from pathlib import Path
import numpy as np
from .coverage_accumulator import CoverageAccumulator
from .terrain_coverage import display_map
from .host_values import host_value


def snapshot(app):
    c=app.cfg.radio_map
    filename=app.cfg.scene_filename
    cached=getattr(app,'_coverage_rules_cache',None)
    # Check changes without reopening/parsing the same file every display frame.
    if cached is None or cached[0]!=filename:
        scene_path=Path(filename).resolve();rules=scene_path.parent/'coverage-surfaces.json'
        cached=[filename,str(scene_path),rules,None,'']
        app._coverage_rules_cache=cached
    rules=cached[2]
    try:stamp=(rules.stat().st_mtime_ns,rules.stat().st_size)
    except FileNotFoundError:stamp=None
    if stamp!=cached[3]:
        cached[3]=stamp;cached[4]=rules.read_text(encoding='utf-8') if stamp else '' 
    names=['center','orientation','size','cell_size','max_depth','los',
           'specular_reflection','diffuse_reflection','refraction','diffraction','edge_diffraction','diffraction_lit_region']
    return dict(scene=cached[1],tx_array=asdict(app.cfg.tx_array),
        rx_array=asdict(app.cfg.rx_array),frequency=float(host_value(app.scene,"frequency").ravel()[0]),
        thickness=app.cfg.radio_material_thickness,scattering=app.cfg.radio_material_scattering_coefficient,
        transmitters=[dict(name=tx.name,position=host_value(tx,"position").ravel().tolist(),
                          orientation=host_value(tx,"orientation").ravel().tolist(),power_dbm=float(host_value(tx,"power_dbm").ravel()[0]))
                      for tx in app.scene.transmitters.values()],
        surface_rules=cached[4],
        mode=c.coverage_mode,height_above_ground=c.height_above_ground,
        domain=c.coverage_domain,air_heights=list(c.air_heights) if c.coverage_domain=='air' else [],
        beam_index=app.beam_demo.selected if c.coverage_mode=='current' else -1,
        solver={k:getattr(c,k) for k in names})


def beam_budget(app):
    c=app.cfg.radio_map
    return c.max_samples_per_beam if c.coverage_mode=='max' else c.accumulate_max_samples_per_tx


def sample_budget(app):
    return beam_budget(app)*(len(app.cfg.radio_map.air_heights) if app.cfg.radio_map.coverage_domain=='air' else len(app.beam_demo.local_dirs) if app.cfg.radio_map.coverage_mode=='max' else 1)


class CoverageJobs:
    def __init__(self,app):
        self.folder=Path(app.cfg.config_path).resolve().parent/'.coverage-cache'
        self.folder.mkdir(exist_ok=True)
        self.process=None;self.pending=None;self.log=None;self.seed=0
        self.next_submit=0.;self.status='Ready to calculate';self.last_seconds=0.;self.last_error=None
        self.states={}
        self.air=None
        atexit.register(self.close)

    def close(self):
        if self.air is not None:self.air.close()
        pending=self.pending
        if self.process is not None:
            if self.process.poll() is None:
                self.process.terminate()
                try:self.process.wait(timeout=2)
                except subprocess.TimeoutExpired:self.process.kill()
            if self.process.stdin:self.process.stdin.close()
            self.process=None
        if self.log is not None:self.log.close();self.log=None
        self.pending=None
        if pending is not None:
            for suffix in ('.npz','.json','.tmp.npz','.tmp.json'):
                Path(pending['output']).with_suffix(suffix).unlink(missing_ok=True)

    def cancel(self):
        if self.air is not None:self.air.cancel()
        if self.pending is not None:self.pending['discard']=True
        self.status='Paused'

    def state(self,app):
        settings=snapshot(app);mode=settings['mode']
        state=self.states.get(mode)
        if state is None or state['settings']!=settings:
            state=dict(settings=settings,acc=CoverageAccumulator(len(app.beam_demo.local_dirs) if mode=='max' else 1),rm=None)
            self.states[mode]=state
        return state

    def restore(self,app):
        if app.cfg.radio_map.coverage_domain=='air':
            if self.air is None:
                from .air_coverage import AirCoverageJobs
                self.air=AirCoverageJobs(app,self)
            return self.air.restore(app)
        state=self.state(app)
        if state['rm'] is None:return False
        app._last_rm_batch_samples=int(state['acc'].counts.sum())
        refining=app.rm_refining
        app.set_radio_map(state['rm'],show=True);app.rm_refining=refining
        return True

    def submit(self,app):
        if app.cfg.radio_map.coverage_domain=='air':
            if self.air is None:
                from .air_coverage import AirCoverageJobs
                self.air=AirCoverageJobs(app,self)
            return self.air.submit(app)
        state=self.state(app)
        if app.radio_map is None:self.restore(app)
        if self.pending is not None:return
        settings=state['settings'];acc=state['acc']
        self.last_error=None
        i=acc.next_beam(beam_budget(app))
        if i is None:app.rm_refining=False;self.status='Budget reached';return
        if len(settings['transmitters'])!=1:
            app.rm_refining=False;self.status='Terrain demo requires one base station';return
        if self.process is None or self.process.poll() is not None:
            self.close()
            self.log=(self.folder/'worker.log').open('a',encoding='utf-8')
            self.process=subprocess.Popen([sys.executable,'-m','sionna_rt_gui.coverage_worker'],
                stdin=subprocess.PIPE,stdout=self.log,stderr=self.log,text=True,
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        samples=min(app.cfg.radio_map.samples_per_it,beam_budget(app)-int(acc.counts[i]))
        beam=i if settings['mode']=='max' else settings['beam_index']
        output=self.folder/uuid.uuid4().hex
        job=dict(settings=settings,samples=samples,seed=self.seed,output=str(output),beam_index=beam)
        self.seed+=1;self.pending=job
        self.process.stdin.write(json.dumps(job)+'\n');self.process.stdin.flush()
        self.status=f'Calculating beam {beam}; {np.count_nonzero(acc.counts)}/{len(acc.counts)} scanned'

    def tick(self,app):
        if app.cfg.radio_map.coverage_domain=='air':
            if self.air is None:
                from .air_coverage import AirCoverageJobs
                self.air=AirCoverageJobs(app,self)
            return self.air.tick(app)
        job=self.pending
        if job is not None:
            output=Path(job['output']);done=output.with_suffix('.json')
            if done.exists():
                result=json.loads(done.read_text());self.pending=None
                if job.get('discard') or snapshot(app)!=job['settings']:
                    self.status='Switching settings' if app.rm_refining else 'Paused'
                elif 'error' in result:
                    self.last_error='No measurable ground. Choose an urban scene or configure ground meshes.' if 'No named ground' in result['error'] else 'Coverage calculation failed. See .coverage-cache/worker.log.'
                    self.status=self.last_error
                    with (self.folder/'worker.log').open('a',encoding='utf-8') as log:log.write(result['error'])
                    app.rm_refining=False
                else:
                    state=self.state(app);acc=state['acc'];is_max=job['settings']['mode']=='max'
                    with np.load(output.with_suffix('.npz')) as data:
                        acc.add(result['beam_index'] if is_max else 0,data['values'],result['samples'])
                        values,winner=acc.result()
                        rm=display_map(app.scene,app.cfg.radio_map,data['centers'].copy(),data['mask'].copy(),values)
                    rm.coverage_mode=job['settings']['mode']
                    rm.best_beam=winner[0] if is_max else np.full(values.shape[1:],result['beam_index'])
                    rm.beams_visited=int(np.count_nonzero(acc.counts));rm.beams_total=len(acc.counts)
                    rm.coverage_label=(f'Best-beam RSS {rm.beams_visited}/{rm.beams_total} beams'+(' (partial)' if rm.beams_visited<rm.beams_total else '')
                                       if is_max else f'Beam {result["beam_index"]} RSS')
                    state['rm']=rm
                    self.restore(app)
                    self.last_seconds=result['seconds'];self.next_submit=time.monotonic()+.03
                    app.demo_status=f'{rm.coverage_label}; {app.cfg.radio_map.height_above_ground:g} m above terrain'
                    if app.rm_accumulated_samples>=sample_budget(app):
                        app.rm_refining=False;self.status='Sample budget complete (not a convergence guarantee)'
                done.unlink(missing_ok=True);output.with_suffix('.npz').unlink(missing_ok=True)
            elif self.process.poll() is not None:
                self.pending=None;app.rm_refining=False;self.last_error='Calculation worker exited. Run again.';self.status=self.last_error
        visible=app.radio_map is None or app.has_visible_radio_map()[0]
        if self.pending is None and app.rm_refining and visible and time.monotonic()>=self.next_submit:
            self.submit(app)


def mark_outdoor_cells(scene,rm):
    # Legacy fixed-height maps outside the terrain demo pipeline.
    centers=rm.cell_centers.numpy();flat=centers.reshape(-1,3)
    import mitsuba as mi
    hit=scene.mi_scene.ray_intersect(mi.Ray3f(mi.Point3f(flat[:,0],flat[:,1],np.full(len(flat),500)),mi.Vector3f(0,0,-1)))
    rm.demo_outdoor_mask=(abs(hit.p.z.numpy().reshape(centers.shape[:2])-(centers[...,2]-1.5))<.8)
