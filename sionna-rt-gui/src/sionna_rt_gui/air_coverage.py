"""Bounded layer jobs, background result I/O, and cached layer display."""
import json
import subprocess
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import polyscope as ps
import polyscope.imgui as ui
from .coverage_accumulator import CoverageAccumulator
from .terrain_coverage import display_map
from .sionna_utils import add_radio_map_to_polyscope


def clear_stack():
    for i in range(8):
        name=f'Air layer {i}'
        if ps.has_surface_mesh(name):ps.remove_surface_mesh(name)


def validate_heights(heights):
    values=[float(h) for h in heights]
    if not 1<=len(values)<=8 or not np.isfinite(values).all() or len(set(values))!=len(values):
        raise ValueError('Use 1-8 distinct finite scene Z heights.')
    return values


def read_payload(job):
    out=Path(job['output'])
    done=out.with_suffix('.json')
    if not done.exists():return None
    try:
        meta=json.loads(done.read_text(encoding='utf-8'))
        if 'error' in meta:return meta,None
        with np.load(out.with_suffix('.npz'),allow_pickle=False) as data:
            arrays={k:data[k].copy() for k in ('values','centers','mask')}
        return meta,arrays
    except Exception as exc:
        return {'error':str(exc)},None
    finally:
        for suffix in ('.npz','.json'):
            out.with_suffix(suffix).unlink(missing_ok=True)


class AirCoverageJobs:
    def __init__(self,app,owner):
        self.owner=owner
        self.folder=Path(app.cfg.config_path).resolve().parent/'.air-cache'
        self.folder.mkdir(exist_ok=True)
        self.io=ThreadPoolExecutor(max_workers=1,thread_name_prefix='air-result')
        self.process=None;self.log=None;self.pending=None;self.future=None
        self.settings=None;self.layers=[];self.seed=0;self.display_key=None

    def close(self):
        process,log=self.process,self.log
        self.process=None;self.log=None
        if process is not None:
            if process.poll() is None:process.terminate()
            def reap():
                try:process.wait(timeout=3)
                except subprocess.TimeoutExpired:process.kill();process.wait()
                if process.stdin:process.stdin.close()
                if log:log.close()
            try:self.io.submit(reap)
            except RuntimeError:pass
        self.pending=None;self.future=None

    def cancel(self):
        if self.pending:self.pending['discard']=True

    def ensure(self,app):
        from .coverage_jobs import snapshot
        settings=snapshot(app)
        if settings!=self.settings:
            self.cancel();self.settings=settings
            self.layers=[dict(height=h,acc=CoverageAccumulator(1),rm=None,data=None,revision=0)
                         for h in validate_heights(settings['air_heights'])]
            self.display_key=None;clear_stack()
        return settings

    def total(self):
        return sum(int(layer['acc'].counts.sum()) for layer in self.layers)

    def restore(self,app):
        self.ensure(app)
        app.rm_accumulated_samples=self.total()
        index=min(max(app.cfg.radio_map.air_selected_layer,0),len(self.layers)-1)
        layer=self.layers[index]
        show=app.cfg.radio_map.air_show_in_3d and app.demo_page==0
        stack=show and app.cfg.radio_map.air_show_stack
        key=(index,tuple(x['revision'] for x in self.layers),show,stack,app.demo_page)
        if key==self.display_key and app.radio_map is layer['rm']:return layer['rm'] is not None
        previous=self.display_key
        self.display_key=key
        main_changed=(previous is None or previous[0]!=index or previous[1][index]!=key[1][index]
                      or previous[2:]!=key[2:] or app.radio_map is not layer['rm'])
        if main_changed and ps.has_surface_mesh('radio_map'):ps.remove_surface_mesh('radio_map')
        if main_changed:app.radio_map=None
        if main_changed and layer['rm'] is not None:
            active=app.rm_refining;app._last_rm_batch_samples=self.total()
            app.set_radio_map(layer['rm'],show=show);app.rm_refining=active
        if stack:
            if ps.has_surface_mesh('radio_map'):ps.get_surface_mesh('radio_map').set_transparency(.3)
            for i,other in enumerate(self.layers):
                name=f'Air layer {i}'
                if i==index or other['rm'] is None:
                    if ps.has_surface_mesh(name):ps.remove_surface_mesh(name)
                    continue
                if (previous is not None and previous[3] and previous[1][i]==key[1][i]
                        and ps.has_surface_mesh(name)):continue
                add_radio_map_to_polyscope(name,other['rm'],app.ps_groups,app.cfg.radio_map)
                ps.get_surface_mesh(name).set_transparency(.3)
        else:clear_stack()
        return layer['rm'] is not None

    def submit(self,app):
        self.ensure(app)
        self.restore(app)
        if self.pending or app.demo_page!=0:return
        if app.cfg.radio_map.coverage_mode!='current':
            app.rm_refining=False;self.owner.last_error='Air preview supports Selected beam only.';return
        budget=app.cfg.radio_map.accumulate_max_samples_per_tx
        # Round-robin by completed sample count: every layer gets an early result.
        index=min(range(len(self.layers)),key=lambda i:self.layers[i]['acc'].counts[0])
        count=int(self.layers[index]['acc'].counts[0])
        if count>=budget:
            app.rm_refining=False;self.owner.status='All layer budgets complete'
            app.demo_status='3D RSS ready. Blue-gray cells have no valid power estimate.'
            return
        if self.process is None or self.process.poll() is not None:
            self.close()
            self.log=(self.folder/'worker.log').open('a',encoding='utf-8')
            self.process=subprocess.Popen([sys.executable,'-m','sionna_rt_gui.coverage_worker'],
                stdin=subprocess.PIPE,stdout=self.log,stderr=self.log,text=True,
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        job=dict(settings=self.settings,layer=index,height=self.layers[index]['height'],
                 samples=min(app.cfg.radio_map.samples_per_it,budget-count),seed=self.seed,
                 beam_index=app.beam_demo.selected,output=str(self.folder/uuid.uuid4().hex),submitted=time.monotonic())
        self.seed+=1;self.owner.last_error=None
        try:
            self.process.stdin.write(json.dumps(job)+'\n');self.process.stdin.flush()
            self.pending=job
            self.owner.status=f"Computing layer {index+1}/{len(self.layers)}, Z={job['height']:g} m"
            app.demo_status=self.owner.status
        except (OSError,ValueError):
            self.close();app.rm_refining=False;self.owner.last_error='Air worker submission failed.'

    def tick(self,app):
        self.ensure(app)
        job=self.pending
        navigating=time.monotonic()<getattr(app,'_navigation_until',0.)
        if job is not None:
            if self.future is None:self.future=self.io.submit(read_payload,job)
            if self.future.done() and not navigating:
                try:payload=self.future.result()
                except Exception as exc:payload=({'error':str(exc)},None)
                self.future=None
                if payload is not None:
                    self.pending=None;meta,data=payload
                    if not job.get('discard') and job['settings']==self.settings:
                        if 'error' in meta:
                            self.owner.last_error='Air calculation failed. See .air-cache/worker.log.'
                            app.rm_refining=False
                        else:
                            layer=self.layers[job['layer']]
                            layer['acc'].add(0,data['values'],meta['samples'])
                            values,_=layer['acc'].result()
                            rm=display_map(app.scene,app.cfg.radio_map,data['centers'],data['mask'],values)
                            rm.coverage_mode='current';rm.air_height=layer['height']
                            rm.air_samples=int(layer['acc'].counts[0])
                            rm.best_beam=np.full(data['mask'].shape,job['beam_index'])
                            rm.beams_visited=1;rm.beams_total=1
                            # All layers, including hidden ones, carry immutable power provenance.
                            rm.beam_power_dbm=job['settings']['transmitters'][0]['power_dbm']
                            rm.beam_index=job['beam_index']
                            layer['rm']=rm;layer['data']=data;layer['revision']+=1
                            self.owner.last_seconds=meta['seconds']
                            self.restore(app)
            if self.pending and (self.process.poll() is not None or time.monotonic()-job['submitted']>120):
                self.close();app.rm_refining=False;self.owner.last_error='Air worker exited or timed out.'
        if not navigating:self.restore(app)
        if self.pending is None and app.rm_refining and not navigating:self.submit(app)

    def export(self,folder,app):
        shape=next(x['data']['mask'].shape for x in self.layers if x['data'] is not None)
        rss=np.full((len(self.layers),*shape),np.nan,dtype=np.float32)
        masks=np.zeros(rss.shape,dtype=bool);available=[]
        for i,layer in enumerate(self.layers):
            available.append(layer['data'] is not None)
            if layer['data'] is None:continue
            mask=layer['data']['mask'];gain=layer['acc'].result()[0][0]
            valid=mask & (gain>0) & np.isfinite(gain)
            masks[i]=mask;rss[i,valid]=10*np.log10(gain[valid])+layer['rm'].beam_power_dbm
        base=next(x['data']['centers'] for x in self.layers if x['data'] is not None)
        np.savez_compressed(folder/'air-volume.npz',rss_dbm=rss,valid_space=masks,
            heights_m=[x['height'] for x in self.layers],xy_centers=base[...,:2],
            layer_available=available,samples=[int(x['acc'].counts[0]) for x in self.layers])
        (folder/'air-volume.json').write_text(json.dumps(dict(settings=self.settings,
            axes=['z','y','x'],height_reference='scene-coordinate Z, metres; not AGL or sea-level altitude',
            mask_policy='conservative open-sky envelope; underpasses excluded; no support is unknown',
            height_selection=app.cfg.radio_map.air_height_mode,
            height_profile=getattr(app,'air_height_profile',{}),
            interpolation='none',source='Sionna RT'),indent=2),encoding='utf-8')


def controls(app):
    c=app.cfg.radio_map
    changed,index=ui.Combo('Coverage space',int(c.coverage_domain=='air'),['Ground + 1.5 m','3D / height layers'])
    if changed:
        app.clear_radio_map();c.coverage_domain='air' if index else 'ground'
        if index:c.coverage_mode='current'
        app.beam_demo.map_signature=None
    if c.coverage_domain!='air':return False
    ui.TextWrapped('Selected beam | independent RSS height layers')
    from .scene_heights import height_labels, configure_scene_heights
    profile=getattr(app,'air_height_profile',{})
    if profile.get('available'):
        ui.TextWrapped(f"Model ground reference: {profile['ground_reference_z_m']:.2f} m; building geometry max Z: {profile['highest_building_z_m']:.2f} m.")
    elif profile:ui.TextWrapped(profile['reason'])
    if ui.Button('Use building-derived heights'):
        app.clear_radio_map();c.air_height_mode='buildings';configure_scene_heights(app);app.beam_demo.map_signature=None
    ui.TextWrapped('Model Z, not local AGL. Presets follow actual loaded geometry.')
    if ui.CollapsingHeader('Height layers'):
        heights=list(c.air_heights)
        for i,h in enumerate(heights):
            changed,value=ui.InputFloat(f'Z {i+1} [m]',h)
            if changed and np.isfinite(value) and value not in heights[:i]+heights[i+1:]:
                app.clear_radio_map();c.air_height_mode='manual';heights[i]=float(value);c.air_heights=heights
                app.beam_demo.map_signature=None
        if len(heights)<8 and ui.Button('Add height'):
            app.clear_radio_map();c.air_height_mode='manual';c.air_heights=heights+[max(heights)+10.];app.beam_demo.map_signature=None
        if len(heights)>1:
            ui.SameLine()
            if ui.Button('Remove last height'):
                app.clear_radio_map();c.air_height_mode='manual';c.air_heights=heights[:-1]
                c.air_selected_layer=min(c.air_selected_layer,len(c.air_heights)-1);app.beam_demo.map_signature=None
    changed,index=ui.Combo('Visible height',min(c.air_selected_layer,len(c.air_heights)-1),height_labels(app))
    if changed:c.air_selected_layer=index
    changed,stack=ui.Checkbox('Show other calculated layers',c.air_show_stack)
    if changed:c.air_show_stack=stack
    ui.TextWrapped('Open-sky space only; roofs/ground and space beneath overhangs are excluded. Gray: no valid sample.')
    if app.coverage_jobs and app.coverage_jobs.air:
        air=app.coverage_jobs.air
        ui.Text(f'Layers with results: {sum(x["rm"] is not None for x in air.layers)} / {len(air.layers)}')
        ui.TextWrapped(app.coverage_jobs.status)
    return True
