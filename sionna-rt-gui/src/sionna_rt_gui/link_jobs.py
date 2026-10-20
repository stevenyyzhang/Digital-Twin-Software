"""Latest-position link jobs; render and input never wait for ray tracing."""
import atexit
import json
import subprocess
import sys
import time
import uuid
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import polyscope as ps
import polyscope.imgui as ui
from .beamforming import link_signature
from .config import DEFAULT_SLICE_PLANE_NAME


def same_environment(a,b):
    """Only receiver translation may differ for a labelled playback snapshot."""
    if a is None or b is None:return False
    def fixed(s):
        return {**s,'rx':[(name,orientation) for name,_,orientation in s['rx']]}
    return fixed(a)==fixed(b)


def current_result(app):
    return (app.scene.get('rx-0') is not None and
            getattr(app,'demo_link_settings',None)==link_signature(app))


class LinkJobs:
    def __init__(self,app):
        self.folder=Path(app.cfg.config_path).resolve().parent/'.link-cache'
        self.folder.mkdir(exist_ok=True)
        self.process=None;self.log=None;self.pending=None;self.desired=None
        self.last_change=0.;self.last_submit=0.;self.error=None;self.failed_settings=None
        self._io=ThreadPoolExecutor(max_workers=1,thread_name_prefix="link-result-io")
        self._result_future=None
        atexit.register(self.close)

    @staticmethod
    def cleanup(job):
        if job:
            for suffix in ('.npz','.json','.tmp.npz','.tmp.json'):
                try:Path(job['output']).with_suffix(suffix).unlink(missing_ok=True)
                except OSError:pass  # A cancelled worker can still be releasing a handle.

    def read_result(self,job):
        """File I/O only. Never call Polyscope, Sionna or GPU operations here."""
        out=Path(job['output']);done=out.with_suffix('.json')
        if not done.exists():return None
        try:
            result=json.loads(done.read_text(encoding='utf-8'))
            data=None
            if 'error' not in result:
                with np.load(out.with_suffix('.npz'),allow_pickle=False) as source:
                    data={key:source[key].copy() for key in ('amplitudes','delays','starts','ends')}
        except Exception:
            result={'error':traceback.format_exc()};data=None
        finally:
            self.cleanup(job)
        if 'error' in result:
            try:
                with (self.folder/'worker.log').open('a',encoding='utf-8') as log:log.write(result['error'])
            except OSError:pass
        return result,data

    @staticmethod
    def finish_close(process,log,job):
        if process is not None:
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                try:process.wait(timeout=2)
                except subprocess.TimeoutExpired:pass
            if process.stdin:process.stdin.close()
        if log:log.close()
        LinkJobs.cleanup(job)

    def close(self):
        # Detach first: a completed old Future can never become a new scene's
        # result. Waiting for I/O or worker exit must not block the event loop.
        process,log,job=self.process,self.log,self.pending
        self.process=None;self.log=None;self.pending=None;self._result_future=None
        if process is not None and process.poll() is None:
            try:process.terminate()
            except OSError:pass
        if process is not None or log is not None or job is not None:
            try:self._io.submit(self.finish_close,process,log,job)
            except RuntimeError:
                # Python may already have shut down the executor at interpreter
                # exit. Termination above is sufficient; OS closes handles.
                pass

    def reset(self):
        self.close();self.desired=None;self.error=None;self.failed_settings=None

    def request(self,app,force=False):
        if app.demo_page!=1 or not app.scene.transmitters or not app.scene.receivers:return
        settings=link_signature(app)
        if settings!=self.desired:
            self.desired=settings;self.last_change=time.monotonic();self.failed_settings=None;self.error=None
        if force:
            self.error=None;self.failed_settings=None
            app.demo_link_settings=None
            self.last_change=0.

    def tick(self,app):
        if app.demo_page!=1:return
        self.request(app)
        compatible=same_environment(app.demo_link_settings,self.desired)
        app.demo_link_snapshot_visible=compatible and app.animation_config.playing
        if ps.has_curve_network('paths'):
            ps.get_curve_network('paths').set_enabled(current_result(app) or app.demo_link_snapshot_visible)
        job=self.pending
        if job:
            if self._result_future is None:
                self._result_future=self._io.submit(self.read_result,job)
            if self._result_future.done() and time.monotonic()>=getattr(app,'_navigation_until',0.):
                try:payload=self._result_future.result()
                except Exception:payload=({'error':traceback.format_exc()},None)
                self._result_future=None
                if payload is not None:
                    result,data=payload;self.pending=None
                    # Revalidate at application time, after background I/O. A
                    # paused position, scene, panel or station may have changed.
                    compatible=same_environment(job['settings'],self.desired)
                    accept=compatible and (app.animation_config.playing or job['settings']==self.desired)
                    if accept and 'error' not in result:
                        app.demo_link_cir=(data['amplitudes'],data['delays'])
                        self.draw_segments(app,data['starts'],data['ends'])
                        app.demo_link_settings=job['settings'];app.demo_link_received=time.monotonic()
                        app.demo_link_seconds=result['seconds'];self.error=None
                        app.update_demo_cir()
                        app.beam_demo.signature=job['settings']
                    elif accept:
                        self.error='Link calculation failed. Use Recalculate to retry.'
                        self.failed_settings=job['settings']
            if self.pending is not None and (self.process is None or self.process.poll() is not None
                    or time.monotonic()-job['submitted']>120):
                self.close();self.error='Link worker exited or timed out. Retry.';self.failed_settings=self.desired
        if self.pending or not self.desired or self.failed_settings==self.desired:return
        if app.demo_link_settings==self.desired:return
        now=time.monotonic()
        # Scrubbing gets instant marker feedback, with a solve after release.
        # Playback instead samples regularly; it must not be starved by debounce.
        if not app.animation_config.playing and (ui.IsMouseDown(0) or now-self.last_change<.12):return
        if now-self.last_submit<max(.12,app.cfg.paths.min_update_delay_s):return
        if now<getattr(app,'_navigation_until',0.):return
        if self.process is None:
            self.log=(self.folder/'worker.log').open('a',encoding='utf-8')
            self.process=subprocess.Popen([sys.executable,'-m','sionna_rt_gui.link_worker'],
                stdin=subprocess.PIPE,stdout=self.log,stderr=self.log,text=True,
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        job=dict(settings=self.desired,output=str(self.folder/uuid.uuid4().hex),submitted=now)
        try:
            self.process.stdin.write(json.dumps(job)+'\n');self.process.stdin.flush()
            self.pending=job;self.last_submit=now
        except (BrokenPipeError,OSError):
            self.close();self.error='Could not submit link calculation. Retry.';self.failed_settings=self.desired

    def draw_segments(self,app,starts,ends):
        if ps.has_curve_network('paths'):ps.remove_curve_network('paths')
        if len(starts):
            vertices=np.stack([starts,ends],axis=1).reshape(-1,3)
            path=ps.register_curve_network('paths',vertices,edges='segments',color=(1.,.62,.18),enabled=True)
            path.set_radius(.10,relative=False);path.add_to_group(app.ps_groups['paths'])
            path.set_ignore_slice_plane(DEFAULT_SLICE_PLANE_NAME,True)
