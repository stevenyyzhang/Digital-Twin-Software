"""Persistent Sionna path solver isolated from the interactive render loop."""
import json
import sys
import time
import traceback
from pathlib import Path
import numpy as np
import mitsuba as mi
import sionna.rt as rt
from .config import AntennaArrayConfig
from .panel_codebook import PANEL_ROTATIONS, matrix_to_orientation, rotation_matrix_numpy


def main():
    scene=None; scene_key=None; solver=rt.PathSolver()
    for line in sys.stdin:
        job=json.loads(line);s=job['settings'];out=Path(job['output']);start=time.monotonic()
        try:
            key=json.dumps({k:s[k] for k in ('scene','frequency','tx_array','rx_array','thickness','scattering')},sort_keys=True)
            if key!=scene_key:
                scene=rt.load_scene(s['scene'])
                scene.tx_array=AntennaArrayConfig(**s['tx_array']).create()
                scene.rx_array=AntennaArrayConfig(**s['rx_array']).create()
                scene.frequency=s['frequency']
                for material in scene.radio_materials.values():
                    if s['thickness'] is not None:material.thickness=s['thickness']
                    if s['scattering'] is not None:material.scattering_coefficient=s['scattering']
                scene_key=key
            for name in list(scene.transmitters)+list(scene.receivers):scene.remove(name)
            for i,(name,p,o) in enumerate(s['tx']):
                matrix=rotation_matrix_numpy(o)@PANEL_ROTATIONS[s['panel']]
                scene.add(rt.Transmitter(name,p,orientation=matrix_to_orientation(matrix),power_dbm=s['power'][i]))
            for name,p,o in s['rx']:scene.add(rt.Receiver(name,p,orientation=o))
            fields=('max_depth','max_num_paths_per_src','samples_per_src','synthetic_array','los',
                    'specular_reflection','diffuse_reflection','refraction','diffraction','edge_diffraction','diffraction_lit_region')
            paths=solver(scene,**{k:s['paths'][k] for k in fields},seed=12345)
            a,tau=paths.cir(normalize_delays=False,out_type='numpy')
            segments=rt.render.paths_to_segments(paths)
            starts,ends=(np.asarray(segments[0]),np.asarray(segments[1])) if segments else (np.empty((0,3)),np.empty((0,3)))
            np.savez(out.with_suffix('.tmp.npz'),amplitudes=a,delays=tau,starts=starts,ends=ends)
            out.with_suffix('.tmp.npz').replace(out.with_suffix('.npz'))
            result=dict(seconds=time.monotonic()-start)
        except Exception:result=dict(error=traceback.format_exc())
        out.with_suffix('.tmp.json').write_text(json.dumps(result),encoding='utf-8')
        out.with_suffix('.tmp.json').replace(out.with_suffix('.json'))


if __name__=='__main__':main()
