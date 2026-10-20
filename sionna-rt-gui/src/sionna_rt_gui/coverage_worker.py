"""Persistent mesh-coverage worker; one bounded beam batch per request."""
import json,sys,time,traceback
from pathlib import Path
import numpy as np
import mitsuba as mi
import sionna.rt as rt
from .config import AntennaArrayConfig
from .beamforming import precoding
from .panel_codebook import make_codebook,PER_PANEL,PANEL_ROTATIONS,matrix_to_orientation,rotation_matrix_numpy
from .terrain_coverage import build_surface,solve_surface


def main():
    scene=None;previous=None;surface=None;surface_key=None;solver=rt.RadioMapSolver()
    panel_dirs=make_codebook()[2]
    for line in sys.stdin:
        job=json.loads(line);output=Path(job['output']);started=time.perf_counter()
        try:
            settings=job['settings']
            scene_key=json.dumps({k:settings[k] for k in ('scene','tx_array','rx_array','frequency','thickness','scattering')},sort_keys=True)
            if previous!=scene_key:
                scene=rt.load_scene(settings['scene'])
                scene.tx_array=AntennaArrayConfig(**settings['tx_array']).create()
                scene.rx_array=AntennaArrayConfig(**settings['rx_array']).create()
                scene.frequency=settings['frequency']
                scene.add(rt.Receiver('rx-0',[0,0,12.4]))
                for material in scene.radio_materials.values():
                    if settings['thickness'] is not None:material.thickness=settings['thickness']
                    if settings['scattering'] is not None:material.scattering_coefficient=settings['scattering']
                previous=scene_key;surface_key=None
            # Base-station placement does not change the scene BVH or terrain.
            # Reuse those expensive structures; only update transmitter devices.
            requested={tx['name'] for tx in settings['transmitters']}
            for name in list(scene.transmitters):
                if name not in requested:scene.remove(name)
            for data in settings['transmitters']:
                tx=scene.get(data['name'])
                if tx is None:scene.add(rt.Transmitter(**data))
                else:
                    tx.position=data['position'];tx.orientation=data['orientation'];tx.power_dbm=data['power_dbm']
            if settings.get('domain')=='air':
                from .air_surface import solve_air
                values,centers,mask=solve_air(solver,scene,settings,job,panel_dirs)
                np.savez(output.with_suffix('.tmp.npz'),values=values,centers=centers,mask=mask)
                output.with_suffix('.tmp.npz').replace(output.with_suffix('.npz'))
                output.with_suffix('.tmp.json').write_text(json.dumps(dict(samples=job['samples'],seconds=time.perf_counter()-started,beam_index=job['beam_index'])),encoding='utf-8')
                output.with_suffix('.tmp.json').replace(output.with_suffix('.json'))
                continue
            terrain_key=json.dumps([settings['scene'],settings['solver']['center'],settings['solver']['orientation'],settings['solver']['size'],
                                    settings['solver']['cell_size'],settings['height_above_ground'],settings.get('surface_rules','')])
            if terrain_key!=surface_key:
                surface=build_surface(scene,settings);surface_key=terrain_key
            beam=job['beam_index']
            for tx_data in settings['transmitters']:
                matrix=rotation_matrix_numpy(tx_data['orientation'])@PANEL_ROTATIONS[beam//PER_PANEL]
                scene.get(tx_data['name']).orientation=mi.Point3f(matrix_to_orientation(matrix))
            values=solve_surface(solver,scene,surface,settings,precoding(scene,panel_dirs[beam]),job['samples'],job['seed'])
            if not np.isfinite(values).all():raise ValueError('Non-finite coverage result')
            temporary=output.with_suffix('.tmp.npz')
            np.savez(temporary,values=values,centers=surface[1],mask=surface[2])
            temporary.replace(output.with_suffix('.npz'))
            result={'samples':job['samples'],'seconds':time.perf_counter()-started,'beam_index':beam}
        except Exception:
            result={'error':traceback.format_exc()}
        temporary=output.with_suffix('.tmp.json')
        temporary.write_text(json.dumps(result),encoding='utf-8')
        temporary.replace(output.with_suffix('.json'))


if __name__=='__main__':main()
