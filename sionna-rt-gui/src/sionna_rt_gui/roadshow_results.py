"""Result summaries and portable evidence, with explicit unknown-cell semantics."""
from datetime import datetime
from pathlib import Path
import json
import numpy as np


def summarize(gain, mask, power_dbm, threshold=-90.):
    valid=mask & np.isfinite(gain) & (gain>0)
    rss=np.full(gain.shape,np.nan)
    rss[valid]=10*np.log10(gain[valid])+power_dbm
    total=int(mask.sum());count=int(valid.sum())
    return dict(total=total,observed=count,unknown=total-count,
        sampled_fraction=count/max(total,1),threshold=float(threshold),
        above_threshold_fraction=float(np.mean(rss[valid]>=threshold)) if count else None,
        median_dbm=float(np.median(rss[valid])) if count else None),rss


def result_data(app):
    rm=app.radio_map
    if rm is None:return None
    threshold=getattr(app,'roadshow_threshold',-90.)
    key=(id(rm),app.rm_accumulated_samples,threshold)
    if getattr(app,'roadshow_result_key',None)==key:return app.roadshow_result
    gain=rm.path_gain.numpy()[0]
    mask=rm.demo_outdoor_mask
    summary,rss=summarize(gain,mask,rm.beam_power_dbm,threshold)
    from .coverage_jobs import snapshot
    settings=snapshot(app)
    if hasattr(rm,'air_height'):settings={**settings,'display_height_m':rm.air_height}
    result=dict(summary=summary,rss=rss,mask=mask.copy(),centers=rm.cell_centers.numpy(),
        settings=settings,samples=app.rm_accumulated_samples,
        beams_visited=getattr(rm,'beams_visited',1),beams_total=getattr(rm,'beams_total',1),
        beam=int(rm.beam_index),mode=getattr(rm,'coverage_mode','current'))
    if hasattr(rm,'air_height'):
        result['selected_layer_samples']=getattr(rm,'air_samples',0)
        result['samples_scope']='total across height layers'
    app.roadshow_result_key=key;app.roadshow_result=result
    return result


def comparable_settings(settings):
    return {k:v for k,v in settings.items() if k not in ('mode','beam_index')}


def compare(current, baseline):
    if baseline is None:return None,'Save a result as A, then change the beam and compute B.'
    if comparable_settings(current['settings'])!=comparable_settings(baseline['settings']):
        return None,'Scene, site or grid changed. Save a new baseline A.'
    common=np.isfinite(current['rss']) & np.isfinite(baseline['rss'])
    if not common.any():return None,'No valid cells shared by both results.'
    return dict(cells=int(common.sum()),median_delta_db=float(np.median(current['rss'][common]-baseline['rss'][common]))),''


def provenance(app):
    if Path(app.cfg.scene_filename).parent.name in ('mong-kok','kam-tin'):
        return ('Buildings, terrain and imagery: Copyright Hong Kong SAR Government, Lands Department '
                '(https://3d.map.gov.hk/). Source elevations retained; stylized facades and assumed RF materials. '
                'Vegetation imagery has no volumetric RF model. No government endorsement.')
    if 'polyu-stylized' in app.cfg.scene_filename.replace('\\','/'):
        return 'PolyU: buildings from Hong Kong Lands Department 3D data; ground references OpenStreetMap. Stylized geometry.'
    return 'Local Sionna RT scene assets; geometry and physical materials follow the source files.'


def export_result(app):
    """Export current values and the settings that generated them; no synthetic fills."""
    if app.demo_page==1:
        from .link_jobs import current_result
        if not current_result(app) or app.animation_config.playing:
            raise ValueError('Pause and wait for the current receiver result before exporting.')
    from .coverage_jobs import snapshot
    folder=Path(app.cfg.config_path).resolve().parent/'exports'/datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    folder.mkdir(parents=True,exist_ok=False)
    meta=dict(created=datetime.now().astimezone().isoformat(),source=provenance(app),
        model='Sionna RT ray tracing; schematic materials; not measurement validated',
        settings=snapshot(app),workspace='coverage' if app.demo_page==0 else 'link')
    if app.demo_page==0:
        data=result_data(app)
        if data is None:raise ValueError('Compute a coverage map first.')
        meta.update({k:v for k,v in data.items() if k not in ('rss','mask','centers')})
        from .coverage_jobs import sample_budget
        meta['state']='running_partial' if app.rm_refining else 'budget_reached' if app.rm_accumulated_samples>=sample_budget(app) else 'paused_partial'
        meta['meaning']='Per-cell best single beam, not simultaneous transmission' if data['mode']=='max' else 'Selected single beam'
        np.savez_compressed(folder/'coverage.npz',rss_dbm=data['rss'],outdoor_mask=data['mask'],
            centers=data['centers'],best_beam=app.radio_map.best_beam if hasattr(app.radio_map,'best_beam') else np.full(data['mask'].shape,data['beam']))
        rows=np.column_stack([data['centers'].reshape(-1,3),data['rss'].ravel(),data['mask'].ravel()])
        np.savetxt(folder/'coverage.csv',rows,delimiter=',',header='x_m,y_m,z_m,rss_dbm,in_measurement_area',comments='')
        if app.cfg.radio_map.coverage_domain=='air' and app.coverage_jobs.air is not None:
            app.coverage_jobs.air.export(folder,app)
        meta['comparison']=compare(data,getattr(app,'roadshow_baseline',None))[0]
    else:
        from dataclasses import asdict
        meta['path_solver']=asdict(app.cfg.paths)
        delays=app.demo_path_delays;powers=app.demo_path_powers
        np.savetxt(folder/'paths.csv',np.column_stack([delays,powers]),delimiter=',',header='delay_ns,path_power_gain_linear',comments='')
        rx=app.scene.get('rx-0')
        meta['receiver']=rx.position.numpy().ravel().tolist() if rx is not None else None
        meta['path_count']=len(powers)
        b=app.beam_demo
        meta['selected_beam']=b.selected;meta['panel']=b.panel
        meta['panel_local_azimuth_deg']=float(b.az[b.selected]);meta['panel_local_elevation_deg']=float(b.el[b.selected])
        meta['route']=app.animation_config.trajectories['rx-0'].points.tolist() if rx is not None else []
        meta['meaning']='Snapshot at the current receiver; noncoherent path powers, not a route-wide coverage map'
    (folder/'metadata.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
    (folder/'README.txt').write_text('Simulation results; not calibrated against measurements.\nIn coverage.csv, NaN means no valid sample or outside the measurement area. Use in_measurement_area to distinguish them.\nThe threshold pass rate uses only valid sampled cells, not the entire area.\nThe best beam envelope selects one beam per cell. It is not simultaneous multi-beam transmission.\npaths.csv is a snapshot at the current receiver. See metadata.json for settings, progress and provenance.\n',encoding='utf-8')
    import polyscope as ps
    # Export is invoked inside the live Polyscope callback. include_UI=True
    # enters an unsupported nested render and can terminate the application.
    ps.screenshot(str(folder/'view.png'),include_UI=False)
    return folder
