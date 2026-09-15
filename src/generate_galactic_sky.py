#!/usr/bin/env python3
"""Generate the canonical-location field, coeval clusters and physical nebulae."""
import argparse
from collections import Counter
from dataclasses import asdict, replace
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path

import numpy as np

import star_generator as field
from galaxy_environment import GalaxyParameters
from cluster_population import Isochrones
from deep_sky import make_populations,realize_visible
from nebula_morphology import add_morphologies


def generate(config, *, generation_id=None, progress=print, population_scale=1.):
    if not np.isfinite(population_scale) or population_scale<=0:raise ValueError('population_scale must be positive')
    gid=generation_id or datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
    grid=Isochrones();attempts=[]
    galpar=GalaxyParameters(observer_radius_pc=config.observer_radius_pc,
        observer_height_pc=config.observer_height_pc,dust_height_pc=config.dust_scale_height_pc)
    for index in range(config.max_catalog_attempts):
        rngs=[np.random.default_rng(np.random.SeedSequence(config.seed,spawn_key=(index,k))) for k in range(3)]
        # Zero dust is a conservative prerequisite cut. Radial dust can be less
        # than the old vertical-only model, so the old extincted list is NOT reused.
        raw,stats=field._sample_population(replace(config,av_per_kpc=0),rngs[0],gid,progress)
        if progress:progress(f'场星零消光候选 {len(raw)}；生成共龄星团和云气。')
        parents,dust,deep_stats=make_populations(rngs[1],galpar,grid,population_scale)
        if progress:progress(f"生成 {deep_stats['open_clusters']} 个疏散星团、{deep_stats['globular_clusters']} 个球状星团、{deep_stats['dust_clouds']} 个云团。")
        av=dust.extinction([s['pos_cartesian'] for s in raw])
        stars=[]
        for star,ext in zip(raw,av):
            star['extinction_Av']=float(ext);star['app_mag']+=float(ext)
            if star['app_mag']<=config.limiting_magnitude:stars.append(star)
        members,objects,clusters=realize_visible(parents,dust,grid,rngs[2],gid,config.limiting_magnitude)
        stars.extend(members)
        counts=Counter((s['spectral_type'],s['luminosity_class']) for s in stars if not s.get('cluster_id'))
        for component in stats['components']:
            component['visible']=counts[(component['spectral_type'],component['luminosity_class'])]
        stats['components'].append(dict(spectral_type='coeval',luminosity_class='PARSEC',
            sampled=deep_stats['living_stars'],visible=len(members)))
        stats['population_stars_sampled']+=deep_stats['living_stars']
        stats['not_visible']=stats['population_stars_sampled']-len(stars)
        accepted=field.count_in_range(len(stars),config)
        attempts.append(dict(attempt=index+1,spawn_key=[index],visible_count=len(stars),accepted=accepted))
        if progress:progress(f"完整尝试 {index+1}: 可见恒星 {len(stars)}（星团成员 {len(members)}），{'接受' if accepted else '重新抽取完整总体'}。")
        if accepted:break
    else:raise field.CountConstraintError(attempts,config)
    morphology=add_morphologies(objects,config.seed)
    names=('star_generator.py','stellar_physics.py','galaxy_environment.py','cluster_population.py','deep_sky.py','generate_galactic_sky.py','export_deep_sky.py','fetch_parsec.py','nebula_morphology.py')
    catalog={'metadata':dict(schema_version=3,generator_version='4.4',generation_id=gid,count=len(stars),
        coordinate_system='galactic',generation_parameters=asdict(config),random_generator='numpy.PCG64',numpy_version=np.__version__,
        random_substreams='SeedSequence(root, spawn_key=(attempt_index, 0 field / 1 clusters+clouds / 2 members))',
        population_model='canonical_radius_field_plus_coeval_clusters_v1',
        population_profile_version=field.POPULATION_PROFILE['profile_version'],
        extinction_model='radial_vertical_disk_plus_gaussian_clouds_v1',
        isochrone_sha256=grid.sha256,source_sha256={n:hashlib.sha256((field.SRC_DIR/n).read_bytes()).hexdigest() for n in names},
        generation_stats=stats,count_selection=dict(mode='whole_catalog_rejection' if config.minimum_visible_stars or config.maximum_visible_stars is not None else 'unconditioned',
            minimum=config.minimum_visible_stars,maximum=config.maximum_visible_stars,max_attempts=config.max_catalog_attempts,
            accepted_attempt=len(attempts),attempts=attempts),
        note='Synthetic SBbc model with declared priors, not a recovered map of the actual Milky Way at 9416 pc. Nebular V photometry and visual detection are approximations.'),
        'stars':stars,'galaxy':dust.to_dict(),'deep_sky':dict(model='PARSEC_Poisson_IMF_Plummer_caseB_v1',population_scale=population_scale,
            population_stats=deep_stats,export_unextinguished_mag_limit=12.,clusters=clusters,objects=objects,morphology=morphology)}
    validation=field.validate_catalog(catalog)
    if not validation['all_passed']:raise ValueError(json.dumps(validation['errors'][:5],ensure_ascii=False))
    catalog['metadata']['validation_stats']={k:v for k,v in validation.items() if k!='errors'}
    return catalog


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed',type=int,default=20260915)
    parser.add_argument('--density',type=float,default=field.DEFAULT_LOCAL_DENSITY)
    parser.add_argument('--unconditioned',action='store_true')
    parser.add_argument('--population-scale',type=float,default=1.,help='研究用总体强度倍率，正式默认为1')
    parser.add_argument('--output-root',type=Path,default=field.OUTPUT_DIR)
    parser.add_argument('--generation-id')
    args=parser.parse_args()
    config=field.GenerationConfig(seed=args.seed,local_density_per_pc3=args.density,
        minimum_visible_stars=0 if args.unconditioned else 9000,maximum_visible_stars=None if args.unconditioned else 9500,
        max_catalog_attempts=1 if args.unconditioned else 8)
    result=generate(config,generation_id=args.generation_id,progress=lambda s:print(s,flush=True),population_scale=args.population_scale)
    path=field.save_catalog(result,args.output_root,plots=False)
    from export_deep_sky import export_browser
    print(export_browser(result,path),flush=True)
    print(path,flush=True)
