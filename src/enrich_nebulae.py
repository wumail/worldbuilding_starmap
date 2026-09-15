#!/usr/bin/env python3
"""Add seeded emissivity morphology to a copied catalogue, preserving its stars."""
import argparse
import hashlib
import json
import tempfile
from pathlib import Path
from collections import Counter
from nebula_morphology import add_morphologies,MODEL
from star_generator import validate_catalog,generate_folders_json
from export_deep_sky import export_browser


def enrich(source, generation_id, output_root):
    source=Path(source);raw=source.read_bytes();catalog=json.loads(raw)
    original_id=catalog['metadata']['generation_id'];original_revision=catalog['metadata'].get('render_revision')
    catalog['deep_sky']['morphology']=add_morphologies(catalog['deep_sky']['objects'],catalog['metadata']['generation_parameters']['seed'])
    # This is a new rendering revision of the SAME physical realization.
    # Keep its generation_id: stellar IDs and member ledgers refer to it.
    catalog['metadata']['render_revision']=generation_id
    catalog['metadata']['derived_from']=dict(generation_id=original_id,render_revision=original_revision,scientific_catalogue=source.name,sha256=hashlib.sha256(raw).hexdigest(),
        operation=f'Set {MODEL} morphology only; retained original stellar IDs and all physical values',
        morphology_source_sha256=hashlib.sha256(Path(__file__).with_name('nebula_morphology.py').read_bytes()).hexdigest())
    validation=validate_catalog(catalog)
    if not validation['all_passed']:raise ValueError(validation['errors'][:3])
    if not generation_id.replace('_','').isalnum():raise ValueError('形态批次名称无效')
    root=Path(output_root);root.mkdir(parents=True,exist_ok=True)
    folder=root/f'output_{generation_id}';destination=folder/f'star_map_{generation_id}.json'
    if folder.exists():raise FileExistsError(f'已有数据集，拒绝覆盖: {folder}')
    with tempfile.TemporaryDirectory(prefix='.nebula-',dir=root) as temporary:
        pending=Path(temporary)/folder.name;pending.mkdir()
        target=pending/destination.name
        target.write_text(json.dumps(catalog,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
        report=export_browser(catalog,target,register=False)
        (pending/'morphology_validation.json').write_text(json.dumps(validation,ensure_ascii=False,indent=2)+'\n')
        pending.rename(folder)
    index_path=root/'catalog_views.json';index=json.loads(index_path.read_text()) if index_path.exists() else {}
    index[folder.name]=f'sky_view_{generation_id}.json';index_path.write_text(json.dumps(index,indent=2)+'\n')
    generate_folders_json(root)
    return dict(path=str(destination),stars=len(catalog['stars']),
        morphologies=dict(Counter(o['morphology']['family'] for o in catalog['deep_sky']['objects'] if o.get('morphology'))),raster=report)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('source',type=Path);p.add_argument('--generation-id',required=True);p.add_argument('--output-root',type=Path,default=Path(__file__).resolve().parents[1]/'output')
    a=p.parse_args();print(json.dumps(enrich(a.source,a.generation_id,a.output_root),ensure_ascii=False,indent=2))
