"""Flux-conserving angular raster for diffuse light, separate from point stars.

The finite grid stores pixel-averaged luminance. Small unresolved profiles
deposit their exact integrated flux; sharp resolved stars never enter this map.
"""
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from scipy.special import erf
from nebula_morphology import projected_weight,validate_morphology

def rasterize(objects,width=2048,height=1024):
    lat=(np.arange(height)+.5)*math.pi/height-math.pi/2
    # Exact solid angle of a longitude/latitude cell, including polar cells.
    omega=2*math.pi/width*(np.sin((np.arange(height)+1)*math.pi/height-math.pi/2)-np.sin(np.arange(height)*math.pi/height-math.pi/2))
    light=np.zeros((height,width),dtype=np.float64);input_flux=0.
    for obj in objects:
        if obj['v_flux']<=0:continue
        flux=obj['v_flux']*2.54e-6;input_flux+=flux
        lon=math.radians(obj['gal_lon']);b=math.radians(obj['gal_lat']);a=obj['angular_scale_rad']
        kind=obj['profile'];cut=obj['profile_truncation'] if kind=='ionized_gaussian' else 10 if kind=='plummer' else 4 if kind=='gaussian' else 1
        radius=min(math.pi,math.atan(math.tan(a)*cut))
        y0=max(0,int((b-radius+math.pi/2)/math.pi*height)-1);y1=min(height,int((b+radius+math.pi/2)/math.pi*height)+2)
        maxlon=math.pi if abs(b)+radius>=math.pi/2 else math.asin(min(1,math.sin(radius)/math.cos(b)))
        xp=np.arange(math.floor((lon-maxlon)/(2*math.pi)*width)-1,math.ceil((lon+maxlon)/(2*math.pi)*width)+1)
        xp=np.unique(xp%width);yp=np.arange(y0,y1)
        l=(xp+.5)*2*math.pi/width
        cosine=np.sin(lat[yp,None])*math.sin(b)+np.cos(lat[yp,None])*math.cos(b)*np.cos(l[None,:]-lon)
        theta=np.arccos(np.clip(cosine,-1,1));u=np.tan(np.minimum(theta,math.pi/2-1e-8))/math.tan(a)
        weights=(1+u*u)**-2 if kind=='plummer' else np.exp(-u*u/2) if kind=='gaussian' else np.sqrt(np.maximum(0,1-u*u))
        if kind=='ionized_gaussian':weights=np.exp(-u*u)*erf(np.sqrt(np.maximum(0,cut*cut-u*u)))
        if kind=='plummer':
            # Exact line integral of the same 3D Plummer density truncated
            # at 10a that supplies the explicitly positioned member stars.
            z=np.sqrt(np.maximum(0,cut*cut-u*u));weights*=z*(3+u*u+2*cut*cut)/(2*(1+cut*cut)**1.5)
        if obj.get('morphology'):
            m=obj['morphology'];validate_morphology(m,obj['kind'])
            # East/north tangent coordinates, then the stored position angle.
            east=np.cos(lat[yp,None])*np.sin(l[None,:]-lon)
            north=np.sin(lat[yp,None])*math.cos(b)-np.cos(lat[yp,None])*math.sin(b)*np.cos(l[None,:]-lon)
            pa=m['position_angle_rad'];den=np.maximum(cosine,1e-12)*math.tan(a)
            x=(east*math.cos(pa)+north*math.sin(pa))/den
            y=(-east*math.sin(pa)+north*math.cos(pa))/den
            weights=projected_weight(x,y,kind,cut,m)
        weights=np.where(theta<=radius,weights,0)*omega[yp,None]
        total=weights.sum()
        if total>0:light[np.ix_(yp,xp)]+=flux*weights/total/omega[yp,None]
        else:
            y=min(height-1,max(0,int((b+math.pi/2)/math.pi*height)));x=int(lon/(2*math.pi)*width)%width
            light[y,x]+=flux/omega[y]
    stored=light.astype('<f4');output=float(np.sum(stored*omega[:,None]))
    return stored,dict(input_flux_lux=input_flux,output_flux_lux=output,
        relative_error=abs(output-input_flux)/max(input_flux,1e-30),width=width,height=height,
        angular_pixel_arcmin=21600/width,profile_cutoffs={'plummer_scale':10,'gaussian_sigma':4,'ionized_gaussian':'mass-consistent ionization front, at most 3 sigma','sphere_radius':1})

def export_browser(catalog,path,*,register=True):
    path=Path(path);folder=path.parent;gid=catalog['metadata'].get('render_revision') or catalog['metadata']['generation_id']
    light,report=rasterize(catalog['deep_sky']['objects'])
    name=f'diffuse_{gid}.f32';(folder/name).write_bytes(light.tobytes())
    report.update(file=name,format='little-endian float32 luminance cd/m2, Galactic longitude left-to-right; latitude -90 to +90 bottom-to-top',sha256=hashlib.sha256(light.tobytes()).hexdigest())
    # Keep the full scientific ledger in star_map_*.json, and avoid making a
    # browser parse tens of thousands of IMF tables just to display the sky.
    view={k:v for k,v in catalog.items() if k not in ('galaxy','deep_sky')}
    view['deep_sky']={'model':catalog['deep_sky']['model'],'objects':catalog['deep_sky']['objects'],
                      'population_stats':catalog['deep_sky']['population_stats'],'raster':report}
    if 'morphology' in catalog['deep_sky']:view['deep_sky']['morphology']=catalog['deep_sky']['morphology']
    view['metadata']={**view['metadata'],'scientific_catalogue':path.name}
    browser_name=f'sky_view_{gid}.json';(folder/browser_name).write_text(json.dumps(view,separators=(',',':'),ensure_ascii=False))
    if register:
        index_path=folder.parent/'catalog_views.json';index=json.loads(index_path.read_text()) if index_path.exists() else {}
        index[folder.name]=browser_name;index_path.write_text(json.dumps(index,indent=2)+'\n')
    (folder/'diffuse_validation.json').write_text(json.dumps(report,indent=2)+'\n')
    return report
