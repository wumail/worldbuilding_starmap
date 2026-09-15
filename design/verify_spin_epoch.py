"""Audit the agreed epoch inclinations and sampled ten-body convergence.

This post-processes saved trajectories; it does not prove continuous-time
closest approaches, stability, or convergence beyond the available samples.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import terrax_spin_candidate as model


def angle_deg(a, b):
    return np.degrees(np.arctan2(np.linalg.norm(np.cross(a, b), axis=-1), np.sum(a*b, axis=-1)))


def orbit(y, index):
    p=y[:, :30].reshape(-1, 10, 3)
    v=y[:, 30:60].reshape(-1, 10, 3)
    r=p[:, index]-p[:, model.HOME]
    speed=v[:, index]-v[:, model.HOME]
    radius=np.linalg.norm(r, axis=1)
    h=np.cross(r, speed)
    mu=model.G*(model.MASS[model.HOME]+model.MASS[index])
    eccentricity=np.cross(speed, h)/mu-r/radius[:, None]
    energy=np.sum(speed*speed, axis=1)/2-mu/radius
    return dict(r=r, radius=radius, h=h, eccentricity=eccentricity, a=-mu/(2*energy))


def audit(reference_path, comparison_path):
    ref=np.load(reference_path)
    other=np.load(comparison_path)
    t=ref['t'];y=ref['y'];z=other['y']
    if not np.array_equal(t, other['t']):
        raise ValueError('Use trajectories with identical output times; no interpolation in this audit.')
    if y.shape!=(len(t), 63) or z.shape!=y.shape or not np.all(np.isfinite(y)) or not np.all(np.isfinite(z)):
        raise ValueError('Invalid trajectory shape or non-finite values.')
    moons=[orbit(y, i) for i in (8, 9)]
    previous=[orbit(z, i) for i in (8, 9)]
    spin=y[:, 60:63]
    initial=model.initial()[None, :]
    epoch={model.NAMES[i]:float(angle_deg(orbit(initial, i)['h'][0], initial[0, 60:63])) for i in (8, 9)}
    if not np.allclose(list(epoch.values()), [.4, 1.1], rtol=0, atol=1e-9):
        raise AssertionError('Initial inclinations disagree with the selected epoch convention.')
    if not np.allclose(y[0], initial[0], rtol=0, atol=1e-14):
        raise AssertionError('Saved trajectory does not start at the current candidate initial state.')
    geometry={}
    for i, m in zip((8, 9), moons):
        geometry[model.NAMES[i]]={
            'sampled_equatorial_inclination_deg':[float(x) for x in (angle_deg(m['h'], spin).min(), angle_deg(m['h'], spin).max())],
            'sampled_radius_km':[float(x*model.AU/1000) for x in (m['radius'].min(), m['radius'].max())],
            'sampled_osculating_eccentricity':[float(x) for x in (np.linalg.norm(m['eccentricity'],axis=1).min(), np.linalg.norm(m['eccentricity'],axis=1).max())],
        }
    separation=np.linalg.norm(moons[1]['r']-moons[0]['r'], axis=1)
    nearest=int(np.argmin(separation))
    hill=((model.MASS[8]+model.MASS[9])/(3*model.MASS[model.HOME]))**(1/3)*(moons[0]['a']+moons[1]['a'])/2
    comparisons=[]
    for year in (0, 1, 10, 50, 75, 100, 110, 120, 125, 130, 135, float(t[-1]/model.YEAR)):
        k=int(np.argmin(abs(t-year*model.YEAR)))
        row={'year':float(t[k]/model.YEAR),'axis_difference_arcsec':float(angle_deg(y[k,60:63], z[k,60:63])*3600),'moons':{}}
        for i,m,n in zip((8,9),moons,previous):
            row['moons'][model.NAMES[i]]={
                'position_difference_km':float(np.linalg.norm(m['r'][k]-n['r'][k])*model.AU/1000),
                'sky_direction_difference_deg':float(angle_deg(m['r'][k],n['r'][k])),
                'orbital_normal_difference_deg':float(angle_deg(m['h'][k],n['h'][k])),
                'radial_difference_km':float(abs(m['radius'][k]-n['radius'][k])*model.AU/1000),
                'eccentricity_vector_difference':float(np.linalg.norm(m['eccentricity'][k]-n['eccentricity'][k])),
            }
        comparisons.append(row)
    return {
        'selection':'moon inclinations are osculating elements at day zero relative to the initial Terrax equator',
        'initial_inclinations_deg':epoch,
        'source_sha256':hashlib.sha256(Path(model.__file__).read_bytes()).hexdigest(),
        'trajectory_sha256':{name:hashlib.sha256(Path(path).read_bytes()).hexdigest() for name,path in [('reference',reference_path),('comparison',comparison_path)]},
        'sample_count':len(t),'max_sample_interval_days':float(np.diff(t).max()),
        'geometry':geometry,
        'sampled_closest_moon_separation':{
            'year':float(t[nearest]/model.YEAR),'distance_km':float(separation[nearest]*model.AU/1000),
            'instantaneous_mutual_Hill_radii_diagnostic_only':float(separation[nearest]/hill[nearest]),
            'minimum_is_an_upper_bound_on_true_continuous_minimum':True,
        },
        'convergence_samples':comparisons,
        'max_axis_difference_arcsec':float(angle_deg(y[:,60:63],z[:,60:63]).max()*3600),
        'scope':'Sampled diagnostics only. Far-future lunar ephemerides remain unconverged; no chaos or long-term stability claim.',
    }


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--comparison',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=audit(args.reference,args.comparison)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({key:result[key] for key in ('initial_inclinations_deg','geometry','sampled_closest_moon_separation','max_axis_difference_arcsec')}))
