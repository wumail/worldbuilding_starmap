import copy
import json
import math
from pathlib import Path
import sys
import unittest
import numpy as np
from scipy.integrate import quad
from scipy.ndimage import gaussian_filter

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from nebula_morphology import morphology_for,add_morphologies,projected_weight,raw_projected_weight,validate_morphology,FAMILIES,MODEL,LEGACY_MODEL,soft_shell_column,FILTER_SIGMA,smoothstep
from export_deep_sky import rasterize


class NebulaMorphologyTests(unittest.TestCase):
    def test_seeded_cloud_structure_preserves_physical_fields_and_pairs(self):
        objects=json.loads((ROOT/'output/output_20260915_galactic_01/sky_view_20260915_galactic_01.json').read_text())['deep_sky']['objects']
        before=copy.deepcopy(objects);add_morphologies(objects,20260915)
        changed=[o for o in objects if 'morphology' in o]
        self.assertEqual(len(changed),160)
        self.assertEqual(set(o['morphology']['family'] for o in changed),set(FAMILIES))
        for old,new in zip(before,objects):
            m=new.get('morphology');self.assertEqual(old,{k:v for k,v in new.items() if k!='morphology'})
            self.assertEqual(m,morphology_for(new,20260915))
            if m:
                validate_morphology(m,new['kind']);self.assertNotEqual(m,morphology_for(new,2))
        pairs={o['id']:o for o in changed}
        for o in changed:
            if o['kind']=='emission_nebula' and 'R'+o['id'][1:] in pairs:
                other=pairs['R'+o['id'][1:]]['morphology']
                for key in ('position_angle_rad','phase_rad'):self.assertEqual(o['morphology'][key],other[key])

    def test_shell_is_a_luminous_projected_shell_not_a_transparent_black_cutout(self):
        m=morphology_for(dict(id='fixture',kind='emission_nebula'),42)
        m.update(family='shell',axis_ratio=1,turbulence=0,phase_rad=0)
        # Near + far sides emit along a central sightline. Limb brightening
        # must nevertheless exceed the central column through the shell.
        values=projected_weight(np.linspace(0,1,1001),np.zeros(1001),'ionized_gaussian',1,m)
        self.assertGreater(values[0],0);self.assertGreater(values.max(),values[0]*1.3)
        self.assertEqual(values[-1],0)

    def test_soft_shell_matches_independent_adaptive_line_integral(self):
        # Independent adaptive integration of the radial emissivity, including
        # near/far material. Covers the allowed thicknesses and outer taper.
        for thickness in np.linspace(.15,.4,6):
            for impact in np.linspace(0,1,81):
                end=math.sqrt(max(0,1-impact*impact))
                split=math.sqrt(max(0,.68**2-impact*impact))
                def emissivity(z):
                    radius=math.hypot(impact,z)
                    u=min(1,max(0,(radius-.68)/.32))
                    return 2*math.exp(-.5*((radius-.60)/(.105+.10*thickness))**2)*(1-3*u*u+2*u**3)
                expected=quad(emissivity,0,end,points=[split] if split else None,epsabs=1e-12,epsrel=1e-12)[0]
                self.assertAlmostEqual(float(soft_shell_column(impact,thickness)),expected,delta=2e-6)

    def test_soft_edge_loses_the_sharp_uniform_shell_interface(self):
        m=morphology_for(dict(id='fixture',kind='emission_nebula'),42)
        m.update(family='shell',axis_ratio=1,turbulence=0)
        r=np.linspace(0,1,10001);slopes=[]
        for model in (LEGACY_MODEL,MODEL):
            m['model']=model
            v=projected_weight(r,np.zeros_like(r),'ionized_gaussian',1,m)
            slopes.append(float(np.abs(np.gradient(v/v.max(),r))[r>.65].max()))
        self.assertLess(slopes[1],slopes[0]*.12)

    def test_upgrade_keeps_existing_families_orientations_and_noise_seeds(self):
        old=json.loads((ROOT/'output/output_20260915_nebula_01/sky_view_20260915_nebula_01.json').read_text())
        objects=copy.deepcopy(old['deep_sky']['objects'])
        add_morphologies(objects,old['metadata']['generation_parameters']['seed'])
        for before,after in zip(old['deep_sky']['objects'],objects):
            if 'morphology' not in before:
                self.assertEqual(before,after)
                continue
            self.assertEqual(before['morphology']['model'],LEGACY_MODEL)
            expected=copy.deepcopy(before);expected['morphology']['model']=MODEL
            self.assertEqual(expected,after)

    def test_morphology_is_nonnegative_finite_and_flux_conserving_at_seams_poles(self):
        m=morphology_for(dict(id='fixture',kind='emission_nebula'),42)
        xy=np.linspace(-4.1,4.1,513);x,y=np.meshgrid(xy,xy)
        for family in FAMILIES:
            m['family']=family
            for profile,cut in [('gaussian',4),('ionized_gaussian',.19),('ionized_gaussian',3)]:
                values=projected_weight(x,y,profile,cut,m)
                self.assertTrue(np.all(np.isfinite(values)));self.assertGreaterEqual(values.min(),0);self.assertGreater(values.max(),0)
            for lon,lat,a in [(359.99,0,.02),(0,89.99,.01),(300,-89.9,.01),(120,0,1e-6)]:
                o=dict(kind='emission_nebula',profile='ionized_gaussian',profile_truncation=.7,
                    gal_lon=lon,gal_lat=lat,angular_scale_rad=a,v_flux=.01,morphology=m)
                image,report=rasterize([o],256,128)
                self.assertLess(report['relative_error'],1e-7);self.assertGreater(image.max(),0)

    def test_invalid_morphology_is_rejected(self):
        m=morphology_for(dict(id='fixture',kind='emission_nebula'),42)
        for change in [dict(axis_ratio=0),dict(phase_rad=math.nan),dict(family='unknown'),dict(model='future')]:
            with self.assertRaises(ValueError):validate_morphology({**m,**change},'emission_nebula')
        with self.assertRaises(ValueError):validate_morphology(m,'open_cluster')
        with self.assertRaises(ValueError):validate_morphology({**m,'family':'shell'},'reflection_nebula')

    def test_filter_is_converged_against_a_twice_finer_independent_grid(self):
        # A separate finer grid and SciPy's Gaussian filter check that the
        # stored angular field is resolved; flux normalization alone cannot.
        n=513;axis=np.linspace(-1,1,n);x,y=np.meshgrid(axis,axis)
        m=morphology_for(dict(id='fixture',kind='emission_nebula'),42)
        for family in FAMILIES:
            m['family']=family
            raw=raw_projected_weight(x,y,'ionized_gaussian',1,m)
            reference=gaussian_filter(raw,FILTER_SIGMA*(n-1)/2,mode='constant',truncate=4)
            reference*=1-smoothstep(.85,1,np.hypot(x,y))
            reference*=1-smoothstep(.99,1,np.hypot(x,y))
            actual=projected_weight(x,y,'ionized_gaussian',1,m)
            self.assertLess(float(np.abs(reference-actual).max()/reference.max()),.003,family)
            # Smooth the full noisy emission, reducing fine edges instead of
            # reintroducing them by multiplying noise after the filter.
            curvature=lambda a:float(np.mean(np.abs(np.diff(a,n=2,axis=0)))+np.mean(np.abs(np.diff(a,n=2,axis=1))))
            self.assertLess(curvature(reference),curvature(raw),family)


if __name__=='__main__':unittest.main()
