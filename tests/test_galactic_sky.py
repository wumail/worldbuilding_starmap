import copy
import json
import math
from pathlib import Path
import sys
import unittest
import numpy as np
from scipy.integrate import quad

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from galaxy_environment import GalaxyParameters,GalacticDust,sample_disk
from cluster_population import Isochrones,imf_integral
from deep_sky import validate_deep_sky,ionized_cloud
from export_deep_sky import rasterize
from star_generator import GenerationConfig,validate_catalog

class GalacticScienceTests(unittest.TestCase):
    def test_canonical_radius_not_old_solar_radius(self):
        self.assertAlmostEqual(GalaxyParameters().observer_radius_pc*3.2615637771674333,30712,places=8)
        self.assertEqual(GenerationConfig().observer_radius_pc,GalaxyParameters().observer_radius_pc)

    def test_disk_area_measure_and_exponential_height(self):
        p=GalaxyParameters();pos=sample_disk(100000,100,0,np.random.default_rng(58),p)
        r=np.hypot(p.observer_radius_pc-pos[:,0],pos[:,1]);cdf=lambda x:1-(1+x/p.disk_scale_pc)*np.exp(-x/p.disk_scale_pc)
        for cut in [4000,8000,12000]:
            expected=(cdf(cut)-cdf(p.inner_star_formation_radius_pc))/(cdf(p.disk_radius_pc)-cdf(p.inner_star_formation_radius_pc))
            self.assertLess(abs(np.mean(r<cut)-expected),.006)
        self.assertLess(abs(np.mean(abs(pos[:,2]+p.observer_height_pc))-100),1.3)

    def test_cloud_foreground_center_and_background_column(self):
        cloud=dict(pos_cartesian=[100.,0.,0.],sigma_pc=2.,central_extinction_Av=4.)
        d=GalacticDust(clouds=[cloud]);values=d.cloud_extinction([[50,0,0],[100,0,0],[150,0,0],[150,30,0]])
        np.testing.assert_allclose(values[:3],[0,2,4],atol=1e-10)
        self.assertLess(values[3],1e-15)
        np.testing.assert_allclose(d.cloud_extinction([[100,0,0]]),values[1:2],atol=1e-12)

    def test_dust_against_independent_adaptive_integrator(self):
        d=GalacticDust();p=d.parameters
        for v in map(np.array,[[20000,0,0],[20000,20,0],[0,0,5000],[8000,30,-20]]):
            f=lambda t:np.exp((p.observer_radius_pc-np.hypot(p.observer_radius_pc-v[0]*t,v[1]*t))/p.dust_scale_pc+(abs(p.observer_height_pc)-abs(p.observer_height_pc+v[2]*t))/p.dust_height_pc)*np.linalg.norm(v)*p.local_diffuse_av_per_kpc/1000
            point=np.clip(p.observer_radius_pc*v[0]/max(1,v[0]**2+v[1]**2),0,1)
            ref=quad(f,0,1,epsabs=1e-9,points=[point])[0]
            self.assertLess(abs(d.smooth([v])[0]-ref),1e-6)

    def test_isochrones_imf_and_stefan_boltzmann(self):
        grid=Isochrones()
        reference=quad(lambda m:m*(m**-1.3 if m<.5 else .5*m**-2.3),.09,100,points=[.5])[0]
        self.assertAlmostEqual(float(imf_integral(.09,100,1)),reference,places=9)
        for key in grid.tracks:
            mid,weight,pars,missing=grid.bins(*key)
            self.assertTrue(np.all(np.diff(mid)>0));self.assertTrue(np.all(weight>=0));self.assertGreaterEqual(missing,0)
            for par in pars:
                self.assertAlmostEqual(par['luminosity_solar'],par['radius_solar']**2*(par['temperature_K']/5772)**4,delta=par['luminosity_solar']*1e-10)
        self.assertEqual(grid.sha256,'ace5bbc63648bf01d63b4e252be1ad6baa369c0d5304443789641c2b1c9c4685')

    def test_ionization_uses_available_gaussian_gas_and_recombination_integral(self):
        for gas,sigma,q in [(283333.33,30.0313,3.18743e50),(2000,2.5,1e50),(1000,3,1e44)]:
            c=ionized_cloud(gas,sigma,q);n0=c['central_electron_density_cm3'];r=c['ionized_radius_pc'];pc=3.0856775814913673e18
            recombinations=4*math.pi*c['case_b_alpha_cm3_s']*n0*n0*pc**3*quad(lambda x:x*x*math.exp(-(x/sigma)**2),0,r,epsabs=1e-12)[0]
            self.assertAlmostEqual(recombinations/(q*c['absorbed_fraction']),1,places=8)
            enclosed=gas*quad(lambda x:4*math.pi*x*x*math.exp(-x*x/(2*sigma*sigma))/(2*math.pi)**1.5/sigma**3,0,r)[0]
            self.assertAlmostEqual(enclosed,c['ionized_gas_mass_solar'],delta=max(1e-8,gas*1e-10))
            self.assertLessEqual(c['ionized_gas_mass_solar'],gas)
            self.assertLessEqual(c['absorbed_fraction'],1+1e-12)

    def test_diffuse_flux_conserved_at_wrap_pole_and_subpixel_scales(self):
        for lon,lat,scale,profile in [(359.99,0,.03,'gaussian'),(.01,89.9,.05,'plummer'),(42,-60,1e-6,'sphere')]:
            light,report=rasterize([dict(v_flux=.1,gal_lon=lon,gal_lat=lat,angular_scale_rad=scale,profile=profile)],256,128)
            self.assertLess(report['relative_error'],1e-7);self.assertGreater(light.max(),0)
            if profile=='gaussian':self.assertGreater(light[:,0].sum(),0);self.assertGreater(light[:,-1].sum(),0)
        dark,report=rasterize([dict(v_flux=0)],64,32);self.assertFalse(dark.any())

class DeliveredCatalogueTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path=ROOT/'output/output_20260915_galactic_01/star_map_20260915_galactic_01.json'
        if not path.is_file():
            raise unittest.SkipTest('完整科学星表未生成，跳过成品核验')
        cls.catalog=json.loads(path.read_text())
    def test_whole_realization_count_physics_and_source_ledger(self):
        c=self.catalog;self.assertTrue(9000<=len(c['stars'])<=9500)
        self.assertEqual(c['metadata']['count_selection']['mode'],'whole_catalog_rejection')
        report=validate_catalog(c);self.assertTrue(report['all_passed'],report['errors'][:2])
    def test_detect_member_overdraw_and_dust_tampering(self):
        c=self.catalog;cluster=next(p for p in c['deep_sky']['clusters'] if p['resolved_member_ids'])
        members=[s for s in c['stars'] if s.get('cluster_id')==cluster['id']]
        obj=next(o for o in c['deep_sky']['objects'] if o['id']==cluster['id'])
        test={'stars':copy.deepcopy(members),'galaxy':c['galaxy'],'deep_sky':{'clusters':[copy.deepcopy(cluster)],'objects':[copy.deepcopy(obj)]}}
        self.assertEqual(validate_deep_sky(test,Isochrones()),[])
        test['deep_sky']['objects'][0]['extinction_Av']+=.2
        self.assertTrue(any('消光' in e for e in validate_deep_sky(test,Isochrones())))
        test['stars'][0]['log_age']+=.5
        self.assertTrue(any('共龄' in e for e in validate_deep_sky(test,Isochrones())))

if __name__=='__main__':unittest.main()
