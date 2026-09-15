"""Physical populations and forward photometry of clusters and associated clouds.

The output is a visibility-oriented catalogue, not a complete galaxy database.
All parent objects are sampled before the unextinguished m=12 export cut.
This cut is well below the naked-eye point limit, and cannot remove a brighter
member: the total flux of positive sources is at least each member's flux.
"""
import math
import numpy as np
from scipy.optimize import brentq

from galaxy_environment import GalaxyParameters, GalacticDust, sample_disk, sample_halo
from cluster_population import Isochrones, stellar_type, ionizing_photons
from stellar_physics import PC_TO_LY, SOLAR_DIAMETER_MAS_AT_PC, cartesian_to_galactic
from nebula_morphology import validate_morphology


def power_law(rng,n,low,high,slope):
    p=1-slope
    return (low**p+rng.random(n)*(high**p-low**p))**(1/p)


def gaussian_recombination_fraction(x):
    """Volume integral of n(r)^2 for n=n0 exp(-r²/(2 sigma²))."""
    if x<.02:
        return 4/math.sqrt(math.pi)*(x**3/3-x**5/5+x**7/14-x**9/54)
    return math.erf(x)-2*x/math.sqrt(math.pi)*math.exp(-x*x)


def ionized_cloud(gas_mass_solar,sigma_pc,q):
    # Gas includes helium; one H nucleus per 1.4 m_H of gas. The snapshot
    # assumes ionized hydrogen, neutral helium and a static density profile.
    pc_cm=3.0856775814913673e18;alpha_b=2.59e-13
    n0=gas_mass_solar*1.98847e33/((2*math.pi)**1.5*(sigma_pc*pc_cm)**3*1.4*1.6735575e-24)
    capacity=alpha_b*n0*n0*math.pi**1.5*(sigma_pc*pc_cm)**3
    fraction=min(q/capacity,gaussian_recombination_fraction(3.))
    x=3. if q>=capacity*gaussian_recombination_fraction(3.) else brentq(lambda v:gaussian_recombination_fraction(v)-fraction,0.,3.,xtol=1e-13)
    mass_fraction=math.erf(x/math.sqrt(2))-math.sqrt(2/math.pi)*x*math.exp(-x*x/2)
    return dict(central_electron_density_cm3=n0,case_b_alpha_cm3_s=alpha_b,
        ionized_radius_pc=x*sigma_pc,profile_truncation=x,gas_mass_solar=gas_mass_solar,
        hydrogen_mass_per_atom_mH=1.4,ionized_gas_mass_solar=gas_mass_solar*max(0,mass_fraction),
        absorbed_fraction=capacity*fraction/q,recombination_capacity_s=capacity*gaussian_recombination_fraction(3.))


def project_object(obj,position,scale_pc,flux10,dust):
    d=float(np.linalg.norm(position));av=float(dust.extinction([position])[0])
    lon,lat=cartesian_to_galactic(*position)
    flux=float(flux10)*100/d**2*10**(-.4*av)
    theta=math.atan2(scale_pc,d)
    obj.update(pos_cartesian=list(map(float,position)),distance_pc=d,gal_lon=lon,gal_lat=lat,
               scale_pc=float(scale_pc),angular_scale_rad=theta,extinction_Av=av,
               intrinsic_v_flux_10pc=float(flux10),v_flux=flux,
               app_mag=-2.5*math.log10(flux) if flux>0 else None)
    return obj


def make_populations(rng,parameters=None,grid=None,population_scale=1.):
    p=parameters or GalaxyParameters();grid=grid or Isochrones()
    ages=[6.5,7.,7.5,8.,8.5,9.,9.5]
    # Explicit survival-weighted cohort prior, not a measured SFH for this galaxy.
    age_weights=[.02,.035,.08,.18,.30,.30,.085]
    parents=[];clouds=[];stats={'open_clusters':0,'globular_clusters':0,'living_stars':0,'post_grid_remnants':0}
    for kind,age,weight in [('open_cluster',a,w) for a,w in zip(ages,age_weights)]+[('globular_cluster',10.,1.)]:
        glob=kind=='globular_cluster';mh=-1. if glob else 0.
        n=int(rng.poisson((p.globular_cluster_expectation if glob else p.open_cluster_expectation*weight)*population_scale))
        stats['globular_clusters' if glob else 'open_clusters']+=n
        mass=(10**rng.normal(5.3,.45,n)) if glob else power_law(rng,n,50,50000,2)
        position=sample_halo(n,rng,p) if glob else sample_disk(n,min(300,55+45*(age-6.5)),max(0,.85-(age-6.5)*.4),rng,p)
        mid,weights,pars,missing=grid.bins(mh,age)
        flux=np.array([10**(-.4*s['abs_mag']) for s in pars])
        photons=np.array([ionizing_photons(s) for s in pars])
        live_mass=np.array([s['mass_solar'] for s in pars])
        for start in range(0,n,512):
            counts=rng.poisson(mass[start:start+512,None]*weights[None,:])
            remnant=rng.poisson(mass[start:start+512]*missing)
            totals=counts@flux; qs=counts@photons; masses=counts@live_mass
            for j,row in enumerate(counts):
                i=start+j; number=int(row.sum());stats['living_stars']+=number;stats['post_grid_remnants']+=int(remnant[j])
                identifier=f"{'GC' if glob else 'OC'}-{age:g}-{i:06d}"
                # Bound Plummer core with an explicit finite spatial truncation.
                scale=float((3.0 if glob else 1.5)*(max(float(masses[j]),10)/(2e5 if glob else 1000))**.15)
                if not glob and age<=7 and (age==6.5 or rng.random()<.2):
                    gas=float(mass[i]*(1-.15)/.15)
                    sig=math.sqrt(gas/(2*math.pi*50))
                    clouds.append(dict(id='D-'+identifier,pos_cartesian=position[i].tolist(),sigma_pc=sig,
                        gas_mass_solar=gas,central_extinction_Av=50/p.cloud_surface_mass_per_av,
                        source_cluster_id=identifier,ionizing_photons_s=float(qs[j]),source_v_flux_10pc=float(totals[j])))
                d=float(np.linalg.norm(position[i]))
                # Retain no source on the basis of a hand-chosen apparent brightness.
                # This is only a conservative post-generation output/storage cut.
                closest=max(1e-6,d-10*scale)
                if totals[j]<=0 or -2.5*math.log10(float(totals[j])*100/closest**2)>12:
                    continue
                parents.append(dict(id=identifier,kind=kind,log_age=age,metallicity_mh=mh,
                    initial_mass_expectation_solar=float(mass[i]),luminous_mass_solar=float(masses[j]),
                    member_count=number,post_grid_remnant_count=int(remnant[j]),
                    bin_counts=row.tolist(),pos_cartesian=position[i].tolist(),scale_pc=scale,
                    intrinsic_v_flux_10pc=float(totals[j]),ionizing_photons_s=float(qs[j])))
    n=int(rng.poisson(p.unlit_cloud_expectation*population_scale))
    mass=power_law(rng,n,1000,1e6,1.7)
    positions=sample_disk(n,65,.9,rng,p)
    for i in range(n):
        clouds.append(dict(id=f'D-U-{i:05d}',pos_cartesian=positions[i].tolist(),
            sigma_pc=math.sqrt(float(mass[i])/(2*math.pi*50)),gas_mass_solar=float(mass[i]),
            central_extinction_Av=50/p.cloud_surface_mass_per_av,source_cluster_id=None,
            ionizing_photons_s=0.,source_v_flux_10pc=0.))
    stats['dust_clouds']=len(clouds)
    return parents,GalacticDust(p,clouds),stats


def realize_visible(parents,dust,grid,rng,generation_id,limit=6.5):
    """Resolved members carry point flux; only the remainder enters the diffuse profile."""
    stars=[];objects=[];clusters=[]
    for parent in parents:
        mh,age=parent['metallicity_mh'],parent['log_age']
        mid,_,pars,_=grid.bins(mh,age)
        pos=np.asarray(parent['pos_cartesian']);d=np.linalg.norm(pos);a=parent['scale_pc']
        total_unresolved=parent['intrinsic_v_flux_10pc'];ids=[]
        # Sample only members which could be bright at the closest point of the
        # truncated Plummer sphere. Fainter members are analytically unresolved.
        for index,(number,par) in enumerate(zip(parent['bin_counts'],pars)):
            if not number or par['abs_mag']+5*math.log10(max(1,d-10*a)/10)>limit:
                continue
            u=rng.uniform(0,1000/(101**1.5),number)
            radii=a/np.sqrt(u**(-2/3)-1)
            directions=rng.normal(size=(number,3));directions/=np.linalg.norm(directions,axis=1)[:,None]
            positions=pos+radii[:,None]*directions
            distances=np.linalg.norm(positions,axis=1);avs=dust.extinction(positions)
            mags=par['abs_mag']+5*np.log10(distances/10)+avs
            for k in np.flatnonzero(mags<=limit):
                stype,lclass,color=stellar_type(par);lon,lat=cartesian_to_galactic(*positions[k])
                sid=f'{generation_id}_C{len(stars)+1:06d}'
                stars.append(dict(id=sid,cluster_id=parent['id'],stellar_model='PARSEC',
                    cluster_mass_bin=index,
                    initial_mass_solar=float(mid[index]),log_age=age,metallicity_mh=mh,**par,
                    spectral_type=stype,luminosity_class=lclass,color_hex=color,
                    pos_cartesian=positions[k].tolist(),distance_pc=float(distances[k]),dist_ly=float(distances[k]*PC_TO_LY),
                    gal_lon=lon,gal_lat=lat,extinction_Av=float(avs[k]),app_mag=float(mags[k]),
                    angular_diameter_mas=SOLAR_DIAMETER_MAS_AT_PC*par['radius_solar']/float(distances[k])))
                ids.append(sid);total_unresolved-=10**(-.4*par['abs_mag'])
        total_unresolved=max(0.,float(total_unresolved))
        obj=project_object(dict(id=parent['id'],kind=parent['kind'],profile='plummer',name=parent['id'],
             member_count=parent['member_count'],resolved_member_ids=ids,color='#cbd5e5'),pos,a,total_unresolved,dust)
        # For Plummer surface density, half of the untruncated flux is inside a.
        obj['profile_truncation']=10.
        clusters.append({**parent,'resolved_member_ids':ids,'unresolved_v_flux_10pc':total_unresolved})
        if obj['v_flux']>0:objects.append(obj)
    # The source budget is tied to the SAME young stellar population.
    for cloud in dust.clouds:
        pos=np.asarray(cloud['pos_cartesian']);sig=cloud['sigma_pc'];q=cloud['ionizing_photons_s']
        base=dict(id=cloud['id'],name=cloud['id'],kind='dark_nebula',profile='gaussian',
                  source_cluster_id=cloud['source_cluster_id'],color='#161c24')
        dark=project_object(base,pos,sig,0,dust);dark['central_extinction_Av']=cloud['central_extinction_Av']
        objects.append(dark)
        if q>1e43:
            ionization=ionized_cloud(cloud['gas_mass_solar'],sig,q)
            # Case-B H-beta; V response-weighted [O III]/H-beta prior. This is
            # not a full photoionization spectrum or narrow-band image model.
            hbeta=4.76e-13*q*ionization['absorbed_fraction']  # erg/s
            line_ratio=3.0
            v_watt=hbeta*1e-7*(.12+.42*line_ratio)
            flux10=v_watt/(4*math.pi*(10*3.085677581491367e16)**2)/3.2e-9
            objects.append(project_object(dict(id='H-'+cloud['id'],name='H-'+cloud['id'],kind='emission_nebula',
                profile='ionized_gaussian',color='#c6d2cf',source_cluster_id=cloud['source_cluster_id'],
                ionizing_photons_s=q,**ionization,hbeta_luminosity_erg_s=hbeta,
                oiii_hbeta_ratio=line_ratio),pos,sig,flux10,dust))
        if cloud['source_v_flux_10pc']>0:
            scattered=.6*(1-math.exp(-cloud['central_extinction_Av']/2/1.085736205))
            objects.append(project_object(dict(id='R-'+cloud['id'],name='R-'+cloud['id'],kind='reflection_nebula',
                profile='gaussian',color='#c3cee0',source_cluster_id=cloud['source_cluster_id'],
                scattering_albedo=.6,intercepted_scattered_fraction=scattered),pos,sig,
                cloud['source_v_flux_10pc']*scattered,dust))
    # Dark cloud extinction is already in every point source and source screen.
    # It is not drawn as a black decal over foreground airglow or nearby stars.
    objects=[o for o in objects if o['kind']=='dark_nebula' or o['app_mag']<=12]
    return stars,objects,clusters


def validate_deep_sky(catalog,grid):
    errors=[];data=catalog.get('deep_sky',{});stars={s['id']:s for s in catalog['stars']}
    dust=GalacticDust.from_dict(catalog['galaxy'])
    objects=data.get('objects',[])
    expected_av=dust.extinction([o['pos_cartesian'] for o in objects]) if objects else []
    all_ids=set();member_ids=set()
    for c in data.get('clusters',[]):
        try:
            mid,_,pars,_=grid.bins(c['metallicity_mh'],c['log_age'])
            counts=c['bin_counts']
            if len(counts)!=len(mid) or any(not isinstance(n,int) or n<0 for n in counts):raise ValueError('无效 IMF 分箱计数')
            total=sum(n*10**(-.4*p['abs_mag']) for n,p in zip(counts,pars))
            if sum(counts)!=c['member_count'] or c['id'] in all_ids:raise ValueError('星团成员数或 ID 不一致')
            all_ids.add(c['id'])
            resolved=0.;used=np.zeros(len(mid),dtype=int)
            for sid in c['resolved_member_ids']:
                if sid in member_ids or stars[sid].get('cluster_id')!=c['id']:raise ValueError('星团成员归属或重复计数错误')
                star=stars[sid];index=star['cluster_mass_bin']
                if not isinstance(index,int) or not 0<=index<len(mid):raise ValueError('无效成员初始质量分箱')
                used[index]+=1
                if not math.isclose(star['initial_mass_solar'],mid[index],rel_tol=1e-12):raise ValueError('成员与质量分箱不匹配')
                if star['log_age']!=c['log_age'] or star['metallicity_mh']!=c['metallicity_mh']:raise ValueError('星团非共龄同金属丰度')
                if math.dist(star['pos_cartesian'],c['pos_cartesian'])>10*c['scale_pc']*(1+1e-9):raise ValueError('成员超出截断半径')
                member_ids.add(sid);resolved+=10**(-.4*star['abs_mag'])
            if np.any(used>counts):raise ValueError('已分辨成员超过生成的成员数')
            if not math.isclose(total,c['unresolved_v_flux_10pc']+resolved,rel_tol=1e-9,abs_tol=1e-10):
                raise ValueError('已分辨和未分辨光量不守恒')
        except (KeyError,TypeError,ValueError) as exc:errors.append(f"星团 {c.get('id')}: {exc}")
    if member_ids!={s['id'] for s in stars.values() if s.get('cluster_id')}:errors.append('有未登记的星团成员')
    ids=set()
    for obj,av in zip(objects,expected_av):
        try:
            validate_morphology(obj.get('morphology'),obj['kind'])
            if obj['id'] in ids:raise ValueError('对象 ID 重复')
            ids.add(obj['id']);d=math.dist(obj['pos_cartesian'],[0,0,0])
            if not math.isclose(obj['distance_pc'],d,rel_tol=1e-10):raise ValueError('距离错误')
            if not math.isclose(obj['extinction_Av'],av,rel_tol=1e-9,abs_tol=1e-10):raise ValueError('尘埃路径消光错误')
            if obj['v_flux']>0 and not math.isclose(obj['app_mag'],-2.5*math.log10(obj['v_flux']),rel_tol=1e-10,abs_tol=1e-10):raise ValueError('视星等错误')
            if not math.isclose(obj['angular_scale_rad'],math.atan2(obj['scale_pc'],d),rel_tol=1e-10):raise ValueError('角尺度错误')
            if not math.isclose(obj['v_flux'],obj['intrinsic_v_flux_10pc']*100/d**2*10**(-.4*obj['extinction_Av']),rel_tol=1e-10):raise ValueError('距离和消光后的光量错误')
            if obj['kind']=='emission_nebula':
                cloud=next(c for c in dust.clouds if c['id']=='D-'+obj['source_cluster_id'])
                if not math.isclose(obj['ionizing_photons_s'],cloud['ionizing_photons_s'],rel_tol=1e-10):raise ValueError('电离光子预算与星群不一致')
                expected=ionized_cloud(cloud['gas_mass_solar'],cloud['sigma_pc'],cloud['ionizing_photons_s'])
                for key,value in expected.items():
                    if not math.isclose(obj[key],value,rel_tol=1e-10,abs_tol=1e-12):raise ValueError('电离范围与气体质量密度不一致: '+key)
                hbeta=4.76e-13*obj['ionizing_photons_s']*obj['absorbed_fraction']
                if not math.isclose(hbeta,obj['hbeta_luminosity_erg_s'],rel_tol=1e-10):raise ValueError('复合线光度错误')
                flux10=hbeta*1e-7*(.12+.42*obj['oiii_hbeta_ratio'])/(4*math.pi*(10*3.085677581491367e16)**2)/3.2e-9
                if not math.isclose(flux10,obj['intrinsic_v_flux_10pc'],rel_tol=1e-10):raise ValueError('星云 V 光量预算不一致')
        except (KeyError,TypeError,ValueError,StopIteration) as exc:errors.append(f"深空对象 {obj.get('id')}: {exc}")
    return errors
