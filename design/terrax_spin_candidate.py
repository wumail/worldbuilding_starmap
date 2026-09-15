#!/usr/bin/env python3
"""Candidate Newtonian 10-body + axisymmetric Terrax spin integration.

This is a design experiment, NOT the deployed Kepler sky clock. Cbar and fluid
Love number are explicit interior assumptions. Units are AU, days, solar mass.
"""
import argparse
import json
import math
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp

AU=149597870700.;DAY=86400.;YEAR=365.25;MSUN=1.98847e30;G_SI=6.67430e-11
G=G_SI*MSUN*DAY**2/AU**3
MPLANET=7.60148e24;R=7016800.;OMEGA=2*math.pi/(26*3600)
CBAR=.3308;KF=.94;Q=OMEGA**2*R**3/(G_SI*MPLANET);J2=KF*Q/3
C=CBAR*MPLANET*R**2;SPIN=C*OMEGA/(MSUN*AU**2/DAY)
NAMES=['Sol','Mercury-Sol','Venus-Sol','Terrax','Mars-Sol','Jupiter-Sol','Saturn-Sol','Neptune-Sol','Luna','Echo']
MASS=np.array([1.04,*[x*5.9722e24/MSUN for x in [.055,.82]],MPLANET/MSUN,*[x*5.9722e24/MSUN for x in [.11,310,80,20]],7.56226e22/MSUN,6.09386e21/MSUN])
HOME=3

def rx(a):
    c,s=np.cos(a),np.sin(a);return np.array([[1,0,0],[0,c,-s],[0,s,c]])
def rz(a):
    c,s=np.cos(a),np.sin(a);return np.array([[c,-s,0],[s,c,0],[0,0,1]])
K0=rx(math.radians(25))@np.array([0,0,1]);X0=np.array([1,0,0]);Y0=np.cross(K0,X0)

def elements(a,e,i,node,peri,mean,mu):
    E=math.radians(mean)
    for _ in range(20):E-=(E-e*math.sin(E)-math.radians(mean))/(1-e*math.cos(E))
    n=math.sqrt(mu/a**3);rot=rz(math.radians(node))@rx(math.radians(i))@rz(math.radians(peri))
    return rot@np.array([a*(math.cos(E)-e),a*math.sqrt(1-e*e)*math.sin(E),0]),rot@np.array([-a*n*math.sin(E),a*n*math.sqrt(1-e*e)*math.cos(E),0])/(1-e*math.cos(E))

def initial():
    p=np.zeros((10,3));v=np.zeros((10,3))
    pars=[(.5,.18,3,18,62,20),(.92,.007,1.2,55,110,110),(1.34,.0167,0,0,283,0),(2.18,.08,1.8,83,145,55),(7.21736,.04,1,118,35,70),(13.94,.05,1.6,164,210,240),(27.38,.015,.8,223,280,180)]
    for i,par in enumerate(pars,1):
        p[i],v[i]=elements(*par,G*(MASS[0]+MASS[i]+(MASS[8]+MASS[9] if i==HOME else 0)))
        p[i]=rx(math.radians(25))@p[i];v[i]=rx(math.radians(25))@v[i]
    moons=[(384825252/AU,.006,.4,40,80,195),(779659047/AU,.012,1.1,190,240,210)]
    for i,par in enumerate(moons,8):p[i],v[i]=elements(*par,G*(MASS[HOME]+MASS[i]))
    total=MASS[HOME]+MASS[8]+MASS[9]
    shift_p=(MASS[8]*p[8]+MASS[9]*p[9])/total;shift_v=(MASS[8]*v[8]+MASS[9]*v[9])/total
    p[HOME]-=shift_p;v[HOME]-=shift_v;p[8:]+=p[HOME];v[8:]+=v[HOME]
    p-=np.sum(p*MASS[:,None],axis=0)/MASS.sum();v-=np.sum(v*MASS[:,None],axis=0)/MASS.sum()
    return np.r_[p.ravel(),v.ravel(),[0,0,1]]

def quadrupole(p,s):
    r=p-p[HOME];r2=np.sum(r*r,axis=1);r2[HOME]=np.inf;z=r@s
    aq=1.5*G*MASS[HOME]*J2*(R/AU)**2/r2[:,None]**2.5*((5*z*z/r2-1)[:,None]*r-2*z[:,None]*s)
    return r,aq

def rhs(t,y):
    p=y[:30].reshape(10,3);v=y[30:60].reshape(10,3);s=y[60:63]
    delta=p[None,:,:]-p[:,None,:];d2=np.sum(delta*delta,axis=2);np.fill_diagonal(d2,np.inf)
    acc=G*np.sum(delta*(MASS[None,:]/d2**1.5)[:,:,None],axis=1)
    r,aq=quadrupole(p,s);acc+=aq;acc[HOME]-=np.sum(aq*MASS[:,None],axis=0)/MASS[HOME]
    ds=-np.sum(np.cross(r,aq)*MASS[:,None],axis=0)/SPIN
    return np.r_[v.ravel(),acc.ravel(),ds]

def invariants(y):
    p=y[:,:30].reshape(-1,10,3);v=y[:,30:60].reshape(-1,10,3);s=y[:,60:63]
    energy=.5*np.sum(MASS[None,:,None]*v*v,axis=(1,2))
    for i in range(10):
        for j in range(i):energy-=G*MASS[i]*MASS[j]/np.linalg.norm(p[:,i]-p[:,j],axis=1)
    rel=p-p[:,HOME,None];r=np.linalg.norm(rel,axis=2);r[:,HOME]=np.inf
    z=np.sum(rel*s[:,None,:],axis=2)/r
    energy+=np.sum(G*MASS[HOME]*MASS[None,:]*J2*(R/AU)**2/r**3*.5*(3*z*z-1),axis=1)
    momentum=np.sum(np.cross(p,v)*MASS[None,:,None],axis=1)+SPIN*s
    return energy,momentum

def report(t,y):
    s=y[:,60:];p=y[:,:30].reshape(-1,10,3);v=y[:,30:60].reshape(-1,10,3)
    energy,h=invariants(y);years=t/YEAR
    phase=np.unwrap(np.arctan2(s@Y0,s@X0));fit=np.polyfit(years,phase,1);residual=(phase-np.polyval(fit,years))*206264.806247
    def angle(a,b):return np.degrees(np.arccos(np.clip(np.sum(a*b,axis=-1)/np.linalg.norm(a,axis=-1)/np.linalg.norm(b,axis=-1),-1,1)))
    normal=np.cross(p[:,HOME]-p[:,0],v[:,HOME]-v[:,0]);obliquity=angle(s,normal)
    moon_normals=[np.cross(p[:,i]-p[:,HOME],v[:,i]-v[:,HOME]) for i in (8,9)]
    moons={NAMES[i]:{'inclination_to_instantaneous_equator_deg':[float(x) for x in [angle(n,s).min(),angle(n,s).max()]],
                    'distance_km':[float(x) for x in [np.linalg.norm(p[:,i]-p[:,HOME],axis=1).min()*AU/1000,np.linalg.norm(p[:,i]-p[:,HOME],axis=1).max()*AU/1000]]} for i,n in zip((8,9),moon_normals)}
    coeff={}
    for name,m,a,e in [('Sol',MASS[0]*MSUN,1.34*AU,.0167),('Luna',MASS[8]*MSUN,384825252.,.006),('Echo',MASS[9]*MSUN,779659047.,.012)]:
        alpha=3*G_SI*m/(2*a**3*(1-e*e)**1.5)*J2/(CBAR*OMEGA)
        coeff[name]={'alpha_arcsec_per_year':alpha*206264.806247*DAY*YEAR}
        if name!='Sol':coeff[name]['orbital_L_over_spin']=(m*MPLANET/(m+MPLANET))*math.sqrt(G_SI*(m+MPLANET)*a*(1-e*e))/(C*OMEGA)
    return {'duration_years':float(years[-1]),'interior_assumptions':{'Cbar':CBAR,'fluid_Love_kf':KF,'J2':J2,'dynamical_ellipticity':J2/CBAR,'q':Q},
        'coefficients_not_precession_rates':coeff,'fitted_axis_longitude_rate_arcsec_per_year':float(fit[0]*206264.806247),
        'extrapolated_period_years_not_integrated_cycle':float(2*math.pi/abs(fit[0])),
        'longitude_residual_arcsec':[float(residual.min()),float(residual.max())],
        'obliquity_deg':[float(obliquity.min()),float(obliquity.max())],'moons':moons,
        'mutual_moon_inclination_deg':[float(angle(*moon_normals).min()),float(angle(*moon_normals).max())],
        'max_energy_relative_drift':float(np.max(abs(energy-energy[0]))/abs(energy[0])),
        'max_angular_momentum_drift_in_spin_units':float(np.max(np.linalg.norm(h-h[0],axis=1))/SPIN),
        'max_spin_norm_error':float(np.max(abs(np.linalg.norm(s,axis=1)-1)))}

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--years',type=float,default=140.49);parser.add_argument('--rtol',type=float,default=2e-10);parser.add_argument('--step',type=float,default=2.);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--compare',type=Path)
    args=parser.parse_args();times=np.linspace(0,args.years*YEAR,math.ceil(args.years*YEAR/5)+1)
    result=solve_ivp(rhs,(0,times[-1]),initial(),method='DOP853',t_eval=times,rtol=args.rtol,atol=args.rtol*.015,max_step=args.step)
    if not result.success:raise RuntimeError(result.message)
    y=result.y.T;stats=report(result.t,y);stats.update(nfev=result.nfev,rtol=args.rtol,max_step_days=args.step)
    if args.compare:
        old=np.load(args.compare);assert np.array_equal(old['t'],result.t)
        stats['convergence']={'max_axis_difference_arcsec':float(np.max(np.linalg.norm(y[:,60:]-old['y'][:,60:],axis=1))*206264.806247),
           'final_moon_relative_position_difference_km':{NAMES[i]:float(np.linalg.norm((y[-1,i*3:i*3+3]-y[-1,9:12])-(old['y'][-1,i*3:i*3+3]-old['y'][-1,9:12]))*AU/1000) for i in (8,9)}}
    args.output.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(args.output,t=result.t,y=y)
    args.output.with_suffix('.json').write_text(json.dumps(stats,indent=2)+'\n');print(json.dumps(stats,indent=2),flush=True)
