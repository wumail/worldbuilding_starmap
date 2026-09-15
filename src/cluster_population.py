"""Stochastic single-age populations on official PARSEC/COLIBRI isochrones.

Poisson counts integrate the Kroupa IMF over every initial-mass interval.
Intervals use midpoint interpolation on the SAME isochrone. No field-star
survival weighting is applied. Post-grid remnants are counted, not assigned
the catalogue's artificial logL=-9.999 sentinel as if it were a white dwarf.
"""
import hashlib
import math
from functools import lru_cache
from pathlib import Path

import numpy as np

from paths import DATA_DIR
from stellar_physics import SUN_M_BOL, SUN_TEFF, COLOR_MAP

GRID_PATH = DATA_DIR/'parsec/cmd39_population.dat'


def imf_integral(low, high, moment=0):
    low,high=np.broadcast_arrays(np.asarray(low,dtype=float),np.asarray(high,dtype=float))
    out=np.zeros_like(low)
    for a,b,slope,factor in ((.09,.5,1.3,1.),(.5,100.,2.3,.5)):
        l=np.maximum(low,a); h=np.minimum(high,b); power=moment-slope+1
        out+=np.where(h>l,factor*(h**power-l**power)/power,0)
    return out


class Isochrones:
    def __init__(self,path=GRID_PATH):
        raw=Path(path).read_bytes()
        self.sha256=hashlib.sha256(raw).hexdigest()
        text=raw.decode()
        self.columns=next(l[2:].split() for l in text.splitlines() if l.startswith('# Zini'))
        table=np.loadtxt(text.splitlines())
        self.tracks={}
        for mh,age in np.unique(table[:,[self.columns.index('MH'),self.columns.index('logAge')]],axis=0):
            rows=table[(table[:,self.columns.index('MH')]==mh)&(table[:,self.columns.index('logAge')]==age)]
            rows=rows[(rows[:,self.columns.index('label')]!=9)&(rows[:,self.columns.index('logL')]>-9)]
            rows=rows[np.argsort(rows[:,self.columns.index('Mini')],kind='stable')]
            _,unique=np.unique(rows[:,self.columns.index('Mini')],return_index=True)
            self.tracks[(float(mh),float(age))]=rows[unique]

    def parameters(self,mh,log_age,initial_mass):
        rows=self.tracks[(float(mh),float(log_age))]
        masses=rows[:,self.columns.index('Mini')]
        if not masses[0] <= initial_mass <= masses[-1]:
            raise ValueError('Initial mass outside the living isochrone')
        at=lambda key:float(np.interp(initial_mass,masses,rows[:,self.columns.index(key)]))
        lum=10**at('logL');temp=10**at('logTe');mbol=SUN_M_BOL-2.5*math.log10(lum)
        return dict(mass_solar=at('Mass'),temperature_K=temp,luminosity_solar=lum,
                    radius_solar=math.sqrt(lum)/(temp/SUN_TEFF)**2,abs_mag=at('Vmag'),
                    bolometric_mag=mbol,bc_correction=mbol-at('Vmag'))

    @lru_cache(maxsize=32)
    def bins(self,mh,age):
        rows=self.tracks[(float(mh),float(age))]
        edges=rows[:,self.columns.index('Mini')]
        mid=(edges[1:]+edges[:-1])/2
        pars=[self.parameters(mh,age,m) for m in mid]
        weight=imf_integral(edges[:-1],edges[1:])/imf_integral(.09,100,1)
        missing=float(imf_integral(edges[-1],100)/imf_integral(.09,100,1))
        return mid,weight,pars,missing


def stellar_type(parameters):
    t=parameters['temperature_K']
    stype=next((s for s,limit in [('O',30000),('B',10000),('A',7500),('F',6000),('G',5200),('K',3700)] if t>=limit),'M')
    logg=4.438+math.log10(parameters['mass_solar']/parameters['radius_solar']**2)
    lclass='V' if logg>=4 else 'IV' if logg>=3.5 else 'III' if logg>=1.5 else 'II' if logg>=.5 else 'I'
    return stype,lclass,COLOR_MAP[stype]


def ionizing_photons(parameters):
    """Blackbody ionizing photon integral; atmosphere-line blanketing is not solved."""
    temp=parameters['temperature_K']
    x=13.6*1.602176634e-19/(1.380649e-23*temp)
    integral=sum(math.exp(-n*x)*(x*x/n+2*x/n**2+2/n**3) for n in range(1,41))
    return 15/math.pi**4*parameters['luminosity_solar']*3.828e26/(1.380649e-23*temp)*integral
