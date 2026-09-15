"""A specified SBbc realization, not an inference of spiral arms from one radius.

Coordinates are observer centred (x towards centre, z towards Galactic north).
Dust is a non-negative volume density. Gaussian cloud columns are integrated
between observer and source, so a foreground star is not extinguished by a
cloud behind it. The smooth disk uses Gauss-Legendre line integration.
"""
from dataclasses import asdict, dataclass
import math

import numpy as np
from scipy.special import erf

PC_TO_LY = 3.2615637771674333


@dataclass(frozen=True)
class GalaxyParameters:
    observer_radius_pc: float = 30712 / PC_TO_LY
    observer_height_pc: float = 20.0
    disk_radius_pc: float = 60000 / PC_TO_LY
    disk_scale_pc: float = 2600.0
    inner_star_formation_radius_pc: float = 2500.0
    arm_count: int = 4
    pitch_deg: float = 12.5
    arm_width_pc: float = 280.0
    arm_reference_radius_pc: float = 7000.0
    arm_phase_deg: float = 15.0
    dust_scale_pc: float = 3500.0
    dust_height_pc: float = 120.0
    local_diffuse_av_per_kpc: float = 0.62
    cloud_surface_mass_per_av: float = 20.0
    open_cluster_expectation: int = 130000
    globular_cluster_expectation: int = 160
    unlit_cloud_expectation: int = 1800


def sample_disk(n, height, arm_fraction, rng, p):
    """Normalized exponential surface density: R is Gamma(2, Rd), not uniform."""
    radius = rng.gamma(2, p.disk_scale_pc, n)
    invalid = (radius < p.inner_star_formation_radius_pc) | (radius > p.disk_radius_pc)
    while np.any(invalid):
        radius[invalid] = rng.gamma(2, p.disk_scale_pc, int(invalid.sum()))
        invalid = (radius < p.inner_star_formation_radius_pc) | (radius > p.disk_radius_pc)
    phi = rng.uniform(-math.pi, math.pi, n)
    use_arm = rng.random(n) < arm_fraction
    pitch = math.radians(p.pitch_deg)
    arm = rng.integers(0, p.arm_count, n)
    arm_phi = (math.radians(p.arm_phase_deg) + arm*2*math.pi/p.arm_count
               + np.log(radius/p.arm_reference_radius_pc)/math.tan(pitch))
    phi[use_arm] = arm_phi[use_arm] + rng.normal(0, 1, int(use_arm.sum())) * (
        p.arm_width_pc/(radius[use_arm]*math.sin(pitch)))
    z = rng.laplace(0, height, n)
    return np.column_stack((p.observer_radius_pc-radius*np.cos(phi),
                            radius*np.sin(phi), z-p.observer_height_pc))


def sample_halo(n, rng, p):
    """Spherical Hernquist number density, scale 3 kpc, truncated at 100 kpc."""
    u = rng.uniform(0, (100000/(100000+3000))**2, n)
    radius = 3000*np.sqrt(u)/(1-np.sqrt(u))
    directions = rng.normal(size=(n,3))
    directions /= np.linalg.norm(directions, axis=1)[:,None]
    positions = directions*radius[:,None]
    positions[:,0] = p.observer_radius_pc-positions[:,0]
    positions[:,2] -= p.observer_height_pc
    return positions


class GalacticDust:
    def __init__(self, parameters=None, clouds=()):
        self.parameters = parameters or GalaxyParameters()
        self.clouds = list(clouds)
        self.centres = np.array([c['pos_cartesian'] for c in clouds], dtype=float).reshape(-1,3)
        self.sigma = np.array([c['sigma_pc'] for c in clouds])
        self.column = np.array([c['central_extinction_Av'] for c in clouds])
        if (np.any(self.sigma <= 0) or np.any(self.column < 0)
                or not np.all(np.isfinite(self.centres))):
            raise ValueError('Invalid cloud density')
        for c in self.clouds:
            if 'gas_mass_solar' in c:
                column=c['gas_mass_solar']/(2*math.pi*c['sigma_pc']**2*self.parameters.cloud_surface_mass_per_av)
                if not math.isclose(column,c['central_extinction_Av'],rel_tol=1e-10):
                    raise ValueError('Cloud mass and extinction column are inconsistent')

    def smooth(self, positions, order=64):
        p = self.parameters
        positions = np.atleast_2d(np.asarray(positions, dtype=float))
        d = np.linalg.norm(positions, axis=1)
        # Split at the midplane kink, then apply quadrature on each smooth segment.
        crossing = np.divide(-p.observer_height_pc, positions[:,2],
                              out=np.zeros(len(d)), where=positions[:,2]!=0)
        crossing = np.where((crossing>0)&(crossing<1), crossing, 1.)
        nodes, weights = np.polynomial.legendre.leggauss(order)
        closest=np.divide(p.observer_radius_pc*positions[:,0],np.sum(positions[:,:2]**2,axis=1),
                          out=np.ones(len(d)),where=np.sum(positions[:,:2]**2,axis=1)>0)
        splits=np.sort(np.column_stack((np.zeros(len(d)),crossing,np.clip(closest,0,1),np.ones(len(d)))),axis=1)
        result = np.zeros(len(d))
        for lower, upper in zip(splits.T[:-1],splits.T[1:]):
            t = lower[:,None]+(nodes[None,:]+1)*.5*(upper-lower)[:,None]
            x,y,z = (positions[:,i,None]*t for i in range(3))
            r = np.hypot(p.observer_radius_pc-x,y)
            density = np.exp((p.observer_radius_pc-r)/p.dust_scale_pc
                     +(abs(p.observer_height_pc)-abs(p.observer_height_pc+z))/p.dust_height_pc)
            result += (density@weights)*.5*(upper-lower)*d*p.local_diffuse_av_per_kpc/1000
        return result

    def cloud_extinction(self, positions):
        positions = np.atleast_2d(np.asarray(positions, dtype=float))
        result = np.zeros(len(positions))
        if not len(self.clouds):
            return result
        centre_sq = np.sum(self.centres**2, axis=1)
        sig2 = self.sigma**2
        for start in range(0,len(positions),128):
            points = positions[start:start+128]
            length = np.linalg.norm(points,axis=1)
            directions = np.divide(points,length[:,None],out=np.zeros_like(points),where=length[:,None]>0)
            along = directions@self.centres.T
            impact_sq = np.maximum(0,centre_sq[None,:]-along**2)
            # At most sum(A0)*exp(-32) mag discarded from perpendicular tails.
            mask = ((impact_sq<64*sig2[None,:]) & (along>-8*self.sigma[None,:])
                    & (along<length[:,None]+8*self.sigma[None,:]))
            rows,cols = np.nonzero(mask)
            t = along[rows,cols]; sig=self.sigma[cols]
            perpendicular=self.centres[cols]-t[:,None]*directions[rows]
            accurate_impact_sq=np.sum(perpendicular**2,axis=1)
            integral = .5*(erf((length[rows]-t)/(math.sqrt(2)*sig))-erf(-t/(math.sqrt(2)*sig)))
            values=self.column[cols]*np.exp(-accurate_impact_sq/(2*sig**2))*integral
            result[start:start+len(points)] = np.bincount(rows,weights=values,minlength=len(points))
        return result

    def extinction(self, positions):
        positions = np.atleast_2d(np.asarray(positions,dtype=float))
        out = np.empty(len(positions))
        for start in range(0,len(positions),1024):
            chunk=positions[start:start+1024]
            out[start:start+len(chunk)] = self.smooth(chunk)+self.cloud_extinction(chunk)
        return out

    def to_dict(self):
        return {'model':'radial_vertical_disk_plus_gaussian_clouds_v1',
                'parameters':asdict(self.parameters), 'clouds':self.clouds,
                'discarded_cloud_column_bound_mag':float(self.column.sum()*math.exp(-32)),
                'assumptions':'Observer radius is canonical. Height, arm phase/shape, density normalization and cloud realization are explicit model choices, not determined by that radius.'}

    @classmethod
    def from_dict(cls, data):
        if data['model']!='radial_vertical_disk_plus_gaussian_clouds_v1':
            raise ValueError('Unsupported Galactic dust model')
        return cls(GalaxyParameters(**data['parameters']),data['clouds'])
