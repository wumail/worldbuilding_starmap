"""Reproducible projected emissivity, normalized by the renderer/exporter.

These are phenomenological H II / reflection-cloud morphologies, not a new
three-dimensional density solution. The parent cloud still supplies its gas,
extinction and integrated photon budget. No star or dust RNG is consumed.
"""
import hashlib
import math
from functools import lru_cache
import numpy as np
from scipy.special import erf
from scipy.ndimage import convolve1d

LEGACY_MODEL = 'projected_emissivity_v1'
SOFT_MODEL = 'projected_emissivity_v2'
MODEL = 'projected_emissivity_v3'
FAMILIES = ('turbulent', 'filament', 'shell', 'blister', 'fan')
SMOKE_DETAIL_SCALE = 1.1
FILTER_SIZE = 257
FILTER_SIGMA = .04  # Fraction of parent support radius, in sky tangent units.


def morphology_for(obj, seed):
    if obj['kind'] not in ('emission_nebula', 'reflection_nebula'):
        return None
    cloud = obj.get('source_cluster_id') or obj['id']
    def draw(tag):
        # Keep each existing cloud's family, orientation and noise realization.
        raw = hashlib.sha256(f'{LEGACY_MODEL}:{seed}:{cloud}:{tag}'.encode()).digest()
        return int.from_bytes(raw[:8], 'big') / 2**64
    choices = ('turbulent', 'filament', 'shell', 'blister') if obj['kind'] == 'emission_nebula' else ('turbulent', 'filament', 'fan')
    family = choices[min(len(choices)-1, int(draw(obj['kind'])*len(choices)))]
    return dict(model=MODEL, family=family,
        # Angle measured in the Galactic tangent plane, east towards north.
        position_angle_rad=2*math.pi*draw('angle'),
        axis_ratio=(.50 if family == 'filament' else .64)+.28*draw('axis'),
        phase_rad=2*math.pi*draw('phase'), noise_seed=int(289*draw('noise')),turbulence=.65+.35*draw('roughness'),
        shell_thickness=.20+.12*draw('shell'), filament_width=.12+.055*draw('width'))


def validate_morphology(m, kind):
    if m is None:
        return
    if m.get('model') not in (MODEL,SOFT_MODEL,LEGACY_MODEL) or m.get('family') not in FAMILIES:
        raise ValueError('星云形态模型或类别无效')
    if kind not in ('emission_nebula', 'reflection_nebula'):
        raise ValueError('星云形态不能应用于星团或暗云消光')
    if kind == 'reflection_nebula' and m['family'] in ('shell', 'blister'):
        raise ValueError('电离壳层不能作为反射云形态')
    bounds = dict(position_angle_rad=(0,2*math.pi),axis_ratio=(.35,1),
        phase_rad=(0,2*math.pi),noise_seed=(0,288),turbulence=(0,1.2),shell_thickness=(.15,.4),filament_width=(.10,.25))
    for key,(low,high) in bounds.items():
        if not math.isfinite(m[key]) or not low <= m[key] <= high:
            raise ValueError('星云形态参数无效: '+key)
    if not isinstance(m['noise_seed'],int):raise ValueError('星云噪声种子必须为整数')


def add_morphologies(objects, seed):
    """Add only emissivity fields; preserve IDs, star positions and photometry."""
    for obj in objects:
        m = morphology_for(obj, seed)
        if m is not None:
            obj['morphology'] = m
    return dict(model=MODEL, seed=seed, coordinates='Galactic tangent plane, east towards north',
        interpretation='Flux-normalized projected emissivity; parent spherical gas/extinction budget retained; not hydrodynamics')


def smoothstep(a,b,x):
    t=np.clip((x-a)/(b-a),0,1)
    return t*t*(3-2*t)


def gradient_noise(x,y,seed):
    # Integer polynomial permutation stays below 2^24, so float32 GLSL and
    # float64 CPU agree on the lattice. Quintic interpolation has C2 joins.
    ix=np.floor(x);iy=np.floor(y);fx=x-ix;fy=y-iy
    def permute(v):
        v=np.mod(v,289);return np.mod((34*v+1)*v,289)
    def corner(ox,oy):
        h=np.mod(permute(permute(ix+ox+seed)+iy+oy),8).astype(int)
        gx=np.array([1,-1,0,0,.707106781,-.707106781,.707106781,-.707106781])[h]
        gy=np.array([0,0,1,-1,.707106781,.707106781,-.707106781,-.707106781])[h]
        return gx*(fx-ox)+gy*(fy-oy)
    u=fx**3*(fx*(fx*6-15)+10);v=fy**3*(fy*(fy*6-15)+10)
    return ((1-u)*corner(0,0)+u*corner(1,0))*(1-v)+((1-u)*corner(0,1)+u*corner(1,1))*v


def structure(x,y,seed,persistence=.5):
    value=0.;amplitude=1.5;x=x*2.3+7.1;y=y*2.3-4.7
    for _ in range(5):
        value+=amplitude*gradient_noise(x,y,seed)
        x,y=1.704*x-1.278*y+17.3,1.278*x+1.704*y+9.2
        amplitude*=persistence
    return value


_SHELL_NODES,_SHELL_WEIGHTS=np.polynomial.legendre.leggauss(8)
_SHELL_NODES=(_SHELL_NODES+1)/2
_SHELL_WEIGHTS=_SHELL_WEIGHTS/2


def soft_shell_column(impact,thickness):
    """Line integral through a smoothly varying, optically thin emitting shell.

    The parent radius is the finite support. Its gas is not a solid surface:
    emissivity varies as a radial Gaussian and fades smoothly at the front.
    """
    end=np.sqrt(np.maximum(0,1-impact*impact));width=.105+.10*thickness
    # Split at the onset of the outer taper so each quadrature interval is
    # smooth. Eight nodes per interval beat a single, longer fixed grid.
    split=np.sqrt(np.maximum(0,.68**2-impact*impact))
    total=np.zeros_like(impact,dtype=float)
    for node,weight in zip(_SHELL_NODES,_SHELL_WEIGHTS):
        for start,length in ((0,split),(split,end-split)):
            radius=np.sqrt(impact*impact+(start+length*node)**2)
            emissivity=np.exp(-.5*((radius-.60)/width)**2)*(1-smoothstep(.68,1,radius))
            total+=weight*length*emissivity
    return 2*total


def raw_projected_weight(x,y,profile,cut,m):
    """Unnormalized radiance in major-axis units; x/y already PA-rotated."""
    y=y/m['axis_ratio'];r=np.hypot(x,y);q=r*r
    filtered=m['model']==MODEL
    soft=m['model'] in (MODEL,SOFT_MODEL)
    extent=min(cut,1.8);px=x/extent;py=y/extent;phase=m['phase_rad']
    # Low-frequency domain warping bends the interior field without changing
    # the outer angular support. This is static cloud structure, not advection.
    strength=.55*m['turbulence']
    bend=gradient_noise(.85*px+19.3,.85*py-7.1,m['noise_seed'])
    px,py=(px+strength*bend,
        py+strength*gradient_noise(.85*px-31.7,.85*py+41.9,m['noise_seed']))
    # Stretched noise and a separate porosity field modulate local emission.
    # The low-frequency bend deforms the smooth shell itself; multiplying a
    # texture onto an unchanged thin shell would keep the solid-surface look.
    detail_scale=.75 if filtered else SMOKE_DETAIL_SCALE
    f=structure(px*(.90*detail_scale if soft else 1),py*(1.80*detail_scale if soft else 1),m['noise_seed'],.30 if soft else .5)
    broad=structure(px*.80*SMOKE_DETAIL_SCALE+8.3,py*.80*SMOKE_DETAIL_SCALE-12.6,m['noise_seed'],.30) if soft else 0
    porosity=(.04+.96*smoothstep(-.70,.40,broad))**2 if soft else 1
    erosion=1+.30*min(1,m['turbulence'])*(porosity-1)
    cloud=np.exp((.55 if soft else 1.65)*m['turbulence']*f)*erosion
    base=np.exp(-q/2) if profile=='gaussian' else np.exp(-q)*erf(np.sqrt(np.maximum(0,cut*cut-q)))
    family=m['family'];edge=1-smoothstep(.35 if soft else .80,1.,r/cut)
    if filtered:
        # Irregular large-scale emission before filtering, not a texture
        # multiplied onto a perfectly radial surface after filtering.
        bend2=gradient_noise(.72*x/extent-11.4,.72*y/extent+8.7,m['noise_seed'])
        warped=r*(1+m['turbulence']*(.65*bend+.35*bend2))
        base=np.exp(-warped*warped/2) if profile=='gaussian' else np.exp(-warped*warped)*erf(np.sqrt(np.maximum(0,cut*cut-warped*warped)))
    if family in ('shell','blister'):
        # Near AND far shell faces remain luminous at the centre. The legacy
        # uniform shell has sharp interfaces; v2 integrates smooth emissivity.
        # Keep the large-scale front smooth when the internal wisps get finer.
        distortion=m['turbulence']*(.65*bend+.35*bend2) if filtered else .28*m['turbulence']*bend if soft else .065*structure(px*.7,py*.7,m['noise_seed'])
        rr=r/cut*(1+distortion)
        if soft:
            base=soft_shell_column(rr,m['shell_thickness'])+.09*np.exp(-3*(r/cut)**2)
            edge=1-smoothstep(.68,1.,r/cut)
        else:
            outer=.78;inner=outer*(1-m['shell_thickness'])
            base=(np.sqrt(np.maximum(0,outer*outer-rr*rr))-np.sqrt(np.maximum(0,inner*inner-rr*rr)))/(outer-inner)
            base+=.055*np.exp(-3*(r/cut)**2)
        cloud=np.exp((.40 if soft else .8)*m['turbulence']*f)*erosion
        if family=='blister':
            base*=(.035 if soft else .12)+( .965 if soft else .88)*(.5+.5*x/np.maximum(r,1e-12))**3
    elif family=='filament':
        width=m['filament_width']*(1.6 if soft else 1)
        a=(py-.28*np.sin(1.9*px+phase))/width
        b=(py+.40-.19*np.sin(2.7*px+1.3*phase))/(width*1.25)
        base*=(.02 if soft else .045)+np.exp(-.5*a*a)+.65*np.exp(-.5*b*b)
    elif family=='fan':
        width=(.35 if soft else .22)+(.55 if soft else .42)*smoothstep(-1,1,px)
        base*=(.10+.90/(1+np.exp(np.clip(-3.5*(px+.15),-700,700))))*np.exp(-.5*(py/width)**2)
    if soft:
        # Broad, faint smoke in one soft patch, not detail over the full cloud.
        # The patch lives in cloud coordinates and fades out before the rim.
        cx=.25*math.cos(phase);cy=.25*math.sin(phase)
        local=np.exp(-.5*((x/cut-cx)**2+(y/cut-cy)**2)/.26**2)*(1-smoothstep(.55,.85,r/cut))
        cloud=1+local*(cloud-1)
    if filtered:cloud*=np.exp(.9*m['turbulence']*bend2)
    return np.where(r<cut,base*cloud*edge,0.)


_FILTER_FIELDS=('family','axis_ratio','phase_rad','noise_seed','turbulence','shell_thickness','filament_width')


@lru_cache(maxsize=32)
def _filtered_grid(profile,cut,parameters):
    m=dict(zip(_FILTER_FIELDS,parameters),model=MODEL)
    v=np.linspace(-cut,cut,FILTER_SIZE);x,y=np.meshgrid(v,v)
    raw=raw_projected_weight(x,y,profile,cut,m)
    sigma=FILTER_SIGMA*(FILTER_SIZE-1)/2
    offsets=np.arange(-math.ceil(4*sigma),math.ceil(4*sigma)+1)
    kernel=np.exp(-.5*(offsets/sigma)**2);kernel/=kernel.sum()
    # Zero extension, positive normalized separable Gaussian, then a C1
    # taper inside the original parent radius. Never clip convolution tails
    # at an opaque edge, and never blur stars or an already tone-mapped image.
    blurred=convolve1d(convolve1d(raw,kernel,axis=1,mode='constant'),kernel,axis=0,mode='constant')
    blurred*=1-smoothstep(.85,1,np.hypot(x,y)/cut)
    return blurred.astype(np.float32)


def filtered_grid(profile,cut,m):
    return _filtered_grid(profile,cut,tuple(m[k] for k in _FILTER_FIELDS))


def projected_weight(x,y,profile,cut,m):
    if m['model']!=MODEL:
        return raw_projected_weight(x,y,profile,cut,m)
    values=filtered_grid(profile,cut,m)
    u=np.clip((np.asarray(x)/cut+1)*(FILTER_SIZE-1)/2,0,FILTER_SIZE-1)
    v=np.clip((np.asarray(y)/cut+1)*(FILTER_SIZE-1)/2,0,FILTER_SIZE-1)
    i=np.minimum(np.floor(u).astype(int),FILTER_SIZE-2);j=np.minimum(np.floor(v).astype(int),FILTER_SIZE-2)
    fx=u-i;fy=v-j
    result=(values[j,i]*(1-fx)+values[j,i+1]*fx)*(1-fy)+(values[j+1,i]*(1-fx)+values[j+1,i+1]*fx)*fy
    # Grid interpolation slightly rounds the last annulus; the same smooth
    # boundary factor is used in JS and GLSL, so the parent support is exact.
    return result*(1-smoothstep(.99,1,np.hypot(x,y)/cut))
