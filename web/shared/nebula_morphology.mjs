// Projected emissivity parameters, shared by all four sky views.
export const LEGACY_MORPHOLOGY_MODEL='projected_emissivity_v1';
export const SOFT_MORPHOLOGY_MODEL='projected_emissivity_v2';
export const MORPHOLOGY_MODEL='projected_emissivity_v3';
export const MORPHOLOGY_FAMILIES=['turbulent','filament','shell','blister','fan'];
export const SMOKE_DETAIL_SCALE=1.1;
export const FILTER_SIZE=257,FILTER_SIGMA=.04;
export function validateMorphology(m,kind){
    if(!m)return null;
    if(![MORPHOLOGY_MODEL,SOFT_MORPHOLOGY_MODEL,LEGACY_MORPHOLOGY_MODEL].includes(m.model)||!MORPHOLOGY_FAMILIES.includes(m.family))throw Error('星云形态模型无效');
    if(!['emission_nebula','reflection_nebula'].includes(kind))throw Error('星云形态不能应用于星团或暗云');
    if(kind==='reflection_nebula'&&['shell','blister'].includes(m.family))throw Error('反射云不能使用电离壳层形态');
    const bounds={position_angle_rad:[0,2*Math.PI],axis_ratio:[.35,1],phase_rad:[0,2*Math.PI],noise_seed:[0,288],turbulence:[0,1.2],shell_thickness:[.15,.4],filament_width:[.10,.25]};
    for(const [k,[a,b]] of Object.entries(bounds))if(!Number.isFinite(m[k])||m[k]<a||m[k]>b)throw Error('星云形态参数无效：'+k);
    if(!Number.isInteger(m.noise_seed))throw Error('星云噪声种子必须为整数');
    return m;
}
const smooth=(a,b,x)=>{const t=Math.min(1,Math.max(0,(x-a)/(b-a)));return t*t*(3-2*t);};
const mod=(x,n)=>x-n*Math.floor(x/n);
const permute=x=>{x=mod(x,289);return mod((34*x+1)*x,289);};
const gx=[1,-1,0,0,.707106781,-.707106781,.707106781,-.707106781],gy=[0,0,1,-1,.707106781,.707106781,-.707106781,-.707106781];
export function gradientNoise(x,y,seed){
    const ix=Math.floor(x),iy=Math.floor(y),fx=x-ix,fy=y-iy;
    const corner=(ox,oy)=>{const h=mod(permute(permute(ix+ox+seed)+iy+oy),8);return gx[h]*(fx-ox)+gy[h]*(fy-oy);};
    const u=fx**3*(fx*(fx*6-15)+10),v=fy**3*(fy*(fy*6-15)+10);
    return ((1-u)*corner(0,0)+u*corner(1,0))*(1-v)+((1-u)*corner(0,1)+u*corner(1,1))*v;
}
export function structure(x,y,seed,persistence=.5){
    let value=0,amplitude=1.5;x=x*2.3+7.1;y=y*2.3-4.7;
    for(let i=0;i<5;i++){
        value+=amplitude*gradientNoise(x,y,seed);
        [x,y]=[1.704*x-1.278*y+17.3,1.278*x+1.704*y+9.2];amplitude*=persistence;
    }
    return value;
}
const SHELL_NODES=[.019855071751231856,.10166676129318664,.23723379504183550,.40828267875217510,.59171732124782490,.76276620495816450,.89833323870681336,.98014492824876810];
const SHELL_WEIGHTS=[.05061426814518813,.11119051722668717,.15685332293894352,.18134189168918088,.18134189168918088,.15685332293894352,.11119051722668717,.05061426814518813];
export function softShellColumn(impact,thickness){
    const end=Math.sqrt(Math.max(0,1-impact*impact)),split=Math.sqrt(Math.max(0,.68**2-impact*impact)),width=.105+.10*thickness;let sum=0;
    for(let i=0;i<SHELL_NODES.length;i++){
        const a=Math.hypot(impact,split*SHELL_NODES[i]),b=Math.hypot(impact,split+(end-split)*SHELL_NODES[i]);
        sum+=SHELL_WEIGHTS[i]*(split*Math.exp(-.5*((a-.60)/width)**2)+(end-split)*Math.exp(-.5*((b-.60)/width)**2)*(1-smooth(.68,1,b)));
    }
    return 2*sum;
}
export function rawMorphologyWeight(x,y,kind,cut,m,radial){
    y/=m.axis_ratio;const r=Math.hypot(x,y);if(r>=cut)return 0;
    const filtered=m.model===MORPHOLOGY_MODEL,soft=filtered||m.model===SOFT_MORPHOLOGY_MODEL,extent=Math.min(cut,1.8),p=m.phase_rad,strength=.55*m.turbulence;
    let px=x/extent,py=y/extent;
    const bend=gradientNoise(.85*px+19.3,.85*py-7.1,m.noise_seed);
    [px,py]=[px+strength*bend,py+strength*gradientNoise(.85*px-31.7,.85*py+41.9,m.noise_seed)];
    const detailScale=filtered?.75:SMOKE_DETAIL_SCALE;
    const f=structure(px*(soft?.90*detailScale:1),py*(soft?1.80*detailScale:1),m.noise_seed,soft?.30:.5);
    const broad=soft?structure(px*.80*SMOKE_DETAIL_SCALE+8.3,py*.80*SMOKE_DETAIL_SCALE-12.6,m.noise_seed,.30):0;
    const porosity=soft?(.04+.96*smooth(-.70,.40,broad))**2:1,erosion=1+.30*Math.min(1,m.turbulence)*(porosity-1);
    let base=radial(kind,r,cut),cloud=Math.exp((soft?.55:1.65)*m.turbulence*f)*erosion,edge=1-smooth(soft?.35:.8,1,r/cut);
    const bend2=filtered?gradientNoise(.72*x/extent-11.4,.72*y/extent+8.7,m.noise_seed):0;
    if(filtered){const warped=r*(1+m.turbulence*(.65*bend+.35*bend2));base=kind==='gaussian'?Math.exp(-warped*warped/2):radial(kind,warped,cut);}
    if(m.family==='shell'||m.family==='blister'){
        const distortion=filtered?m.turbulence*(.65*bend+.35*bend2):soft?.28*m.turbulence*bend:.065*structure(px*.7,py*.7,m.noise_seed),rr=r/cut*(1+distortion),outer=.78,inner=outer*(1-m.shell_thickness);
        if(soft){base=softShellColumn(rr,m.shell_thickness)+.09*Math.exp(-3*(r/cut)**2);edge=1-smooth(.68,1,r/cut);}
        else base=(Math.sqrt(Math.max(0,outer*outer-rr*rr))-Math.sqrt(Math.max(0,inner*inner-rr*rr)))/(outer-inner)+.055*Math.exp(-3*(r/cut)**2);
        cloud=Math.exp((soft?.40:.8)*m.turbulence*f)*erosion;
        if(m.family==='blister')base*=(soft?.035:.12)+(soft?.965:.88)*(.5+.5*x/Math.max(r,1e-12))**3;
    }else if(m.family==='filament'){
        const width=m.filament_width*(soft?1.6:1),a=(py-.28*Math.sin(1.9*px+p))/width,b=(py+.40-.19*Math.sin(2.7*px+1.3*p))/(width*1.25);
        base*=(soft?.02:.045)+Math.exp(-.5*a*a)+.65*Math.exp(-.5*b*b);
    }else if(m.family==='fan'){
        const width=(soft?.35:.22)+(soft?.55:.42)*smooth(-1,1,px);
        base*=(.10+.90/(1+Math.exp(-3.5*(px+.15))))*Math.exp(-.5*(py/width)**2);
    }
    if(soft){
        const cx=.25*Math.cos(p),cy=.25*Math.sin(p),local=Math.exp(-.5*((x/cut-cx)**2+(y/cut-cy)**2)/.26**2)*(1-smooth(.55,.85,r/cut));
        cloud=1+local*(cloud-1);
    }
    if(filtered)cloud*=Math.exp(.9*m.turbulence*bend2);
    return base*cloud*edge;
}
const FILTER_FIELDS=['family','axis_ratio','phase_rad','noise_seed','turbulence','shell_thickness','filament_width'];
const gridCache=new Map(),objectCache=new WeakMap();
export function filteredMorphologyGrid(kind,cut,m,radial){
    let hit=objectCache.get(m);
    if(hit&&hit.kind===kind&&hit.cut===cut&&FILTER_FIELDS.every((k,i)=>m[k]===hit.parameters[i]))return hit.grid;
    const parameters=FILTER_FIELDS.map(k=>m[k]),key=JSON.stringify([kind,cut,parameters]);
    let grid=gridCache.get(key);
    if(!grid){
        const n=FILTER_SIZE,step=2*cut/(n-1),raw=new Float64Array(n*n),temp=new Float64Array(n*n),values=new Float32Array(n*n);
        for(let j=0;j<n;j++)for(let i=0;i<n;i++)raw[j*n+i]=rawMorphologyWeight(-cut+i*step,-cut+j*step,kind,cut,m,radial);
        const sigma=FILTER_SIGMA*(n-1)/2,radius=Math.ceil(4*sigma),kernel=[];let total=0;
        for(let k=-radius;k<=radius;k++){const w=Math.exp(-.5*(k/sigma)**2);kernel.push(w);total+=w;}
        for(let k=0;k<kernel.length;k++)kernel[k]/=total;
        for(let j=0;j<n;j++)for(let i=0;i<n;i++){
            let sum=0;for(let k=Math.max(-radius,-i);k<=Math.min(radius,n-1-i);k++)sum+=kernel[k+radius]*raw[j*n+i+k];temp[j*n+i]=sum;
        }
        for(let j=0;j<n;j++)for(let i=0;i<n;i++){
            let sum=0;for(let k=Math.max(-radius,-j);k<=Math.min(radius,n-1-j);k++)sum+=kernel[k+radius]*temp[(j+k)*n+i];
            values[j*n+i]=sum*(1-smooth(.85,1,Math.hypot(-cut+i*step,-cut+j*step)/cut));
        }
        grid={values,size:n,cut};gridCache.set(key,grid);
        if(gridCache.size>32)gridCache.delete(gridCache.keys().next().value);
    }
    objectCache.set(m,{kind,cut,parameters,grid});return grid;
}
export function filteredMorphologyValue(x,y,grid){
    const {cut,size:n,values}=grid,r=Math.hypot(x,y)/cut;if(r>=1)return 0;
    const u=(x/cut+1)*(n-1)/2,v=(y/cut+1)*(n-1)/2,i=Math.min(n-2,Math.floor(u)),j=Math.min(n-2,Math.floor(v)),fx=u-i,fy=v-j;
    return ((values[j*n+i]*(1-fx)+values[j*n+i+1]*fx)*(1-fy)+(values[(j+1)*n+i]*(1-fx)+values[(j+1)*n+i+1]*fx)*fy)*(1-smooth(.99,1,r));
}
export function morphologyWeight(x,y,kind,cut,m,radial){
    return m.model===MORPHOLOGY_MODEL?filteredMorphologyValue(x,y,filteredMorphologyGrid(kind,cut,m,radial)):rawMorphologyWeight(x,y,kind,cut,m,radial);
}
export function morphologySolidAngle(kind,t,cut,m,radial,nr=256,np=256){
    // Integrate in the elliptical tangent plane. The exact spherical
    // Jacobian is dOmega=t² dx dy/(1+t²(x²+y²))^(3/2).
    const grid=m.model===MORPHOLOGY_MODEL?filteredMorphologyGrid(kind,cut,m,radial):null,axis=grid?1:m.axis_ratio;
    let sum=0;const dr=cut/nr,dp=2*Math.PI/np;
    for(let j=0;j<np;j++){
        const phi=(j+.5)*dp,c=Math.cos(phi),s=Math.sin(phi);
        for(let i=0;i<nr;i++){
            const r=(i+.5)*dr,x=r*c,y=r*s*axis;
            sum+=r*axis*(grid?filteredMorphologyValue(x,y,grid):rawMorphologyWeight(x,y,kind,cut,m,radial))/(1+t*t*(x*x+y*y))**1.5;
        }
    }
    return sum*t*t*dr*dp;
}
export const morphologyGLSL=`
float cloudPermute(float x){x=mod(x,289.);return mod((34.*x+1.)*x,289.);}
float cloudGradient(vec2 i,vec2 f,float seed){
 float h=mod(cloudPermute(cloudPermute(i.x+seed)+i.y),8.);
 if(h<.5)return f.x;if(h<1.5)return -f.x;if(h<2.5)return f.y;if(h<3.5)return -f.y;
 if(h<4.5)return .707106781*(f.x+f.y);if(h<5.5)return .707106781*(-f.x+f.y);
 if(h<6.5)return .707106781*(f.x-f.y);return -.707106781*(f.x+f.y);
}
float cloudNoise(vec2 p,float seed){
 vec2 i=floor(p),f=p-i,u=f*f*f*(f*(f*6.-15.)+10.);
 return mix(mix(cloudGradient(i,f,seed),cloudGradient(i+vec2(1.,0.),f-vec2(1.,0.),seed),u.x),
  mix(cloudGradient(i+vec2(0.,1.),f-vec2(0.,1.),seed),cloudGradient(i+vec2(1.),f-vec2(1.),seed),u.x),u.y);
}
float cloudField(vec2 p,float seed,float persistence){
 float value=0.,amplitude=1.5;p=p*2.3+vec2(7.1,-4.7);
 for(int i=0;i<5;i++){
  value+=amplitude*cloudNoise(p,seed);p=vec2(1.704*p.x-1.278*p.y,1.278*p.x+1.704*p.y)+vec2(17.3,9.2);amplitude*=persistence;
 }
 return value;
}
float cloudStructure(vec2 p,float seed){return cloudField(p,seed,.5);}
float softShellColumn(float impact,float thickness){
 float end=sqrt(max(0.,1.-impact*impact)),split=sqrt(max(0.,.4624-impact*impact)),width=.105+.10*thickness,sum=0.;
 ${SHELL_NODES.map((node,i)=>`{float a=length(vec2(impact,split*${node.toFixed(16)})),b=length(vec2(impact,split+(end-split)*${node.toFixed(16)}));sum+=${SHELL_WEIGHTS[i].toFixed(16)}*(split*exp(-.5*pow((a-.60)/width,2.))+(end-split)*exp(-.5*pow((b-.60)/width,2.))*(1.-smoothstep(.68,1.,b)));}`).join('\n ')}
 return 2.*sum;
}
float cloudWeight(vec2 p,float cut,float base,vec4 shape,vec4 detail){
 // shape = magnitude, family code, minor/major ratio, phase
 // detail = turbulence, shell thickness, filament width, integer noise seed
 bool soft=shape.y>8.;if(soft)shape.y-=8.;
 float r=length(p),phase=shape.w,edge=1.-smoothstep(soft?.35:.8,1.,r/cut);vec2 q=p/min(cut,1.8);
 float bend=cloudNoise(q*.85+vec2(19.3,-7.1),detail.w);
 q+=.55*detail.x*vec2(bend,cloudNoise(q*.85+vec2(-31.7,41.9),detail.w));
 float f=cloudField(q*(soft?vec2(.90,1.80)*${SMOKE_DETAIL_SCALE.toFixed(1)}:vec2(1.)),detail.w,soft?.30:.5);
 float broad=soft?cloudField(q*.80*${SMOKE_DETAIL_SCALE.toFixed(1)}+vec2(8.3,-12.6),detail.w,.30):0.;
 float porosity=soft?pow(.04+.96*smoothstep(-.70,.40,broad),2.):1.,erosion=mix(1.,porosity,.30*min(1.,detail.x));
 float cloud=exp((soft?.55:1.65)*detail.x*f)*erosion;
 if(shape.y>2.5 && shape.y<4.5){
  float distortion=soft?.28*detail.x*bend:.065*cloudStructure(q*.7,detail.w),rr=r/cut*(1.+distortion),outer=.78,inner=outer*(1.-detail.y);
  if(soft){base=softShellColumn(rr,detail.y)+.09*exp(-3.*pow(r/cut,2.));edge=1.-smoothstep(.68,1.,r/cut);}
  else base=(sqrt(max(0.,outer*outer-rr*rr))-sqrt(max(0.,inner*inner-rr*rr)))/(outer-inner)+.055*exp(-3.*pow(r/cut,2.));
  cloud=exp((soft?.40:.8)*detail.x*f)*erosion;
  if(shape.y>3.5)base*=(soft?.035:.12)+(soft?.965:.88)*pow(.5+.5*p.x/max(r,1e-12),3.);
 }else if(shape.y>1.5 && shape.y<2.5){
  float width=detail.z*(soft?1.6:1.),a=(q.y-.28*sin(1.9*q.x+phase))/width,b=(q.y+.40-.19*sin(2.7*q.x+1.3*phase))/(width*1.25);
  base*=(soft?.02:.045)+exp(-.5*a*a)+.65*exp(-.5*b*b);
 }else if(shape.y>4.5){
  float width=(soft?.35:.22)+(soft?.55:.42)*smoothstep(-1.,1.,q.x);
  base*=(.10+.90/(1.+exp(-3.5*(q.x+.15))))*exp(-.5*pow(q.y/width,2.));
 }
 if(soft){
  vec2 centre=.25*vec2(cos(phase),sin(phase)),d=p/cut-centre;
  float local=exp(-.5*dot(d,d)/.0676)*(1.-smoothstep(.55,.85,r/cut));
  cloud=mix(1.,cloud,local);
 }
 return base*cloud*edge;
}
`;
