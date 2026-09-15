// Continuous angular profiles for the existing catalogue. This selects a
// necessary point-flux ceiling; it is NOT a complete extended-source eye model.
import {validateMorphology,MORPHOLOGY_MODEL,SOFT_MORPHOLOGY_MODEL,MORPHOLOGY_FAMILIES,morphologySolidAngle,filteredMorphologyGrid,FILTER_SIZE} from './nebula_morphology.mjs';
export const DEEP_POINT_LIMIT=6.5;
export const SOURCE_GRID=[128,64];
export const SOURCE_TEXELS=5;
const TAU=2*Math.PI;
export function erf(x){
    const sign=x<0?-1:1,t=1/(1+.3275911*Math.abs(x));
    return sign*(1-(((((1.061405429*t-1.453152027)*t)+1.421413741)*t-.284496736)*t+.254829592)*t*Math.exp(-x*x));
}
export function profileValue(kind,u,cut){
    if(u>=cut)return 0;
    const q=u*u;
    if(kind==='plummer'){const z=Math.sqrt(cut*cut-q);return z*(3+q+2*cut*cut)/(2*(1+cut*cut)**1.5*(1+q)**2);}
    if(kind==='gaussian')return Math.exp(-q/2);
    if(kind==='ionized_gaussian')return Math.exp(-q)*erf(Math.sqrt(cut*cut-q));
    return Math.sqrt(Math.max(0,1-q));
}
export function profileSolidAngle(kind,tangentScale,cut){
    // Integrate over solid angle, not over a flat texture grid.
    const n=2048,h=cut/n;
    const f=u=>TAU*tangentScale*tangentScale*u*profileValue(kind,u,cut)/(1+tangentScale*tangentScale*u*u)**1.5;
    let sum=f(0)+f(cut);
    for(let i=1;i<n;i++)sum+=(i%2?4:2)*f(i*h);
    return sum*h/3;
}
export function prepareDeepSources(objects,limit=DEEP_POINT_LIMIT){
    const sources=[],filteredGrids=[];
    for(const o of objects){
        if(!(o.v_flux>0)||!Number.isFinite(o.app_mag)||o.app_mag>limit)continue;
        const a=o.angular_scale_rad,kind=o.profile,cut=kind==='plummer'?10:kind==='gaussian'?4:kind==='ionized_gaussian'?o.profile_truncation:1;
        if(!(a>0 && a<Math.PI/2 && cut>0))throw Error('深空对象角尺度无效');
        const m=validateMorphology(o.morphology,o.kind);
        if(m && !['gaussian','ionized_gaussian'].includes(kind))throw Error('星团不能应用云气形态');
        const l=o.gal_lon*Math.PI/180,b=o.gal_lat*Math.PI/180,t=Math.tan(a),omega=m?morphologySolidAngle(kind,t,cut,m,profileValue):profileSolidAngle(kind,t,cut);
        if(!(omega>0 && Number.isFinite(omega)))throw Error('星云形态光量归一化无效');
        const pa=m?.position_angle_rad||0,c=Math.cos(pa),s=Math.sin(pa);
        const axis=[-Math.sin(l)*c-Math.sin(b)*Math.cos(l)*s,Math.cos(l)*c-Math.sin(b)*Math.sin(l)*s,Math.cos(b)*s];
        const filterTile=m?.model===MORPHOLOGY_MODEL?filteredGrids.length:-1;
        if(filterTile>=0)filteredGrids.push(filteredMorphologyGrid(kind,cut,m,profileValue));
        sources.push({id:o.id,kind,cut,tangentScale:t,flux:o.v_flux*2.54e-6,peak:o.v_flux*2.54e-6/omega,solidAngle:omega,magnitude:o.app_mag,
            morphology:m,filterTile,axis,longitude:l,latitude:b,direction:[Math.cos(b)*Math.cos(l),Math.cos(b)*Math.sin(l),Math.sin(b)],radius:Math.atan(t*cut)});
    }
    const [w,h]=SOURCE_GRID,bins=Array.from({length:w*h},()=>[]),padding=2*TAU/w;
    for(let i=0;i<sources.length;i++){
        const s=sources[i],radius=Math.min(Math.PI,s.radius+padding),b=s.latitude,l=s.longitude;
        const ymin=Math.max(0,Math.floor((b-radius+Math.PI/2)/Math.PI*h)),ymax=Math.min(h-1,Math.floor((b+radius+Math.PI/2)/Math.PI*h));
        const dl=Math.abs(b)+radius>=Math.PI/2?Math.PI:Math.asin(Math.min(1,Math.sin(radius)/Math.cos(b)));
        const xs=new Set();for(let x=Math.floor((l-dl)/TAU*w);x<=Math.floor((l+dl)/TAU*w);x++)xs.add((x%w+w)%w);
        for(let y=ymin;y<=ymax;y++)for(const x of xs)bins[y*w+x].push(i);
    }
    const maxOverlap=Math.max(...bins.map(b=>b.length));
    if(maxOverlap>64)throw Error('深空对象空间索引超出容量，不能截断光源');
    const offsets=new Float32Array(w*h*4),indices=[];
    bins.forEach((bin,i)=>{offsets[i*4]=indices.length;offsets[i*4+1]=bin.length;indices.push(...bin);});
    const indexWidth=256,indexHeight=Math.max(1,Math.ceil(indices.length/indexWidth)),indexData=new Float32Array(indexWidth*indexHeight*4);
    indices.forEach((id,i)=>indexData[i*4]=id);
    const sourceData=new Float32Array(SOURCE_TEXELS*Math.max(1,sources.length)*4);
    sources.forEach((s,i)=>{const m=s.morphology;sourceData.set([...s.direction,s.tangentScale,s.peak,['plummer','gaussian','ionized_gaussian'].indexOf(s.kind),s.cut,s.flux,
        s.magnitude,m?MORPHOLOGY_FAMILIES.indexOf(m.family)+1+(m.model===MORPHOLOGY_MODEL?16:m.model===SOFT_MORPHOLOGY_MODEL?8:0):0,m?.axis_ratio||1,m?.phase_rad||0,
        ...s.axis,m?Math.min(1,s.cut)*Math.sqrt(m.axis_ratio):1,m?.turbulence||0,m?.shell_thickness||.25,m?.filament_width||.15,s.filterTile>=0?s.filterTile:m?.noise_seed||0],i*SOURCE_TEXELS*4);});
    const columns=Math.max(1,Math.ceil(Math.sqrt(filteredGrids.length))),atlasWidth=filteredGrids.length?columns*FILTER_SIZE:1,atlasHeight=filteredGrids.length?Math.ceil(filteredGrids.length/columns)*FILTER_SIZE:1;
    const values=new Float32Array(atlasWidth*atlasHeight);
    filteredGrids.forEach((grid,tile)=>{const x=tile%columns*FILTER_SIZE,y=Math.floor(tile/columns)*FILTER_SIZE;
        for(let row=0;row<FILTER_SIZE;row++)values.set(grid.values.subarray(row*FILTER_SIZE,(row+1)*FILTER_SIZE),(y+row)*atlasWidth+x);
    });
    return {sources,sourceData,offsets,indexData,indexWidth,indexHeight,maxOverlap,limit,filterAtlas:{values,width:atlasWidth,height:atlasHeight,columns,count:filteredGrids.length}};
}
