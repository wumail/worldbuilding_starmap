import {RAD,vector,coordinates,unit,cross,tangentFrame} from './geometry.mjs?revision=candidate-editor-band-1';
import {GRID,fromEquatorial} from './territories.mjs';

const dot=(a,b)=>a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
const clamp=x=>Math.max(-1,Math.min(1,x));
const turn=(a,b,c)=>(b.x-a.x)*(c.y-a.y)-(b.y-a.y)*(c.x-a.x);
// In a common open hemisphere, a gnomonic convex hull maps back to the exact
// spherical convex hull (minor great-circle edges), independent of chart zoom.
export function shapeEnvelope(members){
    const center=unit(members.reduce((sum,s)=>sum.map((x,k)=>x+s.direction[k]),[0,0,0]));
    const c=coordinates(center),frame=tangentFrame(c.longitude,c.latitude);
    const points=members.map(s=>{const z=dot(s.direction,center);if(z<=0)throw Error('星形包络跨过切平面的背面');return {id:s.id,v:s.direction,x:dot(s.direction,frame.east)/z,y:dot(s.direction,frame.north)/z};}).sort((a,b)=>a.x-b.x||a.y-b.y||a.id.localeCompare(b.id));
    const chain=list=>{const r=[];for(const p of list){while(r.length>=2&&turn(r.at(-2),r.at(-1),p)<=1e-13)r.pop();r.push(p);}return r.slice(0,-1);};
    const hull=[...chain(points),...chain([...points].reverse())];
    if(hull.length<3)throw Error('星形包络退化');
    const vertices=hull.map(p=>p.v),edges=vertices.map((a,i)=>{const b=vertices[(i+1)%vertices.length],normal=unit(cross(a,b));return {a,b,normal,start:cross(normal,a),end:cross(b,normal)};});
    return {center,vertices,ids:hull.map(p=>p.id),edges,radius:Math.max(...vertices.map(v=>Math.acos(clamp(dot(v,center)))/RAD))};
}

// Exact distance to the filled spherical hull: zero inside, otherwise the
// closest minor-arc point or endpoint. Do not replace it with a bounding box.
export function envelopeDistance(envelope,v){
    if(envelope.edges.every(e=>dot(e.normal,v)>=-1e-13))return 0;
    let best=-1;
    for(const e of envelope.edges){
        best=Math.max(best,dot(e.a,v));
        const z=dot(e.normal,v),q=v.map((x,k)=>x-z*e.normal[k]);
        if(dot(q,e.start)>=0&&dot(q,e.end)>=0)best=Math.max(best,Math.sqrt(Math.max(0,1-z*z)));
    }
    return Math.acos(clamp(best))/RAD;
}

// A 1° RA/Dec cell is within sqrt(.5²+.5²) degrees of its center: the spherical
// metric is no larger than dRA²+dDec². Distance to a closed set is 1-Lipschitz.
// Subtracting this radius makes the margin apply to EVERY point of each cell,
// not just sampled vertices, including the eventual simplified boundary.
export const CELL_RADIUS_DEGREES=Math.SQRT1_2*GRID.step;
export const cellDirections=Array.from({length:GRID.width*GRID.height},(_,k)=>fromEquatorial(vector(k%GRID.width+.5,Math.floor(k/GRID.width)-89.5)));
export function envelopeMask(envelopes,marginDegrees){
    const mask=new Uint16Array(cellDirections.length),limit=marginDegrees-CELL_RADIUS_DEGREES;
    for(let i=0;i<envelopes.length;i++){
        const e=envelopes[i],minDot=Math.cos((e.radius+limit)*RAD),bit=1<<i;
        for(let k=0;k<mask.length;k++)if(dot(e.center,cellDirections[k])>=minDot&&envelopeDistance(e,cellDirections[k])<=limit+1e-10)mask[k]|=bit;
    }
    return mask;
}

export function cellAreaDegrees(k){
    const dec=Math.floor(k/GRID.width)-90;
    return GRID.step/RAD*(Math.sin((dec+GRID.step)*RAD)-Math.sin(dec*RAD));
}
