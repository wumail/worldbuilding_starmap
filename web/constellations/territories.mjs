import {RAD,vector,coordinates,wrap,delta,dot,owner} from './geometry.mjs?revision=candidate-editor-band-1';

// A shared coordinate tessellation, not fifteen independently perturbed outlines.
// Every cell has exactly one owner. Coordinates are Terrax's day-zero equator.
export const GRID=Object.freeze({type:'equatorial-territories-1',width:360,height:180,step:1,obliquity:25});
const W=GRID.width,H=GRID.height,N=W*H,C=Math.cos(GRID.obliquity*RAD),S=Math.sin(GRID.obliquity*RAD);
const caches=new WeakMap();
export const toEquatorial=v=>[v[0],C*v[1]-S*v[2],S*v[1]+C*v[2]];
export const fromEquatorial=v=>[v[0],C*v[1]+S*v[2],-S*v[1]+C*v[2]];
const snap=x=>Math.abs(x-Math.round(x))<1e-10?Math.round(x):x;
export function cellOf(v){const q=coordinates(toEquatorial(v));return Math.min(H-1,Math.max(0,Math.floor(snap(q.latitude+90))))*W+Math.floor(wrap(snap(q.longitude)));}
const nextCell=(x,y)=>y<0||y>=H?-1:y*W+(x+W)%W;
function neighbours(k){const x=k%W,y=Math.floor(k/W);return [nextCell(x-1,y),nextCell(x+1,y),nextCell(x,y-1),nextCell(x,y+1)];}

class Heap {
    constructor(){this.items=[];}
    push(item){const a=this.items;a.push(item);let i=a.length-1;while(i){const p=(i-1)>>1;if(a[p][0]<=item[0])break;a[i]=a[p];i=p;}a[i]=item;}
    pop(){const a=this.items,first=a[0],last=a.pop();if(a.length){let i=0;while(i*2+1<a.length){let child=i*2+1;if(child+1<a.length&&a[child+1][0]<a[child][0])child++;if(a[child][0]>=last[0])break;a[i]=a[child];i=child;}a[i]=last;}return first;}
}

export function arcCells(a,b){
    const ae=toEquatorial(a),be=toEquatorial(b),cos=Math.max(-1,Math.min(1,dot(ae,be))),angle=Math.acos(cos);
    if(angle<1e-12)return [cellOf(a)];
    const u=be.map((x,i)=>(x-ae[i]*cos)/Math.sin(angle)),at=t=>ae.map((x,i)=>x*Math.cos(t)+u[i]*Math.sin(t));
    const events=[0,angle],add=t=>{if(t>1e-13&&t<angle-1e-13)events.push(t);};
    const ac=coordinates(ae),bc=coordinates(be),end=ac.longitude+delta(bc.longitude,ac.longitude);
    // RA is monotone on a minor arc away from a pole. All candidate figures
    // are within 55° of the equator, so this unwrapped interval has no pole.
    for(let ra=Math.ceil(Math.min(ac.longitude,end));ra<=Math.floor(Math.max(ac.longitude,end));ra++){
        const p=-Math.sin(ra*RAD)*ae[0]+Math.cos(ra*RAD)*ae[1],q=-Math.sin(ra*RAD)*u[0]+Math.cos(ra*RAD)*u[1];
        if(Math.hypot(p,q)>1e-13)add((Math.atan2(-p,q)+Math.PI)%Math.PI);
    }
    // Include interior declination extrema. Uniform samples can miss an arc
    // that rises just above a parallel and returns between two sample points.
    const phase=Math.atan2(u[2],ae[2]),radius=Math.hypot(ae[2],u[2]),z=[ae[2],be[2]];
    for(const t of [phase,phase+Math.PI,phase-Math.PI])if(t>0&&t<angle)z.push(at(t)[2]);
    const low=Math.asin(Math.max(-1,Math.min(...z)))/RAD,high=Math.asin(Math.min(1,Math.max(...z)))/RAD;
    if(radius>1e-13)for(let dec=Math.ceil(low-1e-10);dec<=Math.floor(high+1e-10);dec++){
        const ratio=Math.sin(dec*RAD)/radius;if(Math.abs(ratio)>1+1e-12)continue;
        const offset=Math.acos(Math.max(-1,Math.min(1,ratio)));
        for(const sign of [-1,1])for(const cycle of [-2*Math.PI,0,2*Math.PI])add(phase+sign*offset+cycle);
    }
    events.sort((a,b)=>a-b);const cuts=events.filter((v,i)=>!i||v-events[i-1]>1e-12);
    const cells=new Set();let previous;
    const visit=t=>{
        const k=cellOf(fromEquatorial(at(t)));cells.add(k);
        if(previous!==undefined&&Math.floor(previous/W)!==Math.floor(k/W)&&previous%W!==k%W){
            // Cover both sides of a grid corner, so a protected arc is connected
            // in the same four-neighbour topology used for boundary extraction.
            cells.add(Math.floor(previous/W)*W+k%W);cells.add(Math.floor(k/W)*W+previous%W);
        }
        previous=k;
    };
    for(let i=0;i<cuts.length;i++){visit(cuts[i]);if(i+1<cuts.length)visit((cuts[i]+cuts[i+1])/2);}
    return [...cells];
}

export function growTerritories(figures,outerSeeds,bias=Array(16).fill(0),canAssign=null){
    const labels=new Int8Array(N).fill(-1),fixed=new Int8Array(N).fill(-1),distance=new Float64Array(N).fill(Infinity),heap=new Heap();
    const seed=(k,id)=>{if((canAssign&&!canAssign(k,id))||(fixed[k]>=0&&fixed[k]!==id))return false;if(fixed[k]<0){fixed[k]=id;labels[k]=id;distance[k]=-bias[id];heap.push([-bias[id],k,id]);}return true;};
    for(let i=0;i<figures.length;i++){
        const f=figures[i],byId=new Map(f.members.map(s=>[s.id,s]));
        for(const s of f.members)if(!seed(cellOf(s.direction),i))return null;
        for(const e of f.edges)for(const k of arcCells(byId.get(e.from).direction,byId.get(e.to).direction))if(!seed(k,i))return null;
    }
    for(const v of outerSeeds)if(!seed(cellOf(v),15))return null;
    const ew=Array.from({length:H},(_,y)=>Math.max(.02,Math.cos((y-89.5)*RAD)));
    while(heap.items.length){
        const [cost,k,id]=heap.pop();if(cost!==distance[k]||id!==labels[k])continue;
        const ns=neighbours(k),y=Math.floor(k/W);
        for(let j=0;j<4;j++){
            const q=ns[j];if(q<0||(fixed[q]>=0&&fixed[q]!==id)||(canAssign&&!canAssign(q,id)))continue;
            const d=cost+(j<2?ew[y]:1);
            if(d<distance[q]-1e-10||(Math.abs(d-distance[q])<=1e-10&&id<labels[q])){distance[q]=d;labels[q]=id;heap.push([d,q,id]);}
        }
    }
    if(canAssign)for(let k=0;k<N;k++)if(labels[k]<0)labels[k]=15;
    return labels;
}

export function packGrid(labels){
    const runs=[];let start=0;
    for(let k=1;k<=N;k++)if(k===N||labels[k]!==labels[start]){runs.push([k-start,labels[start]]);start=k;}
    return {...GRID,runs};
}
export function unpackGrid(grid,fresh=false){
    if(!fresh&&caches.has(grid))return caches.get(grid);
    if(Object.entries(GRID).some(([k,v])=>grid?.[k]!==v)||!Array.isArray(grid.runs))throw Error('天区网格格式无效');
    const cells=new Int8Array(N);let start=0;
    for(const [length,id] of grid.runs){if(!Number.isInteger(length)||length<1||!Number.isInteger(id)||id<0||id>15||start+length>N)throw Error('天区归属记录无效');cells.fill(id,start,start+length);start+=length;}
    if(start!==N)throw Error('天区归属记录缺失');caches.set(grid,cells);return cells;
}
export function regionAt(data,v){return data.territories?unpackGrid(data.territories)[cellOf(v)]:owner(v,[...data.regions.map(r=>r.site),...data.remainderSites]);}

// Trace the shared cell edges once per region. Reject islands, holes and
// self-touching contours; none are silently filled by the renderer.
export function territoryBoundary(labels,id){
    const outgoing=new Map();let edgeCount=0;
    const key=(x,y)=>y*W+(x+W)%W;
    const edge=(a,b)=>{if(outgoing.has(a))throw Error('天区边界在一个顶点自接触');outgoing.set(a,b);edgeCount++;};
    for(let k=0;k<N;k++)if(labels[k]===id){
        const x=k%W,y=Math.floor(k/W),ns=neighbours(k);
        if(y===0||y===H-1)throw Error('黄道星座延伸到极点');
        if(labels[ns[2]]!==id)edge(key(x,y),key(x+1,y));
        if(labels[ns[1]]!==id)edge(key(x+1,y),key(x+1,y+1));
        if(labels[ns[3]]!==id)edge(key(x+1,y+1),key(x,y+1));
        if(labels[ns[0]]!==id)edge(key(x,y+1),key(x,y));
    }
    if(edgeCount<4)throw Error('天区为空');
    const start=outgoing.keys().next().value,ring=[];let p=start;
    do{ring.push([p%W,Math.floor(p/W)-90]);const q=outgoing.get(p);if(q===undefined)throw Error('天区边界未闭合');outgoing.delete(p);p=q;}while(p!==start&&ring.length<=edgeCount);
    if(outgoing.size||p!==start)throw Error('天区不连通或含洞');
    return ring.filter((p,i)=>{const a=ring[(i+ring.length-1)%ring.length],b=ring[(i+1)%ring.length];return !((a[0]===p[0]&&p[0]===b[0])||(a[1]===p[1]&&p[1]===b[1]));});
}
export function boundaryPoints(boundary,maxStep=.5){
    return boundary.flatMap((p,i)=>{
        const q=boundary[(i+1)%boundary.length],dx=delta(q[0],p[0]),dy=q[1]-p[1],steps=Math.max(1,Math.ceil(Math.max(Math.abs(dx),Math.abs(dy))/maxStep));
        return Array.from({length:steps},(_,j)=>fromEquatorial(vector(p[0]+dx*j/steps,p[1]+dy*j/steps)));
    });
}

// The only possible owner changes along the ecliptic occur at grid meridians
// and parallels. Compute these crossings, rather than rounding sample labels.
const cuts=[0,360];
for(let ra=0;ra<360;ra++)cuts.push(wrap(Math.atan2(Math.sin(ra*RAD)/C,Math.cos(ra*RAD))/RAD));
for(let dec=-25;dec<=25;dec++){const t=Math.asin(Math.sin(dec*RAD)/S)/RAD;cuts.push(wrap(t),wrap(180-t));}
const eclipticCuts=cuts.sort((a,b)=>a-b).filter((v,i,a)=>!i||v-a[i-1]>1e-9);
export function territoryIntervals(labels){
    const intervals=[];
    for(let i=1;i<eclipticCuts.length;i++){
        const start=eclipticCuts[i-1],end=eclipticCuts[i],index=labels[cellOf(vector((start+end)/2,0))];
        if(intervals.at(-1)?.index===index)intervals.at(-1).end=end;else intervals.push({index,start,end});
    }
    return intervals;
}
