import {dot,cross,unit,separation} from './geometry.mjs?revision=candidate-editor-band-1';
import {arcDistance} from './figure.mjs';

export const REGIONAL_COMPLEXITIES=Object.freeze({
    simple:Object.freeze({label:'简洁',minimum:4,maximum:7,branches:1,loops:0,coreStars:5,coreBranches:1}),
    balanced:Object.freeze({label:'标准',minimum:6,maximum:10,branches:2,loops:1,coreStars:7,coreBranches:1}),
    rich:Object.freeze({label:'丰富',minimum:8,maximum:13,branches:3,loops:2,coreStars:8,coreBranches:2}),
});
export function complexityBudget(style,count){
    if(typeof style!=='string'||!Object.hasOwn(REGIONAL_COMPLEXITIES,style))throw Error('本区星形复杂度无效');
    const profile=REGIONAL_COMPLEXITIES[style];
    // Large mandatory sets remain truthful even when a small-figure budget is
    // impossible. These are soft branch targets, never a reason to drop anchors.
    const extra=Math.floor(Math.max(0,count-12)/6);
    return {...profile,branches:profile.branches+extra,coreBranches:profile.coreBranches+extra};
}
const tangent=(p,q)=>unit(q.map((x,i)=>x-dot(p,q)*p[i]));
export const junctionAngle=(p,q,r)=>Math.acos(Math.max(-1,Math.min(1,dot(tangent(p,q),tangent(p,r)))))*180/Math.PI;
export const acuteCost=angle=>Math.max(0,1-angle/40)**2;

// Three interior probes distinguish a stretch of close parallel arcs from a
// single close endpoint. This is a drawing preference, not a legality test.
function parallelCrowding(edges,lookup){
    const geometry=edges.map(e=>{
        const a=lookup.get(e.from).direction,b=lookup.get(e.to).direction;
        return {e,a,b,normal:unit(cross(a,b)),mid:unit(a.map((x,i)=>x+b[i])),length:separation(a,b),
            probes:[.25,.5,.75].map(t=>unit(a.map((x,i)=>(1-t)*x+t*b[i])))};
    });
    let cost=0;
    for(let i=0;i<geometry.length;i++)for(let j=i+1;j<geometry.length;j++){
        const a=geometry[i],b=geometry[j];
        if([a.e.from,a.e.to].some(id=>id===b.e.from||id===b.e.to)||Math.abs(dot(a.normal,b.normal))<.94)continue;
        const width=Math.max(.15,Math.min(1.2,Math.min(a.length,b.length)*.12));
        if(dot(a.mid,b.mid)<Math.cos(Math.min(180,(a.length+b.length)/2+width)*Math.PI/180))continue;
        const proximity=(points,edge)=>points.map(p=>Math.max(0,1-arcDistance(p,edge.a,edge.b)/width));
        const samples=[...proximity(a.probes,b),...proximity(b.probes,a)];
        if(samples.filter(n=>n>0).length>=2)cost+=samples.reduce((n,x)=>n+x,0)/3;
    }
    return cost;
}

export function figureQuality(members,edges,budget,{parallel=true}={}){
    const lookup=new Map(members.map(s=>[s.id,s])),adj=new Map(members.map(s=>[s.id,[]]));
    for(const {from,to} of edges){adj.get(from).push(to);adj.get(to).push(from);}
    let acutePenalty=0,narrowAngles=0,branches=0,highDegree=0,minAngle=180;
    for(const [id,near] of adj){
        if(near.length>=3)branches++;highDegree+=Math.max(0,near.length-3);
        for(let i=0;i<near.length;i++)for(let j=i+1;j<near.length;j++){
            const angle=junctionAngle(lookup.get(id).direction,lookup.get(near[i]).direction,lookup.get(near[j]).direction);
            acutePenalty+=acuteCost(angle);if(angle<30)narrowAngles++;minAngle=Math.min(minAngle,angle);
        }
    }
    const adjacentBranches=edges.filter(e=>adj.get(e.from).length>=3&&adj.get(e.to).length>=3).length;
    const crowding=parallel?parallelCrowding(edges,lookup):0;
    const loss=4*acutePenalty+.65*crowding+.3*adjacentBranches+.8*highDegree+
        .9*Math.max(0,branches-budget.branches)**2;
    return {loss,acutePenalty,narrowAngles,minAngle,branches,adjacentBranches,highDegree,parallelCrowding:crowding};
}
