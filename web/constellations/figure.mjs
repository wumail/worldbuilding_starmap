import {RAD,coordinates,unit,dot,cross,separation,tangentFrame,projectLocal} from './geometry.mjs?revision=candidate-editor-band-1';
import {randomFrom} from './generator_ordered.mjs';
export const RULES=Object.freeze({count:15,candidateMagnitude:4.5,preferredMagnitude:4,searchLatitude:30,displayScale:2.5,maxEdge:13,minSeparation:.8,maxDiameter:56,maxDegree:4,lineClearance:.18});

// Gnomonic projection maps each minor great-circle edge to a straight segment.
// It is used only for graph intersections; displayed charts stay stereographic.
export function graphPlane(stars) {
    const center=coordinates(unit(stars.reduce((sum,s)=>sum.map((x,i)=>x+s.direction[i]),[0,0,0]))),frame=tangentFrame(center.longitude,center.latitude);
    return new Map(stars.map(s=>[s.id,{x:dot(s.direction,frame.east)/dot(s.direction,frame.center)/RAD,y:dot(s.direction,frame.north)/dot(s.direction,frame.center)/RAD}]));
}
const turn=(a,b,c)=>(b.x-a.x)*(c.y-a.y)-(b.y-a.y)*(c.x-a.x);
export function crossing(a,b,points) {
    if([a.from,a.to].some(id=>id===b.from||id===b.to))return false;
    const p=points.get(a.from),q=points.get(a.to),r=points.get(b.from),s=points.get(b.to);
    return turn(p,q,r)*turn(p,q,s)<-1e-10 && turn(r,s,p)*turn(r,s,q)<-1e-10;
}
export function arcDistance(v,a,b) {
    const n=unit(cross(a,b)),p=unit(v.map((x,i)=>x-dot(v,n)*n[i]));
    if(Math.abs(separation(a,p)+separation(p,b)-separation(a,b))<1e-7)return separation(v,p);
    return Math.min(separation(v,a),separation(v,b));
}
export function pathBetween(from,to,edges) {
    const queue=[[from]],seen=new Set([from]);
    for(let i=0;i<queue.length;i++){
        const path=queue[i],id=path.at(-1);if(id===to)return path;
        for(const e of edges){const next=e.from===id?e.to:e.to===id?e.from:null;if(next&&!seen.has(next)){seen.add(next);queue.push([...path,next]);}}
    }
    return null;
}
function graphStats(members,edges) {
    const degrees=new Map(members.map(s=>[s.id,0]));
    for(const e of edges){degrees.set(e.from,degrees.get(e.from)+1);degrees.set(e.to,degrees.get(e.to)+1);}
    return {loops:edges.length-members.length+1, branches:[...degrees.values()].filter(n=>n>=3).length, tips:[...degrees.values()].filter(n=>n===1).length};
}

function makeGraph(stars,rng,style,acceptArc,config={}) {
    const rules=config.rules??RULES;
    const points=graphPlane(stars),options=[];
    for(let i=0;i<stars.length;i++)for(let j=i+1;j<stars.length;j++){
        const a=stars[i],b=stars[j],degrees=separation(a.direction,b.direction);
        if(!acceptArc(a.direction,b.direction)||degrees>rules.maxEdge||stars.some((s,k)=>k!==i&&k!==j&&arcDistance(s.direction,a.direction,b.direction)<rules.lineClearance))continue;
        options.push({from:a.id,to:b.id,degrees,cost:degrees*(.85+rng()*.3)});
    }
    options.sort((a,b)=>a.cost-b.cost||a.from.localeCompare(b.from)||a.to.localeCompare(b.to));
    const edges=[],degrees=new Map(stars.map(s=>[s.id,0]));
    const canAdd=e=>degrees.get(e.from)<rules.maxDegree&&degrees.get(e.to)<rules.maxDegree&&!edges.some(other=>crossing(e,other,points));
    const add=e=>{edges.push({from:e.from,to:e.to,degrees:e.degrees});degrees.set(e.from,degrees.get(e.from)+1);degrees.set(e.to,degrees.get(e.to)+1);};
    for(const e of options)if(canAdd(e)&&!pathBetween(e.from,e.to,edges))add(e);
    if(edges.length!==stars.length-1)return null;
    const tree=edges.map(e=>({...e})),target=style==='rich'?1+Math.floor(rng()*3):Math.floor(rng()*3);
    for(let loop=0;loop<target;loop++){
        const choices=[];
        for(const e of options){
            if(!canAdd(e))continue;
            const path=pathBetween(e.from,e.to,edges);
            if(path.length<4||path.length>7)continue; // Avoid a dense triangular mesh.
            const poly=path.map(id=>points.get(id));let area=0,perimeter=0;
            for(let i=0;i<poly.length;i++){const a=poly[i],b=poly[(i+1)%poly.length];area+=a.x*b.y-a.y*b.x;perimeter+=Math.hypot(a.x-b.x,a.y-b.y);}
            const compactness=Math.abs(area)/2/(perimeter*perimeter);
            if(compactness<.018)continue; // Reject almost-collinear, decorative slivers.
            choices.push({e,score:e.degrees*.05-compactness*3+rng()*.7});
        }
        choices.sort((a,b)=>a.score-b.score);if(!choices.length)break;add(choices[0].e);
    }
    // The brighter skeleton is a connected subtree of the full figure.
    let core=[...stars],coreEdges=tree;
    const coreRequired=new Set(config.coreRequired??[]);
    while(core.length>7){
        const leaves=core.filter(s=>!coreRequired.has(s.id)&&coreEdges.filter(e=>e.from===s.id||e.to===s.id).length===1).sort((a,b)=>b.app_mag-a.app_mag);
        if(!leaves.length)break;
        const remove=leaves[0];core=core.filter(s=>s!==remove);coreEdges=coreEdges.filter(e=>e.from!==remove.id&&e.to!==remove.id);
    }
    return {core,coreEdges,edges,structure:graphStats(stars,edges)};
}

export function chooseFigure(pool,seed,style,acceptArc=()=>true,required=[],targetCount=null,options={}) {
    const rules=options.rules??RULES;
    for(let i=0;i<required.length;i++)for(let j=i+1;j<required.length;j++){
        const d=separation(required[i].direction,required[j].direction);
        if(d<rules.minSeparation||d>rules.maxDiameter)return null;
    }
    const sorted=[...pool].sort((a,b)=>a.app_mag-b.app_mag||a.id.localeCompare(b.id)),rng=randomFrom(seed),minimum=style==='rich'?10:8;
    for(let attempt=0;attempt<36;attempt++){
        const drawnGoal=minimum+Math.floor(rng()*(style==='rich'?6:4));
        const goal=Math.max(required.length,Math.min(sorted.length,targetCount??drawnGoal)),anchors=sorted.filter(s=>s.app_mag<=3.5).slice(0,5);
        if(!anchors.length)return null;
        if(required.length>(options.maxMembers??(style==='rich'?15:11))||required.some(s=>!pool.includes(s)))return null;
        const chosen=required.length?[...required]:[anchors[Math.floor(rng()*anchors.length)]];
        while(chosen.length<goal){
            const options=sorted.filter(s=>!chosen.includes(s)).map(s=>{
                const gaps=chosen.map(p=>separation(s.direction,p.direction)),near=Math.min(...gaps),far=Math.max(...gaps);
                return {s,near,far,score:s.app_mag*.85+Math.max(0,s.app_mag-4)*1.2+near*.1+Math.max(0,far-42)*.1+rng()*.95};
            }).filter(x=>x.near>=rules.minSeparation&&x.near<=rules.maxEdge&&x.far<=rules.maxDiameter).sort((a,b)=>a.score-b.score||a.s.id.localeCompare(b.s.id));
            if(!options.length)break;chosen.push(options[0].s);
        }
        if(chosen.length<minimum)continue;
        const center=coordinates(unit(chosen.reduce((sum,s)=>sum.map((x,i)=>x+s.direction[i]),[0,0,0]))),frame=tangentFrame(center.longitude,center.latitude);
        if(chosen.some(s=>{const p=projectLocal(s.direction,frame,0,0,1);return !p.visible||Math.abs(p.x)>34||Math.abs(p.y)>25;}))continue;
        const graph=makeGraph(chosen,rng,style,acceptArc,options);if(graph)return {members:chosen,center,...graph};
    }
    return null;
}
