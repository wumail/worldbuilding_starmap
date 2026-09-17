import {separation,cross,unit,dot} from './geometry.mjs?revision=candidate-editor-band-1';
import {arcDistance,pathBetween} from './figure.mjs';
import {edgeKey} from './manual_figures.mjs';
import {junctionAngle,acuteCost,figureQuality} from './regional_quality.mjs';

const insideArc=(a,b,p)=>Math.abs(separation(a,p)+separation(p,b)-separation(a,b))<1e-7;
function arcsCross(a,b,lookup){
    if([a.from,a.to].some(id=>id===b.from||id===b.to))return false;
    const [p,q,r,s]=[a.from,a.to,b.from,b.to].map(id=>lookup.get(id).direction),n=cross(cross(p,q),cross(r,s));
    if(Math.hypot(...n)<1e-12)return false;
    const v=unit(n);return [v,v.map(x=>-x)].some(v=>insideArc(p,q,v)&&insideArc(r,s,v));
}
const noise=rng=>Math.log(-Math.log(Math.max(1e-12,rng())));

// Diagnostic of structure, not a star-ID signature or an isomorphism test.
export function structure(members,edges){
    const adj=new Map(members.map(id=>[id,[]]));
    for(const [a,b] of edges){adj.get(a).push(b);adj.get(b).push(a);}
    const seen=new Set();let components=0,diameter=0;
    for(const id of members){
        if(!seen.has(id))components++;
        const queue=[id],distance=new Map([[id,0]]);
        for(let i=0;i<queue.length;i++){
            const a=queue[i];seen.add(a);diameter=Math.max(diameter,distance.get(a));
            for(const b of adj.get(a))if(!distance.has(b)){distance.set(b,distance.get(a)+1);queue.push(b);}
        }
    }
    const degrees=[...adj.values()].map(a=>a.length);
    return {components,loops:edges.length-members.length+components,branches:degrees.filter(n=>n>=3).length,
        tips:degrees.filter(n=>n===1).length,isolates:degrees.filter(n=>n===0).length,
        elongation:diameter/Math.max(1,members.length-components)};
}

// Only rejects thin closures. Containment/intersection decisions stay spherical,
// including for regions too wide for this local projection.
function compactness(path,lookup){
    const points=path.map(id=>lookup.get(id).direction),sum=points.reduce((a,p)=>a.map((x,i)=>x+p[i]),[0,0,0]);
    if(Math.hypot(...sum)<1e-8)return 0;
    const center=unit(sum);if(points.some(p=>dot(p,center)<.2))return 0;
    const east=unit(cross(Math.abs(center[2])<.9?[0,0,1]:[1,0,0],center)),north=cross(center,east);
    const projected=points.map(p=>[dot(p,east)/dot(p,center),dot(p,north)/dot(p,center)]);
    let area=0,perimeter=0;
    for(let i=0;i<projected.length;i++){
        const a=projected[i],b=projected[(i+1)%projected.length];area+=a[0]*b[1]-a[1]*b[0];perimeter+=Math.hypot(a[0]-b[0],a[1]-b[1]);
    }
    return perimeter>0?Math.abs(area)/2/perimeter**2:0;
}

function graphOptions(members,accepted,include=null){
    const n=members.length,options=[],distance=Array.from({length:n},()=>new Float64Array(n));
    for(let i=0;i<n;i++)for(let j=i+1;j<n;j++)distance[i][j]=distance[j][i]=separation(members[i].direction,members[j].direction);
    const neighbours=members.map((_,i)=>members.map((s,j)=>j).filter(j=>j!==i).sort((a,b)=>distance[i][a]-distance[i][b]||a-b));
    const scale=neighbours.map((near,i)=>Math.max(.1,distance[i][near[1]??near[0]]??1));
    let candidates;
    if(n>32&&!include){
        candidates=new Set();const include=(i,j)=>candidates.add(Math.min(i,j)*n+Math.max(i,j));
        // Large manually enlarged regions can have hundreds of required stars.
        // Use neighbours at several brightness levels, plus an angular MST for
        // bridges between clumps. Every retained edge still passes exact checks.
        for(const limit of [3,4,4.5,Infinity])for(let i=0;i<n;i++)if(members[i].app_mag<=limit){
            for(const j of neighbours[i].filter(j=>members[j].app_mag<=limit).slice(0,8))include(i,j);
        }
        const used=new Uint8Array(n),best=new Float64Array(n).fill(Infinity),parent=new Int32Array(n);let current=0;
        for(let count=1;count<n;count++){
            used[current]=1;let next=-1;
            for(let j=0;j<n;j++)if(!used[j]){
                if(distance[current][j]<best[j]){best[j]=distance[current][j];parent[j]=current;}
                if(next<0||best[j]<best[next])next=j;
            }
            include(parent[next],next);current=next;
        }
    }
    for(let i=0;i<n;i++)for(let j=i+1;j<n;j++){
        if(candidates&&!candidates.has(i*n+j))continue;
        const a=members[i],b=members[j],degrees=distance[i][j];
        if(include&&!include(a,b))continue;
        if(degrees<1e-6||180-degrees<1e-8||!accepted(a,b))continue;
        let clearance=Infinity;
        for(let k=0;k<n&&clearance>=.04;k++)if(k!==i&&k!==j)clearance=Math.min(clearance,arcDistance(members[k].direction,a.direction,b.direction));
        if(clearance<.04)continue;
        // Local scale lets sparse bright pairs coexist with finer limbs. This
        // is inspired by multiscale proximity, not the GC/Delaunay model itself.
        const localScale=(scale[i]+scale[j])/2,bright=Math.max(a.app_mag,b.app_mag);
        const crowding=Math.max(0,1-clearance/Math.max(.15,Math.min(1.2,localScale*.15)))**2;
        const cost=(degrees/localScale)**.9*(bright<=3?.78:1+.1*Math.max(0,bright-3))+1.5*crowding;
        options.push({from:a.id,to:b.id,degrees,cost});
    }
    return options;
}

export function regionalGraph(members,required,rng,intent,accepted,geometryCache=new Map()){
    // This cache lives for one sampleRegion call only: no stale source or border
    // can enter it. Keep the member order, hence the same random-number assignment.
    const key=JSON.stringify(members.map(s=>s.id));
    if(!geometryCache.has(key))geometryCache.set(key,graphOptions(members,accepted));
    const lookup=new Map(members.map(s=>[s.id,s]));
    const options=geometryCache.get(key).map(e=>({...e,jitter:intent.temperature*noise(rng)}));
    const edges=[],adj=new Map(members.map(s=>[s.id,[]])),parent=new Map(members.map(s=>[s.id,s.id]));
    const root=id=>{let p=id;while(parent.get(p)!==p)p=parent.get(p);while(id!==p){const next=parent.get(id);parent.set(id,p);id=next;}return p;};
    const canAdd=e=>adj.get(e.from).length<4&&adj.get(e.to).length<4&&!edges.some(other=>arcsCross(e,other,lookup));
    const add=e=>{edges.push({from:e.from,to:e.to,degrees:e.degrees});adj.get(e.from).push(e.to);adj.get(e.to).push(e.from);parent.set(root(e.from),root(e.to));};
    const extensionCost=(id,other)=>{
        const neighbours=adj.get(id),degree=neighbours.length;
        const angles=neighbours.map(n=>junctionAngle(lookup.get(id).direction,lookup.get(other).direction,lookup.get(n).direction));
        const sharp=angles.reduce((n,a)=>n+4*acuteCost(a),0);
        if(degree>=2)return sharp+intent.branchCost*(degree-1)**2+
            .45*neighbours.filter(n=>adj.get(n).length>=3).length;
        return sharp+(degree===1?intent.continuation*(1+Math.cos(angles[0]*Math.PI/180))*.2:0);
    };
    // Re-score after each addition: attachment degree, continuation and paired
    // subgroups change the graph, not only the order of almost-equal MST edges.
    let expanded=false;
    while(true){
        const branches=[...adj.values()].filter(a=>a.length>=3).length;
        const choices=options.filter(e=>root(e.from)!==root(e.to)&&adj.get(e.from).length<4&&adj.get(e.to).length<4).map(e=>({e,
            score:e.cost+e.jitter+extensionCost(e.from,e.to)+extensionCost(e.to,e.from)+
                1.5*Math.max(0,branches+Number(adj.get(e.from).length===2)+Number(adj.get(e.to).length===2)-intent.budget.branches)+
                (intent.kind==='paired'&&intent.lobe(lookup.get(e.from))!==intent.lobe(lookup.get(e.to))?1.4:0)}));
        choices.sort((a,b)=>a.score-b.score||edgeKey(a.e.from,a.e.to).localeCompare(edgeKey(b.e.from,b.e.to)));
        const choice=choices.find(({e})=>canAdd(e));
        if(choice){add(choice.e);continue;}
        if(expanded||members.length<=32||new Set(members.map(s=>root(s.id))).size<=1)break;
        expanded=true;
        // If a geometric MST bridge is outside the region, its endpoints are
        // not the only possible connection. Lazily inspect omitted cross-piece
        // pairs too; adding them still obeys clearance, degree and crossing rules.
        const groups=new Map();for(const s of members){const r=root(s.id);if(!groups.has(r))groups.set(r,[]);groups.get(r).push(s.id);}
        const bridgeKey=key+'/bridges/'+JSON.stringify([...groups.values()].map(ids=>ids.sort()).sort((a,b)=>a[0].localeCompare(b[0])));
        if(!geometryCache.has(bridgeKey))geometryCache.set(bridgeKey,graphOptions(members,accepted,(a,b)=>root(a.id)!==root(b.id)));
        const existing=new Set(options.map(e=>edgeKey(e.from,e.to)));
        for(const e of geometryCache.get(bridgeKey))if(!existing.has(edgeKey(e.from,e.to)))options.push({...e,jitter:intent.temperature*noise(rng)});
    }
    const loopNodes=new Set();
    for(let i=0;i<intent.loops;i++){
        const choices=[],previous=figureQuality(members,edges,intent.budget);
        for(const e of options){
            if(edges.some(x=>edgeKey(x.from,x.to)===edgeKey(e.from,e.to))||!canAdd(e)||adj.get(e.from).length>=3||adj.get(e.to).length>=3)continue;
            if(previous.branches+Number(adj.get(e.from).length===2)+Number(adj.get(e.to).length===2)>intent.budget.branches)continue;
            const path=pathBetween(e.from,e.to,edges);if(!path||path.length<3||path.length>12)continue;
            const perimeter=path.slice(1).reduce((n,id,k)=>n+separation(lookup.get(path[k]).direction,lookup.get(id).direction),0);
            const area=compactness(path,lookup);if(e.degrees/perimeter>.85||area<.028||path.some(id=>loopNodes.has(id)))continue;
            const angles=[...adj.get(e.from).map(id=>junctionAngle(lookup.get(e.from).direction,lookup.get(e.to).direction,lookup.get(id).direction)),
                ...adj.get(e.to).map(id=>junctionAngle(lookup.get(e.to).direction,lookup.get(e.from).direction,lookup.get(id).direction))];
            if(angles.some(a=>a<32))continue;
            const q=figureQuality(members,[...edges,e],intent.budget);
            if(q.branches>intent.budget.branches||q.loss>previous.loss+.45)continue;
            choices.push({e,path,score:e.cost+intent.temperature*noise(rng)-area*18+
                Math.abs(path.length-intent.loopSize)*.18+q.loss});
        }
        choices.sort((a,b)=>a.score-b.score);if(!choices.length)break;
        add(choices[0].e);for(const id of choices[0].path)loopNodes.add(id);
    }
    const {core,coreEdges}=simplifyCore(members,edges,required,intent.budget,rng);
    return {core,coreEdges,edges};
}

// A skeleton has its own visual budget. Remove redundant cycle edges before
// pruning leaves; required vertices stay, and retained vertices remain connected
// whenever they were connected in the full figure.
export function simplifyCore(members,edges,required,budget,rng=()=>.5){
    let core=[...members],coreEdges=edges.map(e=>({...e}));
    const coreBudget={...budget,branches:budget.coreBranches},maxLoops=Math.min(1,budget.loops);
    const quality=edges=>figureQuality(core,edges,coreBudget).loss;
    while(true){
        const stats=structure(core.map(s=>s.id),coreEdges.map(e=>[e.from,e.to]));
        if(!stats.loops)break;
        const before=quality(coreEdges);
        const options=[];
        for(let i=0;i<coreEdges.length;i++){
            const e=coreEdges[i],next=coreEdges.filter((_,j)=>j!==i);
            if(!pathBetween(e.from,e.to,next))continue;
            const loss=quality(next);
            if(stats.loops>maxLoops||loss<before-.6)options.push({next,loss,edge:e});
        }
        if(!options.length)break;
        options.sort((a,b)=>a.loss-b.loss||b.edge.degrees-a.edge.degrees||edgeKey(a.edge.from,a.edge.to).localeCompare(edgeKey(b.edge.from,b.edge.to)));
        coreEdges=options[0].next;
    }
    const coreGoal=Math.max(budget.coreStars,required.size);
    while(core.length>coreGoal){
        const removable=core.filter(s=>!required.has(s.id)&&coreEdges.filter(e=>e.from===s.id||e.to===s.id).length<=1);
        if(!removable.length)break;
        const ranked=removable.map(s=>({s,score:s.app_mag+rng()*.25-
            figureQuality(core.filter(p=>p.id!==s.id),coreEdges.filter(e=>e.from!==s.id&&e.to!==s.id),coreBudget,{parallel:false}).loss})).sort((a,b)=>b.score-a.score||a.s.id.localeCompare(b.s.id));
        const id=ranked[0].s.id;core=core.filter(s=>s.id!==id);coreEdges=coreEdges.filter(e=>e.from!==id&&e.to!==id);
    }
    return {core,coreEdges};
}
