import {RAD,vector,coordinates,dot,cross,separation,wrap,delta} from './geometry.mjs?revision=candidate-editor-band-1';
import {cellOf,toEquatorial,fromEquatorial,unpackGrid,packGrid,territoryIntervals} from './territories.mjs';
import {normalizeEdits,gridEdits} from './boundary_edits.mjs?revision=candidate-editor-band-1';
import {connectionStats} from './candidate_edits.mjs';
import {brief,brightAudit} from './bright_figures.mjs';

// A separate replay contract: historical manual-1/2 and automatic draws keep
// their original selection, topology and aesthetic constraints.
export const FREE_EDIT_ALGORITHM='terrax-zodiac-manual-3';
export const isFreeEdit=data=>data?.recipe?.algorithm===FREE_EDIT_ALGORITHM;
export const edgeKey=(a,b)=>JSON.stringify([a,b].sort());
const fail=message=>{throw Error(message);};
function ids(value){
    if(!Array.isArray(value)||value.some(id=>typeof id!=='string'||!id.length||id.length>200)||new Set(value).size!==value.length)fail('成员编号无效或重复');
    return [...value].sort();
}
function edges(value,members){
    if(!Array.isArray(value))fail('连线列表无效');
    const known=new Set(members),seen=new Set();
    return value.map(e=>{
        if(!Array.isArray(e)||e.length!==2||e[0]===e[1]||!e.every(id=>known.has(id)))fail('连线必须连接两颗不同的现有成员');
        const pair=[...e].sort(),key=edgeKey(...pair);if(seen.has(key))fail('连线重复');seen.add(key);return pair;
    }).sort((a,b)=>a[0].localeCompare(b[0])||a[1].localeCompare(b[1]));
}
export function normalizeFigures(value=[]){
    if(!Array.isArray(value)||value.length>15)fail('手动星形记录无效');
    const seen=new Set();
    return value.map(f=>{
        if(!Number.isInteger(f.index)||f.index<0||f.index>=15||seen.has(f.index))fail('星区编号无效或重复');seen.add(f.index);
        const members=ids(f.members),coreMembers=ids(f.coreMembers);
        if(coreMembers.some(id=>!members.includes(id)))fail('骨架成员不在完整星形中');
        return {index:f.index,members,coreMembers,edges:edges(f.edges,members),coreEdges:edges(f.coreEdges,coreMembers)};
    }).sort((a,b)=>a.index-b.index);
}
function figure(r,index){return normalizeFigures([{index,members:r.members.map(s=>s.id),coreMembers:r.variants[0].members,
    edges:r.variants[1].edges.map(e=>[e.from,e.to]),coreEdges:r.variants[0].edges.map(e=>[e.from,e.to])}])[0];}
export function manualRecipe(base,next){
    const root=base.recipe,figures=next.regions.map(figure).filter(f=>JSON.stringify(f)!==JSON.stringify(figure(base.regions[f.index],f.index)));
    return {algorithm:FREE_EDIT_ALGORITHM,seed:root.seed,style:root.style,shapeSeeds:root.shapeSeeds,base:root,edits:gridEdits(base,next),figures};
}

// Analytic crossings with ALL meridians and parallels. Unlike the automatic
// short-arc helper this does not assume a narrow, non-polar candidate field.
export function manualArcCells(a,b){
    const ae=toEquatorial(a),be=toEquatorial(b),angle=separation(ae,be)*RAD,c=Math.max(-1,Math.min(1,dot(ae,be)));
    if(angle<1e-12)return [cellOf(a)];
    if(Math.PI-angle<1e-10)fail('两颗星恰好相对，无法唯一确定最短大圆连线');
    const u=be.map((x,i)=>(x-c*ae[i])/Math.sin(angle)),at=t=>ae.map((x,i)=>x*Math.cos(t)+u[i]*Math.sin(t));
    const cuts=[0,angle],add=t=>{if(t>1e-12&&t<angle-1e-12)cuts.push(t);};
    for(let ra=0;ra<180;ra++){
        const p=-Math.sin(ra*RAD)*ae[0]+Math.cos(ra*RAD)*ae[1],q=-Math.sin(ra*RAD)*u[0]+Math.cos(ra*RAD)*u[1];
        if(Math.hypot(p,q)<1e-13)continue;
        const t=Math.atan2(-p,q);for(let n=-1;n<=2;n++)add(t+n*Math.PI);
    }
    const phase=Math.atan2(u[2],ae[2]),radius=Math.hypot(ae[2],u[2]);
    if(radius>1e-13)for(let dec=-89;dec<=89;dec++){
        const ratio=Math.sin(dec*RAD)/radius;if(Math.abs(ratio)>1+1e-12)continue;
        const offset=Math.acos(Math.max(-1,Math.min(1,ratio)));
        for(const sign of [-1,1])for(let n=-1;n<=1;n++)add(phase+sign*offset+n*2*Math.PI);
    }
    cuts.sort((x,y)=>x-y);const unique=cuts.filter((v,i)=>!i||v-cuts[i-1]>1e-11),result=new Set([cellOf(a),cellOf(b)]);
    // Ownership of a boundary itself is immaterial: adjoining open intervals
    // prove containment in the closed cell union, including tangencies.
    for(let i=1;i<unique.length;i++)result.add(cellOf(fromEquatorial(at((unique[i]+unique[i-1])/2))));
    return [...result];
}

// Every region is its actual union of cells. Manual contours may include
// holes, several components or poles; no automatic shape policy is reapplied.
export function manualBoundaryRings(cells,id){
    const outgoing=new Map(),key=(x,y)=>y*360+wrap(x),add=(x,y,X,Y,dir)=>{const a=key(x,y),e={a,b:key(X,Y),dir};if(!outgoing.has(a))outgoing.set(a,[]);outgoing.get(a).push(e);};
    let count=0;
    for(let k=0;k<cells.length;k++)if(cells[k]===id){
        const x=k%360,y=Math.floor(k/360),owned=(X,Y)=>Y>=0&&Y<180&&cells[Y*360+wrap(X)]===id;
        if(!owned(x,y-1)){add(x,y,x+1,y,0);count++;}
        if(!owned(x+1,y)){add(x+1,y,x+1,y+1,1);count++;}
        if(!owned(x,y+1)){add(x+1,y+1,x,y+1,2);count++;}
        if(!owned(x-1,y)){add(x,y+1,x,y,3);count++;}
    }
    const rings=[];
    while(count){
        const start=outgoing.values().next().value[0],steps=[];let edge=start;
        do{
            steps.push(edge);const list=outgoing.get(edge.a);list.splice(list.indexOf(edge),1);if(!list.length)outgoing.delete(edge.a);count--;
            if(edge.b===start.a)break;
            const choices=outgoing.get(edge.b);if(!choices?.length)fail('边界未闭合');
            const priority=d=>[1,0,3,2].indexOf((d-edge.dir+4)%4);
            edge=[...choices].sort((a,b)=>priority(a.dir)-priority(b.dir))[0];
        }while(steps.length<=64800*4);
        // Keep long parallels split at 90 degrees so the rendering's wrapped
        // coordinate interpolation never takes the wrong way around the sky.
        const ring=steps.filter((e,i)=>steps[(i+steps.length-1)%steps.length].dir!==e.dir||(e.dir%2===0&&e.a%90===0))
            .map(e=>[e.a%360,Math.floor(e.a/360)-90]);
        if(ring.length)rings.push(ring);
    }
    return rings;
}
export const regionRings=r=>r.boundaryRings??(r.boundary?.length?[r.boundary]:[]);
export const cornerCount=r=>regionRings(r).reduce((n,ring)=>n+ring.filter((p,i)=>{
    const a=ring[(i+ring.length-1)%ring.length],b=ring[(i+1)%ring.length];return !(a[1]===p[1]&&p[1]===b[1])&&!(a[0]===p[0]&&p[0]===b[0]);
}).length,0);
export function moveManualBoundary(data,index,ringIndex,edgeIndex,target){
    if(!Number.isFinite(target))fail('请输入有限的边界坐标');
    const ring=regionRings(data.regions[index])[ringIndex],a=ring?.[edgeIndex],b=ring?.[(edgeIndex+1)%ring.length];if(!a||!b)fail('请选择一条边界');
    const horizontal=a[1]===b[1],cells=unpackGrid(data.territories),next=new Int8Array(cells);
    const start=horizontal?a[1]+90:a[0],end=horizontal?Math.round(target)+90:a[0]+delta(Math.round(target),a[0]);
    if(horizontal&&(end<0||end>180))fail('赤纬坐标应位于 −90° 至 90°');
    const at=(x,y)=>y<0||y>=180?-1:y*360+wrap(x),lo=Math.min(start,end),hi=Math.max(start,end);
    const alongA=horizontal?a[0]:a[1]+90,alongB=horizontal?a[0]+delta(b[0],a[0]):b[1]+90;
    for(let u=Math.min(alongA,alongB);u<Math.max(alongA,alongB);u++){
        const minus=horizontal?at(u,start-1):at(start-1,u),plus=horizontal?at(u,start):at(start,u),insideMinus=minus>=0&&cells[minus]===index;
        if(insideMinus===(plus>=0&&cells[plus]===index))fail('边界已变化，请重新选择');
        const outsideIndex=insideMinus?plus:minus,outside=outsideIndex<0?15:cells[outsideIndex],expanding=(end>start)===insideMinus;
        for(let v=lo;v<hi;v++){const k=horizontal?at(u,v):at(v,u);if(k<0)continue;if(expanding)next[k]=index;else if(cells[k]===index)next[k]=outside;}
    }
    return next;
}
export function containmentIssues(data){
    const cells=unpackGrid(data.territories),problems=[],used=new Map();
    for(const [i,r] of data.regions.entries()){
        const byId=new Map(r.members.map(s=>[s.id,s]));
        for(const s of r.members){
            if(used.has(s.id))problems.push(`${r.id} 与 ${used.get(s.id)} 共用了成员 ${s.id}`);used.set(s.id,r.id);
            if(cells[cellOf(s.direction)]!==i)problems.push(`${r.id} 的成员 ${s.id} 在边界外`);
        }
        const seen=new Set();for(const v of r.variants)for(const e of v.edges){
            const key=edgeKey(e.from,e.to);if(seen.has(key))continue;seen.add(key);
            if(!manualArcCells(byId.get(e.from).direction,byId.get(e.to).direction).every(k=>cells[k]===i))problems.push(`${r.id} 的连线 ${e.from} → ${e.to} 未被边界完整包住`);
        }
    }
    return problems;
}
export function applyManualFigures(base,source,rawRecipe,{draft=false}={}){
    const recipe={...rawRecipe,edits:normalizeEdits(rawRecipe.edits),figures:normalizeFigures(rawRecipe.figures)},lookup=new Map(source.map(s=>[s.id,s]));
    const cells=new Int8Array(unpackGrid(base.territories));for(const [k,owner] of recipe.edits)cells[k]=owner;
    const overrides=new Map(recipe.figures.map(f=>[f.index,f])),sections=territoryIntervals(cells),groups=Array.from({length:15},()=>[]);
    for(const s of source){const owner=cells[cellOf(s.direction)];if(owner<15)groups[owner].push(s);}
    const regions=base.regions.map((r,i)=>{
        const f=overrides.get(i);let result=structuredClone(r);
        if(f){
            const members=f.members.map(id=>{const s=lookup.get(id);if(!s)fail(`源星表中没有成员 ${id}`);return brief(s);});
            const buildEdges=list=>list.map(([from,to])=>({from,to,degrees:separation(lookup.get(from).direction,lookup.get(to).direction)}));
            const sum=members.reduce((a,s)=>a.map((x,j)=>x+s.direction[j]),[0,0,0]);
            const center=members.length?coordinates(Math.hypot(...sum)>1e-10?sum:members[0].direction):r.center;
            result={...result,center,site:vector(center.longitude,center.latitude),members,
                variants:[{id:'core',label:'亮星骨架',members:[...f.coreMembers],edges:buildEdges(f.coreEdges)},
                    {id:'extended',label:'完整星形',members:[...f.members],edges:buildEdges(f.edges)}],
                brightest:members.length?Math.min(...members.map(s=>s.app_mag)):null,faintest:members.length?Math.max(...members.map(s=>s.app_mag)):null,
                oldSectors:[...new Set(members.map(s=>Math.floor(s.longitude/24)+1))].sort((a,b)=>a-b)};
        }
        const rings=manualBoundaryRings(cells,i),intervals=sections.filter(s=>s.index===i).map(({start,end})=>({start,end}));
        return {...result,boundaryRings:rings,boundary:rings[0]??[],polygon:(rings[0]??[]).map(([ra,dec])=>fromEquatorial(vector(ra,dec))),intervals,
            eclipticSpan:intervals.reduce((n,s)=>n+s.end-s.start,0),candidateCount:groups[i].length,brightCount:groups[i].filter(s=>s.app_mag<=4).length,
            structure:connectionStats(result.variants[1].members,result.variants[1].edges)};
    });
    const {optimality,localOptimality,minimumCornersPolicy,localPolicy,boundaryCleanup,manualBoundary,manualEdits,brightPolicy,brightAudit:oldAudit,envelopePolicy,eclipticPolicy,...rest}=base;
    const data={...rest,schema:13,settings:{mode:'manual',coordinateGridDegrees:1,sourceSelection:'full-catalogue',containment:'members-and-minor-arcs'},recipe,regions,territories:packGrid(cells),sourceCount:source.length,candidateCount:source.length,
        eligibleCandidateCount:groups.reduce((n,g)=>n+g.length,0),selectedExtendedCount:regions.reduce((n,r)=>n+r.members.length,0),selectedCoreCount:regions.reduce((n,r)=>n+r.variants[0].members.length,0)};
    const issues=containmentIssues(data);if(!draft&&issues.length)fail(issues[0]);
    data.brightAudit=brightAudit(data,source,cells);
    data.manualEdits={method:'free-candidate-edit-3',status:issues.length?'draft':'validated',changedCells:recipe.edits.length,changedFigures:recipe.figures.length,
        constraints:['source-catalogue','unique-territory','orthogonal-boundaries','members-and-arcs-contained'],issues,regionCorners:regions.map(cornerCount)};
    return data;
}

// Edits change IDs and graph structure only, never stellar positions or light.
export function editManualFigure(base,current,source,index,action){
    const r=current.regions[index];if(!r)fail('请选择星座');
    const all=source.map(s=>s.id),f=figure(r,index),known=new Set(all);
    const add=id=>{
        if(!known.has(id))fail('请选择源星表中的恒星');
        const other=current.regions.find((q,i)=>i!==index&&q.members.some(s=>s.id===id));
        if(other)fail(`这颗星已是 ${other.id} 的成员，请先在该星座移除它`);
        if(!f.members.includes(id))f.members.push(id);
        // User additions are visible in both variants, not silently omitted by
        // the old automatic bright-core selection policy.
        if(!f.coreMembers.includes(id))f.coreMembers.push(id);
    };
    if(action.type==='add-member')add(action.id);
    else if(action.type==='remove-member'){
        if(!f.members.includes(action.id))fail('这颗星不是当前成员');
        f.members=f.members.filter(id=>id!==action.id);f.coreMembers=f.coreMembers.filter(id=>id!==action.id);
        f.edges=f.edges.filter(e=>!e.includes(action.id));f.coreEdges=f.coreEdges.filter(e=>!e.includes(action.id));
    }else if(action.type==='add-edge'){
        if(action.from===action.to)fail('请选择两颗不同的恒星');add(action.from);add(action.to);
        const key=edgeKey(action.from,action.to);if(f.edges.some(e=>edgeKey(...e)===key))fail('这条连线已存在');
        f.edges.push([action.from,action.to]);if(!f.coreEdges.some(e=>edgeKey(...e)===key))f.coreEdges.push([action.from,action.to]);
    }else if(action.type==='remove-edge'){
        const key=edgeKey(action.from,action.to);f.edges=f.edges.filter(e=>edgeKey(...e)!==key);f.coreEdges=f.coreEdges.filter(e=>edgeKey(...e)!==key);
    }else fail('未知星形操作');
    const recipe=manualRecipe(base,current);recipe.figures=recipe.figures.filter(q=>q.index!==index);recipe.figures.push(f);
    return applyManualFigures(base,source,recipe,{draft:true});
}
