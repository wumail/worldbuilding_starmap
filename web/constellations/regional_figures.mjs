import {separation,cross,unit} from './geometry.mjs?revision=candidate-editor-band-1';
import {randomFrom} from './generator_ordered.mjs';
import {majorStars} from './bright_figures.mjs';
import {arcDistance,pathBetween} from './figure.mjs';
import {cellOf,unpackGrid} from './territories.mjs';
import {manualArcCells,edgeKey,normalizeFigures} from './manual_figures.mjs';

const byBrightness=(a,b)=>a.app_mag-b.app_mag||a.id.localeCompare(b.id);
const signature=f=>JSON.stringify(normalizeFigures([f])[0]);
const regionFigure=(r,index)=>({index,members:r.members.map(s=>s.id),coreMembers:r.variants[0].members,
    edges:r.variants[1].edges.map(e=>[e.from,e.to]),coreEdges:r.variants[0].edges.map(e=>[e.from,e.to])});
const insideArc=(a,b,p)=>Math.abs(separation(a,p)+separation(p,b)-separation(a,b))<1e-7;
function arcsCross(a,b,lookup){
    if([a.from,a.to].some(id=>id===b.from||id===b.to))return false;
    const [p,q,r,s]=[a.from,a.to,b.from,b.to].map(id=>lookup.get(id).direction),n=cross(cross(p,q),cross(r,s));
    if(Math.hypot(...n)<1e-12)return false;
    const v=unit(n);return [v,v.map(x=>-x)].some(v=>insideArc(p,q,v)&&insideArc(r,s,v));
}
function graph(members,required,rng,style,accepted){
    const lookup=new Map(members.map(s=>[s.id,s])),options=[];
    for(let i=0;i<members.length;i++)for(let j=i+1;j<members.length;j++){
        const a=members[i],b=members[j],degrees=separation(a.direction,b.direction);
        if(degrees<.000001||180-degrees<1e-8||!accepted(a,b)||members.some((s,k)=>k!==i&&k!==j&&arcDistance(s.direction,a.direction,b.direction)<.04))continue;
        options.push({from:a.id,to:b.id,degrees,cost:degrees*(.7+rng()*.6)});
    }
    options.sort((a,b)=>a.cost-b.cost||edgeKey(a.from,a.to).localeCompare(edgeKey(b.from,b.to)));
    const edges=[],degree=new Map(members.map(s=>[s.id,0]));
    const canAdd=e=>degree.get(e.from)<4&&degree.get(e.to)<4&&!edges.some(other=>arcsCross(e,other,lookup));
    const add=e=>{edges.push({from:e.from,to:e.to,degrees:e.degrees});degree.set(e.from,degree.get(e.from)+1);degree.set(e.to,degree.get(e.to)+1);};
    // A forest is intentional: separate pieces or holes must never be bridged
    // by a line outside the actual current region merely to force connectivity.
    for(const e of options)if(canAdd(e)&&!pathBetween(e.from,e.to,edges))add(e);
    let core=[...members],coreEdges=edges.map(e=>({...e}));
    while(core.length>Math.max(7,required.size)){
        const removable=core.filter(s=>!required.has(s.id)&&coreEdges.filter(e=>e.from===s.id||e.to===s.id).length<=1).sort((a,b)=>byBrightness(b,a));
        if(!removable.length)break;const id=removable[0].id;core=core.filter(s=>s.id!==id);coreEdges=coreEdges.filter(e=>e.from!==id&&e.to!==id);
    }
    const loopGoal=style==='rich'?1+Math.floor(rng()*3):Math.floor(rng()*2);
    for(let i=0;i<loopGoal;i++){
        const optionsLeft=options.filter(e=>{
            if(!canAdd(e)||edges.some(x=>edgeKey(x.from,x.to)===edgeKey(e.from,e.to)))return false;
            const path=pathBetween(e.from,e.to,edges);if(!path||path.length<4||path.length>7)return false;
            const length=path.slice(1).reduce((sum,id,i)=>sum+separation(lookup.get(path[i]).direction,lookup.get(id).direction),0);
            return e.degrees/length<.9;
        });
        if(!optionsLeft.length)break;add(optionsLeft[Math.floor(rng()*Math.min(8,optionsLeft.length))]);
    }
    return {core,coreEdges,edges};
}
export function sampleRegion(data,source,index,seed,style=data.recipe?.style??'rich'){
    if(!Number.isInteger(index)||index<0||index>=15||!data.territories)throw Error('请选择有边界的星区');
    if(typeof seed!=='string'||!seed.length||seed.length>200)throw Error('本区采样种子无效');
    if(!['rich','balanced'].includes(style))throw Error('本区星形复杂度无效');
    const cells=unpackGrid(data.territories),group=source.filter(s=>cells[cellOf(s.direction)]===index).sort(byBrightness);
    if(!group.length)throw Error('当前星区内没有可用恒星；请先扩大边界，再重新生成');
    const used=new Set(data.regions.filter((r,i)=>i!==index).flatMap(r=>r.members.map(s=>s.id))),major=majorStars(group);
    if(major.some(s=>used.has(s.id)))throw Error('本区的重要亮星仍被邻座使用；请先调整该成员或边界的归属');
    const required=new Set(major.map(s=>s.id)),pool=group.filter(s=>!used.has(s.id)&&(s.app_mag<=4.5||required.has(s.id)));
    const accepts=new Map(),accepted=(a,b)=>{
        const key=edgeKey(a.id,b.id);if(!accepts.has(key))accepts.set(key,manualArcCells(a.direction,b.direction).every(k=>cells[k]===index));return accepts.get(key);
    };
    const before=signature(regionFigure(data.regions[index],index));let last;
    for(let attempt=0;attempt<8;attempt++){
        const rng=randomFrom(`${seed}/${attempt}`),minimum=style==='rich'?10:8,goal=Math.min(pool.length,Math.max(required.size,minimum+Math.floor(rng()*(style==='rich'?6:4))));
        const members=[...major];
        while(members.length<goal){
            // Weighted sampling without replacement. Brightness and proximity
            // favour recognisable stars; a seeded exponential race adds variety.
            const remaining=pool.filter(s=>!members.includes(s)).map(s=>{
                const nearest=Math.min(...members.map(m=>separation(s.direction,m.direction))),weight=Math.pow(10,-.12*(s.app_mag-2))/(1+nearest/18);
                return {s,key:-Math.log(Math.max(1e-12,rng()))/weight};
            }).sort((a,b)=>a.key-b.key||a.s.id.localeCompare(b.s.id));
            if(!remaining.length)break;members.push(remaining[0].s);
        }
        const g=graph(members,required,rng,style,accepted),figure=normalizeFigures([{index,members:members.map(s=>s.id),coreMembers:g.core.map(s=>s.id),edges:g.edges.map(e=>[e.from,e.to]),coreEdges:g.coreEdges.map(e=>[e.from,e.to])}])[0];
        last={figure,method:'regional-figure-sampling-1',seed,attempt,candidateCount:pool.length,regionStarCount:group.length,importantStars:major.map(s=>s.id),changed:signature(figure)!==before};
        if(last.changed)return last;
    }
    return last;
}
