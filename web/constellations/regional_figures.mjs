import {separation} from './geometry.mjs?revision=candidate-editor-band-1';
import {randomFrom} from './generator_ordered.mjs';
import {majorStars} from './bright_figures.mjs';
import {arcDistance} from './figure.mjs';
import {cellOf,unpackGrid} from './territories.mjs';
import {manualArcCells,edgeKey,normalizeFigures} from './manual_figures.mjs';
import {regionalGraph,structure} from './regional_graph.mjs';
import {complexityBudget,figureQuality} from './regional_quality.mjs';

export const REGIONAL_METHOD='regional-figure-sampling-3';
const byBrightness=(a,b)=>a.app_mag-b.app_mag||a.id.localeCompare(b.id);
const signature=f=>JSON.stringify(normalizeFigures([f])[0]);
const regionFigure=(r,index)=>({index,members:r.members.map(s=>s.id),coreMembers:r.variants[0].members,
    edges:r.variants[1].edges.map(e=>[e.from,e.to]),coreEdges:r.variants[0].edges.map(e=>[e.from,e.to])});
const median=a=>a.length?[...a].sort((x,y)=>x-y)[Math.floor(a.length/2)]:0;
const noise=rng=>Math.log(-Math.log(Math.max(1e-12,rng())));
function draw(items,weight,rng){
    let chosen,best=Infinity;
    for(const item of items){const key=-Math.log(Math.max(1e-12,rng()))/Math.max(1e-12,weight(item));if(key<best){best=key;chosen=item;}}
    return chosen;
}
function context(pool){
    const distance=new Map(),gap=(a,b)=>{
        const key=edgeKey(a.id,b.id);if(!distance.has(key))distance.set(key,separation(a.direction,b.direction));return distance.get(key);
    };
    const contrast=new Map(),nearest=[];
    for(const s of pool){
        const neighbours=pool.filter(p=>p!==s).map(p=>({p,d:gap(s,p)})).sort((a,b)=>a.d-b.d||a.p.id.localeCompare(b.p.id));
        // Relative apparent brightness, not intrinsic stellar absolute magnitude.
        contrast.set(s.id,Math.max(0,median(neighbours.slice(0,6).map(x=>x.p.app_mag))-s.app_mag));
        if(neighbours.length)nearest.push(neighbours[0].d);
    }
    return {gap,contrast,spacing:Math.max(.1,median(nearest))};
}
function makeIntent(pool,ctx,rng,budget,kind){
    const brightness=.18+rng()*.2,salience=s=>10**(-brightness*(s.app_mag-3))*Math.exp(.4*ctx.contrast.get(s.id));
    const a=draw(pool,salience,rng),b=draw(pool,s=>salience(s)*(.2+ctx.gap(a,s)/ctx.spacing)**1.2,rng);
    const span=Math.max(2,median(pool.map(s=>ctx.gap(a,s)))),width=Math.max(1,span*(.25+rng()*.5));
    const spacing=ctx.spacing*(.5+rng()),radius=span*(.65+rng()*.5),ab=ctx.gap(a,b);
    const lobe=s=>ctx.gap(a,s)<ctx.gap(b,s)?0:1;
    const field=s=>{
        const da=ctx.gap(a,s),db=ctx.gap(b,s);
        if(kind==='trail'&&ab>1e-6&&ab<179.999)return .22+Math.exp(-.5*(arcDistance(s.direction,a.direction,b.direction)/width)**2);
        if(kind==='ring')return .22+Math.exp(-.5*((da-radius)/width)**2);
        return .22+Math.exp(-.5*((kind==='paired'?Math.min(da,db):da)/width)**2);
    };
    const loops=Math.min(budget.loops,kind==='trail'||kind==='fork'?0:kind==='paired'?2:1);
    return {kind,budget,salience,field,spacing,span,lobe,loops,loopSize:4+Math.floor(rng()*5),temperature:.25+rng()*.3,
        branchCost:kind==='fork'?.35+rng()*.3:kind==='trail'?1+rng():kind==='ring'?.8+rng():.5+rng()*.5,
        continuation:kind==='trail'?.6+rng():kind==='paired'?.8:.3};
}
function selectMembers(pool,major,goal,intent,ctx,rng){
    const members=[...major],selected=new Set(major.map(s=>s.id));
    while(members.length<goal){
        const remaining=pool.filter(s=>!selected.has(s.id));
        const star=draw(remaining,s=>{
            const nearest=Math.min(...members.map(m=>ctx.gap(s,m)));
            // Soft repulsion avoids repeatedly filling the same clump. A positive
            // floor keeps every eligible optional star reachable by sampling.
            const spacing=.2+.8*(1-Math.exp(-.5*(nearest/intent.spacing)**2));
            return intent.salience(s)*intent.field(s)*spacing/(1+nearest/(intent.span*2));
        },rng);
        if(!star)break;members.push(star);selected.add(star.id);
    }
    return members;
}
function setDistance(a,b){
    const A=new Set(a),B=new Set(b),shared=[...A].filter(id=>B.has(id)).length,total=A.size+B.size-shared;
    return total?1-shared/total:0;
}
function novelty(figure,before,required){
    const a=structure(figure.members,figure.edges),b=structure(before.members,before.edges);
    const topology=(Number(a.loops!==b.loops)+Number(a.branches!==b.branches)+Number(a.tips!==b.tips)+Math.min(1,Math.abs(a.elongation-b.elongation)*3))/4;
    return .25*setDistance(figure.members.filter(id=>!required.has(id)),before.members.filter(id=>!required.has(id)))+
        .4*setDistance(figure.edges.map(e=>edgeKey(...e)),before.edges.map(e=>edgeKey(...e)))+.35*topology;
}
// Shared by initial layout, automatic bright-star completion and regional edits.
// Outline anchors can be mandatory in the full figure without forcing them into
// the bright skeleton. Geometry and available stars are supplied by the caller.
export function sampleFigure({pool,major,coreRequired=new Set(major.map(s=>s.id)),index=0,seed,style='balanced',
    accepted=()=>true,before={index,members:[],coreMembers:[],edges:[],coreEdges:[]},attempts=8,acceptFigure=()=>true}){
    const profile=complexityBudget(style,0),required=new Set(major.map(s=>s.id)),graphCache=new Map();
    const beforeSignature=signature(before),ctx=context(pool);
    const plan=randomFrom(`${seed}/${index}/intent`),kind=['trail','fork','ring','paired','body'][Math.floor(plan()*5)],proposals=[];
    const goal=Math.min(pool.length,Math.max(required.size,profile.minimum+Math.floor(plan()*(profile.maximum-profile.minimum+1))));
    const budget=complexityBudget(style,goal);
    for(let attempt=0;attempt<attempts;attempt++){
        const rng=randomFrom(`${seed}/${REGIONAL_METHOD}/${attempt}`),intent=makeIntent(pool,ctx,rng,budget,kind);
        const members=selectMembers(pool,major,goal,intent,ctx,rng),g=regionalGraph(members,coreRequired,rng,intent,accepted,graphCache);
        if(!acceptFigure(members,g))continue;
        const figure=normalizeFigures([{index,members:members.map(s=>s.id),coreMembers:g.core.map(s=>s.id),
            edges:g.edges.map(e=>[e.from,e.to]),coreEdges:g.coreEdges.map(e=>[e.from,e.to])}])[0];
        const stats=structure(figure.members,figure.edges),changed=signature(figure)!==beforeSignature;
        const meanLength=g.edges.reduce((n,e)=>n+e.degrees,0)/Math.max(1,g.edges.length);
        const fullQuality=figureQuality(members,g.edges,budget),coreQuality=figureQuality(g.core,g.coreEdges,{...budget,branches:budget.coreBranches});
        const loss=fullQuality.loss+.6*coreQuality.loss+.025*meanLength;
        const score=.5*novelty(figure,before,required)-.15*noise(rng);
        proposals.push({figure,method:REGIONAL_METHOD,seed,style,attempt,candidateCount:pool.length,
            importantStars:major.map(s=>s.id),changed,stats,loss,score});
    }
    // Prefer the fewest components found by this heuristic. The remaining
    // constraints can still leave pieces; never join them with an illegal edge.
    if(!proposals.length)return null;
    const components=Math.min(...proposals.map(p=>p.stats.components)),connected=proposals.filter(p=>p.stats.components===components);
    // Novelty competes only inside a narrow visual-quality band. It cannot pay
    // for a large increase in sharp angles, crowded branches or parallel arcs.
    const best=Math.min(...connected.map(p=>p.loss)),eligible=connected.filter(p=>p.loss<=best+.45);
    const changed=eligible.filter(p=>p.changed),choices=changed.length?changed:eligible;
    choices.sort((a,b)=>b.score-a.score);
    const {stats,loss,score,...result}=choices[0];return result;
}

export function sampleRegion(data,source,index,seed,style='balanced'){
    if(!Number.isInteger(index)||index<0||index>=15||!data.territories)throw Error('请选择有边界的星区');
    if(typeof seed!=='string'||!seed.length||seed.length>200)throw Error('本区采样种子无效');
    complexityBudget(style,0);
    const cells=unpackGrid(data.territories),group=source.filter(s=>cells[cellOf(s.direction)]===index).sort(byBrightness);
    if(!group.length)throw Error('当前星区内没有可用恒星；请先扩大边界，再重新生成');
    const used=new Set(data.regions.filter((r,i)=>i!==index).flatMap(r=>r.members.map(s=>s.id))),major=majorStars(group);
    if(major.some(s=>used.has(s.id)))throw Error('本区的重要亮星仍被邻座使用；请先调整该成员或边界的归属');
    const required=new Set(major.map(s=>s.id)),pool=group.filter(s=>!used.has(s.id)&&(s.app_mag<=4.5||required.has(s.id))),accepts=new Map();
    const accepted=(a,b)=>{
        const key=edgeKey(a.id,b.id);if(!accepts.has(key))accepts.set(key,manualArcCells(a.direction,b.direction).every(k=>cells[k]===index));return accepts.get(key);
    };
    const before=normalizeFigures([regionFigure(data.regions[index],index)])[0];
    return {...sampleFigure({pool,major,index,seed,style,accepted,before}),regionStarCount:group.length};
}
