import {vector,tangentFrame,projectLocal} from './geometry.mjs?revision=candidate-editor-band-1';
import {randomFrom,equivalentDraw} from './generator_ordered.mjs';
import {findGroups,normalizeRecipe as normalizeFree,validateGeometry} from './generator_free.mjs?revision=candidate-editor-band-1';
import {chooseFigure,RULES} from './figure.mjs?revision=figure-fit-2';
import {growTerritories,packGrid,unpackGrid,cellOf,arcCells,territoryBoundary,territoryIntervals,fromEquatorial} from './territories.mjs?revision=figure-fit-2';
import {CLEANUP,cleanBoundaries} from './boundary_cleanup.mjs?revision=figure-fit-2';
import {shapeEnvelope,envelopeDistance,envelopeMask,cellDirections,CELL_RADIUS_DEGREES} from './shape_envelope.mjs';
import {ECLIPTIC_POLICY,acceptableWidths} from './generator_ecliptic.mjs';
export {RULES,DEFAULT_SEED} from './generator_free.mjs?revision=candidate-editor-band-1';
export const ALGORITHM='terrax-zodiac-draw-6';
export const ENVELOPE_POLICY=Object.freeze({method:'spherical-hull-margin-1',maximumMarginDegrees:8,selectionLatitudeScaleDegrees:12});
const brief=s=>({id:s.id,app_mag:s.app_mag,color_hex:s.color_hex,distance_pc:s.distance_pc,longitude:s.longitude,latitude:s.latitude,direction:[...s.direction]});
const rays=Array.from({length:720},(_,i)=>cellDirections[cellOf(vector(i*.5+.25,0))]);
const margin=ENVELOPE_POLICY.maximumMarginDegrees,centerLimit=margin-CELL_RADIUS_DEGREES;

export function normalizeRecipe(value={}){
    if(value.algorithm&&value.algorithm!==ALGORITHM)throw Error('这份留存使用了不同的抽卡算法，请使用对应版本查看。');
    return {...normalizeFree({...value,algorithm:'terrax-zodiac-draw-3'}),algorithm:ALGORITHM};
}

function horizontalFigure(f,ratio=1.2){
    const frame=tangentFrame(f.center.longitude,f.center.latitude),xy=f.members.map(s=>projectLocal(s.direction,frame,0,0,1));
    const width=Math.max(...xy.map(p=>p.x))-Math.min(...xy.map(p=>p.x)),height=Math.max(...xy.map(p=>p.y))-Math.min(...xy.map(p=>p.y));
    return width>=16&&width>height*ratio;
}

function fittingVariant(group,reference,seed,style,cells,index,rules=RULES){
    const envelope=shapeEnvelope(reference.members),ids=envelope.ids;
    group=group.filter(s=>envelopeDistance(envelope,s.direction)<=1.5);
    const required=ids.map(id=>group.find(s=>s.id===id));
    if(required.some(s=>!s))return null;
    const acceptArc=(a,b)=>arcCells(a,b).every(k=>cells[k]===index);
    // Retain the accepted outline anchors, so a compact reroll cannot leave a
    // large region unsupported. Other members and the line graph may vary.
    for(let attempt=0;attempt<6;attempt++){
        const f=chooseFigure(group,`${seed}/fit/${attempt}`,style,acceptArc,required,null,{rules});
        if(f&&(!horizontalFigure(reference)||horizontalFigure(f)))return f;
    }
    return reference;
}

const partitionCache=new Map();
function partition(pool,seed,rules=RULES){
    // Same immutable candidate values and root seed define the same reference
    // partition across styles and local rerolls. Copies isolate exported data.
    const key=`${seed}\n${JSON.stringify(pool)}\n${JSON.stringify(rules)}`;
    if(partitionCache.has(key))return structuredClone(partitionCache.get(key));
    const result=searchPartition(pool,seed,rules);
    partitionCache.set(key,structuredClone(result));
    if(partitionCache.size>4)partitionCache.delete(partitionCache.keys().next().value);
    return result;
}

function searchPartition(pool,seed,rules=RULES){
    const rng=randomFrom(`${seed}/figures-first`),failures={};
    const miss=s=>{failures[s]=(failures[s]??0)+1;};
    for(let attempt=0;attempt<160;attempt++){
        // A soft selection preference, never a longitude strip or a change to
        // source magnitude. The full requested band remains in the pool.
        const groups=findGroups(pool,rng,s=>Math.exp(-((s.latitude/ENVELOPE_POLICY.selectionLatitudeScaleDegrees)**2)));
        if(!groups||groups.some(g=>g.length<10||g.filter(s=>s.app_mag<=4).length<7)){miss('pool');continue;}
        const options=groups.map((g,i)=>{
            const choices=[];
            for(let k=0;k<16;k++){
                const selection=k%4===0?g.filter(s=>Math.abs(s.latitude)<11):k%4===1?g.filter(s=>Math.abs(s.latitude)<16):g;
                const anchors=selection.filter(s=>s.app_mag<=4&&Math.abs(s.latitude)<12).sort((a,b)=>a.app_mag+.07*Math.abs(a.latitude)-b.app_mag-.07*Math.abs(b.latitude)).slice(0,6);
                const required=k>=4&&anchors.length?[anchors[k%anchors.length]]:[];
                if(k>=12&&anchors.length>1)required.push(anchors[(k+1)%anchors.length]);
                // A connected 10–11-star reference supports both viewing
                // complexities without shrinking its accepted outer anchors.
                const f=chooseFigure(selection,`${seed}/trial/${attempt}/${i}/${k}`,'balanced',()=>true,required,10+k%2,{rules});if(!f||f.members.length<10)continue;
                const envelope=shapeEnvelope(f.members),support=rays.map(v=>envelopeDistance(envelope,v)<=centerLimit),width=support.filter(Boolean).length*.5;
                const frame=tangentFrame(f.center.longitude,f.center.latitude),xy=f.members.map(s=>projectLocal(s.direction,frame,0,0,1));
                const w=Math.max(...xy.map(s=>s.x))-Math.min(...xy.map(s=>s.x)),h=Math.max(...xy.map(s=>s.y))-Math.min(...xy.map(s=>s.y)),horizontal=w>=18&&w>h*1.4;
                choices.push({f,envelope,support,width,horizontal});
            }
            return choices.sort((a,b)=>b.width-a.width);
        });
        if(options.some(g=>!g.length)){miss('figure');continue;}
        const chosen=options.map(g=>g[0]);
        if(chosen.some(c=>c.width<6)||chosen.filter(c=>c.width<16).length>1){miss('small-support');continue;}
        // Select actual alternative figures to cover gaps. The margin is fixed;
        // there is no iterative region bias and no enlargement to meet a width.
        for(let pass=0;pass<4;pass++)for(let i=0;i<15;i++){
            const counts=rays.map((_,k)=>chosen.filter((c,j)=>j!==i&&c.support[k]).length),narrow=chosen.filter((c,j)=>j!==i&&c.width<16).length;
            let best,score=Infinity;
            for(const c of options[i]){
                if(c.width<(narrow?16:6))continue;
                const gap=counts.filter((n,k)=>!n&&!c.support[k]).length,quality=gap*100-c.width*.01-(c.horizontal?1:0);
                if(quality<score){score=quality;best=c;}
            }
            if(best)chosen[i]=best;
        }
        if(rays.some((_,k)=>!chosen.some(c=>c.support[k]))){miss('gap');continue;}
        if(chosen.filter(c=>horizontalFigure(c.f)).length<2){miss('orientation');continue;}
        const mask=envelopeMask(chosen.map(c=>c.envelope),margin),canAssign=(k,i)=>i===15||!!(mask[k]&(1<<i));
        const raw=growTerritories(chosen.map(c=>c.f),[],undefined,canAssign);
        if(!raw){miss('collision');continue;}
        const sections=territoryIntervals(raw),spans=chosen.map((_,i)=>sections.filter(s=>s.index===i).reduce((n,s)=>n+s.end-s.start,0));
        // Exact grid intersections are authoritative; the 0.5° shortlist above
        // only saves computation and is not accepted as proof of coverage.
        if(sections.some(s=>s.index===15)||!acceptableWidths(spans)){miss('width');continue;}
        try{chosen.forEach((_,i)=>territoryBoundary(raw,i));}catch{miss('topology');continue;}
        const clean=cleanBoundaries({territories:packGrid(raw)},pool,canAssign),cells=clean.labels;
        const pools=chosen.map(()=>[]);for(const s of pool){const i=cells[cellOf(s.direction)];if(i<15)pools[i].push(s);}
        return {figures:chosen.map(c=>c.f),pools,cells,sections,boundaries:clean.boundaries};
    }
    throw Error('此种子暂未找到同时贴合星形并满足黄道宽度的十五区方案，请换一个种子；当前轮次已保留。',{cause:failures});
}

export function generateDraw(source,meta,value={}){
    return generateWithRules(source,meta,value,RULES);
}
// Versioned callers can widen initial sampling without changing draw-6 defaults.
export function generateSampledReference(source,meta,value,halfWidth){
    if(!Number.isInteger(halfWidth)||halfWidth<15||halfWidth>60)throw Error('采样范围应为黄道两侧各 15°–60°，精度 1°');
    return generateWithRules(source,meta,value,{...RULES,searchLatitude:halfWidth});
}
function generateWithRules(source,meta,value,rules){
    const recipe=normalizeRecipe(value),stars=source.map(brief),pool=stars.filter(s=>s.app_mag<=rules.candidateMagnitude&&Math.abs(s.latitude)<=rules.searchLatitude),p=partition(pool,recipe.seed,rules);
    const regions=p.pools.map((group,i)=>{
        const first=recipe.shapeSeeds[i]===`${recipe.seed}/shape/${i}`;
        const f=first&&recipe.style==='balanced'?p.figures[i]:fittingVariant(group,p.figures[i],recipe.shapeSeeds[i],recipe.style,p.cells,i,rules);
        if(!f)throw Error(`候选 ${i+1} 暂未找到保留轮廓锚星的连通星形，请重抽。本轮结果未替换当前方案。`);
        const sections=p.sections.filter(s=>s.index===i).map(({start,end})=>({start,end}));
        return {id:`Z${String(i+1).padStart(2,'0')}`,label:`候选 ${String(i+1).padStart(2,'0')}`,site:vector(p.figures[i].center.longitude,p.figures[i].center.latitude),center:f.center,
            boundary:p.boundaries[i],polygon:p.boundaries[i].map(([ra,dec])=>fromEquatorial(vector(ra,dec))),intervals:sections,eclipticSpan:sections.reduce((n,s)=>n+s.end-s.start,0),candidateCount:group.length,brightCount:group.filter(s=>s.app_mag<=4).length,
            members:f.members,variants:[{id:'core',label:'亮星骨架',members:f.core.map(s=>s.id),edges:f.coreEdges},{id:'extended',label:'完整星形',members:f.members.map(s=>s.id),edges:f.edges}],
            structure:f.structure,brightest:Math.min(...f.members.map(s=>s.app_mag)),faintest:Math.max(...f.members.map(s=>s.app_mag)),oldSectors:[...new Set(f.members.map(s=>Math.floor(s.longitude/24)+1))].sort((a,b)=>a-b)};
    });
    const data={schema:7,status:'candidate-review',catalogue:meta.catalogue,sha256:meta.sha256,epoch:'Terrax 第 0 日参考黄道',recipe,settings:rules,sourceCount:source.length,candidateCount:pool.length,eligibleCandidateCount:p.pools.reduce((n,g)=>n+g.length,0),
        selectedCoreCount:105,selectedExtendedCount:regions.reduce((n,r)=>n+r.members.length,0),territories:packGrid(p.cells),remainderSites:[],regions,eclipticPolicy:ECLIPTIC_POLICY,boundaryCleanup:CLEANUP,envelopePolicy:ENVELOPE_POLICY};
    validateWithRules(data,source,rules);return data;
}

export function validateDraw(data,source){
    return validateWithRules(data,source,RULES);
}
function validateWithRules(data,source,rules){
    const recipe=normalizeRecipe(data.recipe);
    if(!equivalentDraw(data.envelopePolicy,ENVELOPE_POLICY))throw Error('候选检查未通过：星形包络余量规则');
    if(!equivalentDraw(data.eclipticPolicy,ECLIPTIC_POLICY))throw Error('候选检查未通过：黄道宽度规则');
    if(!equivalentDraw(data.boundaryCleanup,CLEANUP))throw Error('候选检查未通过：边界整理规则');
    validateGeometry(data,source,{recipe:{...recipe,algorithm:'terrax-zodiac-draw-3'},rules});
    if(!acceptableWidths(data.regions.map(r=>r.eclipticSpan)))throw Error('候选检查未通过：黄道窄区过多或宽度超出范围');
    const cells=unpackGrid(data.territories,true),envelopes=data.regions.map(r=>shapeEnvelope(r.members));
    for(let k=0;k<cells.length;k++)if(cells[k]<15&&envelopeDistance(envelopes[cells[k]],cellDirections[k])>centerLimit+1e-8)throw Error('候选检查未通过：天区超出星形包络余量');
    return true;
}
