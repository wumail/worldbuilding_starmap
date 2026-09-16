import {vector} from './geometry.mjs?revision=candidate-editor-band-1';
import {randomFrom,equivalentDraw} from './generator_ordered.mjs';
import {findGroups,normalizeRecipe as normalizeFree,validateDraw as validateFree} from './generator_free.mjs?revision=candidate-editor-band-1';
import {chooseFigure,RULES} from './figure.mjs';
import {growTerritories,packGrid,cellOf,arcCells,territoryBoundary,territoryIntervals,fromEquatorial} from './territories.mjs';
import {CLEANUP,cleanBoundaries} from './boundary_cleanup.mjs';
export {RULES,DEFAULT_SEED} from './generator_free.mjs?revision=candidate-editor-band-1';
export const ALGORITHM='terrax-zodiac-draw-5';
export const ECLIPTIC_POLICY=Object.freeze({method:'one-narrow-region-1',minimumDegrees:6,ordinaryMinimumDegrees:16,maximumDegrees:44,maximumNarrowRegions:1});
const brief=s=>({id:s.id,app_mag:s.app_mag,color_hex:s.color_hex,distance_pc:s.distance_pc,longitude:s.longitude,latitude:s.latitude,direction:[...s.direction]});

export function normalizeRecipe(value={}){
    if(value.algorithm&&value.algorithm!==ALGORITHM)throw Error('这份留存使用了不同的抽卡算法，请使用对应版本查看。');
    return {...normalizeFree({...value,algorithm:'terrax-zodiac-draw-3'}),algorithm:ALGORITHM};
}

export function acceptableWidths(spans){
    const p=ECLIPTIC_POLICY;
    return spans.length===15&&spans.every(s=>Number.isFinite(s)&&s>=p.minimumDegrees-1e-9&&s<=p.maximumDegrees+1e-9)
        &&spans.filter(s=>s<p.ordinaryMinimumDegrees-1e-9).length<=p.maximumNarrowRegions;
}

// These are limits on the final ecliptic intersections, not longitude strips
// used to select stars. All fifteen two-dimensional figures already exist when
// the negotiation starts, and their stars and arcs remain protected.
function partition(pool,seed){
    const rng=randomFrom(`${seed}/figures-first`),failures={};
    const miss=s=>{failures[s]=(failures[s]??0)+1;};
    for(let attempt=0;attempt<72;attempt++){
        const groups=findGroups(pool,rng);
        if(!groups||groups.some(g=>g.length<12||g.filter(s=>s.app_mag<=4).length<7)){miss('pool');continue;}
        const figures=groups.map((g,i)=>chooseFigure(g,`${seed}/shape/${i}`,'rich'));
        if(figures.some(f=>!f)){miss('figure');continue;}
        const outer=[];
        for(const sign of [-1,1])for(let longitude=0;longitude<360;longitude+=18)outer.push(vector(longitude+rng()*14,sign*(35+rng()*19)));
        outer.push(vector(0,90),vector(0,-90));
        let cells,sections,spans;const bias=Array(16).fill(0);
        for(let negotiation=0;negotiation<32;negotiation++){
            cells=growTerritories(figures,outer,bias);if(!cells)break;
            sections=territoryIntervals(cells);spans=figures.map((_,i)=>sections.filter(s=>s.index===i).reduce((n,s)=>n+s.end-s.start,0));
            const uncovered=sections.some(s=>s.index>=15);
            if(!uncovered&&acceptableWidths(spans))break;
            const narrow=spans.indexOf(Math.min(...spans));
            for(let i=0;i<15;i++){
                const minimum=i===narrow?ECLIPTIC_POLICY.minimumDegrees:ECLIPTIC_POLICY.ordinaryMinimumDegrees;
                if(spans[i]<minimum)bias[i]+=Math.min(2.4,.6*(minimum-spans[i])+.2);
                else if(spans[i]>ECLIPTIC_POLICY.maximumDegrees)bias[i]-=Math.min(2.4,.6*(spans[i]-ECLIPTIC_POLICY.maximumDegrees)+.2);
                if(uncovered)bias[i]+=.8;
            }
        }
        if(!cells){miss('collision');continue;}
        if(sections.some(s=>s.index>=15)||!acceptableWidths(spans)){miss('ecliptic');continue;}
        let boundaries;
        try{boundaries=figures.map((_,i)=>territoryBoundary(cells,i));}catch{miss('topology');continue;}
        const pools=figures.map(()=>[]);for(const s of pool){const i=cells[cellOf(s.direction)];if(i<15)pools[i].push(s);}
        return {figures,pools,cells,sections,boundaries};
    }
    throw Error('此种子暂未找到兼顾星形、边界与黄道宽度的十五区方案，请换一个种子。',{cause:failures});
}

export function generateDraw(source,meta,value={}){
    const recipe=normalizeRecipe(value),stars=source.map(brief),pool=stars.filter(s=>s.app_mag<=RULES.candidateMagnitude&&Math.abs(s.latitude)<=RULES.searchLatitude),p=partition(pool,recipe.seed);
    const regions=p.pools.map((group,i)=>{
        const acceptArc=(a,b)=>arcCells(a,b).every(k=>p.cells[k]===i);
        const f=recipe.style==='rich'&&recipe.shapeSeeds[i]===`${recipe.seed}/shape/${i}`?p.figures[i]:chooseFigure(group,recipe.shapeSeeds[i],recipe.style,acceptArc);
        if(!f)throw Error(`候选 ${i+1} 暂未找到满足约束的星形，请重抽。本轮结果未替换当前方案。`);
        const sections=p.sections.filter(s=>s.index===i).map(({start,end})=>({start,end}));
        return {id:`Z${String(i+1).padStart(2,'0')}`,label:`候选 ${String(i+1).padStart(2,'0')}`,site:vector(p.figures[i].center.longitude,p.figures[i].center.latitude),center:f.center,
            boundary:p.boundaries[i],polygon:p.boundaries[i].map(([ra,dec])=>fromEquatorial(vector(ra,dec))),intervals:sections,eclipticSpan:sections.reduce((n,s)=>n+s.end-s.start,0),candidateCount:group.length,brightCount:group.filter(s=>s.app_mag<=4).length,
            members:f.members,variants:[{id:'core',label:'亮星骨架',members:f.core.map(s=>s.id),edges:f.coreEdges},{id:'extended',label:'完整星形',members:f.members.map(s=>s.id),edges:f.edges}],
            structure:f.structure,brightest:Math.min(...f.members.map(s=>s.app_mag)),faintest:Math.max(...f.members.map(s=>s.app_mag)),oldSectors:[...new Set(f.members.map(s=>Math.floor(s.longitude/24)+1))].sort((a,b)=>a-b)};
    });
    const data={schema:6,status:'candidate-review',catalogue:meta.catalogue,sha256:meta.sha256,epoch:'Terrax 第 0 日参考黄道',recipe,settings:RULES,sourceCount:source.length,candidateCount:pool.length,eligibleCandidateCount:p.pools.reduce((n,g)=>n+g.length,0),
        selectedCoreCount:105,selectedExtendedCount:regions.reduce((n,r)=>n+r.members.length,0),territories:packGrid(p.cells),remainderSites:[],regions,eclipticPolicy:ECLIPTIC_POLICY,boundaryCleanup:CLEANUP};
    const clean=cleanBoundaries(data,source);data.territories=packGrid(clean.labels);
    for(let i=0;i<15;i++){
        data.regions[i].boundary=clean.boundaries[i];
        data.regions[i].polygon=clean.boundaries[i].map(([ra,dec])=>fromEquatorial(vector(ra,dec)));
    }
    validateDraw(data,source);return data;
}

export function validateDraw(data,source){
    const recipe=normalizeRecipe(data.recipe);
    if(!equivalentDraw(data.eclipticPolicy,ECLIPTIC_POLICY))throw Error('候选检查未通过：黄道宽度规则');
    if(!equivalentDraw(data.boundaryCleanup,CLEANUP))throw Error('候选检查未通过：边界整理规则');
    validateFree({...data,recipe:{...recipe,algorithm:'terrax-zodiac-draw-3'}},source);
    if(!acceptableWidths(data.regions.map(r=>r.eclipticSpan)))throw Error('候选检查未通过：黄道窄区过多或宽度超出范围');
    return true;
}
