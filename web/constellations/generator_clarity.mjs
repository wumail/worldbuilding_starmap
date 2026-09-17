import {ENVELOPE_POLICY} from './generator_fitted.mjs?revision=candidate-editor-band-1';
import {generateDraw as generateLayout,normalizeRecipe as normalizeSamplingRecipe,LOCAL_POLICY} from './generator_sampling.mjs';
import {validateGeometry,DEFAULT_SEED} from './generator_free.mjs?revision=candidate-editor-band-1';
import {equivalentDraw} from './generator_ordered.mjs';
import {ECLIPTIC_POLICY,acceptableWidths} from './generator_ecliptic.mjs';
import {shapeEnvelope,envelopeDistance,cellDirections,CELL_RADIUS_DEGREES} from './shape_envelope.mjs';
import {unpackGrid} from './territories.mjs';
import {coordinates,unit,separation,tangentFrame,projectLocal} from './geometry.mjs?revision=candidate-editor-band-1';
import {prepareMinimumSolver} from './minimum_corners.mjs';
import {RULES as BRIGHT_RULES,brief,majorStars,regionalStars,brightAudit} from './bright_figures.mjs';
import {withFigure,rebuildRegions} from './region_data.mjs';
import {samplingInfo} from './sampling.mjs';
import {manualArcCells,edgeKey} from './manual_figures.mjs';
import {sampleFigure,REGIONAL_METHOD} from './regional_figures.mjs';
import {structure,simplifyCore} from './regional_graph.mjs';
import {complexityBudget} from './regional_quality.mjs';

export {DEFAULT_SEED,prepareMinimumSolver};
export const ALGORITHM='terrax-zodiac-draw-10';
export const DEFAULT_STYLE='balanced';
// Sparse bright anchors can use longer limbs. The automatic atlas still keeps
// its common angular window and 56-degree member diameter; local edits are freer.
export const RULES=Object.freeze({...BRIGHT_RULES,minSeparation:0,maxEdge:56});
export const BRIGHT_POLICY=Object.freeze({method:'regional-bright-anchors-2',topCount:3,maximumMagnitude:3,
    finalRegionAudit:'full-catalogue',memberBudget:'soft-preserve-bright-stars-and-outline',coreBudget:'soft-preserve-bright-stars'});
export const FIGURE_POLICY=Object.freeze({method:REGIONAL_METHOD,stages:['seed-figures','fixed-boundary-reroll'],
    layoutAlgorithm:'terrax-zodiac-draw-9',qualityBeforeNovelty:true,maximumCoreLoops:1});
const cache=new Map(),byBrightness=(a,b)=>a.app_mag-b.app_mag||a.id.localeCompare(b.id);

export function normalizeRecipe(value={}){
    if(value.algorithm&&value.algorithm!==ALGORITHM)throw Error('这份轮次使用不同的抽卡算法');
    const style=value.style??DEFAULT_STYLE;complexityBudget(style,0);
    const recipe=normalizeSamplingRecipe({...value,algorithm:'terrax-zodiac-draw-9',style:style==='simple'?'balanced':style});
    return {...recipe,algorithm:ALGORITHM,style};
}

export function chooseClearFigure(group,seed,style,{anchors=[],cells=null,index=0,attempts=8}={}){
    if(group.length<3)return null;
    const bright=majorStars(group),required=new Set([...anchors,...bright.map(s=>s.id)]);
    const pool=group.filter(s=>s.app_mag<=RULES.candidateMagnitude||required.has(s.id)).sort(byBrightness);
    const major=pool.filter(s=>required.has(s.id));if(major.length!==required.size)return null;
    const accepts=new Map(),accepted=(a,b)=>{
        const key=edgeKey(a.id,b.id);if(!accepts.has(key))accepts.set(key,!cells||manualArcCells(a.direction,b.direction).every(k=>cells[k]===index));
        return accepts.get(key);
    };
    const centerOf=members=>coordinates(unit(members.reduce((v,s)=>v.map((x,k)=>x+s.direction[k]),[0,0,0])));
    const acceptFigure=(members,g)=>{
        if(structure(members.map(s=>s.id),g.edges.map(e=>[e.from,e.to])).components!==1)return false;
        for(let i=0;i<members.length;i++)for(let j=i+1;j<members.length;j++)if(separation(members[i].direction,members[j].direction)>RULES.maxDiameter)return false;
        const center=centerOf(members),frame=tangentFrame(center.longitude,center.latitude);
        return members.every(s=>{const p=projectLocal(s.direction,frame,0,0,1);return p.visible&&Math.abs(p.x)<=34&&Math.abs(p.y)<=25;});
    };
    const result=sampleFigure({pool,major,coreRequired:new Set(bright.map(s=>s.id)),index,seed,style,accepted,attempts,acceptFigure});
    if(!result)return null;
    const {figure}=result,lookup=new Map(pool.map(s=>[s.id,s])),members=figure.members.map(id=>lookup.get(id));
    const edges=pairs=>pairs.map(([from,to])=>({from,to,degrees:separation(lookup.get(from).direction,lookup.get(to).direction)}));
    return {members,core:figure.coreMembers.map(id=>lookup.get(id)),center:centerOf(members),edges:edges(figure.edges),coreEdges:edges(figure.coreEdges),
        structure:structure(figure.members,figure.edges)};
}

function chooseWithinLayout(group,reference,seed,style,cells,index){
    const anchors=shapeEnvelope(reference.members).ids;
    const figure=chooseClearFigure(group,seed,style,{anchors,cells,index});
    if(figure)return figure;
    // A concave region can require intermediate stars to connect its anchors.
    // Recover those from a legal reference tree before trying again; a member
    // target is never allowed to force a disconnected automatic constellation.
    const required=new Set([...anchors,...majorStars(group).map(s=>s.id)]),profile=complexityBudget(style,required.size);
    const support=simplifyCore(reference.members,reference.variants[1].edges,required,
        {...profile,loops:0,coreStars:required.size,coreBranches:profile.branches}).core;
    return chooseClearFigure(group,`${seed}/connected`,style,{anchors:support.map(s=>s.id),cells,index})??
        chooseClearFigure(reference.members,`${seed}/reference-members`,style,{anchors:reference.members.map(s=>s.id),cells,index});
}

function reference(source,meta,recipe,onProgress){
    const {seed,style,samplingHalfWidthDegrees:halfWidth}=recipe,stars=source.map(brief);
    const key=JSON.stringify([seed,style,halfWidth,stars.filter(s=>s.app_mag<=4.5)]);
    if(cache.has(key))return structuredClone(cache.get(key));
    // Keep the established seed partition and its proof. Every displayed figure
    // is then selected and connected by the shared clarity sampler. Retaining
    // the layout hull anchors keeps its existing 8-degree envelope supported.
    const layout=generateLayout(source,meta,{algorithm:'terrax-zodiac-draw-9',seed,style:'rich',samplingHalfWidthDegrees:halfWidth},{onProgress});
    const cells=unpackGrid(layout.territories),groups=regionalStars(stars,cells);
    onProgress(`正在按“${complexityBudget(style,0).label}”生成十五座清晰星形…`);
    const regions=layout.regions.map((r,i)=>{
        const f=chooseWithinLayout(groups[i],r,`${seed}/clarity/reference/${i}`,style,cells,i);
        if(!f)throw Error(`候选 ${i+1} 未找到完整保留亮星与轮廓锚点的清晰星形；当前轮次已保留。`);
        return withFigure(r,f);
    });
    const result=rebuildRegions({...layout,schema:14,settings:RULES,regions},stars,cells,recipe);
    cache.set(key,structuredClone(result));if(cache.size>4)cache.delete(cache.keys().next().value);
    return result;
}

function figuresFor(base,source,recipe,cells){
    const groups=regionalStars(source.map(brief),cells);
    return base.regions.map((r,i)=>{
        if(recipe.shapeSeeds[i]===`${recipe.seed}/shape/${i}`)return r;
        const f=chooseWithinLayout(groups[i],r,recipe.shapeSeeds[i],recipe.style,cells,i);
        // The fallback was itself made with this method and this complexity.
        return f?withFigure(r,f):r;
    });
}

export function generateDraw(source,meta,value={},options={}){
    const recipe=normalizeRecipe(value),base=reference(source,meta,recipe,options.onProgress??(()=>{})),cells=unpackGrid(base.territories);
    const regions=figuresFor(base,source,recipe,cells);
    const data=rebuildRegions({...base,catalogue:meta.catalogue,sha256:meta.sha256,regions,sampling:samplingInfo(source,recipe.samplingHalfWidthDegrees),
        brightPolicy:BRIGHT_POLICY,localPolicy:LOCAL_POLICY,figurePolicy:FIGURE_POLICY},source,cells,recipe);
    data.brightAudit=brightAudit(data,source,cells);validateDraw(data,source);return data;
}

export function validateDraw(data,source){
    const recipe=normalizeRecipe(data.recipe);
    if(!equivalentDraw(data.sampling,samplingInfo(source,recipe.samplingHalfWidthDegrees)))throw Error('采样带与本轮配方不符');
    for(const [actual,expected] of [[data.settings,RULES],[data.brightPolicy,BRIGHT_POLICY],[data.localPolicy,LOCAL_POLICY],
        [data.figurePolicy,FIGURE_POLICY],[data.envelopePolicy,ENVELOPE_POLICY],[data.eclipticPolicy,ECLIPTIC_POLICY]]){
        if(!equivalentDraw(actual,expected))throw Error('清晰度、亮星或边界规则不符');
    }
    validateGeometry(data,source,{recipe,rules:RULES,adaptiveCore:true,minimumMembers:3,maximumMembers:Infinity,minimumCoreMembers:3,arcCoverage:manualArcCells});
    if(!acceptableWidths(data.regions.map(r=>r.eclipticSpan)))throw Error('黄道宽度不合法');
    const cells=unpackGrid(data.territories),audit=brightAudit(data,source,cells);
    if(!equivalentDraw(audit,data.brightAudit)||audit.some(r=>r.missing.length||r.missingFromCore.length))throw Error('星区的重要亮星未完整纳入星形和骨架');
    const envelopes=data.regions.map(r=>shapeEnvelope(r.members));
    for(let k=0;k<cells.length;k++)if(cells[k]<15&&envelopeDistance(envelopes[cells[k]],cellDirections[k])>8-CELL_RADIUS_DEGREES+1e-8)throw Error('天区超出实际星形的余量');
    const expected=reference(source,{catalogue:data.catalogue,sha256:data.sha256},recipe,()=>{});
    if(!equivalentDraw(data.territories,expected.territories)||!equivalentDraw(data.localOptimality,expected.localOptimality))throw Error('边界与局部优先求解证据不符');
    if(!equivalentDraw(data.regions.map(r=>r.boundary.length),data.localOptimality.regionCorners))throw Error('实际各区拐点与证明不符');
    if(!equivalentDraw(data.regions,figuresFor(expected,source,recipe,cells)))throw Error('星形成员与连线不符合本轮清晰度配方');
    return true;
}
