import {generateDraw as fittedDraw,ENVELOPE_POLICY} from './generator_fitted.mjs?revision=candidate-editor-band-1';
import {normalizeRecipe as normalizeFree,validateGeometry,DEFAULT_SEED} from './generator_free.mjs?revision=candidate-editor-band-1';
import {equivalentDraw} from './generator_ordered.mjs';
import {ECLIPTIC_POLICY,acceptableWidths} from './generator_ecliptic.mjs';
import {shapeEnvelope,envelopeMask,envelopeDistance,cellDirections,CELL_RADIUS_DEGREES} from './shape_envelope.mjs';
import {unpackGrid,cellOf,arcCells} from './territories.mjs';
import {vector} from './geometry.mjs?revision=candidate-editor-band-1';
import {prepareMinimumSolver,solveLocalFirstCorners} from './minimum_corners.mjs';
import {RULES,BRIGHT_POLICY,brief,regionalStars,chooseBrightFigure,brightAudit} from './bright_figures.mjs';
import {withFigure,rebuildRegions} from './region_data.mjs';
export {RULES,DEFAULT_SEED,prepareMinimumSolver};
export const ALGORITHM='terrax-zodiac-draw-8';
export const LOCAL_POLICY=Object.freeze({method:'local-excess-lexicographic-1',coordinateStepDegrees:1,
    edgeDirections:'reference-ra-dec',protectedGeometry:'actual-reference-figures-bright-pool-and-ecliptic',
    priorities:['individual-lower-bounds','descending-excess-corners','total-corners','owned-cells']});
const cache=new Map();
export function normalizeRecipe(value={}){
    if(value.algorithm&&value.algorithm!==ALGORITHM)throw Error('这份轮次使用不同的抽卡算法');
    return {...normalizeFree({...value,algorithm:'terrax-zodiac-draw-3'}),algorithm:ALGORITHM};
}
export function protectedFigureCells(data,source,cells){
    const fixed=new Uint8Array(cells.length);
    for(const s of source)if(s.app_mag<=RULES.candidateMagnitude)fixed[cellOf(s.direction)]=1;
    for(const [i,r] of data.regions.entries()){
        const members=new Map(r.members.map(s=>[s.id,s]));
        for(const s of r.members)fixed[cellOf(s.direction)]=1;
        for(const v of r.variants)for(const e of v.edges)for(const k of arcCells(members.get(e.from).direction,members.get(e.to).direction)){
            if(cells[k]!==i)throw Error('参考星形的连线越界');fixed[k]=1;
        }
    }
    for(let l=0;l<360;l+=10)for(const k of arcCells(vector(l,0),vector(l+10,0)))fixed[k]=1;
    return fixed;
}
function reference(source,meta,seed,onProgress){
    const stars=source.map(brief),key=`${seed}\n${JSON.stringify(stars.filter(s=>s.app_mag<=4.5))}`;
    if(cache.has(key))return structuredClone(cache.get(key));
    let base,layoutSeed,lastError;
    for(let attempt=0;attempt<12;attempt++){
        layoutSeed=attempt?(seed.length>60?`bright-layout-${attempt}/${seed.slice(0,60)}`:`${seed}/bright-layout/${attempt}`):seed;
        onProgress(`正在组织亮星与星形${attempt?`，尝试第 ${attempt+1} 个布局`:''}…`);
        try{
            const old=fittedDraw(stars,meta,{seed:layoutSeed,style:'balanced'}),cells=unpackGrid(old.territories),groups=regionalStars(stars,cells);
            const regions=old.regions.map((r,i)=>withFigure(r,chooseBrightFigure(groups[i],r,`${seed}/reference/${attempt}/${i}`,'rich',cells,i)));
            const {boundaryCleanup,...rest}=old;
            base=rebuildRegions({...rest,settings:RULES,regions},stars,cells);break;
        }catch(error){lastError=error;}
    }
    if(!base)throw Error(`此种子未找到同时保留重要亮星与合法星形的布局：${lastError?.message??''}。当前轮次已保留。`);
    const cells=unpackGrid(base.territories),mask=envelopeMask(base.regions.map(r=>shapeEnvelope(r.members)),8),fixed=protectedFigureCells(base,stars,cells);
    const solved=solveLocalFirstCorners({labels:cells,mask,fixed},{onProgress});
    const result=rebuildRegions({...base,localOptimality:solved.certificate,layoutSeed},stars,solved.labels);
    cache.set(key,structuredClone(result));if(cache.size>4)cache.delete(cache.keys().next().value);
    return result;
}
export function generateDraw(source,meta,value={},options={}){
    const recipe=normalizeRecipe(value),onProgress=options.onProgress??(()=>{}),base=reference(source,meta,recipe.seed,onProgress);
    const cells=unpackGrid(base.territories),groups=regionalStars(source.map(brief),cells);
    const regions=base.regions.map((r,i)=>{
        if(recipe.style==='rich'&&recipe.shapeSeeds[i]===`${recipe.seed}/shape/${i}`)return r;
        try{return withFigure(r,chooseBrightFigure(groups[i],r,recipe.shapeSeeds[i],recipe.style,cells,i));}
        catch{return r;} // Preserve the already valid bright reference, never an incomplete figure.
    });
    const data=rebuildRegions({...base,schema:9,catalogue:meta.catalogue,sha256:meta.sha256,regions,brightPolicy:BRIGHT_POLICY,localPolicy:LOCAL_POLICY},source,cells,recipe);
    data.brightAudit=brightAudit(data,source,cells);
    validateDraw(data,source);return data;
}
export function validateDraw(data,source){
    const recipe=normalizeRecipe(data.recipe);
    if(!equivalentDraw(data.settings,RULES)||!equivalentDraw(data.brightPolicy,BRIGHT_POLICY)||!equivalentDraw(data.localPolicy,LOCAL_POLICY))throw Error('新版选形与局部边界规则不符');
    if(!equivalentDraw(data.envelopePolicy,ENVELOPE_POLICY)||!equivalentDraw(data.eclipticPolicy,ECLIPTIC_POLICY))throw Error('星形余量或黄道宽度规则不符');
    validateGeometry(data,source,{recipe,rules:RULES,adaptiveCore:true,maximumMembers:15});
    if(!acceptableWidths(data.regions.map(r=>r.eclipticSpan)))throw Error('黄道宽度不合法');
    const cells=unpackGrid(data.territories),audit=brightAudit(data,source,cells);
    if(!equivalentDraw(audit,data.brightAudit)||audit.some(r=>r.missing.length||r.missingFromCore.length))throw Error('星区的重要亮星未完整纳入星形和骨架');
    const envelopes=data.regions.map(r=>shapeEnvelope(r.members));
    for(let k=0;k<cells.length;k++)if(cells[k]<15&&envelopeDistance(envelopes[cells[k]],cellDirections[k])>8-CELL_RADIUS_DEGREES+1e-8)throw Error('天区超出实际星形的余量');
    const expected=reference(source,{catalogue:data.catalogue,sha256:data.sha256},recipe.seed,()=>{});
    if(!equivalentDraw(data.territories,expected.territories)||!equivalentDraw(data.localOptimality,expected.localOptimality))throw Error('边界与局部优先求解证据不符');
    if(!equivalentDraw(data.regions.map(r=>r.boundary.length),data.localOptimality.regionCorners))throw Error('实际各区拐点与证明不符');
    return true;
}
