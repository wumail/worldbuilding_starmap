import {generateDraw as fittedDraw,ENVELOPE_POLICY} from './generator_fitted.mjs?revision=candidate-editor-band-1';
import {normalizeRecipe as normalizeFree,validateDraw as validateFree,RULES,DEFAULT_SEED} from './generator_free.mjs?revision=candidate-editor-band-1';
import {equivalentDraw} from './generator_ordered.mjs';
import {ECLIPTIC_POLICY,acceptableWidths} from './generator_ecliptic.mjs';
import {protectedCells} from './boundary_cleanup.mjs';
import {shapeEnvelope,envelopeMask,envelopeDistance,cellDirections,CELL_RADIUS_DEGREES} from './shape_envelope.mjs';
import {unpackGrid,packGrid,territoryBoundary,fromEquatorial} from './territories.mjs';
import {vector} from './geometry.mjs?revision=candidate-editor-band-1';
import {prepareMinimumSolver,solveMinimumCorners} from './minimum_corners.mjs';
export {RULES,DEFAULT_SEED,prepareMinimumSolver};
export const ALGORITHM='terrax-zodiac-draw-7';
export const MINIMUM_POLICY=Object.freeze({method:'joint-grid-corner-milp-1',coordinateStepDegrees:1,edgeDirections:'reference-ra-dec',
    envelopeReference:'root-balanced-draw-6',protectedGeometry:'candidate-pools-legal-arcs-and-ecliptic-cells',
    objective:'sum-of-region-corners',tieBreak:'minimum-owned-cells',solver:'highs-js-1.15.3'});
const cache=new Map();
export function normalizeRecipe(value={}){
    if(value.algorithm&&value.algorithm!==ALGORITHM)throw Error('这份留存使用了不同的抽卡算法，请使用对应版本查看。');
    return {...normalizeFree({...value,algorithm:'terrax-zodiac-draw-3'}),algorithm:ALGORITHM};
}
function optimum(source,meta,seed){
    const pool=source.filter(s=>s.app_mag<=4.5&&Math.abs(s.latitude)<=30).map(({id,app_mag,longitude,latitude,direction,color_hex,distance_pc})=>({id,app_mag,longitude,latitude,direction,color_hex,distance_pc}));
    const key=`${seed}\n${JSON.stringify(pool)}`;
    if(cache.has(key))return structuredClone(cache.get(key));
    const reference=fittedDraw(source,meta,{seed,style:'balanced'}),labels=unpackGrid(reference.territories),fixed=protectedCells(reference,source);
    const mask=envelopeMask(reference.regions.map(r=>shapeEnvelope(r.members)),ENVELOPE_POLICY.maximumMarginDegrees);
    const solved=solveMinimumCorners({labels,fixed,mask});
    for(let k=0;k<labels.length;k++)if(fixed[k]&&labels[k]!==solved.labels[k])throw Error('最少拐点求解改变了受保护归属');
    const boundaries=Array.from({length:15},(_,i)=>territoryBoundary(solved.labels,i));
    if(boundaries.reduce((n,b)=>n+b.length,0)!==solved.certificate.minimumCorners)throw Error('最少拐点证明与实际边界不一致');
    const result={territories:packGrid(solved.labels),boundaries,certificate:solved.certificate};
    cache.set(key,structuredClone(result));if(cache.size>4)cache.delete(cache.keys().next().value);
    return result;
}
export function generateDraw(source,meta,value={}){
    const recipe=normalizeRecipe(value),base=fittedDraw(source,meta,{...recipe,algorithm:'terrax-zodiac-draw-6'}),p=optimum(source,meta,recipe.seed);
    const {boundaryCleanup,...rest}=base;
    const data={...rest,schema:8,recipe,territories:p.territories,minimumCornersPolicy:MINIMUM_POLICY,optimality:p.certificate,
        regions:base.regions.map((r,i)=>({...r,boundary:p.boundaries[i],polygon:p.boundaries[i].map(([ra,dec])=>fromEquatorial(vector(ra,dec)))}))};
    validateDraw(data,source);return data;
}
export function validateDraw(data,source){
    const recipe=normalizeRecipe(data.recipe);
    if(!equivalentDraw(data.minimumCornersPolicy,MINIMUM_POLICY))throw Error('候选检查未通过：最少拐点规则');
    if(!equivalentDraw(data.envelopePolicy,ENVELOPE_POLICY)||!equivalentDraw(data.eclipticPolicy,ECLIPTIC_POLICY))throw Error('候选检查未通过：包络或黄道宽度规则');
    validateFree({...data,recipe:{...recipe,algorithm:'terrax-zodiac-draw-3'}},source);
    if(!acceptableWidths(data.regions.map(r=>r.eclipticSpan)))throw Error('候选检查未通过：黄道宽度');
    const expected=optimum(source,{catalogue:data.catalogue,sha256:data.sha256},recipe.seed);
    if(!equivalentDraw(data.territories,expected.territories)||!equivalentDraw(data.optimality,expected.certificate))throw Error('候选检查未通过：边界与最少拐点证明不符');
    if(data.regions.reduce((n,r)=>n+r.boundary.length,0)!==expected.certificate.minimumCorners)throw Error('候选检查未通过：实际拐点数量');
    const cells=unpackGrid(data.territories),envelopes=data.regions.map(r=>shapeEnvelope(r.members));
    for(let k=0;k<cells.length;k++)if(cells[k]<15&&envelopeDistance(envelopes[cells[k]],cellDirections[k])>ENVELOPE_POLICY.maximumMarginDegrees-CELL_RADIUS_DEGREES+1e-8)throw Error('候选检查未通过：实际星形的包络余量');
    return true;
}
