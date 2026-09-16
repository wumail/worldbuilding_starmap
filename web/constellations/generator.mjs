import * as current from './generator_sampling.mjs';
import * as balanced from './generator_balanced.mjs';
import * as minimum from './generator_minimum.mjs';
import * as fitted from './generator_fitted.mjs?revision=candidate-editor-band-1';
import * as ecliptic from './generator_ecliptic.mjs';
import * as clean from './generator_clean.mjs';
import * as free from './generator_free.mjs?revision=candidate-editor-band-1';
import * as ordered from './generator_ordered.mjs';
import * as legacy from './generator_legacy.mjs';
import {MANUAL_ALGORITHM,normalizeEdits,applyBoundaryEdits} from './boundary_edits.mjs?revision=candidate-editor-band-1';
import {EDIT_ALGORITHM,normalizeRemovedEdges,applyCandidateEdits} from './candidate_edits.mjs';
import {FREE_EDIT_ALGORITHM,normalizeFigures,applyManualFigures} from './manual_figures.mjs';
export {MANUAL_ALGORITHM,EDIT_ALGORITHM,FREE_EDIT_ALGORITHM};
export const isManual=recipe=>[MANUAL_ALGORITHM,EDIT_ALGORITHM,FREE_EDIT_ALGORITHM].includes(recipe?.algorithm);
export const ALGORITHM=current.ALGORITHM;
export const SUPPORTED_ALGORITHMS=Object.freeze([current.ALGORITHM,balanced.ALGORITHM,FREE_EDIT_ALGORITHM,EDIT_ALGORITHM,MANUAL_ALGORITHM,minimum.ALGORITHM,fitted.ALGORITHM,ecliptic.ALGORITHM,clean.ALGORITHM,free.ALGORITHM,ordered.ALGORITHM,legacy.ALGORITHM]);
export const RULES=current.RULES;
export const DEFAULT_SEED=current.DEFAULT_SEED;
export const equivalentDraw=ordered.equivalentDraw;
export const randomFrom=ordered.randomFrom;
const implementation=value=>value?.algorithm===legacy.ALGORITHM?legacy:value?.algorithm===ordered.ALGORITHM?ordered:value?.algorithm===free.ALGORITHM?free:value?.algorithm===clean.ALGORITHM?clean:value?.algorithm===ecliptic.ALGORITHM?ecliptic:value?.algorithm===fitted.ALGORITHM?fitted:value?.algorithm===minimum.ALGORITHM?minimum:value?.algorithm===balanced.ALGORITHM?balanced:current;
export async function prepareDraw(value={}){
    const recipe=normalizeRecipe(value);if(isManual(recipe))return prepareDraw(recipe.base);
    if([current,balanced,minimum].includes(implementation(recipe)))await current.prepareMinimumSolver();
}
export function normalizeRecipe(value={}){
    if(!isManual(value))return implementation(value).normalizeRecipe(value);
    if(!value.base||isManual(value.base))throw Error('手动编辑需要一份自动原始轮次');
    const base=normalizeRecipe(value.base);
    const normalized={algorithm:value.algorithm,seed:base.seed,style:base.style,shapeSeeds:base.shapeSeeds,base,edits:normalizeEdits(value.edits)};
    if(value.algorithm===EDIT_ALGORITHM)normalized.removedEdges=normalizeRemovedEdges(value.removedEdges);
    if(value.algorithm===FREE_EDIT_ALGORITHM)normalized.figures=normalizeFigures(value.figures);
    return normalized;
}
export function generateDraw(source,meta,value={},options={}){
    const recipe=normalizeRecipe(value);
    if(recipe.algorithm===MANUAL_ALGORITHM)return applyBoundaryEdits(generateDraw(source,meta,recipe.base,options),source,recipe.edits,recipe);
    if(recipe.algorithm===EDIT_ALGORITHM)return applyCandidateEdits(generateDraw(source,meta,recipe.base,options),source,recipe);
    if(recipe.algorithm===FREE_EDIT_ALGORITHM)return applyManualFigures(generateDraw(source,meta,recipe.base,options),source,recipe);
    return implementation(recipe).generateDraw(source,meta,recipe,options);
}
export function validateDraw(data,stars){
    if(!isManual(data.recipe))return implementation(data.recipe).validateDraw(data,stars);
    const expected=generateDraw(stars,{catalogue:data.catalogue,sha256:data.sha256},data.recipe);
    if(!equivalentDraw(expected,data))throw Error('手动边界与保存配方不一致');return true;
}
export function redrawRecipe(recipe,locked,nonce){
    const next=normalizeRecipe(recipe),protectedRegions=new Set(locked);
    if(isManual(next))throw Error('手动编辑轮次保留现有星形；如需重抽星形，请返回自动原轮次');
    if([...protectedRegions].some(i=>!Number.isInteger(i)||i<0||i>=15))throw Error('锁定的区域编号无效。');
    next.shapeSeeds=next.shapeSeeds.map((s,i)=>protectedRegions.has(i)?s:`${String(nonce).slice(0,90)}/shape/${i}`);
    return next;
}
