import {vector} from './geometry.mjs?revision=candidate-editor-band-1';
import {generateDraw as generateFree,validateDraw as validateFree,normalizeRecipe as normalizeFree,ALGORITHM as FREE_ALGORITHM} from './generator_free.mjs?revision=candidate-editor-band-1';
import {equivalentDraw} from './generator_ordered.mjs';
import {packGrid,fromEquatorial} from './territories.mjs';
import {CLEANUP,cleanBoundaries} from './boundary_cleanup.mjs';
export {RULES,DEFAULT_SEED} from './generator_free.mjs?revision=candidate-editor-band-1';
export const ALGORITHM='terrax-zodiac-draw-4';

export function normalizeRecipe(value={}){
    if(value.algorithm&&value.algorithm!==ALGORITHM)throw Error('这份留存使用了不同的抽卡算法，请使用对应版本查看。');
    return {...normalizeFree({...value,algorithm:FREE_ALGORITHM}),algorithm:ALGORITHM};
}

export function generateDraw(source,meta,value={}){
    const recipe=normalizeRecipe(value),data=generateFree(source,meta,{...recipe,algorithm:FREE_ALGORITHM}),clean=cleanBoundaries(data,source);
    data.schema=5;data.recipe=recipe;data.boundaryCleanup=CLEANUP;data.territories=packGrid(clean.labels);
    for(let i=0;i<15;i++){
        data.regions[i].boundary=clean.boundaries[i];
        data.regions[i].polygon=clean.boundaries[i].map(([ra,dec])=>fromEquatorial(vector(ra,dec)));
    }
    // Counts, all figures and the exact ecliptic intervals stay unchanged.
    // The common validator independently checks them against the cleaned map.
    validateDraw(data,source);return data;
}

export function validateDraw(data,source){
    const recipe=normalizeRecipe(data.recipe);
    if(!equivalentDraw(data.boundaryCleanup,CLEANUP))throw Error('候选检查未通过：边界整理规则');
    return validateFree({...data,recipe:{...recipe,algorithm:FREE_ALGORITHM}},source);
}
