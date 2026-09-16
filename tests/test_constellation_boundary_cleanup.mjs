import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {catalogueStars} from '../src/build_zodiac_candidates.mjs';
import {generateDraw,redrawRecipe,validateDraw} from '../web/constellations/generator.mjs';
import {unpackGrid} from '../web/constellations/territories.mjs';
import {protectedCells} from '../web/constellations/boundary_cleanup.mjs';

const meta=JSON.parse(fs.readFileSync(new URL('../design/zodiac_candidates_v1.json',import.meta.url)));
const stars=catalogueStars(JSON.parse(fs.readFileSync(new URL(`../${meta.catalogue}`,import.meta.url))));
const figures=d=>d.regions.map(({boundary,polygon,...r})=>r);

test('cleanup reduces boundary fragmentation on 120 draws without resampling stars or altering ecliptic ownership',()=>{
    const results=[];
    for(let k=0;k<60;k++)for(const style of ['rich','balanced']){
        const seed=k===0?'terrax-b1bxb1':k===1?'terrax-001':`verify-${k-2}`;
        const before=generateDraw(stars,meta,{seed,style,algorithm:'terrax-zodiac-draw-3'}),after=generateDraw(stars,meta,{seed,style,algorithm:'terrax-zodiac-draw-4'});
        assert.equal(after.recipe.algorithm,'terrax-zodiac-draw-4');
        assert.deepEqual(figures(after),figures(before),'only boundaries may change');
        assert.equal(after.candidateCount,before.candidateCount);assert.equal(after.eligibleCandidateCount,before.eligibleCandidateCount);
        const oldCount=before.regions.reduce((n,r)=>n+r.boundary.length,0),newCount=after.regions.reduce((n,r)=>n+r.boundary.length,0);
        assert.ok(newCount<oldCount*.6,`${seed}/${style}: insufficient simplification ${oldCount} -> ${newCount}`);
        const oldCells=unpackGrid(before.territories),newCells=unpackGrid(after.territories),fixed=protectedCells(before,stars);
        for(let i=0;i<fixed.length;i++)if(fixed[i])assert.equal(newCells[i],oldCells[i],'protected stars, potential arcs and ecliptic must keep their owner');
        results.push({seed,style,before:oldCount,after:newCount,reduction:1-newCount/oldCount});
    }
    fs.mkdirSync(new URL('../reports/zodiac_boundary_clean/',import.meta.url),{recursive:true});
    fs.writeFileSync(new URL('../reports/zodiac_boundary_clean/seed_validation.json',import.meta.url),JSON.stringify(results,null,2)+'\n');
});

test('cleanup of a locally rerolled recipe keeps the same custom figures and all future cleanup boundaries',()=>{
    const first=generateDraw(stars,meta,{seed:'terrax-b1bxb1',algorithm:'terrax-zodiac-draw-3'});
    const custom=redrawRecipe(first.recipe,[0,4],'custom-before-cleanup'),before=generateDraw(stars,meta,custom);
    const cleaned=generateDraw(stars,meta,{...custom,algorithm:'terrax-zodiac-draw-4'});
    assert.deepEqual(figures(before),figures(cleaned));
    const next=generateDraw(stars,meta,redrawRecipe(cleaned.recipe,[0,4],'custom-after-cleanup'));
    assert.deepEqual(next.territories,cleaned.territories);
    for(let i=0;i<15;i++){
        assert.deepEqual(next.regions[i].boundary,cleaned.regions[i].boundary);
        if([0,4].includes(i))assert.deepEqual(next.regions[i],cleaned.regions[i]);
    }
    assert.deepEqual(generateDraw(stars,meta,JSON.parse(JSON.stringify(cleaned.recipe))),cleaned);
    assert.notDeepEqual(figures(next),figures(cleaned));
});

test('the previous fine-grid export remains byte-for-byte reproducible, and cleanup metadata is checked',()=>{
    const before=JSON.parse(fs.readFileSync(new URL('../design/zodiac_draw_free_example.json',import.meta.url))).data;
    assert.equal(JSON.stringify(generateDraw(stars,meta,before.recipe)),JSON.stringify(before));
    const next=generateDraw(stars,meta,{seed:'terrax-b1bxb1',algorithm:'terrax-zodiac-draw-4'});
    next.boundaryCleanup={...next.boundaryCleanup,blockDegrees:8};
    assert.throws(()=>validateDraw(next,stars),/边界整理规则/);
});
