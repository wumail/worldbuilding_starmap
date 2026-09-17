import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {catalogueStars} from '../src/build_zodiac_candidates.mjs';
import {ALGORITHM,normalizeRecipe,prepareDraw,generateDraw,validateDraw,redrawRecipe,equivalentDraw} from '../web/constellations/generator.mjs';
import {REGIONAL_METHOD,sampleRegion} from '../web/constellations/regional_figures.mjs';
import {complexityBudget} from '../web/constellations/regional_quality.mjs';
import {unpackGrid,cellOf} from '../web/constellations/territories.mjs';
import {manualArcCells,manualRecipe} from '../web/constellations/manual_figures.mjs';
import {graphMeasures} from '../src/audit_regional_figures.mjs';

const read=p=>JSON.parse(fs.readFileSync(new URL('../'+p,import.meta.url)));
const meta=read('design/zodiac_candidates_v1.json'),stars=catalogueStars(read(meta.catalogue)),sourceBefore=JSON.stringify(stars);
let standard;
test('new seed defaults are versioned; historical recipes and invalid input retain their contracts',()=>{
    assert.equal(ALGORITHM,'terrax-zodiac-draw-10');assert.equal(normalizeRecipe().style,'balanced');
    for(const style of ['simple','balanced','rich'])assert.equal(normalizeRecipe({style}).style,style);
    for(const version of [2,3,4,5,6,7,8,9])assert.equal(normalizeRecipe({algorithm:`terrax-zodiac-draw-${version}`}).style,'rich');
    for(const value of [{style:'toString'},{style:'unknown'},{algorithm:'unknown'},{seed:''},{samplingHalfWidthDegrees:61}])assert.throws(()=>normalizeRecipe(value));
    for(const algorithm of ['terrax-zodiac-manual-1','terrax-zodiac-manual-2']){
        assert.throws(()=>normalizeRecipe({algorithm,base:normalizeRecipe(),edits:[],removedEdges:[]}),/旧手动格式/);
    }
});
test('all fifteen seed-generated figures use the clarity rules and retain real bright stars and exact boundaries',async()=>{
    await prepareDraw();standard=generateDraw(stars,meta);
    assert.equal(standard.recipe.style,'balanced');assert.equal(standard.figurePolicy.method,REGIONAL_METHOD);
    assert.deepEqual(standard.figurePolicy.stages,['seed-figures','fixed-boundary-reroll']);
    assert.equal(standard.figurePolicy.layoutAlgorithm,'terrax-zodiac-draw-9');
    const cells=unpackGrid(standard.territories),lookup=new Map(stars.map(s=>[s.id,s]));
    for(const [i,r] of standard.regions.entries()){
        const group=stars.filter(s=>cells[cellOf(s.direction)]===i).sort((a,b)=>a.app_mag-b.app_mag||a.id.localeCompare(b.id));
        const required=group.filter((s,j)=>j<3||s.app_mag<=3).map(s=>s.id);
        for(const variant of r.variants){
            assert.ok(required.every(id=>variant.members.includes(id)));
            const stats=graphMeasures(variant.members,variant.edges.map(e=>[e.from,e.to]));assert.equal(stats.components,1);
            assert.ok(stats.loops<=complexityBudget('balanced',r.members.length).loops);assert.ok(stats.maxDegree<=4);
            for(const e of variant.edges)assert.ok(manualArcCells(lookup.get(e.from).direction,lookup.get(e.to).direction).every(k=>cells[k]===i));
        }
    }
    assert.equal(validateDraw(standard,stars),true);assert.deepEqual(generateDraw(stars,meta,standard.recipe),standard);
    assert.equal(JSON.stringify(stars),sourceBefore);
    const bad=structuredClone(standard);bad.regions[0].structure.loops+=1;assert.throws(()=>validateDraw(bad,stars));
});
test('new-version locked rerolls and manual-3 replay retain borders, anchors and historical figure freedom',()=>{
    const recipe=redrawRecipe(standard.recipe,[0,5,10],'clarity-reroll'),changed=generateDraw(stars,meta,recipe);
    assert.deepEqual(changed.territories,standard.territories);
    for(const i of [0,5,10])assert.deepEqual(changed.regions[i],standard.regions[i]);
    assert.ok(changed.regions.some((r,i)=>JSON.stringify(r.variants)!==JSON.stringify(standard.regions[i].variants)));
    const manual=manualRecipe(standard,standard),{figure}=sampleRegion(standard,stars,0,'seed-local','simple');manual.figures=[figure];
    const applied=generateDraw(stars,meta,manual);assert.equal(applied.recipe.base.algorithm,ALGORITHM);assert.equal(applied.figurePolicy,undefined);
    assert.deepEqual(generateDraw(stars,meta,applied.recipe),applied);assert.equal(validateDraw(applied,stars),true);
});
test('saved draw-9 remains byte-for-byte equivalent through the current dispatcher',async()=>{
    const previous=read('design/zodiac_draw_sampling_example.json').data;await prepareDraw(previous.recipe);
    assert.ok(equivalentDraw(generateDraw(stars,meta,previous.recipe),previous));
});
