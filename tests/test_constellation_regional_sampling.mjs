import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {catalogueStars} from '../src/build_zodiac_candidates.mjs';
import {sampleRegion} from '../web/constellations/regional_figures.mjs';
import {majorStars} from '../web/constellations/bright_figures.mjs';
import {manualRecipe,applyManualFigures,manualArcCells,moveManualBoundary} from '../web/constellations/manual_figures.mjs';
import {cellOf,unpackGrid,packGrid} from '../web/constellations/territories.mjs';
import {separation} from '../web/constellations/geometry.mjs';
import {prepareDraw,generateDraw,equivalentDraw} from '../web/constellations/generator.mjs';
const read=p=>JSON.parse(fs.readFileSync(new URL('../'+p,import.meta.url))),base=read('design/zodiac_draw_sampling_example.json').data,meta=read('design/zodiac_candidates_v1.json'),stars=catalogueStars(read(meta.catalogue)),lookup=new Map(stars.map(s=>[s.id,s])),cells=unpackGrid(base.territories);
const shape=r=>({members:r.members,variants:r.variants});
function apply(current,result){const r=manualRecipe(base,current);r.figures=r.figures.filter(f=>f.index!==result.figure.index);r.figures.push(result.figure);return applyManualFigures(base,stars,r);}
function smallRegion(chosen){const next=new Int8Array(cells);for(let i=0;i<next.length;i++)if(next[i]===0)next[i]=15;for(const s of chosen)next[cellOf(s.direction)]=0;const recipe=manualRecipe(base,{...base,territories:packGrid(next)});recipe.figures=[{index:0,members:[],coreMembers:[],edges:[],coreEdges:[]}];return applyManualFigures(base,stars,recipe);}

test('all fifteen regions resample actual in-bound stars, preserve their important stars and keep every edge inside',()=>{
    const original=JSON.stringify(base),source=JSON.stringify(stars);
    for(let i=0;i<15;i++)for(let j=0;j<3;j++){
        const r=sampleRegion(base,stars,i,`region-sample-${j}`),group=stars.filter(s=>cells[cellOf(s.direction)]===i),required=majorStars(group).map(s=>s.id);
        assert.ok(r.changed);assert.ok(r.figure.members.every(id=>cells[cellOf(lookup.get(id).direction)]===i));
        assert.ok(required.every(id=>r.figure.members.includes(id)&&r.figure.coreMembers.includes(id)));
        for(const edge of [...r.figure.edges,...r.figure.coreEdges])assert.ok(manualArcCells(...edge.map(id=>lookup.get(id).direction)).every(k=>cells[k]===i));
    }
    assert.equal(JSON.stringify(base),original);assert.equal(JSON.stringify(stars),source);
});
test('same input and seed reproduce; different seeds genuinely sample different members',()=>{
    const a=sampleRegion(base,stars,0,'repeat'),b=sampleRegion(base,[...stars].reverse(),0,'repeat');assert.deepEqual(a,b);
    const signatures=new Set(Array.from({length:8},(_,i)=>JSON.stringify(sampleRegion(base,stars,0,`variety-${i}`).figure.members)));assert.ok(signatures.size>=3);
});
test('regeneration changes only the current figure, preserves manual boundary changes and survives the existing manual-3 replay',async()=>{
    const next=moveManualBoundary(base,0,0,0,-61),current=applyManualFigures(base,stars,manualRecipe(base,{...base,territories:packGrid(next)})),r=sampleRegion(current,stars,0,'expanded-1'),d=apply(current,r);
    assert.deepEqual(d.territories,current.territories);for(let i=1;i<15;i++)assert.deepEqual(shape(d.regions[i]),shape(current.regions[i]));
    assert.ok(r.regionStarCount>sampleRegion(base,stars,0,'expanded-1').regionStarCount);
    assert.ok(r.figure.members.some(id=>cells[cellOf(lookup.get(id).direction)]!==0));
    await prepareDraw(d.recipe);assert.ok(equivalentDraw(generateDraw(stars,meta,d.recipe),d));
});
test('sparse or disconnected manual regions work without inventing stars or crossing the gaps',()=>{
    const free=stars.filter(s=>cells[cellOf(s.direction)]===15&&s.app_mag>4.5),a=free[0],b=free.find(s=>separation(a.direction,s.direction)>60);
    const sparse=smallRegion([a]),r=sampleRegion(sparse,stars,0,'sparse');assert.ok(r.figure.members.includes(a.id));const done=apply(sparse,r);assert.equal(done.manualEdits.status,'validated');
    const split=smallRegion([a,b]),d=apply(split,sampleRegion(split,stars,0,'split'));assert.ok(d.regions[0].structure.components>=2);assert.equal(d.manualEdits.status,'validated');
    const empty=smallRegion([]);assert.throws(()=>sampleRegion(empty,stars,0,'empty'),/没有可用恒星/);
});
test('no-alternative result is reported honestly and a neighbour-owned bright member is never stolen',()=>{
    const s=stars.find(s=>cells[cellOf(s.direction)]===15),small=smallRegion([s]),one=apply(small,sampleRegion(small,stars,0,'one'));
    const next=sampleRegion(one,stars,0,'another');if(next.figure.members.length===1)assert.equal(next.changed,false);
    const bad=structuredClone(base),member=bad.regions[1].members[0],moved=new Int8Array(cells);moved[cellOf(member.direction)]=0;bad.territories=packGrid(moved);
    // Make the conflict unambiguously a major anchor, without altering the real fixture.
    const inputs=stars.map(s=>s.id===member.id?{...s,app_mag:-10}:s);
    assert.throws(()=>sampleRegion(bad,inputs,0,'conflict'),/邻座/);
});
