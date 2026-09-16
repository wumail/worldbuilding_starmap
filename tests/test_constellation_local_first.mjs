import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {prepareDraw,generateDraw as generateAny,validateDraw,redrawRecipe,normalizeRecipe,MANUAL_ALGORITHM} from '../web/constellations/generator.mjs';
import {solveLocalFirstCorners} from '../web/constellations/minimum_corners.mjs';
import {brightAudit,majorStars} from '../web/constellations/bright_figures.mjs';
import {moveBoundaryEdge,applyEditedGrid,gridEdits,normalizeEdits} from '../web/constellations/boundary_edits.mjs';
import {packGrid,unpackGrid,territoryBoundary,cellOf,regionAt} from '../web/constellations/territories.mjs';
import {cellAreaDegrees} from '../web/constellations/shape_envelope.mjs';
import {catalogueStars} from '../src/build_zodiac_candidates.mjs';

// This file is the draw-8 regression contract, independent of the UI default.
const generateDraw=(stars,meta,recipe={})=>generateAny(stars,meta,{algorithm:'terrax-zodiac-draw-8',...recipe});

await prepareDraw();
const meta=JSON.parse(fs.readFileSync(new URL('../design/zodiac_candidates_v1.json',import.meta.url)));
const stars=catalogueStars(JSON.parse(fs.readFileSync(new URL(`../${meta.catalogue}`,import.meta.url))));
const out=new URL('../reports/zodiac_local_first/',import.meta.url);fs.mkdirSync(out,{recursive:true});

// Independent directed boundary tracing, not the MILP's four-cell formula.
function trace(labels,id,W,H){
    const edges=new Map(),key=(x,y)=>y*W+(x+W)%W,at=(x,y)=>y<0||y>=H?-1:labels[key(x,y)];
    const add=(a,b)=>{if(edges.has(a))throw Error('touch');edges.set(a,b);};
    for(let y=0;y<H;y++)for(let x=0;x<W;x++)if(at(x,y)===id){
        if(!y||y===H-1)throw Error('pole');
        if(at(x,y-1)!==id)add(key(x,y),key(x+1,y));if(at(x+1,y)!==id)add(key(x+1,y),key(x+1,y+1));
        if(at(x,y+1)!==id)add(key(x+1,y+1),key(x,y+1));if(at(x-1,y)!==id)add(key(x,y+1),key(x,y));
    }
    if(!edges.size)throw Error('empty');
    const first=edges.keys().next().value,ring=[];let k=first;
    do{ring.push(k);const next=edges.get(k);if(next===undefined)throw Error('open');edges.delete(k);k=next;}while(k!==first);
    if(edges.size)throw Error('holes');
    return ring.filter((p,i)=>{const a=ring[(i+ring.length-1)%ring.length],b=ring[(i+1)%ring.length];return !((a%W===p%W&&b%W===p%W)||(Math.floor(a/W)===Math.floor(p/W)&&Math.floor(b/W)===Math.floor(p/W)));}).length;
}
const conflict={width:4,height:5,count:2,
    labels:[2,2,2,2,0,1,2,2,2,2,1,2,2,2,0,2,2,2,2,2],
    fixed:[0,0,0,0,1,1,0,0,0,0,1,0,0,0,1,0,0,0,0,0],
    mask:[0,0,0,0,1,2,2,1,1,3,2,2,2,1,1,2,0,0,0,0]};
function enumerate(problem){
    const result=[],labels=[...problem.labels],{width:W,height:H,count}=problem;
    function next(k){
        if(k===labels.length){try{result.push({labels:[...labels],corners:Array.from({length:count},(_,i)=>trace(labels,i,W,H))});}catch{}return;}
        if(problem.fixed[k]){next(k+1);return;}
        for(let id=0;id<=count;id++)if(id===count||problem.mask[k]&(1<<id)){labels[k]=id;next(k+1);}
    }
    next(0);return result;
}
test('local lower bounds and coordinated excess match exhaustive conflict-grid enumeration',()=>{
    const joint=enumerate(conflict),local=Array.from({length:2},(_,i)=>Math.min(...enumerate({...conflict,count:1,
        labels:conflict.labels.map(id=>id===i?0:1),mask:conflict.mask.map(m=>m&(1<<i)?1:0)}).map(r=>r.corners[0])));
    assert.equal(joint.length,8);assert.deepEqual(local,[10,4]);
    const key=r=>[...r.corners.map((n,i)=>n-local[i]).sort((a,b)=>b-a),r.corners.reduce((a,b)=>a+b,0)];
    joint.sort((a,b)=>{const x=key(a),y=key(b);for(let i=0;i<x.length;i++)if(x[i]!==y[i])return x[i]-y[i];return 0;});
    const d=solveLocalFirstCorners(conflict,{seconds:20}),c=d.certificate;
    assert.deepEqual(c.localMinima,local);assert.deepEqual([...c.sortedExcessCorners,c.minimumTotalAtLocalPriority],key(joint[0]));
    assert.deepEqual(c.regionCorners,Array.from({length:2},(_,i)=>trace(d.labels,i,4,5)));assert.deepEqual(c.topExcessIntegerLowerBounds,[2,2]);
    const swapped={...conflict,labels:conflict.labels.map(x=>x<2?1-x:2),mask:conflict.mask.map(x=>(x&1?2:0)|(x&2?1:0))};
    assert.deepEqual(solveLocalFirstCorners(swapped).certificate.sortedExcessCorners,c.sortedExcessCorners);
});
test('bright inclusion ranks by smaller apparent magnitude and checks outside the old latitude band',()=>{
    const pool=[{id:'a',app_mag:4,latitude:0},{id:'b',app_mag:-1,latitude:35},{id:'c',app_mag:2,latitude:-34},{id:'d',app_mag:2.5},{id:'e',app_mag:3},{id:'f',app_mag:3.1}];
    assert.deepEqual(majorStars(pool).map(s=>s.id),['b','c','d','e']);
});
test('all four edge directions cross the RA seam without mutating the source cache',()=>{
    const cells=new Int8Array(64800).fill(15);
    for(let y=80;y<84;y++)for(const x of [358,359,0,1])cells[y*360+x]=0;
    const data={territories:packGrid(cells),regions:[{boundary:territoryBoundary(cells,0)}]},copy=new Int8Array(unpackGrid(data.territories));
    for(let e=0;e<4;e++)for(const step of [-1,1]){
        const a=data.regions[0].boundary[e],b=data.regions[0].boundary[(e+1)%4],value=a[1]===b[1]?a[1]:a[0],next=moveBoundaryEdge(data,0,e,value+step);
        assert.ok([12,20].includes(next.filter(i=>i===0).length));assert.equal(territoryBoundary(next,0).length,4);assert.deepEqual(unpackGrid(data.territories),copy);
    }
    assert.throws(()=>normalizeEdits([[1,0],[1,1]]),/重复/);assert.throws(()=>normalizeEdits([[64800,1]]),/越界/);
});
let example;
test('12 draws include every regional major star and preserve original physical data',()=>{
    const measurements=[];
    for(const seed of ['terrax-1ptws5s','terrax-001','terrax-b1bxb1','local-first-1','local-first-2','local-first-3']){
        const start=performance.now();
        for(const style of ['rich','balanced']){
            const d=generateDraw(stars,meta,{seed,style}),cells=unpackGrid(d.territories),audit=brightAudit(d,stars,cells);
            assert.equal(validateDraw(d,stars),true);assert.ok(audit.every(r=>!r.missing.length&&!r.missingFromCore.length));
            for(const [i,r] of d.regions.entries()){
                const actual=stars.filter(s=>regionAt(d,s.direction)===i).sort((a,b)=>a.app_mag-b.app_mag||a.id.localeCompare(b.id));
                assert.ok(r.members.some(s=>s.id===actual[0].id));
                assert.equal(r.boundary.length,d.localOptimality.regionCorners[i]);
            }
            let area=0;for(let k=0;k<cells.length;k++)if(cells[k]<15)area+=cellAreaDegrees(k);
            measurements.push({seed,style,milliseconds:performance.now()-start,layoutSeed:d.layoutSeed,members:d.selectedExtendedCount,core:d.selectedCoreCount,
                proof:d.localOptimality,areaSquareDegrees:area,missing:audit.flatMap(r=>r.missing)});
            if(seed==='terrax-1ptws5s'&&style==='rich')example=d;
        }
        console.log(`verified ${seed}`);
    }
    fs.writeFileSync(new URL('seed_validation.json',out),JSON.stringify(measurements,null,2)+'\n');
    fs.writeFileSync(new URL('example-data.json',out),JSON.stringify(example,null,2)+'\n');
});
test('manual changes replay exactly, preserve stars, reject illegal edits and remove optimum badges',()=>{
    const d=example??generateDraw(stars,meta,{seed:'terrax-1ptws5s'});let edited,chosen;
    for(let i=0;i<15&&!edited;i++)for(let e=0;e<d.regions[i].boundary.length&&!edited;e++)for(const step of [-1,1]){
        const a=d.regions[i].boundary[e],b=d.regions[i].boundary[(e+1)%d.regions[i].boundary.length],value=a[1]===b[1]?a[1]:a[0];
        try{const next=applyEditedGrid(d,stars,moveBoundaryEdge(d,i,e,value+step),d.recipe);if(gridEdits(d,next).length){edited=next;chosen={index:i,edge:e,target:value+step,coordinate:value};break;}}catch{}
    }
    assert.ok(edited,'at least one useful legal manual edge move');
    const recipe=normalizeRecipe({algorithm:MANUAL_ALGORITHM,base:d.recipe,edits:gridEdits(d,edited)}),rebuilt=generateDraw(stars,meta,recipe);
    assert.deepEqual(rebuilt.territories,edited.territories);assert.ok(!rebuilt.optimality&&!rebuilt.localOptimality);assert.equal(validateDraw(rebuilt,stars),true);
    assert.deepEqual(rebuilt.regions.map(r=>r.members),d.regions.map(r=>r.members));assert.ok(rebuilt.brightAudit.every(r=>!r.missing.length));
    const bad=structuredClone(rebuilt);bad.manualBoundary.status='optimal';assert.throws(()=>validateDraw(bad,stars),/不一致/);
    const broken=new Int8Array(unpackGrid(d.territories));broken[cellOf(d.regions[0].members[0].direction)]=15;
    assert.throws(()=>applyEditedGrid(d,stars,broken,d.recipe));assert.throws(()=>redrawRecipe(recipe,[],'r'),/手动/);
    fs.writeFileSync(new URL('manual-fixture.json',out),JSON.stringify({chosen,recipe,data:rebuilt},null,2)+'\n');
});
test('local rerolls keep locked bright figures and all seven historical examples still reproduce',()=>{
    const d=generateDraw(stars,meta,{seed:'terrax-1ptws5s'}),r=generateDraw(stars,meta,redrawRecipe(d.recipe,[0,6],'local-reroll'));
    assert.deepEqual(r.territories,d.territories);assert.deepEqual(r.regions[0],d.regions[0]);assert.deepEqual(r.regions[6],d.regions[6]);
    assert.notDeepEqual(r.regions,d.regions);assert.ok(r.brightAudit.every(a=>!a.missing.length&&!a.missingFromCore.length));
    const changedMeta={catalogue:'same-stars-new-reference.json',sha256:'different-catalogue-fingerprint'},fresh=generateDraw(stars,changedMeta,d.recipe);
    assert.equal(fresh.catalogue,changedMeta.catalogue);assert.equal(fresh.sha256,changedMeta.sha256);assert.deepEqual(fresh.territories,d.territories);
    for(const name of ['example','ordered_example','free_example','clean_example','ecliptic_example','fitted_example','minimum_example']){
        const saved=JSON.parse(fs.readFileSync(new URL(`../design/zodiac_draw_${name}.json`,import.meta.url))).data;
        assert.equal(JSON.stringify(generateDraw(stars,meta,saved.recipe)),JSON.stringify(saved));
    }
});
