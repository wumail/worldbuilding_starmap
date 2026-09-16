import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {catalogueStars} from '../src/build_zodiac_candidates.mjs';
import {prepareDraw,generateDraw,normalizeRecipe,validateDraw,redrawRecipe,EDIT_ALGORITHM,equivalentDraw} from '../web/constellations/generator.mjs';
import {coordinates,vector,separation,tangentFrame,projectLocal,unprojectLocal} from '../web/constellations/geometry.mjs';
import {samplingCurve,samplingDeclination,samplingInfo} from '../web/constellations/sampling.mjs';
import {connectionStats,candidateRecipe,applyCandidateEdits,normalizeRemovedEdges} from '../web/constellations/candidate_edits.mjs';
import {moveBoundaryEdge,gridEdits} from '../web/constellations/boundary_edits.mjs';
import {packGrid,unpackGrid,toEquatorial} from '../web/constellations/territories.mjs';
import {validateGeometry} from '../web/constellations/generator_free.mjs';

const root=new URL('../',import.meta.url),read=p=>JSON.parse(fs.readFileSync(new URL(p,root)));
const meta=read('design/zodiac_candidates_v1.json'),stars=catalogueStars(read(meta.catalogue));
const out=new URL('reports/zodiac_candidate_editor/',root);fs.mkdirSync(out,{recursive:true});
await prepareDraw();let example;
test('sampling angle is canonical, bounded, and retained by local rerolls',()=>{
    assert.equal(normalizeRecipe().samplingHalfWidthDegrees,40);
    const r=normalizeRecipe({seed:'slider',samplingHalfWidthDegrees:47});assert.equal(redrawRecipe(r,[1],'reroll').samplingHalfWidthDegrees,47);
    for(const v of [NaN,Infinity,14,61,35.5,'40',null])assert.throws(()=>normalizeRecipe({samplingHalfWidthDegrees:v}));
    assert.equal(normalizeRecipe({algorithm:'terrax-zodiac-draw-8'}).samplingHalfWidthDegrees,undefined);
});
test('projected band edges have constant true angular distance to the ecliptic, including the seam',()=>{
    for(const width of [15,30,40,60])for(const sign of [-1,1]){
        const beta=sign*width;
        for(const [ra,v] of samplingCurve(beta).entries()){
            const c=coordinates(v),eq=coordinates(toEquatorial(v));
            assert.ok(Math.abs(c.latitude-beta)<1e-10);
            assert.ok(Math.abs(separation(v,vector(c.longitude,0))-width)<1e-10);
            assert.ok(Math.abs(eq.latitude-samplingDeclination(ra,beta))<1e-10);
        }
    }
    for(const ra of [0,45,90,180,270,360])assert.ok(Math.abs(coordinates(samplingCurve(0)[ra]).latitude)<1e-10);
});
test('candidate stereographic pointer inverse preserves direction away from center and across RA seam',()=>{
    for(const center of [[359,15],[15,-25],[90,45]]){
        const frame=tangentFrame(...center);
        for(const [dx,dy] of [[0,0],[16,10],[-20,-12]]){
            const v=vector(center[0]+dx,center[1]+dy),p=projectLocal(v,frame,421,283,8.6),back=unprojectLocal(p.x,p.y,frame,421,283,8.6);
            assert.ok(separation(v,back)<1e-10);
        }
    }
});
test('requested sampling bands retain geometry and brightness; a difficult sample may reject an unproven optimum',()=>{
    const measured=[];
    for(const seed of ['terrax-1ptws5s','terrax-b1bxb1'])for(const width of [30,40,55]){
        const start=performance.now();let data;
        try{data=generateDraw(stars,meta,{seed,samplingHalfWidthDegrees:width});}
        catch(error){
            assert.equal(seed,'terrax-b1bxb1');assert.equal(width,55);assert.match(error.message,/尚未证明最优/);
            measured.push({seed,width,status:'rejected-unproven',reason:error.message,seconds:(performance.now()-start)/1000});continue;
        }
        assert.equal(data.recipe.algorithm,'terrax-zodiac-draw-9');assert.deepEqual(data.sampling,samplingInfo(stars,width));
        assert.ok(data.brightAudit.every(r=>!r.missing.length&&!r.missingFromCore.length));assert.equal(validateDraw(data,stars),true);
        measured.push({seed,width,status:'validated',candidates:data.sampling.candidateCount,layoutSeed:data.layoutSeed,corners:data.localOptimality.minimumTotalAtLocalPriority,seconds:(performance.now()-start)/1000});
        if(seed==='terrax-1ptws5s'&&width===40)example=data;
        console.log(`verified ${seed} +/-${width}`);
    }
    assert.ok(measured[0].candidates<measured[1].candidates&&measured[1].candidates<measured[2].candidates);
    assert.ok(measured.filter(r=>r.status==='validated').length>=5);
    fs.writeFileSync(new URL('sample-validation.json',out),JSON.stringify(measured,null,2));
    fs.writeFileSync(new URL('example-data.json',out),JSON.stringify(example,null,2));
});
test('manual deletion can disconnect the figure without deleting stars or weakening automatic connectedness',()=>{
    const d=example??generateDraw(stars,meta,{seed:'terrax-1ptws5s'}),recipe=candidateRecipe(d,d),unique=new Map();
    for(const v of d.regions[0].variants)for(const e of v.edges)unique.set([e.from,e.to].sort().join('/'),[0,e.from,e.to]);
    recipe.removedEdges=[...unique.values()];const next=applyCandidateEdits(d,stars,recipe);
    assert.deepEqual(next.regions.map(r=>r.members),d.regions.map(r=>r.members));assert.deepEqual(next.territories,d.territories);
    assert.ok(next.regions[0].variants.every(v=>v.edges.length===0));assert.equal(next.regions[0].structure.loops,0);
    assert.equal(next.regions[0].structure.components,d.regions[0].members.length);assert.ok(!next.localOptimality&&!next.optimality);
    assert.ok(next.brightAudit.every(r=>!r.missing.length&&!r.missingFromCore.length));
    assert.throws(()=>validateGeometry(next,stars,{recipe:d.recipe,rules:d.settings,adaptiveCore:true,maximumMembers:15}),/不连通/);
    assert.deepEqual(generateDraw(stars,meta,normalizeRecipe(recipe)),next);assert.equal(validateDraw(next,stars),true);
    assert.deepEqual(connectionStats(['a','b','c'],[{from:'a',to:'b'}]),{loops:0,branches:0,tips:2,components:2,isolated:1});
    const bad=structuredClone(next);bad.localOptimality=d.localOptimality;assert.throws(()=>validateDraw(bad,stars));
    assert.throws(()=>normalizeRemovedEdges([[0,'a','b'],[0,'b','a']]));assert.throws(()=>applyCandidateEdits(d,stars,{...recipe,removedEdges:[[0,'absent','star']]}));
    assert.throws(()=>redrawRecipe(recipe,[],'reroll'));assert.throws(()=>normalizeRecipe({algorithm:EDIT_ALGORITHM,base:recipe,edits:[],removedEdges:[]}));
});
test('combined line and boundary edits replay, preserve both changes, and reject moving members outside their region',()=>{
    const d=example??generateDraw(stars,meta,{seed:'terrax-1ptws5s'}),first=d.regions[0].variants[1].edges[0],recipe=candidateRecipe(d,d);
    recipe.removedEdges=[[0,first.from,first.to]];const removed=applyCandidateEdits(d,stars,recipe);let next,action;
    for(let i=0;i<15&&!next;i++)for(let edge=0;edge<removed.regions[i].boundary.length&&!next;edge++)for(const step of [-1,1]){
        const a=removed.regions[i].boundary[edge],b=removed.regions[i].boundary[(edge+1)%removed.regions[i].boundary.length],coordinate=a[1]===b[1]?a[1]:a[0];
        try{
            const moved={...removed,territories:packGrid(moveBoundaryEdge(removed,i,edge,coordinate+step))},r=candidateRecipe(d,moved);
            const candidate=applyCandidateEdits(d,stars,r);if(gridEdits(removed,candidate).length){next=candidate;action={index:i,edge,target:coordinate+step,coordinate};break;}
        }catch{}
    }
    assert.ok(next);assert.equal(next.manualEdits.removedEdgeCount,1);assert.ok(gridEdits(d,next).length);
    assert.deepEqual(generateDraw(stars,meta,normalizeRecipe(next.recipe)),next);assert.equal(validateDraw(next,stars),true);
    const originalOnly=applyCandidateEdits(d,stars,{...next.recipe,edits:[]});assert.equal(originalOnly.manualEdits.removedEdgeCount,1);
    assert.deepEqual(originalOnly.territories,d.territories);
    const bad=structuredClone(next.recipe);bad.edits=[[Math.floor(unpackGrid(d.territories).length/2),99]];assert.throws(()=>applyCandidateEdits(d,stars,bad));
    fs.writeFileSync(new URL('manual-fixture.json',out),JSON.stringify({action,line:[0,first.from,first.to],recipe:next.recipe,data:next},null,2));
});
test('draw-8 and the saved manual-1 round retain their exact historical results',()=>{
    for(const path of ['design/zodiac_draw_local_example.json','reports/zodiac_local_first/exported-manual.json']){
        const saved=read(path),rebuilt=generateDraw(stars,meta,saved.data.recipe);
        if(path.startsWith('design/'))assert.deepEqual(rebuilt,saved.data);else assert.ok(equivalentDraw(rebuilt,saved.data));
    }
});
