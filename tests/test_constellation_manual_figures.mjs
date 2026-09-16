import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {catalogueStars} from '../src/build_zodiac_candidates.mjs';
import {prepareDraw,generateDraw,validateDraw,normalizeRecipe,equivalentDraw,FREE_EDIT_ALGORITHM} from '../web/constellations/generator.mjs';
import {manualRecipe,editManualFigure,applyManualFigures,containmentIssues,manualArcCells,manualBoundaryRings,moveManualBoundary,regionRings} from '../web/constellations/manual_figures.mjs';
import {moveBoundaryEdge,gridEdits} from '../web/constellations/boundary_edits.mjs';
import {candidateRecipe,applyCandidateEdits} from '../web/constellations/candidate_edits.mjs';
import {cellOf,unpackGrid,packGrid,fromEquatorial,boundaryPoints} from '../web/constellations/territories.mjs';
import {equatorialPath} from '../web/constellations/map_paths.mjs';
import {vector,arc,separation,delta,cross,unit} from '../web/constellations/geometry.mjs';
const read=path=>JSON.parse(fs.readFileSync(new URL('../'+path,import.meta.url))),meta=read('design/zodiac_candidates_v1.json'),stars=catalogueStars(read(meta.catalogue));
const base=read('reports/zodiac_candidate_editor/example-data.json'),cells=unpackGrid(base.territories),save=current=>applyManualFigures(base,stars,manualRecipe(base,current));
const baseFingerprint=JSON.stringify(base),starsFingerprint=JSON.stringify(stars);
let sample;

test('manual member edits use real stars beyond the automatic magnitude/member limits and replay IDs and edges',()=>{
    const group=stars.filter(s=>cells[cellOf(s.direction)]===0),dim=group.find(s=>s.app_mag>4.5&&!base.regions[0].members.some(m=>m.id===s.id));assert.ok(dim);
    let d=editManualFigure(base,base,stars,0,{type:'add-member',id:dim.id});
    assert.ok(d.regions[0].variants.every(v=>v.members.includes(dim.id)));assert.deepEqual(d.regions[0].members.find(s=>s.id===dim.id).direction,dim.direction);
    for(const s of group.slice(0,25))d=editManualFigure(base,d,stars,0,{type:'add-member',id:s.id});
    assert.ok(d.regions[0].members.length>15);assert.deepEqual(containmentIssues(d),[]);sample=save(d);
    assert.ok(!sample.optimality&&!sample.localOptimality&&!sample.brightPolicy&&!sample.envelopePolicy&&!sample.eclipticPolicy);
    assert.equal(sample.settings.mode,'manual');assert.equal(sample.manualEdits.status,'validated');assert.equal(normalizeRecipe(sample.recipe).algorithm,FREE_EDIT_ALGORITHM);
    assert.deepEqual(applyManualFigures(base,stars,normalizeRecipe(sample.recipe)),sample);
});
test('the brightest member, every member and all incident edges can be removed without auto-restoration',()=>{
    const brightest=[...base.regions[0].members].sort((a,b)=>a.app_mag-b.app_mag)[0];let d=editManualFigure(base,base,stars,0,{type:'remove-member',id:brightest.id});
    assert.ok(d.regions[0].variants.every(v=>!v.members.includes(brightest.id)&&v.edges.every(e=>e.from!==brightest.id&&e.to!==brightest.id)));
    assert.ok(d.brightAudit[0].missing.includes(brightest.id));assert.equal(save(d).manualEdits.status,'validated');
    for(const s of [...d.regions[0].members])d=editManualFigure(base,d,stars,0,{type:'remove-member',id:s.id});
    const result=save(d);assert.equal(result.regions[0].members.length,0);assert.equal(result.regions[0].brightest,null);assert.equal(result.regions[0].faintest,null);
    assert.ok(result.regions[0].variants.every(v=>!v.members.length&&!v.edges.length));assert.equal(result.regions[0].structure.components,0);assert.equal(result.regions[0].structure.loops,0);
    assert.deepEqual(JSON.parse(JSON.stringify(result)),result);
});
test('manual edges may exceed 13 degrees and the automatic degree limit, provided their true arcs fit',()=>{
    const region=base.regions[0],group=stars.filter(s=>cells[cellOf(s.direction)]===0),pivot=region.members[0];
    const targets=group.filter(s=>s.id!==pivot.id&&manualArcCells(s.direction,pivot.direction).every(k=>cells[k]===0));assert.ok(targets.length>5);
    let d=base;for(const s of targets.slice(0,8)){
        if(!d.regions[0].variants[1].edges.some(e=>[e.from,e.to].includes(s.id)&&[e.from,e.to].includes(pivot.id)))d=editManualFigure(base,d,stars,0,{type:'add-edge',from:pivot.id,to:s.id});
    }
    assert.ok(d.regions[0].variants[1].edges.filter(e=>e.from===pivot.id||e.to===pivot.id).length>4);
    const distant=targets.find(s=>separation(s.direction,pivot.direction)>13);assert.ok(distant);
    if(!d.regions[0].variants[1].edges.some(e=>[e.from,e.to].includes(distant.id)&&[e.from,e.to].includes(pivot.id)))d=editManualFigure(base,d,stars,0,{type:'add-edge',from:pivot.id,to:distant.id});
    assert.ok(save(d).regions[0].variants[1].edges.some(e=>e.degrees>13));
    const longEdge=d.regions[0].variants[1].edges.find(e=>e.degrees>13),before=d.regions[0].members.length;
    d=editManualFigure(base,d,stars,0,{type:'remove-edge',...longEdge});assert.equal(save(d).regions[0].members.length,before);
});
test('out-of-bound stars remain an editable draft but cannot be confirmed; fictitious and shared members are rejected',()=>{
    const outside=stars.find(s=>cells[cellOf(s.direction)]===15),d=editManualFigure(base,base,stars,0,{type:'add-member',id:outside.id});
    assert.equal(d.manualEdits.status,'draft');assert.ok(containmentIssues(d).some(s=>s.includes(outside.id)));assert.throws(()=>save(d),/边界外/);
    assert.throws(()=>editManualFigure(base,base,stars,0,{type:'add-member',id:'not-a-star'}),/源星表/);
    assert.throws(()=>editManualFigure(base,base,stars,0,{type:'add-member',id:base.regions[1].members[0].id}),/Z02/);
    assert.throws(()=>editManualFigure(base,base,stars,0,{type:'add-edge',from:outside.id,to:outside.id}),/不同/);
});
test('containment checks the whole spherical edge, both variants and neighbouring constellations',()=>{
    let hole;
    for(const r of base.regions)for(const v of r.variants)for(const e of v.edges){
        const byId=new Map(r.members.map(s=>[s.id,s]));const ends=new Set(stars.filter(s=>r.members.some(m=>m.id===s.id)).map(s=>cellOf(s.direction)));
        const k=manualArcCells(byId.get(e.from).direction,byId.get(e.to).direction).find(k=>!ends.has(k));if(k!==undefined){hole=k;break;}
    }
    assert.ok(Number.isInteger(hole));const recipe=manualRecipe(base,base);recipe.edits=[[hole,15]];assert.throws(()=>applyManualFigures(base,stars,recipe),/连线|边界外/);
    const other=base.regions[1].members[0];recipe.edits=[[cellOf(other.direction),0]];assert.throws(()=>applyManualFigures(base,stars,recipe),/Z02/);
});
test('manual boundaries are not capped by 8 degree envelopes, 40 degree moves, ecliptic width or automatic topology',()=>{
    let chosen;
    outer:for(let i=0;i<15;i++){
        const r=base.regions[i];for(let j=0;j<r.boundary.length;j++)for(const step of [-45,45]){
            const a=r.boundary[j],b=r.boundary[(j+1)%r.boundary.length];if(a[1]!==b[1])continue;const target=a[1]+step;if(Math.abs(target)>90)continue;
            try{const next=moveManualBoundary(base,i,0,j,target),recipe=manualRecipe(base,{...base,territories:packGrid(next)});if(!recipe.edits.length)continue;
                const data=applyManualFigures(base,stars,recipe);chosen={i,j,target,data};break outer;}catch{}
        }
    }
    assert.ok(chosen);assert.throws(()=>moveBoundaryEdge(base,chosen.i,chosen.j,chosen.target),/40°/);
    assert.throws(()=>applyCandidateEdits(base,stars,{...candidateRecipe(base,base),edits:chosen.data.recipe.edits}),/余量|黄道|候选|极点|洞/);
    assert.deepEqual(containmentIssues(chosen.data),[]);assert.ok(gridEdits(base,chosen.data).length>100);
    const isolated=new Int8Array(64800).fill(15);isolated[0]=0;isolated[360*90+120]=0;
    const rings=manualBoundaryRings(isolated,0);assert.equal(rings.length,2);assert.ok(rings.flat().some(p=>p[1]===-90));
    for(const ring of rings)for(let i=0;i<ring.length;i++)assert.ok(ring[i][0]===ring[(i+1)%ring.length][0]||ring[i][1]===ring[(i+1)%ring.length][1]);
});
test('global arc containment matches dense independent sampling across seams and near both poles',()=>{
    for(const pair of [[[359,20],[5,-15]],[[179,85],[359,84.8]],[[0,-84],[179,-86]],[[90,10],[270.1,10.1]]]){
        const [a,b]=pair.map(p=>fromEquatorial(vector(...p))),covered=new Set(manualArcCells(a,b));
        for(const p of arc(a,b,.0097))assert.ok(covered.has(cellOf(p)),JSON.stringify(pair));
    }
    assert.throws(()=>manualArcCells(vector(0,0),vector(180,0)),/无法唯一/);
});
test('manual-3 survives generator replay and tamper validation while historical manual-2 and draw-9 stay unchanged',async()=>{
    await prepareDraw(base.recipe);
    assert.ok(equivalentDraw(generateDraw(stars,meta,base.recipe),base));
    assert.ok(equivalentDraw(generateDraw(stars,meta,sample.recipe),sample));assert.equal(validateDraw(sample,stars),true);
    const forged=structuredClone(sample);forged.regions[0].members[0].app_mag-=1;assert.throws(()=>validateDraw(forged,stars));
    forged.regions[0].members[0].app_mag+=1;forged.localOptimality=base.localOptimality;assert.throws(()=>validateDraw(forged,stars));
    const oldest=read('reports/zodiac_local_first/exported-manual.json').data;assert.ok(equivalentDraw(generateDraw(stars,meta,oldest.recipe),oldest));
    const old=read('reports/zodiac_candidate_editor/exported-manual.json').data;assert.ok(equivalentDraw(generateDraw(stars,meta,old.recipe),old));
    const migrated=applyManualFigures(base,stars,manualRecipe(base,old));assert.deepEqual(migrated.territories,old.territories);
    for(let i=0;i<15;i++)assert.deepEqual(new Set(migrated.regions[i].variants[1].edges.map(e=>[e.from,e.to].sort().join('/'))),new Set(old.regions[i].variants[1].edges.map(e=>[e.from,e.to].sort().join('/'))));
    assert.equal(JSON.stringify(base),baseFingerprint);assert.equal(JSON.stringify(stars),starsFingerprint);
});

test('a manual figure may deliberately contain crossing edges and disconnected members',()=>{
    const group=stars.filter(s=>cells[cellOf(s.direction)]===0).slice(0,28),links=[];
    for(let i=0;i<group.length;i++)for(let j=i+1;j<group.length;j++)if(manualArcCells(group[i].direction,group[j].direction).every(k=>cells[k]===0))links.push([group[i],group[j]]);
    const inside=(a,b,p)=>Math.abs(separation(a,p)+separation(p,b)-separation(a,b))<1e-7;
    let found;
    outer:for(const [a,b] of links)for(const [c,d] of links){
        if(new Set([a.id,b.id,c.id,d.id]).size<4)continue;
        const normal=cross(cross(a.direction,b.direction),cross(c.direction,d.direction));if(Math.hypot(...normal)<1e-9)continue;
        for(const sign of [-1,1]){const p=unit(normal).map(x=>x*sign);if(inside(a.direction,b.direction,p)&&inside(c.direction,d.direction,p)){found=[a,b,c,d];break outer;}}
    }
    assert.ok(found);const [a,b,c,d]=found,recipe=manualRecipe(base,base);
    recipe.figures=[{index:0,members:found.map(s=>s.id),coreMembers:[],edges:[[a.id,b.id],[c.id,d.id]],coreEdges:[]}];
    const result=applyManualFigures(base,stars,recipe);assert.equal(result.regions[0].structure.components,2);assert.equal(result.regions[0].variants[1].edges.length,2);assert.equal(result.manualEdits.status,'validated');
});
test('a winding polar boundary closes on the sky without a spurious full-width line on the map',()=>{
    const grid=new Int8Array(64800).fill(15);
    for(let y=150;y<180;y++)for(let x=0;x<360;x++)if(x<180||x>=270||y>=160)grid[y*360+x]=0;
    const ring=manualBoundaryRings(grid,0).find(r=>r.some(p=>p[1]===60)&&r.some(p=>p[1]===70));assert.ok(ring);
    const path=equatorialPath(boundaryPoints(ring),0,true);assert.ok(Math.abs(path.at(-1).longitude-path[0].longitude)>359);
    for(let i=1;i<path.length;i++)assert.ok(Math.abs(path[i].longitude-path[i-1].longitude)<=.51);
    const badSegment=path.some((p,i)=>i&&Math.abs(p.latitude-60)<1e-8&&Math.abs(path[i-1].latitude-60)<1e-8&&p.longitude>180&&p.longitude<270);
    assert.equal(badSegment,false);
});
