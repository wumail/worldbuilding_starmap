import fs from 'node:fs';
import test from 'node:test';
import assert from 'node:assert/strict';
import {wrap} from '../web/constellations/geometry.mjs';
import {unpackGrid,packGrid} from '../web/constellations/territories.mjs';
import {shiftBoundarySegment,boundaryStep,isBoundaryCorner,moveBoundaryCorner,removeBoundaryCorner} from '../web/constellations/boundary_geometry.mjs';
import {manualBoundaryRings,manualRecipe,applyManualFigures,containmentIssues,cornerCount} from '../web/constellations/manual_figures.mjs';
import {catalogueStars} from '../src/build_zodiac_candidates.mjs';
import {prepareDraw,generateDraw,equivalentDraw} from '../web/constellations/generator.mjs';
const read=path=>JSON.parse(fs.readFileSync(new URL('../'+path,import.meta.url)));
const rect=(left,right,bottom,top,outside=15)=>{
    const cells=new Int8Array(64800).fill(outside);
    for(let y=bottom+90;y<top+90;y++)for(let x=left;x<right;x++)cells[y*360+wrap(x)]=0;return cells;
};
const count=c=>c.reduce((n,v)=>n+(v===0),0),point=(r,x,y)=>r.findIndex(p=>p[0]===x&&p[1]===y);
const difference=(a,b)=>Array.from(a.keys()).filter(k=>a[k]!==b[k]);
const equalCells=(a,b)=>assert.equal(difference(a,b).length,0,'the complete ownership grids must match');
const orthogonal=c=>{for(const r of manualBoundaryRings(c,0))for(let i=0;i<r.length;i++){const a=r[i],b=r[(i+1)%r.length];assert.ok(a[0]===b[0]||a[1]===b[1]);}};

test('inserting a partial step adds real corners without shifting the remainder of the edge',()=>{
    const cells=rect(10,30,-10,10),before=new Int8Array(cells),a=[30,10],b=[10,10];
    const step=boundaryStep(cells,0,a,b,[20,10],4);assert.equal(step.width,4);assert.equal(step.target,11);
    const added=shiftBoundarySegment(cells,0,step.a,step.b,step.target);assert.equal(count(added),count(cells)+4);
    assert.equal(cornerCount({boundaryRings:manualBoundaryRings(added,0)}),8);
    for(const k of difference(cells,added)){assert.equal(Math.floor(k/360)-90,10);assert.ok(k%360>=18&&k%360<22);}
    const notched=shiftBoundarySegment(cells,0,step.a,step.b,7);assert.equal(count(notched),count(cells)-12);
    assert.equal(cornerCount({boundaryRings:manualBoundaryRings(notched,0)}),8);orthogonal(added);orthogonal(notched);assert.deepEqual(cells,before);
});

test('partial steps preserve their intended span and side across the zero-degree seam',()=>{
    const cells=rect(350,371,0,20),a=[11,20],b=[350,20],step=boundaryStep(cells,0,a,b,[359,20],6);
    const next=shiftBoundarySegment(cells,0,step.a,step.b,22);assert.equal(count(next)-count(cells),12);
    const columns=new Set(difference(cells,next).map(k=>k%360));assert.deepEqual(columns,new Set([356,357,358,359,0,1]));
    assert.deepEqual(shiftBoundarySegment(cells,0,step.b,step.a,22),next);orthogonal(next);
});

test('moving each rectangle corner changes both incident edges and exactly the expected cell union',()=>{
    for(const [old,target,bounds] of [
        [[30,10],[33,13],[10,33,-10,13]],[[30,10],[27,7],[10,27,-10,7]],
        [[10,10],[7,13],[7,30,-10,13]],[[10,-10],[7,-13],[7,30,-13,10]],
        [[30,-10],[33,-13],[10,33,-13,10]]
    ]){
        const cells=rect(10,30,-10,10),ring=manualBoundaryRings(cells,0)[0],next=moveBoundaryCorner(cells,0,ring,point(ring,...old),target);
        equalCells(next,rect(...bounds));orthogonal(next);
    }
    const seam=rect(350,370,-10,10),ring=manualBoundaryRings(seam,0)[0];
    equalCells(moveBoundaryCorner(seam,0,ring,point(ring,10,10),[12,12]),rect(350,372,-10,12));
    const wide=rect(20,250,-10,10),wideRing=manualBoundaryRings(wide,0)[0];
    equalCells(moveBoundaryCorner(wide,0,wideRing,point(wideRing,250,10),[255,12]),rect(20,255,-10,12));
});

test('removing an inserted corner can merge the step, while both collapse directions remain selectable',()=>{
    const cells=rect(10,30,-10,10),step=boundaryStep(cells,0,[30,10],[10,10],[20,10],4),expanded=shiftBoundarySegment(cells,0,step.a,step.b,12);
    const ring=manualBoundaryRings(expanded,0)[0],i=point(ring,18,12);assert.ok(i>=0);
    assert.deepEqual(removeBoundaryCorner(expanded,0,ring,i,'previous'),cells);
    assert.deepEqual(removeBoundaryCorner(expanded,0,ring,i,'next'),cells);
    const base=read('design/zodiac_draw_sampling_example.json').data;let choices;
    for(const [index,r] of base.regions.entries())for(let j=0;j<r.boundary.length;j++)try{
        const original=unpackGrid(base.territories),a=removeBoundaryCorner(original,index,r.boundary,j,'previous'),b=removeBoundaryCorner(original,index,r.boundary,j,'next');
        if(difference(a,b).length){choices={index,a,b,p:r.boundary[j]};break;}
    }catch{}
    assert.ok(choices);for(const c of [choices.a,choices.b]){
        assert.ok(!manualBoundaryRings(c,choices.index).some(r=>r.some((p,j)=>p[0]===choices.p[0]&&p[1]===choices.p[1]&&isBoundaryCorner(r,j))));
    }
});

test('shared neighbours receive retracted strips and a cancelled proposal leaves all original cells intact',()=>{
    const cells=rect(10,30,-10,10,1),original=new Int8Array(cells),step=boundaryStep(cells,0,[30,10],[10,10],[20,10],4);
    const expanded=shiftBoundarySegment(cells,0,step.a,step.b,13);assert.deepEqual(cells,original);
    assert.deepEqual(shiftBoundarySegment(expanded,0,[step.a[0],13],[step.b[0],13],10),cells);
    assert.equal(expanded.filter(v=>v===0).length+expanded.filter(v=>v===1).length,64800);
});

test('holes, disconnected components and both polar limits support local steps without changing remote parts',()=>{
    const cells=rect(10,35,-15,15);for(let y=85;y<95;y++)for(let x=18;x<24;x++)cells[y*360+x]=15;cells[100]=0;
    const hole=manualBoundaryRings(cells,0).find(r=>r.some(p=>p[0]===18&&p[1]===5));assert.ok(hole);
    const i=hole.findIndex((a,j)=>a[1]===hole[(j+1)%hole.length][1]&&Math.abs(a[0]-hole[(j+1)%hole.length][0])>=3);
    const a=hole[i],b=hole[(i+1)%hole.length],step=boundaryStep(cells,0,a,b,[(a[0]+b[0])/2,a[1]],2),next=shiftBoundarySegment(cells,0,step.a,step.b,step.target);
    assert.equal(next[100],0);assert.ok(difference(cells,next).length>0);orthogonal(next);
    for(const [bottom,top,lat] of [[80,90,90],[-90,-80,-90]]){
        const polar=rect(30,50,bottom,top),s=boundaryStep(polar,0,[30,lat],[50,lat],[40,lat],4);
        assert.equal(s.target,lat===90?89:-89);assert.equal(count(shiftBoundarySegment(polar,0,s.a,s.b,s.target)),count(polar)-4);
    }
});

test('invalid widths, polar coordinates and degenerate or stale edges fail without mutating the source',()=>{
    const cells=rect(10,30,-10,10),saved=new Int8Array(cells),ring=manualBoundaryRings(cells,0)[0];
    assert.throws(()=>boundaryStep(cells,0,[10,10],[12,10],[11,10],4),/不足 3/);
    assert.throws(()=>boundaryStep(cells,0,[10,10],[30,10],[20,10],NaN),/宽度/);
    assert.throws(()=>moveBoundaryCorner(cells,0,ring,0,[20,91]),/赤纬/);
    assert.throws(()=>shiftBoundarySegment(cells,0,[10,10],[30,5],11),/横边或竖边/);
    assert.throws(()=>shiftBoundarySegment(cells,0,[10,9],[30,9],11),/变化/);
    assert.deepEqual(cells,saved);
});

test('new corners persist through manual-3 replay, preserving stars and full-arc containment checks',async()=>{
    const base=read('design/zodiac_draw_sampling_example.json').data,stars=catalogueStars(read(base.catalogue)),original=JSON.stringify(base),source=JSON.stringify(stars);let result;
    outer:for(const [index,r] of base.regions.entries())for(let i=0;i<r.boundary.length;i++)try{
        const a=r.boundary[i],b=r.boundary[(i+1)%r.boundary.length],s=boundaryStep(unpackGrid(base.territories),index,a,b,[(a[0]+b[0])/2,(a[1]+b[1])/2],3);
        const cells=shiftBoundarySegment(unpackGrid(base.territories),index,s.a,s.b,s.target),recipe=manualRecipe(base,{...base,territories:packGrid(cells)});
        const next=applyManualFigures(base,stars,recipe);if(cornerCount(next.regions[index])>cornerCount(r)){result=next;break outer;}
    }catch{}
    assert.ok(result);assert.deepEqual(containmentIssues(result),[]);
    for(let i=0;i<15;i++){assert.deepEqual(result.regions[i].members,base.regions[i].members);assert.deepEqual(result.regions[i].variants,base.regions[i].variants);}
    await prepareDraw(result.recipe);assert.ok(equivalentDraw(generateDraw(stars,base,result.recipe),result));
    assert.equal(JSON.stringify(base),original);assert.equal(JSON.stringify(stars),source);
});
