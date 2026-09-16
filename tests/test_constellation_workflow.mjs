import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import crypto from 'node:crypto';
import {catalogueStars} from '../src/build_zodiac_candidates.mjs';
import {generateDraw,validateDraw,redrawRecipe,normalizeRecipe,equivalentDraw} from '../web/constellations/generator_ordered.mjs';
import {vector,dot,cross,unit,owner,separation,tangentFrame,projectLocal} from '../web/constellations/geometry.mjs';
import {layoutIssues,regionEnvelopes,longitudeEnvelope} from '../web/constellations/region_layout.mjs';

const initialBytes=fs.readFileSync(new URL('../design/zodiac_candidates_v1.json',import.meta.url)),initial=JSON.parse(initialBytes);
const sourceBytes=fs.readFileSync(new URL(`../${initial.catalogue}`,import.meta.url)),stars=catalogueStars(JSON.parse(sourceBytes));
const original=JSON.stringify(stars),first=generateDraw(stars,initial,{seed:'terrax-001'});
const signature=d=>JSON.stringify(d.regions.map(r=>({site:r.site,members:r.variants[1].members,edges:r.variants[1].edges})));

test('same recipe reproduces the whole draw without changing catalogue or first draft',()=>{
    assert.equal(JSON.stringify(generateDraw(stars,initial,first.recipe)),JSON.stringify(first));
    assert.equal(JSON.stringify(stars),original);
    assert.equal(crypto.createHash('sha256').update(initialBytes).digest('hex'),'07e79a209f40517171995ce7cecc0e4a4d0b73d2d862eaf651245b9fbfbb5ad7');
    assert.equal(crypto.createHash('sha256').update(sourceBytes).digest('hex'),initial.sha256);
    assert.notEqual(signature(first),signature(generateDraw(stars,initial,{seed:'terrax-002'})));
});

test('local reroll preserves all boundaries and every locked region, including notes recipe reconstruction',()=>{
    const locks=[0,4,14],recipe=redrawRecipe(first.recipe,locks,'second-shape'),next=generateDraw(stars,initial,recipe);
    for(let i=0;i<15;i++){
        assert.deepEqual(next.regions[i].site,first.regions[i].site);
        assert.deepEqual(next.regions[i].polygon,first.regions[i].polygon);
        assert.deepEqual(next.regions[i].intervals,first.regions[i].intervals);
        if(locks.includes(i))assert.deepEqual(next.regions[i],first.regions[i]);
        else assert.notEqual(recipe.shapeSeeds[i],first.recipe.shapeSeeds[i]);
    }
    assert.notEqual(signature(next),signature(first));
    assert.deepEqual(redrawRecipe(first.recipe,Array.from({length:15},(_,i)=>i),'ignored'),first.recipe);
    assert.deepEqual(generateDraw(stars,initial,JSON.parse(JSON.stringify(recipe))),next);
});

test('60 independent seeds in both complexities obey the physical, geometric and layout constraints',()=>{
    const signatures=new Set();let closed=0;
    for(let k=0;k<60;k++)for(const style of ['balanced','rich']){
        const d=generateDraw(stars,initial,{seed:`verify-${k}`,style}),sites=[...d.regions.map(r=>r.site),...d.remainderSites];
        assert.equal(validateDraw(d,stars),true);signatures.add(signature(d));
        assert.deepEqual(layoutIssues(d.regions),[]);
        const envelopes=d.regions.map(r=>longitudeEnvelope(r.members.map(s=>s.direction),r.site));
        for(let i=0;i<15;i++)for(let j=i+1;j<15;j++)for(const shift of [-360,0,360])assert.ok(Math.min(envelopes[i].end,envelopes[j].end+shift)-Math.max(envelopes[i].start,envelopes[j].start+shift)<1e-9,'figures overlap in longitude');
        for(const [i,r] of d.regions.entries()){
            const v=r.variants[1],members=new Map(r.members.map(s=>[s.id,s]));closed+=r.structure.loops>0?1:0;
            // Independent great-circle intersection check, not the generator's
            // gnomonic segment test. Ignore intersections at shared endpoints.
            for(let a=0;a<v.edges.length;a++)for(let b=a+1;b<v.edges.length;b++){
                const e=v.edges[a],f=v.edges[b];if([e.from,e.to].some(id=>id===f.from||id===f.to))continue;
                const [p,q,u,w]=[e.from,e.to,f.from,f.to].map(id=>members.get(id).direction),axis=cross(cross(p,q),cross(u,w));
                if(Math.hypot(...axis)<1e-12)continue;
                for(const sign of [-1,1]){
                    const x=unit(axis).map(n=>n*sign),onArc=(a,b)=>Math.abs(separation(a,x)+separation(x,b)-separation(a,b))<1e-7;
                    assert.ok(!(onArc(p,q)&&onArc(u,w)),`${d.recipe.seed} ${r.id} crossing`);
                }
            }
            const normals=r.polygon.map((p,j)=>{const n=cross(p,r.polygon[(j+1)%r.polygon.length]);return dot(n,r.site)<0?n.map(x=>-x):n;});
            for(const s of r.members){assert.equal(owner(s.direction,sites),i);assert.equal(owner(vector(s.longitude,0),sites),i);assert.ok(normals.every(n=>dot(n,s.direction)>=-1e-9));}
            for(const [width,height] of [[1046,610],[630,500],[334,460],[294,460]]){
                const frame=tangentFrame(r.center.longitude,r.center.latitude),ppd=Math.min((width-44)/58,(height-44)/52);
                for(const s of r.members){const p=projectLocal(s.direction,frame,width/2,height/2,ppd);assert.ok(p.visible&&p.x>5&&p.y>5&&p.x<width-5&&p.y<height-5);}
            }
        }
        for(let longitude=0;longitude<360;longitude+=.25){const v=vector(longitude,0),i=owner(v,sites);assert.ok(i<15);assert.ok(d.regions[i].intervals.some(s=>longitude>=s.start-1e-9&&longitude<=s.end+1e-9));}
        assert.equal(owner(vector(0,0),sites),owner(vector(360,0),sites));
    }
    assert.equal(signatures.size,120);assert.ok(closed>900);
});

test('validation rejects missing figures, altered stars, false boundaries, duplicate edges and false ecliptic intervals',()=>{
    const corruptions=[
        d=>{d.regions[0].members=[];d.regions[0].variants=[];},
        d=>{d.regions[0].variants=[];},
        d=>{d.regions[0].variants[1].members.pop();},
        d=>{d.regions[0].members[0].app_mag-=1;},
        d=>{d.regions[0].polygon=[];},
        d=>{d.regions[0].variants[1].edges.push({...d.regions[0].variants[1].edges[0]});},
        d=>{d.regions.forEach((r,i)=>{r.intervals=[{start:i*24,end:(i+1)*24}];r.eclipticSpan=24;});},
    ];
    for(const mutate of corruptions){const d=structuredClone(first);mutate(d);assert.throws(()=>validateDraw(d,stars),/候选检查未通过/);}
    assert.throws(()=>normalizeRecipe({seed:''}),/种子/);
    assert.throws(()=>normalizeRecipe({algorithm:'unknown'}),/不同的抽卡算法/);
    assert.throws(()=>normalizeRecipe({shapeSeeds:['incomplete']}),/不完整/);
});

test('portable recipe comparison tolerates derived trigonometric roundoff but never altered photometry or identity',()=>{
    const other=structuredClone(first);other.regions[0].members[0].longitude+=1e-12;other.regions[0].polygon[0][0]+=1e-14;
    assert.equal(equivalentDraw(first,other),true);
    other.regions[0].members[0].app_mag+=1e-12;assert.equal(equivalentDraw(first,other),false);
    const boundary=structuredClone(first);boundary.regions[0].intervals[0].end+=.00001;assert.equal(equivalentDraw(first,boundary),false);
});

test('reported terrax-b1bxb1 layout reproduces the old longitude containment and removes it in new draws',()=>{
    const before=generateDraw(stars,initial,{seed:'terrax-b1bxb1',algorithm:'terrax-zodiac-draw-1'}),after=generateDraw(stars,initial,{seed:'terrax-b1bxb1'});
    assert.equal(before.selectedExtendedCount,186);assert.equal(after.selectedExtendedCount,185);
    assert.ok(layoutIssues(before.regions).includes('Z05/Z06: longitude containment'));
    assert.ok(layoutIssues(before.regions).includes('Z08/Z09: longitude containment'));
    assert.deepEqual(layoutIssues(after.regions),[]);
    const ranges=regionEnvelopes(after.regions);assert.ok(ranges.some(r=>r.end>360||r.start<0));
    // Independent area test distinguishes longitude containment from actual
    // overlap of two sky regions. Both old and new area partitions are valid.
    for(const data of [before,after]){
        const sites=[...data.regions.map(r=>r.site),...data.remainderSites],normals=data.regions.map(r=>r.polygon.map((p,i)=>{const n=cross(p,r.polygon[(i+1)%r.polygon.length]);return dot(n,r.site)<0?n.map(x=>-x):n;}));
        for(let k=0;k<100000;k++){
            const z=1-2*(k+.5)/100000,phi=k*Math.PI*(3-Math.sqrt(5)),v=[Math.sqrt(1-z*z)*Math.cos(phi),Math.sqrt(1-z*z)*Math.sin(phi),z];
            const ids=normals.flatMap((ns,i)=>ns.every(n=>dot(n,v)>=-1e-10)?[i]:[]),index=owner(v,sites);
            assert.deepEqual(ids,index<15?[index]:[]);
        }
    }
});

test('older exported recipes retain their precise original output and locked rerolls use the same old partition',()=>{
    const example=JSON.parse(fs.readFileSync(new URL('../design/zodiac_draw_example.json',import.meta.url))).data;
    const restored=generateDraw(stars,initial,example.recipe);assert.equal(JSON.stringify(restored),JSON.stringify(example));
    const next=generateDraw(stars,initial,redrawRecipe(example.recipe,[2,14],'legacy-reroll'));
    assert.equal(next.recipe.algorithm,'terrax-zodiac-draw-1');
    for(let i=0;i<15;i++){assert.deepEqual(next.regions[i].polygon,example.regions[i].polygon);if([2,14].includes(i))assert.deepEqual(next.regions[i],example.regions[i]);}
});
