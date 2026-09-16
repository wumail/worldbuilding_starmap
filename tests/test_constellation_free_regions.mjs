import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import crypto from 'node:crypto';
import {catalogueStars} from '../src/build_zodiac_candidates.mjs';
import {generateDraw as generateAny,validateDraw,redrawRecipe,equivalentDraw} from '../web/constellations/generator.mjs';
// Preserve the draw-5 free-layout baseline; draw-6 has its own fitted-area audit.
const generateDraw=(source,meta,recipe={})=>generateAny(source,meta,{algorithm:'terrax-zodiac-draw-5',...recipe});
import {vector,coordinates,delta,owner,cross,unit,separation,tangentFrame,projectLocal} from '../web/constellations/geometry.mjs';
import {longitudeEnvelope} from '../web/constellations/region_layout.mjs';
import {regionAt,arcCells,cellOf,fromEquatorial,growTerritories} from '../web/constellations/territories.mjs';

const initial=JSON.parse(fs.readFileSync(new URL('../design/zodiac_candidates_v1.json',import.meta.url)));
const bytes=fs.readFileSync(new URL(`../${initial.catalogue}`,import.meta.url)),stars=catalogueStars(JSON.parse(bytes));
const original=JSON.stringify(stars),first=generateDraw(stars,initial,{seed:'terrax-b1bxb1'});

function shapeMetrics(data,old){
    const sites=[...old.regions.map(r=>r.site),...old.remainderSites],envelopes=data.regions.map(r=>longitudeEnvelope(r.members.map(s=>s.direction),r.site));
    let horizontal=0,crossOld=0,overlappingPairs=0,latitudeChanges=0,crossOldEdges=0;
    for(const r of data.regions){
        const frame=tangentFrame(r.center.longitude,r.center.latitude),p=r.members.map(s=>projectLocal(s.direction,frame,0,0,1));
        const width=Math.max(...p.map(s=>s.x))-Math.min(...p.map(s=>s.x)),height=Math.max(...p.map(s=>s.y))-Math.min(...p.map(s=>s.y));
        if(width>=16&&width>height*1.2)horizontal++;
        const ids=new Map(r.members.map(s=>[s.id,owner(s.direction,sites)]));
        if(new Set(ids.values()).size>1)crossOld++;
        crossOldEdges+=r.variants[1].edges.filter(e=>ids.get(e.from)!==ids.get(e.to)).length;
    }
    for(let i=0;i<15;i++)for(let j=i+1;j<15;j++)if(Math.abs(data.regions[i].center.latitude-data.regions[j].center.latitude)>6&&[-360,0,360].some(s=>Math.min(envelopes[i].end,envelopes[j].end+s)-Math.max(envelopes[i].start,envelopes[j].start+s)>=3))overlappingPairs++;
    for(let l=.5;l<360;l++){const a=regionAt(data,vector(l,15)),b=regionAt(data,vector(l,-15));if(a<15&&b<15&&a!==b)latitudeChanges++;}
    return {horizontal,crossOld,crossOldEdges,overlappingPairs,latitudeChanges};
}

test('new results are reproducible and preserve source photometry and all four earlier exported algorithms',()=>{
    assert.deepEqual(generateDraw(stars,initial,JSON.parse(JSON.stringify(first.recipe))),first);
    assert.equal(JSON.stringify(stars),original);
    assert.equal(crypto.createHash('sha256').update(bytes).digest('hex'),initial.sha256);
    for(const name of ['zodiac_draw_example.json','zodiac_draw_ordered_example.json','zodiac_draw_free_example.json','zodiac_draw_clean_example.json']){
        const saved=JSON.parse(fs.readFileSync(new URL(`../design/${name}`,import.meta.url))).data;
        assert.equal(JSON.stringify(generateDraw(stars,initial,saved.recipe)),JSON.stringify(saved));
        const next=generateDraw(stars,initial,redrawRecipe(saved.recipe,[0,14],'legacy-round'));
        assert.deepEqual(next.regions[0],saved.regions[0]);assert.deepEqual(next.regions[14],saved.regions[14]);
    }
});

test('free-region local rerolls preserve every boundary and locked figure while changing the other figures',()=>{
    const locks=[0,4,14],recipe=redrawRecipe(first.recipe,locks,'free-reroll'),next=generateDraw(stars,initial,recipe);
    assert.deepEqual(first.territories,next.territories);
    for(let i=0;i<15;i++){
        for(const key of ['site','polygon','boundary','intervals','candidateCount'])assert.deepEqual(next.regions[i][key],first.regions[i][key]);
        if(locks.includes(i))assert.deepEqual(next.regions[i],first.regions[i]);
    }
    assert.notDeepEqual(first.regions.map(r=>r.members),next.regions.map(r=>r.members));
    assert.deepEqual(generateDraw(stars,initial,JSON.parse(JSON.stringify(recipe))),next);
    assert.deepEqual(redrawRecipe(first.recipe,Array.from({length:15},(_,i)=>i),'ignored'),first.recipe);
});

test('60 seeds in both styles contain actual lateral figures and staggered selection, rather than renamed strips',()=>{
    const results=[];
    for(let k=0;k<60;k++){
        const seed=k===0?'terrax-b1bxb1':k===1?'terrax-001':`verify-${k-2}`;
        const old=generateDraw(stars,initial,{seed,algorithm:'terrax-zodiac-draw-2'});
        for(const style of ['balanced','rich']){
            const data=generateDraw(stars,initial,{seed,style}),m=shapeMetrics(data,old),spans=data.regions.map(r=>r.eclipticSpan);
            assert.ok(Math.min(...spans)>=6-1e-9&&Math.max(...spans)<=44+1e-9);
            assert.ok(spans.filter(s=>s<16-1e-9).length<=1,`${seed}/${style}: too many narrow ecliptic crossings`);
            assert.ok(m.horizontal>=2,`${seed}/${style}: lateral figures ${m.horizontal}`);
            assert.ok(m.crossOld>=7&&m.crossOldEdges>=10,`${seed}/${style}: real members/edges must cross former ownership`);
            assert.ok(m.overlappingPairs>=3,`${seed}/${style}: no substantial north/south staggering`);
            assert.ok(m.latitudeChanges>=160,`${seed}/${style}: region ownership still follows longitude strips`);
            const ids=data.regions.flatMap(r=>r.members.map(s=>s.id));assert.equal(new Set(ids).size,ids.length);
            for(const [i,r] of data.regions.entries()){
                for(const s of r.members){const source=stars.find(x=>x.id===s.id);assert.equal(s.app_mag,source.app_mag);assert.deepEqual(s.direction,source.direction);assert.equal(regionAt(data,s.direction),i);}
                for(const [width,height] of [[1046,610],[630,500],[334,460],[294,460]]){
                    const ppd=Math.min((width-44)/72,(height-44)/56),frame=tangentFrame(r.center.longitude,r.center.latitude);
                    for(const s of r.members){const p=projectLocal(s.direction,frame,width/2,height/2,ppd);assert.ok(p.visible&&p.x>5&&p.y>5&&p.x<width-5&&p.y<height-5,`${seed} ${r.id}: clipped`);}
                }
            }
            const sections=data.regions.flatMap(r=>r.intervals.map(s=>({...s,id:r.id}))).sort((a,b)=>a.start-b.start);
            assert.equal(sections[0].start,0);assert.equal(sections.at(-1).end,360);
            for(let i=1;i<sections.length;i++)assert.ok(Math.abs(sections[i].start-sections[i-1].end)<1e-9);
            for(let l=.073;l<360;l+=.173){const i=regionAt(data,vector(l,0));assert.ok(i<15);assert.ok(data.regions[i].intervals.some(s=>l>=s.start&&l<=s.end));}
            assert.equal(regionAt(data,vector(0,0)),regionAt(data,vector(360,0)));
            results.push({seed,style,...m,spans});
        }
    }
    // This aggregate also prevents a regression to predominantly vertical figures.
    assert.ok(results.reduce((n,r)=>n+r.horizontal,0)/results.length>=4);
    fs.mkdirSync(new URL('../reports/zodiac_ecliptic_widths/',import.meta.url),{recursive:true});
    fs.writeFileSync(new URL('../reports/zodiac_ecliptic_widths/seed_validation.json',import.meta.url),JSON.stringify(results,null,2)+'\n');
});

test('shared concave boundaries agree with an independent polygon ray cast on 30000 sky directions',()=>{
    // Independent planar point-in-polygon in the boundary's reference frame;
    // this does not call the grid lookup to decide which polygon contains a point.
    const c=Math.cos(25*Math.PI/180),s=Math.sin(25*Math.PI/180);
    const eq=v=>coordinates([v[0],c*v[1]-s*v[2],s*v[1]+c*v[2]]);
    for(const data of [first,generateDraw(stars,initial,{seed:'terrax-001'})]){
        const polygons=data.regions.map(r=>{const center=eq(r.site).longitude;return {center,points:r.boundary.map(([x,y])=>[center+delta(x,center),y])};});
        const inside=(p,poly)=>{const x=poly.center+delta(p.longitude,poly.center),y=p.latitude;let yes=false;for(let a=0,b=poly.points.length-1;a<poly.points.length;b=a++){
            const [ax,ay]=poly.points[a],[bx,by]=poly.points[b];if((ay>y)!==(by>y)&&x<(bx-ax)*(y-ay)/(by-ay)+ax)yes=!yes;
        }return yes;};
        for(let k=0;k<15000;k++){
            const z=1-2*(k+.5)/15000,phi=k*Math.PI*(3-Math.sqrt(5)),v=[Math.sqrt(1-z*z)*Math.cos(phi),Math.sqrt(1-z*z)*Math.sin(phi),z],p=eq(v);
            const hits=polygons.flatMap((poly,i)=>inside(p,poly)?[i]:[]),index=regionAt(data,v);assert.deepEqual(hits,index<15?[index]:[]);
        }
    }
});

test('minor great-circle edges never cross other figures, including near the longitude seam',()=>{
    const edges=first.regions.flatMap(r=>{const byId=new Map(r.members.map(s=>[s.id,s.direction]));return r.variants[1].edges.map(e=>({...e,a:byId.get(e.from),b:byId.get(e.to)}));});
    for(let i=0;i<edges.length;i++)for(let j=i+1;j<edges.length;j++){
        const e=edges[i],f=edges[j];if([e.from,e.to].some(x=>x===f.from||x===f.to))continue;
        const axis=cross(cross(e.a,e.b),cross(f.a,f.b));if(Math.hypot(...axis)<1e-12)continue;
        for(const sign of [-1,1]){const x=unit(axis).map(v=>v*sign),on=(a,b)=>Math.abs(separation(a,x)+separation(x,b)-separation(a,b))<1e-7;assert.ok(!(on(e.a,e.b)&&on(f.a,f.b)));}
    }
});

test('arc protection finds brief parallel crossings between samples, including across the longitude seam',()=>{
    const R=Math.PI/180,dec=Math.atan(Math.tan(20.000001*R)*Math.cos(.5*R))/R;
    for(const start of [40,359.5]){
        const a=fromEquatorial(vector(start,dec)),b=fromEquatorial(vector(start+1,dec)),middle=unit(a.map((x,i)=>x+b[i]));
        assert.ok(arcCells(a,b).includes(cellOf(middle)),'missed the interior latitude maximum');
        const f={members:[{id:'a',direction:a},{id:'b',direction:b}],edges:[{from:'a',to:'b'}]};
        assert.equal(growTerritories([f],[middle]),null,'must reject an external seed on the actual arc');
    }
});

test('validation rejects altered physical data, grid ownership, outlines, intervals and graph contents',()=>{
    for(const mutate of [
        d=>{d.regions[0].members[0].app_mag-=1;},d=>{d.regions[0].variants=[];},d=>{d.regions[0].polygon=[];},
        d=>{d.regions[0].boundary[0][0]+=.1;},d=>{d.territories.runs[0][1]=16;},d=>{d.territories.runs[0][0]++;},
        d=>{d.regions[0].variants[1].edges.push({...d.regions[0].variants[1].edges[0]});},d=>{d.regions[0].intervals[0].end+=.1;},
        d=>{d.regions[0].center.longitude+=3;},d=>{d.regions[0].members[0]={...d.regions[1].members[0]};}
    ]){const d=structuredClone(first);mutate(d);assert.throws(()=>validateDraw(d,stars),/候选检查未通过/);}
    const d=structuredClone(first);validateDraw(d,stars);d.territories.runs[0][1]=17;assert.throws(()=>validateDraw(d,stars),/候选检查未通过/,'do not validate a stale cached grid');
    const portable=structuredClone(first);portable.regions[0].polygon[0][0]+=1e-14;assert.ok(equivalentDraw(first,portable));portable.regions[0].members[0].app_mag+=1e-12;assert.ok(!equivalentDraw(first,portable));
});
