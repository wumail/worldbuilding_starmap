import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {catalogueStars} from '../src/build_zodiac_candidates.mjs';
import {generateDraw,validateDraw,redrawRecipe} from '../web/constellations/generator.mjs';
import {ENVELOPE_POLICY} from '../web/constellations/generator_fitted.mjs';
import {shapeEnvelope,envelopeDistance,cellDirections,cellAreaDegrees} from '../web/constellations/shape_envelope.mjs';
import {vector,arc,separation,tangentFrame,projectLocal} from '../web/constellations/geometry.mjs';
import {unpackGrid,boundaryPoints,regionAt} from '../web/constellations/territories.mjs';

const meta=JSON.parse(fs.readFileSync(new URL('../design/zodiac_candidates_v1.json',import.meta.url)));
const stars=catalogueStars(JSON.parse(fs.readFileSync(new URL(`../${meta.catalogue}`,import.meta.url))));

test('spherical hull distance agrees with known angles and independent dense arc samples across the seam',()=>{
    const e=shapeEnvelope([[-5,-5],[5,-5],[5,5],[-5,5]].map(([l,b],i)=>({id:String(i),direction:vector(l,b)})));
    assert.equal(envelopeDistance(e,vector(359,0)),0);
    assert.ok(Math.abs(envelopeDistance(e,vector(8,0))-3)<1e-10);
    const samples=e.vertices.flatMap((a,i)=>arc(a,e.vertices[(i+1)%e.vertices.length],.04));
    for(const [l,b] of [[0,8],[30,20],[180,0],[358,-10]]){
        const v=vector(l,b),sampled=Math.min(...samples.map(q=>separation(v,q))),exact=envelopeDistance(e,v);
        assert.ok(sampled>=exact-1e-9&&sampled-exact<.021);
    }
});

test('120 fitted draws retain physical stars, width coverage and lateral figures within a bounded envelope',()=>{
    const results=[],original=JSON.stringify(stars);
    for(let k=0;k<60;k++){
        const seed=k===0?'terrax-1ptws5s':k===1?'terrax-b1bxb1':k===2?'terrax-001':`fit-${k-3}`;
        for(const style of ['rich','balanced']){
            const started=performance.now(),d=generateDraw(stars,meta,{seed,style,algorithm:'terrax-zodiac-draw-6'}),cells=unpackGrid(d.territories),envelopes=d.regions.map(r=>shapeEnvelope(r.members));
            assert.equal(d.recipe.algorithm,'terrax-zodiac-draw-6');
            let area=0,maxBoundaryDistance=0,horizontal=0;
            for(let n=0;n<cells.length;n++)if(cells[n]<15)area+=cellAreaDegrees(n);
            for(let i=0;i<15;i++){
                const r=d.regions[i],frame=tangentFrame(r.center.longitude,r.center.latitude),p=r.members.map(s=>projectLocal(s.direction,frame,0,0,1));
                const w=Math.max(...p.map(s=>s.x))-Math.min(...p.map(s=>s.x)),h=Math.max(...p.map(s=>s.y))-Math.min(...p.map(s=>s.y));
                if(w>=16&&w>h*1.2)horizontal++;
                for(const v of boundaryPoints(r.boundary,.25))maxBoundaryDistance=Math.max(maxBoundaryDistance,envelopeDistance(envelopes[i],v));
                for(const s of r.members){assert.equal(regionAt(d,s.direction),i);assert.deepEqual(s,stars.find(x=>x.id===s.id));}
                for(const [width,height] of [[1046,610],[294,460]]){
                    const ppd=Math.min((width-44)/72,(height-44)/56);
                    for(const s of r.members){const v=projectLocal(s.direction,frame,width/2,height/2,ppd);assert.ok(v.visible&&v.x>5&&v.x<width-5&&v.y>5&&v.y<height-5);}
                }
            }
            assert.ok(horizontal>=2,`${seed}/${style}: lateral figures disappeared`);
            assert.ok(maxBoundaryDistance<=8+1e-8,`${seed}: unsupported border ${maxBoundaryDistance}`);
            const spans=d.regions.map(r=>r.eclipticSpan);
            assert.ok(Math.min(...spans)>=6-1e-9&&Math.max(...spans)<=44+1e-9&&spans.filter(s=>s<16-1e-9).length<=1);
            assert.ok(Math.abs(spans.reduce((a,b)=>a+b,0)-360)<1e-9);
            for(let l=.013;l<360;l+=.113)assert.ok(regionAt(d,vector(l,0))<15);
            results.push({seed,style,milliseconds:performance.now()-started,areaSquareDegrees:area,maxBoundaryDistance,horizontal,spans,members:d.selectedExtendedCount});
        }
    }
    assert.equal(JSON.stringify(stars),original);
    fs.mkdirSync(new URL('../reports/zodiac_fitted_regions/',import.meta.url),{recursive:true});
    fs.writeFileSync(new URL('../reports/zodiac_fitted_regions/seed_validation.json',import.meta.url),JSON.stringify(results,null,2)+'\n');
});

test('fitted local rerolls preserve boundaries and locks without detaching from their figures',()=>{
    for(const style of ['rich','balanced']){
        const d=generateDraw(stars,meta,{seed:'terrax-1ptws5s',style,algorithm:'terrax-zodiac-draw-6'}),locks=[0,6,14];
        const next=generateDraw(stars,meta,redrawRecipe(d.recipe,locks,'fitted-reroll'));
        assert.deepEqual(next.territories,d.territories);
        for(const i of locks)assert.deepEqual(next.regions[i],d.regions[i]);
        assert.notDeepEqual(next.regions,d.regions);assert.ok(validateDraw(next,stars));
        const saved=JSON.stringify(d);d.regions[0].members[0].app_mag=-99;d.regions[0].boundary[0][0]+=1;
        assert.equal(JSON.stringify(generateDraw(stars,meta,d.recipe)),saved,'export mutation must not poison cached reference data');
    }
});

test('previous expanded regions cannot pass as fitted data and all five historical exports still reproduce',()=>{
    const d=generateDraw(stars,meta,{seed:'terrax-1ptws5s',algorithm:'terrax-zodiac-draw-5'});
    d.recipe.algorithm='terrax-zodiac-draw-6';d.envelopePolicy=ENVELOPE_POLICY;
    assert.throws(()=>validateDraw(d,stars),/超出星形包络/);
    for(const name of ['zodiac_draw_example.json','zodiac_draw_ordered_example.json','zodiac_draw_free_example.json','zodiac_draw_clean_example.json','zodiac_draw_ecliptic_example.json']){
        const saved=JSON.parse(fs.readFileSync(new URL(`../design/${name}`,import.meta.url))).data;
        assert.equal(JSON.stringify(generateDraw(stars,meta,saved.recipe)),JSON.stringify(saved));
    }
});
