import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {catalogueStars} from '../src/build_zodiac_candidates.mjs';
import {generateDraw as generateAny,validateDraw,redrawRecipe} from '../web/constellations/generator.mjs';
const generateDraw=(source,meta,recipe={})=>generateAny(source,meta,{algorithm:'terrax-zodiac-draw-5',...recipe});
import {ECLIPTIC_POLICY} from '../web/constellations/generator_ecliptic.mjs';
import {vector} from '../web/constellations/geometry.mjs';
import {regionAt} from '../web/constellations/territories.mjs';

const meta=JSON.parse(fs.readFileSync(new URL('../design/zodiac_candidates_v1.json',import.meta.url)));
const stars=catalogueStars(JSON.parse(fs.readFileSync(new URL(`../${meta.catalogue}`,import.meta.url))));

test('the reported seed has one small crossing after negotiation, measured on the actual sky',()=>{
    const before=generateDraw(stars,meta,{seed:'terrax-1ptws5s',algorithm:'terrax-zodiac-draw-4'});
    assert.equal(before.regions.filter(r=>r.eclipticSpan<14).length,5);
    for(const style of ['rich','balanced']){
        const data=generateDraw(stars,meta,{seed:'terrax-1ptws5s',style});
        assert.equal(data.recipe.algorithm,'terrax-zodiac-draw-5');
        // Independently accumulate ownership of 36,000 equal ecliptic samples.
        const measured=Array(15).fill(0);
        for(let k=0;k<36000;k++){const i=regionAt(data,vector((k+.5)/100,0));assert.ok(i<15);measured[i]+=.01;}
        for(let i=0;i<15;i++)assert.ok(Math.abs(measured[i]-data.regions[i].eclipticSpan)<.03);
        assert.ok(measured.filter(s=>s<15.97).length<=1);
        for(const i of [2,6,10])assert.ok(measured[i]>=16);
        assert.ok(Math.abs(data.regions.reduce((n,r)=>n+r.eclipticSpan,0)-360)<1e-9);
        const rerolled=generateDraw(stars,meta,redrawRecipe(data.recipe,[2,6,10],'width-reroll'));
        assert.deepEqual(rerolled.territories,data.territories);
        for(const i of [2,6,10])assert.deepEqual(rerolled.regions[i],data.regions[i]);
        assert.deepEqual(rerolled.regions.map(r=>r.intervals),data.regions.map(r=>r.intervals));
    }
});

test('new validation rejects a geometrically valid old layout with too many narrow crossings',()=>{
    const d=generateDraw(stars,meta,{seed:'terrax-1ptws5s',algorithm:'terrax-zodiac-draw-4'});
    d.recipe.algorithm='terrax-zodiac-draw-5';d.eclipticPolicy=ECLIPTIC_POLICY;
    assert.throws(()=>validateDraw(d,stars),/黄道窄区过多/);
    const next=generateDraw(stars,meta,{seed:'terrax-1ptws5s'});
    next.eclipticPolicy={...ECLIPTIC_POLICY,maximumNarrowRegions:4};
    assert.throws(()=>validateDraw(next,stars),/黄道宽度规则/);
});

test('browser stars with additional display fields yield the same portable result',()=>{
    const recipe={seed:'terrax-1ptws5s'},expected=generateDraw(stars,meta,recipe);
    const renderStars=stars.map(s=>({...s,baseDirection:s.direction,renderOnly:'ignored'}));
    assert.deepEqual(generateDraw(renderStars,meta,recipe),expected);
});
