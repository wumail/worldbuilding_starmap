import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {prepareDeepSources,profileSolidAngle,erf,SOURCE_GRID} from '../web/shared/deep_sky_profiles.mjs';

test('continuous profiles match independent small-angle integrated mass and emission formulas',()=>{
    const t=1e-5,c=2.3;
    const expected={plummer:Math.PI*t*t*1000/101**1.5,gaussian:2*Math.PI*t*t*(1-Math.exp(-8)),ionized_gaussian:Math.PI*t*t*(erf(c)-2*c/Math.sqrt(Math.PI)*Math.exp(-c*c))};
    for(const [kind,wanted] of Object.entries(expected)){
        const actual=profileSolidAngle(kind,t,kind==='plummer'?10:kind==='gaussian'?4:c);
        assert.ok(Math.abs(actual/wanted-1)<2e-5,`${kind}: ${actual/wanted}`);
    }
});
test('deep objects retain physical flux while rejecting sources below the necessary point-flux ceiling',()=>{
    const objects=JSON.parse(readFileSync(new URL('../output/output_20260915_galactic_01/sky_view_20260915_galactic_01.json',import.meta.url))).deep_sky.objects;
    const before=JSON.stringify(objects),p=prepareDeepSources(objects);
    assert.equal(p.sources.length,160);assert.equal(p.sources.length,objects.filter(o=>o.v_flux>0&&o.app_mag<=6.5).length);
    assert.equal(JSON.stringify(objects),before);
    for(const s of p.sources)assert.ok(Math.abs(s.peak*s.solidAngle/s.flux-1)<1e-12);
    assert.ok(p.maxOverlap<=64);
});
test('spatial source index includes source centers through longitude wrap and polar caps',()=>{
    const objects=[0,.001,359.999,180].flatMap(l=>[-89.99,-30,0,30,89.99].map(b=>({id:`${l}/${b}`,v_flux:.01,app_mag:5,angular_scale_rad:.002,profile:'plummer',gal_lon:l,gal_lat:b})));
    const p=prepareDeepSources(objects),[w,h]=SOURCE_GRID;
    for(const [i,s] of p.sources.entries()){
        const x=Math.floor((s.longitude/(2*Math.PI)%1+1)%1*w),y=Math.min(h-1,Math.floor((s.latitude/Math.PI+.5)*h)),k=4*(y*w+x),offset=p.offsets[k],count=p.offsets[k+1];
        assert.ok(Array.from({length:count},(_,j)=>p.indexData[4*(offset+j)]).includes(i),s.id);
    }
});
