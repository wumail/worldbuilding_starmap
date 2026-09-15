import test from 'node:test';
import assert from 'node:assert/strict';
import {existsSync, readFileSync} from 'node:fs';
import {EYE_DEFAULTS,screenCalibration,fovForFocal,magnitudeToLux,luxToMagnitude,diskPeakLuminance,diskSolidAngle,airMass,observedMagnitude,gaussianKernel,observerSky,pointHidden,toneLuminance,nakedEyeLimit,magnitudeToLuminance} from '../eye_model.mjs';
import {sanitizeState,timeAt,changeState,STORAGE_KEY,SkyClock} from '../sky_state.mjs';
import {DEG,MAX_PERIOD,skyState,lambertPhase} from '../../shared/solar_system.mjs';
import {projectionFocal} from '../../shared/sky_projection.mjs';
const near=(a,b,tol=1e-9)=>assert.ok(Math.abs(a-b)<=tol,`${a} != ${b} (±${tol})`);
test('27 inch 16:9 screen at 60 cm agrees with independent viewing geometry',()=>{
    const e=EYE_DEFAULTS,c=screenCalibration(e,1920,1080),physicalHeight=27*2.54*9/Math.sqrt(16**2+9**2);
    near(c.focal,60*1080/physicalHeight);
    const lunarPixels=2*c.focal*Math.tan(.5*DEG/2);
    near(2*Math.atan(lunarPixels/(2*c.pixelsPerCm)/60)/DEG,.5);
    for(const mode of ['perspective','stereographic'])for(const h of [256,600,1080])near(projectionFocal(h,fovForFocal(h,c.focal,mode),mode),c.focal);
});
test('ruler calibration overrides diagonal estimate; all objects share zoom',()=>{
    const e={...EYE_DEFAULTS,pixelsPerCm:32},f=screenCalibration(e).focal;near(f,1920);
    for(const diameter of [.5,.114,.01])near((2*f*Math.tan(diameter*DEG/2)*4)/(2*f*Math.tan(diameter*DEG/2)),4);
});
test('magnitude uses integrated flux and correct five-magnitude ratio',()=>{
    near(magnitudeToLux(0),2.54e-6,1e-15);near(magnitudeToLux(0)/magnitudeToLux(5),100);
    for(const m of [-26.5,-12.7,-9.8,0,6.5])near(luxToMagnitude(magnitudeToLux(m)),m);
    assert.ok(magnitudeToLux(-12.7)>.30 && magnitudeToLux(-12.7)<.31);
});
test('independent projected-disk integration recovers Lambert total flux and phase',()=>{
    const n=1600,area=diskSolidAngle(.53);let worst=0;
    for(const angle of [0,Math.PI/3,Math.PI/2,5*Math.PI/6]){
        const body={id:'Luna',angularDiameter:.53,phaseAngle:angle,magnitude:-12.7-2.5*Math.log10(lambertPhase(angle))};
        let sum=0;for(let y=0;y<n;y++)for(let x=0;x<n;x++){const u=2*(x+.5)/n-1,v=2*(y+.5)/n-1,r=u*u+v*v;if(r<1)sum+=Math.max(0,u*Math.sin(angle)+Math.sqrt(1-r)*Math.cos(angle));}
        const flux=diskPeakLuminance(body)*sum*4/(n*n)*area/Math.PI;
        worst=Math.max(worst,Math.abs(flux/magnitudeToLux(body.magnitude)-1));
    }assert.ok(worst<.001,`relative integration error ${worst}`);
});
test('airmass and brightness threshold respond to altitude, moonlight and daylight',()=>{
    near(airMass(90),.9997119919,1e-9);assert.ok(airMass(30)>1.99 && airMass(30)<2);
    assert.ok(airMass(0)>37 && airMass(0)<39);
    near(observedMagnitude(2,30,EYE_DEFAULTS)-2,.2*airMass(30));
    assert.ok(nakedEyeLimit(magnitudeToLuminance(21.7))>6.4);
    const night=observerSky(0,sanitizeState()),day=observerSky(.54,sanitizeState());
    assert.ok(night.environment.limit<6);assert.ok(day.environment.limit<night.environment.limit);
    assert.equal(observerSky(0,sanitizeState({mode:'center'})).environment.limit,6.5);
});
test('V2 preserves orbital outputs, true angular diameters and original time span',()=>{
    near(MAX_PERIOD,51313.9725);
    for(const days of [0,1234.5,MAX_PERIOD]){
        const s=sanitizeState(),original=skyState(days,s),eye=observerSky(days,s);
        for(let i=0;i<eye.bodies.length;i++)for(const field of ['view','angularDiameter','magnitude','phaseAngle','distance'])assert.deepEqual(eye.bodies[i][field],original.bodies[i][field]);
    }
    const bodies=observerSky(0,sanitizeState()).bodies;
    near(bodies.find(b=>b.id==='Luna').angularDiameter,.531544,1e-6);
    near(bodies.find(b=>b.id==='Echo').angularDiameter,.114561,1e-6);
});
test('opaque new moon blocks stars even at sub-arcminute diameters',()=>{
    const disk={id:'Echo',distance:.01,view:[1,0,0],angularDiameter:.000001};
    assert.ok(pointHidden([1,0,0],[disk]));assert.ok(!pointHidden([1,0,0],[disk],.001));
    const a=.001*DEG;assert.ok(!pointHidden([Math.cos(a),Math.sin(a),0],[disk]));
});
test('PSF conserves energy, rejects sparse sampling; tone preserves monotonicity',()=>{
    for(const s of [0,.2,.8,3,8]){const k=gaussianKernel(s);near(k.weights[0]+2*k.weights.slice(1).reduce((a,b)=>a+b,0),1);assert.equal(k.step,1);}
    assert.throws(()=>gaussianKernel(50));
    assert.ok(toneLuminance(.005,.005)<.003,'night should not become middle gray');
    const samples=[0,.0001,.001,.1,1,100,1e9].map(l=>toneLuminance(l,.005));
    assert.ok(samples.every((v,i)=>i===0 || v>samples[i-1]));
});
test('V2 state is independent, blur can be disabled, clock stops at endpoint',()=>{
    assert.equal(STORAGE_KEY,'terrax-naked-eye-v2');const s=sanitizeState({eye:{opticalBlur:false},magnification:4});
    assert.equal(s.eye.opticalBlur,false);assert.equal(s.magnification,4);assert.equal(s.mode,'surface');
    const play=changeState(s,{days:MAX_PERIOD-1,playing:true,speed:2},1000);near(timeAt(play,2000),MAX_PERIOD);
    assert.equal(changeState(play,{},2000).playing,false);
});
test('historical revision records stay consistent; versioned sources still exist',()=>{
    // Historical SHA keys still use v4/ paths; they map onto src/, web/, data/, output/.
    const record=JSON.parse(readFileSync(new URL('../data/v1_preservation.json',import.meta.url)));
    const revision=JSON.parse(readFileSync(new URL('../data/v1_display_revision.json',import.meta.url)));
    const galactic=JSON.parse(readFileSync(new URL('../data/v1_galactic_revision.json',import.meta.url)));
    const allowed=new Set(['v4/sky_atlas.html','v4/star_map.html','v4/sky_controls.mjs','v4/sky_render.mjs','v4/sky_atlas.mjs','v4/sky_state.mjs','v4/sky_points.js','v4/sky_motion_3d.js','v4/sky_atlas_bodies.mjs']);
    for(const [path,change] of Object.entries(revision.changes)){assert.ok(allowed.has(path),path);assert.equal(change.before,record.sha256[path],path);}
    const galacticAllowed=new Set(['v4/sky_atlas.html','v4/star_map.html','v4/sky_controls.mjs','v4/sky_render.mjs','v4/sky_atlas.mjs','v4/sky_state.mjs','v4/star_generator.py','v4/stellar_physics.py','v4/folders.json']);
    for(const [path,change] of Object.entries(galactic.changes)){assert.ok(galacticAllowed.has(path),path);assert.equal(change.before,revision.changes[path]?.after ?? record.sha256[path],path);}
    const relocate=path=>{
        if(path==='v4/.DS_Store')return null;
        if(path.startsWith('v4/v2/'))return path==='v4/v2/solar_system.mjs'?'web/shared/solar_system.mjs':'web/v2/'+path.slice(6);
        if(path.startsWith('v4/output_'))return 'output/'+path.slice(3);
        if(path==='v4/folders.json')return 'output/folders.json';
        if(path.startsWith('v4/data/'))return path.slice(3);
        if(/^v4\/[^/]+\.py$/.test(path))return 'src/'+path.slice(3);
        if(path.startsWith('v4/design/'))return path.slice(3);
        if(path.startsWith('v4/')){
            const name=path.slice(3),shared=new Set(['catalog_data.mjs','deep_sky_view.js','zodiac.mjs','sky_projection.js','sky_projection.mjs','sky_render.mjs','sky_point_kernel.mjs','sky_motion.css','solar_system.mjs']);
            return 'web/'+(shared.has(name)?'shared/':'v1/')+name;
        }
        return path;
    };
    let found=0;
    for(const path of Object.keys(record.sha256)){
        const relocated=relocate(path);if(!relocated)continue;
        if(existsSync(new URL('../../../'+relocated,import.meta.url)))found++;
    }
    assert.ok(found>=20,`relocated sources ${found}`);
});
test('blocked storage getter falls back to a working in-memory clock',()=>{
    const descriptor=Object.getOwnPropertyDescriptor(globalThis,'localStorage');
    try{
        Object.defineProperty(globalThis,'localStorage',{configurable:true,get(){throw Error('blocked getter');}});
        const clock=new SkyClock({eventTarget:{addEventListener(){}},now:()=>1000,id:'isolated-storage-test'});
        clock.set({days:12.5});near(clock.time(),12.5);
    }finally{if(descriptor)Object.defineProperty(globalThis,'localStorage',descriptor);else delete globalThis.localStorage;}
});
