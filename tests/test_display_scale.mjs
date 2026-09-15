import test from 'node:test';
import assert from 'node:assert/strict';
import * as V1 from '../web/v1/sky_state.mjs';
import * as V2 from '../web/v2/sky_state.mjs';
import {DEFAULT_DISPLAY_SCALE,MAX_DISPLAY_SCALE} from '../web/shared/sky_render.mjs';
import {observerSky} from '../web/v2/eye_model.mjs';

const savedState={days:1234.5,anchor:1000,playing:true,speed:2,mode:'surface',latitude:-32,longitude:81,spinPhase:65,selected:'Echo',daylight:false,eye:{opticalBlur:false},angles:{Luna:{node:36,peri:74,mean:18}}};
function clock(module,record){
    let value=JSON.stringify(record);return new module.SkyClock({id:'size-upgrade',now:()=>3000,eventTarget:{addEventListener(){}},storage:{getItem:()=>value,setItem:(_key,next)=>value=next}});
}
test('old automatic display presets upgrade without resetting time or observer preferences',()=>{
    for(const module of [V1,V2]){
        const state=module.sanitizeState(savedState,1000);
        if(module===V1)state.displayScale=1.6;else delete state.displayScale;
        const c=clock(module,{state,versions:{time:[1000,'old'],angles:[1001,'old']}});
        assert.deepEqual(c.state,{...module.sanitizeState(state,3000),displayScale:DEFAULT_DISPLAY_SCALE});
        assert.equal(c.time(),1238.5);
        c.set({markers:false});assert.equal(c.read().state.displayScale,DEFAULT_DISPLAY_SCALE);
        assert.deepEqual(c.record.versions.time,[1000,'old']);
        assert.deepEqual(c.record.versions.angles,[1001,'old']);
    }
});
test('explicit size preferences persist, including the old 1.6 setting and natural 1x',()=>{
    for(const module of [V1,V2])for(const gain of [1,1.6,3,6,8]){
        const c=clock(module,{state:{...savedState,displayScale:gain},versions:{displayScale:[1002,'user']}});
        assert.equal(c.state.displayScale,gain);c.set({displayScale:gain});assert.equal(c.read().state.displayScale,gain);
        assert.equal(module.sanitizeState({displayScale:100}).displayScale,MAX_DISPLAY_SCALE);
    }
});
test('V2 display size changes neither visibility and atmosphere nor physical body data',()=>{
    const baseline=observerSky(0,V2.sanitizeState({...savedState,displayScale:1}));
    for(const gain of [DEFAULT_DISPLAY_SCALE,MAX_DISPLAY_SCALE])assert.deepEqual(observerSky(0,V2.sanitizeState({...savedState,displayScale:gain})),baseline);
});
