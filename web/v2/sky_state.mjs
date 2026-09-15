import {sanitizeEye,EYE_DEFAULTS} from './eye_model.mjs';
import {DEFAULT_DISPLAY_SCALE,MAX_DISPLAY_SCALE} from '../shared/sky_render.mjs';
import {MAX_PERIOD, LOCAL_DAY, INITIAL_ANGLES, SYSTEM, clamp, mod} from '../shared/solar_system.mjs';

export const STORAGE_KEY = 'terrax-naked-eye-v2';
export const DEFAULT_STATE = {
    days:0, anchor:0, playing:false, speed:LOCAL_DAY/20,
    mode:'surface', projection:'perspective', latitude:30, longitude:0, spinPhase:0,
    daylight:true, eye:EYE_DEFAULTS, markers:false, grid:false, trail:'off', selected:'Luna',
    zodiac:true, deepSky:true, folder:'', magnification:1, displayScale:DEFAULT_DISPLAY_SCALE, angles:INITIAL_ANGLES,
};
const finite = (v,fallback) => typeof v==='number' && Number.isFinite(v) ? v : fallback;

export function sanitizeState(value = {}, now = Date.now()) {
    if (!value || typeof value!=='object') value={};
    const angles={};
    for (const body of SYSTEM.bodies) {
        const a=value.angles?.[body.id] || {}, initial=INITIAL_ANGLES[body.id];
        angles[body.id]={node:mod(finite(a.node,initial.node),360),peri:mod(finite(a.peri,initial.peri),360),mean:mod(finite(a.mean,initial.mean),360)};
        if (body.canonicalPeri) { angles[body.id].node=0; angles[body.id].peri=283; }
    }
    return {
        days:clamp(finite(value.days,0),0,MAX_PERIOD), anchor:finite(value.anchor,now),
        playing:value.playing===true, speed:clamp(finite(value.speed,DEFAULT_STATE.speed),1e-6,3652.5),
        mode:['center','surface'].includes(value.mode)?value.mode:DEFAULT_STATE.mode,
        projection:['perspective','stereographic'].includes(value.projection)?value.projection:DEFAULT_STATE.projection,
        eye:sanitizeEye(value.eye),
        magnification:clamp(finite(value.magnification,1),.2,40),
        displayScale:clamp(finite(value.displayScale,DEFAULT_DISPLAY_SCALE),1,MAX_DISPLAY_SCALE),
        latitude:clamp(finite(value.latitude,30),-90,90), longitude:clamp(finite(value.longitude,0),-180,180),
        spinPhase:mod(finite(value.spinPhase,0),360),
        zodiac:value.zodiac!==false,deepSky:value.deepSky!==false,daylight:value.daylight!==false,markers:value.markers===true,grid:value.grid===true,
        trail:['day','year'].includes(value.trail)?value.trail:'off',
        selected:SYSTEM.bodies.some(b=>b.id===value.selected && b.id!=='Terrax') || value.selected==='Sol'?value.selected:'Luna',
        folder:typeof value.folder==='string' && /^output_[A-Za-z0-9_]+$/.test(value.folder)?value.folder:'', angles,
    };
}

export function timeAt(state, now = Date.now()) {
    return clamp(state.days+(state.playing?Math.max(0,now-state.anchor)/1000*state.speed:0),0,MAX_PERIOD);
}

// 两页只共享一个时间锚点；每一帧由锚点计算，不分别累加或互相写回时间。
export function changeState(state, patch, now = Date.now()) {
    const next=sanitizeState({...state, days:timeAt(state,now), ...patch, anchor:now},now);
    if (next.days>=MAX_PERIOD) next.playing=false;
    return next;
}

// 时间是不可拆分的一组字段；其它设置各自合并，避免两页同时操作丢失修改。
const GROUPS={time:['days','anchor','playing','speed'],...Object.fromEntries(Object.keys(DEFAULT_STATE).filter(k=>!['days','anchor','playing','speed'].includes(k)).map(k=>[k,[k]]))};
function recordOf(value,now) {
    const state=sanitizeState(value?.state || value,now),versions={};
    for(const key of Object.keys(GROUPS)) {
        const v=value?.versions?.[key];versions[key]=Array.isArray(v) && Number.isFinite(v[0]) && typeof v[1]==='string'?[v[0],v[1]]:[0,''];
    }
    return {state,versions};
}
const newer=(a,b)=>a[0]>b[0] || (a[0]===b[0] && a[1]>b[1]);
function mergeRecords(a,b) {
    const state={},versions={};
    for(const [key,fields] of Object.entries(GROUPS)) {
        const chosen=newer(b.versions[key],a.versions[key])?b:a;versions[key]=chosen.versions[key];
        for(const field of fields)state[field]=chosen.state[field];
    }
    return {state:sanitizeState(state),versions};
}

export class SkyClock {
    constructor({storage,eventTarget=globalThis.window,now=()=>Date.now(),id=globalThis.crypto.randomUUID()}={}) {
        try{this.storage=storage??globalThis.localStorage;}
        catch{this.storage={getItem:()=>null,setItem:()=>{throw Error('Storage unavailable');}};}
        this.now=now;this.id=id;this.record=this.read();this.state=this.record.state;
        this.listeners=new Set();
        eventTarget.addEventListener('storage',event=>{
            if (event.key!==STORAGE_KEY || !event.newValue) return;
            try {
                const stored=this.read(),merged=mergeRecords(mergeRecords(this.record,recordOf(JSON.parse(event.newValue),this.now())),stored);
                const changed=JSON.stringify(merged)!==JSON.stringify(this.record);
                this.record=merged;this.state=merged.state;
                // 修复同时写入造成的字段丢失；旧事件永远不能让状态倒退。
                if(JSON.stringify(merged)!==JSON.stringify(stored))this.persist();
                if(changed)this.emit();
            } catch { /* 忽略损坏的数据 */ }
        });
    }
    read() {
        try {
            const saved=this.storage.getItem(STORAGE_KEY);
            if(saved!==null)return recordOf(JSON.parse(saved),this.now());
            const record=recordOf({},this.now());
            this.persist(record);
            return record;
        }
        catch {return recordOf({},this.now());}
    }
    persist(record=this.record) {
        try {this.storage.setItem(STORAGE_KEY,JSON.stringify(record));}
        catch {if(globalThis.document)document.documentElement.dataset.storageUnavailable='true';}
    }
    subscribe(listener) { this.listeners.add(listener); return ()=>this.listeners.delete(listener); }
    emit() { for (const listener of this.listeners) listener(this.state); }
    set(patch) {
        this.record=mergeRecords(this.record,this.read());
        const now=this.now(),timeChange=GROUPS.time.some(key=>key in patch);
        const next=timeChange?changeState(this.record.state,patch,now):sanitizeState({...this.record.state,...patch},now);
        const revision=Math.max(now,...Object.values(this.record.versions).map(v=>v[0]+1));
        for(const [key,fields] of Object.entries(GROUPS))if(fields.some(field=>field in patch))this.record.versions[key]=[revision,this.id];
        this.record.state=this.state=next;this.persist();
        this.emit();
    }
    stopAtEnd() {
        const latest=mergeRecords(this.record,this.read());
        if(timeAt(latest.state,this.now())>=MAX_PERIOD)this.set({days:MAX_PERIOD,playing:false});
        else {this.record=latest;this.state=latest.state;this.emit();}
    }
    time(now=this.now()) { return timeAt(this.state,now); }
}
