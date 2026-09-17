import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {mountWorkflow,STORAGE_KEY} from '../web/constellations/workflow.mjs';
import {ALGORITHM,DEFAULT_STYLE,normalizeRecipe} from '../web/constellations/generator.mjs';

const read=p=>JSON.parse(fs.readFileSync(new URL('../'+p,import.meta.url)));
const baseline=read('design/zodiac_draw_sampling_example.json').data;
const html=fs.readFileSync(new URL('../web/constellations/index.html',import.meta.url),'utf8');
// Exercise actual workflow routing and persistence using a worker/DOM adapter.
// Geometry and browser layout are tested separately.
function setup(t,{saved=null,url='http://localhost/'}={}){
    const make=()=>({value:'',checked:false,disabled:false,hidden:false,textContent:'',children:[],
        classList:{add(){},remove(){},toggle(){}},setAttribute(){},addEventListener(){},replaceChildren(...children){this.children=children;}});
    const elements=new Map([...html.matchAll(/id="([^"]+)"/g)].map(m=>[m[1],make()]));
    for(const id of ['complexity','regional-complexity']){
        const select=html.match(new RegExp(`<select id="${id}"[^>]*>([\\s\\S]*?)<\\/select>`))[1];
        elements.get(id).value=select.match(/<option value="([^"]+)" selected>/)[1];
    }
    elements.get('sampling-width').value='40';
    const storage=new Map(saved?[[STORAGE_KEY,JSON.stringify(saved)]]:[]),requests=[];
    const replacements={document:{body:{dataset:{}},getElementById:id=>{assert.ok(elements.has(id),id);return elements.get(id);},createElement:make},
        location:{href:url},history:{replaceState(_a,_b,next){replacements.location.href=String(next);}},
        localStorage:{getItem:k=>storage.get(k)??null,setItem:(k,v)=>storage.set(k,v)},
        Worker:class{postMessage(message){if(message.type==='init')return;requests.push(message.recipe);
            queueMicrotask(()=>this.onmessage({data:{id:message.id,result:{...structuredClone(baseline),recipe:message.recipe}}}));}}};
    for(const [key,value] of Object.entries(replacements)){
        const original=Object.getOwnPropertyDescriptor(globalThis,key);
        Object.defineProperty(globalThis,key,{value,configurable:true,writable:true});
        t.after(()=>{if(original)Object.defineProperty(globalThis,key,original);else delete globalThis[key];});
    }
    const workflow=mountWorkflow({stars:[],baseline,onChange(){},getLayout:()=>null,getView:()=>({center:null,zoom:1}),redraw(){},onEditing(){}});
    return {$:id=>elements.get(id),workflow,requests,storage};
}

test('first visit, explicit seed and next-round buttons use current generation with standard defaults',async t=>{
    const {$,workflow,requests}=setup(t);await workflow.start();
    assert.equal(requests[0].algorithm,ALGORITHM);assert.equal(requests[0].style,DEFAULT_STYLE);
    assert.equal($('complexity').value,'balanced');assert.equal($('regional-complexity').value,'balanced');
    $('seed').value='entered-seed';await $('reproduce').onclick();
    assert.equal(requests.at(-1).seed,'entered-seed');assert.equal(requests.at(-1).style,'balanced');
    $('complexity').value='rich';await $('new-round').onclick();
    assert.equal(requests.at(-1).algorithm,ALGORITHM);assert.equal(requests.at(-1).style,'rich');
});

test('URL seed uses standard while legacy favourites replay their own recipe without changing next-round settings',async t=>{
    const record={id:'legacy',title:'旧收藏',recipe:baseline.recipe,locks:[0],notes:{Z01:'保留'},favourite:true};
    const saved={schema:2,algorithm:'terrax-zodiac-draw-9',sha256:baseline.sha256,records:[record],current:'legacy'};
    const {$,workflow,requests,storage}=setup(t,{saved,url:'http://localhost/?seed=url-seed&sampling=47'});await workflow.start();
    assert.equal(requests[0].algorithm,ALGORITHM);assert.equal(requests[0].style,'balanced');assert.equal(requests[0].samplingHalfWidthDegrees,47);
    assert.equal(new URL(location.href).searchParams.has('seed'),false);
    await $('rounds').onchange({target:{value:'legacy'}});
    assert.deepEqual(requests.at(-1),normalizeRecipe(baseline.recipe));assert.equal(workflow.record.notes.Z01,'保留');
    assert.equal($('complexity').value,'balanced');
    await $('reroll').onclick();assert.equal(requests.at(-1).algorithm,'terrax-zodiac-draw-9');assert.equal(requests.at(-1).style,'rich');
    assert.equal(requests.at(-1).shapeSeeds[0],baseline.recipe.shapeSeeds[0]);
    await $('reproduce').onclick();assert.equal(requests.at(-1).algorithm,ALGORITHM);assert.equal(requests.at(-1).style,'balanced');
    const retained=JSON.parse(storage.get(STORAGE_KEY));assert.equal(retained.algorithm,ALGORITHM);
    assert.deepEqual(retained.records,[record]);assert.equal(retained.current,'legacy');
});
