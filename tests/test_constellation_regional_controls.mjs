import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {mountCandidateEditor} from '../web/constellations/candidate_editor.mjs';
import {sampleRegion} from '../web/constellations/regional_figures.mjs';
import {catalogueStars} from '../src/build_zodiac_candidates.mjs';

const read=p=>JSON.parse(fs.readFileSync(new URL('../'+p,import.meta.url)));
const base=read('design/zodiac_draw_sampling_example.json').data,stars=catalogueStars(read(base.catalogue));
const html=fs.readFileSync(new URL('../web/constellations/index.html',import.meta.url),'utf8');
// A DOM adapter for controller tests, not a browser or layout verification.
function controls(t){
    const make=()=>({value:'',checked:false,disabled:false,hidden:false,textContent:'',
        classList:{add(){},remove(){},toggle(){}},setAttribute(){},addEventListener(){}});
    const elements=new Map([...html.matchAll(/id="([^"]+)"/g)].map(m=>[m[1],make()]));
    const select=html.match(/<select id="regional-complexity"[^>]*>([\s\S]*?)<\/select>/)[1];
    elements.get('regional-complexity').value=select.match(/<option value="([^"]+)" selected>/)[1];
    const previous=globalThis.document;
    globalThis.document={getElementById:id=>{assert.ok(elements.has(id),`missing actual HTML control ${id}`);return elements.get(id);}};
    t.after(()=>{if(previous===undefined)delete globalThis.document;else globalThis.document=previous;});
    return id=>elements.get(id);
}

test('local complexity reaches the sampler, stays locked during work and participates in undo, cancel and confirm',async t=>{
    const $=controls(t),seen=[],saved=[];let current=base,fail=false;
    const editor=mountCandidateEditor({stars,getLayout:()=>null,getView:()=>({center:null,zoom:1}),redraw(){},
        onPreview:data=>{current=data;},onClose:applied=>{if(!applied)current=base;},onSave:async recipe=>{saved.push(recipe);},
        onResample:async(data,index,style)=>{
            assert.equal($('regional-complexity').disabled,true);assert.equal($('regenerate-candidate').disabled,true);
            seen.push(style);if(fail)throw Error('controlled failure');return sampleRegion(data,stars,index,'local-controls',style);
        }});
    assert.equal($('regional-complexity').value,'balanced');assert.equal(base.recipe.style,'rich');
    editor.start(base,base,10,'lines');$('regional-complexity').value='simple';await editor.regenerate();
    assert.equal(seen.at(-1),'simple');assert.match($('candidate-message').textContent,/简洁/);
    assert.equal($('regional-complexity').disabled,false);assert.ok(current.regions[10].members.length<=7);
    const generated=JSON.stringify(current);$('candidate-undo').onclick();assert.deepEqual(current,base);
    $('candidate-redo').onclick();assert.equal(JSON.stringify(current),generated);
    const position=editor.status.historyPosition;fail=true;await editor.regenerate();
    assert.equal(editor.status.historyPosition,position);assert.equal(JSON.stringify(current),generated);
    assert.equal($('regional-complexity').disabled,false);assert.match($('candidate-message').textContent,/controlled failure/);
    $('candidate-cancel').onclick();assert.deepEqual(current,base);assert.equal(editor.active,false);
    fail=false;editor.start(base,base,10,'lines');$('regional-complexity').value='rich';await editor.regenerate();
    assert.equal(seen.at(-1),'rich');await $('candidate-confirm').onclick();
    assert.equal(saved.length,1);assert.equal(saved[0].algorithm,'terrax-zodiac-manual-3');
    assert.equal(saved[0].style,base.recipe.style);assert.ok(saved[0].figures.some(f=>f.index===10));assert.equal(editor.active,false);
});

test('removed lines are hidden at session start, can be inspected and never change the draft',async t=>{
    const $=controls(t);let current=base;
    const editor=mountCandidateEditor({stars,getLayout:()=>null,getView:()=>({center:null,zoom:1}),redraw(){},onClose(){},onSave(){},
        onPreview:data=>{current=data;},onResample:async(data,index,style)=>sampleRegion(data,stars,index,'removed-lines',style)});
    $('candidate-show-removed').checked=true;editor.start(base,base,10,'lines');
    assert.equal($('candidate-show-removed').checked,false);await editor.regenerate();
    const generated=JSON.stringify(current),dashes=[];
    const context={save(){},restore(){},beginPath(){},moveTo(){},lineTo(){},stroke(){},arc(){},setLineDash(value){dashes.push(value.join(','));}};
    const layout={project:p=>({x:p[0]*100,y:p[1]*100,visible:true}),hits:[]};
    editor.paint(context,layout);assert.ok(!dashes.includes('3,5'));
    $('candidate-show-removed').checked=true;$('candidate-show-removed').onchange();editor.paint(context,layout);
    assert.ok(dashes.includes('3,5'));assert.equal(JSON.stringify(current),generated);
    $('candidate-cancel').onclick();editor.start(base,base,10,'lines');assert.equal($('candidate-show-removed').checked,false);
});
