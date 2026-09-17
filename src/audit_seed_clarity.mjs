// Compare old/new displayed figures inside the same seeded automatic territories.
import fs from 'node:fs';
import path from 'node:path';
import assert from 'node:assert/strict';
import {catalogueStars} from './build_zodiac_candidates.mjs';
import {prepareDraw,generateDraw,ALGORITHM} from '../web/constellations/generator.mjs';
import {unpackGrid,cellOf} from '../web/constellations/territories.mjs';
import {manualArcCells} from '../web/constellations/manual_figures.mjs';
import {graphMeasures} from './audit_regional_figures.mjs';
import {angleMeasures,summarizeAngles} from './audit_regional_clarity.mjs';

const args=process.argv.slice(2),out=args[args.indexOf('--out')+1];
if(!args.includes('--out')||!out)throw Error('Usage: node src/audit_seed_clarity.mjs --out NEW.json');
if(fs.existsSync(out))throw Error('Refusing to overwrite an existing audit');
const root=new URL('../',import.meta.url),read=p=>JSON.parse(fs.readFileSync(new URL(p,root)));
const meta=read('design/zodiac_candidates_v1.json'),stars=catalogueStars(read(meta.catalogue)),lookup=new Map(stars.map(s=>[s.id,s]));
const beforeSource=JSON.stringify(stars),runs=[],rows=[];
await prepareDraw();
for(const seed of ['terrax-001','terrax-1ptws5s','terrax-002']){
    let boundaries;
    for(const algorithm of ['terrax-zodiac-draw-9',ALGORITHM])for(const style of algorithm===ALGORITHM?['simple','balanced','rich']:['balanced','rich']){
        const start=performance.now();let data;
        try{data=generateDraw(stars,meta,{algorithm,seed,style});}
        catch(error){runs.push({algorithm,seed,style,status:'rejected',error:error.message,milliseconds:performance.now()-start});console.log('REJECTED',algorithm,seed,style,error.message);continue;}
        const cells=unpackGrid(data.territories);
        if(boundaries)assert.deepEqual(data.territories,boundaries);else boundaries=data.territories;
        for(const [index,r] of data.regions.entries()){
            const group=stars.filter(s=>cells[cellOf(s.direction)]===index).sort((a,b)=>a.app_mag-b.app_mag||a.id.localeCompare(b.id));
            const required=group.filter((s,i)=>i<3||s.app_mag<=3).map(s=>s.id);
            const variants=r.variants.map(v=>{
                assert.ok(required.every(id=>v.members.includes(id)));
                for(const e of v.edges)assert.ok(manualArcCells(lookup.get(e.from).direction,lookup.get(e.to).direction).every(k=>cells[k]===index));
                const pairs=v.edges.map(e=>[e.from,e.to]),graph=graphMeasures(v.members,pairs);assert.equal(graph.components,1);
                return {...graph,...angleMeasures(v.members,pairs,lookup),members:v.members.length};
            });
            rows.push({algorithm,seed,style,index,full:variants[1],core:variants[0]});
        }
        const name=`${algorithm}-${seed}-${style}.json`,output=path.join(path.dirname(out),name);
        fs.mkdirSync(path.dirname(output),{recursive:true});fs.writeFileSync(output,JSON.stringify(data,null,2)+'\n',{flag:'wx'});
        runs.push({algorithm,seed,style,status:'validated',milliseconds:performance.now()-start,members:data.selectedExtendedCount,file:output});
        console.log('VALIDATED',algorithm,seed,style,data.selectedExtendedCount);
    }
}
assert.equal(JSON.stringify(stars),beforeSource);
const summaries=Object.fromEntries(['terrax-zodiac-draw-9',ALGORITHM].map(algorithm=>[algorithm,summarizeAngles(rows.filter(r=>r.algorithm===algorithm))]));
fs.mkdirSync(path.dirname(out),{recursive:true});fs.writeFileSync(out,JSON.stringify({runs,summaries,rows,checks:{sourceUnchanged:true,boundariesIdenticalAcrossMethodsAndStyles:true,brightStarsPreserved:true,connected:true,completeArcsContained:true}},null,2)+'\n',{flag:'wx'});
