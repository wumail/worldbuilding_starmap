// Compare regional samplers on an identical, committed sky and boundary fixture.
// Run once before a change and once after it, with different output filenames.
import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {createHash} from 'node:crypto';
import {execFileSync} from 'node:child_process';
import assert from 'node:assert/strict';
import {catalogueStars} from './build_zodiac_candidates.mjs';
import {sampleRegion} from '../web/constellations/regional_figures.mjs';
import {cellOf,unpackGrid} from '../web/constellations/territories.mjs';
import {manualArcCells} from '../web/constellations/manual_figures.mjs';
import {separation} from '../web/constellations/geometry.mjs';

export function graphMeasures(members,edges){
    const adj=new Map(members.map(id=>[id,new Set()]));
    for(const [a,b] of edges){adj.get(a).add(b);adj.get(b).add(a);}
    const seen=new Set();let components=0,diameter=0;
    for(const id of members){
        if(!seen.has(id))components++;
        const queue=[id],dist=new Map([[id,0]]);
        for(let i=0;i<queue.length;i++){
            const a=queue[i];seen.add(a);diameter=Math.max(diameter,dist.get(a));
            for(const b of adj.get(a))if(!dist.has(b)){dist.set(b,dist.get(a)+1);queue.push(b);}
        }
    }
    const degrees=[...adj.values()].map(a=>a.size),loops=edges.length-members.length+components;
    const branches=degrees.filter(n=>n>=3).sort((a,b)=>a-b),tips=degrees.filter(n=>n===1).length;
    return {components,loops,branches:branches.length,tips,isolates:degrees.filter(n=>n===0).length,
        maxDegree:Math.max(0,...degrees),diameterFraction:diameter/Math.max(1,members.length-components),
        // No star IDs or number of degree-two nodes: detects branching/closure
        // changes, rather than counting a different member as a different shape.
        topology:JSON.stringify([components,loops,tips,branches])};
}
export function jaccardDistance(a,b){
    const x=new Set(a),y=new Set(b),intersection=[...x].filter(v=>y.has(v)).length;
    return x.size+y.size===0?0:1-intersection/(x.size+y.size-intersection);
}
const mean=a=>a.reduce((s,x)=>s+x,0)/Math.max(1,a.length);
export function summarize(rows){
    const groups=[];
    for(const style of [...new Set(rows.map(r=>r.style))])for(let index=0;index<15;index++){
        const samples=rows.filter(r=>r.index===index&&r.style===style);if(!samples.length)continue;
        const optional=samples.map(r=>r.figure.members.filter(id=>!r.required.includes(id))),memberDistances=[],edgeDistances=[];
        for(let i=0;i<samples.length;i++)for(let j=i+1;j<samples.length;j++){
            memberDistances.push(jaccardDistance(optional[i],optional[j]));
            edgeDistances.push(jaccardDistance(samples[i].figure.edges.map(e=>JSON.stringify(e)),samples[j].figure.edges.map(e=>JSON.stringify(e))));
        }
        groups.push({index,style,samples:samples.length,candidateCount:samples[0].candidates,requiredCount:samples[0].required.length,
            uniqueMembers:new Set(samples.map(r=>JSON.stringify(r.figure.members))).size,
            uniqueTopologies:new Set(samples.map(r=>r.full.topology)).size,
            uniqueCoreTopologies:new Set(samples.map(r=>r.core.topology)).size,
            optionalMemberDistance:mean(memberDistances),edgeDistance:mean(edgeDistances)});
    }
    const ms=rows.map(r=>r.milliseconds).sort((a,b)=>a-b);
    return {samples:rows.length,groups,meanUniqueTopologies:mean(groups.map(g=>g.uniqueTopologies)),
        meanUniqueCoreTopologies:mean(groups.map(g=>g.uniqueCoreTopologies)),
        meanOptionalMemberDistance:mean(groups.map(g=>g.optionalMemberDistance)),meanEdgeDistance:mean(groups.map(g=>g.edgeDistance)),
        meanLoops:mean(rows.map(r=>r.full.loops)),meanCoreLoops:mean(rows.map(r=>r.core.loops)),
        meanDiameterFraction:mean(rows.map(r=>r.full.diameterFraction)),meanMembers:mean(rows.map(r=>r.figure.members.length)),
        disconnected:rows.filter(r=>r.full.components>1).length,isolates:rows.reduce((n,r)=>n+r.full.isolates,0),
        meanEdgeDegrees:mean(rows.map(r=>r.meanEdgeDegrees)),maxEdgeDegrees:Math.max(...rows.map(r=>r.maxEdgeDegrees)),
        milliseconds:{median:ms[Math.floor(ms.length*.5)],p95:ms[Math.floor(ms.length*.95)],max:ms.at(-1)}};
}

if(process.argv[1]&&path.resolve(process.argv[1])===fileURLToPath(import.meta.url)){
    const args=process.argv.slice(2),output=args[args.indexOf('--out')+1],countAt=args.indexOf('--samples');
    const count=countAt<0?12:Number(args[countAt+1]);
    const styleAt=args.indexOf('--styles'),styles=styleAt<0?['balanced','rich']:args[styleAt+1]?.split(',');
    if(!styles?.length||new Set(styles).size!==styles.length||styles.some(s=>!['simple','balanced','rich'].includes(s)))throw Error('Invalid --styles');
    if(!args.includes('--out')||!output||!Number.isInteger(count)||count<2||count>100)throw Error('Usage: node src/audit_regional_figures.mjs --out NEW.json [--samples 12]');
    if(fs.existsSync(output))throw Error('Refusing to overwrite an existing audit');
    const root=fileURLToPath(new URL('../',import.meta.url)),read=p=>JSON.parse(fs.readFileSync(path.join(root,p)));
    const baselineAt=args.indexOf('--baseline-ref'),baselineRef=baselineAt<0?null:args[baselineAt+1];
    let sampler=sampleRegion,baselineSource;
    if(baselineAt>=0){
        if(!/^[0-9a-f]{7,40}$/.test(baselineRef??''))throw Error('Baseline must be an explicit git commit hash');
        baselineSource=execFileSync('git',['show',`${baselineRef}:web/constellations/regional_figures.mjs`],{cwd:root,encoding:'utf8'});
        const code=baselineSource.replaceAll("from './",`from '${new URL('../web/constellations/',import.meta.url).href}`);
        sampler=(await import(`data:text/javascript;base64,${Buffer.from(code).toString('base64')}`)).sampleRegion;
    }
    const base=read('design/zodiac_draw_sampling_example.json').data,meta=read('design/zodiac_candidates_v1.json'),stars=catalogueStars(read(meta.catalogue));
    const cells=unpackGrid(base.territories),lookup=new Map(stars.map(s=>[s.id,s])),rows=[],sourceBefore=JSON.stringify(stars),before=JSON.stringify(base);
    for(const style of styles)for(let index=0;index<15;index++)for(let j=0;j<count;j++){
        const group=stars.filter(s=>cells[cellOf(s.direction)]===index).sort((a,b)=>a.app_mag-b.app_mag||a.id.localeCompare(b.id));
        const required=group.filter((s,i)=>i<3||s.app_mag<=3).map(s=>s.id),seed=`morphology-${j}`,t=performance.now();
        const result=sampler(base,stars,index,seed,style),milliseconds=performance.now()-t,{figure}=result;
        assert.ok(required.every(id=>figure.members.includes(id)&&figure.coreMembers.includes(id)));
        assert.ok(figure.members.every(id=>cells[cellOf(lookup.get(id).direction)]===index));
        for(const pair of [...figure.edges,...figure.coreEdges])assert.ok(manualArcCells(...pair.map(id=>lookup.get(id).direction)).every(k=>cells[k]===index));
        const lengths=figure.edges.map(pair=>separation(...pair.map(id=>lookup.get(id).direction)));
        rows.push({index,style,seed,method:result.method,candidates:result.candidateCount,required,figure,milliseconds,
            full:graphMeasures(figure.members,figure.edges),core:graphMeasures(figure.coreMembers,figure.coreEdges),
            meanEdgeDegrees:mean(lengths),maxEdgeDegrees:Math.max(0,...lengths)});
    }
    assert.equal(JSON.stringify(base),before);assert.equal(JSON.stringify(stars),sourceBefore);
    const hashes={};for(const p of ['design/zodiac_draw_sampling_example.json',meta.catalogue,'web/constellations/regional_figures.mjs','web/constellations/regional_graph.mjs','web/constellations/regional_quality.mjs']){
        if(fs.existsSync(path.join(root,p)))hashes[p]=createHash('sha256').update(fs.readFileSync(path.join(root,p))).digest('hex');
    }
    if(baselineSource)hashes['web/constellations/regional_figures.mjs']=createHash('sha256').update(baselineSource).digest('hex');
    if(baselineSource){delete hashes['web/constellations/regional_graph.mjs'];delete hashes['web/constellations/regional_quality.mjs'];}
    const summary=summarize(rows),report={fixture:'design/zodiac_draw_sampling_example.json',baselineRef,hashes,samplesPerRegionAndStyle:count,
        sky:{regions:base.regions.map(r=>({id:r.id,center:r.center})),stars:stars.filter(s=>s.app_mag<=4.5).map(s=>({...s,region:cells[cellOf(s.direction)]}))},
        checks:{importantStarsPreserved:true,membersAndArcsContained:true,inputsUnchanged:true},summary,rows};
    fs.mkdirSync(path.dirname(output),{recursive:true});fs.writeFileSync(output,JSON.stringify(report,null,2)+'\n',{flag:'wx'});
    console.log(JSON.stringify({...summary,groups:undefined},null,2));
}
