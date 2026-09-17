import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {sampleRegion} from '../web/constellations/regional_figures.mjs';
import {catalogueStars} from '../src/build_zodiac_candidates.mjs';
import {graphMeasures} from '../src/audit_regional_figures.mjs';
import {angleMeasures} from '../src/audit_regional_clarity.mjs';
import {simplifyCore,regionalGraph} from '../web/constellations/regional_graph.mjs';
import {complexityBudget} from '../web/constellations/regional_quality.mjs';
import {randomFrom} from '../web/constellations/generator_ordered.mjs';
import {packGrid,unpackGrid,cellOf} from '../web/constellations/territories.mjs';
import {manualArcCells,manualRecipe,applyManualFigures} from '../web/constellations/manual_figures.mjs';
import {vector,separation,arc,cross,unit} from '../web/constellations/geometry.mjs';

const read=p=>JSON.parse(fs.readFileSync(new URL('../'+p,import.meta.url)));
const base=read('design/zodiac_draw_sampling_example.json').data,meta=read('design/zodiac_candidates_v1.json');
const stars=catalogueStars(read(meta.catalogue)),cells=unpackGrid(base.territories),lookup=new Map(stars.map(s=>[s.id,s]));
const blank=()=>({territories:packGrid(new Int8Array(64800)),regions:Array.from({length:15},()=>({members:[],variants:[{members:[],edges:[]},{members:[],edges:[]}]}))});
const fixedStars=n=>Array.from({length:n},(_,i)=>({id:`fixed-${i}`,app_mag:2+i/(n*2),
    direction:vector(32+8*Math.cos(i*2.4)*Math.sqrt(i/n),12+8*Math.sin(i*2.4)*Math.sqrt(i/n))}));
const edgeKey=e=>JSON.stringify([...e].sort());
function assertClearArcs(figure,lookup){
    const edges=figure.edges.map(e=>e.map(id=>lookup.get(id).direction));
    for(let i=0;i<edges.length;i++){
        const [a,b]=edges[i],length=separation(a,b);assert.ok(Number.isFinite(length)&&length>0&&length<180);
        // An independent dense probe checks the clearance contract.
        const samples=arc(a,b,.025);
        for(const id of figure.members)if(!figure.edges[i].includes(id))assert.ok(samples.every(p=>separation(p,lookup.get(id).direction)>=.039999));
        for(let k=i+1;k<edges.length;k++){
            if(figure.edges[i].some(id=>figure.edges[k].includes(id)))continue;
            const [c,d]=edges[k],intersection=cross(cross(a,b),cross(c,d));if(Math.hypot(...intersection)<1e-12)continue;
            const v=unit(intersection);
            for(const p of [v,v.map(x=>-x)])assert.ok(Math.abs(separation(a,p)+separation(p,b)-length)>1e-7||
                Math.abs(separation(c,p)+separation(p,d)-separation(c,d))>1e-7);
        }
    }
}

test('fixed bright members produce different graph structures, including open chains, branches and preserved cycles',()=>{
    const source=fixedStars(12),data=blank(),rows=[],lines=new Set();
    for(let i=0;i<24;i++){
        const {figure}=sampleRegion(data,source,0,`fixed-members-${i}`,'rich');
        assert.deepEqual(figure.members,[...source.map(s=>s.id)].sort());
        assert.deepEqual(figure.coreMembers,figure.members);
        const full=graphMeasures(figure.members,figure.edges),core=graphMeasures(figure.coreMembers,figure.coreEdges);
        assert.equal(full.components,1);assert.equal(core.components,1);assert.ok(full.maxDegree<=4);assert.ok(core.loops<=Math.min(1,full.loops));
        assert.ok(figure.coreEdges.every(e=>figure.edges.some(f=>edgeKey(e)===edgeKey(f))));rows.push(full);lines.add(JSON.stringify(figure.edges));
    }
    assert.ok(new Set(rows.map(r=>r.topology)).size>=4,'bounded simple structures must still vary with fixed members');
    assert.ok(lines.size>=20,'readability must not collapse fixed members into a single wiring');
    assert.ok(rows.some(r=>r.loops===0&&r.diameterFraction>=.8));
    assert.ok(rows.some(r=>r.branches>=2));assert.ok(rows.some(r=>r.loops>=1));
});

test('real regions retain absolute and relative bright anchors, and vary the bright skeleton itself',()=>{
    const counts=[];
    for(const index of [0,5,10])for(const style of ['balanced','rich']){
        const group=stars.filter(s=>cells[cellOf(s.direction)]===index).sort((a,b)=>a.app_mag-b.app_mag||a.id.localeCompare(b.id));
        const required=group.filter((s,i)=>i<3||s.app_mag<=3).map(s=>s.id),shapes=new Set();
        for(let j=0;j<12;j++){
            const {figure}=sampleRegion(base,stars,index,`skeleton-check-${j}`,style);
            assert.ok(required.every(id=>figure.members.includes(id)&&figure.coreMembers.includes(id)));
            for(const e of [...figure.edges,...figure.coreEdges])assert.ok(manualArcCells(...e.map(id=>lookup.get(id).direction)).every(k=>cells[k]===index));
            const full=graphMeasures(figure.members,figure.edges),core=graphMeasures(figure.coreMembers,figure.coreEdges);
            assert.ok(full.maxDegree<=4);assert.equal(core.components,full.components);assert.ok(core.loops<=Math.min(1,full.loops));shapes.add(core.topology);
        }
        counts.push(shapes.size);
    }
    assert.ok(counts.reduce((n,x)=>n+x,0)/counts.length>=3,'skeletons must differ structurally, not merely select new IDs');
});

test('the user-identified crowded figure becomes readable with exactly the same eleven real stars',()=>{
    const old=read('tests/fixtures/regional_clutter.json'),source=old.figure.members.map(id=>lookup.get(id)),recipe=manualRecipe(base,base);
    recipe.figures=[old.figure];const data=applyManualFigures(base,stars,recipe);
    const {figure}=sampleRegion(data,source,10,'morphology-1','rich');
    assert.deepEqual(figure.members,old.figure.members);assert.ok(old.required.every(id=>figure.coreMembers.includes(id)));
    const original=angleMeasures(old.figure.members,old.figure.edges,lookup),current=angleMeasures(figure.members,figure.edges,lookup);
    assert.ok(original.narrow30>=5);assert.equal(current.narrow30,0);assert.ok(current.adjacentBranchEdges<=1&&current.adjacentBranchEdges<original.adjacentBranchEdges);
    const stats=graphMeasures(figure.members,figure.edges);assert.equal(stats.components,1);assert.ok(stats.branches<=3&&stats.loops<=1);
    assertClearArcs(figure,lookup);
    for(const e of [...figure.edges,...figure.coreEdges])assert.ok(manualArcCells(...e.map(id=>lookup.get(id).direction)).every(k=>cells[k]===10));
});

test('skeleton reduction removes redundant loop edges while preserving all required stars and connectivity',()=>{
    const old=read('tests/fixtures/regional_clutter.json'),source=old.figure.members.map(id=>lookup.get(id));
    const edges=old.figure.edges.map(([from,to])=>({from,to,degrees:separation(lookup.get(from).direction,lookup.get(to).direction)}));
    const {core,coreEdges}=simplifyCore(source,edges,new Set(old.required),complexityBudget('balanced',source.length));
    const ids=core.map(s=>s.id),pairs=coreEdges.map(e=>[e.from,e.to]),stats=graphMeasures(ids,pairs);
    assert.equal(stats.components,1);assert.ok(stats.loops<=1);assert.ok(old.required.every(id=>ids.includes(id)));
    assert.ok(pairs.every(e=>old.figure.edges.some(f=>edgeKey(e)===edgeKey(f))));
    assert.ok(angleMeasures(ids,pairs,lookup).narrow30<angleMeasures(old.figure.coreMembers,old.figure.coreEdges,lookup).narrow30);
});

test('a clear primary ring survives skeleton reduction, while a thin loop is not added around required close pairs',()=>{
    const source=Array.from({length:6},(_,i)=>({id:`ring-${i}`,app_mag:2,direction:vector(30+5*Math.cos(i*Math.PI/3),10+5*Math.sin(i*Math.PI/3))}));
    const edges=source.map((s,i)=>({from:s.id,to:source[(i+1)%6].id,degrees:separation(s.direction,source[(i+1)%6].direction)}));
    const required=new Set(source.map(s=>s.id)),budget=complexityBudget('balanced',source.length);
    const core=simplifyCore(source,edges,required,budget);
    assert.equal(graphMeasures(core.core.map(s=>s.id),core.coreEdges.map(e=>[e.from,e.to])).loops,1);
    const narrow=[[20,10],[40,10],[40,10.2],[20,10.2]].map(([lon,lat],i)=>({id:`pair-${i}`,app_mag:2,direction:vector(lon,lat)}));
    for(let i=0;i<4;i++){
        const {figure}=sampleRegion(blank(),narrow,0,`thin-ring-${i}`,'rich');
        assert.equal(figure.members.length,4);assert.equal(figure.coreMembers.length,4);
        assert.equal(graphMeasures(figure.members,figure.edges).loops,0);
    }
});

test('local complexity controls member and loop budgets without changing source brightness or defaulting to the round style',()=>{
    let simpleMembers=0,richMembers=0,simpleBranches=0,richBranches=0;
    const original=JSON.stringify(base),source=JSON.stringify(stars);
    for(let i=0;i<12;i++){
        const generated=['simple','balanced','rich'].map(style=>sampleRegion(base,stars,10,`complexity-${i}`,style));
        for(let j=0;j<generated.length;j++){
            const r=generated[j],stats=graphMeasures(r.figure.members,r.figure.edges);
            assert.equal(stats.components,1);assert.ok(stats.loops<=j);
            assert.ok(r.importantStars.every(id=>r.figure.members.includes(id)&&r.figure.coreMembers.includes(id)));
            if(j)assert.ok(r.figure.members.length>=generated[j-1].figure.members.length);
        }
        simpleMembers+=generated[0].figure.members.length;richMembers+=generated[2].figure.members.length;
        simpleBranches+=graphMeasures(generated[0].figure.members,generated[0].figure.edges).branches;
        richBranches+=graphMeasures(generated[2].figure.members,generated[2].figure.edges).branches;
    }
    assert.ok(simpleMembers<richMembers);assert.ok(simpleBranches<richBranches);
    assert.deepEqual(sampleRegion(base,stars,10,'local-default'),sampleRegion(base,stars,10,'local-default','balanced'));
    assert.equal(JSON.stringify(base),original);assert.equal(JSON.stringify(stars),source);
});

test('regional suggestions avoid intersecting arcs and lines through other selected stars',()=>{
    for(const index of [2,7,12])for(let j=0;j<5;j++){
        const {figure}=sampleRegion(base,stars,index,`geometry-${j}`,'rich');
        assertClearArcs(figure,lookup);
    }
});

test('large required sets retain distant clumps and connect them with legal sparse-graph bridges',()=>{
    const source=Array.from({length:48},(_,i)=>{
        const k=i%24,radius=3*Math.sqrt((k+1)/24);
        return {id:`clump-${i}`,app_mag:2+i/96,
            direction:vector((i<24?20:42)+radius*Math.cos(k*2.4),12+radius*Math.sin(k*2.4))};
    });
    const data=blank(),before=JSON.stringify({data,source}),{figure}=sampleRegion(data,source,0,'bright-clumps','balanced');
    assert.deepEqual(figure.members,source.map(s=>s.id).sort());assert.deepEqual(figure.coreMembers,figure.members);
    const stats=graphMeasures(figure.members,figure.edges);assert.equal(stats.components,1);assert.ok(stats.maxDegree<=4);
    const left=new Set(source.slice(0,24).map(s=>s.id));assert.ok(figure.edges.some(([a,b])=>left.has(a)!==left.has(b)));
    const localLookup=new Map(source.map(s=>[s.id,s]));assertClearArcs(figure,localLookup);
    const grid=unpackGrid(data.territories);
    for(const e of figure.edges)assert.ok(manualArcCells(...e.map(id=>localLookup.get(id).direction)).every(k=>grid[k]===0));
    assert.equal(JSON.stringify({data,source}),before);
});

test('large sparse graphs inspect alternative cross-clump pairs when their nearest bridge is rejected',()=>{
    const source=Array.from({length:34},(_,i)=>{
        const right=i>=17,k=i%17,angle=k*2.4;
        return {id:`bridge-${i}`,app_mag:2,direction:vector((right?30:10)+(i===0||i===33?0:3*Math.cos(angle)),i===0||i===33?0:4+2*Math.sin(angle))};
    });
    let bridgeChecked=false;
    const accepted=(a,b)=>{
        const i=Number(a.id.slice(7)),j=Number(b.id.slice(7));
        if(i===0&&j===33){bridgeChecked=true;return true;}
        return (i<17)===(j<17);
    };
    const intent={kind:'trail',loops:0,temperature:.3,branchCost:.8,continuation:.8,budget:complexityBudget('balanced',34)};
    const g=regionalGraph(source,new Set(source.map(s=>s.id)),randomFrom('alternative-bridge'),intent,accepted);
    assert.equal(bridgeChecked,true);assert.equal(graphMeasures(source.map(s=>s.id),g.edges.map(e=>[e.from,e.to])).components,1);
    assert.equal(g.core.length,34);assertClearArcs({members:source.map(s=>s.id),edges:g.edges.map(e=>[e.from,e.to])},new Map(source.map(s=>[s.id,s])));
});

test('important stars override the member budget; ordering and method replay are deterministic',()=>{
    const data=blank(),source=fixedStars(20),a=sampleRegion(data,source,0,'twenty-bright','balanced');
    assert.equal(a.figure.members.length,20);assert.equal(a.figure.coreMembers.length,20);
    assert.deepEqual(sampleRegion(data,[...source].reverse(),0,'twenty-bright','balanced'),a);
    assert.throws(()=>sampleRegion(data,source,0,'','rich'),/种子/);
    assert.throws(()=>sampleRegion(data,source,0,'test','unknown'),/复杂度/);
    assert.throws(()=>sampleRegion(data,source,0,'test','toString'),/复杂度/);
});

test('coincident and antipodal required stars remain truthful points without undefined arcs',()=>{
    const data=blank(),source=[{id:'a',app_mag:2,direction:vector(0,0)},{id:'b',app_mag:2.1,direction:vector(180,0)},
        {id:'c',app_mag:2.2,direction:vector(0,0)}];
    const {figure}=sampleRegion(data,source,0,'degenerate','rich');
    assert.deepEqual(figure.members,['a','b','c']);assert.deepEqual(figure.edges,[]);assert.deepEqual(figure.coreEdges,[]);
});
