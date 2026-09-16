import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {prepareDraw,generateDraw as generateCurrent,validateDraw,redrawRecipe} from '../web/constellations/generator.mjs';
import {cornerProblem,solveMinimumCorners,topologyCuts} from '../web/constellations/minimum_corners.mjs';
import {unpackGrid,territoryBoundary,cellOf,arcCells,boundaryPoints,regionAt} from '../web/constellations/territories.mjs';
import {shapeEnvelope,envelopeDistance,cellAreaDegrees} from '../web/constellations/shape_envelope.mjs';
import {protectedCells} from '../web/constellations/boundary_cleanup.mjs';
import {catalogueStars} from '../src/build_zodiac_candidates.mjs';
await prepareDraw({algorithm:'terrax-zodiac-draw-7'});
const generateDraw=(stars,meta,recipe={})=>generateCurrent(stars,meta,{algorithm:'terrax-zodiac-draw-7',...recipe});
const meta=JSON.parse(fs.readFileSync(new URL('../design/zodiac_candidates_v1.json',import.meta.url)));
const stars=catalogueStars(JSON.parse(fs.readFileSync(new URL(`../${meta.catalogue}`,import.meta.url))));

// Independent oriented-edge tracing for a small periodic grid, also rejecting
// extra rings. Count direction changes, not the solver's four-cell formula.
function bruteCorners(labels,owner,w,h){
    const edges=new Map();let n=0;
    const key=(x,y)=>y*w+(x+w)%w,at=(x,y)=>y<0||y>=h?-1:labels[y*w+(x+w)%w];
    const add=(a,b)=>{if(edges.has(a))throw Error('touch');edges.set(a,b);n++;};
    for(let y=0;y<h;y++)for(let x=0;x<w;x++)if(at(x,y)===owner){
        if(y===0||y===h-1)throw Error('pole');
        if(at(x,y-1)!==owner)add(key(x,y),key(x+1,y));if(at(x+1,y)!==owner)add(key(x+1,y),key(x+1,y+1));
        if(at(x,y+1)!==owner)add(key(x+1,y+1),key(x,y+1));if(at(x-1,y)!==owner)add(key(x,y+1),key(x,y));
    }
    if(!n)throw Error('empty');
    const start=edges.keys().next().value;let p=start;const ring=[];
    do{ring.push(p);const next=edges.get(p);if(next===undefined)throw Error('open');edges.delete(p);p=next;}while(p!==start&&ring.length<=n);
    if(edges.size||p!==start)throw Error('hole/island');
    return ring.filter((p,i)=>{const a=ring[(i+ring.length-1)%ring.length],b=ring[(i+1)%ring.length];return !((a%w===p%w&&b%w===p%w)||(Math.floor(a/w)===Math.floor(p/w)&&Math.floor(b/w)===Math.floor(p/w)));}).length;
}
test('joint optimum and area tie-break agree with exhaustive enumeration of all small-grid assignments',()=>{
    for(const count of [1,2])for(const variant of [0,1,2]){
        const width=4,height=4,n=16,labels=new Int8Array(n).fill(count),fixed=new Uint8Array(n),mask=new Uint16Array(n);
        for(let k=4;k<12;k++)mask[k]=(1<<count)-1;
        labels[4]=0;fixed[4]=1;if(count===2){labels[11]=1;fixed[11]=1;}
        if(variant){mask[6]=0;if(variant===2)mask[9]=1;}
        const problem={width,height,count,labels,fixed,mask},model=cornerProblem(problem);let best=Infinity,area=Infinity,valid=0;
        function enumerate(k,out){
            if(k===12){let corners;try{corners=Array.from({length:count},(_,i)=>bruteCorners(out,i,width,height)).reduce((a,b)=>a+b,0);}catch{return;}
                valid++;const size=out.filter(x=>x<count).length;
                if(corners<best||corners===best&&size<area){best=corners;area=size;}
                // Every generated separation inequality must retain every legal
                // result; additional cut soundness is tested below.
                assert.equal(topologyCuts(out,model).length,0);return;
            }
            for(let i=0;i<=count;i++)if((i===count||mask[k]&(1<<i))&&(!fixed[k]||labels[k]===i)){out[k]=i;enumerate(k+1,out);}
        }
        enumerate(4,new Int8Array(n).fill(count));assert.ok(valid);
        const result=solveMinimumCorners(problem);
        assert.equal(result.certificate.minimumCorners,best);assert.equal(result.certificate.integerLowerBound,best);
        assert.equal(result.certificate.minimumCellsAtMinimumCorners,area);
        assert.equal(result.labels.filter(i=>i<count).length,area);
    }
});
test('fixed figures outside the envelope and incomplete solves never receive an optimal certificate',()=>{
    const labels=new Int8Array(16).fill(1),fixed=new Uint8Array(16),mask=new Uint16Array(16);labels[4]=0;fixed[4]=1;mask[5]=1;
    assert.throws(()=>solveMinimumCorners({width:4,height:4,count:1,labels,fixed,mask}),/无法保证包围/);
    mask[4]=1;assert.throws(()=>solveMinimumCorners({width:4,height:4,count:1,labels,fixed,mask},{seconds:0}),/尚未证明最优/);
});
test('48 sky draws prove minimum corners while preserving figures, protected geometry and angular limits',()=>{
    const measurements=[];
    for(let i=0;i<24;i++){
        const seed=i===0?'terrax-1ptws5s':i===1?'terrax-b1bxb1':i===2?'terrax-001':`minimum-${i-3}`;
        for(const style of ['rich','balanced']){
            const old=generateDraw(stars,meta,{seed,style,algorithm:'terrax-zodiac-draw-6'}),start=performance.now(),d=generateDraw(stars,meta,{seed,style}),cells=unpackGrid(d.territories),previous=unpackGrid(old.territories),fixed=protectedCells(old,stars);
            const withoutBoundary=data=>data.regions.map(({boundary,polygon,...r})=>r);
            assert.deepEqual(withoutBoundary(d),withoutBoundary(old));
            for(let k=0;k<cells.length;k++)if(fixed[k])assert.equal(cells[k],previous[k]);
            let area=0,maxGap=0;for(let k=0;k<cells.length;k++)if(cells[k]<15)area+=cellAreaDegrees(k);
            for(const [j,r] of d.regions.entries()){
                assert.deepEqual(r.boundary,territoryBoundary(cells,j));
                const e=shapeEnvelope(r.members);for(const v of boundaryPoints(r.boundary,.25))maxGap=Math.max(maxGap,envelopeDistance(e,v));
                for(const s of r.members)assert.equal(regionAt(d,s.direction),j);
                const members=new Map(r.members.map(s=>[s.id,s]));for(const e of r.variants[1].edges)assert.ok(arcCells(members.get(e.from).direction,members.get(e.to).direction).every(k=>cells[k]===j));
                for(let k=0;k<r.boundary.length;k++){const a=r.boundary[k],b=r.boundary[(k+1)%r.boundary.length];assert.ok(a[0]===b[0]||a[1]===b[1]);}
            }
            const corners=d.regions.reduce((n,r)=>n+r.boundary.length,0),oldCorners=old.regions.reduce((n,r)=>n+r.boundary.length,0);
            assert.equal(corners,d.optimality.minimumCorners);assert.equal(corners,d.optimality.integerLowerBound);assert.ok(corners<=oldCorners);assert.ok(maxGap<=8+1e-8);
            measurements.push({seed,style,milliseconds:performance.now()-start,corners,oldCorners,maxBoundaryDistance:maxGap,areaSquareDegrees:area,certificate:d.optimality});
        }
    }
    fs.mkdirSync(new URL('../reports/zodiac_minimum_corners/',import.meta.url),{recursive:true});
    fs.writeFileSync(new URL('../reports/zodiac_minimum_corners/seed_validation.json',import.meta.url),JSON.stringify(measurements,null,2)+'\n');
});
test('minimum borders remain fixed across local rerolls, six historical exports reproduce and false certificates fail',()=>{
    const d=generateDraw(stars,meta,{seed:'terrax-1ptws5s'}),next=generateDraw(stars,meta,redrawRecipe(d.recipe,[0,6],'minimum-reroll'));
    assert.deepEqual(next.territories,d.territories);assert.deepEqual(next.regions[0],d.regions[0]);assert.deepEqual(next.regions[6],d.regions[6]);assert.notDeepEqual(next.regions,d.regions);
    const bad=structuredClone(d);bad.optimality.integerLowerBound--;assert.throws(()=>validateDraw(bad,stars),/证明不符/);
    const before=JSON.stringify(d);d.regions[0].boundary[0][0]+=1;d.territories.runs[0][0]++;
    assert.equal(JSON.stringify(generateDraw(stars,meta,d.recipe)),before);
    for(const name of ['example','ordered_example','free_example','clean_example','ecliptic_example','fitted_example']){
        const saved=JSON.parse(fs.readFileSync(new URL(`../design/zodiac_draw_${name}.json`,import.meta.url))).data;
        assert.equal(JSON.stringify(generateDraw(stars,meta,saved.recipe)),JSON.stringify(saved));
    }
});
