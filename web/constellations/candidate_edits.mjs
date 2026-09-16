import {normalizeEdits,gridEdits,applyEditedGrid} from './boundary_edits.mjs?revision=candidate-editor-band-1';
import {unpackGrid} from './territories.mjs';

export const EDIT_ALGORITHM='terrax-zodiac-manual-2';
const edgeKey=(a,b)=>JSON.stringify([a,b].sort());
export function connectionStats(ids,edges){
    const neighbours=new Map(ids.map(id=>[id,[]]));
    for(const e of edges){neighbours.get(e.from).push(e.to);neighbours.get(e.to).push(e.from);}
    const seen=new Set();let components=0;
    for(const id of ids)if(!seen.has(id)){
        components++;const queue=[id];seen.add(id);
        for(let k=0;k<queue.length;k++)for(const next of neighbours.get(queue[k]))if(!seen.has(next)){seen.add(next);queue.push(next);}
    }
    const degrees=[...neighbours.values()].map(a=>a.length);
    return {loops:edges.length-ids.length+components,branches:degrees.filter(n=>n>=3).length,tips:degrees.filter(n=>n===1).length,
        components,isolated:degrees.filter(n=>n===0).length};
}
export function normalizeRemovedEdges(value=[]){
    if(!Array.isArray(value)||value.length>1000)throw Error('删除连线记录无效');
    const seen=new Set();
    const entries=value.map(entry=>{
        if(!Array.isArray(entry)||entry.length!==3)throw Error('删除连线格式无效');
        const [index,a,b]=entry;
        if(!Number.isInteger(index)||index<0||index>=15||[a,b].some(id=>typeof id!=='string'||!id.length||id.length>200)||a===b)throw Error('删除连线端点无效');
        const ids=[a,b].sort(),key=JSON.stringify([index,...ids]);if(seen.has(key))throw Error('重复删除同一连线');seen.add(key);return [index,...ids];
    });
    return entries.sort((a,b)=>a[0]-b[0]||a[1].localeCompare(b[1])||a[2].localeCompare(b[2]));
}
export function removedEdges(original,next){
    const result=[];
    for(const [i,r] of original.regions.entries()){
        const remaining=new Set(next.regions[i].variants.flatMap(v=>v.edges.map(e=>edgeKey(e.from,e.to)))),seen=new Set();
        for(const v of r.variants)for(const e of v.edges){const key=edgeKey(e.from,e.to);if(!seen.has(key)&&!remaining.has(key))result.push([i,e.from,e.to]);seen.add(key);}
    }
    return normalizeRemovedEdges(result);
}
export function candidateRecipe(base,next){
    const root=base.recipe;
    return {algorithm:EDIT_ALGORITHM,seed:root.seed,style:root.style,shapeSeeds:root.shapeSeeds,base:root,
        edits:gridEdits(base,next),removedEdges:removedEdges(base,next)};
}
export function applyCandidateEdits(base,source,recipe){
    const removals=normalizeRemovedEdges(recipe.removedEdges),edits=normalizeEdits(recipe.edits),regions=structuredClone(base.regions);
    recipe={...recipe,edits,removedEdges:removals};
    for(const [i,a,b] of removals){
        const key=edgeKey(a,b),region=regions[i];
        if(!region.variants.some(v=>v.edges.some(e=>edgeKey(e.from,e.to)===key)))throw Error('待删除的连线不在自动原轮次中');
        for(const v of region.variants)v.edges=v.edges.filter(e=>edgeKey(e.from,e.to)!==key);
    }
    for(const r of regions)r.structure=connectionStats(r.variants[1].members,r.variants[1].edges);
    const cells=new Int8Array(unpackGrid(base.territories));for(const [k,owner] of edits)cells[k]=owner;
    const data=applyEditedGrid({...base,regions},source,cells,recipe,{allowDisconnected:true});
    data.schema=12;
    data.manualEdits={method:'candidate-edits-2',status:'validated',removedEdgeCount:removals.length,
        disconnectedRegions:regions.filter(r=>r.structure.components>1).map(r=>r.id)};
    return data;
}
