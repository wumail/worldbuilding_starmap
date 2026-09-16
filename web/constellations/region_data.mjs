import {vector} from './geometry.mjs?revision=candidate-editor-band-1';
import {packGrid,cellOf,territoryBoundary,territoryIntervals,fromEquatorial} from './territories.mjs';

export function withFigure(region,f){
    return {...region,center:f.center,members:f.members,
        variants:[{id:'core',label:'亮星骨架',members:f.core.map(s=>s.id),edges:f.coreEdges},
            {id:'extended',label:'完整星形',members:f.members.map(s=>s.id),edges:f.edges}],
        structure:f.structure,brightest:Math.min(...f.members.map(s=>s.app_mag)),faintest:Math.max(...f.members.map(s=>s.app_mag)),
        oldSectors:[...new Set(f.members.map(s=>Math.floor(s.longitude/24)+1))].sort((a,b)=>a-b)};
}
export function rebuildRegions(data,source,cells,recipe=data.recipe){
    const pool=source.filter(s=>s.app_mag<=data.settings.candidateMagnitude&&Math.abs(s.latitude)<=data.settings.searchLatitude);
    const groups=Array.from({length:15},()=>[]),sections=territoryIntervals(cells);
    for(const s of pool){const i=cells[cellOf(s.direction)];if(i<15)groups[i].push(s);}
    const regions=data.regions.map((r,i)=>{
        const boundary=territoryBoundary(cells,i),intervals=sections.filter(s=>s.index===i).map(({start,end})=>({start,end}));
        return {...r,boundary,polygon:boundary.map(([ra,dec])=>fromEquatorial(vector(ra,dec))),intervals,
            eclipticSpan:intervals.reduce((n,s)=>n+s.end-s.start,0),candidateCount:groups[i].length,brightCount:groups[i].filter(s=>s.app_mag<=4).length};
    });
    return {...data,recipe,regions,territories:packGrid(cells),sourceCount:source.length,candidateCount:pool.length,
        eligibleCandidateCount:groups.reduce((n,g)=>n+g.length,0),selectedExtendedCount:regions.reduce((n,r)=>n+r.members.length,0),
        selectedCoreCount:regions.reduce((n,r)=>n+r.variants[0].members.length,0)};
}
