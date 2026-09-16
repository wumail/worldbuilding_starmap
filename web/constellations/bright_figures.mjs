import {chooseFigure,RULES as LEGACY_RULES} from './figure.mjs';
import {cellOf,arcCells} from './territories.mjs';
import {shapeEnvelope} from './shape_envelope.mjs';

// These are drawing choices, not changes to stellar photometry. Close bright
// cluster members must not be discarded by the old 0.8-degree spacing rule.
export const RULES=Object.freeze({...LEGACY_RULES,searchLatitude:90,minSeparation:.05,lineClearance:.04});
export const BRIGHT_POLICY=Object.freeze({method:'regional-bright-anchors-1',topCount:3,maximumMagnitude:3,
    finalRegionAudit:'full-catalogue',maximumMembers:15,minimumCoreMembers:7});
export const brief=s=>({id:s.id,app_mag:s.app_mag,color_hex:s.color_hex,distance_pc:s.distance_pc,longitude:s.longitude,latitude:s.latitude,direction:[...s.direction]});
const byBrightness=(a,b)=>a.app_mag-b.app_mag||a.id.localeCompare(b.id);
export function regionalStars(source,cells){
    const groups=Array.from({length:15},()=>[]);
    for(const s of source){const i=cells[cellOf(s.direction)];if(i<15)groups[i].push(s);}
    return groups.map(g=>g.sort(byBrightness));
}
export function majorStars(group){
    return [...group].sort(byBrightness).filter((s,i)=>i<BRIGHT_POLICY.topCount||s.app_mag<=BRIGHT_POLICY.maximumMagnitude);
}
export function brightAudit(data,source,cells){
    return regionalStars(source,cells).map((group,i)=>{
        const r=data.regions[i],members=new Set(r.members.map(s=>s.id)),core=new Set(r.variants[0].members);
        const major=majorStars(group);
        return {id:r.id,brightest:group[0]?{id:group[0].id,app_mag:group[0].app_mag}:null,
            required:major.map(s=>s.id),missing:major.filter(s=>!members.has(s.id)).map(s=>s.id),
            missingFromCore:major.filter(s=>!core.has(s.id)).map(s=>s.id),regionStarCount:group.length};
    });
}
export function chooseBrightFigure(group,reference,seed,style,cells,index){
    const pool=group.filter(s=>s.app_mag<=RULES.candidateMagnitude),major=majorStars(group);
    const requiredIds=new Set([...shapeEnvelope(reference.members).ids,...major.map(s=>s.id)]);
    const required=pool.filter(s=>requiredIds.has(s.id));
    if(required.length!==requiredIds.size)throw Error(`Z${String(index+1).padStart(2,'0')} 的重要亮星不在可连接星表中`);
    const acceptArc=(a,b)=>arcCells(a,b).every(k=>cells[k]===index);
    for(let attempt=0;attempt<8;attempt++){
        const target=Math.max(required.length,style==='rich'?12+attempt%4:10+attempt%2);
        const f=chooseFigure(pool,`${seed}/bright/${attempt}`,style,acceptArc,required,target,
            {rules:RULES,maxMembers:BRIGHT_POLICY.maximumMembers,coreRequired:major.map(s=>s.id)});
        if(f)return f;
    }
    throw Error(`Z${String(index+1).padStart(2,'0')} 暂未找到完整纳入重要亮星的连线；不会静默略过亮星`);
}
