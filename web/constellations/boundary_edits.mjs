import {delta,wrap} from './geometry.mjs?revision=candidate-editor-band-1';
import {unpackGrid} from './territories.mjs';
import {rebuildRegions} from './region_data.mjs';
import {validateGeometry} from './generator_free.mjs?revision=candidate-editor-band-1';
import {brightAudit} from './bright_figures.mjs';
import {acceptableWidths} from './generator_ecliptic.mjs';
import {shapeEnvelope,envelopeDistance,cellDirections,CELL_RADIUS_DEGREES} from './shape_envelope.mjs';

export const MANUAL_ALGORITHM='terrax-zodiac-manual-1';
const cell=(x,y)=>{if(y<0||y>=180)throw Error('边界不能越过参考极点');return y*360+wrap(x);};
export function normalizeEdits(value){
    if(!Array.isArray(value)||value.length>64800)throw Error('手动边界记录无效');
    const seen=new Set();
    return value.map(pair=>{
        if(!Array.isArray(pair)||pair.length!==2)throw Error('手动边界格记录无效');
        const [k,owner]=pair;
        if(!Number.isInteger(k)||k<0||k>=64800||!Number.isInteger(owner)||owner<0||owner>15||seen.has(k))throw Error('手动边界包含重复或越界格');
        seen.add(k);return [k,owner];
    }).sort((a,b)=>a[0]-b[0]);
}
export function gridEdits(original,next){
    const a=unpackGrid(original.territories),b=unpackGrid(next.territories),out=[];
    for(let k=0;k<a.length;k++)if(a[k]!==b[k])out.push([k,b[k]]);
    return out;
}
// Sweep an entire existing orthogonal edge to a snapped coordinate. Adjacent
// regions receive retracted cells from their original side of the shared edge.
export function moveBoundaryEdge(data,index,edge,target){
    if(!Number.isInteger(index)||index<0||index>=15||!Number.isInteger(edge)||!Number.isFinite(target))throw Error('请选择一条边界');
    const boundary=data.regions[index].boundary;if(edge<0||edge>=boundary.length)throw Error('边界编号无效');
    const a=boundary[edge],b=boundary[(edge+1)%boundary.length],horizontal=a[1]===b[1],cells=unpackGrid(data.territories),next=new Int8Array(cells);
    const start=horizontal?a[1]+90:a[0],end=horizontal?Math.round(target)+90:a[0]+delta(Math.round(target),a[0]);
    if(Math.abs(end-start)>40)throw Error('单次移动请控制在 40° 内');
    if(end===start)return next;
    const lo=Math.min(start,end),hi=Math.max(start,end);
    const alongA=horizontal?a[0]:a[1]+90,alongB=horizontal?a[0]+delta(b[0],a[0]):b[1]+90;
    for(let u=Math.min(alongA,alongB);u<Math.max(alongA,alongB);u++){
        const minus=horizontal?cell(u,start-1):cell(start-1,u),plus=horizontal?cell(u,start):cell(start,u);
        const insideMinus=cells[minus]===index;
        if(insideMinus===(cells[plus]===index))throw Error('这条边界已变化，请重新选择');
        const outside=cells[insideMinus?plus:minus],expanding=(end>start)===insideMinus;
        for(let v=lo;v<hi;v++){
            const k=horizontal?cell(u,v):cell(v,u);
            if(expanding)next[k]=index;else if(cells[k]===index)next[k]=outside;
        }
    }
    return next;
}
export function applyEditedGrid(base,source,cells,recipe,{allowDisconnected=false}={}){
    const {optimality,localOptimality,minimumCornersPolicy,localPolicy,boundaryCleanup,manualBoundary,...rest}=base;
    const data=rebuildRegions({...rest,schema:10},source,cells,recipe);
    const adaptive=!!base.brightPolicy;
    validateGeometry(data,source,{recipe:base.recipe,rules:base.settings,adaptiveCore:adaptive,maximumMembers:adaptive?15:null,allowDisconnected});
    if(base.eclipticPolicy&&!acceptableWidths(data.regions.map(r=>r.eclipticSpan)))throw Error('此次移动会使黄道交段过窄、过宽或窄区过多');
    if(base.envelopePolicy){
        const envelopes=data.regions.map(r=>shapeEnvelope(r.members)),limit=base.envelopePolicy.maximumMarginDegrees-CELL_RADIUS_DEGREES;
        for(let k=0;k<cells.length;k++)if(cells[k]<15&&envelopeDistance(envelopes[cells[k]],cellDirections[k])>limit+1e-8)throw Error('此次移动超出了星形允许的 8° 余量');
    }
    const previous=brightAudit(base,source,unpackGrid(base.territories)),audit=brightAudit(data,source,cells);
    for(let i=0;i<15;i++){
        const omitted=new Set([...previous[i].missing,...previous[i].missingFromCore]);
        const newlyMissing=[...new Set([...audit[i].missing,...audit[i].missingFromCore])].filter(id=>!omitted.has(id));
        if(newlyMissing.length)throw Error(`${audit[i].id} 将纳入尚未参与星形的重要亮星 ${newlyMissing[0]}，请保留其原归属或重新生成星形`);
    }
    data.brightAudit=audit;
    data.manualBoundary={method:'orthogonal-edge-edit-1',status:'validated',baseAlgorithm:base.recipe.algorithm,
        changedCells:gridEdits(base,data).length,regionCorners:data.regions.map(r=>r.boundary.length)};
    return data;
}
export function applyBoundaryEdits(base,source,edits,recipe){
    const cells=new Int8Array(unpackGrid(base.territories));for(const [k,id] of normalizeEdits(edits))cells[k]=id;
    return applyEditedGrid(base,source,cells,recipe);
}
