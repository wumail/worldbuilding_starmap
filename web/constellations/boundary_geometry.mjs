import {delta,wrap} from './geometry.mjs';

const fail=message=>{throw Error(message);};
const cell=(x,y)=>y<0||y>=180?-1:y*360+wrap(x);
const same=(a,b)=>wrap(a[0])===wrap(b[0])&&a[1]===b[1];
const shifted=(p,horizontal,target)=>horizontal?[p[0],Math.round(target)]:[Math.round(target),p[1]];

// Sweep a whole edge or a subsegment through the shared ownership grid.
// A retracted strip returns to the owner on its original outside, per column.
export function shiftBoundarySegment(cells,index,a,b,target){
    return sweep(cells,index,a,b,target,false);
}
function sweep(cells,index,a,b,target,unwrapped){
    if(!Number.isFinite(target))fail('请输入有限的边界坐标');
    if(!a||!b||same(a,b)||(a[1]!==b[1]&&wrap(a[0])!==wrap(b[0])))fail('请选择非零长度的横边或竖边');
    const horizontal=a[1]===b[1],next=new Int8Array(cells);
    const start=horizontal?a[1]+90:a[0],end=horizontal?Math.round(target)+90:unwrapped?Math.round(target):a[0]+delta(Math.round(target),a[0]);
    if(horizontal&&(end<0||end>180))fail('赤纬坐标应位于 −90° 至 90°');
    const lo=Math.min(start,end),hi=Math.max(start,end);
    const alongA=horizontal?a[0]:a[1]+90,alongB=horizontal?(unwrapped?b[0]:a[0]+delta(b[0],a[0])):b[1]+90;
    for(let u=Math.min(alongA,alongB);u<Math.max(alongA,alongB);u++){
        const minus=horizontal?cell(u,start-1):cell(start-1,u),plus=horizontal?cell(u,start):cell(start,u),insideMinus=minus>=0&&cells[minus]===index;
        if(insideMinus===(plus>=0&&cells[plus]===index))fail('这段边界已合并或变化，请重新选择');
        const outsideIndex=insideMinus?plus:minus,outside=outsideIndex<0?15:cells[outsideIndex],expanding=(end>start)===insideMinus;
        for(let v=lo;v<hi;v++){const k=horizontal?cell(u,v):cell(v,u);if(k<0)continue;if(expanding)next[k]=index;else if(cells[k]===index)next[k]=outside;}
    }
    return next;
}

export function isBoundaryCorner(ring,point){
    const a=ring[(point+ring.length-1)%ring.length],b=ring[point],c=ring[(point+1)%ring.length];
    return !!a&&!!b&&!!c&&(a[1]===b[1])!==(b[1]===c[1]);
}
function corner(ring,point){
    if(!ring||!Number.isInteger(point)||!isBoundaryCorner(ring,point))fail('请选择一个真正的拐点');
    // Rendering divides long parallels at 90-degree meridians. These are not
    // editable corners: walk through them and retain the full unwrapped span.
    const neighbour=(start,step)=>{
        let i=start,x=ring[i][0];
        do{const j=(i+step+ring.length)%ring.length;x+=delta(ring[j][0],ring[i][0]);i=j;}while(i!==start&&!isBoundaryCorner(ring,i));
        return {point:i,position:[x,ring[i][1]]};
    };
    return [neighbour(point,-1).position,ring[point],neighbour(point,1).position];
}

// Move the two incident edges in sequence. Try the reverse order if the first
// temporarily collapses an edge; each sweep must still start on a real boundary.
export function moveBoundaryCorner(cells,index,ring,point,target){
    if(!Array.isArray(target)||target.length!==2||!target.every(Number.isFinite))fail('请输入有限的赤经和赤纬');
    const goal=[wrap(Math.round(target[0])),Math.round(target[1])];
    if(Math.abs(goal[1])>90)fail('赤纬坐标应位于 −90° 至 90°');
    const [a,b,c]=corner(ring,point);if(same(b,goal))return new Int8Array(cells);
    goal[0]=b[0]+delta(goal[0],b[0]);
    let lastError;
    for(const reverse of [false,true])try{
        const p=reverse?c:a,q=reverse?a:c,horizontal=p[1]===b[1],normal=horizontal?goal[1]:goal[0];
        const first=sweep(cells,index,p,b,normal,true),moved=shifted(b,horizontal,normal);
        if(same(moved,goal))return first;
        if(same(moved,q))fail('这次移动会先合并相邻边');
        return sweep(first,index,moved,q,horizontal?goal[0]:goal[1],true);
    }catch(error){lastError=error;}
    fail(`拐点无法直接移动到这里：${lastError.message}。可先调整相邻边。`);
}

// Orthogonal outlines cannot lose just one vertex. Collapse either incident
// edge onto its other neighbour, removing the selected corner and redundant turns.
export function removeBoundaryCorner(cells,index,ring,point,side='previous'){
    const [a,b,c]=corner(ring,point);
    if(side==='previous')return sweep(cells,index,a,b,a[1]===b[1]?c[1]:c[0],true);
    if(side==='next')return sweep(cells,index,b,c,b[1]===c[1]?a[1]:a[0],true);
    fail('请选择拐点合并方向');
}

// Inserting a collinear point would vanish when rebuilding the grid boundary.
// Insert a real, one-cell-deep step, with one cell left at either end of the edge.
export function boundaryStep(cells,index,a,b,at,width){
    if(!Number.isFinite(width)||width<1)fail('新增段宽度至少为 1°');
    const horizontal=a[1]===b[1],begin=horizontal?a[0]:a[1],end=horizontal?a[0]+delta(b[0],a[0]):b[1];
    const lo=Math.min(begin,end),hi=Math.max(begin,end);if(hi-lo<3)fail('这段边不足 3°，请先拉长它或选择较长的边');
    const span=Math.min(Math.round(width),hi-lo-2),along=horizontal?a[0]+delta(at[0],a[0]):at[1];
    const start=Math.max(lo+1,Math.min(hi-1-span,Math.round(along-span/2))),finish=start+span;
    const p=horizontal?[wrap(start),a[1]]:[a[0],start],q=horizontal?[wrap(finish),a[1]]:[a[0],finish];
    const mid=Math.floor((start+finish)/2),normal=horizontal?a[1]:a[0],minus=horizontal?cell(mid,a[1]+89):cell(a[0]-1,mid+90);
    let target=normal+(minus>=0&&cells[minus]===index?1:-1);
    if(horizontal&&Math.abs(target)>90)target=normal-(target-normal);
    return {a:p,b:q,horizontal,width:span,target:horizontal?target:wrap(target)};
}
