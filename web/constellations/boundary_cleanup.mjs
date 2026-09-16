import {vector,separation} from './geometry.mjs?revision=candidate-editor-band-1';
import {GRID,arcCells,cellOf,unpackGrid,territoryBoundary} from './territories.mjs';

export const CLEANUP=Object.freeze({method:'protected-adaptive-blocks-1',blockDegrees:4});
const W=GRID.width,H=GRID.height;

// Protect the entire old candidate pool and every possible legal old edge, not
// just today's chosen figure. Cleanup is therefore independent of shapeSeeds:
// local rerolls retain the same boundary, and no existing figure is resampled.
export function protectedCells(data,stars){
    const labels=unpackGrid(data.territories),fixed=new Uint8Array(labels.length),groups=Array.from({length:15},()=>[]);
    for(const s of stars)if(s.app_mag<=4.5&&Math.abs(s.latitude)<=30){
        const k=cellOf(s.direction);fixed[k]=1;if(labels[k]<15)groups[labels[k]].push(s);
    }
    for(let i=0;i<15;i++)for(let a=0;a<groups[i].length;a++)for(let b=a+1;b<groups[i].length;b++){
        const p=groups[i][a].direction,q=groups[i][b].direction;
        if(separation(p,q)>13+1e-9)continue;
        const cells=arcCells(p,q);
        if(cells.every(k=>labels[k]===i))for(const k of cells)fixed[k]=1;
    }
    // Preserve the exact ecliptic ownership, including tangencies and the seam.
    for(let longitude=0;longitude<360;longitude+=10)for(const k of arcCells(vector(longitude,0),vector(longitude+10,0)))fixed[k]=1;
    return fixed;
}

function coarseTargets(labels,fixed,canAssign=null){
    const target=new Int8Array(labels);
    function tile(x,y,size){
        const counts=new Int16Array(16),protectedOwners=new Set();
        for(let yy=y;yy<y+size;yy++)for(let xx=x;xx<x+size;xx++){
            const k=yy*W+xx;counts[labels[k]]++;if(fixed[k])protectedOwners.add(labels[k]);
        }
        if(protectedOwners.size>1){
            if(size>1)for(const dy of [0,size/2])for(const dx of [0,size/2])tile(x+dx,y+dy,size/2);
            return;
        }
        let id=protectedOwners.size?[...protectedOwners][0]:counts.reduce((best,n,i)=>n>counts[best]?i:best,0);
        if(canAssign){
            // A fitted region may not absorb a whole block near its margin.
            // Prefer a valid neighbouring owner (including the remainder), or
            // subdivide. Otherwise the transfer guard would leave each tiny
            // stair step in place instead of coarsening the shared boundary.
            const choices=protectedOwners.size?[id]:Array.from({length:16},(_,i)=>i).filter(i=>counts[i]).sort((a,b)=>counts[b]-counts[a]||a-b);
            id=choices.find(owner=>{
                for(let yy=y;yy<y+size;yy++)for(let xx=x;xx<x+size;xx++)if(!canAssign(yy*W+xx,owner))return false;
                return true;
            });
            if(id===undefined){
                if(size>1)for(const dy of [0,size/2])for(const dx of [0,size/2])tile(x+dx,y+dy,size/2);
                return;
            }
        }
        for(let yy=y;yy<y+size;yy++)target.fill(id,yy*W+x,yy*W+x+size);
    }
    for(let y=0;y<H;y+=4)for(let x=0;x<W;x+=4)tile(x,y,4);
    return target;
}

function mayTransfer(labels,k,id){
    const x=k%W,y=Math.floor(k/W);if(y===0||y===H-1)return false;
    const left=(x+W-1)%W,right=(x+1)%W,old=labels[k];
    // Clockwise eight-neighbour ring. A single run of both the old and new
    // labels preserves connectivity of each region and its complement. Require
    // a shared side too, so regions can never join by a diagonal corner alone.
    const ring=[labels[(y-1)*W+x],labels[(y-1)*W+right],labels[y*W+right],labels[(y+1)*W+right],
        labels[(y+1)*W+x],labels[(y+1)*W+left],labels[y*W+left],labels[(y-1)*W+left]];
    if(![0,2,4,6].some(i=>ring[i]===id)||![0,2,4,6].some(i=>ring[i]===old))return false;
    for(const label of [old,id]){
        let transitions=0;for(let i=0;i<8;i++)if((ring[i]===label)!==(ring[(i+1)%8]===label))transitions++;
        if(transitions!==2)return false;
    }
    // Reject checkerboard corners: a diagonal contact creates a self-touching
    // outline even when another route happens to keep the region connected.
    for(const [a,b,c] of [[0,1,2],[2,3,4],[4,5,6],[6,7,0]]){
        if(ring[b]===id&&ring[a]!==id&&ring[c]!==id)return false;
        if(ring[a]===ring[c]&&ring[a]!==id&&ring[b]!==ring[a])return false;
    }
    return true;
}

export function cleanBoundaries(data,stars,canAssign=null){
    const original=unpackGrid(data.territories),fixed=protectedCells(data,stars),target=coarseTargets(original,fixed,canAssign),labels=new Int8Array(original);
    const pending=[];for(let k=0;k<labels.length;k++)if(!fixed[k]&&target[k]!==labels[k])pending.push(k);
    let changed=0;
    for(let pass=0;pass<16;pass++){
        let count=0;
        for(let i=0;i<pending.length;i++){
            const k=pending[pass%2?pending.length-1-i:i];
            if(labels[k]===target[k]||(canAssign&&!canAssign(k,target[k]))||!mayTransfer(labels,k,target[k]))continue;
            labels[k]=target[k];count++;changed++;
        }
        if(!count)break;
    }
    // Verify the complete result, including multi-region junctions. All outlines
    // come from this one label map; neighbouring regions share the exact edge.
    const boundaries=Array.from({length:15},(_,i)=>territoryBoundary(labels,i));
    return {labels,boundaries,changed};
}
