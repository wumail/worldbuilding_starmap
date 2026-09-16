import loadHighs from './vendor/highs-1.15.3/highs.mjs';

let runtime,loading;
export async function prepareMinimumSolver(){
    if(!loading)loading=loadHighs().then(value=>runtime=value).catch(error=>{loading=null;throw error;});
    await loading;
}

// Binary x(i,k): region i owns cell k. The unclaimed label is the remainder.
// At a grid vertex, D = NW - NE - SW + SE. With checkerboards excluded,
// |D| is EXACTLY one at a turn and zero along a straight edge or in the interior.
// Continuous c >= D, c >= -D becomes |D| when sum(c) is minimized.
export function cornerProblem({labels,fixed,mask,width=360,height=180,count=15}){
    const n=width*height,lookup=Array.from({length:count},()=>new Int32Array(n).fill(-1)),cells=[],owners=[];
    if(!Number.isInteger(width)||width<3||!Number.isInteger(height)||height<3||!Number.isInteger(count)||count<1||count>15||[labels,fixed,mask].some(a=>a?.length!==n))throw Error('最少拐点模型的网格或区域数无效');
    for(let k=0;k<n;k++){
        if(!Number.isInteger(labels[k])||labels[k]<0||labels[k]>count||![0,1].includes(fixed[k]))throw Error('最少拐点模型的固定归属无效');
        if(fixed[k]&&labels[k]<count&&!(mask[k]&(1<<labels[k])))throw Error('固定星形超出允许星区，无法保证包围');
    }
    for(let i=0;i<count;i++)for(let k=0;k<n;k++)if((mask[k]&(1<<i))&&(!fixed[k]||labels[k]===i)){
        if(k<width||k>=n-width)throw Error('黄道区域不能触及参考极点');
        lookup[i][k]=cells.length;cells.push(k);owners.push(i);
    }
    const binary=cells.length,terms=[],cornerGroups=[];
    for(let i=0;i<count;i++){
        const first=terms.length;
        const vertices=new Set();
        for(let k=0;k<n;k++)if(lookup[i][k]>=0)for(const dy of [0,1])for(const dx of [0,1])vertices.add((Math.floor(k/width)+dy)*width+(k%width+dx)%width);
        for(const k of [...vertices].sort((a,b)=>a-b)){
            const x=k%width,y=Math.floor(k/width),left=(x+width-1)%width;
            terms.push([[(y-1)*width+left,1],[(y-1)*width+x,-1],[y*width+left,-1],[y*width+x,1]].filter(([q])=>q>=0&&q<n&&lookup[i][q]>=0).map(([q,v])=>[lookup[i][q],v]));
        }
        cornerGroups.push(Array.from({length:terms.length-first},(_,j)=>binary+first+j));
    }
    const size=binary+terms.length,lower=new Float64Array(size),upper=new Float64Array(size).fill(1),cost=new Float64Array(size),integrality=new Int32Array(size);
    for(let v=0;v<binary;v++){lower[v]=fixed[cells[v]]?1:0;integrality[v]=1;}
    cost.fill(1,binary);
    const starts=[0],indices=[],values=[],rowLower=[],rowUpper=[];
    const add=(entries,lo,hi)=>{for(const [c,v] of entries){indices.push(c);values.push(v);}starts.push(indices.length);rowLower.push(lo);rowUpper.push(hi);};
    for(let j=0;j<terms.length;j++){
        const d=terms[j];add(d,-1,1);add([...d,[binary+j,-1]],-Infinity,0);add([...d.map(([v,s])=>[v,-s]),[binary+j,-1]],-Infinity,0);
    }
    for(let k=0;k<n;k++){
        const entries=lookup.flatMap(a=>a[k]>=0?[[a[k],1]]:[]);if(entries.length>1)add(entries,0,1);
    }
    const roots=lookup.map((_,i)=>{for(let k=0;k<n;k++)if(labels[k]===i&&fixed[k])return k;throw Error('最少拐点求解缺少固定星形');});
    return {width,height,count,n,lookup,cells,owners,binary,size,roots,cornerGroups,cornerTerms:terms,
        model:{numCols:size,numRows:rowLower.length,colCost:cost,colLower:lower,colUpper:upper,integrality,rowLower,rowUpper,
            matrix:{format:'csr',numCols:size,numRows:rowLower.length,starts,indices,values}}};
}

function neighbours(k,p){
    const x=k%p.width,y=Math.floor(k/p.width);
    return [y*p.width+(x+p.width-1)%p.width,y*p.width+(x+1)%p.width,k-p.width,k+p.width].filter(q=>q>=0&&q<p.n);
}
function components(labels,id,inverted,p){
    const seen=new Uint8Array(p.n),groups=[];
    for(let start=0;start<p.n;start++)if(!seen[start]&&((labels[start]===id)!==inverted)){
        const group=[start];seen[start]=1;
        for(let j=0;j<group.length;j++)for(const k of neighbours(group[j],p))if(!seen[k]&&((labels[k]===id)!==inverted)){seen[k]=1;group.push(k);}
        groups.push(group);
    }
    return groups;
}

// Valid separation cuts: a selected component away from the fixed root must
// connect through its outer ring. A void enclosed by selected cells must open
// its ring or fill a void anchor. No valid connected, hole-free solution is cut.
export function topologyCuts(labels,p){
    const cuts=[];
    for(let i=0;i<p.count;i++){
        for(const group of components(labels,i,false,p)){
            if(group.includes(p.roots[i]))continue;
            const set=new Set(group),ring=new Set();
            for(const k of group)for(const q of neighbours(k,p))if(p.lookup[i][q]>=0&&!set.has(q))ring.add(q);
            const anchor=Math.min(...group),entries=[...ring].sort((a,b)=>a-b).map(k=>[p.lookup[i][k],1]);entries.push([p.lookup[i][anchor],-1]);
            cuts.push({entries,lower:0,upper:Infinity});
        }
        for(const group of components(labels,i,true,p)){
            if(group.includes(0))continue;
            const ring=new Set();for(const k of group)for(const q of neighbours(k,p))if(labels[q]===i)ring.add(q);
            const anchor=Math.min(...group),entries=[...ring].sort((a,b)=>a-b).map(k=>[p.lookup[i][k],-1]);
            if(p.lookup[i][anchor]>=0)entries.push([p.lookup[i][anchor],1]);
            cuts.push({entries,lower:1-ring.size,upper:Infinity});
        }
    }
    return cuts;
}
function decode(solution,p){
    const labels=new Int8Array(p.n).fill(p.count);
    for(let v=0;v<p.binary;v++){
        const x=solution[v];if(!Number.isFinite(x)||Math.abs(x-Math.round(x))>1e-6)throw Error('最少拐点求解未返回整数区域');
        if(x>.5){if(labels[p.cells[v]]!==p.count)throw Error('最少拐点求解出现区域重叠');labels[p.cells[v]]=p.owners[v];}
    }
    return labels;
}

export function solveMinimumCorners(problem,{seconds=45,cornersOnly=false}={}){
    if(!runtime)throw Error('最少拐点求解器尚未就绪');
    const p=cornerProblem(problem),model=runtime.createModel(p.model),started=performance.now();let cuts=0;
    try{
        model.options.set({output_flag:false,random_seed:0,mip_rel_gap:0,mip_abs_gap:0,mip_feasibility_tolerance:1e-7});
        function optimum(){
            for(let attempt=0;attempt<128;attempt++){
                const remaining=seconds-(performance.now()-started)/1000;
                if(remaining<=0)throw Error('求解尚未证明最优，当前轮次已保留。请稍后重试或换一个种子。');
                model.options.set('time_limit',remaining);model.run();
                if(model.getModelStatus()!==runtime.constants.modelStatus.optimal)throw Error('求解尚未证明最优，当前轮次已保留。请稍后重试或换一个种子。');
                const value=model.getObjectiveValue(),bound=model.info.get('mip_dual_bound'),minimum=Math.round(value);
                if(Math.abs(value-minimum)>1e-5||!Number.isFinite(bound)||bound<=minimum-1+1e-6)throw Error('求解结果缺少有效的整数最优下界');
                const labels=decode(model.getSolution().colValue,p),separation=topologyCuts(labels,p);
                if(!separation.length)return {labels,minimum};
                for(const c of separation){model.addRow(c.lower,c.upper,{indices:c.entries.map(([i])=>i),values:c.entries.map(([,v])=>v)});cuts++;}
            }
            throw Error('求解尚未完成区域拓扑证明，当前轮次已保留。');
        }
        const primary=optimum(),cornerIndices=Array.from({length:p.size-p.binary},(_,i)=>p.binary+i);
        if(cornersOnly)return {labels:primary.labels,certificate:{minimumCorners:primary.minimum,integerLowerBound:primary.minimum}};
        // Exact lexicographic objectives. Never trade an extra corner for area.
        model.addRow(primary.minimum,primary.minimum,{indices:cornerIndices,values:cornerIndices.map(()=>1)});
        for(let i=0;i<p.size;i++)model.changeColCost(i,i<p.binary?1:0);
        const secondary=optimum();
        return {labels:secondary.labels,certificate:{status:'optimal',minimumCorners:primary.minimum,integerLowerBound:primary.minimum,
            minimumCellsAtMinimumCorners:secondary.minimum,domain:{width:p.width,height:p.height,regions:p.count,edges:'orthogonal',constraints:'mask-protected-cells-connected-hole-free'},
            solver:`HiGHS ${runtime.version.string}`,wrapper:'highs-js 1.15.3'}};
    }finally{model.dispose();}
}

export function actualCornerCounts(labels,{width=360,height=180,count=15}={}){
    const result=Array(count).fill(0);
    for(let y=1;y<height;y++)for(let x=0;x<width;x++){
        const left=(x+width-1)%width,ids=[labels[(y-1)*width+left],labels[(y-1)*width+x],labels[y*width+left],labels[y*width+x]];
        for(const i of new Set(ids))if(i<count){const d=Number(ids[0]===i)-Number(ids[1]===i)-Number(ids[2]===i)+Number(ids[3]===i);if(Math.abs(d)>1)throw Error('边界对角自接触');result[i]+=Math.abs(d);}
    }
    return result;
}

function trimOuterEdges(original,problem,p){
    const labels=new Int8Array(original),counts=actualCornerCounts(labels,p),W=p.width,H=p.height;
    const at=(x,y)=>y<0||y>=H?-1:y*W+(x+W)%W;
    const canRemove=k=>{
        const x=k%W,y=Math.floor(k/W),old=labels[k],ring=[at(x,y-1),at(x+1,y-1),at(x+1,y),at(x+1,y+1),at(x,y+1),at(x-1,y+1),at(x-1,y),at(x-1,y-1)].map(q=>q<0?-1:labels[q]);
        if(problem.fixed[k]||![0,2,4,6].some(i=>ring[i]===p.count))return false;
        for(const id of [old,p.count]){
            let changes=0;for(let i=0;i<8;i++)if((ring[i]===id)!==(ring[(i+1)%8]===id))changes++;
            if(changes!==2)return false;
        }
        return true;
    };
    let passes=0;
    for(;passes<12;passes++){
        let changed=0;
        for(const [dx,dy] of [[0,-1],[0,1],[-1,0],[1,0]]){
            const rows=dy?H:W,cols=dy?W:H;
            for(let row=0;row<rows;row++){
                let line=[];
                const apply=()=>{
                    if(!line.length)return;const owner=labels[line[0]],removed=[],vertices=new Set();
                    for(const k of line){const x=k%W,y=Math.floor(k/W);for(const dx of [0,1])for(const dy of [0,1])vertices.add((y+dy)*W+(x+dx)%W);}
                    const turns=()=>{let sum=0;for(const k of vertices){const x=k%W,y=Math.floor(k/W),ids=[at(x-1,y-1),at(x,y-1),at(x-1,y),at(x,y)].map(q=>q>=0&&labels[q]===owner?1:0);const d=ids[0]-ids[1]-ids[2]+ids[3];if(Math.abs(d)>1)return Infinity;sum+=Math.abs(d);}return sum;};
                    const before=turns();
                    for(const k of line){if(!canRemove(k))break;labels[k]=p.count;removed.push(k);}
                    const same=removed.length===line.length&&turns()===before;
                    if(!same)for(const k of removed)labels[k]=owner;else changed+=removed.length;
                    line=[];
                };
                for(let col=0;col<=cols;col++){
                    const x=dy?col:row,y=dy?row:col,k=col<cols?at(x,y):-1,q=col<cols?at(x+dx,y+dy):-1;
                    if(k<0||q<0||labels[k]>=p.count||labels[q]!==p.count||(line.length&&labels[line[0]]!==labels[k]))apply();
                    if(k>=0&&q>=0&&labels[k]<p.count&&labels[q]===p.count)line.push(k);
                }
            }
        }
        if(!changed)break;
    }
    if(topologyCuts(labels,p).length||actualCornerCounts(labels,p).some((n,i)=>n!==counts[i]))throw Error('收紧外缘未通过拓扑或拐点检查');
    return {labels,passes,ownedCells:labels.filter(i=>i<p.count).length};
}

// First establish each region's independent lower bound with all protected
// ownership intact. Then lexicographically minimize the descending vector of
// excess corners, so the second-worst region cannot be sacrificed for a sum.
export function solveLocalFirstCorners(problem,{seconds=90,onProgress=()=>{}}={}){
    if(!runtime)throw Error('边界求解器尚未就绪');
    const started=performance.now(),width=problem.width??360,height=problem.height??180,count=problem.count??15;
    const remaining=()=>seconds-(performance.now()-started)/1000,localMinima=[];
    for(let i=0;i<count;i++){
        onProgress(`正在整理第 ${i+1} / ${count} 座的局部边界…`);
        const labels=Int8Array.from(problem.labels,id=>id===i?0:1),mask=Uint16Array.from(problem.mask,m=>m&(1<<i)?1:0);
        const solved=solveMinimumCorners({width,height,count:1,labels,mask,fixed:problem.fixed},{seconds:remaining(),cornersOnly:true});
        localMinima.push(solved.certificate.minimumCorners);
    }
    onProgress('正在检验各区最简边界能否同时成立…');
    const joint=solveMinimumCorners(problem,{seconds:remaining(),cornersOnly:true});
    const p=cornerProblem(problem),model=runtime.createModel(p.model);let columns=p.size;
    const resetCosts=()=>{for(let i=0;i<columns;i++)model.changeColCost(i,0);};
    const addColumn=(cost=0)=>{const index=columns++;model.addCol(cost,0,Infinity,{indices:[],values:[]});return index;};
    const stage=()=>{
        for(let attempt=0;attempt<128;attempt++){
            if(remaining()<=0)throw Error('局部与整体边界协调尚未完成，当前轮次已保留');
            model.options.set('time_limit',remaining());model.run();
            if(model.getModelStatus()!==runtime.constants.modelStatus.optimal)throw Error('局部与整体边界协调尚未证明最优，当前轮次已保留');
            const value=model.getObjectiveValue(),minimum=Math.round(value),bound=model.info.get('mip_dual_bound');
            if(Math.abs(value-minimum)>1e-5||!Number.isFinite(bound)||bound<=minimum-1+1e-6)throw Error('边界协调缺少有效最优下界');
            const labels=decode(model.getSolution().colValue,p),cuts=topologyCuts(labels,p);
            if(!cuts.length)return {labels,minimum};
            for(const c of cuts)model.addRow(c.lower,c.upper,{indices:c.entries.map(([i])=>i),values:c.entries.map(([,v])=>v)});
        }
        throw Error('边界协调未通过拓扑检查，当前轮次已保留');
    };
    try{
        model.options.set({output_flag:false,random_seed:0,mip_rel_gap:0,mip_abs_gap:0,mip_feasibility_tolerance:1e-7});
        const localSum=localMinima.reduce((a,b)=>a+b,0),allLocalAttainable=joint.certificate.minimumCorners===localSum;
        const topSums=allLocalAttainable?Array(count-1).fill(0):[];
        for(let k=1;!allLocalAttainable&&k<count;k++){
            onProgress(`正在协调公共边界：优先减轻第 ${k} 处局部负担…`);resetCosts();
            const t=addColumn(k),z=Array.from({length:count},()=>addColumn(1));
            for(let i=0;i<count;i++)model.addRow(-localMinima[i],Infinity,
                {indices:[t,z[i],...p.cornerGroups[i]],values:[1,1,...p.cornerGroups[i].map(()=>-1)]});
            const result=stage(),minimum=result.minimum;
            model.addRow(-Infinity,minimum,{indices:[t,...z],values:[k,...z.map(()=>1)]});
            topSums.push(minimum);
            // A zero next excess means all later excesses are also zero; their
            // top-k sums are already fixed. Avoid redundant identical solves.
            if(minimum===(topSums.at(-2)??0)){while(topSums.length<count-1)topSums.push(minimum);break;}
        }
        resetCosts();const corners=p.cornerGroups.flat();for(const i of corners)model.changeColCost(i,1);
        const total=allLocalAttainable?{labels:joint.labels,minimum:joint.certificate.minimumCorners}:stage();topSums.push(total.minimum-localSum);
        onProgress('正在收紧整套边界…');
        const compact=trimOuterEdges(total.labels,problem,p),counts=actualCornerCounts(compact.labels,p),excess=counts.map((v,i)=>v-localMinima[i]),sorted=[...excess].sort((a,b)=>b-a);
        let sum=0;for(let i=0;i<count;i++){sum+=sorted[i];if(excess[i]<0||sum!==topSums[i])throw Error('实际边界与局部优先证明不一致');}
        if(counts.reduce((a,b)=>a+b,0)!==total.minimum)throw Error('实际总拐点与协调证明不一致');
        return {labels:compact.labels,certificate:{status:'optimal',method:'local-excess-lexicographic-1',localMinima,regionCorners:counts,excessCorners:excess,
            sortedExcessCorners:sorted,topExcessIntegerLowerBounds:topSums,minimumTotalAtLocalPriority:total.minimum,
            compactness:{status:'locally-trimmed',method:'fixed-order-inward-edge-trim-1',ownedCells:compact.ownedCells,passes:compact.passes},
            domain:{width,height,regions:count,edges:'orthogonal',constraints:'mask-protected-cells-connected-hole-free'},solver:`HiGHS ${runtime.version.string}`,wrapper:'highs-js 1.15.3'}};
    }finally{model.dispose();}
}
