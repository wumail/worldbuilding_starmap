import {RAD, vector, coordinates, unit, dot, cross, separation, owner, cellPolygon, eclipticIntervals, tangentFrame, projectLocal} from './geometry.mjs?revision=candidate-editor-band-1';

// These are cultural drawing constraints, not thresholds derived from gravity
// or a calibrated human-vision experiment. The source catalogue is immutable.
export const ALGORITHM = 'terrax-zodiac-draw-1';
export const RULES = Object.freeze({count:15, candidateMagnitude:4.5, preferredMagnitude:4, searchLatitude:30, siteLatitude:8, displayScale:2.5, maxEdge:13, minSeparation:.8, maxDiameter:42, maxDegree:4, lineClearance:.18, outerSiteLatitude:50, boundaryClearance:2});
export const DEFAULT_SEED = 'terrax-001';
const brief = s => ({id:s.id, app_mag:s.app_mag, color_hex:s.color_hex, distance_pc:s.distance_pc, longitude:s.longitude, latitude:s.latitude, direction:[...s.direction]});

// JavaScript engines can differ in the last bits of trigonometric results.
// Only derived angles/vectors have a tolerance; identity and photometry remain exact.
export function equivalentDraw(a,b,key='',tolerance=0) {
    if(['longitude','latitude','degrees','eclipticSpan','start','end'].includes(key))tolerance=1e-9;
    if(['direction','site','polygon','remainderSites'].includes(key))tolerance=1e-11;
    if(typeof a==='number'||typeof b==='number')return typeof a==='number'&&typeof b==='number'&&Number.isFinite(a)&&Number.isFinite(b)&&Math.abs(a-b)<=tolerance;
    if(a===b)return true;
    if(!a||!b||typeof a!=='object'||typeof b!=='object'||Array.isArray(a)!==Array.isArray(b))return false;
    const keys=Object.keys(a);return keys.length===Object.keys(b).length&&keys.every(k=>Object.hasOwn(b,k)&&equivalentDraw(a[k],b[k],k,tolerance));
}

export function randomFrom(seed) {
    let h=2166136261;
    for(const char of String(seed)){h^=char.codePointAt(0);h=Math.imul(h,16777619);}
    return ()=>{h+=0x6D2B79F5;let t=h;t=Math.imul(t^t>>>15,t|1);t^=t+Math.imul(t^t>>>7,t|61);return ((t^t>>>14)>>>0)/4294967296;};
}

export function normalizeRecipe(value={}) {
    const seed=String(value.seed??DEFAULT_SEED).trim();
    if(!seed||seed.length>80)throw Error('种子需要 1–80 个字符。');
    if(value.algorithm && value.algorithm!==ALGORITHM)throw Error('这份留存使用了不同的抽卡算法，请保留原文件并使用对应版本查看。');
    const style=value.style??'rich';
    if(!['balanced','rich'].includes(style))throw Error('未知的星形复杂度。');
    if(value.shapeSeeds && (!Array.isArray(value.shapeSeeds)||value.shapeSeeds.length!==15||value.shapeSeeds.some(s=>typeof s!=='string'||!s.length||s.length>120)))throw Error('星形种子不完整。');
    return {algorithm:ALGORITHM, seed, style, shapeSeeds:value.shapeSeeds?[...value.shapeSeeds]:Array.from({length:15},(_,i)=>`${seed}/shape/${i}`)};
}

export function redrawRecipe(recipe, locked, nonce) {
    const next=normalizeRecipe(recipe),protectedRegions=new Set(locked);
    if([...protectedRegions].some(i=>!Number.isInteger(i)||i<0||i>=15))throw Error('锁定的区域编号无效。');
    next.shapeSeeds=next.shapeSeeds.map((s,i)=>protectedRegions.has(i)?s:`${String(nonce).slice(0,90)}/shape/${i}`);
    return next;
}

// Gnomonic projection maps each minor great-circle edge to a straight segment.
// It is used only for graph intersections; displayed charts stay stereographic.
function graphPlane(stars) {
    const center=coordinates(unit(stars.reduce((sum,s)=>sum.map((x,i)=>x+s.direction[i]),[0,0,0]))),frame=tangentFrame(center.longitude,center.latitude);
    return new Map(stars.map(s=>[s.id,{x:dot(s.direction,frame.east)/dot(s.direction,frame.center)/RAD,y:dot(s.direction,frame.north)/dot(s.direction,frame.center)/RAD}]));
}
const turn=(a,b,c)=>(b.x-a.x)*(c.y-a.y)-(b.y-a.y)*(c.x-a.x);
function crossing(a,b,points) {
    if([a.from,a.to].some(id=>id===b.from||id===b.to))return false;
    const p=points.get(a.from),q=points.get(a.to),r=points.get(b.from),s=points.get(b.to);
    return turn(p,q,r)*turn(p,q,s)<-1e-10 && turn(r,s,p)*turn(r,s,q)<-1e-10;
}
function arcDistance(v,a,b) {
    const n=unit(cross(a,b)),p=unit(v.map((x,i)=>x-dot(v,n)*n[i]));
    if(Math.abs(separation(a,p)+separation(p,b)-separation(a,b))<1e-7)return separation(v,p);
    return Math.min(separation(v,a),separation(v,b));
}
function pathBetween(from,to,edges) {
    const queue=[[from]],seen=new Set([from]);
    for(let i=0;i<queue.length;i++){
        const path=queue[i],id=path.at(-1);if(id===to)return path;
        for(const e of edges){const next=e.from===id?e.to:e.to===id?e.from:null;if(next&&!seen.has(next)){seen.add(next);queue.push([...path,next]);}}
    }
    return null;
}
function graphStats(members,edges) {
    const degrees=new Map(members.map(s=>[s.id,0]));
    for(const e of edges){degrees.set(e.from,degrees.get(e.from)+1);degrees.set(e.to,degrees.get(e.to)+1);}
    return {loops:edges.length-members.length+1, branches:[...degrees.values()].filter(n=>n>=3).length, tips:[...degrees.values()].filter(n=>n===1).length};
}

function makeGraph(stars,rng,style) {
    const points=graphPlane(stars),options=[];
    for(let i=0;i<stars.length;i++)for(let j=i+1;j<stars.length;j++){
        const a=stars[i],b=stars[j],degrees=separation(a.direction,b.direction);
        if(degrees>RULES.maxEdge||stars.some((s,k)=>k!==i&&k!==j&&arcDistance(s.direction,a.direction,b.direction)<RULES.lineClearance))continue;
        options.push({from:a.id,to:b.id,degrees,cost:degrees*(.85+rng()*.3)});
    }
    options.sort((a,b)=>a.cost-b.cost||a.from.localeCompare(b.from)||a.to.localeCompare(b.to));
    const edges=[],degrees=new Map(stars.map(s=>[s.id,0]));
    const canAdd=e=>degrees.get(e.from)<RULES.maxDegree&&degrees.get(e.to)<RULES.maxDegree&&!edges.some(other=>crossing(e,other,points));
    const add=e=>{edges.push({from:e.from,to:e.to,degrees:e.degrees});degrees.set(e.from,degrees.get(e.from)+1);degrees.set(e.to,degrees.get(e.to)+1);};
    for(const e of options)if(canAdd(e)&&!pathBetween(e.from,e.to,edges))add(e);
    if(edges.length!==stars.length-1)return null;
    const tree=edges.map(e=>({...e})),target=style==='rich'?1+Math.floor(rng()*3):Math.floor(rng()*3);
    for(let loop=0;loop<target;loop++){
        const choices=[];
        for(const e of options){
            if(!canAdd(e))continue;
            const path=pathBetween(e.from,e.to,edges);
            if(path.length<4||path.length>7)continue; // Avoid a dense triangular mesh.
            const poly=path.map(id=>points.get(id));let area=0,perimeter=0;
            for(let i=0;i<poly.length;i++){const a=poly[i],b=poly[(i+1)%poly.length];area+=a.x*b.y-a.y*b.x;perimeter+=Math.hypot(a.x-b.x,a.y-b.y);}
            const compactness=Math.abs(area)/2/(perimeter*perimeter);
            if(compactness<.018)continue; // Reject almost-collinear, decorative slivers.
            choices.push({e,score:e.degrees*.05-compactness*3+rng()*.7});
        }
        choices.sort((a,b)=>a.score-b.score);if(!choices.length)break;add(choices[0].e);
    }
    // The brighter skeleton is a connected subtree of the full figure.
    let core=[...stars],coreEdges=tree;
    while(core.length>7){
        const leaves=core.filter(s=>coreEdges.filter(e=>e.from===s.id||e.to===s.id).length===1).sort((a,b)=>b.app_mag-a.app_mag);
        const remove=leaves[0];core=core.filter(s=>s!==remove);coreEdges=coreEdges.filter(e=>e.from!==remove.id&&e.to!==remove.id);
    }
    return {core,coreEdges,edges,structure:graphStats(stars,edges)};
}

function chooseFigure(pool,seed,style) {
    const sorted=[...pool].sort((a,b)=>a.app_mag-b.app_mag||a.id.localeCompare(b.id)),rng=randomFrom(seed),minimum=style==='rich'?10:8;
    for(let attempt=0;attempt<36;attempt++){
        const goal=Math.min(sorted.length,minimum+Math.floor(rng()*(style==='rich'?6:4))),anchors=sorted.filter(s=>s.app_mag<=3.5).slice(0,5);
        if(!anchors.length)return null;
        const chosen=[anchors[Math.floor(rng()*anchors.length)]];
        while(chosen.length<goal){
            const options=sorted.filter(s=>!chosen.includes(s)).map(s=>{
                const gaps=chosen.map(p=>separation(s.direction,p.direction)),near=Math.min(...gaps),far=Math.max(...gaps);
                return {s,near,far,score:s.app_mag*.85+Math.max(0,s.app_mag-4)*1.2+near*.1+Math.max(0,far-30)*.1+rng()*.95};
            }).filter(x=>x.near>=RULES.minSeparation&&x.near<=RULES.maxEdge&&x.far<=RULES.maxDiameter).sort((a,b)=>a.score-b.score||a.s.id.localeCompare(b.s.id));
            if(!options.length)break;chosen.push(options[0].s);
        }
        if(chosen.length<minimum)continue;
        const center=coordinates(unit(chosen.reduce((sum,s)=>sum.map((x,i)=>x+s.direction[i]),[0,0,0]))),frame=tangentFrame(center.longitude,center.latitude);
        if(chosen.some(s=>{const p=projectLocal(s.direction,frame,0,0,1);return !p.visible||Math.abs(p.x)>26||Math.abs(p.y)>23;}))continue;
        const graph=makeGraph(chosen,rng,style);if(graph)return {members:chosen,center,...graph};
    }
    return null;
}

function partition(pool,seed) {
    const rng=randomFrom(`${seed}/partition`);
    for(let attempt=0;attempt<24;attempt++){
        const phase=rng()*24,weights=pool.map(s=>(5-s.app_mag)**1.5*(.65+rng()*.7));
        let centers=Array.from({length:15},(_,i)=>vector(i*24+phase+(rng()-.5)*10,(rng()-.5)*10));
        for(let iteration=0;iteration<75;iteration++){
            const sums=centers.map(()=>[0,0,0]);
            pool.forEach((s,j)=>{const i=owner(s.direction,centers);for(let k=0;k<3;k++)sums[i][k]+=s.direction[k]*weights[j];});
            const next=sums.map((sum,i)=>{if(Math.hypot(...sum)<1e-12)return centers[i];const c=coordinates(sum);return vector(c.longitude,Math.max(-8,Math.min(8,c.latitude)));});
            const change=Math.max(...centers.map((c,i)=>1-dot(c,next[i])));centers=next;if(change<1e-11)break;
        }
        centers.sort((a,b)=>coordinates(a).longitude-coordinates(b).longitude);
        const groups=centers.map(()=>[]);for(const s of pool)groups[owner(s.direction,centers)].push(s);
        if(groups.some(g=>g.length<12||g.filter(s=>s.app_mag<=4).length<7))continue;
        const intervals=eclipticIntervals(centers),spans=centers.map((_,i)=>intervals.filter(s=>s.index===i).reduce((n,s)=>n+s.end-s.start,0));
        if(spans.some(s=>s<12||s>40))continue;
        // Feasibility is independent of shape seeds: a local reroll must never
        // secretly change the partition or a locked neighbour's boundary.
        if(groups.some((g,i)=>!chooseFigure(g,`${seed}/feasibility/${attempt}/${i}`,'rich')))continue;
        const remainderSites=[];
        for(const sign of [-1,1])for(let longitude=12;longitude<360;longitude+=24){
            let site;
            for(let latitude=RULES.outerSiteLatitude;latitude<=90;latitude++){
                site=vector(longitude,sign*latitude);
                if(groups.every((g,i)=>g.every(s=>separation(s.direction,site)>=separation(s.direction,centers[i])+RULES.boundaryClearance)))break;
            }
            remainderSites.push(site);
        }
        const all=[...centers,...remainderSites];
        if(groups.some((g,i)=>g.some(s=>owner(s.direction,all)!==i)))continue;
        if(eclipticIntervals(all).some(i=>i.index>=15))continue;
        return {centers,remainderSites,groups,attempt};
    }
    throw Error('此种子没有找到满足亮星与连线约束的十五区。请换一个种子；没有放宽星等或改变星表。');
}

export function generateDraw(source,meta,value={}) {
    const recipe=normalizeRecipe(value),stars=source.map(brief),pool=stars.filter(s=>s.app_mag<=RULES.candidateMagnitude&&Math.abs(s.latitude)<=RULES.searchLatitude);
    const p=partition(pool,recipe.seed),sites=[...p.centers,...p.remainderSites],intervals=eclipticIntervals(sites);
    const regions=p.groups.map((group,i)=>{
        const f=chooseFigure(group,recipe.shapeSeeds[i],recipe.style);
        if(!f)throw Error(`候选 ${i+1} 暂未找到满足约束的星形，请重抽。本轮结果未替换当前方案。`);
        const sections=intervals.filter(x=>x.index===i).map(({start,end})=>({start,end}));
        return {id:`Z${String(i+1).padStart(2,'0')}`,label:`候选 ${String(i+1).padStart(2,'0')}`,site:p.centers[i],center:f.center,
            polygon:cellPolygon(i,sites),intervals:sections,eclipticSpan:sections.reduce((n,s)=>n+s.end-s.start,0),candidateCount:group.length,brightCount:group.filter(s=>s.app_mag<=4).length,
            members:f.members,variants:[{id:'core',label:'亮星骨架',members:f.core.map(s=>s.id),edges:f.coreEdges},{id:'extended',label:'完整星形',members:f.members.map(s=>s.id),edges:f.edges}],
            structure:f.structure,brightest:Math.min(...f.members.map(s=>s.app_mag)),faintest:Math.max(...f.members.map(s=>s.app_mag)),oldSectors:[...new Set(f.members.map(s=>Math.floor(s.longitude/24)+1))].sort((a,b)=>a-b)};
    });
    const result={schema:2,status:'candidate-review',catalogue:meta.catalogue,sha256:meta.sha256,epoch:'Terrax 第 0 日参考黄道',recipe,settings:RULES,sourceCount:stars.length,candidateCount:pool.length,
        selectedCoreCount:regions.reduce((n,r)=>n+r.variants[0].members.length,0),selectedExtendedCount:regions.reduce((n,r)=>n+r.members.length,0),remainderSites:p.remainderSites,regions};
    validateDraw(result,stars);
    return result;
}

// The checks run on every draw, including imported recipes and local rerolls.
export function validateDraw(data,stars) {
    const fail=message=>{throw Error(`候选检查未通过：${message}`);},byId=new Map(stars.map(s=>[s.id,s])),sites=[...data.regions.map(r=>r.site),...data.remainderSites],used=new Set();
    if(data.regions.length!==15)fail('区域数量');
    const recipe=normalizeRecipe(data.recipe),minimum=recipe.style==='rich'?10:8,maximum=recipe.style==='rich'?15:11;
    if(data.sourceCount!==stars.length||data.candidateCount!==stars.filter(s=>s.app_mag<=RULES.candidateMagnitude&&Math.abs(s.latitude)<=RULES.searchLatitude).length)fail('星表计数');
    if(sites.some(s=>!Array.isArray(s)||s.length!==3||s.some(n=>!Number.isFinite(n))||Math.abs(Math.hypot(...s)-1)>1e-9))fail('区域中心');
    const exactIntervals=eclipticIntervals(sites);
    for(const [i,r] of data.regions.entries()){
        if(r.id!==`Z${String(i+1).padStart(2,'0')}`||r.members.length<minimum||r.members.length>maximum)fail('区域编号或成员数量');
        if(r.variants.length!==2||r.variants[0].id!=='core'||r.variants[1].id!=='extended')fail('缺少骨架或完整星形');
        const points=graphPlane(r.members),members=new Map(r.members.map(s=>[s.id,s]));
        const full=r.variants[1].members;
        if(full.length!==r.members.length||new Set(full).size!==full.length||full.some(id=>!members.has(id)))fail('完整星形成员不一致');
        if(r.variants[0].members.length!==7)fail('亮星骨架数量');
        for(const s of r.members){
            const original=byId.get(s.id);
            if(!original||!equivalentDraw(brief(original),s)||used.has(s.id))fail('成员不是唯一的原始恒星');
            used.add(s.id);
            if(s.app_mag>RULES.candidateMagnitude||Math.abs(s.latitude)>RULES.searchLatitude||owner(s.direction,sites)!==i)fail('成员亮度或区域归属');
        }
        for(let a=0;a<r.members.length;a++)for(let b=a+1;b<r.members.length;b++){
            const d=separation(r.members[a].direction,r.members[b].direction);
            if(d<RULES.minSeparation-1e-9||d>RULES.maxDiameter+1e-9)fail('图形角尺度');
        }
        for(const v of r.variants){
            const degrees=new Map(v.members.map(id=>[id,0]));
            if(degrees.size!==v.members.length||v.members.some(id=>!members.has(id)))fail('连线方案成员重复或不存在');
            const edgeKeys=v.edges.map(e=>[e.from,e.to].sort().join('/'));
            if(new Set(edgeKeys).size!==edgeKeys.length)fail('重复连线');
            for(const [j,e] of v.edges.entries()){
                if(!degrees.has(e.from)||!degrees.has(e.to)||e.from===e.to)fail('连线端点');
                const a=members.get(e.from),b=members.get(e.to),d=separation(a.direction,b.direction);
                if(d>RULES.maxEdge+1e-9||Math.abs(d-e.degrees)>1e-9)fail('连线长度');
                if(v.edges.slice(j+1).some(f=>crossing(e,f,points)))fail('连线交叉');
                if(r.members.some(s=>s.id!==e.from&&s.id!==e.to&&arcDistance(s.direction,a.direction,b.direction)<RULES.lineClearance-1e-9))fail('连线穿过其他成员');
                degrees.set(e.from,degrees.get(e.from)+1);degrees.set(e.to,degrees.get(e.to)+1);
            }
            if([...degrees.values()].some(n=>n>RULES.maxDegree))fail('连接过密');
            if(v.members.some(id=>!pathBetween(v.members[0],id,v.edges)))fail('星形不连通');
        }
        const expected=cellPolygon(i,sites);
        if(r.polygon.length<3||r.polygon.length!==expected.length||r.polygon.some((v,j)=>v.some(n=>!Number.isFinite(n))||Math.abs(Math.hypot(...v)-1)>1e-9||separation(v,expected[j])>1e-7))fail('区域边界');
        const sections=exactIntervals.filter(s=>s.index===i);
        if(r.intervals.length!==sections.length||r.intervals.some((s,j)=>Math.abs(s.start-sections[j].start)>1e-9||Math.abs(s.end-sections[j].end)>1e-9)||Math.abs(r.eclipticSpan-sections.reduce((n,s)=>n+s.end-s.start,0))>1e-9)fail('黄道区间归属');
        const frame=tangentFrame(r.center.longitude,r.center.latitude);
        if(r.members.some(s=>{const p=projectLocal(s.direction,frame,0,0,1);return !p.visible||Math.abs(p.x)>26||Math.abs(p.y)>23;}))fail('局部图裁剪');
    }
    const intervals=data.regions.flatMap(r=>r.intervals).sort((a,b)=>a.start-b.start);
    if(intervals[0]?.start!==0||intervals.at(-1)?.end!==360||data.regions.some(r=>r.eclipticSpan<=0))fail('黄道未覆盖全周');
    for(let i=1;i<intervals.length;i++)if(Math.abs(intervals[i].start-intervals[i-1].end)>1e-8)fail('黄道区域交叠或遗漏');
    if(data.selectedExtendedCount!==used.size||data.selectedCoreCount!==data.regions.reduce((n,r)=>n+r.variants[0].members.length,0))fail('成员总数');
    return true;
}
