import {vector,coordinates,unit,dot,separation,tangentFrame,projectLocal} from './geometry.mjs?revision=candidate-editor-band-1';
import {randomFrom,equivalentDraw,normalizeRecipe as normalizeOrdered} from './generator_ordered.mjs';
import {chooseFigure,RULES,graphPlane,crossing,arcDistance,pathBetween} from './figure.mjs';
import {growTerritories,packGrid,unpackGrid,cellOf,arcCells,territoryBoundary,territoryIntervals,fromEquatorial} from './territories.mjs';
export {RULES};
export const ALGORITHM='terrax-zodiac-draw-3';
export const DEFAULT_SEED='terrax-001';
const brief=s=>({id:s.id,app_mag:s.app_mag,color_hex:s.color_hex,distance_pc:s.distance_pc,longitude:s.longitude,latitude:s.latitude,direction:[...s.direction]});
export function normalizeRecipe(value={}){
    if(value.algorithm&&value.algorithm!==ALGORITHM)throw Error('这份留存使用了不同的抽卡算法，请使用对应版本查看。');
    return {...normalizeOrdered({...value,algorithm:'terrax-zodiac-draw-2'}),algorithm:ALGORITHM};
}

// Provisional groups are discovered in TWO dimensions, before any boundary or
// ecliptic interval exists. Random orientations explore different groupings;
// the weights never modify a star's position or magnitude.
export function findGroups(pool,rng,starWeight=null){
    const phase=rng()*24;
    let centers=Array.from({length:15},(_,i)=>vector(i*24+phase+(rng()-.5)*14,(rng()-.5)*36));
    const angles=centers.map(()=>rng()*Math.PI),stretch=centers.map(()=>1.15+rng()*.9);
    const weights=pool.map(s=>(5-s.app_mag)**1.5*(.7+rng()*.6)*(starWeight?starWeight(s):1));
    let groups;
    for(let iteration=0;iteration<36;iteration++){
        const frames=centers.map((v,i)=>{const c=coordinates(v);return {...tangentFrame(c.longitude,c.latitude),cos:Math.cos(angles[i]),sin:Math.sin(angles[i]),stretch:stretch[i]};});
        groups=centers.map(()=>[]);
        for(let j=0;j<pool.length;j++){
            const s=pool[j];let best=0,score=Infinity;
            for(let i=0;i<15;i++){
                const f=frames[i],z=dot(s.direction,f.center);if(z<.5)continue;
                const x=dot(s.direction,f.east)/z,y=dot(s.direction,f.north)/z;
                const a=x*f.cos+y*f.sin,b=-x*f.sin+y*f.cos,cost=a*a/f.stretch+b*b*f.stretch;
                if(cost<score){score=cost;best=i;}
            }
            groups[best].push({s,w:weights[j]});
        }
        if(groups.some(g=>!g.length))return null;
        const next=groups.map(g=>unit(g.reduce((sum,{s,w})=>sum.map((v,k)=>v+s.direction[k]*w),[0,0,0])));
        const change=Math.max(...next.map((v,i)=>separation(v,centers[i])));centers=next;
        if(change<.015)break;
    }
    return groups.map(g=>g.map(x=>x.s)).sort((a,b)=>coordinates(unit(a.reduce((v,s)=>v.map((x,i)=>x+s.direction[i]),[0,0,0]))).longitude-coordinates(unit(b.reduce((v,s)=>v.map((x,i)=>x+s.direction[i]),[0,0,0]))).longitude);
}

function partition(pool,seed){
    const rng=randomFrom(`${seed}/figures-first`),failures={};
    const miss=s=>{failures[s]=(failures[s]??0)+1;};
    for(let attempt=0;attempt<72;attempt++){
        const groups=findGroups(pool,rng);
        if(!groups||groups.some(g=>g.length<12||g.filter(s=>s.app_mag<=4).length<7)){miss('pool');continue;}
        const figures=groups.map((g,i)=>chooseFigure(g,`${seed}/shape/${i}`,'rich'));
        if(figures.some(f=>!f)){miss('figure');continue;}
        const outer=[];
        for(const sign of [-1,1])for(let longitude=0;longitude<360;longitude+=18)outer.push(vector(longitude+rng()*14,sign*(35+rng()*19)));
        outer.push(vector(0,90),vector(0,-90));
        let cells,sections,spans;const bias=Array(16).fill(0);
        // Neighbouring boundaries negotiate room for all fifteen ecliptic
        // crossings. Protected stars and arcs cannot be stolen by this step.
        for(let negotiation=0;negotiation<12;negotiation++){
            cells=growTerritories(figures,outer,bias);if(!cells)break;
            sections=territoryIntervals(cells);spans=figures.map((_,i)=>sections.filter(s=>s.index===i).reduce((n,s)=>n+s.end-s.start,0));
            if(sections.every(s=>s.index<15)&&spans.every(s=>s>=4&&s<=65))break;
            for(let i=0;i<15;i++){if(spans[i]<4)bias[i]+=1.8;else if(spans[i]>65)bias[i]-=1.8;if(sections.some(s=>s.index>=15))bias[i]+=.8;}
        }
        if(!cells){miss('collision');continue;}
        if(sections.some(s=>s.index>=15)||spans.some(s=>s<4||s>65)){miss('ecliptic');continue;}
        let boundaries;
        try{boundaries=figures.map((_,i)=>territoryBoundary(cells,i));}catch{miss('topology');continue;}
        const pools=figures.map(()=>[]);for(const s of pool){const i=cells[cellOf(s.direction)];if(i<15)pools[i].push(s);}
        return {figures,pools,cells,sections,boundaries,attempt};
    }
    throw Error('此种子暂未找到十五组兼顾星形、边界与黄道覆盖的方案，请换一个种子。',{cause:failures});
}

export function generateDraw(source,meta,value={}){
    const recipe=normalizeRecipe(value),stars=source.map(brief),pool=stars.filter(s=>s.app_mag<=RULES.candidateMagnitude&&Math.abs(s.latitude)<=RULES.searchLatitude),p=partition(pool,recipe.seed);
    const regions=p.pools.map((group,i)=>{
        const acceptArc=(a,b)=>arcCells(a,b).every(k=>p.cells[k]===i);
        const f=recipe.style==='rich'&&recipe.shapeSeeds[i]===`${recipe.seed}/shape/${i}`?p.figures[i]:chooseFigure(group,recipe.shapeSeeds[i],recipe.style,acceptArc);
        if(!f)throw Error(`候选 ${i+1} 暂未找到满足约束的星形，请重抽。本轮结果未替换当前方案。`);
        const sections=p.sections.filter(s=>s.index===i).map(({start,end})=>({start,end}));
        return {id:`Z${String(i+1).padStart(2,'0')}`,label:`候选 ${String(i+1).padStart(2,'0')}`,site:vector(p.figures[i].center.longitude,p.figures[i].center.latitude),center:f.center,
            boundary:p.boundaries[i],polygon:p.boundaries[i].map(([ra,dec])=>fromEquatorial(vector(ra,dec))),intervals:sections,eclipticSpan:sections.reduce((n,s)=>n+s.end-s.start,0),candidateCount:group.length,brightCount:group.filter(s=>s.app_mag<=4).length,
            members:f.members,variants:[{id:'core',label:'亮星骨架',members:f.core.map(s=>s.id),edges:f.coreEdges},{id:'extended',label:'完整星形',members:f.members.map(s=>s.id),edges:f.edges}],
            structure:f.structure,brightest:Math.min(...f.members.map(s=>s.app_mag)),faintest:Math.max(...f.members.map(s=>s.app_mag)),oldSectors:[...new Set(f.members.map(s=>Math.floor(s.longitude/24)+1))].sort((a,b)=>a-b)};
    });
    const result={schema:4,status:'candidate-review',catalogue:meta.catalogue,sha256:meta.sha256,epoch:'Terrax 第 0 日参考黄道',recipe,settings:RULES,sourceCount:stars.length,candidateCount:pool.length,eligibleCandidateCount:p.pools.reduce((n,g)=>n+g.length,0),
        selectedCoreCount:105,selectedExtendedCount:regions.reduce((n,r)=>n+r.members.length,0),territories:packGrid(p.cells),remainderSites:[],regions};
    validateDraw(result,stars);return result;
}

export function validateDraw(data,stars){
    return validateGeometry(data,stars,{recipe:normalizeRecipe(data.recipe)});
}

// Shared geometry contract; historical wrappers retain their original limits.
export function validateGeometry(data,stars,{recipe,rules=RULES,adaptiveCore=false,maximumMembers=null,allowDisconnected=false}={}){
    const fail=m=>{throw Error(`候选检查未通过：${m}`);};
    if(!Array.isArray(data.regions)||data.regions.length!==15)fail('区域数量');
    let cells;try{cells=unpackGrid(data.territories,true);}catch(e){fail(e.message);}
    const byId=new Map(stars.map(s=>[s.id,s])),used=new Set(),minimum=recipe.style==='rich'?10:8,maximum=maximumMembers??(recipe.style==='rich'?15:11);
    const pool=stars.filter(s=>s.app_mag<=rules.candidateMagnitude&&Math.abs(s.latitude)<=rules.searchLatitude),pools=Array.from({length:15},()=>[]);
    for(const s of pool){const i=cells[cellOf(s.direction)];if(i<15)pools[i].push(s);}
    if(data.sourceCount!==stars.length||data.candidateCount!==pool.length||data.eligibleCandidateCount!==pools.reduce((n,g)=>n+g.length,0))fail('星表计数');
    const intervals=territoryIntervals(cells);
    if(intervals.some(s=>s.index>=15))fail('黄道未覆盖全周');
    for(let i=0;i<15;i++){
        const r=data.regions[i];
        if(r.id!==`Z${String(i+1).padStart(2,'0')}`||r.members.length<minimum||r.members.length>maximum)fail('成员数量或区域编号');
        if(!Array.isArray(r.site)||r.site.length!==3||r.site.some(x=>!Number.isFinite(x))||Math.abs(Math.hypot(...r.site)-1)>1e-9)fail('区域参考方向');
        if(r.candidateCount!==pools[i].length||r.brightCount!==pools[i].filter(s=>s.app_mag<=4).length)fail('选星池');
        let boundary;try{boundary=territoryBoundary(cells,i);}catch(e){fail(e.message);}
        if(!equivalentDraw(boundary,r.boundary)||!equivalentDraw(boundary.map(([ra,dec])=>fromEquatorial(vector(ra,dec))),r.polygon,'polygon'))fail('区域边界');
        const sections=intervals.filter(s=>s.index===i).map(({start,end})=>({start,end}));
        if(!equivalentDraw(sections,r.intervals)||Math.abs(r.eclipticSpan-sections.reduce((n,s)=>n+s.end-s.start,0))>1e-9||r.eclipticSpan<4-1e-9||r.eclipticSpan>65+1e-9)fail('黄道区间归属');
        if(r.variants.length!==2||r.variants[0].id!=='core'||r.variants[1].id!=='extended'||(adaptiveCore?r.variants[0].members.length<7:r.variants[0].members.length!==7))fail('缺少骨架或完整星形');
        const members=new Map(r.members.map(s=>[s.id,s])),points=graphPlane(r.members);
        if(r.variants[1].members.length!==r.members.length||r.variants[1].members.some(id=>!members.has(id)))fail('完整成员不一致');
        for(const s of r.members){
            if(!byId.has(s.id)||!equivalentDraw(brief(byId.get(s.id)),s)||used.has(s.id))fail('成员不是唯一的原始恒星');
            used.add(s.id);if(s.app_mag>rules.candidateMagnitude||Math.abs(s.latitude)>rules.searchLatitude||cells[cellOf(s.direction)]!==i)fail('成员区域归属');
        }
        for(let a=0;a<r.members.length;a++)for(let b=a+1;b<r.members.length;b++){const d=separation(r.members[a].direction,r.members[b].direction);if(d<rules.minSeparation-1e-9||d>rules.maxDiameter+1e-9)fail('图形角尺度');}
        for(const v of r.variants){
            const degrees=new Map(v.members.map(id=>[id,0])),keys=new Set();
            if(degrees.size!==v.members.length||v.members.some(id=>!members.has(id)))fail('连线方案成员');
            for(let j=0;j<v.edges.length;j++){
                const e=v.edges[j],key=[e.from,e.to].sort().join('/');if(keys.has(key))fail('重复连线');keys.add(key);
                if(e.from===e.to||!degrees.has(e.from)||!degrees.has(e.to))fail('连线端点');
                const a=members.get(e.from),b=members.get(e.to),d=separation(a.direction,b.direction);
                if(d>rules.maxEdge+1e-9||Math.abs(d-e.degrees)>1e-9)fail('连线长度');
                if(v.edges.slice(j+1).some(f=>crossing(e,f,points)))fail('连线交叉');
                if(!arcCells(a.direction,b.direction).every(k=>cells[k]===i))fail('连线穿过其他星座');
                if(r.members.some(s=>s.id!==e.from&&s.id!==e.to&&arcDistance(s.direction,a.direction,b.direction)<rules.lineClearance-1e-9))fail('连线穿过成员');
                degrees.set(e.from,degrees.get(e.from)+1);degrees.set(e.to,degrees.get(e.to)+1);
            }
            if([...degrees.values()].some(n=>n>rules.maxDegree)||(!allowDisconnected&&v.members.some(id=>!pathBetween(v.members[0],id,v.edges))))fail('星形不连通或过密');
        }
        const frame=tangentFrame(r.center.longitude,r.center.latitude);
        const center=coordinates(unit(r.members.reduce((v,s)=>v.map((x,k)=>x+s.direction[k]),[0,0,0])));
        if(!equivalentDraw(center,r.center))fail('局部图中心');
        if(r.members.some(s=>{const p=projectLocal(s.direction,frame,0,0,1);return !p.visible||Math.abs(p.x)>34||Math.abs(p.y)>25;}))fail('局部图裁剪');
    }
    if(data.selectedExtendedCount!==used.size||data.selectedCoreCount!==(adaptiveCore?data.regions.reduce((n,r)=>n+r.variants[0].members.length,0):105))fail('成员总数');
    return true;
}
