import {prepareBackground} from '../shared/solar_system.mjs';
import {eclipticCoordinates} from '../shared/zodiac.mjs';
import {SYMBOL_REFERENCE} from '../shared/sky_render.mjs';
import {AtlasStarPainter} from '../v1/sky_atlas_stars.mjs';
import {RAD,delta,vector,coordinates,arc,tangentFrame,projectLocal,unprojectLocal} from './geometry.mjs?revision=favourites-regional-1';
import {mountWorkflow} from './workflow.mjs?revision=favourites-regional-1';
import {regionAt,boundaryPoints,unpackGrid,toEquatorial,fromEquatorial} from './territories.mjs';
import {brightAudit} from './bright_figures.mjs';
import {connectionStats} from './candidate_edits.mjs';
import {isFreeEdit,regionRings,cornerCount} from './manual_figures.mjs';
import {equatorialPath} from './map_paths.mjs';
import {roundSampling,samplingInfo,samplingCurve} from './sampling.mjs';

const $=id=>document.getElementById(id), overview=$('overview'),detail=$('detail');
const contexts=new Map([overview,detail].map(c=>[c,c.getContext('2d')]));
const painters=new Map([...contexts].map(([canvas,ctx])=>[canvas,new AtlasStarPainter(ctx)]));
const state={region:0,variant:'extended',scale:2.5,limit:6.5,lines:true,boundaries:true,numbers:false,sectors:false,compare:false,inspected:null};
let data,stars,overviewLayout,detailLayout,workflow,hits=[],frames=0,regionalAudit=[],savedView=null;
const colors=['#acc9d7','#aaa6ce','#d5b393','#b4ccaa','#cbb9cb'];
const rgb=(color,alpha)=>{const n=parseInt(color.slice(1),16);return `rgba(${n>>16},${n>>8&255},${n&255},${alpha})`;};
const num=x=>Number(x).toFixed(1);

function canvasStart(canvas) {
    const rect=canvas.getBoundingClientRect(),width=rect.width,height=rect.height,dpr=window.devicePixelRatio||1;
    if(canvas.width!==Math.round(width*dpr)||canvas.height!==Math.round(height*dpr)){canvas.width=Math.round(width*dpr);canvas.height=Math.round(height*dpr);}
    const ctx=contexts.get(canvas);ctx.setTransform(dpr,0,0,dpr,0,0);ctx.globalAlpha=1;ctx.fillStyle='#03060d';ctx.fillRect(0,0,width,height);
    return {ctx,width,height,dpr};
}
function pathLine(ctx,points,project) {
    ctx.beginPath();let active=false;
    for(const v of points){const p=project(v);if(!Number.isFinite(p.x+p.y)||p.visible===false){active=false;continue;}if(active)ctx.lineTo(p.x,p.y);else ctx.moveTo(p.x,p.y);active=true;}
    ctx.stroke();
}
const polygonCache=new WeakMap();
function polygonRings(region) {
    if(!polygonCache.has(region))polygonCache.set(region,region.boundary?regionRings(region).map(r=>boundaryPoints(r)):[region.polygon.flatMap((p,i)=>arc(p,region.polygon[(i+1)%region.polygon.length],.7).slice(0,-1))]);
    return polygonCache.get(region);
}
function currentVariant(region) {return region.variants.find(v=>v.id===state.variant);}
function overviewProject(longitude,latitude) {
    const l=overviewLayout;return {x:l.right-longitude/360*l.plotWidth,y:l.cy-latitude*l.ppd};
}
function overviewPath(ctx,points,center,close=false,fill=false) {
    const mapped=equatorialPath(points,center,close);
    for(const offset of [-360,0,360]) {
        ctx.beginPath();mapped.forEach((c,i)=>{const p=overviewProject(c.longitude+offset,c.latitude);if(i)ctx.lineTo(p.x,p.y);else ctx.moveTo(p.x,p.y);});
        if(fill){ctx.closePath();ctx.fill();}else ctx.stroke();
    }
}
function previewSampling(){return Number($('sampling-width').value);}
function drawOverview() {
    const currentWidth=roundSampling(data),previewWidth=previewSampling(),extent=isFreeEdit(data)?90:Math.min(89,Math.max(50,currentWidth+28,previewWidth+28));
    const proposedWidth=overview.getBoundingClientRect().width;
    overview.style.height=`${Math.max(280,(proposedWidth-66)/360*(2*extent)+50)}px`;
    const {ctx,width,height}=canvasStart(overview),left=40,right=width-26,ppd=(right-left)/360;
    overviewLayout={left,right,plotWidth:right-left,ppd,cy:(height-28)/2,width,height,extent};
    ctx.save();ctx.beginPath();ctx.rect(left,12,right-left,height-40);ctx.clip();
    if(!state.compare){
        ctx.fillStyle='rgba(100,147,181,.075)';overviewPath(ctx,[...samplingCurve(currentWidth),...samplingCurve(-currentWidth).reverse()],0,true,true);
        ctx.strokeStyle='rgba(124,176,210,.55)';ctx.lineWidth=.9;
        for(const beta of [-currentWidth,currentWidth])overviewPath(ctx,samplingCurve(beta),0);
        if(previewWidth!==currentWidth){ctx.setLineDash([6,5]);ctx.strokeStyle='rgba(211,185,131,.65)';for(const beta of [-previewWidth,previewWidth])overviewPath(ctx,samplingCurve(beta),0);ctx.setLineDash([]);}
        ctx.strokeStyle='rgba(121,143,171,.10)';ctx.lineWidth=.6;
        for(let ra=0;ra<=360;ra+=30){const p=overviewProject(ra,0);ctx.beginPath();ctx.moveTo(p.x,12);ctx.lineTo(p.x,height-28);ctx.stroke();}
        for(let dec=-60;dec<=60;dec+=30){const p=overviewProject(0,dec);ctx.beginPath();ctx.moveTo(left,p.y);ctx.lineTo(right,p.y);ctx.stroke();}
    }
    if(state.boundaries&&!state.compare){
        // Fill the actual cell union in manual rounds, including holes, islands
        // and polar caps; a single polygon would incorrectly fill these gaps.
        if(isFreeEdit(data)){
            const cells=unpackGrid(data.territories);
            for(let y=0;y<180;y++)for(let x=0;x<360;){const owner=cells[y*360+x],start=x;while(x<360&&cells[y*360+x]===owner)x++;
                if(owner<15){const p=overviewProject(x,y-89);ctx.fillStyle=rgb(colors[owner%colors.length],owner===state.region?.095:.022);ctx.fillRect(p.x,p.y,(x-start)*ppd,ppd+.05);}
            }
        }
        for(const [i,r] of data.regions.entries()){
            const center=coordinates(toEquatorial(r.site)).longitude;
            for(const points of polygonRings(r)){
                if(!isFreeEdit(data)){ctx.fillStyle=rgb(colors[i%colors.length],i===state.region?.095:.022);overviewPath(ctx,points,center,true,true);}
                ctx.strokeStyle=rgb(colors[i%colors.length],i===state.region?.46:.22);ctx.lineWidth=.7;overviewPath(ctx,points,center,true);
            }
        }
    }
    if(state.sectors&&!state.compare){ctx.strokeStyle='rgba(131,142,160,.25)';ctx.setLineDash([3,5]);for(let l=0;l<360;l+=24)overviewPath(ctx,Array.from({length:121},(_,i)=>vector(l,-60+i)),coordinates(toEquatorial(vector(l,0))).longitude);ctx.setLineDash([]);}
    if(!state.compare){ctx.strokeStyle='rgba(212,179,108,.58)';ctx.lineWidth=1;overviewPath(ctx,samplingCurve(0),0);}
    if(state.lines&&!state.compare)for(const [i,r] of data.regions.entries()){
        const byId=new Map(r.members.map(s=>[s.id,s]));ctx.strokeStyle=rgb(colors[i%colors.length],i===state.region?.64:.29);ctx.lineWidth=i===state.region?.9:.65;
        for(const edge of currentVariant(r).edges){const a=byId.get(edge.from),b=byId.get(edge.to);if(a.app_mag>state.limit||b.app_mag>state.limit)continue;overviewPath(ctx,arc(a.direction,b.direction,.5),coordinates(toEquatorial(a.direction)).longitude);}
    }
    const symbolScale=ppd/(SYMBOL_REFERENCE.perspectiveFocal*RAD)*state.scale;
    for(const star of stars)if(star.app_mag<=state.limit){const c=star.equatorial,p=overviewProject(c.longitude,c.latitude);if(p.y>10&&p.y<height-26)painters.get(overview).draw(star,p,symbolScale);}
    if(!state.compare)for(const [i,r] of data.regions.entries()){
        const c=coordinates(toEquatorial(vector(r.center.longitude,r.center.latitude))),p=overviewProject(c.longitude,c.latitude);ctx.fillStyle=rgb(colors[i%colors.length],i===state.region?.9:.56);ctx.font='11px system-ui';ctx.textAlign='center';ctx.fillText(r.id.slice(1),p.x,p.y-15);
    }
    ctx.restore();ctx.font='10px system-ui';ctx.textAlign='center';ctx.fillStyle='#8295ad';
    for(let ra=0;ra<=360;ra+=30){const p=overviewProject(ra,0);ctx.fillText(`${ra}°`,p.x,height-8);}
    ctx.textAlign='right';for(let latitude=-60;latitude<=60;latitude+=30){const p=overviewProject(0,latitude);if(p.y>12&&p.y<height-28)ctx.fillText(`${latitude>0?'+':''}${latitude}°`,left-7,p.y+3);}
    ctx.textAlign='left';ctx.fillText('赤纬 ↑ · 赤经 ←',left+6,25);
    const info=samplingInfo(stars,previewWidth);$('sampling-value').textContent=`${previewWidth}°`;
    $('sampling-note').textContent=`本轮初始采样：黄道两侧各 ${currentWidth}°。${previewWidth!==currentWidth?'虚线预览：':'下轮范围：'}两侧各 ${previewWidth}°，带内有 ${info.candidateCount} 颗 ≤4.5 等候选星。拖动仅预览，生成新一轮后应用；最终区域内的重要亮星仍会补入。`;
    overview.dataset.samplingHalfWidth=String(currentWidth);overview.dataset.previewHalfWidth=String(previewWidth);overview.dataset.samplingCandidates=String(info.candidateCount);
}
function drawDetail() {
    const {ctx,width,height}=canvasStart(detail),r=data.regions[state.region],view=workflow?.editView;
    const center=view?.center??r.center,frame=tangentFrame(center.longitude,center.latitude);
    let ppd=Math.min((width-44)/(data.territories?72:58),(height-44)/(data.territories?56:52));
    const fitScale=(points,fitCenter)=>{
        const f=tangentFrame(fitCenter.longitude,fitCenter.latitude),p=points.map(v=>projectLocal(v,f,0,0,1)).filter(p=>Number.isFinite(p.x+p.y));
        if(p.length)ppd=Math.min(ppd,(width-44)/(2*(Math.max(...p.map(p=>Math.abs(p.x)))+4)),(height-44)/(2*(Math.max(...p.map(p=>Math.abs(p.y)))+4)));
    };
    if(view){fitScale(view.fitPoints,view.fitCenter);ppd*=view.zoom;}
    else if(isFreeEdit(data))for(const region of data.regions)fitScale(region.members.map(s=>s.direction),region.center);
    const project=v=>{const p=projectLocal(v,frame,width/2,height/2,ppd);if(isFreeEdit(data)||view)p.visible=Number.isFinite(p.x+p.y);return p;};
    hits=[];detailLayout={frame,ppd,width,height,project,hits,unprojectEquatorial:(x,y)=>coordinates(toEquatorial(unprojectLocal(x,y,frame,width/2,height/2,ppd)))};
    if(!state.compare){ctx.strokeStyle='rgba(124,176,210,.4)';ctx.lineWidth=.8;for(const beta of [-roundSampling(data),roundSampling(data)])pathLine(ctx,samplingCurve(beta),project);}
    if(state.boundaries&&!state.compare)for(const [i,region] of data.regions.entries()){
        ctx.strokeStyle=rgb(colors[i%colors.length],i===state.region?.32:.12);ctx.lineWidth=.8;for(const points of polygonRings(region))if(points.length)pathLine(ctx,[...points,points[0]],project);
    }
    if(!state.compare){ctx.strokeStyle='rgba(184,158,99,.30)';ctx.lineWidth=.8;pathLine(ctx,Array.from({length:241},(_,i)=>vector(r.center.longitude-60+i*.5,0)),project);}
    if(state.sectors&&!state.compare){ctx.strokeStyle='rgba(132,145,163,.24)';ctx.setLineDash([3,6]);for(let l=0;l<360;l+=24)if(Math.abs(delta(l,r.center.longitude))<60)pathLine(ctx,Array.from({length:121},(_,i)=>vector(l,-60+i)),project);ctx.setLineDash([]);}
    const variant=currentVariant(r),byId=new Map(r.members.map(s=>[s.id,s]));
    if(state.lines&&!state.compare){ctx.strokeStyle=rgb(colors[state.region%colors.length],.46);ctx.lineWidth=.85;
        for(const edge of variant.edges){const a=byId.get(edge.from),b=byId.get(edge.to);if(a.app_mag>state.limit||b.app_mag>state.limit)continue;
            const points=arc(a.direction,b.direction,Math.min(.2,edge.degrees/12)),screen=points.map(project),first=screen[0],last=screen.at(-1),gap=Math.min(3,Math.hypot(last.x-first.x,last.y-first.y)*.18);ctx.beginPath();let started=false;
            for(let i=1;i<screen.length-1;i++){const p=screen[i];if(!p.visible||!Number.isFinite(p.x+p.y)){started=false;continue;}if(Math.hypot(p.x-first.x,p.y-first.y)<gap||Math.hypot(p.x-last.x,p.y-last.y)<gap)continue;if(!started)ctx.moveTo(p.x,p.y);else ctx.lineTo(p.x,p.y);started=true;}ctx.stroke();
        }
    }
    const symbolScale=ppd/(SYMBOL_REFERENCE.perspectiveFocal*RAD)*state.scale;
    for(const star of stars)if(star.app_mag<=state.limit){const p=project(star.direction);if(p.visible&&p.x>-5&&p.y>-5&&p.x<width+5&&p.y<height+5){painters.get(detail).draw(star,p,symbolScale);hits.push({star,...p});}}
    if(!state.compare){
        if(state.numbers){ctx.font='11px system-ui';ctx.fillStyle='rgba(191,202,217,.62)';variant.members.forEach((id,i)=>{const s=byId.get(id);if(s.app_mag>state.limit)return;const p=project(s.direction);ctx.fillText(String(i+1).padStart(2,'0'),p.x+7,p.y-7);});}
        if(state.inspected){const s=stars.find(s=>s.id===state.inspected);if(s.app_mag<=state.limit){const p=project(s.direction);ctx.strokeStyle='#c7b18b';ctx.lineWidth=.8;ctx.setLineDash([2,3]);ctx.beginPath();ctx.arc(p.x,p.y,8,0,2*Math.PI);ctx.stroke();ctx.setLineDash([]);}}
    }
    ctx.fillStyle='#74849b';ctx.font='10px system-ui';ctx.textAlign='left';ctx.fillText('北 ↑    东 ←',16,24);ctx.textAlign='right';ctx.fillText(`${r.id} · ${variant.label} · ${state.scale.toFixed(1)}×`,width-16,24);ctx.textAlign='left';
    const length=ppd*5;ctx.strokeStyle='#66768b';ctx.lineWidth=.7;ctx.beginPath();ctx.moveTo(18,height-24);ctx.lineTo(18+length,height-24);ctx.stroke();ctx.fillText('5° · 中央角尺度',18,height-33);
    if(data.recipe){ctx.textAlign='right';ctx.fillText(`${data.recipe.seed} · ${workflow?.record.id??''}`,width-16,height-20,Math.max(90,width-190));ctx.textAlign='left';}
    workflow?.paintEditor(ctx,detailLayout);
    $('scale-note').textContent=`显示 ${state.scale.toFixed(1)}× · 中央 ${ppd.toFixed(1)} 像素 / 度`;
    $('view-scale-note').textContent=view?'编辑中 · 可拖动与缩放视野 · 立体投影':'十五张局部图共用角尺度 · 立体投影';
    detail.dataset.pixelsPerDegree=String(ppd);detail.dataset.visibleStars=String(hits.length);
}
function updateInformation() {
    const r=data.regions[state.region],variant=currentVariant(r),byId=new Map(r.members.map(s=>[s.id,s]));
    $('detail-title').textContent=`${r.label} · ${variant.label}`;$('region-id').textContent=`${r.id} / ${String(state.region+1).padStart(2,'0')} OF 15`;$('region-name').textContent=`${variant.members.length} 颗星的${state.variant==='extended'?'星形':'骨架'}`;
    $('region-description').textContent=r.oldSectors.length>1?`成员跨越原 ${r.oldSectors.map(n=>String(n).padStart(2,'0')).join('、')} 天区；区域按星群重新划分。`:'成员集中在一片天空中；区域宽度由整体分布决定。';
    const {loops,branches,components,isolated}=connectionStats(variant.members,variant.edges);
    $('structure-note').textContent=`${loops} 个独立闭环 · ${branches} 处岔枝${components>1?` · ${components} 个分段（含 ${isolated} 个孤立点）`:''}。`;
    const audit=regionalAudit[state.region];
    $('bright-note').replaceChildren();$('bright-note').classList.toggle('warning',!!audit?.missing.length);
    if(audit?.brightest){$('bright-note').append(document.createTextNode(`本区最亮 ${audit.brightest.app_mag.toFixed(2)} 等 · ${isFreeEdit(data)?'参考亮星':'重要亮星'}已选 ${audit.required.length-audit.missing.length} / ${audit.required.length}。${isFreeEdit(data)?'手动选择，不强制补星。':audit.missingFromCore.length?`骨架尚缺 ${audit.missingFromCore.length} 颗。`:'骨架也已保留。'}`));
        if(audit.missing.length){const button=document.createElement('button');button.textContent='查看遗漏亮星';button.onclick=()=>inspect(stars.find(s=>s.id===audit.missing[0]));$('bright-note').append(button);}}
    else $('bright-note').textContent=isFreeEdit(data)?'当前区域没有可作参考的背景星。':'初稿尚未按最终天区检查重要亮星。';
    const local=data.localOptimality;
    $('corner-note').textContent=local?`本区局部最低 ${local.localMinima[state.region]} 个拐点 · 当前 ${r.boundary.length} 个 · 协调增加 ${local.excessCorners[state.region]} 个。`:r.boundary?`本区 ${cornerCount(r)} 个拐点${data.manualBoundary||isFreeEdit(data)?' · 手动边界已验证，不标为自动最优。':' · 此旧轮未按局部优先协调。'}`:'';
    const range=variant.members.map(id=>byId.get(id).app_mag);
    const metrics=[['黄道占据宽度',`${num(r.eclipticSpan)}°`],['成员星等',range.length?`${num(Math.min(...range))} 至 ${num(Math.max(...range))}`:'暂无成员'],['本区选星池',`${r.candidateCount} 颗`],['最长连线',variant.edges.length?`${num(Math.max(...variant.edges.map(e=>e.degrees)))}°`:'无连线']];
    $('metrics').replaceChildren(...metrics.flatMap(([key,value])=>{const dt=document.createElement('dt'),dd=document.createElement('dd');dt.textContent=key;dd.textContent=value;return [dt,dd];}));
    $('members').replaceChildren(...variant.members.map((id,i)=>{
        const s=byId.get(id),row=document.createElement('tr');row.classList.toggle('selected',id===state.inspected);
        const values=[String(i+1).padStart(2,'0'),s.app_mag.toFixed(2),`${num(s.longitude)}°`,`${s.latitude>0?'+':''}${num(s.latitude)}°`];
        values.forEach((value,j)=>{const td=document.createElement('td');if(j===0){const b=document.createElement('button');b.textContent=value;b.title=s.app_mag>state.limit?'当前亮度筛选已隐藏此星':s.id;b.disabled=s.app_mag>state.limit;b.setAttribute('aria-label',`查看成员 ${value}`);b.onclick=()=>workflow?.status.editing?workflow.chooseStar(s.id):inspect(s);td.append(b);}else td.textContent=value;row.append(td);});return row;
    }));
    const filtered=range.filter(m=>m>state.limit).length;$('filtered-note').textContent=filtered?`当前背景星筛选隐藏了 ${filtered} 颗成员及相关连线。`:'两套连线方案共用原始星表，没有另外添加恒星。';
    [...$('region-tabs').children].forEach((b,i)=>{b.classList.toggle('active',i===state.region);b.classList.toggle('locked',!!workflow?.status.locks.includes(i));b.setAttribute('aria-pressed',String(i===state.region));});
    workflow?.regionChanged(state.region);
}
function inspect(star){state.inspected=star.id;$('inspection').textContent=`${star.id} · ${star.app_mag.toFixed(2)} 等 · 黄经 ${star.longitude.toFixed(2)}° · 黄纬 ${star.latitude.toFixed(2)}° · ${star.distance_pc.toFixed(1)} pc`;draw();}
function clearInspection(){state.inspected=null;$('inspection').textContent='点击星点可查看原始编号和视星等。';}
function draw(){if(!data)return;drawOverview();drawDetail();updateInformation();frames++;document.body.dataset.region=data.regions[state.region].id;document.body.dataset.variant=state.variant;document.body.dataset.displayScale=String(state.scale);}
function selectRegion(index){if(workflow?.status.editing||workflow?.status.busy)return;state.region=(index+15)%15;clearInspection();history.replaceState(null,'',`#${data.regions[state.region].id}`);draw();}

function applyData(next){
    data=next;clearInspection();
    regionalAudit=data.territories?brightAudit(data,stars,unpackGrid(data.territories)):[];
    $('summary').textContent=`${data.sourceCount.toLocaleString()} 颗背景星 · ${data.selectedExtendedCount} 颗完整成员 · 15 个不等宽区域`;
    const spans=data.regions.map(r=>r.eclipticSpan),measured=spans.every(Number.isFinite);
    $('coverage-note').hidden=!measured;
    if(measured)$('coverage-note').textContent=`黄道交段 ${num(Math.min(...spans))}°–${num(Math.max(...spans))}° · 小于 16° 的窄区 ${spans.filter(s=>s<16-1e-9).length} / 15 座 · 合计 ${num(spans.reduce((n,s)=>n+s,0))}°${data.envelopePolicy?` · 星形包络外余量 ≤${data.envelopePolicy.maximumMarginDegrees}°`:''}`;
    if(data.optimality)$('coverage-note').textContent+=` · 最少拐点 ${data.optimality.minimumCorners}（下界 ${data.optimality.integerLowerBound}）· 当前 1° 精度及归属约束下已证最优`;
    if(data.localOptimality)$('coverage-note').textContent+=` · 局部优先总拐点 ${data.localOptimality.minimumTotalAtLocalPriority} · ${data.localOptimality.excessCorners.filter(n=>n===0).length} / 15 区达到各自局部下界`;
    if(isFreeEdit(data))$('coverage-note').textContent+=` · 手动设计：黄道宽度只作记录，${data.manualEdits.status==='draft'?'待完成包围':'包围已验证'}`;
    if(data.manualBoundary)$('coverage-note').textContent+=' · 手动边界已验证，自动最优标记已移除';
    $('method-counts').textContent=`${isFreeEdit(data)?'手动设计可从完整源星表选择的':data.brightPolicy?'初始分组搜索黄道附近，最终按完整星表的实际区域归属检查亮星；使用视星等 ≤4.5 的':'固定使用黄纬 ±30°、视星等 ≤4.5 的'} ${data.candidateCount} 颗候选${data.eligibleCandidateCount?`，其中 ${data.eligibleCandidateCount} 颗位于本轮划定的天区内`:''}；本轮亮星骨架共 ${data.selectedCoreCount} 颗，完整星形共 ${data.selectedExtendedCount} 颗。恒星 ID、位置、星等与颜色均沿用源星表。`;
    $('variant').replaceChildren(...data.regions[0].variants.map(v=>{const option=document.createElement('option');option.value=v.id;option.textContent=v.label;return option;}));$('variant').value=state.variant;
    draw();
}

async function init(){
    const response=await fetch('../../design/zodiac_candidates_v1.json');if(!response.ok)throw Error('候选文件读取失败');data=await response.json();
    const source=await fetch(`../../${data.catalogue}`);if(!source.ok)throw Error('原始星表读取失败');const bytes=await source.arrayBuffer();
    const digest=[...new Uint8Array(await crypto.subtle.digest('SHA-256',bytes))].map(x=>x.toString(16).padStart(2,'0')).join('');
    if(digest!==data.sha256)throw Error('星表与这份候选不匹配，请重新生成候选后查看。');
    stars=prepareBackground(JSON.parse(new TextDecoder().decode(bytes)).stars).map(s=>{const c=eclipticCoordinates(s.baseDirection);const direction=vector(c.longitude,c.latitude);return {...s,...c,direction,equatorial:coordinates(toEquatorial(direction))};});
    data.regions.forEach((r,i)=>{const b=document.createElement('button');b.textContent=r.id.slice(1);b.setAttribute('aria-label',`查看${r.label}`);b.onclick=()=>selectRegion(i);$('region-tabs').append(b);});
    state.region=Math.max(0,data.regions.findIndex(r=>`#${r.id}`===location.hash));
    $('sampling-width').oninput=()=>draw();
    $('scale').oninput=e=>{state.scale=Number(e.target.value);$('scale-value').textContent=`${state.scale.toFixed(1)}×`;draw();};
    $('variant').onchange=e=>{state.variant=e.target.value;clearInspection();draw();};$('limit').onchange=e=>{state.limit=Number(e.target.value);if(state.inspected&&stars.find(s=>s.id===state.inspected).app_mag>state.limit)clearInspection();draw();};
    for(const id of ['lines','boundaries','numbers','sectors'])$(id).onchange=e=>{state[id]=e.target.checked;draw();};
    const compare=on=>{state.compare=on;$('compare').classList.toggle('active',on);draw();};
    $('compare').onpointerdown=e=>{e.currentTarget.setPointerCapture(e.pointerId);compare(true);};
    for(const event of ['pointerup','pointercancel','lostpointercapture'])$('compare').addEventListener(event,()=>{if(state.compare)compare(false);});
    $('compare').onkeydown=e=>{if(e.code==='Space'||e.code==='Enter'){e.preventDefault();compare(true);}};$('compare').onkeyup=e=>{if(e.code==='Space'||e.code==='Enter'){e.preventDefault();compare(false);}};$('compare').onblur=()=>{if(state.compare)compare(false);};
    $('previous').onclick=()=>selectRegion(state.region-1);$('next').onclick=()=>selectRegion(state.region+1);
    overview.onclick=e=>{const rect=overview.getBoundingClientRect(),x=e.clientX-rect.left,y=e.clientY-rect.top,l=overviewLayout;if(x<l.left||x>l.right)return;const index=regionAt(data,fromEquatorial(vector((l.right-x)/l.plotWidth*360,(l.cy-y)/l.ppd)));if(index<15)selectRegion(index);};
    detail.onclick=e=>{if(workflow?.status.editing||workflow?.status.busy)return;const rect=detail.getBoundingClientRect(),x=e.clientX-rect.left,y=e.clientY-rect.top;const hit=hits.map(h=>({...h,d:Math.hypot(h.x-x,h.y-y)})).filter(h=>h.d<9).sort((a,b)=>a.d-b.d)[0];if(hit)inspect(hit.star);};
    $('export').onclick=()=>{const round=data.recipe?`${data.recipe.seed.replace(/[^a-zA-Z0-9_-]/g,'_')}-${workflow.record.id}-`:'';const filename=`Terrax-${round}${data.regions[state.region].id}-${state.variant}-${state.scale.toFixed(1)}x-${state.compare||!state.lines?'no-lines':'lines'}.png`;detail.toBlob(blob=>{if(!blob){$('inspection').textContent='图片保存失败，请重试。';return;}const url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download=filename;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);},'image/png');};
    workflow=mountWorkflow({stars,baseline:data,onChange:applyData,onMetadata:updateInformation,getLayout:()=>detailLayout,redraw:draw,
        onEditing(mode){
            if(mode){savedView={variant:state.variant,limit:state.limit,lines:state.lines,boundaries:state.boundaries};Object.assign(state,{variant:'extended',limit:6.5,lines:true,boundaries:true,compare:false});}
            else if(savedView){Object.assign(state,savedView);savedView=null;}
            $('variant').value=state.variant;$('limit').value=String(state.limit);$('lines').checked=state.lines;$('boundaries').checked=state.boundaries;clearInspection();
        }});
    $('loading').hidden=true;$('review').hidden=false;await workflow.start();new ResizeObserver(()=>draw()).observe(detail);draw();document.body.dataset.state='ready';
    window.constellationReview={get status(){return {...state,frames,sourceCount:stars.length,sourceHash:digest,regions:data.regions.length,ppd:detailLayout.ppd,visibleStars:hits.length,workflow:workflow.status};},get editor(){return workflow.editor;},get data(){return workflow.data;},get record(){return workflow.record;},get projectedMembers(){return data.regions[state.region].members.map(s=>({id:s.id,app_mag:s.app_mag,...detailLayout.project(s.direction)}));}};
}
init().catch(error=>{$('loading').hidden=false;$('loading').classList.add('error');$('loading').textContent=error.message;document.body.dataset.state='error';console.error(error);});
