import {preferredCatalog} from '../shared/catalog_data.mjs';
import {SKY_OVERLAYS} from '../shared/sky_overlays.mjs';
import {referenceLines} from '../shared/sky_guides.mjs';
import {drawZodiac} from '../shared/zodiac.mjs';
import {DiffuseAtlasPainter,loadDiffuseTexture} from '../shared/deep_sky_view.js';
import {DEG,applyFrame,backgroundDirection,projectHemisphere} from '../shared/solar_system.mjs';
import {SkyControls} from './sky_controls.mjs';
import {SYMBOL_REFERENCE,displaySky,catalogFolders,loadCatalog,backgroundVisible,pointRadius,skyBackground,trajectory,trajectoryDirections} from '../shared/sky_render.mjs';
import {AtlasBodyPainter} from './sky_atlas_bodies.mjs';
import {AtlasStarPainter} from './sky_atlas_stars.mjs';

const controls=new SkyControls(1),clock=controls.clock;
const canvas=document.getElementById('atlas'),stage=document.getElementById('atlas-stage'),ctx=canvas.getContext('2d');
let width=1,height=1,dpr=1,zoom=1,pan=[0,0],pointer=null,white=false;
let diffuseTexture=null,deepSkyCount=0;
const diffusePainter=new DiffuseAtlasPainter(ctx);
let stars=[],folder='',pendingFolder='',sharedFolder=clock.state.folder,loadRevision=0,lastDraw='',hitPoints=[];
let trail=[],trailKey='';
let contextLost=false;
const bodyPainter=new AtlasBodyPainter(ctx);
const starPainter=new AtlasStarPainter(ctx);
const tools=document.getElementById('page-tools');
tools.innerHTML='<details><summary>背景恒星数据集</summary><select id="atlas-catalog" aria-label="背景恒星数据集"></select><p class="small-note">新数据集包含共龄星团与云气；原有星表和图片保留。</p></details>';
const selector=document.getElementById('atlas-catalog');

async function chooseCatalog(next) {
    if(pendingFolder===next)return;
    const revision=++loadRevision;pendingFolder=next;selector.disabled=true;
    try {
        const data=await loadCatalog(next),texture=await loadDiffuseTexture(data.diffuse,data.deepSky);if(revision!==loadRevision){texture?.dispose();return;}
        diffuseTexture?.dispose();diffuseTexture=texture;deepSkyCount=data.deepSky.length;
        starPainter.clear();stars=data.stars;folder=next;selector.value=next;lastDraw='';controls.clearError();
        if(clock.state.folder!==next)clock.set({folder:next});
    } catch(error) { if(revision===loadRevision) { selector.value=folder;controls.error(`${error.message}；保留已加载的星空。`); } }
    finally { if(revision===loadRevision) {selector.disabled=false;pendingFolder='';} }
}
catalogFolders().then(folders=>{
    selector.replaceChildren(...folders.map(f=>new Option(f.slice(7),f)));
    selector.onchange=()=>chooseCatalog(selector.value);
    return chooseCatalog(preferredCatalog(folders,clock.state.folder));
}).catch(error=>controls.error(error.message));
clock.subscribe(s=>{
    lastDraw='';
    if(s.folder===sharedFolder)return;
    sharedFolder=s.folder;
    if(s.folder && s.folder!==folder && [...selector.options].some(o=>o.value===s.folder))chooseCatalog(s.folder);
    else if(s.folder===folder && pendingFolder && pendingFolder!==folder) {loadRevision++;pendingFolder='';selector.disabled=false;selector.value=folder;}
});

function resize() {
    const rect=stage.getBoundingClientRect();width=rect.width;height=rect.height;dpr=Math.min(window.devicePixelRatio || 1,2);
    const w=Math.max(1,Math.round(width*dpr)),h=Math.max(1,Math.round(height*dpr));
    if(canvas.width!==w)canvas.width=w;
    if(canvas.height!==h)canvas.height=h;
    lastDraw='';
}
new ResizeObserver(resize).observe(stage);
canvas.addEventListener('contextlost',()=>{
    contextLost=true;lastDraw='';canvas.dataset.renderStatus='context-lost';
    controls.error('星图绘图资源暂时不可用，浏览器恢复后会自动重画。');
});
canvas.addEventListener('contextrestored',()=>{
    starPainter.clear();bodyPainter.clear();contextLost=false;lastDraw='';
    canvas.dataset.renderStatus='restoring';controls.clearError();
});
function layout() {
    const horizontal=Math.min((width-90)/4,(height-105)/2),vertical=Math.min((width-55)/2,(height-130)/4);
    if(horizontal>=vertical) { const radius=Math.max(10,horizontal);return {radius,north:[width/2-radius-18,height/2+5],south:[width/2+radius+18,height/2+5]}; }
    const radius=Math.max(10,vertical);return {radius,north:[width/2,height/2-radius-20],south:[width/2,height/2+radius+20]};
}
const toScreen=p=>({x:width/2+(p.x-width/2)*zoom+pan[0],y:height/2+(p.y-height/2)*zoom+pan[1]});

function drawLine(points,color,map,options={}) {
    ctx.save();ctx.globalAlpha=SKY_OVERLAYS.line;
    ctx.strokeStyle=color;ctx.lineWidth=(options.width || .8)/zoom;ctx.setLineDash(options.dashed?[3/zoom,4/zoom]:[]);ctx.beginPath();let prev=null;
    for(const direction of points) {
        const p=map(direction);
        if(!prev || p.north!==prev.north || Math.hypot(p.x-prev.x,p.y-prev.y)>options.radius*.4)ctx.moveTo(p.x,p.y);else ctx.lineTo(p.x,p.y);
        prev=p;
    }ctx.stroke();ctx.restore();
}

function draw(sky) {
    const state=clock.state,l=layout(),orientation=sky.frame.surface?180:13.564125;
    sky=displaySky(sky,state.displayScale);
    const symbolScale=l.radius/SYMBOL_REFERENCE.atlasRadius*state.displayScale;
    const map=(v)=>projectHemisphere(v,l.radius,l,orientation);
    ctx.setTransform(dpr,0,0,dpr,0,0);ctx.fillStyle='#03060d';ctx.fillRect(0,0,width,height);
    ctx.translate(width/2+pan[0],height/2+pan[1]);ctx.scale(zoom,zoom);ctx.translate(-width/2,-height/2);
    const background=`rgb(${skyBackground(sky).join(',')})`;
    for(const north of [true,false]) {
        const c=north?l.north:l.south;
        ctx.beginPath();ctx.arc(...c,l.radius,0,Math.PI*2);
        ctx.fillStyle=sky.frame.surface && !north?'#0a0e14':background;ctx.fill();
        ctx.save();ctx.globalAlpha=SKY_OVERLAYS.border;
        ctx.strokeStyle=north?'#596777':'#344353';ctx.lineWidth=.8/zoom;ctx.stroke();ctx.globalAlpha=SKY_OVERLAYS.text;
        ctx.fillStyle='#8795a7';ctx.font=`${11/zoom}px -apple-system,sans-serif`;ctx.textAlign='center';
        ctx.fillText(sky.frame.surface?(north?'地平线上 · 天顶半球':'地平线下 · 仅辅助位置'):(north?'北天半球':'南天半球'),c[0],c[1]-l.radius-12/zoom);ctx.restore();
    }
    diffusePainter.draw(sky,diffuseTexture,l,orientation,{width,height,dpr,zoom,pan,enabled:state.deepSky});
    if(state.zodiac)drawZodiac(ctx,sky,map,{scale:zoom,maxJump:l.radius*.4});
    if(state.grid)for(const line of referenceLines)drawLine(line.points.map(v=>applyFrame(v,sky.frame)),line.color,map,{radius:l.radius});
    let visibleCount=0;hitPoints=[];
    for(const star of stars) {
        const direction=backgroundDirection(star,sky.frame);
        if(!backgroundVisible(star,direction,sky))continue;
        // 使用星图坐标中的半径，让星点与位置一起缩放；不按投影位置放大。
        visibleCount++;const p=map(direction),screen=toScreen(p),radius=pointRadius(star.app_mag)*symbolScale*zoom;
        // 放大后只绘制落在窗口内的光点，不影响全天可见性计数。
        if(screen.x+radius<0 || screen.x-radius>width || screen.y+radius<0 || screen.y-radius>height)continue;
        starPainter.draw(star,p,symbolScale,white);
    }
    const key=JSON.stringify([state.trail,state.selected,state.mode,state.latitude,state.longitude,state.spinPhase,state.angles,Math.floor(sky.days/(state.trail==='day'?.003:1))]);
    if(key!==trailKey) {trail=trajectory(sky.days,state,state.selected,state.trail);trailKey=key;}
    if(trail.length)drawLine(trajectoryDirections(trail,sky.frame,state.trail),'#9e8961',map,{radius:l.radius,dashed:true,width:1});
    bodyPainter.drawSky(sky,l,orientation,{symbolScale,lunarGlow:state.lunarGlow,planetGlow:state.planetGlow,solarGlow:state.solarGlow});
    for(const body of sky.bodies) {
        const p=map(body.view);
        if(body.visible || state.markers)hitPoints.push({...toScreen(p),id:body.id});
        if(!state.markers)continue;
        ctx.save();ctx.globalAlpha=SKY_OVERLAYS.marker;
        const r=(body.id===state.selected?10:7)/zoom;
        ctx.beginPath();
        if(body.kind==='moon') {ctx.moveTo(p.x+r,p.y);ctx.lineTo(p.x+r+3/zoom,p.y);}else ctx.arc(p.x,p.y,r,0,Math.PI*2);
        ctx.strokeStyle=body.visible?(body.id===state.selected?'#ebcd8a':'#9f8e6b'):'#4e5a6a';ctx.lineWidth=.8/zoom;
        ctx.setLineDash(body.visible?[]:[2/zoom,3/zoom]);ctx.stroke();ctx.setLineDash([]);
        ctx.fillStyle=body.visible?'#d7c39b':'#68768b';ctx.font=`${10/zoom}px -apple-system,sans-serif`;ctx.textAlign='left';ctx.fillText(body.id,p.x+12/zoom,p.y-4/zoom);ctx.restore();
    }
    if(sky.frame.surface) {
        ctx.save();ctx.globalAlpha=SKY_OVERLAYS.text;
        ctx.textAlign='center';ctx.fillStyle='#98a6b8';ctx.font=`${10/zoom}px -apple-system,sans-serif`;
        for(const [name,v] of [['北',[1,0,0]],['东',[0,-1,0]],['南',[-1,0,0]],['西',[0,1,0]]]) {
            const p=map(v),c=l.north;ctx.fillText(name,c[0]+(p.x-c[0])*.94,c[1]+(p.y-c[1])*.94+3/zoom);
        }
        ctx.restore();
    }
    document.getElementById('atlas-title').textContent=`${sky.frame.surface?'地平星图':'双半球平面星图'} · ${zoom.toFixed(2)}×`;
    document.getElementById('atlas-scale').textContent=`天体显示 ${state.displayScale.toFixed(1)}× · 角尺度 ${(l.radius*zoom/90).toFixed(2)} px/°`;
    document.getElementById('atlas-hint').textContent=sky.frame.surface?'仰视展开，东在左 · 拖动平移 · 滚轮缩放':'方位等距双半球 · 拖动平移 · 滚轮缩放';
    controls.catalogStatus(stars.length,visibleCount);
    canvas.dataset.days=String(sky.days);canvas.dataset.visibleStars=String(visibleCount);canvas.dataset.catalogCount=String(stars.length);
    canvas.dataset.selectedDirection=JSON.stringify(sky.bodies.find(b=>b.id===state.selected).view);
    canvas.dataset.renderStatus='ready';canvas.dataset.deepSkyCount=String(deepSkyCount);canvas.dataset.zodiac=String(state.zodiac);
    canvas.dataset.zoom=String(zoom);canvas.dataset.symbolScale=String(symbolScale*zoom);canvas.dataset.displayScale=String(state.displayScale);canvas.dataset.pixelsPerDegree=String(l.radius*zoom/90);canvas.dataset.deepSkyCandidates=String(diffuseTexture?.userData.profiles?.count||0);
}

canvas.addEventListener('pointerdown',e=>{ if(e.button!==0)return;pointer={id:e.pointerId,x:e.clientX,y:e.clientY,startX:e.clientX,startY:e.clientY};canvas.setPointerCapture(e.pointerId); });
canvas.addEventListener('pointermove',e=>{ if(!pointer || e.pointerId!==pointer.id)return;pan[0]+=e.clientX-pointer.x;pan[1]+=e.clientY-pointer.y;pointer.x=e.clientX;pointer.y=e.clientY;lastDraw=''; });
canvas.addEventListener('pointerup',e=>{
    if(!pointer || e.pointerId!==pointer.id)return;
    if(Math.hypot(e.clientX-pointer.startX,e.clientY-pointer.startY)<5) {
        const rect=canvas.getBoundingClientRect(),x=e.clientX-rect.left,y=e.clientY-rect.top;
        const nearest=hitPoints.map(p=>({...p,d:Math.hypot(p.x-x,p.y-y)})).sort((a,b)=>a.d-b.d)[0];if(nearest?.d<18)controls.choose(nearest.id);
    }pointer=null;canvas.releasePointerCapture(e.pointerId);
});
canvas.addEventListener('pointercancel',()=>{pointer=null;});
canvas.addEventListener('wheel',e=>{
    e.preventDefault();const rect=canvas.getBoundingClientRect(),x=e.clientX-rect.left-width/2,y=e.clientY-rect.top-height/2;
    const next=Math.max(.55,Math.min(128,zoom*Math.exp(-e.deltaY*(e.deltaMode===1?16:e.deltaMode===2?height:1)*.001)));
    pan=[x-(x-pan[0])*next/zoom,y-(y-pan[1])*next/zoom];zoom=next;lastDraw='';
},{passive:false});
function focusBody(id,inspect=false) {
    const sky=controls.frame(),body=sky.bodies.find(b=>b.id===id);if(!body)return;
    const l=layout(),p=projectHemisphere(body.view,l.radius,l,sky.frame.surface?180:13.564125);
    if(inspect)zoom=Math.min(128,Math.max(1,height*.15/(l.radius/(Math.PI/2)*body.angularDiameter*DEG)));
    pan=[-(p.x-width/2)*zoom,-(p.y-height/2)*zoom];lastDraw='';
}
window.addEventListener('sky-focus',event=>focusBody(event.detail));
window.addEventListener('sky-inspect',event=>focusBody(event.detail,true));
document.getElementById('atlas-reset').onclick=()=>{zoom=1;pan=[0,0];lastDraw='';};
document.getElementById('atlas-colors').onclick=e=>{white=!white;e.target.textContent=white?'切换彩色':'切换白色';lastDraw='';};
document.getElementById('atlas-export').onclick=()=>{
    const sky=controls.frame();draw(sky);
    const snapshot=document.createElement('canvas');snapshot.width=canvas.width;snapshot.height=canvas.height;const out=snapshot.getContext('2d');out.drawImage(canvas,0,0);
    out.globalAlpha=SKY_OVERLAYS.text;out.fillStyle='#bec7d4';out.font=`${11*dpr}px sans-serif`;out.fillText(`Terrax · 第 ${sky.days.toFixed(5)} 地球日 · ${sky.frame.surface?'地表':'质心全天'}`,20*dpr,25*dpr);
    snapshot.toBlob(blob=>{if(!blob)return;const url=URL.createObjectURL(blob),link=document.createElement('a');link.href=url;link.download=`Terrax-sky-day-${sky.days.toFixed(5)}.png`;link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);},'image/png');
};
function animate() {
    requestAnimationFrame(animate);const sky=controls.frame();
    if(contextLost || ctx.isContextLost?.() || width<=0 || height<=0)return;
    const key=JSON.stringify([sky.days,clock.state]);if(key===lastDraw)return;
    draw(sky);lastDraw=key;
}
resize();animate();
