import {drawZodiac} from '../shared/zodiac.mjs';
import {SKY_OVERLAYS} from '../shared/sky_overlays.mjs';
import {referenceLines} from '../shared/sky_guides.mjs';
import {preferredCatalog} from '../shared/catalog_data.mjs';
import {loadDiffuseTexture} from '../shared/deep_sky_view.js';
import * as THREE from 'three';
import {SkyControls} from './sky_controls.mjs';
import {SkyView} from './sky_view.js';
import {EyeSkyRenderer} from './eye_renderer.mjs';
import {screenCalibration,fovForFocal,observerSky} from './eye_model.mjs';
import {projectionFocal} from '../shared/sky_projection.mjs';
import {backgroundDirection,DEG,clamp,applyFrame} from '../shared/solar_system.mjs';
import {catalogFolders,loadCatalog,trajectory,trajectoryDirections,diskBasis,fmt} from './sky_render.mjs';
import {DEFAULT_DISPLAY_SCALE,MAX_DISPLAY_SCALE,displaySky} from '../shared/sky_render.mjs';

const page=document.body.dataset.page==='atlas'?1:2;
const controls=new SkyControls(page),clock=controls.clock,$=id=>document.getElementById(id);
const stage=$('eye-stage'),canvas=$('eye-canvas'),overlay=$('eye-overlay'),ctx=overlay.getContext('2d');
const camera=new THREE.PerspectiveCamera(40,1,.01,10);let painter;
let diffuseTexture=null,deepSkyCount=0;
let stars=[],overview=page===1,white=false,dirty=true,failed=false,lost=false,lastDraw=0,frames=0;
let trailKey='',trailPoints=[],loadVersion=0,loadedFolder='',selectedStar=null;
let lastMode=clock.state.mode,width=1,height=1,calibration;
const view=new SkyView(camera,canvas,{onZoom(){
    const factor=projectionFocal(height,camera.fov,camera.userData.skyProjection)/calibration.focal;
    clock.set({magnification:factor});
}});
const settings=document.createElement('details');settings.className='eye-settings';settings.open=true;
settings.innerHTML=`<summary>观看尺度与光感</summary>
 <label>天体显示大小 <output id="display-scale-value"></output><input id="display-scale" type="range" min="1" max="${MAX_DISPLAY_SCALE}" step=".1"></label>
 <p class="small-note">默认 ${DEFAULT_DISPLAY_SCALE} 倍观看增强；背景星、日月和行星一起放大，位置与真实角径读数不变。点击“肉眼尺度 1×”关闭增强。</p>
 <label class="check"><input id="eye-blur" type="checkbox">轻微光扩散</label>
 <p class="small-note">增强模式的背景星保持清晰；光扩散用于日月和 1× 自然星点，不改变天体位置。</p>
 <div class="coordinate-fields"><label>屏幕对角线 / 英寸<input id="eye-screen" type="number" min="8" max="80" step=".1"></label><label>观看距离 / 厘米<input id="eye-distance" type="number" min="20" max="200" step="1"></label></div>
 <p class="small-note">初始 27 英寸、60 厘米为示例，请按实际设备调整。观看倍率与天体显示均为 1× 时，对应校准后的观看尺度。</p>
 <details><summary>用实物尺精确校准</summary><p class="small-note">调整下方数值，让线段在屏幕上恰好长 5 厘米。校准后再设置观看距离；更换显示器或网页缩放后需要重校准。</p>
 <label>每厘米像素<input id="eye-density" type="number" min="5" max="200" step=".1"></label><div class="ruler-area"><div id="eye-ruler"><span>5 cm</span></div></div><button id="eye-auto-density">恢复按屏幕尺寸估算</button></details>
 <details><summary>光感与天空参数</summary>
 <label>显示亮度 <output id="eye-exposure-value"></output><input id="eye-exposure" type="range" min="-3" max="3" step=".1"></label>
 <label>光扩散宽度 / 角分<input id="eye-acuity" type="number" min=".5" max="4" step=".1"></label>
 <label>空气消光 / 星等<input id="eye-extinction" type="number" min="0" max=".8" step=".01"></label>
 <label>无月夜空 / mag·arcsec⁻²<input id="eye-darkness" type="number" min="16" max="22" step=".1"></label>
 <p class="small-note">这些是可编辑的晴空与视觉近似参数。显示亮度不改变“亮度达标”的判断。</p></details>`;
document.querySelector('.observer-settings').after(settings);
const setEye=patch=>clock.set({eye:{...clock.state.eye,...patch}});
$('display-scale').oninput=e=>clock.set({displayScale:Number(e.target.value)});
$('eye-blur').onchange=e=>setEye({opticalBlur:e.target.checked});
for(const [id,key] of [['eye-screen','screenInches'],['eye-distance','distanceCm'],['eye-density','pixelsPerCm'],['eye-acuity','acuityArcmin'],['eye-extinction','extinction'],['eye-darkness','skyMagnitude'],['eye-exposure','exposureEV']]){
    $(id).oninput=e=>{if(e.target.value!=='' && e.target.checkValidity())setEye({[key]:Number(e.target.value),...(key==='screenInches'?{pixelsPerCm:0}:{})});};
    $(id).onchange=()=>sync();
}
$('eye-auto-density').onclick=()=>setEye({pixelsPerCm:0});
const tools=$('page-tools');tools.innerHTML=`<details><summary>查找背景恒星</summary><input id="star-search" type="text" placeholder="输入恒星编号或名称" aria-label="查找背景恒星"><div id="star-results"></div><p id="star-reading" class="small-note"></p></details>
<details><summary>背景星表</summary><select id="catalog-folder" aria-label="选择背景星表"></select></details><button id="catalog-retry" hidden>重试加载星表</button>`;
$('star-search').oninput=e=>{
    const query=e.target.value.trim().toLowerCase();$('star-results').replaceChildren();if(!query)return;
    for(const star of stars.filter(s=>`${s.id} ${s.name||''}`.toLowerCase().includes(query)).slice(0,20)){
        const b=document.createElement('button');b.textContent=`${star.name||star.id} · ${fmt(star.app_mag,2)} 等`;
        b.onclick=()=>focusStar(star);$('star-results').append(b);
    }
};
$('catalog-folder').onchange=e=>clock.set({folder:e.target.value});
$('catalog-retry').onclick=()=>{loadedFolder='';initCatalog();};
function applyScale(){
    calibration=screenCalibration(clock.state.eye,screen.width,screen.height);
    camera.aspect=width/height;camera.userData.skyProjection=page===1?'stereographic':clock.state.projection;
    camera.fov=fovForFocal(height,calibration.focal*clock.state.magnification,camera.userData.skyProjection);
    camera.updateProjectionMatrix();dirty=true;
}
function sync(){
    const e=clock.state.eye;applyScale();
    $('display-scale').value=clock.state.displayScale;$('display-scale-value').textContent=`${clock.state.displayScale.toFixed(1)}×`;
    for(const [id,value] of [['eye-screen',e.screenInches],['eye-distance',e.distanceCm],['eye-density',calibration.pixelsPerCm],['eye-acuity',e.acuityArcmin],['eye-extinction',e.extinction],['eye-darkness',e.skyMagnitude],['eye-exposure',e.exposureEV]])
        if(document.activeElement!==$(id))$(id).value=Math.round(value*100)/100;
    $('eye-blur').checked=e.opticalBlur;$('toolbar-blur').setAttribute('aria-pressed',String(e.opticalBlur));$('toolbar-blur').textContent=`光扩散：${e.opticalBlur?'开':'关'}`;
    $('eye-exposure-value').textContent=`${e.exposureEV>0?'+':''}${e.exposureEV.toFixed(1)} EV`;
    $('eye-ruler').style.width=`${5*calibration.pixelsPerCm}px`;
    if(lastMode!==clock.state.mode){lastMode=clock.state.mode;focusBody(clock.state.selected,0);}
    if(clock.state.folder && clock.state.folder!==loadedFolder)loadStars(clock.state.folder);
    trailKey='';dirty=true;
}
clock.subscribe(sync);
function setOverview(value){overview=!!value;view.enabled=!overview;view.transition=null;$('view-overview').setAttribute('aria-pressed',String(overview));dirty=true;}
function focusBody(id,duration=450){
    const sky=observerSky(clock.time(),clock.state),body=sky.bodies.find(b=>b.id===id);if(!body)return;
    selectedStar=null;if(overview){focusAtlas(body.view);return;}view.focus(new THREE.Vector3(...body.view),duration);dirty=true;
}
function focusStar(star){
    clock.set({playing:false});selectedStar=star;
    if(overview){focusAtlas(backgroundDirection(star,observerSky(clock.time(),clock.state).frame));return;}
    view.focus(new THREE.Vector3(...backgroundDirection(star,observerSky(clock.time(),clock.state).frame)),450);
    $('star-reading').textContent=`${star.name||star.id} · 大气外 ${fmt(star.app_mag,2)} 等；定位不代表当前肉眼可见。`;dirty=true;
}
window.addEventListener('sky-focus',e=>focusBody(e.detail));
window.addEventListener('sky-inspect',e=>{
    const b=displaySky(observerSky(clock.time(),clock.state),clock.state.displayScale).bodies.find(b=>b.id===e.detail);
    if(overview){zoomAtlas(80/Math.max(.01,painter.focal*b.angularDiameter*DEG));focusBody(e.detail);}
    else {clock.set({magnification:clamp(90/(calibration.focal*b.angularDiameter*DEG),1,40)});focusBody(e.detail);}
});
$('view-natural').onclick=()=>{setOverview(false);clock.set({magnification:1,displayScale:1});};
$('view-overview').hidden=page!==1;
$('view-overview').onclick=()=>{Object.assign(atlasTransform,{zoom:1,panX:0,panY:0});setOverview(true);};
$('view-horizon').onclick=()=>{setOverview(false);clock.set({mode:'surface'});view.focus(new THREE.Vector3(1,0,.08),500);dirty=true;};
$('view-minus').onclick=()=>overview?zoomAtlas(1/1.4):clock.set({magnification:clock.state.magnification/1.4});
$('view-plus').onclick=()=>overview?zoomAtlas(1.4):clock.set({magnification:clock.state.magnification*1.4});
$('toolbar-blur').onclick=()=>setEye({opticalBlur:!clock.state.eye.opticalBlur});
$('view-colors').onclick=()=>{white=!white;$('view-colors').textContent=white?'恢复颜色':'改为白色';dirty=true;};
$('view-export').onclick=()=>{
    draw(performance.now(),true);
    const out=document.createElement('canvas');out.width=canvas.width;out.height=canvas.height;const context=out.getContext('2d');context.drawImage(canvas,0,0);context.drawImage(overlay,0,0,out.width,out.height);
    const a=document.createElement('a');a.download=`terrax-v2-${page===1?'atlas':'sky'}-day-${clock.time().toFixed(5)}.png`;a.href=out.toDataURL('image/png');a.click();out.width=out.height=1;
};
const atlasTransform={zoom:1,panX:0,panY:0};
function zoomAtlas(factor,x=width/2,y=height/2){
    const old=atlasTransform.zoom,next=clamp(old*factor,.55,128),k=next/old;
    atlasTransform.panX=x-width/2-(x-width/2-atlasTransform.panX)*k;
    atlasTransform.panY=y-height/2-(y-height/2-atlasTransform.panY)*k;
    atlasTransform.zoom=next;dirty=true;
}
function focusAtlas(direction){
    // Configure immediately so switching observer mode uses the new frame.
    painter.configure(camera,observerSky(clock.time(),clock.state),{focal:calibration.focal,overview:true,atlasTransform});
    const p=painter.project(direction);atlasTransform.panX+=width/2-p.x;atlasTransform.panY+=height/2-p.y;dirty=true;
}
let press,drag;
canvas.addEventListener('wheel',e=>{if(!overview)return;e.preventDefault();const rect=canvas.getBoundingClientRect();zoomAtlas(Math.exp(-e.deltaY*.0012),e.clientX-rect.left,e.clientY-rect.top);},{passive:false});
canvas.addEventListener('pointerdown',e=>{press=[e.clientX,e.clientY];if(overview){drag=[...press,atlasTransform.panX,atlasTransform.panY];canvas.setPointerCapture(e.pointerId);}});
canvas.addEventListener('pointermove',e=>{if(drag && overview){atlasTransform.panX=drag[2]+e.clientX-drag[0];atlasTransform.panY=drag[3]+e.clientY-drag[1];}dirty=true;});
canvas.addEventListener('pointercancel',()=>{drag=null;press=null;});
canvas.addEventListener('pointerup',e=>{
    drag=null;
    if(!press || Math.hypot(e.clientX-press[0],e.clientY-press[1])>4 || !painter)return;
    const rect=canvas.getBoundingClientRect(),x=e.clientX-rect.left,y=e.clientY-rect.top,sky=painter.sky;
    let distance=18,chosen=null;
    for(const body of sky.bodies){
        if(!body.visible && !clock.state.markers)continue;
        const p=painter.project(body.view),d=Math.hypot(p.x-x,p.y-y);if(p.visible && d<distance){distance=d;chosen=body;}
    }
    if(chosen){controls.choose(chosen.id);return;}
    distance=10;let star=null;
    for(const s of stars){const d=backgroundDirection(s,sky.frame);if(sky.frame.surface && d[2]<0)continue;const p=painter.project(d),r=Math.hypot(p.x-x,p.y-y);if(p.visible && r<distance){distance=r;star=s;}}
    if(star)focusStar(star);
});
async function loadStars(folder){
    loadedFolder=folder;const version=++loadVersion;
    try{const data=await loadCatalog(folder),texture=await loadDiffuseTexture(data.diffuse,data.deepSky);if(version!==loadVersion){texture?.dispose();return;}diffuseTexture?.dispose();diffuseTexture=texture;painter.setDiffuse(texture);deepSkyCount=data.deepSky.length;stars=data.stars;selectedStar=null;$('star-results').replaceChildren();$('star-search').value='';$('star-reading').textContent='';$('catalog-folder').value=folder;$('catalog-retry').hidden=true;controls.clearError();dirty=true;}
    catch(error){if(version===loadVersion){controls.error(error.message);$('catalog-retry').hidden=false;loadedFolder='';}}
}
function resize(){
    const r=stage.getBoundingClientRect();width=Math.max(1,Math.round(r.width));height=Math.max(1,Math.round(r.height));
    if(painter && !lost)painter.resize(width,height,devicePixelRatio);
    const d=Math.min(devicePixelRatio,2);overlay.width=Math.round(width*d);overlay.height=Math.round(height*d);ctx.setTransform(d,0,0,d,0,0);applyScale();
}
function lineThrough(points,close=false){
    let previous=null;ctx.beginPath();
    for(const p of points){if(!p.visible || !Number.isFinite(p.x+p.y)){previous=null;continue;}
        if(previous && p.north===previous.north && Math.hypot(p.x-previous.x,p.y-previous.y)<width*.5)ctx.lineTo(p.x,p.y);else ctx.moveTo(p.x,p.y);previous=p;
    }if(close)ctx.closePath();
}
function drawOverlay(sky){
    ctx.clearRect(0,0,width,height);ctx.save();if(overview){ctx.beginPath();for(const c of [painter.layout.north,painter.layout.south]){ctx.moveTo(c[0]+painter.layout.radius,c[1]);ctx.arc(...c,painter.layout.radius,0,Math.PI*2);}ctx.clip();}ctx.font='11px -apple-system, sans-serif';
    if(clock.state.grid){
        ctx.globalAlpha=SKY_OVERLAYS.line;ctx.lineWidth=.8;
        for(const line of referenceLines){
            const points=line.points.map(v=>{const d=applyFrame(v,sky.frame),p=painter.project(d);return {...p,visible:p.visible&&(!sky.frame.surface||d[2]>=0)};});
            ctx.strokeStyle=line.color;lineThrough(points);ctx.stroke();
        }
        ctx.globalAlpha=1;
    }
    if(clock.state.zodiac)drawZodiac(ctx,sky,d=>painter.project(d),{maxJump:width*.4});
    if(clock.state.trail!=='off'){
        const key=JSON.stringify([clock.state.selected,clock.state.trail,Math.floor(sky.days*4),clock.state.angles,clock.state.latitude,clock.state.longitude,clock.state.spinPhase,clock.state.mode]);
        if(trailKey!==key){trailKey=key;trailPoints=trajectory(sky.days,clock.state,clock.state.selected,clock.state.trail);}
        const points=trajectoryDirections(trailPoints,sky.frame,clock.state.trail).map(d=>({...painter.project(d),visible:(!sky.frame.surface || d[2]>=0)&&painter.project(d).visible}));
        ctx.globalAlpha=SKY_OVERLAYS.line;ctx.strokeStyle='#b0a077';ctx.lineWidth=1;ctx.setLineDash([3,5]);lineThrough(points);ctx.stroke();ctx.setLineDash([]);ctx.globalAlpha=1;
    }
    if(clock.state.grid || clock.state.zodiac || clock.state.trail!=='off'){
        // Auxiliary lines must not show through opaque day/night disks.
        ctx.globalCompositeOperation='destination-out';
        for(const b of sky.bodies){
            if(overview){
                for(const p of painter.bodyCircles(b)){ctx.beginPath();ctx.arc(p.x,p.y,p.radius,0,Math.PI*2);ctx.fill();}
                continue;
            }
            const basis=diskBasis(b.view),r=b.angularDiameter*DEG/2,points=[];
            for(let i=0;i<64;i++){const a=i*Math.PI/32,v=b.view.map((c,j)=>c+Math.tan(r)*(Math.cos(a)*basis.x[j]+Math.sin(a)*basis.y[j]));points.push(painter.project(v));}
            lineThrough(points,true);ctx.fill();
        }ctx.globalCompositeOperation='source-over';
    }
    if(clock.state.markers){
        ctx.globalAlpha=SKY_OVERLAYS.marker;
        for(const b of sky.bodies){const p=painter.project(b.view);if(!p.visible)continue;
            ctx.strokeStyle=ctx.fillStyle=b.visible?'#ddc087':'#8791a1';ctx.setLineDash(b.visible?[]:[2,3]);ctx.beginPath();ctx.arc(p.x,p.y,8,0,Math.PI*2);ctx.stroke();ctx.fillText(`${b.name}${b.aboveHorizon?'':' · 地平下'}`,p.x+12,p.y+4);
        }ctx.setLineDash([]);
        if(selectedStar){const p=painter.project(backgroundDirection(selectedStar,sky.frame));if(p.visible){ctx.fillStyle='#d0d7e0';ctx.fillText(selectedStar.name||selectedStar.id,p.x+10,p.y-5);}}
    }
    ctx.restore();if(overview){ctx.save();ctx.globalAlpha=SKY_OVERLAYS.text;ctx.fillStyle='#8795a7';ctx.textAlign='center';for(const [key,label] of [['north',sky.frame.surface?'地平线上 · 天顶半球':'北天半球'],['south',sky.frame.surface?'地平线下 · 仅辅助位置':'南天半球']]){const c=painter.layout[key];ctx.fillText(label,c[0],c[1]-painter.layout.radius-12);}ctx.restore();}
}
function draw(now,force=false){
    if(!painter || failed || lost)return;
    if(!force && !dirty && !clock.state.playing && !view.transition)return;
    if(!force && now-lastDraw<33)return;
    view.update(now);const sky=controls.frame(now);
    const stats=painter.render(camera,sky,stars,{focal:calibration.focal*clock.state.magnification,displayScale:clock.state.displayScale,overview,deepSky:clock.state.deepSky,white,atlasTransform});
    drawOverlay(painter.sky);controls.catalogStatus(stars.length,stats.eligible);
    $('view-title').textContent=overview?`双半球平面星图 · ${atlasTransform.zoom.toFixed(2)}×`:`${page===1?'局部星图':'沉浸天球'} · ${clock.state.magnification.toFixed(2)}×`;
    $('view-hint').textContent=overview?'方位等距双半球 · 拖动平移 · 滚轮缩放':`拖动转头 · 滚轮统一缩放 · 垂直视场 ${camera.fov.toFixed(1)}°`;
    const gainLabel=`天体显示 ${clock.state.displayScale.toFixed(1)}×`;
    $('view-scale').textContent=overview?`${gainLabel} · 角尺度 ${(painter.focal*DEG).toFixed(2)} px/°`:`${gainLabel} · ${clock.state.displayScale===1?'校准尺度':'观看增强'} · ${clock.state.eye.distanceCm} cm 观看距离`;
    stage.dataset.state='ready';stage.dataset.deepSkyCount=String(deepSkyCount);stage.dataset.zodiac=String(clock.state.zodiac);stage.dataset.frames=String(++frames);stage.dataset.time=String(sky.days);stage.dataset.stars=String(stars.length);stage.dataset.eligible=String(stats.eligible);stage.dataset.focal=String(painter.focal);stage.dataset.displayScale=String(clock.state.displayScale);stage.dataset.pixelsPerDegree=String(painter.focal*DEG);stage.dataset.deepSkyCandidates=String(diffuseTexture?.userData.profiles?.count||0);
    lastDraw=now;dirty=false;
}
function tick(now){try{draw(now);}catch(error){failed=true;controls.error(error.message);stage.dataset.state='error';console.error(error);}requestAnimationFrame(tick);}
canvas.addEventListener('webglcontextlost',e=>{e.preventDefault();lost=true;stage.dataset.state='lost';controls.error('绘图暂时中断，正在等待浏览器恢复…');});
canvas.addEventListener('webglcontextrestored',()=>{
    // Three recreates its context before this listener. Rebuild our targets
    // as well, including while paused; preserve clock and calibration.
    painter.dispose();painter=new EyeSkyRenderer(canvas);painter.setDiffuse(diffuseTexture);lost=false;failed=false;controls.clearError();resize();dirty=true;
});
window.addEventListener('resize',resize);document.addEventListener('visibilitychange',()=>{dirty=true;});
try{
    painter=new EyeSkyRenderer(canvas);resize();sync();setOverview(overview);if(!overview)focusBody(clock.state.selected,0);requestAnimationFrame(tick);
}catch(error){failed=true;controls.error(error.message);stage.dataset.state='error';console.error(error);}
async function initCatalog(){
    try{
        const folders=await catalogFolders();$('catalog-folder').replaceChildren();for(const folder of folders)$('catalog-folder').add(new Option(folder.slice(7),folder));
        const folder=preferredCatalog(folders,clock.state.folder);
        if(folder===clock.state.folder){if(folder!==loadedFolder)await loadStars(folder);}else clock.set({folder});
    }catch(error){controls.error(error.message);$('catalog-retry').hidden=false;}
}
if(!failed)await initCatalog();

// A narrow, read-only diagnostic surface for the browser acceptance page.
window.terraxV2={get status(){return {state:stage.dataset.state,frames,deepSkyCount,zodiac:clock.state.zodiac,stars:stars.length,time:clock.time(),focal:painter?.focal,magnification:clock.state.magnification,displayScale:clock.state.displayScale,blur:clock.state.eye.opticalBlur,overview,atlasZoom:atlasTransform.zoom,projection:camera.userData.skyProjection,resources:painter?.resourceGeneration};},readPixels:()=>painter.readPixels()};
