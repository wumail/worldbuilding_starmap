import {AtlasStarPainter} from '../web/v1/sky_atlas_stars.mjs';
import {AtlasBodyPainter} from '../web/v1/sky_atlas_bodies.mjs';
import {skyState,prepareBackground,backgroundDirection,projectHemisphere} from '../web/shared/solar_system.mjs';
import {sanitizeState,STORAGE_KEY} from '../web/v1/sky_state.mjs';
import {backgroundVisible,atlasSymbolScale} from '../web/shared/sky_render.mjs';

const status=document.querySelector('#status'),metrics=document.querySelector('#metrics'),canvas=document.querySelector('#sky'),ctx=canvas.getContext('2d');
const results=document.querySelector('#results'),createElement=document.createElement;
let canvases=0,losses=0,frames=0,warm=0,started=0,lastReport=-1;
document.createElement=function(tag,...args){if(tag==='canvas')canvases++;return createElement.call(this,tag,...args);};
canvas.addEventListener('contextlost',()=>losses++);
const starsPainter=new AtlasStarPainter(ctx),bodiesPainter=new AtlasBodyPainter(ctx);
const state=sanitizeState({mode:'center'}),layout={radius:320,north:[355,380],south:[1045,380]},initialDay=3.040*365.25;
let total=0,failures=0,report;
function test(name,fn) {
    total++;const item=document.createElement('li');
    try{fn();item.className='pass';item.textContent='通过：'+name;}catch(e){failures++;item.className='fail';item.textContent=`失败：${name} — ${e.message}`;}
    results.append(item);
}
function assert(ok,message){if(!ok)throw Error(message);}
const pause=ms=>new Promise(resolve=>setTimeout(resolve,ms));
async function waitFor(check) {const start=performance.now();while(!check()){if(performance.now()-start>10000)throw Error('等候生产页重画超时');await pause(100);}}

try {
    const response=await fetch('../output/output_20260915_galactic_01/sky_view_20260915_galactic_01.json');
    if(!response.ok)throw Error('星表加载失败');
    const data=await response.json(),stars=prepareBackground([...(data.neighbors || []),...data.stars]);
    function draw(days,symbolScale=atlasSymbolScale(layout.radius)) {
        const sky=skyState(days,state);ctx.resetTransform();ctx.globalAlpha=1;ctx.fillStyle='#03060d';ctx.fillRect(0,0,canvas.width,canvas.height);
        for(const star of stars) {
            const direction=backgroundDirection(star,sky.frame);
            if(backgroundVisible(star,direction,sky))starsPainter.draw(star,projectHemisphere(direction,layout.radius,layout,13.564125),symbolScale);
        }
        bodiesPainter.drawSky(sky,layout,13.564125,{symbolScale,lunarGlow:true,solarGlow:true});
    }
    // 预先覆盖小、大星纹理及月相；计数从预热结束后开始。
    for(const scale of [.8,1,1.6,2,3])draw(initialDay,scale);
    warm=canvases;started=performance.now();
    await new Promise((resolve,reject)=>{
        function frame(now) {
            try {
                const elapsed=(now-started)/1000;draw(initialDay+elapsed*26/24);frames++;
                if(Math.floor(elapsed)!==lastReport) {
                    lastReport=Math.floor(elapsed);status.textContent=`连续播放 ${lastReport} / 60 秒`;
                    metrics.textContent=JSON.stringify({frames,canvases,warm,newCanvases:canvases-warm,contextLosses:losses},null,2);
                }
                if(elapsed>=60)resolve();else requestAnimationFrame(frame);
            }catch(e){reject(e);}
        }
        requestAnimationFrame(frame);
    });
    const pixels=ctx.getImageData(0,0,canvas.width,canvas.height).data;let bright=0,checksum=0;
    for(let k=0;k<pixels.length;k+=4){if(Math.max(pixels[k]-3,pixels[k+1]-6,pixels[k+2]-13)>2)bright++;checksum=(checksum+pixels[k]+pixels[k+1]+pixels[k+2])>>>0;}
    report={durationSeconds:(performance.now()-started)/1000,frames,catalogCount:stars.length,symbolScale:atlasSymbolScale(layout.radius),warmCanvases:warm,playbackNewCanvases:canvases-warm,contextLosses:losses,brightPixels:bright,checksum};
    test('完整星表连续绘制至少 60 秒，完成至少 120 帧',()=>assert(stars.length===9356 && report.durationSeconds>=60 && frames>=120,JSON.stringify(report)));
    test('预热后星点与天体绘制没有新增画布',()=>assert(canvases===warm,JSON.stringify(report)));
    test('绘图上下文保持有效，画布仍有实际星光像素',()=>assert(losses===0 && !ctx.isContextLost?.() && bright>500,JSON.stringify(report)));
    document.createElement=createElement;starsPainter.clear();bodiesPainter.clear();

    // 合成事件只验证生产页面的恢复处理，不冒充真实 GPU 失效复现。
    status.textContent='连续播放完成；检查暂停状态下的恢复处理…';
    const iframe=document.createElement('iframe');iframe.title='生产页面恢复验收';
    // 运行正式 HTML/模块，但隔离本次测试的时钟存储，避免影响正在操作的主页面。
    const html=await (await fetch('../web/v1/sky_atlas.html')).text();
    const initial=JSON.stringify(sanitizeState({days:3928.342761,playing:false,folder:'output_20260915_galactic_01'}));
    const baseURL=new URL('../web/v1/',location.href).href;
    const init='<base href="'+baseURL+'"><script>const memory=new Map('+JSON.stringify([[STORAGE_KEY,initial]])+');Object.defineProperty(window,"localStorage",{value:{getItem:key=>memory.get(key)??null,setItem:(key,value)=>memory.set(key,value)}});<'+ '/script>';
    iframe.srcdoc=html.replace('<head>','<head>'+init);
    document.querySelector('#recovery').append(iframe);
    await new Promise((resolve,reject)=>{iframe.onload=resolve;iframe.onerror=reject;});
    await waitFor(()=>iframe.contentDocument.querySelector('#atlas')?.dataset.renderStatus==='ready' && +iframe.contentDocument.querySelector('#atlas').dataset.catalogCount===9356);
    const target=iframe.contentDocument.querySelector('#atlas'),before=target.dataset.days;
    const button=[...iframe.contentDocument.querySelectorAll('button')].find(b=>b.textContent==='播放');
    assert(button,'恢复检查的独立时钟没有暂停');
    target.dispatchEvent(new iframe.contentWindow.Event('contextlost'));await pause(150);
    test('生产页绘图中断事件会暂停绘制并显示状态',()=>assert(target.dataset.renderStatus==='context-lost','未进入 context-lost'));
    target.width=target.width;target.dispatchEvent(new iframe.contentWindow.Event('contextrestored'));
    await waitFor(()=>target.dataset.renderStatus==='ready');
    const sample=target.getContext('2d').getImageData(0,0,1,1).data;
    report.recovery={syntheticEvents:true,hardwareLossReproduced:false,daysBefore:before,daysAfter:target.dataset.days,status:target.dataset.renderStatus,pixel:[...sample]};
    test('生产页恢复后自动重画，暂停时刻及星表保持不变',()=>assert(target.dataset.days===before && +target.dataset.catalogCount===9356 && sample[3]===255 && sample[0]===3,JSON.stringify(report.recovery)));
    iframe.remove();
    metrics.textContent=JSON.stringify(report,null,2);for(const [key,value] of Object.entries(report)){const li=document.createElement('li');li.textContent=key+': '+JSON.stringify(value);results.append(li);}status.textContent=`${total-failures}/${total} 通过；${failures} 失败`;
    status.dataset.complete='true';status.dataset.failures=String(failures);
} catch(e) {
    document.createElement=createElement;status.textContent='验收中断：'+e.message;status.className='fail';console.error(e);
}
