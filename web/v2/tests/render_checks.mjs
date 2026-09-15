import * as THREE from 'three';
import {AtlasStarPainter} from '../../v1/sky_atlas_stars.mjs';
import {runNativeStarChecks} from './native_star_checks.mjs?v=20260915-atlas-disks-1';
import {AtlasBodyPainter} from '../../v1/sky_atlas_bodies.mjs';
import {SYMBOL_REFERENCE,DEFAULT_DISPLAY_SCALE} from '../../shared/sky_render.mjs';
import {EyeSkyRenderer} from '../eye_renderer.mjs';
import {EYE_DEFAULTS,magnitudeToLux,fovForFocal,observerSky} from '../eye_model.mjs';
import {sanitizeState,STORAGE_KEY} from '../sky_state.mjs';
import {DEG,lambertPhase} from '../../shared/solar_system.mjs';
import {loadCatalog,bodyLightSample} from '../sky_render.mjs';
let total=0,failures=0;const metrics={samples:0},results=document.querySelector('#results');
const assert=(ok,message)=>{if(!ok)throw Error(message);};
async function test(name,fn){total++;const li=document.createElement('li');try{await fn();li.className='pass';li.textContent='通过：'+name;}catch(e){failures++;li.className='fail';li.textContent='失败：'+name+' — '+e.message;console.error(e);}results.append(li);}
const near=(a,b,tol,message)=>assert(Math.abs(a-b)<=tol,`${message}: ${a} vs ${b}`);
const delay=ms=>new Promise(r=>setTimeout(r,ms));
async function until(fn,timeout=12000){const start=performance.now();while(!fn()){if(performance.now()-start>timeout)throw Error('等待页面状态超时');await delay(60);}}
const canvas=document.createElement('canvas'),r=new EyeSkyRenderer(canvas),camera=new THREE.PerspectiveCamera(60,1,.01,10);
camera.up.set(0,0,1);camera.lookAt(1,0,0);
const base=observerSky(0,sanitizeState({mode:'center'}));
base.frame.position=[0,0,0];base.environment={atmosphere:false,extinction:0,nightL:0,dayL:0,moons:[],adaptation:.005,limit:6.5};base.eye={...EYE_DEFAULTS,opticalBlur:false};
const body=(id='Luna',angle=0)=>({id,kind:id==='Sol'?'star':'moon',color:'#ffffff',view:[1,0,0],distance:.01,angularDiameter:id==='Echo'?.114561:.531544,altitude:0,aboveHorizon:true,brightEnough:true,visible:true,
    lightDirection:[-Math.cos(angle),-Math.sin(angle),0],phaseAngle:angle,magnitude:-12.7-2.5*Math.log10(lambertPhase(angle)),observedMagnitude:-12.7-2.5*Math.log10(lambertPhase(angle))});
const star=(m=0,d=[1,0,0])=>({id:'fixture',app_mag:m,color_hex:'#ffffff',baseDirection:d,baseDistanceAU:1e10});
function render(bodies=[],stars=[],f=1500,{size=256,dpr=1,mode='perspective',blur=false,overview=false,displayScale=overview?DEFAULT_DISPLAY_SCALE:1,eye={},atlasTransform}={}){
    r.resize(size,size,dpr);camera.aspect=1;camera.userData.skyProjection=mode;camera.fov=fovForFocal(size,f,mode);camera.updateProjectionMatrix();
    const sky={...base,bodies,eye:{...base.eye,...eye,opticalBlur:blur}};
    r.render(camera,sky,stars,{focal:f,overview,displayScale,atlasTransform});metrics.samples++;return r.readLinear();
}
function energy(image){let sum=0;for(let i=0;i<image.data.length;i+=4)sum+=.2126*image.data[i]+.7152*image.data[i+1]+.0722*image.data[i+2];return sum;}
function span(image){let x0=Infinity,x1=-1,y0=Infinity,y1=-1;for(let y=0;y<image.height;y++)for(let x=0;x<image.width;x++)if(image.data[4*(y*image.width+x)]>1){x0=Math.min(x0,x);x1=Math.max(x1,x);y0=Math.min(y0,y);y1=Math.max(y1,y);}return {width:x1-x0+1,height:y1-y0+1};}
function copySample(label){const figure=document.createElement('figure'),c=document.createElement('canvas'),caption=document.createElement('figcaption');c.width=canvas.width;c.height=canvas.height;c.style.width='256px';c.getContext('2d').drawImage(canvas,0,0);caption.textContent=label;figure.append(c,caption);document.querySelector('#samples').append(figure);}

await test('星点的实际浮点像素积分符合星等光量，覆盖亚像素位置与 DPR',()=>{
    let worst=0;for(const dpr of [1,2])for(const m of [-2,0,4,6.5])for(const x of [0,.21,.49]){
        const f=1500,d=new THREE.Vector3(1,x/f,0).normalize().toArray(),im=render([],[star(m,d)],f,{dpr});
        const expected=2.54e-6*10**(-.4*m)*(f*dpr)**2;worst=Math.max(worst,Math.abs(energy(im)/expected-1));
    }metrics.pointFluxError=worst;assert(worst<.0001,`积分相对误差 ${worst}`);
});
await test('Luna、Echo 的真实角径映射到像素，缩放及两种局部投影一致',()=>{
    let worst=0;for(const mode of ['perspective','stereographic'])for(const dpr of [1,2])for(const id of ['Luna','Echo'])for(const f of [750,1500,6000]){
        const b=body(id),im=render([b],[],f,{dpr,mode}),actual=span(im),a=b.angularDiameter*DEG/2;
        const expected=(mode==='perspective'?2*f*Math.tan(a):4*f*Math.tan(a/2))*dpr;
        worst=Math.max(worst,Math.abs(actual.width-expected),Math.abs(actual.height-expected));
    }metrics.diskDiameterPixelError=worst;assert(worst<2,`${worst} px`);
});
await test('月面光量在相位变化、点源到可分辨盘面之间连续',()=>{
    let worst=0;for(const id of ['Luna','Echo'])for(const f of [500,1050,1250,1500,6000])for(const phase of [0,Math.PI/2,2*Math.PI/3]){
        const b=body(id,phase),im=render([b],[],f),expected=magnitudeToLux(b.observedMagnitude)*f*f;
        worst=Math.max(worst,Math.abs(energy(im)/expected-1));
    }metrics.diskFluxError=worst;assert(worst<.055,`采样积分误差 ${worst}`);
});
await test('Sol 与六颗行星按各自星等发光，Neptune 保持低于阈值',()=>{
    const checks=[];
    for(const source of base.bodies.filter(b=>b.kind!=='moon')){
        const b={...source,view:[1,0,0],altitude:0,aboveHorizon:true,lightDirection:[-Math.cos(source.phaseAngle),-Math.sin(source.phaseAngle),0]};
        const im=render([b],[],1500),actual=energy(im),expected=b.brightEnough?magnitudeToLux(b.observedMagnitude)*1500**2:0;
        if(expected)near(actual/expected,1,.055,b.id+' 光量');else near(actual,0,0,b.id+' 可见性');
        checks.push({id:b.id,visible:b.brightEnough,fluxRatio:expected?actual/expected:0});
    }assert(checks.length===7,'没有覆盖所有行星');metrics.systemFlux=checks;
});
await test('新月和半月的暗面阻挡背景亮星与后方太阳',()=>{
    let worst=0;for(const id of ['Luna','Echo'])for(const phase of [Math.PI/2,Math.PI])for(const f of [1500,6000])for(const blur of [false,true]){
        const b=body(id,phase),blank=render([b],[],f,{blur}),withStar=render([b],[star(-6)],f,{blur});
        for(let i=0;i<blank.data.length;i+=4)worst=Math.max(worst,Math.abs(withStar.data[i]-blank.data[i]));
        const sun={...body('Sol'),distance:1,angularDiameter:b.angularDiameter*.9,magnitude:-26.5,observedMagnitude:-26.5};
        const eclipsed=render([b,sun],[],f,{blur});near(energy(eclipsed),energy(blank),.01,'全食有剩余日光');
    }metrics.hiddenStarLeak=worst;assert(worst===0,`暗面漏光 ${worst}`);
});
await test('光扩散开关不改变原始光量与盘面，最终画面确有变化',()=>{
    const b=body('Luna',Math.PI/2),a=render([b],[star(0,[.9999,.01414,0])],1500),off=r.readPixels();copySample('半月 · 光扩散关');
    const c=render([b],[star(0,[.9999,.01414,0])],1500,{blur:true}),on=r.readPixels();copySample('相同半月 · 光扩散开');
    near(energy(a),energy(c),0,'开关修改了原始光量');let different=0;for(let i=0;i<on.data.length;i+=4)if(on.data[i]!==off.data[i])different++;
    assert(different>10,`开关改变的像素 ${different}`);metrics.blurChangedPixels=different;
});
await test('放大后的光扩散连续、保光量、资源可复用',()=>{
    render([],[star(0)],18000,{blur:true});const raw=energy(r.readLinear());
    r.blurBranch=1;const target=r.blurTo(r.targets[0],r.targets[1],r.targets[3],18000*2*DEG/60),data=new Float32Array(target.width*target.height*4);
    r.renderer.readRenderTargetPixels(target,0,0,target.width,target.height,data);
    const flux=energy({data})*(256/target.width)*(256/target.height);near(flux/raw,1,.025,'降采样光量');
    const half=target.height/2|0;let gaps=0,seen=false;for(let x=0;x<target.width;x++){const v=data[4*(half*target.width+x)];if(v>1e-8)seen=true;else if(seen && data[4*(half*target.width+Math.min(x+1,target.width-1))]>1e-8)gaps++;}
    assert(gaps===0,`发现光扩散空隙 ${gaps}`);const count=r.pyramid.length,generation=r.resourceGeneration;
    for(let i=0;i<5;i++)render([],[star(0)],18000,{blur:true});assert(r.pyramid.length===count && r.resourceGeneration===generation,'重复绘制新增缓冲');
    metrics.wideBlurFluxError=Math.abs(flux/raw-1);
});
await test('双半球总览实际绘制上下半球，不受背向相机影响',()=>{
    const upper=star(0,[0,0,1]),lower={...star(0,[0,0,-1]),id:'lower'};render([],[upper,lower],1500,{overview:true});
    const im=r.readLinear();let left=0,right=0;for(let y=0;y<im.height;y++)for(let x=0;x<im.width;x++){if(x<im.width/2)left+=im.data[4*(y*im.width+x)];else right+=im.data[4*(y*im.width+x)];}
    assert(left>0 && right>0,`${left}, ${right}`);near(left/right,1,.001,'南北半球光量');
});
await test('奇数画面尺寸与最近邻浮点过滤下，宽光扩散仍保持总光量',()=>{
    let worst=0;
    for(const size of [255,257,511])for(const filtered of [false,true]){
        r.linearFiltering=filtered;r.resize(size+2,size+2,1);render([],[star(0)],18000,{size,blur:true});const raw=energy(r.readLinear());
        r.blurBranch=1;const target=r.blurTo(r.targets[0],r.targets[1],r.targets[3],18000*2*DEG/60),data=new Float32Array(target.width*target.height*4);
        r.renderer.readRenderTargetPixels(target,0,0,target.width,target.height,data);
        const flux=energy({data})*(size/target.width)*(size/target.height);worst=Math.max(worst,Math.abs(flux/raw-1));
    }metrics.oddSizeBlurFluxError=worst;assert(worst<.0001,`奇数尺寸积分误差 ${worst}`);
});
await test('极宽立体视角后半球点源使用正确投影面积',()=>{
    const f=50,d=[-.5,-Math.sqrt(.75),0],im=render([],[star(0,d)],f,{size:512,mode:'stereographic'});
    near(energy(im)/(magnitudeToLux(0)*f*f*16),1,.0001,'后半球点源亮度');
});
await test('未分辨天体中心被挡住时，未被挡住的部分仍贡献光量',()=>{
    const b={...body('Echo'),distance:1},a=.00015;
    const front={...body('Front',Math.PI),distance:.01,view:[Math.cos(a),Math.sin(a),0],angularDiameter:b.angularDiameter*.8,brightEnough:false};
    const sample=bodyLightSample(b,{...base,bodies:[b,front]});assert(sample.fraction>0 && sample.fraction<1,'样本不是部分遮挡');
    const im=render([b,front],[],500),expected=magnitudeToLux(b.observedMagnitude)*500**2*sample.fraction;
    near(energy(im)/expected,1,.0001,'剩余可见光被重复扣除');metrics.partialPointVisibleFraction=sample.fraction;
});
await test('正式 9356 星表加载、夜间与白昼可见性分离',async()=>{
    const {stars}=await loadCatalog('output_20260915_galactic_01');assert(stars.length===9356,'星表变化');
    const counts=[];for(const [days,mode] of [[0,'center'],[0,'surface'],[.54,'surface']]){
        const sky=observerSky(days,sanitizeState({mode}));r.resize(256,256,1);camera.fov=fovForFocal(256,100);camera.userData.skyProjection='perspective';camera.updateProjectionMatrix();counts.push(r.render(camera,sky,stars,{focal:100,overview:mode==='center'}).eligible);
    }assert(counts[0]===9356 && counts[1]>0 && counts[1]<counts[0] && counts[2]<counts[1],counts.join(','));metrics.catalogCounts=counts;
});
await test('相同底层像素尺寸下切换 CSS 尺寸与 DPR，仍更新实际观看尺度',()=>{
    render([body('Sol')],[],910,{size:256,dpr:2});render([body('Sol')],[],910,{size:512,dpr:1});
    assert(r.width===512 && r.height===512 && r.dpr===1,'保留了旧 CSS 尺寸或 DPR');
    near(span(r.readLinear()).width,2*910*Math.tan(.531544*DEG/2),2,'DPR 切换后盘面');
});
const linear=c=>{c/=255;return c<=.04045?c/12.92:((c+.055)/1.055)**2.4;};
const rgb=(im,x,y)=>Array.from(im.data.slice(4*(Math.floor(y)*im.width+Math.floor(x)),4*(Math.floor(y)*im.width+Math.floor(x))+3));
await test('最终半月外缘与受光符合独立圆面积积分，关闭扩散没有高亮台阶',()=>{
    let worst=0,count=0;
    for(const dpr of [1,2])for(const f of [900,1500,4200]){
        const b=body('Luna',Math.PI/2);render([b],[],f,{dpr});const im=r.readPixels(),c=im.width/2,R=f*dpr*Math.tan(b.angularDiameter*DEG/2);
        for(let y=Math.floor(c-R-1);y<c+R+1;y++)for(let x=Math.ceil(c+1);x<c+R+1;x++){
            let coverage=0,lit=0;for(let j=0;j<64;j++)for(let i=0;i<64;i++)if((x+(i+.5)/64-c)**2+(y+(j+.5)/64-c)**2<=R*R){coverage+=1/4096;lit+=(x+(i+.5)/64-c)/R/4096;}
            if(coverage<.03 || coverage>.97)continue;
            const actual=linear(rgb(im,x,y)[0]);worst=Math.max(worst,Math.abs(actual-lit));count++;
        }
    }
    metrics.finalLimbCoverageError=worst;assert(count>60 && worst<.075,`外缘覆盖误差 ${worst}，样本 ${count}`);
});
function radialWidths(im,threshold=.08,maxRadius=im.width*.48){
    const c=im.width/2,values=[];
    const sample=(x,y)=>{const ix=Math.floor(x-.5),iy=Math.floor(y-.5),fx=x-.5-ix,fy=y-.5-iy;
        const at=(a,b)=>linear(rgb(im,Math.max(0,Math.min(im.width-1,a)),Math.max(0,Math.min(im.height-1,b)))[0]);
        return (at(ix,iy)*(1-fx)+at(ix+1,iy)*fx)*(1-fy)+(at(ix,iy+1)*(1-fx)+at(ix+1,iy+1)*fx)*fy;};
    for(let n=0;n<64;n++){const a=n*Math.PI/32;let edge=0;for(let d=0;d<maxRadius;d+=.1)if(sample(c+d*Math.cos(a),c+d*Math.sin(a))>threshold)edge=d;values.push(edge);}
    return {min:Math.min(...values),max:Math.max(...values)};
}
await test('可分辨月面保留受光明暗层次，高亮不会把整个亮面冲成白色',()=>{
    const b=body('Luna');render([b],[],12000);const im=r.readPixels(),center=rgb(im,128,128)[0],nearLimb=rgb(im,178,128)[0];
    assert(center>235 && nearLimb>90 && nearLimb<center-25,`月面层次 ${center}, ${nearLimb}`);metrics.lunarSurfaceContrast={center,nearLimb};copySample('满相月面 · 保留明暗层次');
});
await test('真实太阳亮度下的最终轮廓为圆，默认和旧扩散参数均无方块',()=>{
    let worst=0;for(const f of [910,1500,6000,18000])for(const blur of [false,true])for(const eye of [{},{glareFraction:.025,glareArcmin:6}]){
        const sol={...body('Sol'),magnitude:-26.5,observedMagnitude:-26.5};render([sol],[],f,{size:512,blur,eye});const im=r.readPixels(),w=radialWidths(im);
        worst=Math.max(worst,w.max-w.min);assert(w.min>0 && w.max-w.min<1.3,`f=${f}, blur=${blur}: ${JSON.stringify(w)}`);
        if(f===1500 && blur && eye.glareArcmin===6)copySample('Sol · 真实太阳亮度 · 旧扩散参数');
    }metrics.finalSolarRadialSpread=worst;
});
await test('新月、半月、满月的最终暗面遮挡不因扩散而漏出背景星',()=>{
    let worst=0;for(const phase of [0,Math.PI/2,Math.PI])for(const blur of [false,true])for(const id of ['Luna','Echo']){
        const b=body(id,phase);render([b],[],6000,{blur});const blank=r.readPixels();render([b],[star(-6)],6000,{blur});const covered=r.readPixels();
        for(let i=0;i<blank.data.length;i++)worst=Math.max(worst,Math.abs(blank.data[i]-covered.data[i]));
    }metrics.finalOccultationLeak=worst;assert(worst===0,`最终像素漏光 ${worst}`);
});
await test('开启大气的深夜保持暗蓝，月光不会把整幅天空曝光为灰',()=>{
    const samples=[];for(const day of [0,.03,.08]){
        const sky=observerSky(day,sanitizeState({mode:'surface'}));assert(sky.bodies[0].altitude<-18,'测试不是深夜');
        sky.bodies=[];camera.lookAt(1,0,.8);r.resize(256,256,1);r.render(camera,sky,[],{focal:1500});const color=rgb(r.readPixels(),128,128);
        assert(color[0]<=22 && color[1]<=30 && color[2]<=43 && color[2]>color[0]+3,`夜空 ${color}`);samples.push(color);
    }camera.lookAt(1,0,0);metrics.finalNightRGB=samples;copySample('有月光的夜空 · 大气开启');
});
await test('双半球图内外有清晰边界，极亮边缘星不漏到圆外',()=>{
    render([],[],1500,{size:512,overview:true});const blank=r.readPixels(),l=r.layout;
    const inside=rgb(blank,...l.north),outside=rgb(blank,5,5),edge=rgb(blank,l.north[0]+l.radius-.5,l.north[1]);
    assert(inside.reduce((a,v,i)=>a+Math.abs(v-outside[i]),0)>=7,'圆内外颜色不能区分');assert(Math.max(...edge)>45,'边框不可见');
    render([],[star(-6,[1,0,0])],1500,{size:512,overview:true,blur:true});const lit=r.readPixels();let leak=0;
    for(let y=0;y<512;y++)for(let x=0;x<512;x++)if(Math.hypot(x+.5-l.north[0],y+.5-l.north[1])>l.radius+2 && Math.hypot(x+.5-l.south[0],y+.5-l.south[1])>l.radius+2){
        const a=rgb(lit,x,511-y),b=rgb(blank,x,511-y);for(let c=0;c<3;c++)leak=Math.max(leak,Math.abs(a[c]-b[c]));}
    metrics.atlasBackground={inside,outside,edge,leak};assert(leak===0,`圆外漏光 ${leak}`);copySample('双半球 · 清晰边界');
});
await test('Sol 和双月缩放跨越旧点源阈值时，最终光量连续且保持实体轮廓',()=>{
    let worst=0;for(const id of ['Sol','Luna','Echo']){
        const b=body(id);if(id==='Sol')b.observedMagnitude=-26.5;
        let previous=null;for(const radius of [1.05,1.15,1.25,1.35]){
            render([b],[],radius/(b.angularDiameter*DEG/2));const im=r.readPixels();let sum=0;
            for(let i=0;i<im.data.length;i+=4)sum+=Math.max(0,linear(im.data[i])-linear(5));
            const normalized=sum/(radius*radius);if(previous!==null)worst=Math.max(worst,Math.abs(normalized/previous-1));previous=normalized;
        }
    }metrics.finalSmallDiskContinuity=worst;assert(worst<.1,`缩放跳变 ${worst}`);
});
await test('离轴点像的光扩散与投影面积同步变化，不因沿用中心核而异常增亮',()=>{
    const sum=im=>{let n=0;for(let i=0;i<im.data.length;i+=4)n+=Math.max(0,linear(im.data[i])-linear(5));return n;};
    const ratios=[];for(const [mode,angle,expected] of [['perspective',Math.PI/3,8],['stereographic',Math.PI/2,4]]){
        render([],[star(-6)],50,{mode,blur:true});const center=sum(r.readPixels());
        render([],[star(-6,[Math.cos(angle),Math.sin(angle),0])],50,{mode,blur:true});const ratio=sum(r.readPixels())/center;
        near(ratio/expected,1,.065,'离轴显示面积');ratios.push(ratio);
    }metrics.finalOffAxisAreas=ratios;
});
await test('中心在屏外的亮星仍贡献可见光尾，越过旧 3px 边界不会消失',()=>{
    const f=6000,d=new THREE.Vector3(1,-132/f,0).normalize().toArray();render([],[star(-6,d)],f,{blur:true});const im=r.readPixels();let peak=0;
    for(let y=100;y<156;y++)for(let x=250;x<256;x++)peak=Math.max(peak,rgb(im,x,y)[0]);assert(peak>25,`屏边没有星光 ${peak}`);metrics.offscreenStarPeak=peak;
});
await test('V2 双半球星点与 V1 Canvas 星点使用相同大小和亮度规则',()=>{
    const c=document.createElement('canvas');c.width=c.height=512;const ctx=c.getContext('2d'),painter=new AtlasStarPainter(ctx);let worst=0;
    for(const mag of [-2,0,3,6.5]){
        render([],[star(mag,[0,0,1])],1500,{size:512,overview:true});const im=r.readPixels(),point=r.project([0,0,1]);
        ctx.fillStyle='#050911';ctx.fillRect(0,0,512,512);painter.draw({app_mag:mag,color_hex:'#ffffff'},point,r.focal/SYMBOL_REFERENCE.perspectiveFocal*DEFAULT_DISPLAY_SCALE);
        for(let y=Math.floor(point.y)-4;y<=point.y+4;y++)for(let x=Math.floor(point.x)-4;x<=point.x+4;x++){
            const expected=ctx.getImageData(x,y,1,1).data,actual=rgb(im,x,511-y);for(let channel=0;channel<3;channel++)worst=Math.max(worst,Math.abs(actual[channel]-expected[channel]));
        }
    }painter.clear();metrics.chartV1PixelDifference=worst;assert(worst<=2,`两版同一星点像素差 ${worst}`);
});
await test('双半球缩略图采用可读星点，正式星表不是近乎空白的图',async()=>{
    const {stars}=await loadCatalog('output_20260915_galactic_01'),counts=[];
    for(const displayScale of [1.6,DEFAULT_DISPLAY_SCALE]){
        render([],stars,1500,{size:1024,overview:true,displayScale});const im=r.readPixels(),l=r.layout,count=[0,0];
        for(let y=0;y<1024;y++)for(let x=0;x<1024;x++)for(const [i,c] of [l.north,l.south].entries())if(Math.hypot(x+.5-c[0],1023.5-y-c[1])<l.radius-4 && rgb(im,x,y)[0]>24)count[i]++;
        counts.push(count);
    }
    const old=counts[0].reduce((a,b)=>a+b,0),bright=counts[1].reduce((a,b)=>a+b,0);
    // A fixed contrast threshold checks visible signal against the old preset.
    // Both hemispheres must improve, independent of the star population split.
    assert(counts[1].every((n,i)=>n>500 && n>counts[0][i]*3) && bright>old*3,`复位整图亮像素（旧/新、北/南） ${JSON.stringify(counts)}`);
    metrics.chartStarPixels={threshold:24,old:counts[0],current:counts[1]};copySample(`正式星表 · 默认 ${DEFAULT_DISPLAY_SCALE} 倍`);
});
await test('V2 最终背景星按星等连续分层，亮星光点大于暗星',()=>{
    const areas=[],light=[];for(const m of [-2,0,2,4,6]){render([],[star(m)],1500,{blur:true});const im=r.readPixels();let area=0,sum=0;for(let i=0;i<im.data.length;i+=4){if(im.data[i]>25)area++;sum+=Math.max(0,linear(im.data[i])-linear(5));}areas.push(area);light.push(sum);}
    for(let i=1;i<areas.length;i++)assert(areas[i]<=areas[i-1] && light[i]<light[i-1]*.85,`星点层次 ${areas}; ${light}`);metrics.finalStarAreas=areas;metrics.finalStarEffectiveAreas=light;
});
await test('6 倍增强只改变最终画面，物理光量与物理遮挡完全保留',()=>{
    const b=body('Luna'),s=star(-2,new THREE.Vector3(1,.012,0).normalize().toArray());
    const a=render([b],[s],1500,{displayScale:1}),c=render([b],[s],1500,{displayScale:DEFAULT_DISPLAY_SCALE});
    let difference=0;for(let i=0;i<a.data.length;i++)difference=Math.max(difference,Math.abs(a.data[i]-c.data[i]));
    near(difference,0,0,'显示增强修改物理缓冲');metrics.enhancedPhysicalDifference=difference;
});
await test('两种局部视图的日月显示直径按同一倍率扩大，双月比例不变',()=>{
    const measures=[];let worst=0;
    for(const mode of ['perspective','stereographic'])for(const dpr of [1,2])for(const id of ['Sol','Luna','Echo']){
        const b=body(id);if(id==='Sol'){b.angularDiameter=.44084;b.magnitude=b.observedMagnitude=-26.5;}
        render([b],[],1500,{mode,dpr,displayScale:DEFAULT_DISPLAY_SCALE});const im=r.readPixels(),m={width:0,height:0};let x0=Infinity,x1=-1,y0=Infinity,y1=-1;
        for(let y=0;y<im.height;y++)for(let x=0;x<im.width;x++)if(rgb(im,x,y)[0]>25){x0=Math.min(x0,x);x1=Math.max(x1,x);y0=Math.min(y0,y);y1=Math.max(y1,y);}
        m.width=x1-x0+1;m.height=y1-y0+1;
        const a=Math.atan(DEFAULT_DISPLAY_SCALE*Math.tan(b.angularDiameter*DEG/2)),expected=(mode==='perspective'?3000*Math.tan(a):6000*Math.tan(a/2))*dpr;
        worst=Math.max(worst,Math.abs(m.width-expected),Math.abs(m.height-expected));measures.push({mode,dpr,id,...m,expected});
    }
    assert(worst<2,`增强盘径偏差 ${worst}`);metrics.enhancedDisks={worst,measures};copySample('Echo · 默认增强后盘面');
});
await test('增强后的暗面外扩区同样遮星，两种投影与双半球均不透明',()=>{
    let worst=0;
    for(const mode of ['perspective','stereographic','atlas'])for(const id of ['Luna','Echo']){
        const b=body(id,Math.PI),a=b.angularDiameter*DEG,overview=mode==='atlas';
        if(overview){b.view=[0,0,1];b.lightDirection=[0,0,1];}
        const direction=overview?[Math.sin(a),0,Math.cos(a)]:[Math.cos(a),Math.sin(a),0];
        const options={mode:overview?'perspective':mode,overview,size:512,displayScale:DEFAULT_DISPLAY_SCALE};
        render([b],[],1500,options);const blank=r.readPixels();render([b],[star(-6,direction)],1500,options);const covered=r.readPixels();
        for(let i=0;i<blank.data.length;i++)worst=Math.max(worst,Math.abs(blank.data[i]-covered.data[i]));
    }
    near(worst,0,0,'增强暗面透星');metrics.enhancedDarkSideLeak=worst;
});
function centeredAtlas(b,zoom=8,size=512,dpr=1){
    render([],[],1500,{overview:true,size,dpr});const p=r.project(b.view);
    return {overview:true,size,dpr,atlasTransform:{zoom,panX:(size/2-p.x)*zoom,panY:(size/2-p.y)*zoom}};
}
await test('V1 与 V2 平面 Sol 在南北半球、离极点位置和缩放后均为圆形',()=>{
    const c=document.createElement('canvas'),painter=new AtlasBodyPainter(c.getContext('2d'));let worst=0,diameterDifference=0,glareSpread=0;
    for(const dec of [70,35,10,-10,-60])for(const zoom of [4,12])for(const dpr of [1,2]){
        const sol={...body('Sol'),view:[Math.cos(dec*DEG),0,Math.sin(dec*DEG)],angularDiameter:.44084,magnitude:-26.5,observedMagnitude:-26.5};
        const options=centeredAtlas(sol,zoom,512,dpr);render([sol],[],1500,options);
        // Limit measurements to the solar neighborhood so the atlas's own
        // hemisphere outline cannot be mistaken for a distant solar limb.
        const extent=r.focal*dpr*3*6*.44084*DEG/2;
        const gpu=radialWidths(r.readPixels(),.08,extent);worst=Math.max(worst,gpu.max-gpu.min);
        c.width=c.height=512*dpr;const ctx=c.getContext('2d');ctx.scale(dpr,dpr);ctx.fillStyle='#050911';ctx.fillRect(0,0,512,512);
        painter.drawSky(r.sky,r.layout,r.orientation,{planetGlow:false});
        const canvasImage={data:ctx.getImageData(0,0,c.width,c.height).data,width:c.width,height:c.height},v1=radialWidths(canvasImage,.08,extent);
        worst=Math.max(worst,v1.max-v1.min);diameterDifference=Math.max(diameterDifference,Math.abs((gpu.min+gpu.max)-(v1.min+v1.max)));
        assert(gpu.min>2 && v1.min>2,'日面丢失');
        ctx.fillRect(0,0,512,512);painter.drawSky(r.sky,r.layout,r.orientation,{planetGlow:false,solarGlow:true});
        const glow=radialWidths({data:ctx.getImageData(0,0,c.width,c.height).data,width:c.width,height:c.height},.08,extent);
        glareSpread=Math.max(glareSpread,glow.max-glow.min);
    }
    painter.clear();assert(worst<1.3 && diameterDifference<1.4 && glareSpread<1.3,`日面方向差 ${worst}，两版直径差 ${diameterDifference}，柔光方向差 ${glareSpread}`);
    metrics.atlasSolarRoundness={maxRadialSpread:worst,maxV1V2DiameterDifference:diameterDifference,glareSpread,samples:20};copySample('平面 Sol · 离轴与放大仍为圆形');
});
await test('圆形日面按实际显示边缘遮星，旧椭圆外扩区域不再误挡星点',()=>{
    const dec=10*DEG,sol={...body('Sol'),view:[Math.cos(dec),0,Math.sin(dec)],angularDiameter:.44084,distance:1,brightEnough:false};
    const options=centeredAtlas(sol,16);render([sol],[],1500,options);const blank=r.readPixels(),p=r.project(sol.view),l=r.layout,c=l.north;
    const radial=[(p.x-c[0])/Math.hypot(p.x-c[0],p.y-c[1]),(p.y-c[1])/Math.hypot(p.x-c[0],p.y-c[1])],R=r.bodyCircles(r.sky.bodies[0])[0].radius;
    const directionAt=(x,y)=>{const theta=Math.hypot(x-c[0],y-c[1])/l.radius*Math.PI/2,phi=Math.atan2(-(x-c[0]),y-c[1])+r.orientation*DEG;return [Math.sin(theta)*Math.cos(phi),Math.sin(theta)*Math.sin(phi),Math.cos(theta)];};
    const inside=directionAt(p.x+radial[0]*R*.93,p.y+radial[1]*R*.93),outside=directionAt(p.x-radial[1]*R*1.12,p.y+radial[0]*R*1.12);
    render([sol],[star(-2,inside)],1500,options);const covered=r.readPixels();let leak=0;for(let i=0;i<blank.data.length;i++)leak=Math.max(leak,Math.abs(blank.data[i]-covered.data[i]));
    near(leak,0,0,'日面内的背景星仍有漏光');
    render([sol],[star(-2,outside)],1500,options);const exposed=r.readPixels();let visible=0;for(let i=0;i<blank.data.length;i+=4)visible+=Math.max(0,exposed.data[i]-blank.data[i]);
    assert(visible>500,'圆外星点仍被旧椭圆误挡');metrics.circularSolarOccultation={coveredLeak:leak,exposedSignal:visible};
});
await test('Sol 越过双半球边界保留两侧圆弧，圆外无漏光且物理缓冲不变',()=>{
    let leak=0,physicalDifference=0;const sides=[];
    for(const dec of [-.4,0,.4]){
        const b={...body('Sol'),view:[Math.cos(dec*DEG),0,Math.sin(dec*DEG)],magnitude:-26.5,observedMagnitude:-26.5};
        const a=render([b],[],1500,{overview:true,size:1024,displayScale:1});
        render([],[],1500,{overview:true,size:1024});const blank=r.readPixels();
        const enhanced=render([b],[],1500,{overview:true,size:1024,blur:true}),lit=r.readPixels(),l=r.layout,count=[0,0];
        for(let i=0;i<a.data.length;i++)physicalDifference=Math.max(physicalDifference,Math.abs(a.data[i]-enhanced.data[i]));
        for(let y=0;y<1024;y++)for(let x=0;x<1024;x++){
            const d=[l.north,l.south].map(c=>Math.hypot(x+.5-c[0],1023.5-y-c[1])-l.radius),k=4*(y*1024+x);
            if(d.every(v=>v>2))leak=Math.max(leak,Math.abs(lit.data[k]-blank.data[k]));
            d.forEach((v,j)=>{if(v<0 && lit.data[k]>150)count[j]++;});
        }
        assert(count.every(v=>v>0),`接缝缺半边 ${dec}: ${count}`);sides.push(count);
    }
    near(leak,0,0,'太阳光漏到图外');near(physicalDifference,0,0,'圆形显示修改物理缓冲');metrics.solarSeam={leak,physicalDifference,sides};
});
await test('远离视野的亮星不会借助离轴拉伸把整片天空照灰',()=>{
    let worst=0;
    for(const mode of ['perspective','stereographic'])for(const displayScale of [1,DEFAULT_DISPLAY_SCALE]){
        const options={mode,blur:true,displayScale};render([],[],1500,options);const blank=r.readPixels();
        const outside=[60,85,89,89.9].map(degrees=>star(-6,[Math.cos(degrees*DEG),Math.sin(degrees*DEG),0]));
        render([],outside,1500,options);const im=r.readPixels();for(let i=0;i<im.data.length;i++)worst=Math.max(worst,Math.abs(im.data[i]-blank.data[i]));
    }
    near(worst,0,0,'远方星光污染背景');metrics.farOffscreenLeak=worst;
});
await test('加载帧之后扩充星表，超过第 1024 个实例的恒星仍完整绘出',()=>{
    const warm=new EyeSkyRenderer(document.createElement('canvas')),fresh=new EyeSkyRenderer(document.createElement('canvas'));
    try{
        camera.userData.skyProjection='perspective';camera.aspect=1;camera.fov=fovForFocal(256,1500);camera.updateProjectionMatrix();
        const sky={...base,bodies:[]},options={focal:1500,displayScale:1};
        warm.resize(256,256);fresh.resize(256,256);
        warm.render(camera,sky,[star(0)],options);
        const initial=warm.pointGeometry,stars=Array.from({length:1300},()=>star(6,new THREE.Vector3(1,-.04,0).normalize().toArray()));
        stars.push(star(-2,new THREE.Vector3(1,.04,0).normalize().toArray()));
        warm.render(camera,sky,stars,options);fresh.render(camera,sky,stars,options);
        const a=warm.readPixels(),b=fresh.readPixels();let worst=0;for(let i=0;i<a.data.length;i++)worst=Math.max(worst,Math.abs(a.data[i]-b.data[i]));
        assert(warm.pointGeometry!==initial,'扩容没有刷新实例上限');near(worst,0,0,'后加载的恒星被截断');
        const stable=warm.pointGeometry;warm.render(camera,sky,stars,options);assert(warm.pointGeometry===stable,'每帧重建绘图资源');
        metrics.catalogGrowth={sources:stars.length,maxPixelDifference:worst,stableAfterGrowth:true};
    }finally{warm.dispose();fresh.dispose();}
});
r.dispose();
await runNativeStarChecks({test,assert,metrics});

// Real production pages, kept in a dedicated test window. Save and restore
// only V2 state; the original application's storage is never written.
const oldKey='terrax-sky-motion-v2',old=localStorage.getItem(oldKey);let first,second;
// Pin the historical point-source fixture; newer default catalogues also have
// diffuse objects and a different, independently validated point count.
const testValues=new Map([[STORAGE_KEY,JSON.stringify(sanitizeState({folder:'output_20260915_galactic_01'}))]]),testWindows=new Set();
window.testStorageFor=win=>{testWindows.add(win);return {
    getItem:key=>testValues.get(key)??null,
    setItem(key,value){testValues.set(key,String(value));for(const other of testWindows)if(other!==win)setTimeout(()=>other.dispatchEvent(new other.StorageEvent('storage',{key,newValue:String(value)})),0);},
    removeItem:key=>testValues.delete(key),
};};
function eventValue(doc,id,value){const el=doc.getElementById(id);el.value=String(value);el.dispatchEvent(new Event('input',{bubbles:true}));el.dispatchEvent(new Event('change',{bubbles:true}));}
async function frame(path,setup=''){
    const f=document.createElement('iframe'),url=new URL(path,location.href),html=await (await fetch(url)).text();
    f.srcdoc=html.replace('<head>',`<head><base href="${new URL('.',url)}"><script>Object.defineProperty(window,'localStorage',{value:parent.testStorageFor(window)});${setup}<\/script>`);
    document.querySelector('#frames').append(f);await until(()=>f.contentWindow?.terraxV2?.status?.stars===9356,25000);return f;
}
try{
    await test('两个 V2 正式入口可加载，均采用相同校准角尺度',async()=>{
        first=await frame('../star_map.html');second=await frame('../sky_atlas.html');
        assert(second.contentWindow.terraxV2.status.overview,'平面页没有默认双半球');
        assert([first,second].every(f=>f.contentWindow.terraxV2.status.displayScale===DEFAULT_DISPLAY_SCALE),'正式页面没有采用新的默认大小');
        second.contentDocument.getElementById('view-natural').click();await until(()=>!second.contentWindow.terraxV2.status.overview && second.contentWindow.terraxV2.status.focal===first.contentWindow.terraxV2.status.focal);
        const a=first.contentWindow.terraxV2.status,b=second.contentWindow.terraxV2.status;
        assert(a.state==='ready' && b.state==='ready','页面未 ready');near(a.focal,b.focal,1e-8,'两页尺度');
    });
    await test('两页共享天体显示大小；复位保留选择；肉眼 1× 确实关闭增强',async()=>{
        const d=second.contentDocument;eventValue(d,'display-scale',DEFAULT_DISPLAY_SCALE);
        await until(()=>first.contentWindow.terraxV2.status.displayScale===DEFAULT_DISPLAY_SCALE);
        d.getElementById('view-overview').click();await until(()=>second.contentWindow.terraxV2.status.overview);
        assert(second.contentWindow.terraxV2.status.displayScale===DEFAULT_DISPLAY_SCALE,'复位丢失增强');
        d.getElementById('view-natural').click();await until(()=>[first,second].every(f=>f.contentWindow.terraxV2.status.displayScale===1));
        assert(second.contentWindow.terraxV2.status.magnification===1,'肉眼按钮未恢复相机倍率');
    });
    await test('两页共享时刻与光扩散开关，切换及缩放后仍然对齐',async()=>{
        const d=first.contentDocument;eventValue(d,'time-days',7.25);d.getElementById('toolbar-blur').click();d.getElementById('view-plus').click();
        await until(()=>second.contentWindow.terraxV2.status.time===7.25 && second.contentWindow.terraxV2.status.blur===false && second.contentWindow.terraxV2.status.magnification===1.4);
        near(first.contentWindow.terraxV2.status.focal,second.contentWindow.terraxV2.status.focal,1e-8,'缩放后的两页尺度');
    });
    await test('调整窗口只改变可见范围，不改变校准后的盘面像素尺度',async()=>{
        const before=first.contentWindow.terraxV2.status.focal;first.style.width='780px';first.style.height='600px';await delay(350);
        near(first.contentWindow.terraxV2.status.focal,before,1e-8,'resize 改变焦距');
    });
    await test('平面总览与天体定位可往返，恢复到肉眼尺度',async()=>{
        const d=second.contentDocument;d.getElementById('view-overview').click();await until(()=>second.contentWindow.terraxV2.status.overview);
        d.querySelector('[data-body="Echo"]').click();assert(second.contentWindow.terraxV2.status.overview,'定位意外退出平面图');
        const z=second.contentWindow.terraxV2.status.atlasZoom;d.getElementById('view-plus').click();await until(()=>second.contentWindow.terraxV2.status.atlasZoom>z);
        d.getElementById('view-natural').click();await until(()=>d.getElementById('view-title').textContent.includes('1.00×'));
        assert(second.contentWindow.terraxV2.status.projection==='stereographic','平面局部投影不正确');
    });
    await test('极宽视角下滚轮缩小仍然缩小，不突然反向放大',async()=>{
        const d=second.contentDocument;eventValue(d,'eye-density',5);eventValue(d,'eye-distance',20);await delay(100);
        const before=second.contentWindow.terraxV2.status.focal;d.getElementById('eye-canvas').dispatchEvent(new WheelEvent('wheel',{deltaY:80,cancelable:true}));await delay(150);
        assert(second.contentWindow.terraxV2.status.focal<before,'滚轮缩小反向');
        eventValue(d,'eye-distance',60);d.getElementById('eye-auto-density').click();d.getElementById('view-natural').click();await delay(100);
    });
    await test('两页持续播放 12 秒并同步暂停，画布保持有效',async()=>{
        const d=first.contentDocument;eventValue(d,'time-days',0);d.getElementById('time-play').click();const start=performance.now(),frames=first.contentWindow.terraxV2.status.frames;
        await delay(12050);d.getElementById('time-play').click();await delay(160);

        const a=first.contentWindow.terraxV2.status,b=second.contentWindow.terraxV2.status;
        assert(a.state==='ready' && b.state==='ready' && a.frames>frames+50 && a.time>.3,'播放没有推进');near(a.time,b.time,1e-8,'暂停时刻');
        metrics.playback={seconds:(performance.now()-start)/1000,frames:a.frames-frames,day:a.time};
    });
    await test('暂停时模拟丢失并恢复 GPU 上下文，画面自动恢复',async()=>{
        const win=first.contentWindow,stage=first.contentDocument.getElementById('eye-stage'),before=win.terraxV2.status.time;
        const gl=first.contentDocument.getElementById('eye-canvas').getContext('webgl2'),extension=gl.getExtension('WEBGL_lose_context');assert(extension,'缺少 context-loss 扩展');
        extension.loseContext();await until(()=>stage.dataset.state==='lost');await delay(250);extension.restoreContext();await until(()=>stage.dataset.state==='ready');
        near(win.terraxV2.status.time,before,1e-8,'恢复改变暂停时刻');assert(win.terraxV2.status.stars===9356,'恢复丢失星表');metrics.simulatedContextRecovery=true;
    });
    await test('星表列表临时失败时天体仍可播放，并可重试恢复',async()=>{
        first.remove();second.remove();testWindows.clear();testValues.set(STORAGE_KEY,JSON.stringify(sanitizeState({folder:'output_20260915_galactic_01'})));
        const setup=`const originalFetch=window.fetch;window.fetch=(...args)=>{if(String(args[0]).includes('folders.json')&&!window.failedOnce){window.failedOnce=true;return Promise.resolve(new Response('temporary fixture',{status:503}));}return originalFetch(...args);};
        const timer=setInterval(()=>{const b=document.getElementById('catalog-retry');if(!b||b.hidden)return;clearInterval(timer);const before=Number(document.getElementById('eye-stage').dataset.frames)||0;document.getElementById('time-play').click();const started=performance.now(),progress=setInterval(()=>{if(+document.getElementById('eye-stage').dataset.frames<=before && performance.now()-started<8000)return;clearInterval(progress);window.continuedAfterFailure=+document.getElementById('eye-stage').dataset.frames>before;document.getElementById('time-play').click();b.click();},100);},40);`;
        const f=await frame('../star_map.html',setup);
        try{await until(()=>f.contentWindow.continuedAfterFailure!==undefined);assert(f.contentWindow.continuedAfterFailure===true,'列表失败冻结天体绘图');}
        finally{testWindows.delete(f.contentWindow);f.remove();}
    });
    await test('原版保存的时间和设置没有被 V2 写入',()=>assert(localStorage.getItem(oldKey)===old,'原版存储变化'));
}finally{
    first?.remove();second?.remove();testWindows.clear();
}
document.querySelector('#status').textContent=`${total-failures}/${total} 通过；${failures} 失败`;
document.querySelector('#metrics').textContent=JSON.stringify(metrics,null,2);

window.renderCheckReport={total,failures,metrics};
