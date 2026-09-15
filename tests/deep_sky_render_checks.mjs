import * as THREE from 'three';
import {EyeSkyRenderer} from '../web/v2/eye_renderer.mjs';
import {EYE_DEFAULTS,fovForFocal,observerSky} from '../web/v2/eye_model.mjs';
import {sanitizeState} from '../web/v2/sky_state.mjs';
import {galacticToEquatorial,projectHemisphere,skyState,applyFrame,observerFrame,DEG} from '../web/shared/solar_system.mjs';
import {DiffuseAtlasPainter,attachDiffuseSources} from '../web/shared/deep_sky_view.js';
import {eclipticDirection,zodiacLabels,zodiacLines} from '../web/shared/zodiac.mjs';
let total=0,failures=0;const metrics={};
const assert=(v,m)=>{if(!v)throw Error(m);};
function test(name,fn){total++;const li=document.createElement('li');try{fn();li.className='pass';li.textContent='通过：'+name;}catch(e){failures++;li.className='fail';li.textContent='失败：'+name+' — '+e.message;}document.querySelector('#results').append(li);}
const w=512,h=256,data=new Float32Array(w*h);
for(let y=0;y<h;y++)for(let x=0;x<w;x++){const l=(x+.5)/w*2*Math.PI,b=(y+.5)/h*Math.PI-Math.PI/2;data[y*w+x]=.02*Math.max(0,Math.cos(b)*Math.cos(l))**32;}
const texture=new THREE.DataTexture(data,w,h,THREE.RedFormat,THREE.FloatType);texture.wrapS=THREE.RepeatWrapping;texture.minFilter=texture.magFilter=THREE.LinearFilter;texture.needsUpdate=true;
const direction=galacticToEquatorial(0,0),canvas=document.createElement('canvas'),painter=new EyeSkyRenderer(canvas),camera=new THREE.PerspectiveCamera(60,1,.01,10);camera.up.set(0,0,1);camera.lookAt(new THREE.Vector3(...direction));
const sky=observerSky(0,sanitizeState({mode:'center'}));sky.bodies=[];sky.eye={...EYE_DEFAULTS,opticalBlur:false};sky.environment={atmosphere:false,extinction:0,nightL:0,dayL:0,moons:[],adaptation:.005};painter.setDiffuse(texture);
const center=im=>im.data[4*((Math.floor(im.height/2))*im.width+Math.floor(im.width/2))];
function render(f=500,options={},s=sky){painter.resize(256,256,options.dpr||1);camera.userData.skyProjection=options.mode||'perspective';camera.fov=fovForFocal(256,f,camera.userData.skyProjection);camera.updateProjectionMatrix();painter.render(camera,s,[],{focal:f,displayScale:1,deepSky:true,...options});return painter.readLinear();}
test('V2 HDR 采样对齐银河方向，表面亮度不随焦距与 DPR 改变',()=>{
    const values=[];for(const f of [300,600])for(const dpr of [1,2])for(const mode of ['perspective','stereographic'])values.push(center(render(f,{dpr,mode})));
    metrics.peakRadiance=values;assert(values.every(x=>Math.abs(x/.02-1)<.004),String(values));
});
test('关闭星云图层确实去除弥散光，保留独立背景与恒星入口',()=>{assert(center(render(500,{deepSky:false}))===0,'图层仍在发光');});
test('不发光的新月完整遮住星云，包括月面暗侧',()=>{
    const b={id:'Luna',kind:'moon',color:'#ffffff',view:direction,angularDiameter:8,distance:.01,altitude:30,brightEnough:false,visible:false,lightDirection:direction,observedMagnitude:Infinity};
    const im=render(500,{}, {...sky,bodies:[b]});assert(center(im)===0,'月面泄漏弥散光');
});
test('地表模式正确裁掉地平线以下的星云',()=>{
    // This fixture direction has negative equatorial z; an identity surface
    // frame places the source below the horizon and must suppress its light.
    const im=render(500,{}, {...sky,frame:{...sky.frame,surface:true}});assert(center(im)===0,'地平下弥散光可见');
});
test('V1 双半球星云位置与背景星采用同一坐标，缩放后仍对齐',()=>{
    const c=document.createElement('canvas');c.width=800;c.height=400;const ctx=c.getContext('2d'),atlas=new DiffuseAtlasPainter(ctx);
    const layout={radius:160,north:[190,200],south:[610,200]},s=skyState(0,{mode:'center'});
    for(const zoom of [1,1.5]){
        const p=projectHemisphere(direction,160,layout,13.564125),pan=[-(p.x-400)*zoom,-(p.y-200)*zoom];ctx.clearRect(0,0,800,400);
        atlas.draw(s,texture,layout,13.564125,{width:800,height:400,dpr:1,zoom,pan,enabled:true});
        const v=ctx.getImageData(400,200,1,1).data;assert(v[0]>100,`中心未对齐 ${v}`);
    }document.querySelector('#samples').append(c);
});
test('15 个天区标签、15 条经度边界及带边界共用固定黄道',()=>{
    assert(zodiacLabels.length===15 && zodiacLines.length===18,'天区数量错误');
    for(const day of [0,200,30000]){const f=observerFrame(day,{mode:'surface',latitude:45});for(let i=0;i<15;i++){const v=applyFrame(eclipticDirection(i*24),f);assert(Math.abs(Math.hypot(...v)-1)<1e-12,'天区方向错误');}}
});
const sourceFixture=(profile,angularScale,extra={})=>({id:'continuous-fixture',profile,angular_scale_rad:angularScale,profile_truncation:3,gal_lon:0,gal_lat:0,app_mag:4,v_flux:10**(-.4*4),...extra});
function sourceTexture(objects){const t=new THREE.DataTexture(new Float32Array(4),2,2,THREE.RedFormat,THREE.FloatType);t.needsUpdate=true;return attachDiffuseSources(t,objects);}
function integratedFlux(im,f,mode){
    let sum=0;
    const corner=(x,y)=>Math.atan2(x*y,Math.sqrt(1+x*x+y*y));
    for(let y=0;y<im.height;y++)for(let x=0;x<im.width;x++){
        const px=(x+.5-im.width/2)/f,py=(y+.5-im.height/2)/f;
        let omega;
        if(mode==='stereographic')omega=1/(f*f*(1+(px*px+py*py)/4)**2);
        else{const x0=px-.5/f,x1=px+.5/f,y0=py-.5/f,y1=py+.5/f;omega=corner(x1,y1)-corner(x0,y1)-corner(x1,y0)+corner(x0,y0);}
        sum+=im.data[4*(y*im.width+x)]*omega;
    }return sum;
}
test('连续剖面跨越亚像素、缩放与高分屏时守恒总光量',()=>{
    let worst=0,witness;const cases=[];
    const right=new THREE.Vector3().setFromMatrixColumn(camera.matrixWorld,0),up=new THREE.Vector3().setFromMatrixColumn(camera.matrixWorld,1);
    for(const profile of ['plummer','gaussian','ionized_gaussian'])for(const a of [.0001,.0005,.002,.008]){
        const o=sourceFixture(profile,a),t=sourceTexture([o]);painter.setDiffuse(t);
        for(const focal of [250,750])for(const dpr of [1,2])for(const mode of ['perspective','stereographic'])for(const offset of [0,.37]){
            camera.lookAt(new THREE.Vector3(...direction).addScaledVector(right,offset/focal).addScaledVector(up,offset*.7/focal));
            const im=render(focal,{dpr,mode}),ratio=integratedFlux(im,focal*dpr,mode)/(o.v_flux*2.54e-6),error=Math.abs(ratio-1),entry={profile,a,focal,dpr,mode,offset,ratio};
            if(error>worst){worst=error;witness=entry;}cases.push(entry);
        }t.dispose();
    }
    camera.lookAt(new THREE.Vector3(...direction));metrics.continuousFlux={worstRelativeError:worst,witness,cases};assert(worst<.025,JSON.stringify(witness));
});
test('高倍连续星团核心为平滑圆形，不继承旧光图的方形像素',()=>{
    const t=sourceTexture([sourceFixture('plummer',.012)]);painter.setDiffuse(t);
    const im=render(1000),sample=(dx,dy)=>{
        // Interpolate at equal distances from the optical center; integer
        // pixels are centered at half-integers, not at (128, 128).
        const x=127.5+dx,y=127.5+dy,ix=Math.floor(x),iy=Math.floor(y),fx=x-ix,fy=y-iy;
        const at=(i,j)=>im.data[4*(j*im.width+i)];
        return (1-fy)*((1-fx)*at(ix,iy)+fx*at(ix+1,iy))+fy*((1-fx)*at(ix,iy+1)+fx*at(ix+1,iy+1));
    };
    const radii=[4,8,16,24],ratios=radii.map(r=>sample(r,0)/sample(0,r));
    const diagonal=sample(16/Math.sqrt(2),16/Math.sqrt(2))/sample(16,0);
    metrics.continuousShape={axisRatios:ratios,equalRadiusDiagonalRatio:diagonal};
    assert(ratios.every(v=>Math.abs(v-1)<.015)&&Math.abs(diagonal-1)<.04,'方形或方向性核心 '+JSON.stringify(metrics.continuousShape));
    const c=document.createElement('canvas');c.width=c.height=256;const ctx=c.getContext('2d'),pixels=painter.readPixels(),image=ctx.createImageData(256,256);
    for(let y=0;y<256;y++)image.data.set(pixels.data.subarray(4*(255-y)*256,4*(256-y)*256),4*y*256);ctx.putImageData(image,0,0);document.querySelector('#samples').append(c);t.dispose();
});
test('总光量低于必要门槛的深空记录不发光，切换对象与释放纹理无残留',()=>{
    const t=sourceTexture([sourceFixture('plummer',.002,{app_mag:10,v_flux:1e-4})]);painter.setDiffuse(t);
    const im=render(500);assert(im.data.every((v,i)=>i%4===3||v===0),'过暗对象仍发光');
    let disposed=0;for(const key of ['sources','cells','indices'])t.userData.profiles[key].addEventListener('dispose',()=>disposed++);
    t.dispose();assert(disposed===3,'连续剖面纹理未释放');
});
test('连续星团核心在银河经度接缝和两极无缺失',()=>{
    const ratios=[];
    for(const [lon,lat] of [[0,0],[359.9999,0],[120,89.9999],[320,-89.9999]]){
        const o=sourceFixture('gaussian',.002,{gal_lon:lon,gal_lat:lat}),t=sourceTexture([o]);painter.setDiffuse(t);camera.lookAt(new THREE.Vector3(...galacticToEquatorial(lon,lat)));
        const im=render(600),ratio=integratedFlux(im,600,'perspective')/(o.v_flux*2.54e-6);ratios.push(ratio);t.dispose();
    }
    camera.lookAt(new THREE.Vector3(...direction));metrics.continuousSeams=ratios;assert(ratios.every(r=>Math.abs(r-1)<.01),JSON.stringify(ratios));
});
test('连续弥散源同样受月面暗侧完整遮挡',()=>{
    const t=sourceTexture([sourceFixture('gaussian',.004)]);painter.setDiffuse(t);
    const b={id:'Luna',kind:'moon',color:'#ffffff',view:direction,angularDiameter:8,distance:.01,altitude:30,brightEnough:false,visible:false,lightDirection:direction,observedMagnitude:Infinity};
    assert(center(render(500,{}, {...sky,bodies:[b]}))===0,'连续光穿过月面暗侧');t.dispose();
});
test('双半球连续光量按真实立体角守恒，V1 同一方向在高倍地图中仍对齐',()=>{
    const o=sourceFixture('plummer',.002),t=sourceTexture([o]);painter.setDiffuse(t);const ratios=[];
    for(const zoom of [1,8,33.84])for(const dpr of [1,2]){
        const atlasTransform={zoom,panX:0,panY:0},options={overview:true,dpr,atlasTransform};
        render(500,options);const anchor=painter.project(direction);atlasTransform.panX=128-anchor.x;atlasTransform.panY=128-anchor.y;
        const im=render(500,options),l=painter.layout,f=painter.focal*dpr;let flux=0;
        for(let y=0;y<im.height;y++)for(let x=0;x<im.width;x++){
            const px=(x+.5)/dpr,py=(im.height-y-.5)/dpr;
            let r=Math.hypot(px-l.north[0],py-l.north[1]);if(r>l.radius)r=Math.hypot(px-l.south[0],py-l.south[1]);if(r>l.radius)continue;
            const a=r/l.radius*Math.PI/2,omega=(a<1e-8?1:Math.sin(a)/a)/(f*f);flux+=im.data[4*(y*im.width+x)]*omega;
        }
        ratios.push({zoom,dpr,ratio:flux/(o.v_flux*2.54e-6)});
    }
    metrics.continuousAtlasFlux=ratios;assert(ratios.every(v=>Math.abs(v.ratio-1)<.025),JSON.stringify(ratios));
    const c=document.createElement('canvas');c.width=800;c.height=400;const ctx=c.getContext('2d'),atlas=new DiffuseAtlasPainter(ctx),layout={radius:160,north:[190,200],south:[610,200]};
    const p=projectHemisphere(direction,160,layout,13.564125),zoom=33.84,pan=[-(p.x-400)*zoom,-(p.y-200)*zoom];
    atlas.draw(skyState(0,{mode:'center'}),t,layout,13.564125,{width:800,height:400,dpr:1,zoom,pan,enabled:true});
    const v=ctx.getImageData(400,200,1,1).data;assert(v[0]>50,`V1 连续核心未对齐 ${v}`);
    atlas.layer.mesh.geometry.dispose();atlas.layer.mesh.material.dispose();atlas.renderer.dispose();t.dispose();
});
test('连续剖面在宽视野离轴处保持光量，地平线只裁掉下半片光源',()=>{
    const o=sourceFixture('gaussian',.002),t=sourceTexture([o]);painter.setDiffuse(t);const ratios=[];
    const right=new THREE.Vector3().setFromMatrixColumn(camera.matrixWorld,0);
    for(const mode of ['perspective','stereographic'])for(const angle of [30,55])for(const dpr of [1,2]){
        camera.lookAt(new THREE.Vector3(...direction).multiplyScalar(Math.cos(angle*DEG)).addScaledVector(right,Math.sin(angle*DEG)));
        const im=render(70,{mode,dpr}),ratio=integratedFlux(im,70*dpr,mode)/(o.v_flux*2.54e-6);ratios.push({mode,angle,dpr,ratio});
    }
    t.dispose();assert(ratios.every(v=>Math.abs(v.ratio-1)<.025),JSON.stringify(ratios));metrics.continuousOffAxis=ratios;
    const axes=[[0,0],[90,0],[0,90]].map(([l,b])=>galacticToEquatorial(l,b)),g=axes.map(a=>a[0]);
    const horizon=sourceFixture('gaussian',.006,{gal_lon:(Math.atan2(g[1],g[0])/DEG+360)%360,gal_lat:Math.asin(g[2])/DEG}),ht=sourceTexture([horizon]);
    painter.setDiffuse(ht);camera.lookAt(1,0,0);
    const im=render(500,{}, {...sky,frame:{...sky.frame,surface:true}}),ratio=integratedFlux(im,500,'perspective')/(horizon.v_flux*2.54e-6);
    metrics.continuousHorizonFraction=ratio;assert(Math.abs(ratio-.5)<.01,`地平线截断不对称 ${ratio}`);ht.dispose();camera.lookAt(new THREE.Vector3(...direction));
});
painter.dispose();document.querySelector('#status').textContent=`${total-failures}/${total} 通过；${failures} 失败`;window.renderCheckReport={total,failures,metrics};
