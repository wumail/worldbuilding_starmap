import * as THREE from 'three';
import {renderFixture,pixelMetrics} from './sky_render_fixture.mjs';
import {skyState,equatorialDirection,galacticToEquatorial,projectHemisphere} from '../web/shared/solar_system.mjs';
import {sanitizeState} from '../web/v1/sky_state.mjs';
import {DiffuseAtlasPainter,attachDiffuseSources} from '../web/shared/deep_sky_view.js';
import {EyeSkyRenderer} from '../web/v2/eye_renderer.mjs';
import {EYE_DEFAULTS,observerSky,fovForFocal} from '../web/v2/eye_model.mjs';
import {sanitizeState as eyeState} from '../web/v2/sky_state.mjs';
let total=0,failures=0;const metrics={};
const assert=(ok,message)=>{if(!ok)throw Error(message);};
function test(name,fn){total++;const li=document.createElement('li');try{fn();li.className='pass';li.textContent='通过：'+name;}catch(e){failures++;li.className='fail';li.textContent='失败：'+name+' — '+e.message;}document.querySelector('#results').append(li);}
function sample(p,label){const item=document.createElement('div'),text=document.createElement('p'),c=document.createElement('canvas');text.textContent=label;c.width=p.width;c.height=p.height;const ctx=c.getContext('2d');ctx.putImageData(new ImageData(new Uint8ClampedArray(p.data),p.width,p.height),0,0);item.append(text,c);document.querySelector('#samples').append(item);}
const renderer=new THREE.WebGLRenderer({antialias:false,preserveDrawingBuffer:true}),bodies=skyState(0,sanitizeState()).bodies;
renderer.debug.onShaderError=(gl,program,vs,fs)=>{throw Error(gl.getShaderInfoLog(vs)+' / '+gl.getShaderInfoLog(fs));};
test('用户截图视向 RA 133.9° / Dec 0.6° 不被反方向 Luna 铺满',()=>{
    const fixture=renderFixture('stereographic',renderer,384),luna=bodies.find(b=>b.id==='Luna'),options={focus:equatorialDirection(133.9,.6),focal:384/(4*Math.tan(60*Math.PI/720)),gain:6,enhanced:true};
    try{const blank=fixture.show([],options),actual=fixture.show([luna],options);let changed=0;
        for(let i=0;i<blank.data.length;i+=4)if(Math.abs(blank.data[i]-actual.data[i])>1)changed++;
        metrics.userLunaAntipode={changedPixels:changed,pixels:actual.width*actual.height,center:[...actual.data.slice(4*(192*384+192),4*(192*384+192)+4)]};
        sample(actual,'用户视向：背景应完整保留');assert(changed===0,JSON.stringify(metrics.userLunaAntipode));
    }finally{fixture.close();}
});
test('太阳、双月和行星的反方向不产生盘面或光晕，保留前景星点',()=>{
    const cases=[];
    for(const mode of ['perspective','stereographic']){
        const fixture=renderFixture(mode,renderer,192);
        try{for(const id of ['Sol','Luna','Echo','Venus-Sol'])for(const gain of [1,6,8])for(const dpr of [1,2]){
            const body=bodies.find(b=>b.id===id),focus=body.view.map(v=>-v),options={focus,focal:mode==='perspective'?60:95,gain,dpr,enhanced:true,solarGlow:true,planetGlow:true,stars:[{view:focus,app_mag:-2,color_hex:'#ffffff'}]};
            const blank=fixture.show([],options),actual=fixture.show([body],options);let changed=0;
            for(let i=0;i<blank.data.length;i++)if(blank.data[i]!==actual.data[i])changed++;
            cases.push({id,mode,gain,dpr,changed});
        }}finally{fixture.close();}
    }
    metrics.bodyAntipodes=cases;assert(cases.every(c=>c.changed===0),JSON.stringify(cases.filter(c=>c.changed)));
});
renderer.dispose();
const catalog=await (await fetch('../output/output_20260915_galactic_01/sky_view_20260915_galactic_01.json')).json();
const objects=catalog.deep_sky.objects;
const sourceTexture=o=>{const t=new THREE.DataTexture(new Float32Array(4),2,2,THREE.RedFormat,THREE.FloatType);t.needsUpdate=true;return attachDiffuseSources(t,[o]);};
test('完整星表和恒星系在用户视向下保留星空，不出现整屏覆盖',()=>{
    const r=new THREE.WebGLRenderer({antialias:false,preserveDrawingBuffer:true}),fixture=renderFixture('stereographic',r,512);
    try{
        const stars=catalog.stars.map(s=>({...s,view:galacticToEquatorial(s.gal_lon,s.gal_lat)}));
        const im=fixture.show(bodies,{focus:equatorialDirection(133.9,.6),focal:512/(4*Math.tan(60*Math.PI/720)),gain:6,enhanced:true,solarGlow:true,planetGlow:true,stars});
        const p=pixelMetrics(im);metrics.catalogueAntipode={catalogue:stars.length,brightPixels:p.count,fraction:p.count/(512*512)};
        sample(im,'完整星表与恒星系：用户截图视向');assert(p.count>100 && p.count<512*512*.2,JSON.stringify(metrics.catalogueAntipode));
    }finally{fixture.close();r.dispose();}
});
test('真实发射云和反射云的显示对比更柔和，星团像素保持一致',()=>{
    const c=document.createElement('canvas');c.width=c.height=256;const ctx=c.getContext('2d'),painter=new DiffuseAtlasPainter(ctx),sky=skyState(0,{mode:'center'}),cases=[];
    const ids=['H-D-OC-7-000786','H-D-OC-6.5-001558','OC-7.5-001432',objects.find(o=>o.kind==='reflection_nebula'&&o.app_mag<=6.5).id];
    try{for(const id of ids){
        const o=objects.find(o=>o.id===id),texture=sourceTexture(o),direction=galacticToEquatorial(o.gal_lon,o.gal_lat),layout={radius:200,north:[64,128],south:[520,128]},zoom=128;
        const p=projectHemisphere(direction,200,layout,13.564125),pan=[(128-p.x)*zoom,(128-p.y)*zoom];
        const draw=soften=>{ctx.setTransform(1,0,0,1,0,0);ctx.fillStyle='#050911';ctx.fillRect(0,0,256,256);painter.layer.uniforms.softenGas.value=soften;painter.draw(sky,texture,layout,13.564125,{width:256,height:256,dpr:1,zoom,pan,enabled:true});return ctx.getImageData(0,0,256,256).data;};
        const before=draw(false),after=draw(true);let peakBefore=0,peakAfter=0,difference=0,gradientBefore=0,gradientAfter=0;
        for(let y=0;y<255;y++)for(let x=0;x<255;x++){const i=4*(y*256+x);peakBefore=Math.max(peakBefore,before[i]-5);peakAfter=Math.max(peakAfter,after[i]-5);difference=Math.max(difference,Math.abs(before[i]-after[i]));for(const offset of [4,1024]){gradientBefore=Math.max(gradientBefore,Math.abs(before[i]-before[i+offset]));gradientAfter=Math.max(gradientAfter,Math.abs(after[i]-after[i+offset]));}}
        const result={id,kind:o.kind,peakBefore,peakAfter,gradientBefore,gradientAfter,difference};cases.push(result);texture.dispose();
        if(o.kind==='open_cluster')assert(difference===0,JSON.stringify(result));
        else assert(peakBefore>0 && peakAfter<.75*peakBefore && gradientAfter<=gradientBefore,JSON.stringify(result));
    }}finally{painter.layer.mesh.geometry.dispose();painter.layer.mesh.material.dispose();painter.renderer.dispose();}
    metrics.gasDisplay=cases;
});
test('气体云柔化不改变 V2 物理光量缓冲',()=>{
    const painter=new EyeSkyRenderer(document.createElement('canvas')),camera=new THREE.PerspectiveCamera(),o=objects.find(o=>o.id==='H-D-OC-7-000786'),texture=sourceTexture(o);
    const sky=observerSky(0,eyeState({mode:'center'}));sky.bodies=[];sky.environment={atmosphere:false,extinction:0,nightL:0,dayL:0,moons:[],adaptation:.005};sky.eye={...EYE_DEFAULTS,opticalBlur:false};
    painter.resize(256,256,1);painter.setDiffuse(texture);camera.up.set(0,0,1);camera.lookAt(...galacticToEquatorial(o.gal_lon,o.gal_lat));camera.fov=fovForFocal(256,20000,'perspective');camera.updateProjectionMatrix();
    try{const render=soften=>{painter.u.softenGas.value=soften;painter.render(camera,sky,[],{focal:20000,displayScale:1,deepSky:true});return painter.readLinear().data;};
        const before=render(false),after=render(true);let difference=0,total=0;for(let i=0;i<before.length;i+=4){difference=Math.max(difference,Math.abs(before[i]-after[i]));total+=before[i];}
        metrics.gasPhysical={maxDifference:difference,positiveRadianceSum:total};assert(difference===0 && total>0,JSON.stringify(metrics.gasPhysical));
    }finally{texture.dispose();painter.dispose();}
});
document.querySelector('#status').textContent=`${total-failures}/${total} 通过；${failures} 失败`;window.renderCheckReport={total,failures,metrics};
