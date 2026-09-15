import * as THREE from 'three';
import {EyeSkyRenderer} from '../web/v2/eye_renderer.mjs';
import {EYE_DEFAULTS,fovForFocal,observerSky} from '../web/v2/eye_model.mjs';
import {sanitizeState} from '../web/v2/sky_state.mjs';
import {galacticToEquatorial,skyState,projectHemisphere} from '../web/shared/solar_system.mjs';
import {attachDiffuseSources,DiffuseAtlasPainter,diffuseGLSL,diffuseUniforms,setDiffuseUniforms} from '../web/shared/deep_sky_view.js';
import {MORPHOLOGY_MODEL,MORPHOLOGY_FAMILIES,morphologyGLSL} from '../web/shared/nebula_morphology.mjs';
let total=0,failures=0;const metrics={};
const assert=(v,m)=>{if(!v)throw Error(m);};
async function test(name,fn){
    total++;const li=document.createElement('li');
    try{await fn();li.className='pass';li.textContent='通过：'+name;}catch(e){failures++;li.className='fail';li.textContent='失败：'+name+' — '+e.message;}
    document.querySelector('#results').append(li);await new Promise(r=>setTimeout(r,0));
}
const morphology=family=>({model:MORPHOLOGY_MODEL,family,position_angle_rad:.8,axis_ratio:.64,phase_rad:2.1,noise_seed:143,turbulence:.85,shell_thickness:.27,filament_width:.14});
const object=(family,a=.01,extra={})=>({id:family,kind:'emission_nebula',profile:'ionized_gaussian',profile_truncation:.8,
    angular_scale_rad:a,gal_lon:0,gal_lat:0,app_mag:4,v_flux:10**(-1.6),morphology:morphology(family),...extra});
const tex=objects=>{const t=new THREE.DataTexture(new Float32Array(4),2,2,THREE.RedFormat,THREE.FloatType);t.needsUpdate=true;return attachDiffuseSources(t,objects);};
const canvas=document.createElement('canvas'),painter=new EyeSkyRenderer(canvas),camera=new THREE.PerspectiveCamera(60,1,.01,10),direction=galacticToEquatorial(0,0);
camera.up.set(0,0,1);camera.lookAt(new THREE.Vector3(...direction));
const sky=observerSky(0,sanitizeState({mode:'center'}));sky.bodies=[];sky.eye={...EYE_DEFAULTS,opticalBlur:false};
sky.environment={atmosphere:false,extinction:0,nightL:0,dayL:0,moons:[],adaptation:.005};
function render(f=500,options={},s=sky){
    painter.resize(256,256,options.dpr||1);camera.userData.skyProjection=options.mode||'perspective';
    camera.fov=fovForFocal(256,f,camera.userData.skyProjection);camera.updateProjectionMatrix();
    painter.render(camera,s,[],{focal:f,displayScale:1,deepSky:true,...options});return painter.readLinear();
}
function flux(im,f,mode){
    let sum=0;const corner=(x,y)=>Math.atan2(x*y,Math.sqrt(1+x*x+y*y));
    for(let y=0;y<im.height;y++)for(let x=0;x<im.width;x++){
        const px=(x+.5-im.width/2)/f,py=(y+.5-im.height/2)/f;
        const w=mode==='stereographic'?1/(f*f*(1+(px*px+py*py)/4)**2):corner(px+.5/f,py+.5/f)-corner(px-.5/f,py+.5/f)-corner(px+.5/f,py-.5/f)+corner(px-.5/f,py-.5/f);
        sum+=im.data[4*(y*im.width+x)]*w;
    }return sum;
}
await test('五种形态在两种投影、缩放及 DPR 下保持原始 HDR 总光量',()=>{
    let worst=0,witness;const cases=[];
    for(const family of MORPHOLOGY_FAMILIES)for(const a of [.0001,.001,.005,.03]){
        const o=object(family,a),t=tex([o]);painter.setDiffuse(t);
        for(const focal of [200,500])for(const dpr of [1,2])for(const mode of ['perspective','stereographic']){
            const im=render(focal,{dpr,mode}),ratio=flux(im,focal*dpr,mode)/(o.v_flux*2.54e-6),error=Math.abs(ratio-1),entry={family,a,focal,dpr,mode,ratio};
            cases.push(entry);if(error>worst){worst=error;witness=entry;}
        }t.dispose();
    }
    metrics.flux={worst,witness,cases};assert(worst<.035,JSON.stringify(witness));
});
await test('转动相机时云丝固定在天球，旋转后的图像与原方向一致',()=>{
    const t=tex([object('filament',.045)]);painter.setDiffuse(t);camera.lookAt(new THREE.Vector3(...direction));
    const before=render(1200),q=camera.quaternion.clone();camera.rotateZ(Math.PI/2);const after=render(1200);camera.quaternion.copy(q);
    let difference=0,totalLight=0;
    for(let y=0;y<256;y++)for(let x=0;x<256;x++){
        const a=after.data[4*(y*256+x)],b=before.data[4*(x*256+255-y)];difference+=Math.abs(a-b);totalLight+=b;
    }
    metrics.cameraRollRelativeDifference=difference/totalLight;assert(difference/totalLight<.002,String(difference/totalLight));t.dispose();
});
await test('结构化星云在接缝、两极及离轴视角保持光量',()=>{
    const cases=[];
    for(const [lon,lat] of [[359.999,0],[120,89.999],[320,-89.999]]){
        const o=object('blister',.006,{gal_lon:lon,gal_lat:lat}),t=tex([o]);painter.setDiffuse(t);camera.up.set(0,0,1);camera.lookAt(new THREE.Vector3(...galacticToEquatorial(lon,lat)));
        const ratio=flux(render(600),600,'perspective')/(o.v_flux*2.54e-6);cases.push({lon,lat,ratio});t.dispose();
    }
    const o=object('filament',.006),t=tex([o]);painter.setDiffuse(t);camera.lookAt(new THREE.Vector3(...direction));camera.updateMatrixWorld();const right=new THREE.Vector3().setFromMatrixColumn(camera.matrixWorld,0);
    for(const mode of ['perspective','stereographic'])for(const a of [30,55]){
        const angle=a*Math.PI/180;camera.lookAt(new THREE.Vector3(...direction).multiplyScalar(Math.cos(angle)).addScaledVector(right,Math.sin(angle)));
        const ratio=flux(render(70,{mode}),70,mode)/(o.v_flux*2.54e-6);cases.push({mode,angle:a,ratio});
    }
    metrics.directions=cases;assert(cases.every(c=>Math.abs(c.ratio-1)<.035),JSON.stringify(cases));t.dispose();camera.lookAt(new THREE.Vector3(...direction));
});
await test('完整月面、地平线和图层开关仍能正确遮挡结构化星云',()=>{
    const t=tex([object('shell',.01)]);painter.setDiffuse(t);
    const moon={id:'Luna',kind:'moon',color:'#ffffff',view:direction,angularDiameter:8,distance:.01,altitude:30,brightEnough:false,visible:false,lightDirection:direction,observedMagnitude:Infinity};
    for(const [s,options] of [[{...sky,bodies:[moon]},{}],[{...sky,frame:{...sky.frame,surface:true}},{}],[sky,{deepSky:false}]]){
        const im=render(500,options,s);assert(im.data.every((v,i)=>i%4===3||v===0),'弥散光穿透遮挡或图层开关');
    }
    painter.u.softenGas.value=false;const a=render(500);painter.u.softenGas.value=true;const b=render(500);
    assert(a.data.every((v,i)=>v===b.data[i]),'显示设置改动了物理 HDR');t.dispose();
});
await test('五种形态对照及 V1 双半球缩放的方向一致',()=>{
    const names=['不规则云气','弯曲云丝','投影壳层','开口弧壳','反射云扇面'];
    for(const [i,family] of MORPHOLOGY_FAMILIES.entries()){
        const o=object(family,.006,{app_mag:2,v_flux:10**(-.8)}),t=tex([o]);painter.setDiffuse(t);render(20000);
        const pixels=painter.readPixels(),c=document.createElement('canvas');c.width=c.height=256;const ctx=c.getContext('2d'),im=ctx.createImageData(256,256);
        for(let y=0;y<256;y++)im.data.set(pixels.data.subarray(4*(255-y)*256,4*(256-y)*256),4*y*256);ctx.putImageData(im,0,0);
        const figure=document.createElement('figure'),caption=document.createElement('figcaption');caption.textContent=names[i];figure.append(c,caption);document.querySelector('#samples').append(figure);t.dispose();
    }
    const o=object('filament',.008),t=tex([o]),c=document.createElement('canvas');c.width=800;c.height=400;const ctx=c.getContext('2d'),atlas=new DiffuseAtlasPainter(ctx),layout={radius:160,north:[190,200],south:[610,200]};
    const p=projectHemisphere(direction,160,layout,13.564125);
    for(const zoom of [8,24]){
        ctx.clearRect(0,0,800,400);atlas.draw(skyState(0,{mode:'center'}),t,layout,13.564125,{width:800,height:400,dpr:1,zoom,pan:[-(p.x-400)*zoom,-(p.y-200)*zoom],enabled:true});
        const im=ctx.getImageData(0,0,800,400);let weight=0,xsum=0,ysum=0;
        for(let y=0;y<400;y++)for(let x=0;x<800;x++){const v=im.data[4*(y*800+x)];weight+=v;xsum+=v*x;ysum+=v*y;}
        assert(weight>0 && Math.hypot(xsum/weight-400,ysum/weight-200)<zoom*.8,'V1 结构剖面偏离预期方向');
    }
    atlas.layer.mesh.geometry.dispose();atlas.layer.mesh.material.dispose();atlas.renderer.dispose();t.dispose();
});
const data=await fetch('../output/output_20260915_nebula_03/sky_view_20260915_nebula_03.json').then(r=>r.json());
await test('逐一核验正式目录中 18 个云气候选的实际角尺度、截断与 HDR 光量',()=>{
    const cases=[];
    for(const o of data.deep_sky.objects.filter(o=>o.morphology&&o.app_mag<=6.5)){
        const t=tex([o]);painter.setDiffuse(t);camera.up.set(0,0,1);camera.lookAt(new THREE.Vector3(...galacticToEquatorial(o.gal_lon,o.gal_lat)));
        const cut=o.profile==='gaussian'?4:o.profile_truncation,f=45/(Math.tan(o.angular_scale_rad)*cut);
        for(const mode of ['perspective','stereographic']){
            const ratio=flux(render(f,{mode}),f,mode)/(o.v_flux*2.54e-6);cases.push({id:o.id,family:o.morphology.family,mode,ratio});
        }t.dispose();
    }
    metrics.actualFlux=cases;assert(cases.every(c=>Math.abs(c.ratio-1)<.025),JSON.stringify(cases));
});
await test('正式新目录保留 9356 颗恒星，18 个明亮云气候选包含全部五类形态',()=>{
    const clouds=data.deep_sky.objects.filter(o=>o.morphology),visible=clouds.filter(o=>o.app_mag<=6.5);
    metrics.catalog={stars:data.stars.length,clouds:clouds.length,candidates:visible.length,families:Object.fromEntries(MORPHOLOGY_FAMILIES.map(f=>[f,visible.filter(o=>o.morphology.family===f).length]))};
    assert(data.stars.length===9356&&clouds.length===160&&visible.length===18,'目录发生意外变化');
    assert(Object.values(metrics.catalog.families).every(n=>n>0),'可见候选缺少形态类别');
    assert(clouds.every(o=>o.morphology.model===MORPHOLOGY_MODEL),'新目录仍包含旧硬壳模型');
});
await test('GPU 噪声与 Python 导出器一致，无整数散列或浮点纹理伪影',async()=>{
    const refs=await fetch('./fixtures/nebula_noise_reference.json').then(r=>r.json()),values=new Float32Array(refs.length*4);
    refs.forEach((r,i)=>values.set([r.x,r.y,r.seed,0],i*4));
    const samples=new THREE.DataTexture(values,refs.length,1,THREE.RGBAFormat,THREE.FloatType);samples.needsUpdate=true;samples.minFilter=samples.magFilter=THREE.NearestFilter;
    const target=new THREE.WebGLRenderTarget(refs.length,1,{type:THREE.FloatType,format:THREE.RGBAFormat,depthBuffer:false}),scene=new THREE.Scene(),geometry=new THREE.PlaneGeometry(2,2);
    const material=new THREE.ShaderMaterial({uniforms:{samples:{value:samples}},vertexShader:'void main(){gl_Position=vec4(position.xy,0.,1.);}',fragmentShader:morphologyGLSL+`uniform sampler2D samples;void main(){vec3 p=texture2D(samples,vec2(gl_FragCoord.x/${refs.length}.,.5)).xyz;gl_FragColor=vec4(cloudNoise(p.xy,p.z),cloudStructure(p.xy,p.z),0.,1.);}`});
    scene.add(new THREE.Mesh(geometry,material));const renderer=painter.renderer;renderer.setRenderTarget(target);renderer.render(scene,new THREE.Camera());
    const out=new Float32Array(refs.length*4);renderer.readRenderTargetPixels(target,0,0,refs.length,1,out);renderer.setRenderTarget(null);
    const worst=Math.max(...refs.flatMap((r,i)=>[Math.abs(out[i*4]-r.noise),Math.abs(out[i*4+1]-r.structure)]));metrics.gpuNoiseError=worst;
    samples.dispose();geometry.dispose();material.dispose();target.dispose();assert(worst<.001,String(worst));
});
await test('GPU 柔和壳层与独立视线积分一致，包含中心、亮缘和渐隐外缘',async()=>{
    const refs=await fetch('./fixtures/nebula_soft_shell_reference.json').then(r=>r.json()),values=new Float32Array(refs.samples.length*4);
    refs.samples.forEach((r,i)=>values.set([r.impact,r.thickness,0,0],i*4));
    const samples=new THREE.DataTexture(values,refs.samples.length,1,THREE.RGBAFormat,THREE.FloatType);samples.needsUpdate=true;samples.minFilter=samples.magFilter=THREE.NearestFilter;
    const target=new THREE.WebGLRenderTarget(refs.samples.length,1,{type:THREE.FloatType,format:THREE.RGBAFormat,depthBuffer:false}),scene=new THREE.Scene(),geometry=new THREE.PlaneGeometry(2,2);
    const material=new THREE.ShaderMaterial({uniforms:{samples:{value:samples}},vertexShader:'void main(){gl_Position=vec4(position.xy,0.,1.);}',fragmentShader:morphologyGLSL+`uniform sampler2D samples;void main(){vec2 p=texture2D(samples,vec2(gl_FragCoord.x/${refs.samples.length}.,.5)).xy;gl_FragColor=vec4(softShellColumn(p.x,p.y),0.,0.,1.);}`});
    scene.add(new THREE.Mesh(geometry,material));const renderer=painter.renderer;renderer.setRenderTarget(target);renderer.render(scene,new THREE.Camera());
    const out=new Float32Array(refs.samples.length*4);renderer.readRenderTargetPixels(target,0,0,refs.samples.length,1,out);renderer.setRenderTarget(null);
    const worst=Math.max(...refs.samples.map((r,i)=>Math.abs(out[i*4]-r.column)));metrics.gpuSoftShellError=worst;
    samples.dispose();geometry.dispose();material.dispose();target.dispose();assert(worst<4e-6,String(worst));
});
await test('滤波后的 GPU 云气场与 Python 导出逐点一致，含多图块、外缘和不同轴比',async()=>{
    const refs=await fetch('./fixtures/nebula_filtered_reference.json').then(r=>r.json()),objects=[],tiles=new Map(),samples=new Float32Array(refs.samples.length*4);
    refs.samples.forEach((r,i)=>{
        const key=r.morphology.family+':'+r.profile+':'+r.cut;
        if(!tiles.has(key)){tiles.set(key,objects.length);objects.push(object(r.morphology.family,.01,{id:key,profile:r.profile,profile_truncation:r.cut,morphology:r.morphology}));}
        samples.set([r.x,r.y,r.cut,tiles.get(key)],i*4);
    });
    const clouds=tex(objects),input=new THREE.DataTexture(samples,refs.samples.length,1,THREE.RGBAFormat,THREE.FloatType);input.needsUpdate=true;input.minFilter=input.magFilter=THREE.NearestFilter;
    const uniforms={...diffuseUniforms(),samples:{value:input}};setDiffuseUniforms(uniforms,clouds);
    const target=new THREE.WebGLRenderTarget(refs.samples.length,1,{type:THREE.FloatType,format:THREE.RGBAFormat,depthBuffer:false}),scene=new THREE.Scene(),geometry=new THREE.PlaneGeometry(2,2);
    const material=new THREE.ShaderMaterial({uniforms,vertexShader:'void main(){gl_Position=vec4(position.xy,0.,1.);}',fragmentShader:diffuseGLSL+`uniform sampler2D samples;void main(){vec4 s=texture2D(samples,vec2(gl_FragCoord.x/${refs.samples.length}.,.5));gl_FragColor=vec4(filteredCloud(s.xy,s.z,s.w),0.,0.,1.);}`});
    scene.add(new THREE.Mesh(geometry,material));const renderer=painter.renderer;renderer.setRenderTarget(target);renderer.render(scene,new THREE.Camera());
    const out=new Float32Array(refs.samples.length*4);renderer.readRenderTargetPixels(target,0,0,refs.samples.length,1,out);renderer.setRenderTarget(null);
    metrics.gpuFilteredError=Math.max(...refs.samples.map((r,i)=>Math.abs(out[4*i]-r.value)));
    assert(metrics.gpuFilteredError<2e-6,String(metrics.gpuFilteredError));
    input.dispose();clouds.dispose();geometry.dispose();material.dispose();target.dispose();
});
painter.dispose();document.querySelector('#status').textContent=`${total-failures}/${total} 通过；${failures} 失败`;window.renderCheckReport={total,failures,metrics};
