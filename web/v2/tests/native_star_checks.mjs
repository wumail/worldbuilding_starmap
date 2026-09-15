import * as THREE from 'three';
import {AtlasStarPainter} from '../../v1/sky_atlas_stars.mjs';
import {SYMBOL_REFERENCE,pointRadius} from '../../shared/sky_render.mjs';
import {EyeSkyRenderer} from '../eye_renderer.mjs';
import {EYE_DEFAULTS,fovForFocal,magnitudeToLux,observerSky} from '../eye_model.mjs';
import {sanitizeState} from '../sky_state.mjs';

export async function runNativeStarChecks({test,assert,metrics}){
    const r=new EyeSkyRenderer(document.createElement('canvas')),camera=new THREE.PerspectiveCamera();
    camera.up.set(0,0,1);camera.lookAt(1,0,0);
    const base=observerSky(0,sanitizeState({mode:'center'}));base.frame.position=[0,0,0];
    base.bodies=[];base.environment={atmosphere:false,extinction:0,nightL:0,dayL:0,moons:[],adaptation:.005,limit:6.5};
    base.eye={...EYE_DEFAULTS,opticalBlur:false};
    const c=document.createElement('canvas'),ctx=c.getContext('2d'),reference=new AtlasStarPainter(ctx);
    const star=(m,d=[1,0,0])=>({id:'native-fixture',app_mag:m,color_hex:'#ffffff',baseDirection:d,baseDistanceAU:1e10});
    const render=(renderer,sources,{width=256,height=256,dpr=1,focal=338,gain=8,projection='perspective',blur=false,overview=false,bodies=[],atlasTransform}={})=>{
        renderer.resize(width,height,dpr);camera.aspect=width/height;camera.userData.skyProjection=projection;
        camera.fov=fovForFocal(height,focal,projection);camera.updateProjectionMatrix();
        renderer.render(camera,{...base,bodies,eye:{...base.eye,opticalBlur:blur}},sources,{focal,displayScale:gain,overview,atlasTransform});metrics.samples++;
        return renderer.readPixels();
    };
    const compare=(sources,im,{width=256,height=256,focal=338,gain=8}={})=>{
        c.width=im.width;c.height=im.height;ctx.setTransform(im.width/width,0,0,im.height/height,0,0);
        ctx.fillStyle='#050911';ctx.fillRect(0,0,width,height);
        const points=sources.map(s=>r.project(s.baseDirection)),scale=focal/SYMBOL_REFERENCE.perspectiveFocal*gain;
        sources.forEach((s,i)=>reference.draw(s,points[i],scale));
        const expected=ctx.getImageData(0,0,c.width,c.height).data;let worst=0;
        for(let i=0;i<sources.length;i++){
            const p=points[i],dpr=im.width/width,span=pointRadius(sources[i].app_mag)*scale*dpr+4,cx=p.x*dpr,cy=p.y*dpr;
            for(let y=Math.max(0,Math.floor(cy-span));y<Math.min(im.height,Math.ceil(cy+span));y++)for(let x=Math.max(0,Math.floor(cx-span));x<Math.min(im.width,Math.ceil(cx+span));x++)for(let channel=0;channel<3;channel++)
            {
                const actual=im.data[4*((im.height-1-y)*im.width+x)+channel],wanted=expected[4*(y*im.width+x)+channel],difference=Math.abs(actual-wanted);
                if(difference>worst){worst=difference;compare.witness={magnitude:sources[i].app_mag,x,y,channel,actual,wanted,width,height,focal,gain,dpr,point:p};}
            }
        }
        return worst;
    };
    try{
        await test('高倍双半球图按相同角尺度与显示倍率对齐，有色星点与 V1 一致',()=>{
            let worst=0,cases=0,witness;
            const width=760,height=600;
            for(const zoom of [8,33.84,64])for(const dpr of [1,2])for(const gain of [6,8]){
                const atlasTransform={zoom,panX:0,panY:0},options={width,height,dpr,gain,overview:true,atlasTransform};
                render(r,[],options);
                const anchor=r.project([.6,.3,Math.sqrt(.55)]);
                atlasTransform.panX=width/2-anchor.x;atlasTransform.panY=height/2-anchor.y;
                render(r,[],options);
                const c0=r.layout.north,rad=r.layout.radius;
                const sources=[-2,0,3,6.5].map((m,i)=>{
                    const x=width/2+(i-1.5)*170+.21,y=height/2+.49;
                    const dx=x-c0[0],dy=y-c0[1],theta=Math.hypot(dx,dy)/rad*Math.PI/2,phi=Math.atan2(-dx,dy)+r.orientation*Math.PI/180;
                    return {...star(m,[Math.sin(theta)*Math.cos(phi),Math.sin(theta)*Math.sin(phi),Math.cos(theta)]),color_hex:['#b8cbff','#ffcc90','#dbe7ff','#f2daae'][i]};
                });
                const im=render(r,sources,options),difference=compare(sources,im,{...options,focal:r.focal});
                if(difference>worst){worst=difference;witness={...compare.witness,zoom};}cases++;
            }
            metrics.nativeHighZoomAtlas={maxChannelDifference:worst,cases,witness};assert(worst<=3,`高倍地图星点差 ${worst}: ${JSON.stringify(witness)}`);
        });
        await test('局部增强星点与 V1 清晰符号一致，覆盖宽视野、星等、缩放与亚像素位置',()=>{
            let worst=0,cases=0,witness;
            for(const projection of ['perspective','stereographic'])for(const dpr of [1,2])for(const focal of [338,1500])for(const gain of [3,8]){
                for(const offset of [.21,.49]){
                    const sources=[-2,0,3,6.5].map((m,i)=>star(m,new THREE.Vector3(1,((i-1.5)*45+offset)/focal,offset/focal).normalize().toArray()));
                    const options={projection,dpr,focal,gain},im=render(r,sources,options);
                    const difference=compare(sources,im,options);if(difference>worst){worst=difference;witness={...compare.witness,projection,offset};}cases++;
                }
            }
            metrics.nativeSymbolReference={maxChannelDifference:worst,cases,witness};assert(worst<=3,`局部星点与清晰 Canvas 参考差 ${worst}: ${JSON.stringify(witness)}`);
        });
        await test('0.22×、8×、88.3° 场景跨越物理分辨率上限后，最终星点仍以原生像素绘制',()=>{
            const width=1140,height=656,focal=height/(2*Math.tan(88.3*Math.PI/360)),sources=[-2,0,3,6.5].map((m,i)=>star(m,new THREE.Vector3(1,(i-1.5)*.27,.0007).normalize().toArray()));
            let worst=0,witness;const sizes=[];
            for(const dpr of [1.5,1.75,2]){
                const options={width,height,dpr,focal,gain:8,blur:true},im=render(r,sources,options);
                assert(Math.abs(camera.fov-88.3)<1e-8,'未复现用户宽视野');
                assert(im.width===Math.round(width*dpr) && im.height===Math.round(height*dpr),'星点画布仍被缩小后放大');
                assert(r.targets[0].width*r.targets[0].height<2202000 && r.targets.every(t=>t.texture.type===THREE.FloatType),'物理缓冲被扩大或降低精度');
                if(dpr>=1.75)assert(r.presentation?.width===im.width && r.presentation.texture.type===THREE.HalfFloatType,'未使用原生线性合成缓冲');
                const difference=compare(sources,im,options);if(difference>worst){worst=difference;witness=compare.witness;}sizes.push({dpr,output:[im.width,im.height],physical:[r.targets[0].width,r.targets[0].height]});
            }
            metrics.nativeWideField={maxChannelDifference:worst,fov:88.3,magnification:.22,displayScale:8,sizes,witness};assert(worst<=3,`超过分辨率上限后星点变模糊，参考差 ${worst}: ${JSON.stringify(witness)}`);
        });
        await test('清晰显示与光扩散开关均不改变真实点通量',()=>{
            const options={width:1140,height:656,dpr:2,focal:338,gain:1},source=star(0);
            render(r,[source],options);const raw=r.readLinear();
            render(r,[source],{...options,gain:8,blur:true});const enhanced=r.readLinear(),on=r.readPixels();
            render(r,[source],{...options,gain:8,blur:false});const off=r.readPixels();
            let physical=0,blurLeak=0,total=0;
            for(let i=0;i<raw.data.length;i+=4){physical=Math.max(physical,Math.abs(raw.data[i]-enhanced.data[i]));total+=raw.data[i];}
            for(let i=0;i<on.data.length;i+=4)blurLeak=Math.max(blurLeak,Math.abs(on.data[i]-off.data[i]));
            const expected=magnitudeToLux(0)*(338*r.dpr)**2;
            assert(physical===0 && Math.abs(total/expected-1)<.0001,'显示流程改变了物理点通量');
            assert(blurLeak===0,'光扩散仍把增强星点扩成光斑');metrics.nativePhysical={physicalDifference:physical,enhancedBlurDifference:blurLeak,fluxRatio:total/expected};
        });
        await test('高分屏原生星点仍被新月暗面及平面日月的圆形边缘正确遮挡',()=>{
            const shadow={id:'Luna',kind:'moon',color:'#ffffff',view:[1,0,0],distance:.01,angularDiameter:.53,altitude:0,aboveHorizon:true,brightEnough:false,visible:false,lightDirection:[1,0,0],phaseAngle:Math.PI,magnitude:20,observedMagnitude:20};
            const options={width:1140,height:656,dpr:2,focal:338,gain:8,bodies:[shadow]};
            const dark=render(r,[],options),withStar=render(r,[star(-6)],options);let moonLeak=0;
            for(let i=0;i<dark.data.length;i++)moonLeak=Math.max(moonLeak,Math.abs(dark.data[i]-withStar.data[i]));
            assert(moonLeak===0,'原生星点穿过新月暗面');
            const mapCases=[];
            for(const id of ['Sol','Luna','Echo']){
            const sun={...shadow,id,kind:id==='Sol'?'star':'moon',view:[.5,0,Math.sqrt(.75)],distance:id==='Sol'?1:.01,angularDiameter:.44084},mapOptions={...options,overview:true,bodies:[sun]};
            const blank=render(r,[],mapOptions),limb=r.bodyCircles(r.sky.bodies[0])[0],l=r.layout,c=l.north;
            const x=limb.x+limb.radius+.2,y=limb.y,theta=Math.hypot(x-c[0],y-c[1])/l.radius*Math.PI/2,phi=Math.atan2(-(x-c[0]),y-c[1])+r.orientation*Math.PI/180;
            const direction=[Math.sin(theta)*Math.cos(phi),Math.sin(theta)*Math.sin(phi),Math.cos(theta)];
            const exposed=render(r,[star(-6,direction)],mapOptions);let solarLeak=0,visible=0;
            for(let y=0;y<exposed.height;y++)for(let x=0;x<exposed.width;x++){
                const delta=Math.abs(exposed.data[4*(y*exposed.width+x)]-blank.data[4*(y*blank.width+x)]);
                if(Math.hypot(x+.5-limb.x*2,exposed.height-y-.5-limb.y*2)<limb.radius*2-1)solarLeak=Math.max(solarLeak,delta);
                visible+=delta;
            }
            assert(solarLeak===0 && visible>100,id+' 圆形原生像素遮挡错位');mapCases.push({id,leak:solarLeak,visibleOutside:visible});
            }metrics.nativeOccultation={moonLeak,mapCases};
        });
        await test('原生合成缓冲正确复用和释放，扩容后第 1024 颗之后的星点仍完整',()=>{
            const fresh=new EyeSkyRenderer(document.createElement('canvas'));
            try{
                const options={width:1140,height:656,dpr:2,focal:338,gain:8};
                render(r,[star(0)],options);const target=r.presentation;let released=0;target.addEventListener('dispose',()=>released++);
                render(r,[star(0)],options);assert(r.presentation===target,'相同尺寸每帧重建合成缓冲');
                render(r,[],{...options,dpr:1.5});assert(released===1 && !r.presentation,'回到原生低分屏未释放额外缓冲');
                const sources=Array.from({length:1300},()=>star(6,new THREE.Vector3(1,-.6,0).normalize().toArray()));sources.push(star(-2,new THREE.Vector3(1,.6,0).normalize().toArray()));
                const a=render(r,sources,options),b=render(fresh,sources,options);let difference=0;
                for(let i=0;i<a.data.length;i++)difference=Math.max(difference,Math.abs(a.data[i]-b.data[i]));
                assert(difference===0,'扩容/切换 DPR 后星点与新建渲染器不同');
                const finalTarget=r.presentation;let disposed=0;finalTarget.addEventListener('dispose',()=>disposed++);r.dispose();
                assert(disposed===1,'最终合成缓冲未释放');metrics.nativeResources={maxPixelDifference:difference,reused:true,resizeReleased:released,disposeReleased:disposed};
            }finally{fresh.dispose();}
        });
    }finally{reference.clear();if(!metrics.nativeResources)r.dispose();}
}
