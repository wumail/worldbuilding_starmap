import * as THREE from 'three';
import {skyState,DEG,equatorialDirection,projectHemisphere,unit,add,scale} from '../web/shared/solar_system.mjs';
import {diskBasis,diskLight,phaseImage,pointRadius,pointOpacity,skyBackground,srgbToLinear,solarGlareVisibility} from '../web/shared/sky_render.mjs';
import {AtlasBodyPainter} from '../web/v1/sky_atlas_bodies.mjs';
import {SkyMotion3D} from '../web/v1/sky_motion_3d.js';
import {sanitizeState} from '../web/v1/sky_state.mjs';

export function runBodyRenderAudit({test,assert,renderer}) {
    const snapshot=(bodies,extra={})=>({days:0,night:1,frame:{surface:false,matrix:[[1,0,0],[0,1,0],[0,0,1]]},bodies,...extra});
    const sample=skyState(3928.342761);
    const sol=sample.bodies.find(b=>b.id==='Sol');
    const base={id:'Venus-Sol',kind:'planet',color:'#ffffff',view:[1,0,0],distance:.5,angularDiameter:.5,altitude:45,brightEnough:true,visible:true,lightDirection:[-1,0,0],magnitude:-3};
    function fixture(fov=60,dpr=1) {
        renderer.setPixelRatio(dpr);renderer.setSize(256,256);
        const camera=new THREE.PerspectiveCamera(fov,1,.1,5000);camera.up.set(0,0,1);camera.lookAt(1,0,0);camera.updateMatrixWorld();
        const scene=new THREE.Scene(),controls={clock:{state:sanitizeState({displayScale:1,projection:'perspective',markers:false,solarGlow:false,lunarGlow:false,planetGlow:false})}};
        const motion=new SkyMotion3D(scene,camera,document.createElement('div'),renderer,controls);
        return {camera,controls,motion,show(sky) {
            motion.update(sky,pointRadius,pointOpacity);renderer.render(scene,camera);
            const gl=renderer.getContext(),width=gl.drawingBufferWidth,height=gl.drawingBufferHeight,data=new Uint8Array(width*height*4);
            gl.readPixels(0,0,width,height,gl.RGBA,gl.UNSIGNED_BYTE,data);return {data,width,height};
        }};
    }
    function canvasFixture() {
        const canvas=document.createElement('canvas');canvas.width=canvas.height=512;
        const ctx=canvas.getContext('2d'),painter=new AtlasBodyPainter(ctx);
        return {canvas,ctx,painter};
    }
    function bounds({data,width,height},cx=width/2,cy=height/2,span=Math.min(width,height)/2,threshold=40) {
        let minX=Infinity,maxX=-Infinity,minY=Infinity,maxY=-Infinity,count=0,xSum=0,ySum=0,sum=0;
        for(let y=Math.max(0,Math.floor(cy-span));y<Math.min(height,Math.ceil(cy+span));y++)for(let x=Math.max(0,Math.floor(cx-span));x<Math.min(width,Math.ceil(cx+span));x++) {
            const k=4*(y*width+x),v=Math.max(data[k],data[k+1],data[k+2]);if(v<=threshold)continue;
            minX=Math.min(minX,x);maxX=Math.max(maxX,x);minY=Math.min(minY,y);maxY=Math.max(maxY,y);count++;
            sum+=v;xSum+=(x+.5)*v;ySum+=(y+.5)*v;
        }
        return {count,width:maxX-minX+1,height:maxY-minY+1,x:xSum/sum,y:ySum/sum};
    }
    const canvasPixels=({canvas,ctx})=>({data:ctx.getImageData(0,0,canvas.width,canvas.height).data,width:canvas.width,height:canvas.height});
    const fill=(ctx,sky)=>{ctx.setTransform(1,0,0,1,0,0);ctx.fillStyle='rgb('+skyBackground(sky).join(',')+')';ctx.fillRect(0,0,512,512);};

    test('实际 Sol、Luna、Echo 盘面按角直径绘制，Luna 大于 Sol 大于 Echo',()=>{
        const f=fixture(2),bodies=['Sol','Luna','Echo'].map((id,i)=>{
            const b=sample.bodies.find(b=>b.id===id),view=equatorialDirection([.68,0,-.68][i],0);
            return {...b,view,lightDirection:scale(view,-1),brightEnough:true,visible:true};
        });
        const pixels=f.show(snapshot(bodies)),sizes={};
        for(const b of bodies) {
            const p=new THREE.Vector3(...b.view).project(f.camera),expected=256*Math.tan(b.angularDiameter*DEG/2)/Math.tan(DEG);
            const a=bounds(pixels,(p.x+1)*128,(p.y+1)*128,expected/2+2,25);
            assert(Math.abs(a.width-expected)<2,JSON.stringify({id:b.id,actual:a.width,expected}));sizes[b.id]=a.width;
        }
        assert(sizes.Luna>sizes.Sol && sizes.Sol>sizes.Echo,JSON.stringify(sizes));
    });
    test('关闭行星增强后六颗行星显示真实盘面，暗行星仍不伪造可见亮面',()=>{
        for(const b of sample.bodies.filter(b=>b.kind==='planet')) {
            const f=fixture(b.angularDiameter*8);f.camera.lookAt(...b.view);f.camera.updateMatrixWorld();
            const pixels=f.show(snapshot([b])),record=f.motion.disks.get(b.id);
            assert(!record.point.visible,b.id+' 的光点仍盖住盘面');
            const a=bounds(pixels,128,128,30,40);
            if(b.brightEnough)assert(a.count>0,b.id+' 的亮面丢失');else assert(a.count===0,b.id+' 低于阈值仍有亮面');
        }
        const c=canvasFixture(),b={...base,view:equatorialDirection(0,45)},l={radius:18000,north:[0,0],south:[40000,0]},sky=snapshot([b]);
        fill(c.ctx,sky);const p=projectHemisphere(b.view,l.radius,l,0);c.ctx.translate(256-p.x,256-p.y);
        c.painter.draw(b,l,0,sky,{symbolScale:90,planetGlow:false});
        const a=bounds(canvasPixels(c));assert(a.width<160 && a.height<160,'放大后仍绘制巨型行星光点');
        assert(c.painter.disks.get(b.id).width>64,'大盘面仍只使用 64 像素纹理');
    });
    test('两页最终相位亮面朝向太阳，覆盖南北半球和两个天极',()=>{
        for(const dec of [90,45,-45,-90]) {
            const view=equatorialDirection(30,dec),basis=diskBasis(view),light=unit(add(scale(basis.x,Math.sqrt(.75)),scale(basis.z,.5)));
            const b={...base,view,lightDirection:light},sky=snapshot([b]),target=unit(add(view,scale(light,.01)));
            const f=fixture(4);f.camera.lookAt(...view);f.camera.updateMatrixWorld();const gpu=bounds(f.show(sky));
            const q=new THREE.Vector3(...target).project(f.camera),gx=gpu.x-128,gy=gpu.y-128;
            assert((gx*q.x+gy*q.y)>Math.hypot(gx,gy)*Math.hypot(q.x,q.y)*.95,'三维亮面反向，赤纬 '+dec);
            const c=canvasFixture(),l={radius:6000,north:[0,0],south:[14000,0]},p=projectHemisphere(view,l.radius,l,0),t=projectHemisphere(target,l.radius,l,0,p.north);
            fill(c.ctx,sky);c.ctx.translate(256-p.x,256-p.y);c.painter.draw(b,l,0,sky,{planetGlow:false});
            const atlas=bounds(canvasPixels(c)),ax=atlas.x-256,ay=atlas.y-256,dx=t.x-p.x,dy=t.y-p.y;
            assert((ax*dx+ay*dy)>Math.hypot(ax,ay)*Math.hypot(dx,dy)*.95,'平面亮面反向，赤纬 '+dec);
        }
    });
    test('Lambert 盘面在线性亮度积分后的四分相与满相通量比为 1/π',()=>{
        const b={...base,kind:'moon'},full=phaseImage(b,[0,0,1],256),quarter=phaseImage(b,[1,0,0],256);
        let a=0,q=0;for(let i=0;i<full.data.length;i+=4){a+=srgbToLinear(full.data[i]/255);q+=srgbToLinear(quarter.data[i]/255);}
        assert(Math.abs(q/a-1/Math.PI)<.003,'通量比 '+q/a);
    });
    test('Canvas 相位与真实 GPU 使用一致的线性 Lambert 明暗和 sRGB 输出',()=>{
        const f=fixture(1),b={...base,color:'#a08060',lightDirection:[-.5,0,Math.sqrt(.75)]},sky=snapshot([b]);
        const gpu=f.show(sky),image=phaseImage(b,diskLight(b),256,{background:skyBackground(sky)}),k=4*(128*256+128);
        for(let channel=0;channel<3;channel++)assert(Math.abs(gpu.data[k+channel]-image.data[k+channel])<=3,JSON.stringify({gpu:[...gpu.data.slice(k,k+3)],canvas:[...image.data.slice(k,k+3)]}));
    });
    test('白昼中低于阈值的天体保留遮挡，但不会在天空上留下黑斑',()=>{
        const b={...base,brightEnough:false,visible:false,magnitude:7,view:equatorialDirection(0,45)};
        const sky=snapshot([b],{night:0,frame:{surface:true,matrix:[[1,0,0],[0,1,0],[0,0,1]]}}),background=skyBackground(sky);
        const f=fixture(2);f.camera.lookAt(...b.view);f.camera.updateMatrixWorld();const gpu=f.show(sky),k=4*(128*256+128);
        assert(background.every((v,i)=>Math.abs(gpu.data[k+i]-v)<=1),'GPU 留下黑斑');
        const c=canvasFixture(),l={radius:6000,north:[0,0],south:[14000,0]},p=projectHemisphere(b.view,l.radius,l,0);
        fill(c.ctx,sky);c.ctx.translate(256-p.x,256-p.y);c.painter.draw(b,l,0,sky,{planetGlow:false});
        const data=c.ctx.getImageData(256,256,1,1).data;assert(background.every((v,i)=>Math.abs(data[i]-v)<=1),'Canvas 留下黑斑');
    });
    test('部分日食中日面剩余光衰减，但太阳光晕不能穿透前景暗月面',()=>{
        const f=fixture(8);f.controls.clock.state.solarGlow=true;
        const b={...sol,view:[1,0,0]},moon={...base,id:'Luna',kind:'moon',view:equatorialDirection(.1,0),angularDiameter:.3,distance:.01,brightEnough:false,visible:false};
        const sky=snapshot([b,moon]),visibility=solarGlareVisibility(sky);assert(visibility>0 && visibility<1,'不是偏食样本');
        const pixels=f.show(sky),p=new THREE.Vector3(...moon.view).project(f.camera),k=4*(128*256+Math.floor((p.x+1)*128));
        assert(skyBackground(sky).every((v,i)=>Math.abs(pixels.data[k+i]-v)<=1),'太阳光晕穿透前景暗月面');
    });
    test('刚升起的行星识读光点按地平线逐像素裁剪',()=>{
        const f=fixture(60),b={...base,angularDiameter:.005,view:equatorialDirection(0,.1),altitude:.1};
        f.controls.clock.state.planetGlow=true;
        const pixels=f.show(snapshot([b],{frame:{surface:true,matrix:[[1,0,0],[0,1,0],[0,0,1]]}}));
        for(let y=121;y<127;y++)for(let x=121;x<135;x++)assert(Math.max(...pixels.data.slice(4*(y*256+x),4*(y*256+x)+3))<40,'行星光点漏到地平线下');
    });
    renderer.setPixelRatio(1);renderer.setSize(256,256);
}
