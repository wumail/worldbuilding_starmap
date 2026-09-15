import * as THREE from 'three';
import {createStarMaterial} from '../web/v1/sky_points.js';
import {AtlasStarPainter} from '../web/v1/sky_atlas_stars.mjs';
import {pointRadius} from '../web/shared/sky_render.mjs';

export function runStarRenderAudit({test,assert,renderer}) {
    function canvasFixture(size=512) {
        const canvas=document.createElement('canvas');canvas.width=canvas.height=size;
        const ctx=canvas.getContext('2d');return {canvas,ctx,painter:new AtlasStarPainter(ctx)};
    }
    function measure(data,width,cx,cy,span=40,threshold=0) {
        let count=0,sum=0,peak=0,minX=Infinity,maxX=-Infinity;
        for(let y=Math.max(0,Math.floor(cy-span));y<Math.min(data.length/4/width,Math.ceil(cy+span));y++)for(let x=Math.max(0,Math.floor(cx-span));x<Math.min(width,Math.ceil(cx+span));x++) {
            const k=4*(y*width+x),v=Math.max(data[k],data[k+1],data[k+2]);sum+=v;peak=Math.max(peak,v);
            if(v>threshold){count++;minX=Math.min(minX,x);maxX=Math.max(maxX,x);}
        }
        return {count,sum,peak,width:maxX-minX+1};
    }
    function canvasStar(f,mag,zoom=1,x=256,y=256) {
        f.ctx.setTransform(1,0,0,1,0,0);f.ctx.fillStyle='#000';f.ctx.fillRect(0,0,f.canvas.width,f.canvas.height);
        f.ctx.translate(x,y);f.ctx.scale(zoom,zoom);f.painter.draw({app_mag:mag,color_hex:'#ffffff'},{x:0,y:0});
        return f.ctx.getImageData(0,0,f.canvas.width,f.canvas.height).data;
    }
    function gpuStars(magnitudes,dpr=1,offset=0,scale=1) {
        renderer.setPixelRatio(dpr);renderer.setSize(512,256);renderer.setClearColor(0);
        const camera=new THREE.PerspectiveCamera(2*Math.atan(Math.tan(Math.PI/6)/scale)*180/Math.PI,2,.1,5000);camera.up.set(0,0,1);camera.lookAt(1,0,0);camera.updateMatrixWorld();
        const scene=new THREE.Scene(),geometry=new THREE.BufferGeometry(),positions=[],colors=[];
        const centers=magnitudes.map((mag,i)=>({x:40+i*80+offset,y:128+offset}));
        for(const p of centers) {
            positions.push(...new THREE.Vector3(p.x/256-1,p.y/128-1,.5).unproject(camera).normalize().multiplyScalar(1000).toArray());
            colors.push(1,1,1);
        }
        geometry.setAttribute('position',new THREE.Float32BufferAttribute(positions,3));
        geometry.setAttribute('color',new THREE.Float32BufferAttribute(colors,3));
        geometry.setAttribute('mag',new THREE.Float32BufferAttribute(magnitudes,1));
        const material=createStarMaterial({pixelRatio:dpr}),stars=new THREE.Points(geometry,material);stars.frustumCulled=false;scene.add(stars);
        renderer.render(scene,camera);const gl=renderer.getContext(),width=gl.drawingBufferWidth,data=new Uint8Array(width*gl.drawingBufferHeight*4);
        gl.readPixels(0,0,width,gl.drawingBufferHeight,gl.RGBA,gl.UNSIGNED_BYTE,data);
        const metrics=centers.map(p=>measure(data,width,p.x*dpr,p.y*dpr,16*dpr));geometry.dispose();material.dispose();return metrics;
    }
    test('平面恒星从 0 到 6.5 等逐级变小变暗，放大后的光点不再是实心圆片',()=>{
        const f=canvasFixture(),mags=[0,2,4,5,6,6.5];let previous=null;
        for(const mag of mags) {
            const data=canvasStar(f,mag,256),a=measure(data,512,256,256,80,16);
            assert(a.count>0,'星等 '+mag+' 消失');
            if(previous)assert(a.count<previous.count && a.sum<previous.sum,JSON.stringify({mag,a,previous}));previous=a;
            const radius=pointRadius(mag)*256;
            const at=fraction=>data[4*(256*512+Math.floor(256+radius*fraction))];
            assert(at(0)>at(.55) && at(.55)>at(.85),'星等 '+mag+' 没有亮核与渐隐边缘');
        }
    });
    test('平面恒星的柔光与位置同比缩放，改变颜色不影响大小和亮度曲线',()=>{
        const f=canvasFixture();
        for(const mag of [0,2,4,6.5]) {
            let full=null;
            for(const zoom of [512,256,128]) {
                const data=canvasStar(f,mag,zoom),a=measure(data,512,256,256,120,16);full??=a;
                assert(Math.abs(a.width-full.width*zoom/512)<=2,JSON.stringify({mag,zoom,a,full}));
                assert(Math.abs(a.sum/(full.sum*(zoom/512)**2)-1)<.12,JSON.stringify({mag,zoom,a,full}));
            }
        }
        const a=canvasStar(f,4,256);f.ctx.setTransform(1,0,0,1,0,0);f.ctx.fillStyle='#000';f.ctx.fillRect(0,0,512,512);
        f.ctx.translate(256,256);f.ctx.scale(256,256);f.painter.draw({app_mag:4,color_hex:'#ff0000'},{x:0,y:0});
        const b=f.ctx.getImageData(0,0,512,512).data;
        for(let k=0;k<a.length;k+=4)assert(a[k]===b[k] && b[k+1]===0 && b[k+2]===0,'颜色改变了光点轮廓');
    });
    test('实际 GPU 中不同星等的累计亮度严格排序，覆盖 DPR 1 和 2',()=>{
        for(const dpr of [1,2]) {
            const metrics=gpuStars([0,2,4,5,6,6.5],dpr);
            for(let i=0;i<metrics.length;i++) {
                assert(metrics[i].count>0,JSON.stringify({dpr,i,metrics}));
                if(i)assert(metrics[i].sum<metrics[i-1].sum,JSON.stringify({dpr,metrics}));
            }
        }
    });
    test('6.5 等亚像素暗星在不同像素落点仍可见，Canvas 与 GPU 均保留覆盖',()=>{
        const f=canvasFixture();
        for(const dpr of [1,2])for(const offset of [0,.25,.5,.75]) {
            const a=gpuStars([6.5],dpr,offset)[0];assert(a.count>0,JSON.stringify({dpr,offset,a}));
            const data=canvasStar(f,6.5,dpr,256+offset,256+offset),b=measure(data,512,256+offset,256+offset,8);
            assert(b.count>0,JSON.stringify({dpr,offset,b}));
        }
    });
    test('星点跨过 2 个设备像素的采样接口时，亮度和轮廓保持连续',()=>{
        const f=canvasFixture();
        for(const offset of [0,.25,.5,.75]) {
            const samples=[1.999,2.001].map(r=>{
                const zoom=r/pointRadius(0),canvas=measure(canvasStar(f,0,zoom,256+offset,256+offset),512,256+offset,256+offset,8,0);
                return {Canvas:canvas,WebGL:gpuStars([0],1,offset,zoom)[0]};
            });
            for(const kind of ['Canvas','WebGL']) {
                const a=samples[0][kind],b=samples[1][kind];assert(Math.abs(a.sum/b.sum-1)<.05 && Math.abs(a.width-b.width)<=1,JSON.stringify({kind,offset,a,b}));
            }
        }
        f.painter.clear();
    });
    renderer.setPixelRatio(1);renderer.setSize(256,256);
}
