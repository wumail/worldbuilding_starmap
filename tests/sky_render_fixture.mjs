import * as THREE from 'three';
import {DEG,projectHemisphere,unit,add,scale} from '../web/shared/solar_system.mjs';
import {atlasSymbolScale,displaySky,pointRadius,pointOpacity,skyBackground,diskBasis} from '../web/shared/sky_render.mjs';
import {AtlasBodyPainter} from '../web/v1/sky_atlas_bodies.mjs';
import {AtlasStarPainter} from '../web/v1/sky_atlas_stars.mjs';
import {SkyMotion3D} from '../web/v1/sky_motion_3d.js';
import {createStarMaterial} from '../web/v1/sky_points.js';
import {projectSkyPosition} from '../web/shared/sky_projection.js';
import {sanitizeState} from '../web/v1/sky_state.mjs';

export const kinds=['Canvas','perspective','stereographic'];
export function sampleMoon(id,phase=0,view=[0,0,1],magnitude=-10) {
    const basis=diskBasis(view);
    return {id,kind:'moon',color:'#ffffff',view,distance:id==='Luna'?.0026:.0052,
        angularDiameter:id==='Luna'?.5275:.1149,magnitude,altitude:Math.asin(view[2])/DEG,brightEnough:true,visible:true,
        lightDirection:add(scale(basis.z,Math.cos(phase*DEG)),scale(basis.x,Math.sin(phase*DEG)))};
}
export function renderFixture(kind,renderer,size=384) {
    const canvas=document.createElement('canvas'),ctx=canvas.getContext('2d'),painter=new AtlasBodyPainter(ctx),starPainter=new AtlasStarPainter(ctx);
    const scene=new THREE.Scene(),camera=new THREE.PerspectiveCamera(60,1,.1,5000);
    const controls={clock:{state:sanitizeState({displayScale:1,projection:kind,markers:false,solarGlow:false,planetGlow:false})}};
    const motion=new SkyMotion3D(scene,camera,document.createElement('div'),renderer,controls);
    const geometry=new THREE.BufferGeometry(),material=createStarMaterial(),points=new THREE.Points(geometry,material);scene.add(points);points.frustumCulled=false;
    return {scene,camera,motion,controls,show(bodies,{focus=bodies[0]?.view || [0,0,1],focal=24000,dpr=1,enhanced=false,solarGlow=false,planetGlow=false,stars=[],surface=false,night=1,gain=1,north=focus[2]>=0}={}) {
        const sky={days:0,night,limitingMagnitude:6.5,frame:{surface,matrix:[[1,0,0],[0,1,0],[0,0,1]]},bodies},background=skyBackground(sky);
        const layout={radius:focal*Math.PI/2,north:[0,0],south:[focal*4,0]},origin=projectHemisphere(focus,layout.radius,layout,0,north);
        const width=size*dpr;
        camera.fov=kind==='stereographic'?4*Math.atan(size/(4*focal))/DEG:2*Math.atan(size/(2*focal))/DEG;
        camera.up.set(...(Math.abs(focus[2])>.99?[0,1,0]:[0,0,1]));camera.lookAt(...focus);camera.updateProjectionMatrix();camera.updateMatrixWorld();
        camera.userData.skyProjection=kind==='stereographic'?kind:'perspective';
        const locate=view=>{
            if(kind==='Canvas') {const p=projectHemisphere(view,layout.radius,layout,0,north);return [(size/2+p.x-origin.x)*dpr,(size/2+p.y-origin.y)*dpr];}
            const p=projectSkyPosition(new THREE.Vector3(...view),camera);return [(p.x+1)*width/2,(1-p.y)*width/2];
        };
        let data;
        if(kind==='Canvas') {
            if(canvas.width!==width)canvas.width=canvas.height=width;
            ctx.resetTransform();ctx.fillStyle='rgb('+background.join(',')+')';ctx.fillRect(0,0,width,width);
            ctx.setTransform(dpr,0,0,dpr,(size/2-origin.x)*dpr,(size/2-origin.y)*dpr);
            for(const star of stars)starPainter.draw(star,projectHemisphere(star.view,layout.radius,layout,0,north),atlasSymbolScale(layout.radius)*gain);
            painter.drawSky(displaySky(sky,gain),layout,0,{symbolScale:atlasSymbolScale(layout.radius)*gain,lunarGlow:enhanced,solarGlow,planetGlow});
            data=ctx.getImageData(0,0,width,width).data;
        } else {
            renderer.setSize(size,size);renderer.setPixelRatio(dpr);
            for(const record of motion.disks.values()){record.mesh.visible=record.point.visible=false;if(record.glare)record.glare.visible=false;}
            geometry.setAttribute('position',new THREE.Float32BufferAttribute(stars.flatMap(s=>s.view.map(v=>v*1000)),3));
            geometry.setAttribute('color',new THREE.Float32BufferAttribute(stars.flatMap(s=>new THREE.Color(s.color_hex).toArray()),3));
            geometry.setAttribute('mag',new THREE.Float32BufferAttribute(stars.map(s=>s.app_mag),1));points.visible=stars.length>0;
            material.uniforms.displayScale.value=gain;material.uniforms.pixelRatio.value=dpr;material.uniforms.surface.value=surface;
            Object.assign(controls.clock.state,{displayScale:gain,lunarGlow:enhanced,solarGlow,planetGlow});motion.update(sky,pointRadius,pointOpacity);renderer.render(scene,camera);
            const gl=renderer.getContext(),raw=new Uint8Array(width*width*4);data=new Uint8Array(raw.length);
            gl.readPixels(0,0,width,width,gl.RGBA,gl.UNSIGNED_BYTE,raw);
            for(let y=0;y<width;y++)data.set(raw.subarray((width-y-1)*width*4,(width-y)*width*4),y*width*4);
        }
        return {data,width,height:width,background,locate,focal,dpr,kind};
    },close(){
        painter.clear();starPainter.clear();canvas.width=canvas.height=0;
        scene.traverse(o=>{o.geometry?.dispose();o.material?.uniforms?.map?.value?.dispose();o.material?.dispose();});
    }};
}
export function pixelMetrics(p,threshold=2) {
    let sum=0,count=0,minX=Infinity,maxX=-Infinity,minY=Infinity,maxY=-Infinity,xSum=0,ySum=0;
    for(let i=0;i<p.data.length;i+=4) {
        const value=Math.max(0,...p.background.map((v,c)=>p.data[i+c]-v)),x=i/4%p.width,y=Math.floor(i/4/p.width);
        sum+=value;xSum+=(x+.5)*value;ySum+=(y+.5)*value;
        if(value>threshold){count++;minX=Math.min(minX,x);maxX=Math.max(maxX,x);minY=Math.min(minY,y);maxY=Math.max(maxY,y);}
    }
    return {sum,count,width:count?maxX-minX+1:0,height:count?maxY-minY+1:0,x:xSum/sum,y:ySum/sum};
}
export const offsetDirection=(view,x,y)=>{const b=diskBasis(view);return unit(add(view,add(scale(b.x,x),scale(b.y,y))));};
