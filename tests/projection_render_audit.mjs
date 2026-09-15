import * as THREE from 'three';
import {SkyProjection,skyDirectionAt,projectSkyPosition} from '../web/shared/sky_projection.js';
import {SkyView,pickNearestStar} from '../web/v1/sky_view.js';
import {SkyMotion3D} from '../web/v1/sky_motion_3d.js';
import {pointRadius,pointOpacity} from '../web/shared/sky_render.mjs';
import {sampleMoon} from './sky_render_fixture.mjs';
import {DEG} from '../web/shared/solar_system.mjs';

export function runProjectionRenderAudit({test,assert,renderer}) {
    function read(size=512) {const gl=renderer.getContext(),data=new Uint8Array(size*size*4);gl.readPixels(0,0,size,size,gl.RGBA,gl.UNSIGNED_BYTE,data);return data;}
    test('实际 GPU：立体投影的离轴天球圆保持圆形；透视按几何产生椭圆',()=>{
        renderer.setPixelRatio(1);renderer.setSize(512,512);renderer.setClearColor(0);
        const scene=new THREE.Scene(),camera=new THREE.PerspectiveCamera(100,1,.1,5000),projection=new SkyProjection(scene,camera,renderer);
        const line=new THREE.LineLoop(new THREE.BufferGeometry(),new THREE.LineBasicMaterial({color:0xffffff}));scene.add(line);
        const ratios=[];
        for(const mode of ['perspective','stereographic'])for(const offset of [0,30,40]) {
            const a=offset*DEG,r=10*DEG,c=new THREE.Vector3(Math.sin(a),0,-Math.cos(a)),t=new THREE.Vector3(Math.cos(a),0,Math.sin(a));
            line.geometry.dispose();line.geometry=new THREE.BufferGeometry().setFromPoints(Array.from({length:512},(_,i)=>c.clone().multiplyScalar(Math.cos(r)).addScaledVector(t,Math.sin(r)*Math.cos(i/512*Math.PI*2)).addScaledVector(new THREE.Vector3(0,1,0),Math.sin(r)*Math.sin(i/512*Math.PI*2)).multiplyScalar(1000)));
            projection.update(mode);renderer.render(scene,camera);const data=read();let minX=512,maxX=-1,minY=512,maxY=-1;
            for(let y=0;y<512;y++)for(let x=0;x<512;x++)if(data[4*(y*512+x)]>80){minX=Math.min(minX,x);maxX=Math.max(maxX,x);minY=Math.min(minY,y);maxY=Math.max(maxY,y);}
            const width=maxX-minX+1,height=maxY-minY+1,expected=mode==='stereographic'?1:1/Math.sqrt(1-Math.sin(a)**2/Math.cos(r)**2);
            assert(width>20 && height>20,'没有绘制圆');assert(Math.abs(width/height-expected)<.035,JSON.stringify({mode,offset,width,height,expected}));ratios.push({mode,offset,ratio:width/height});
        }
        document.querySelector('#status').dataset.projectionCircles=JSON.stringify(ratios);line.geometry.dispose();line.material.dispose();
    });
    test('投影、反投影、DOM 标签、拾取与抓取拖动使用同一方向',()=>{
        renderer.setPixelRatio(1);renderer.setSize(512,512);
        const scene=new THREE.Scene(),camera=new THREE.PerspectiveCamera(80,1,.1,5000),stage=document.createElement('div');stage.style.cssText='width:512px;height:512px;position:absolute;left:-10000px';document.body.append(stage);
        const element=document.createElement('canvas');element.style.cssText='width:512px;height:512px';stage.append(element);
        const view=new SkyView(camera,element),controls={clock:{state:{projection:'perspective',trail:'off',markers:true,solarGlow:false,lunarGlow:false,planetGlow:false}}},motion=new SkyMotion3D(scene,camera,stage,renderer,controls);
        for(const mode of ['perspective','stereographic'])for(const pair of [[0,0],[.6,.35],[-.75,-.5]]) {
            camera.fov=80;camera.updateProjectionMatrix();camera.userData.skyProjection=mode;controls.clock.state.projection=mode;camera.updateMatrixWorld();
            const direction=skyDirectionAt(...pair,camera),body=sampleMoon('Luna',0,direction.toArray());
            const sky={days:0,night:1,frame:{surface:false,matrix:[[1,0,0],[0,1,0],[0,0,1]]},bodies:[body]};motion.update(sky,pointRadius,pointOpacity);
            const p=projectSkyPosition(body.normalizedPos,camera),x=(pair[0]+1)*256,y=(1-pair[1])*256;
            assert(Math.hypot(p.x-pair[0],p.y-pair[1])<1e-10,'正反投影不一致');
            const label=motion.disks.get('Luna').label;assert(label.style.display!=='none' && Math.abs(parseFloat(label.style.left)-x)<1e-6 && Math.abs(parseFloat(label.style.top)-y)<1e-6,'标签错位');
            assert(pickNearestStar([body],camera,x,y,{left:0,top:0,width:512,height:512},1)===body,'屏幕天体不可拾取');
            view.rotate(25,-18,x,y);camera.updateMatrixWorld();const after=projectSkyPosition(body.normalizedPos,camera);
            assert(Math.hypot((after.x+1)*256-x-25,(1-after.y)*256-y+18)<1e-6,'拖动未抓住原方向');
            view.focus(direction,0);camera.updateMatrixWorld();assert(Math.hypot(...Object.values(projectSkyPosition(body.normalizedPos,camera)).slice(0,2))<1e-9,'聚焦未居中');
        }
        stage.remove();view.dispose();
    });
    test('两种投影在倾斜视角下：实际地平线、月面裁剪与逆投影高度一致',()=>{
        renderer.setPixelRatio(1);renderer.setSize(512,512);
        const scene=new THREE.Scene(),camera=new THREE.PerspectiveCamera(80,1,.1,5000),controls={clock:{state:{trail:'off',markers:false,solarGlow:false,lunarGlow:false,planetGlow:false}}};
        camera.up.set(0,0,1);camera.lookAt(Math.cos(10*DEG),Math.sin(10*DEG),.2);camera.rotateZ(.37);camera.updateMatrixWorld();
        const motion=new SkyMotion3D(scene,camera,document.createElement('div'),renderer,controls),body={...sampleMoon('Luna',0,[1,0,0]),angularDiameter:12};
        for(const mode of ['perspective','stereographic']) {
            controls.clock.state.projection=mode;motion.update({days:0,night:1,frame:{surface:true,matrix:[[1,0,0],[0,1,0],[0,0,1]]},bodies:[body]},pointRadius,pointOpacity);renderer.render(scene,camera);
            const data=read();let bright=0,leak=0;
            for(let y=0;y<512;y++)for(let x=0;x<512;x++)if(data[4*(y*512+x)]>100){bright++;if(skyDirectionAt((x+.5)/256-1,(y+.5)/256-1,camera).z<-.002)leak++;}
            assert(bright>100 && leak===0,JSON.stringify({mode,bright,leak}));
        }
    });
    renderer.setPixelRatio(1);renderer.setSize(256,256);
}
