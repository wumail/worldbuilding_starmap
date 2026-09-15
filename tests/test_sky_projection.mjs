import test from 'node:test';
import assert from 'node:assert/strict';
import {projectionFocal,projectViewDirection,unprojectViewDirection} from '../web/shared/sky_projection.mjs';
const DEG=Math.PI/180;
test('正反投影保持同一单位方向，覆盖极宽屏、离轴和背向半球',()=>{
    for(const mode of ['perspective','stereographic'])for(const fov of [1,10,60,100])for(const aspect of [.5,1,2,4])for(const x of [-.98,-.4,0,.5,.98])for(const y of [-.98,-.2,0,.7,.98]) {
        const direction=unprojectViewDirection(x,y,fov,aspect,mode),p=projectViewDirection(direction,fov,aspect,mode);
        assert.ok(Math.abs(Math.hypot(...direction)-1)<1e-12);assert.ok(p.visible);assert.ok(Math.hypot(p.x-x,p.y-y)<1e-11);
    }
});
test('垂直视场、中心角尺度与定义一致，不把两种投影的焦距混用',()=>{
    for(const mode of ['perspective','stereographic'])for(const fov of [1,10,60,100]) {
        const edge=unprojectViewDirection(0,1,fov,1,mode);assert.ok(Math.abs(Math.acos(-edge[2])/DEG-fov/2)<1e-9);
        const h=800,eps=1e-7,p=projectViewDirection([Math.sin(eps),0,-Math.cos(eps)],fov,1,mode);
        assert.ok(Math.abs(p.x*h/2/eps-projectionFocal(h,fov,mode))<1e-6);
    }
});
test('立体投影的球面小圆映成圆，透视天空盒和球壳的结果相同',()=>{
    for(const offset of [0,30,40,50]) {
        const a=offset*DEG,r=10*DEG,project=t=>{const v=[Math.sin(a)*Math.cos(r)+Math.cos(a)*Math.sin(r)*Math.cos(t),Math.sin(r)*Math.sin(t),-Math.cos(a)*Math.cos(r)+Math.sin(a)*Math.sin(r)*Math.cos(t)];return projectViewDirection(v,100,1,'stereographic');};
        const left=project(Math.PI),right=project(0),cx=(left.x+right.x)/2,radius=(right.x-left.x)/2;
        for(let i=0;i<720;i++){const p=project(i*Math.PI/360);assert.ok(Math.abs(Math.hypot(p.x-cx,p.y)-radius)<1e-12);}
        for(const v of [[.3,.4,-.8],[.8,.2,-.6]]) {
            const cube=v.map(x=>x/Math.max(...v.map(Math.abs))),p=projectViewDirection(v,80,2),q=projectViewDirection(cube,80,2);assert.ok(Math.hypot(p.x-q.x,p.y-q.y)<1e-12);
        }
    }
});
