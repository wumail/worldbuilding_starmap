import * as THREE from 'three';
import {projectionMode,projectionFocal,projectViewDirection,unprojectViewDirection,projectionVertexGLSL,projectionFragmentGLSL} from './sky_projection.mjs';

export function projectSkyPosition(position,camera) {
    const v=position.clone().applyMatrix4(camera.matrixWorldInverse);
    return projectViewDirection(v.toArray(),camera.fov,camera.aspect,camera.userData.skyProjection);
}
export function skyDirectionAt(x,y,camera) {
    return new THREE.Vector3(...unprojectViewDirection(x,y,camera.fov,camera.aspect,camera.userData.skyProjection)).applyQuaternion(camera.quaternion);
}
export function updateProjectionUniforms(u,camera,renderer) {
    const stereo=camera.userData.skyProjection==='stereographic',half=camera.fov*Math.PI/360,t=Math.tan(half/(stereo?2:1));
    u.skyStereo.value=stereo;u.skyStereoFactor.value=Math.tan(half)/Math.tan(half/2);
    u.skyFocal.value=projectionFocal(renderer.getSize(u.skyViewport.value).y,camera.fov,camera.userData.skyProjection);
    renderer.getDrawingBufferSize(u.skyViewport.value);u.skyHalfSpan.value.set(camera.aspect*t,t);
    const m=camera.matrixWorld.elements;u.skyWorldUp.value.set(m[2],m[6],m[10]);
}

// 一个控制器覆盖场景全部材质，包括参考文字、辅助标记和新加载的星表。
export class SkyProjection {
    constructor(scene,camera,renderer) {
        this.scene=scene;this.camera=camera;this.renderer=renderer;this.materials=new WeakSet();
        this.uniforms={skyStereo:{value:false},skyStereoFactor:{value:2},skyFocal:{value:1},
            skyViewport:{value:new THREE.Vector2()},skyHalfSpan:{value:new THREE.Vector2()},skyWorldUp:{value:new THREE.Vector3()}};
    }
    update(mode=this.camera.userData.skyProjection) {
        const camera=this.camera,u=this.uniforms,stereo=projectionMode(mode)==='stereographic';
        camera.userData.skyProjection=stereo?'stereographic':'perspective';camera.updateMatrixWorld();
        updateProjectionUniforms(u,camera,this.renderer);
        this.scene.traverse(object=>{
            if(!object.material)return;
            // Three 的线性透视包围盒不能代替立体投影的可见域。
            object.frustumCulled=false;
            for(const material of Array.isArray(object.material)?object.material:[object.material])this.attach(material);
        });
    }
    attach(material) {
        if(this.materials.has(material))return;
        this.materials.add(material);
        if(material.uniforms)Object.assign(material.uniforms,this.uniforms);
        const previous=material.onBeforeCompile;
        material.onBeforeCompile=(shader,renderer)=>{
            previous.call(material,shader,renderer);Object.assign(shader.uniforms,this.uniforms);
            if(!shader.vertexShader.includes('uniform bool skyStereo;'))shader.vertexShader=projectionVertexGLSL+shader.vertexShader;
            if(!shader.fragmentShader.includes('uniform bool skyStereo;'))shader.fragmentShader=projectionFragmentGLSL+shader.fragmentShader;
            if(shader.vertexShader.includes('#include <project_vertex>'))
                shader.vertexShader=shader.vertexShader.replace('#include <project_vertex>','#include <project_vertex>\ngl_Position=skyProject(mvPosition.xyz);');
            else shader.vertexShader=shader.vertexShader.replace(/gl_Position\s*=\s*projectionMatrix\s*\*\s*mvPosition\s*;/g,'gl_Position=skyProject(mvPosition.xyz);');
        };
        material.customProgramCacheKey=()=>`terrax-projection-1`;material.needsUpdate=true;
    }
}
