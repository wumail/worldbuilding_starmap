import * as THREE from 'three';
import {projectSkyPosition,skyDirectionAt} from '../shared/sky_projection.js';

// 天球的观测者固定在原点。拖动只改变朝向，滚轮只改变视场。
export class SkyView {
    constructor(camera, element, {onZoom=()=>{}}={}) {
        this.onZoom=onZoom;this.enabled=true;
        this.camera = camera;
        this.element = element;
        this.pointer = null;
        this.transition = null;
        this.handlers = {
            pointerdown: event => {
                if (!this.enabled || event.button !== 0 || this.pointer) return;
                this.transition = null;
                this.pointer = { id: event.pointerId, x: event.clientX, y: event.clientY };
                element.setPointerCapture(event.pointerId);
            },
            pointermove: event => {
                if (!this.pointer || event.pointerId !== this.pointer.id) return;
                const bounds=element.getBoundingClientRect();
                this.rotate(event.clientX - this.pointer.x, event.clientY - this.pointer.y,this.pointer.x-bounds.left,this.pointer.y-bounds.top);
                this.pointer.x = event.clientX;
                this.pointer.y = event.clientY;
            },
            pointerup: event => this.releasePointer(event),
            pointercancel: event => this.releasePointer(event),
            wheel: event => {
                event.preventDefault();
                if(this.enabled)this.zoom(event.deltaY, event.deltaMode);
            },
        };
        element.style.touchAction = 'none';
        for (const [event, handler] of Object.entries(this.handlers)) {
            element.addEventListener(event, handler, event === 'wheel' ? { passive: false } : undefined);
        }
        this.reset();
    }

    releasePointer(event) {
        if (!this.pointer || event.pointerId !== this.pointer.id) return;
        if (this.element.hasPointerCapture(event.pointerId)) this.element.releasePointerCapture(event.pointerId);
        this.pointer = null;
    }

    rotate(dx,dy,x=this.element.clientWidth/2,y=this.element.clientHeight/2) {
        this.transition=null;
        const width=Math.max(1,this.element.clientWidth),height=Math.max(1,this.element.clientHeight);
        const before=skyDirectionAt(2*x/width-1,1-2*y/height,this.camera);
        const after=skyDirectionAt(2*(x+dx)/width-1,1-2*(y+dy)/height,this.camera);
        this.camera.quaternion.premultiply(new THREE.Quaternion().setFromUnitVectors(after,before));
    }

    zoom(deltaY, mode = 0) {
        const pixels = deltaY * (mode === 1 ? 16 : mode === 2 ? this.element.clientHeight : 1);
        const divisor=this.camera.userData.skyProjection==='stereographic'?4:2;
        const tangent=Math.tan(this.camera.fov*Math.PI/180/divisor)*Math.exp(pixels*.001);
        this.camera.fov=divisor*Math.atan(tangent)*180/Math.PI;
        this.camera.updateProjectionMatrix();this.onZoom();
    }

    focus(direction, duration = 700, now = performance.now()) {
        if (!direction || direction.lengthSq() === 0) return;
        const target = direction.clone().normalize();
        const up = Math.abs(target.z) > 1 - 1e-10 ? new THREE.Vector3(0, 1, 0) : new THREE.Vector3(0, 0, 1);
        const rotation = new THREE.Matrix4().lookAt(new THREE.Vector3(), target, up);
        this.transition = { from: this.camera.quaternion.clone(), to: new THREE.Quaternion().setFromRotationMatrix(rotation),
            startedAt: now, duration: Math.max(0, duration) };
        this.update(now);
    }

    update(now = performance.now()) {
        if (!this.transition) return;
        const { from, to, startedAt, duration } = this.transition;
        const progress = duration === 0 ? 1 : THREE.MathUtils.clamp((now - startedAt) / duration, 0, 1);
        this.camera.quaternion.slerpQuaternions(from, to, 1 - (1 - progress) ** 3);
        if (progress === 1) this.transition = null;
    }

    reset() {
        this.transition = null;
        this.camera.position.set(0, 0, 0);
        this.camera.up.set(0, 0, 1);
        this.camera.lookAt(new THREE.Vector3(1, 0, 0));
        this.camera.fov = 60;
        this.camera.updateProjectionMatrix();
    }

    dispose() {
        for (const [event, handler] of Object.entries(this.handlers)) this.element.removeEventListener(event, handler);
        this.transition = null;
    }
}

export function pickNearestStar(stars, camera, clientX, clientY, bounds, tolerancePixels = 25) {
    camera.updateMatrixWorld();
    let nearest = null;
    let best = tolerancePixels;
    for (const star of stars) {
        if (!star.normalizedPos) continue;
        const projected = projectSkyPosition(star.normalizedPos,camera);
        if (!projected.visible) continue;
        const x = bounds.left + (projected.x + 1) * bounds.width / 2;
        const y = bounds.top + (1 - projected.y) * bounds.height / 2;
        const distance = Math.hypot(x - clientX, y - clientY);
        if (distance < best) {
            best = distance;
            nearest = star;
        }
    }
    return nearest;
}
