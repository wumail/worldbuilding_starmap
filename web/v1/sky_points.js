import * as THREE from 'three';
import {POINT_STYLE,SYMBOL_REFERENCE} from '../shared/sky_render.mjs';
import {projectionVertexGLSL,projectionFragmentGLSL} from '../shared/sky_projection.mjs';
import {pointKernelAtlas} from '../shared/sky_point_kernel.mjs';
import {updateProjectionUniforms} from '../shared/sky_projection.js';

// 从共享参数生成 GLSL，避免两页各自维护不同的星等映射。
const glsl=value=>Number(value).toFixed(8);
export const vertexShader = projectionVertexGLSL+`
            attribute float mag;
            attribute vec3 color;
            uniform float pixelRatio;
            uniform float pointWeight;
            uniform float displayScale;
            uniform float viewportHeight;
            uniform float limitingMagnitude;
            uniform bool surface;
            varying vec3 vColor;
            varying float vAlpha;
            varying float vVisible;
            varying float vRadius;
            varying float vSize;
            varying vec2 vCenter;
            varying vec3 vViewDirection;
            void main() {
                vColor = color;
                vVisible = (mag <= min(6.5, limitingMagnitude) && (!surface || position.z >= 0.0)) ? 1.0 : 0.0;
                float relativeMagnitude = ${glsl(POINT_STYLE.referenceMagnitude)} - mag;
                vAlpha = pointWeight * min(1.0, ${glsl(POINT_STYLE.referenceOpacity)} * pow(10.0, ${glsl(POINT_STYLE.opacityExponent)} * relativeMagnitude));
                float radius = min(${glsl(POINT_STYLE.maximumRadius)}, ${glsl(POINT_STYLE.referenceRadius)} * pow(10.0, ${glsl(POINT_STYLE.radiusExponent)} * relativeMagnitude));
                float symbolScale = (skyFocal>0.0?skyFocal:viewportHeight*projectionMatrix[1][1]/2.0) / ${glsl(SYMBOL_REFERENCE.perspectiveFocal)};
                vRadius = radius * pixelRatio * symbolScale * displayScale;
                // 留出像素采样边界；这是透明绘制区域，不是把暗星半径抬到统一下限。
                vSize = vRadius<2.0?5.0:2.0*vRadius+4.0;
                gl_PointSize = vSize;
                vec3 viewPosition=(modelViewMatrix * vec4(position, 1.0)).xyz;
                vViewDirection=normalize(viewPosition);
                gl_Position = skyProject(viewPosition);
                vCenter=(gl_Position.xy/gl_Position.w+1.0)*skyViewport/2.0;
            }
        `;
export const fragmentShader = projectionFragmentGLSL+`
            uniform bool useWhiteColor;
            uniform sampler2D kernelAtlas;
            uniform bool surface;
            varying vec3 vColor;
            varying float vAlpha;
            varying float vVisible;
            varying float vRadius;
            varying float vSize;
            varying vec2 vCenter;
            varying vec3 vViewDirection;
            float resolvedPixel(vec2 p,float radius) {
                float coverage=0.0;
                for(int y=0;y<4;y++)for(int x=0;x<4;x++) {
                    vec2 o=(vec2(float(x),float(y))+.5)/4.0-.5;
                    float profile=1.0-smoothstep(${glsl(POINT_STYLE.coreFraction)},1.0,length(p+o)/radius);
                    float energy=${glsl(POINT_STYLE.exposure)}*profile;
                    coverage+=energy/(1.0+energy);
                }
                return coverage/16.0;
            }
            void main() {
                if (vVisible < .5 || dot(skyRay(),vViewDirection)<=0.0 || (surface && belowSkyHorizon())) discard;
                vec2 p = skyViewport.x>0.0?gl_FragCoord.xy-vCenter:(gl_PointCoord-.5)*vSize;
                float opacity;
                if(vRadius<2.0) {
                    float index=ceil(vRadius*128.0);
                    vec2 origin=vec2(mod(index,16.0),floor(index/16.0))*7.0+1.0;
                    opacity=texture2D(kernelAtlas,(origin+2.5+p)/vec2(112.0,119.0)).a*pow(vRadius/(index/128.0),2.0);
                } else {
                    float radius=ceil(vRadius*128.0)/128.0;
                    vec2 q=floor(p),f=fract(p);
                    opacity=mix(mix(resolvedPixel(q,radius),resolvedPixel(q+vec2(1.0,0.0),radius),f.x),
                                mix(resolvedPixel(q+vec2(0.0,1.0),radius),resolvedPixel(q+vec2(1.0),radius),f.x),f.y)*pow(vRadius/radius,2.0);
                }
                gl_FragColor = vec4(useWhiteColor ? vec3(1.0) : vColor, vAlpha * opacity);
                #include <colorspace_fragment>
            }
        `;


let sharedKernelTexture;
function kernelTexture() {
    if(!sharedKernelTexture) {
        const image=pointKernelAtlas();sharedKernelTexture=new THREE.DataTexture(image.data,image.width,image.height);
        sharedKernelTexture.minFilter=sharedKernelTexture.magFilter=THREE.LinearFilter;sharedKernelTexture.needsUpdate=true;
    }
    return sharedKernelTexture;
}
export function createStarMaterial({white=false,pixelRatio=1,surface=false,viewportHeight=256}={}) {
    const material=new THREE.ShaderMaterial({
        uniforms: {displayScale:{value:1},kernelAtlas:{value:kernelTexture()},pointWeight:{value:1},useWhiteColor:{value:white},pixelRatio:{value:pixelRatio},viewportHeight:{value:viewportHeight},limitingMagnitude:{value:6.5},surface:{value:surface},skyStereo:{value:false},skyStereoFactor:{value:2},skyFocal:{value:-1},skyViewport:{value:new THREE.Vector2()},skyHalfSpan:{value:new THREE.Vector2(1,1)},skyWorldUp:{value:new THREE.Vector3(0,1,0)}},
        vertexShader,fragmentShader,transparent:true,depthTest:false,depthWrite:false,toneMapped:false,
    });
    material.onBeforeRender=(renderer,scene,camera)=>updateProjectionUniforms(material.uniforms,camera,renderer);
    return material;
}
