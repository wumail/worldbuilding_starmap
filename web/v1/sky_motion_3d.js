import {SKY_OVERLAYS} from '../shared/sky_overlays.mjs';
import * as THREE from 'three';
import {DEG} from '../shared/solar_system.mjs';
import {SYMBOL_REFERENCE,displaySky,trajectory,unresolvedPointWeight,skyBackground,srgbToLinear,SOLAR_GLARE_RADIUS_RATIO,bodyGlareImage,solarGlareVisibility,bodyFullyOcculted,lunarSurfaceScale} from '../shared/sky_render.mjs';

import {SkyProjection,projectSkyPosition} from '../shared/sky_projection.js';
import {projectionFocal,projectionMode} from '../shared/sky_projection.mjs';
import {createStarMaterial} from './sky_points.js';

const diskVertex=`varying vec2 diskUV; varying float worldHeight;varying vec3 diskViewCenter;
void main(){
    vec4 center=modelViewMatrix*vec4(0.0,0.0,0.0,1.0);
    diskViewCenter=normalize(center.xyz);
    float diameter=length(modelMatrix[0].xyz)*skyFocal/max(1.0,length(center.xyz));
    float rasterScale=max(1.0,4.0/max(.0001,diameter));
    diskUV=(uv*2.0-1.0)*rasterScale;
    vec4 p=modelMatrix*vec4(position*rasterScale,1.0);worldHeight=p.z;gl_Position=skyProject((viewMatrix*p).xyz);
}`;
const diskFragment=`uniform vec3 lightDirection;uniform vec3 tint;uniform vec3 background;uniform float lightScale;uniform bool luminous;uniform bool brightEnough;uniform bool surface;
varying vec2 diskUV;varying float worldHeight;varying vec3 diskViewCenter;
vec3 diskColor(vec2 uv){vec3 n=vec3(uv,sqrt(max(0.0,1.0-dot(uv,uv))));
float shade=luminous?1.0:(brightEnough?max(0.0,dot(n,lightDirection)):0.0);
return min(vec3(1.0),background+tint*shade*lightScale);}
void main(){
// A tangent disk cannot cover rays on the opposite side of its plane.
// Stereographic triangles crossing the antipode can otherwise fill the
// viewport while still interpolating valid-looking, opaque disk UVs.
vec2 dx=dFdx(diskUV),dy=dFdy(diskUV);float hx=dFdx(worldHeight),hy=dFdy(worldHeight);
if(dot(skyRay(),diskViewCenter)<=0.0)discard;
float footprint=max(length(dx),length(dy)),r=length(diskUV),alpha=0.0;vec3 color=vec3(0.0);
if(r>1.0+footprint)discard;
if(footprint<.1 && r<1.0-footprint && (!surface || worldHeight>abs(hx)+abs(hy))){color=diskColor(diskUV);alpha=1.0;}
else {
    for(int y=0;y<16;y++)for(int x=0;x<16;x++){
        vec2 o=(vec2(float(x),float(y))+.5)/16.0-.5,uv=diskUV+dx*o.x+dy*o.y;
        if(dot(uv,uv)>1.0 || (surface && worldHeight+hx*o.x+hy*o.y<0.0))continue;
        color+=diskColor(uv)/256.0;alpha+=1.0/256.0;
    }
    if(alpha==0.0)discard;color/=alpha;
}
gl_FragColor=vec4(color,alpha);
#include <colorspace_fragment>
}`;

export class SkyMotion3D {
    constructor(scene,camera,stage,renderer,controls) {
        this.scene=scene;this.camera=camera;this.stage=stage;this.renderer=renderer;this.controls=controls;
        this.projection=new SkyProjection(scene,camera,renderer);
        this.disks=new Map();this.currentBodies=[];this.trailKey='';
        this.labels=document.createElement('div');this.labels.id='body-labels';stage.append(this.labels);
        this.compass=document.createElement('div');this.compass.className='compass';stage.append(this.compass);
        this.clip=new THREE.Plane(new THREE.Vector3(0,0,1),0);
        this.trail=new THREE.LineSegments(new THREE.BufferGeometry(),new THREE.LineBasicMaterial({color:0x958367,transparent:true,opacity:SKY_OVERLAYS.line,depthTest:false}));
        this.trail.frustumCulled=false;this.trail.renderOrder=2;this.trail.matrixAutoUpdate=false;scene.add(this.trail);
        const groundMaterial=new THREE.ShaderMaterial({side:THREE.BackSide,depthTest:false,depthWrite:false,
            vertexShader:'varying float height;void main(){height=position.z;gl_Position=skyProject((modelViewMatrix*vec4(position,1.0)).xyz);}',
            fragmentShader:'varying float height;void main(){if(height>=0.0)discard;gl_FragColor=vec4(0.022,0.03,0.045,1.0);}',toneMapped:false});
        this.ground=new THREE.Mesh(new THREE.SphereGeometry(1200,64,32),groundMaterial);this.ground.renderOrder=-2;scene.add(this.ground);
        const horizonPoints=Array.from({length:256},(_,i)=>new THREE.Vector3(1000*Math.cos(i/256*Math.PI*2),1000*Math.sin(i/256*Math.PI*2),0));
        this.horizon=new THREE.LineLoop(new THREE.BufferGeometry().setFromPoints(horizonPoints),new THREE.LineBasicMaterial({color:0x5d6c80,transparent:true,opacity:SKY_OVERLAYS.line,depthTest:false}));scene.add(this.horizon);
        this.compassLabels=['北','西','南','东'].map((name,i)=>{const label=document.createElement('span');label.className='sky-label compass-label';label.textContent=name;this.labels.append(label);return {label,position:new THREE.Vector3(1000*Math.cos(i*Math.PI/2),1000*Math.sin(i*Math.PI/2),8)};});
    }
    createDisk(body) {
        const material=new THREE.ShaderMaterial({vertexShader:diskVertex,fragmentShader:diskFragment,
            uniforms:{lightDirection:{value:new THREE.Vector3(0,0,1)},tint:{value:new THREE.Color(body.color)},background:{value:new THREE.Color(0)},lightScale:{value:1},luminous:{value:body.id==='Sol'},brightEnough:{value:true},surface:{value:false}},
            transparent:true,depthTest:false,depthWrite:false,side:THREE.DoubleSide,toneMapped:false,extensions:{derivatives:true}});
        const mesh=new THREE.Mesh(new THREE.PlaneGeometry(1,1,24,24),material);mesh.frustumCulled=false;this.scene.add(mesh);
        let glare=null;
        if(body.id==='Sol') {
            const image=bodyGlareImage(body.color),texture=new THREE.DataTexture(image.data,image.width,image.height);
            texture.colorSpace=THREE.SRGBColorSpace;texture.minFilter=texture.magFilter=THREE.LinearFilter;texture.needsUpdate=true;
            // 太阳为柔光；双月为不透明的显示盘面，并按地平线裁剪。
            const material=new THREE.ShaderMaterial({
                uniforms:{map:{value:texture},opacity:{value:1},surface:{value:false}},
                vertexShader:'varying vec2 glareUV;varying float worldHeight;varying vec3 glareViewCenter;void main(){glareViewCenter=normalize((modelViewMatrix*vec4(0.,0.,0.,1.)).xyz);glareUV=uv;vec4 p=modelMatrix*vec4(position,1.0);worldHeight=p.z;gl_Position=skyProject((viewMatrix*p).xyz);}',
                fragmentShader:'uniform sampler2D map;uniform float opacity;uniform bool surface;varying vec2 glareUV;varying float worldHeight;varying vec3 glareViewCenter;void main(){if(dot(skyRay(),glareViewCenter)<=0.0 || (surface && worldHeight<0.0))discard;gl_FragColor=texture2D(map,glareUV);gl_FragColor.a*=opacity;\n#include <colorspace_fragment>\n}',
                transparent:true,depthTest:false,depthWrite:false,side:THREE.DoubleSide,toneMapped:false});
            glare=new THREE.Mesh(new THREE.PlaneGeometry(1,1,24,24),material);
            glare.frustumCulled=false;this.scene.add(glare);
        }
        // 行星亮度增强独立于实体盘面；大小与背景星采用相同的投影倍率。
        const pointMaterial=createStarMaterial({pixelRatio:this.renderer.getPixelRatio()});
        const geometry=new THREE.BufferGeometry().setFromPoints([new THREE.Vector3()]);
        geometry.setAttribute('mag',new THREE.Float32BufferAttribute([body.magnitude],1));
        geometry.setAttribute('color',new THREE.Float32BufferAttribute(new THREE.Color(body.color).toArray(),3));
        const point=new THREE.Points(geometry,pointMaterial);point.frustumCulled=false;this.scene.add(point);
        const label=document.createElement('span');label.className=body.kind==='moon'?'sky-label lunar-body':'sky-label';label.innerHTML=`<i class="body-marker"></i>${body.id}`;this.labels.append(label);
        const record={mesh,point,label,glare};this.disks.set(body.id,record);return record;
    }
    placeLabel(label,position) {
        const p=projectSkyPosition(position,this.camera);
        const visible=p.visible;
        label.style.display=visible?'':'none';if(!visible)return;
        label.style.left=`${(p.x+1)*this.stage.clientWidth/2}px`;label.style.top=`${(1-p.y)*this.stage.clientHeight/2}px`;
    }
    update(sky,pointRadius,pointOpacity) {
        const state=this.controls.clock.state,surface=sky.frame.surface,gain=state.displayScale ?? 1;
        sky=displaySky(sky,gain);
        this.currentBodies=sky.bodies;
        this.renderer.clippingPlanes=surface?[this.clip]:[];
        this.ground.visible=this.horizon.visible=surface;
        const background=new THREE.Color().fromArray(skyBackground(sky).map(c=>srgbToLinear(c/255)));this.renderer.setClearColor(background);
        this.camera.updateMatrixWorld();
        const mode=projectionMode(state.projection);
        this.camera.userData.skyProjection=mode;
        const focal=projectionFocal(this.renderer.getSize(new THREE.Vector2()).y,this.camera.fov,mode);
        const symbolScale=focal/SYMBOL_REFERENCE.perspectiveFocal;
        const sorted=[...sky.bodies].sort((a,b)=>b.distance-a.distance);
        const glareVisibility=state.solarGlow?solarGlareVisibility(sky):0;
        sorted.forEach((body,index)=>{
            const {mesh,point,label,glare}=this.disks.get(body.id) || this.createDisk(body);
            const radius=900;
            mesh.position.fromArray(body.view).multiplyScalar(radius);mesh.lookAt(0,0,0);
            mesh.scale.setScalar(2*radius*Math.tan(body.angularDiameter*DEG/2));
            const occulted=bodyFullyOcculted(body,sky);
            mesh.visible=!occulted && (!surface || body.altitude+body.angularDiameter/2>=0);
            mesh.material.uniforms.surface.value=surface;mesh.material.uniforms.brightEnough.value=body.brightEnough;
            mesh.material.uniforms.background.value.copy(background);
            mesh.material.uniforms.lightScale.value=lunarSurfaceScale(body,state.lunarGlow);
            if(body.lightDirection)mesh.material.uniforms.lightDirection.value.fromArray(body.lightDirection).applyQuaternion(mesh.quaternion.clone().invert());
            mesh.renderOrder=21+index*3;
            if(glare) {
                glare.position.copy(mesh.position);glare.quaternion.copy(mesh.quaternion);
                glare.scale.setScalar(mesh.scale.x*SOLAR_GLARE_RADIUS_RATIO);
                glare.visible=glareVisibility>0;
                glare.material.uniforms.opacity.value=glareVisibility;
                glare.material.uniforms.surface.value=surface;
                glare.renderOrder=mesh.renderOrder+1;
            }
            const referenceDiameter=2*SYMBOL_REFERENCE.perspectiveFocal*Math.tan(body.angularDiameter*DEG/2)/gain;
            const weight=unresolvedPointWeight(referenceDiameter);
            point.position.copy(mesh.position);point.visible=!occulted && state.planetGlow!==false && body.kind==='planet' && body.visible && weight>0;
            point.geometry.getAttribute('mag').setX(0,body.magnitude);point.geometry.getAttribute('mag').needsUpdate=true;
            // 把点的位置存进几何，使背景星与行星共用的地平筛选读取世界方向。
            point.position.set(0,0,0);point.geometry.getAttribute('position').setXYZ(0,...mesh.position.toArray());point.geometry.getAttribute('position').needsUpdate=true;
            point.material.uniforms.surface.value=surface;point.material.uniforms.limitingMagnitude.value=sky.limitingMagnitude;
            point.material.uniforms.displayScale.value=gain;point.material.uniforms.pointWeight.value=weight;point.material.uniforms.pixelRatio.value=this.renderer.getPixelRatio();point.renderOrder=mesh.renderOrder+1;
            body.normalizedPos=mesh.position.clone();
            label.classList.toggle('dim',!body.visible);label.classList.toggle('selected-body',body.id===state.selected);
            if(state.markers)this.placeLabel(label,mesh.position);else label.style.display='none';
        });
        for(const {label,position} of this.compassLabels) { if(surface)this.placeLabel(label,position);else label.style.display='none'; }
        const direction=this.camera.getWorldDirection(new THREE.Vector3());
        this.compass.textContent=surface?`方位 ${((Math.atan2(-direction.y,direction.x)/DEG+360)%360).toFixed(1)}° · 高度 ${(Math.asin(direction.z)/DEG).toFixed(1)}°`:`赤经 ${((Math.atan2(direction.y,direction.x)/DEG+360)%360).toFixed(1)}° · 赤纬 ${(Math.asin(direction.z)/DEG).toFixed(1)}°`;
        const key=JSON.stringify([state.trail,state.selected,state.mode,state.latitude,state.longitude,state.spinPhase,state.angles,Math.floor(sky.days/(state.trail==='day'?.003:1))]);
        if(key!==this.trailKey) {
            const points=trajectory(sky.days,state,state.selected,state.trail),vertices=[];
            const coordinateKey=state.trail==='year'?'equatorial':'direction';
            for(let i=1;i<points.length;i++)vertices.push(...points[i-1][coordinateKey].map(v=>v*998),...points[i][coordinateKey].map(v=>v*998));
            this.trail.geometry.dispose();this.trail.geometry=new THREE.BufferGeometry();this.trail.geometry.setAttribute('position',new THREE.Float32BufferAttribute(vertices,3));this.trailKey=key;
        }
        if(state.trail==='year') {const m=sky.frame.matrix;this.trail.matrix.set(...m[0],0,...m[1],0,...m[2],0,0,0,0,1);}else this.trail.matrix.identity();
        this.trail.matrixWorldNeedsUpdate=true;
        this.trail.visible=state.trail!=='off';
        this.projection.update(mode);
        this.renderer.domElement.dataset.projection=mode;
        this.renderer.domElement.dataset.symbolScale=String(symbolScale);this.renderer.domElement.dataset.fov=String(this.camera.fov);
    }
}
