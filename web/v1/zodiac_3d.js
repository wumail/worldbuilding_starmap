import * as THREE from 'three';
import {zodiacLines,zodiacLabels} from '../shared/zodiac.mjs';
import {SKY_OVERLAYS} from '../shared/sky_overlays.mjs';

export class Zodiac3D {
    constructor(scene){
        this.group=new THREE.Group();this.group.matrixAutoUpdate=false;scene.add(this.group);
        this.uniforms={surface:{value:false}};
        const positions=[];
        for(const line of zodiacLines)for(let i=1;i<line.points.length;i++)positions.push(...line.points[i-1].map(x=>1000*x),...line.points[i].map(x=>1000*x));
        const geometry=new THREE.BufferGeometry();geometry.setAttribute('position',new THREE.Float32BufferAttribute(positions,3));
        const material=new THREE.ShaderMaterial({uniforms:this.uniforms,depthTest:false,depthWrite:false,transparent:true,toneMapped:false,
            vertexShader:'varying float h;void main(){vec4 p=modelMatrix*vec4(position,1.);h=p.z;gl_Position=skyProject((viewMatrix*p).xyz);}',
            fragmentShader:`uniform bool surface;varying float h;void main(){if(surface && h<0.)discard;gl_FragColor=vec4(.48,.38,.22,${SKY_OVERLAYS.line});\n#include <colorspace_fragment>\n}`});
        const mesh=new THREE.LineSegments(geometry,material);mesh.renderOrder=1;this.group.add(mesh);
        for(const label of zodiacLabels){
            const canvas=document.createElement('canvas');canvas.width=256;canvas.height=64;const ctx=canvas.getContext('2d');
            ctx.font='32px sans-serif';ctx.textAlign='center';ctx.fillStyle='#c8b68e';ctx.fillText(label.text,128,43);
            const map=new THREE.CanvasTexture(canvas);map.colorSpace=THREE.SRGBColorSpace;
            const sprite=new THREE.Sprite(new THREE.SpriteMaterial({map,transparent:true,opacity:SKY_OVERLAYS.text,depthTest:false,depthWrite:false,toneMapped:false}));
            sprite.position.fromArray(label.direction).multiplyScalar(1000);sprite.scale.set(130,32.5,1);sprite.renderOrder=1;this.group.add(sprite);
        }
    }
    update(sky,enabled){
        this.group.visible=enabled;this.uniforms.surface.value=sky.frame.surface;
        const m=sky.frame.matrix;this.group.matrix.set(...m[0],0,...m[1],0,...m[2],0,0,0,0,1);this.group.matrixWorldNeedsUpdate=true;
        // Sprite shaders inherit Three's world clipping plane from the viewer.
    }
}
