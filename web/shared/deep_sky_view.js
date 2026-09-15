import * as THREE from 'three';
import {galacticToEquatorial} from './solar_system.mjs';
import {prepareDeepSources,SOURCE_GRID,SOURCE_TEXELS} from './deep_sky_profiles.mjs';
import {morphologyGLSL,FILTER_SIZE} from './nebula_morphology.mjs';

// Analytic pixel-to-direction Jacobians. Unlike fragment derivatives these
// remain defined at a clipped horizon and at either atlas hemisphere border.
// Include after the renderer's shared projection uniforms are declared.
export const diffuseFootprintGLSL=`
void diffuseFootprint(vec2 pixel,out vec3 dx,out vec3 dy){
 if(projectionKind==2){
   vec2 p=pixel-atlasNorth.xy;float z=1.,radius=atlasNorth.z;
   if(length(p)>radius){p=pixel-atlasSouth.xy;z=-1.;radius=atlasSouth.z;}
   float r=length(p),k=1.5707963267948966/radius;
   vec3 ex=vec3(z*sin(orientation),-z*cos(orientation),0.),ey=vec3(-cos(orientation),-sin(orientation),0.),n=vec3(0.,0.,z);
   if(r<.001){dx=k*ex-k*k*p.x*n;dy=k*ey-k*k*p.y*n;return;}
   float a=r*k,s=sin(a),c=cos(a),slope=(a*c-s)/(r*r*r);
   vec3 tangent=ex*p.x+ey*p.y;
   dx=ex*(s/r)+tangent*(slope*p.x)-n*(k*s*p.x/r);
   dy=ey*(s/r)+tangent*(slope*p.y)-n*(k*s*p.y/r);return;
 }
 vec2 p=(pixel-resolution*.5)/focal;
 if(projectionKind==1){
   float q=dot(p,p)*.25;vec3 v=vec3(p,q-1.)/(1.+q);
   dx=(vec3(1.,0.,p.x*.5)-v*(p.x*.5))/((1.+q)*focal);
   dy=(vec3(0.,1.,p.y*.5)-v*(p.y*.5))/((1.+q)*focal);
 }else{
   vec3 raw=vec3(p,-1.),v=normalize(raw);
   dx=(vec3(1.,0.,0.)-v*v.x)/(focal*length(raw));
   dy=(vec3(0.,1.,0.)-v*v.y)/(focal*length(raw));
 }
 dx=cameraToWorld*dx;dy=cameraToWorld*dy;
}
`;

export const diffuseGLSL=morphologyGLSL+`
uniform bool hasDiffuse;
uniform sampler2D diffuseMap;
uniform mat3 equatorialToGalactic;
uniform bool hasDiffuseSources;
uniform sampler2D diffuseSourceData,diffuseSourceCells,diffuseSourceIndices;
uniform sampler2D diffuseMorphologyAtlas;
uniform vec2 diffuseMorphologySize;
uniform float diffuseMorphologyColumns;
uniform vec2 diffuseIndexSize;
uniform float diffuseSourceCount,deepMagnitudeLimit;
uniform bool softenGas;
uniform float gasDisplayGain;
float sourceErf(float x){float t=1./(1.+.3275911*x);return 1.-(((((1.061405429*t-1.453152027)*t)+1.421413741)*t-.284496736)*t+.254829592)*t*exp(-x*x);}
float filteredCloud(vec2 p,float cut,float tile){
 float r=length(p)/cut;if(r>=1.)return 0.;
 vec2 uv=(p/cut+1.)*${((FILTER_SIZE-1)/2).toFixed(1)},i=min(floor(uv),vec2(${(FILTER_SIZE-2).toFixed(1)})),f=uv-i;
 vec2 origin=vec2(mod(tile,diffuseMorphologyColumns),floor(tile/diffuseMorphologyColumns))*${FILTER_SIZE.toFixed(1)};
 vec2 a=(origin+i+.5)/diffuseMorphologySize,step=1./diffuseMorphologySize;
 // Explicit bilinear reconstruction of the same fixed angular field used
 // for CPU normalization. No float-linear extension or screen blur needed.
 float value=mix(mix(texture2D(diffuseMorphologyAtlas,a).r,texture2D(diffuseMorphologyAtlas,a+vec2(step.x,0.)).r,f.x),
  mix(texture2D(diffuseMorphologyAtlas,a+vec2(0.,step.y)).r,texture2D(diffuseMorphologyAtlas,a+step).r,f.x),f.y);
 return value*(1.-smoothstep(.99,1.,r));
}
float sourceProfile(vec3 v,vec4 source,vec4 params,vec4 shape,vec4 axis,vec4 detail,bool display){
 float c=dot(v,source.xyz);if(c<=0.)return 0.;
 vec3 crossV=cross(v,source.xyz);float q=dot(crossV,crossV)/(c*c*source.w*source.w),cut=params.z;
 vec2 p=vec2(0.);
 if(shape.y>.5){
  p=vec2(dot(v,axis.xyz),dot(v,cross(source.xyz,axis.xyz)))/(c*source.w);
  if(shape.y>16.)return filteredCloud(p,cut,detail.w);
  p.y/=shape.z;q=dot(p,p);
 }
 if(q>=cut*cut)return 0.;
 if(params.y<-.5)return sqrt(max(0.,1.-q));
 if(params.y<.5)return sqrt(cut*cut-q)*(3.+q+2.*cut*cut)/(2.*pow(1.+cut*cut,1.5)*pow(1.+q,2.));
 float value=params.y<1.5?exp(-q*.5):exp(-q)*sourceErf(sqrt(cut*cut-q));
 if(shape.y>.5)return cloudWeight(p,cut,value,shape,detail);
 if(params.y<1.5)return value;
 // The sharp physical ionization front remains in HDR. A display-only
 // contrast taper prevents its idealized chord profile reading as a solid
 // disk. This is not a new density field or a change to the cloud's radius.
 if(display && softenGas)value*=1.-smoothstep(.55,1.,sqrt(q)/cut);
 return value;
}
float sourceRadiance(vec3 g,vec3 dx,vec3 dy,vec4 source,vec4 params,vec4 shape,vec4 axis,vec4 detail,bool display){
 float xx=dot(dx,dx),xy=dot(dx,dy),yy=dot(dy,dy),det=max(1e-24,xx*yy-xy*xy),omega=sqrt(det);
 float scale=source.w*axis.w/sqrt(omega);
 vec3 delta=source.xyz-g;
 vec2 d=vec2(dot(delta,dx),dot(delta,dy));
 vec2 pixel=vec2(yy*d.x-xy*d.y,xx*d.y-xy*d.x)/det;
 // A subpixel source deposits its integrated flux in neighbouring pixels.
 // It never inherits the fixed angular size of a stored sky-map texel.
 float unresolved=params.w/omega*max(0.,1.-abs(pixel.x))*max(0.,1.-abs(pixel.y));
 if(scale<=.08)return unresolved;
 float resolved=0.;int samples=4;
 if(shape.y>.5)samples=scale<3.?16:8;
 for(int y=0;y<16;y++){
  if(y>=samples)break;
  for(int x=0;x<16;x++){
   if(x>=samples)break;
   vec2 p=(vec2(float(x),float(y))+.5)/float(samples)-.5;
   resolved+=sourceProfile(normalize(g+p.x*dx+p.y*dy),source,params,shape,axis,detail,display)/float(samples*samples);
  }
 }
 return mix(unresolved,params.x*resolved,smoothstep(.08,.6,scale));
}
vec3 diffuseAt(vec3 equatorial,vec3 pixelDx,vec3 pixelDy,bool display){
 if(!hasDiffuse || dot(equatorial,equatorial)<.1)return vec3(0.);
 vec3 g=normalize(equatorialToGalactic*equatorial);
 vec2 uv=vec2(fract(atan(g.y,g.x)/6.28318530718),asin(clamp(g.z,-1.,1.))/3.14159265359+.5);
 if(hasDiffuseSources){
   vec3 dx=equatorialToGalactic*pixelDx,dy=equatorialToGalactic*pixelDy;
   vec2 cell=floor(clamp(uv,vec2(0.),vec2(.999999))*vec2(128.,64.));
   vec2 range=texture2D(diffuseSourceCells,(cell+.5)/vec2(128.,64.)).rg;
   float light=0.;
   for(int j=0;j<64;j++){
     if(float(j)>=range.y)break;
     float index=range.x+float(j);
     float id=texture2D(diffuseSourceIndices,(vec2(mod(index,diffuseIndexSize.x),floor(index/diffuseIndexSize.x))+.5)/diffuseIndexSize).r;
     float row=(id+.5)/max(1.,diffuseSourceCount);
     vec4 source=texture2D(diffuseSourceData,vec2(.1,row)),params=texture2D(diffuseSourceData,vec2(.3,row));
     vec4 shape=texture2D(diffuseSourceData,vec2(.5,row));
     if(shape.x>deepMagnitudeLimit)continue;
     float angular=atan(source.w*params.z)+4.*max(length(dx),length(dy));
     if(dot(g,source.xyz)<cos(angular))continue;
     vec4 axis=texture2D(diffuseSourceData,vec2(.7,row)),detail=texture2D(diffuseSourceData,vec2(.9,row));
     float value=sourceRadiance(g,dx,dy,source,params,shape,axis,detail,display);
     if(display && softenGas && params.y>.5)value*=gasDisplayGain;
     light+=value;
   }
   return vec3(light);
 }
 float light=texture2D(diffuseMap,uv).r;
 return vec3(light); // V-band luminance, intentionally no photographic red/blue.
}
`;
export function diffuseUniforms(){
    const axes=[[0,0],[90,0],[0,90]].map(([l,b])=>galacticToEquatorial(l,b));
    return {hasDiffuse:{value:false},diffuseMap:{value:null},equatorialToGalactic:{value:new THREE.Matrix3().set(...axes.flat())},hasDiffuseSources:{value:false},diffuseSourceData:{value:null},diffuseSourceCells:{value:null},diffuseSourceIndices:{value:null},diffuseIndexSize:{value:new THREE.Vector2(1,1)},diffuseSourceCount:{value:0},diffuseMorphologyAtlas:{value:null},diffuseMorphologySize:{value:new THREE.Vector2(1,1)},diffuseMorphologyColumns:{value:1},deepMagnitudeLimit:{value:6.5},softenGas:{value:true},gasDisplayGain:{value:.25}};
}
export function attachDiffuseSources(texture,objects){
    if(!texture || !Array.isArray(objects))return texture;
    const data=prepareDeepSources(objects);
    const make=(values,w,h)=>{const t=new THREE.DataTexture(values,w,h,THREE.RGBAFormat,THREE.FloatType);t.minFilter=t.magFilter=THREE.NearestFilter;t.needsUpdate=true;return t;};
    const sources=make(data.sourceData,SOURCE_TEXELS,Math.max(1,data.sources.length)),cells=make(data.offsets,...SOURCE_GRID),indices=make(data.indexData,data.indexWidth,data.indexHeight);
    const a=data.filterAtlas,filtered=new THREE.DataTexture(a.values,a.width,a.height,THREE.RedFormat,THREE.FloatType);filtered.minFilter=filtered.magFilter=THREE.NearestFilter;filtered.needsUpdate=true;
    texture.userData.profiles={sources,cells,indices,filtered,filterSize:[a.width,a.height],filterColumns:a.columns,count:data.sources.length,indexSize:[data.indexWidth,data.indexHeight]};
    texture.addEventListener('dispose',()=>{sources.dispose();cells.dispose();indices.dispose();filtered.dispose();});return texture;
}
export function setDiffuseUniforms(u,texture){
    u.diffuseMap.value=texture;u.hasDiffuse.value=!!texture;const p=texture?.userData.profiles;
    u.hasDiffuseSources.value=!!p;u.diffuseSourceData.value=p?.sources||null;u.diffuseSourceCells.value=p?.cells||null;u.diffuseSourceIndices.value=p?.indices||null;u.diffuseSourceCount.value=p?.count||0;u.diffuseIndexSize.value.set(...(p?.indexSize||[1,1]));
    u.diffuseMorphologyAtlas.value=p?.filtered||null;u.diffuseMorphologySize.value.set(...(p?.filterSize||[1,1]));u.diffuseMorphologyColumns.value=p?.filterColumns||1;
}
export async function loadDiffuseTexture(info,objects){
    if(!info)return null;
    const r=await fetch(info.url);if(!r.ok)throw Error(`深空光图加载失败（${r.status}）`);
    const bytes=await r.arrayBuffer();if(bytes.byteLength!==info.width*info.height*4)throw Error('深空光图尺寸不匹配');
    if(info.sha256){const digest=await crypto.subtle.digest('SHA-256',bytes),hash=Array.from(new Uint8Array(digest),b=>b.toString(16).padStart(2,'0')).join('');if(hash!==info.sha256)throw Error('深空光图校验失败');}
    const texture=new THREE.DataTexture(new Float32Array(bytes),info.width,info.height,THREE.RedFormat,THREE.FloatType);
    texture.minFilter=texture.magFilter=THREE.LinearFilter;texture.wrapS=THREE.RepeatWrapping;texture.wrapT=THREE.ClampToEdgeWrapping;texture.needsUpdate=true;
    return attachDiffuseSources(texture,objects);
}

// Both versions share the necessary integrated-flux filter and angular
// profiles. V1 compresses surface brightness; V2 adds the radiance to HDR.
const vertex='void main(){gl_Position=vec4(position.xy,0.,1.);}';
const fragment=diffuseGLSL+`
uniform vec2 resolution;
uniform mat3 cameraToWorld,viewToEquatorial;
uniform float focal,orientation,night;
uniform int projectionKind;
uniform vec3 atlasNorth,atlasSouth;
uniform bool surface;
`+diffuseFootprintGLSL+`
void main(){
 vec3 v;
 if(projectionKind==2){
   vec2 p=gl_FragCoord.xy-atlasNorth.xy;float z=1.,r=atlasNorth.z;
   if(length(p)>r){p=gl_FragCoord.xy-atlasSouth.xy;z=-1.;r=atlasSouth.z;}
   if(length(p)>r)discard;
   float a=length(p)/r*1.57079632679,phi=atan(-z*p.x,-p.y)+orientation;
   v=vec3(sin(a)*cos(phi),sin(a)*sin(phi),z*cos(a));
 }else{
   vec2 p=(gl_FragCoord.xy-resolution*.5)/focal;
   if(projectionKind==1){p*=.5;v=vec3(2.*p,dot(p,p)-1.)/(1.+dot(p,p));}else v=normalize(vec3(p,-1.));
   v=cameraToWorld*v;
 }
 if(surface && v.z<0.)discard;
 vec3 dx,dy;diffuseFootprint(gl_FragCoord.xy,dx,dy);
 float l=diffuseAt(viewToEquatorial*v,viewToEquatorial*dx,viewToEquatorial*dy,true).r*night*night;
 float response=.35*l/(.01+l);
 gl_FragColor=vec4(vec3(response),1.);
 #include <colorspace_fragment>
}`;

export class DiffuseLayer {
    constructor(scene){
        this.uniforms={...diffuseUniforms(),resolution:{value:new THREE.Vector2()},cameraToWorld:{value:new THREE.Matrix3()},viewToEquatorial:{value:new THREE.Matrix3()},focal:{value:1},orientation:{value:0},night:{value:1},projectionKind:{value:0},atlasNorth:{value:new THREE.Vector3()},atlasSouth:{value:new THREE.Vector3()},surface:{value:false}};
        const material=new THREE.ShaderMaterial({vertexShader:vertex,fragmentShader:fragment,uniforms:this.uniforms,depthTest:false,depthWrite:false,transparent:true,blending:THREE.AdditiveBlending,toneMapped:false});
        this.mesh=new THREE.Mesh(new THREE.PlaneGeometry(2,2),material);this.mesh.frustumCulled=false;this.mesh.renderOrder=-1;scene.add(this.mesh);
    }
    setTexture(texture){setDiffuseUniforms(this.uniforms,texture);}
    update(sky,enabled,camera,width,height,focal){
        const u=this.uniforms;this.mesh.visible=enabled && u.hasDiffuse.value;u.surface.value=sky.frame.surface;u.night.value=sky.night;
        u.viewToEquatorial.value.set(...sky.frame.matrix.flat()).transpose();u.resolution.value.set(width,height);u.focal.value=focal;
        if(camera){camera.updateMatrixWorld();u.cameraToWorld.value.setFromMatrix4(camera.matrixWorld);u.projectionKind.value=camera.userData.skyProjection==='stereographic'?1:0;}
    }
}
export class DiffuseAtlasPainter {
    constructor(ctx){
        this.ctx=ctx;this.renderer=new THREE.WebGLRenderer({alpha:true,antialias:false});this.renderer.outputColorSpace=THREE.SRGBColorSpace;
        this.scene=new THREE.Scene();this.camera=new THREE.Camera();this.layer=new DiffuseLayer(this.scene);this.renderer.setClearColor(0,0);
    }
    draw(sky,texture,layout,orientation,{width,height,dpr,zoom,pan,enabled}){
        if(!texture || !enabled)return;
        const w=Math.round(width*dpr),h=Math.round(height*dpr),r=this.renderer;if(r.domElement.width!==w || r.domElement.height!==h)r.setSize(w,h,false);
        this.layer.setTexture(texture);this.layer.update(sky,true,null,w,h,1);
        const u=this.layer.uniforms;u.projectionKind.value=2;u.orientation.value=orientation*Math.PI/180;
        for(const [key,uniform] of [['north','atlasNorth'],['south','atlasSouth']]){
            const c=layout[key],x=width/2+(c[0]-width/2)*zoom+pan[0],y=height/2+(c[1]-height/2)*zoom+pan[1];
            u[uniform].value.set(x*dpr,(height-y)*dpr,layout.radius*zoom*dpr);
        }
        r.render(this.scene,this.camera);this.ctx.save();this.ctx.setTransform(1,0,0,1,0,0);this.ctx.globalCompositeOperation='lighter';this.ctx.drawImage(r.domElement,0,0);this.ctx.restore();
    }
}
