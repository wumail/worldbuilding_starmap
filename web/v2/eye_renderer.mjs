import {diffuseGLSL,diffuseFootprintGLSL,diffuseUniforms,setDiffuseUniforms} from '../shared/deep_sky_view.js';
import {SKY_OVERLAYS} from '../shared/sky_overlays.mjs';
import * as THREE from 'three';
import {POINT_STYLE,SYMBOL_REFERENCE,DEFAULT_DISPLAY_SCALE,displaySky as chartSky,pointRadius,pointOpacity,unresolvedPointWeight,atlasBodyCircles} from '../shared/sky_render.mjs';
import {pointKernelAtlas} from '../shared/sky_point_kernel.mjs';
import {DEG,dot,backgroundDirection,projectHemisphere} from '../shared/solar_system.mjs';
import {diskBasis,diskLight,bodyLightSample,bodyFullyOcculted} from './sky_render.mjs';
import {ARC_MIN,observedMagnitude,magnitudeToLux,diskPeakLuminance,pointHidden,nakedEyeLimit,skyLuminance,gaussianKernel,displayKey,luxToMagnitude} from './eye_model.mjs';
import {projectSkyPosition} from '../shared/sky_projection.js';

const quadVertex=`uniform vec4 bounds;void main(){gl_Position=vec4(mix(bounds.xy,bounds.zw,position.xy*.5+.5),0.,1.);}`;
const ATLAS_OCCLUDER_LIMIT=32; // Up to two clipped circles for each system body.
// Display response is evaluated BEFORE silhouette coverage is resolved. The
// physical HDR target remains separate for photometric diagnostics.
const displayGLSL=`
uniform bool displayPass,white;
uniform float adaptation,exposureEV,displayKey;
float response(float l){float x=displayKey*exp2(exposureEV)*l/max(.005,adaptation);return x/(1.+x);}
vec3 displaySource(vec3 radiance){
 float l=dot(radiance,vec3(.2126,.7152,.0722));
 float chroma=white?0.:smoothstep(-2.,1.,log(max(l,1e-9))/log(10.));
 return clamp(mix(vec3(1.),radiance/max(l,1e-20),chroma)*response(l),0.,1.);
}
`;
const rayGLSL=displayGLSL+`
uniform vec2 resolution;
uniform float focal;
uniform int projectionKind;
uniform mat3 cameraToWorld;
uniform mat3 toEquatorial;
uniform vec3 atlasNorth,atlasSouth;
uniform vec4 atlasOccluders[${ATLAS_OCCLUDER_LIMIT}];
uniform int atlasOccluderCount;
uniform bool atlasMap;
uniform float orientation;
uniform bool surface,atmosphere;
uniform float nightL,dayL,extinction;
uniform vec3 moonDirection[2];
uniform float moonStrength[2];
const float PI=3.141592653589793;
bool behindAtlasLimb(vec2 pixel,float distance){
    if(!atlasMap)return false;
    bool north=length(pixel-atlasNorth.xy)<=atlasNorth.z;
    for(int i=0;i<${ATLAS_OCCLUDER_LIMIT};i++){
        if(i>=atlasOccluderCount)break;
        vec4 circle=atlasOccluders[i];
        if(north==(circle.z>0.) && distance>circle.w && length(pixel-circle.xy)<abs(circle.z))return true;
    }
    return false;
}
vec3 rayAt(vec2 pixel) {
    if(projectionKind==2){
        vec2 d=pixel-atlasNorth.xy;float signZ=1.,r=atlasNorth.z;
        if(length(d)>r){d=pixel-atlasSouth.xy;signZ=-1.;r=atlasSouth.z;}
        if(length(d)>r)return vec3(0.);
        float theta=length(d)/r*PI/2.;
        float phi=atan(-signZ*d.x,-d.y)+orientation;
        return vec3(sin(theta)*cos(phi),sin(theta)*sin(phi),signZ*cos(theta));
    }
    vec2 p=(pixel-resolution*.5)/focal;
    vec3 v;
    if(projectionKind==1){p*=.5;v=vec3(2.*p,dot(p,p)-1.)/(1.+dot(p,p));}
    else v=normalize(vec3(p,-1.));
    return cameraToWorld*v;
}
float massAt(vec3 v){float h=degrees(asin(clamp(v.z,0.,1.)));return 1./(sin(radians(h))+.50572*pow(h+6.07995,-1.6364));}
vec3 unitY(vec3 color){return color/dot(color,vec3(.2126,.7152,.0722));}
vec3 skyRadiance(vec3 v){
    if(dot(v,v)<.1)return vec3(0.);
    if(surface && v.z<0.)return vec3(.19,.21,.22)*(nightL*.06+dayL*.003);
    if(!atmosphere)return unitY(vec3(.68,.80,1.))*nightL;
    float a=massAt(v),scatter=0.;
    for(int i=0;i<2;i++){
        float rho=clamp(degrees(acos(clamp(dot(v,moonDirection[i]),-1.,1.))),10.,180.);
        float f=pow(10.,5.36)*(1.06+pow(cos(radians(rho)),2.))+pow(10.,6.15-rho/40.);
        scatter+=f*moonStrength[i]*(1.-pow(10.,-.4*extinction*a));
    }
    return unitY(vec3(.68,.8,1.))*(nightL*(.6+.4*sqrt(a))+scatter)+unitY(vec3(.36,.65,1.24))*dayL*(.65+.35*sqrt(a));
}
// A dark display baseline, independent of photometric sky luminance used for
// extinction and visibility. Moonlight lifts it modestly; daylight stays blue.
vec3 displaySky(vec3 v){
 if(dot(v,v)<.1)return vec3(.00091,.00182,.00402); // #03060d atlas surround
 if(surface && v.z<0.)return vec3(.00304,.00439,.00699); // #0a0e14
 float physical=dot(skyRadiance(v),vec3(.2126,.7152,.0722));
 float moonLift=clamp(log(1.+max(0.,physical-nightL)/.002)/log(10.),0.,1.);
 vec3 night=mix(vec3(.001518,.002732,.005605),vec3(.003677,.006049,.01096),moonLift);
 float day=clamp(log(1.+dayL/.04)/log(1.+5000./.04),0.,1.);
 vec3 color=mix(night,vec3(.085,.25,.52),day*day);
 return clamp(color*exp2(exposureEV),0.,1.);
}
`;
const backgroundFragment=rayGLSL+diffuseGLSL+diffuseFootprintGLSL+`
uniform bool deepSky;
void main(){
    vec3 v=rayAt(gl_FragCoord.xy),color=displayPass?displaySky(v):skyRadiance(v);
    if(deepSky && dot(v,v)>.1 && (!surface || v.z>=0.)){
        vec3 dx,dy;diffuseFootprint(gl_FragCoord.xy,dx,dy);
        vec3 radiance=diffuseAt(toEquatorial*v,toEquatorial*dx,toEquatorial*dy,displayPass);
        if(atmosphere)radiance*=pow(10.,-.4*extinction*massAt(v));
        color+=displayPass?displaySource(radiance):radiance;
    }
    gl_FragColor=vec4(color,1.);
}`;
const bodyFragment=rayGLSL+`
uniform vec3 bodyDirection,axisX,axisY,light,tint;
uniform float angularRadius,peak,sourceWeight;
uniform bool luminous,roundAtlasBody;
uniform vec3 bodyNorth,bodySouth,lightNorth,lightSouth;
vec3 bodyCircleAt(vec2 pixel){return length(pixel-atlasNorth.xy)<=atlasNorth.z?bodyNorth:bodySouth;}
vec4 sampleBody(vec2 pixel){
    vec3 v=rayAt(pixel);
    if(dot(v,v)<.1 || (surface && v.z<0.))return vec4(0.);
    float c=dot(v,bodyDirection);
    vec2 uv;
    vec3 illumination=light;
    if(roundAtlasBody){
        vec3 circle=bodyCircleAt(pixel);if(circle.z<=0.)return vec4(0.);
        uv=(pixel-circle.xy)/circle.z;if(dot(uv,uv)>1.)return vec4(0.);
        illumination=length(pixel-atlasNorth.xy)<=atlasNorth.z?lightNorth:lightSouth;
    }else{
        if(length(v-bodyDirection)>2.*sin(angularRadius*.5))return vec4(0.);
        uv=vec2(dot(v,axisX),dot(v,axisY))/(c*tan(angularRadius));
    }
    float shade=luminous?1.:max(0.,dot(vec3(uv,sqrt(max(0.,1.-dot(uv,uv)))),illumination));
    // Compress the surface peak once, retaining Lambert contrast across a
    // resolved disk instead of saturating every sunlit sample to white.
    return vec4(displayPass?min(vec3(1.),displaySky(v)+sourceWeight*displaySource(tint*peak)*shade):(atmosphere?skyRadiance(v):vec3(0.))+tint*peak*shade,1.);
}
void main(){
    vec3 v=rayAt(gl_FragCoord.xy);
    float angle=2.*asin(clamp(length(v-bodyDirection)*.5,0.,1.));
    vec3 circle=bodyCircleAt(gl_FragCoord.xy);
    bool interior=roundAtlasBody?length(gl_FragCoord.xy-circle.xy)<circle.z-2.5:angle<angularRadius-2.5/focal;
    if(dot(v,v)>.1 && interior && (!surface || v.z>2.5/focal)){
        gl_FragColor=sampleBody(gl_FragCoord.xy);return;
    }
    vec4 sum=vec4(0.);
    for(int y=0;y<12;y++)for(int x=0;x<12;x++)sum+=sampleBody(gl_FragCoord.xy+(vec2(float(x),float(y))+.5)/12.-.5);
    if(sum.a==0.)discard;
    // Integrate linear radiance and coverage separately. Alpha only encodes
    // subpixel silhouette coverage; darkness never makes the disk transparent.
    gl_FragColor=vec4(sum.rgb/sum.a,sum.a/144.);
}`;
const pointVertex=displayGLSL+`
attribute vec2 center;attribute vec3 radiance;attribute vec4 sourceAxes;attribute vec4 sourceWeights;attribute vec2 chartSymbol;attribute vec3 chartTint;
uniform vec2 resolution;uniform float pointSigma,pointWingSigma,pointWingWeight,pointGain,pointSymbolMix,rasterScale;uniform bool mapSymbols;
varying vec2 sourceCenter;varying vec3 sourceRadiance;varying vec4 axes;varying vec4 drawWeights;varying vec2 symbol;varying vec3 symbolTint;
void main(){
 sourceCenter=center*rasterScale;sourceRadiance=radiance*rasterScale*rasterScale;axes=sourceAxes;drawWeights=sourceWeights;symbol=vec2(chartSymbol.x*rasterScale,chartSymbol.y);symbolTint=chartTint;
 float determinant=abs(sourceAxes.x*sourceAxes.w-sourceAxes.y*sourceAxes.z);
 float e=dot(sourceRadiance,vec3(.2126,.7152,.0722))*pointGain*pointGain/max(1e-8,determinant);
 // Adapt the support to intensity; its tail is below an 8-bit display step.
 float factor=displayKey*exp2(exposureEV)/max(.005,adaptation);
 float extent=2.;
 if(displayPass){
  float a=e*factor/(6.2831853*pointSigma*pointSigma);
  float b=e*factor*pointWingWeight/(6.2831853*pointWingSigma*pointWingSigma);
  extent=1.+max(length(sourceAxes.xy),length(sourceAxes.zw))*max(pointSigma*sqrt(2.*max(0.,log(max(1.,a*8192.)))),pointWingSigma*sqrt(2.*max(0.,log(max(1.,b*8192.)))));
 }
 float symbolExtent=symbol.x<2.?2.5:symbol.x+2.;
 if(displayPass && (mapSymbols || pointSymbolMix>=1.))extent=symbolExtent;
 else if(displayPass && pointSymbolMix>0.)extent=max(extent,symbolExtent);
 gl_Position=vec4((sourceCenter+position.xy*extent)/resolution*2.-1.,0.,1.);
}
`;
const pointFragment=rayGLSL+`
varying vec2 sourceCenter;varying vec3 sourceRadiance;varying vec4 axes;varying vec4 drawWeights;varying vec2 symbol;varying vec3 symbolTint;
uniform float pointSigma,pointWingSigma,pointWingWeight,pointGain,pointSymbolMix;uniform bool mapSymbols;uniform sampler2D kernelAtlas;
vec3 chartSRGB(vec3 c){return mix(1.055*pow(max(vec3(0.),c),vec3(1./2.4))-.055,c*12.92,step(c,vec3(.0031308)));}
vec3 chartLinear(vec3 c){return mix(pow((c+.055)/1.055,vec3(2.4)),c/12.92,step(c,vec3(.04045)));}
float chartPixel(vec2 p,float radius){
 float coverage=0.;for(int y=0;y<16;y++)for(int x=0;x<16;x++){
  vec2 o=(vec2(float(x),float(y))+.5)/16.-.5;
  float profile=1.-smoothstep(${POINT_STYLE.coreFraction.toFixed(8)},1.,length(p+o)/radius);
  float energy=${POINT_STYLE.exposure.toFixed(8)}*profile;coverage+=energy/(1.+energy);
 }return floor(coverage/256.*255.+.5)/255.;
}
float chartAlpha(vec2 p,float radius){
 if(radius<2.){float index=ceil(radius*128.);vec2 origin=vec2(mod(index,16.),floor(index/16.))*7.+1.;return texture2D(kernelAtlas,(origin+2.5+p)/vec2(112.,119.)).a*pow(radius/(index/128.),2.);}
 float r=ceil(radius*128.)/128.;vec2 q=floor(p),f=fract(p);
 return mix(mix(chartPixel(q,r),chartPixel(q+vec2(1.,0.),r),f.x),mix(chartPixel(q+vec2(0.,1.),r),chartPixel(q+1.,r),f.x),f.y)*pow(radius/r,2.);
}
float gaussian(vec2 d,float s){return exp(-dot(d,d)/(2.*s*s))/(6.2831853*s*s);}
void main(){
 vec2 d=gl_FragCoord.xy-sourceCenter;
 if(!displayPass){if(drawWeights.x==0.)discard;vec2 a=abs(d);float w=max(0.,1.-a.x)*max(0.,1.-a.y);if(w==0.)discard;gl_FragColor=vec4(sourceRadiance*w,1.);return;}
 // Tone each subpixel before resolving coverage, including unresolved Sol.
 // A circular analytic PSF has no rectangular, intensity-visible cutoff.
 if(drawWeights.y==0.)discard;
 if(drawWeights.z!=0.){vec3 circle=drawWeights.z>0.?atlasNorth:atlasSouth;if(length(gl_FragCoord.xy-circle.xy)>circle.z)discard;}
 vec3 sharp=vec3(0.);
 if(mapSymbols || pointSymbolMix>0.){vec3 base=displaySky(rayAt(gl_FragCoord.xy));float coverage=0.;
  for(int y=0;y<4;y++)for(int x=0;x<4;x++)if(!behindAtlasLimb(gl_FragCoord.xy+(vec2(float(x),float(y))+.5)/4.-.5,drawWeights.w))coverage+=1./16.;
  float alpha=chartAlpha(d,symbol.x)*symbol.y*drawWeights.y*coverage;
  sharp=max(vec3(0.),chartLinear(mix(chartSRGB(base),chartSRGB(symbolTint),alpha))-base);
  if(mapSymbols || pointSymbolMix>=1.){gl_FragColor=vec4(sharp,1.);return;}}
 float det=axes.x*axes.w-axes.y*axes.z;
 mat2 inverseAxes=mat2(axes.w,-axes.z,-axes.y,axes.x)/det;
 vec3 sum=vec3(0.);
 for(int y=0;y<4;y++)for(int x=0;x<4;x++){
  vec2 q=d+(vec2(float(x),float(y))+.5)/4.-.5;
  q=inverseAxes*q;float k=mix(gaussian(q,pointSigma),gaussian(q,pointWingSigma),pointWingWeight)/abs(det);
  sum+=displaySource(sourceRadiance*k*pointGain*pointGain);
 }
 gl_FragColor=vec4(mix(sum/16.*drawWeights.y,sharp,pointSymbolMix),1.);
}
`;
const blurFragment=`
uniform sampler2D inputImage;uniform vec2 passResolution,axis;uniform float weights[25];uniform int radius;
void main(){vec2 uv=gl_FragCoord.xy/passResolution;vec3 sum=texture2D(inputImage,uv).rgb*weights[0];
for(int i=1;i<=24;i++){if(i>radius)break;vec2 d=axis*float(i)/passResolution;sum+=(texture2D(inputImage,uv+d).rgb+texture2D(inputImage,uv-d).rgb)*weights[i];}
gl_FragColor=vec4(sum,1.);}
`;
const downFragment=`uniform sampler2D inputImage;uniform vec2 passResolution,sourceResolution;
void main(){
vec2 start=floor(gl_FragCoord.xy)*sourceResolution/passResolution,end=(floor(gl_FragCoord.xy)+1.)*sourceResolution/passResolution;
vec2 base=floor(start);vec4 sum=vec4(0.);
for(int y=0;y<3;y++)for(int x=0;x<3;x++){
    vec2 p=base+vec2(float(x),float(y)),area=max(vec2(0.),min(end,p+1.)-max(start,p));
    sum+=texture2D(inputImage,(p+.5)/sourceResolution)*area.x*area.y;
}
gl_FragColor=sum/((end.x-start.x)*(end.y-start.y));}`;
const toneFragment=`
uniform sampler2D coreImage,wingImage;uniform vec2 resolution,coreSize,wingSize;uniform float wingWeight;
vec3 linearSample(sampler2D tex,vec2 uv,vec2 size){
vec2 p=uv*size-.5,f=fract(p),base=(floor(p)+.5)/size;
return mix(mix(texture2D(tex,base).rgb,texture2D(tex,base+vec2(1./size.x,0.)).rgb,f.x),
mix(texture2D(tex,base+vec2(0.,1./size.y)).rgb,texture2D(tex,base+1./size).rgb,f.x),f.y);}
void main(){vec2 uv=gl_FragCoord.xy/resolution;
gl_FragColor=vec4(mix(linearSample(coreImage,uv,coreSize),linearSample(wingImage,uv,wingSize),wingWeight),1.);
}`;
const outputFragment=rayGLSL+`
uniform sampler2D inputImage;
void main(){
 vec3 color=texture2D(inputImage,gl_FragCoord.xy/resolution).rgb;
 if(projectionKind==2){
  float n=length(gl_FragCoord.xy-atlasNorth.xy)-atlasNorth.z;
  float s=length(gl_FragCoord.xy-atlasSouth.xy)-atlasSouth.z;
  float edge=min(n,s),aa=max(1.,resolution.x/1600.);
  vec3 border=n<s?vec3(.0999,.1356,.1845):vec3(.03434,.05613,.0865);
  color=mix(color,vec3(.00091,.00182,.00402),smoothstep(-aa,0.,edge));
  color=mix(color,border,(1.-smoothstep(0.,aa,abs(edge)))*${SKY_OVERLAYS.border});
 }
 gl_FragColor=vec4(clamp(color,0.,1.),1.);
 #include <colorspace_fragment>
}`;

const colorCache=new Map();
function luminanceTint(hex) {
    if(!colorCache.has(hex)){
        const c=new THREE.Color(hex),y=.2126*c.r+.7152*c.g+.0722*c.b;
        const tint=new THREE.Vector3(c.r/y,c.g/y,c.b/y);tint.displayRGB=[c.r,c.g,c.b];colorCache.set(hex,tint);
    }
    return colorCache.get(hex);
}
function uniforms() {
    return {...diffuseUniforms(),deepSky:{value:true},displayPass:{value:false},white:{value:false},adaptation:{value:.005},exposureEV:{value:0},displayKey:{value:.002},bounds:{value:new THREE.Vector4(-1,-1,1,1)},resolution:{value:new THREE.Vector2(1,1)},focal:{value:1},projectionKind:{value:0},
        cameraToWorld:{value:new THREE.Matrix3()},toEquatorial:{value:new THREE.Matrix3()},atlasNorth:{value:new THREE.Vector3()},atlasSouth:{value:new THREE.Vector3()},orientation:{value:Math.PI},
        atlasMap:{value:false},atlasOccluderCount:{value:0},atlasOccluders:{value:Array.from({length:ATLAS_OCCLUDER_LIMIT},()=>new THREE.Vector4())},
        surface:{value:true},atmosphere:{value:true},nightL:{value:.0002},dayL:{value:0},extinction:{value:.2},
        moonDirection:{value:[new THREE.Vector3(0,0,1),new THREE.Vector3(0,0,1)]},moonStrength:{value:[0,0]}};
}
function material(vertexShader,fragmentShader,u,options={}) {
    return new THREE.ShaderMaterial({vertexShader,fragmentShader,uniforms:u,depthTest:false,depthWrite:false,toneMapped:false,...options});
}

export class EyeSkyRenderer {
    constructor(canvas) {
        this.renderer=new THREE.WebGLRenderer({canvas,antialias:false,preserveDrawingBuffer:true,powerPreference:'high-performance'});
        this.renderer.debug.onShaderError=(gl,program,vs,fs)=>{throw Error(gl.getShaderInfoLog(vs)+' / '+gl.getShaderInfoLog(fs));};
        if(!this.renderer.capabilities.isWebGL2 || !this.renderer.extensions.has('EXT_color_buffer_float'))throw Error('V2 需要浏览器支持 WebGL 2 浮点绘图。原版仍可使用。');
        this.linearFiltering=this.renderer.extensions.has('OES_texture_float_linear');
        this.renderer.outputColorSpace=THREE.SRGBColorSpace;this.renderer.autoClear=false;
        this.u=uniforms();
        // V2 already uses its much lower, adaptation-based display response.
        // Keep that exposure; only soften the ideal gas front in display.
        this.u.gasDisplayGain.value=1;
        this.quadGeometry=new THREE.PlaneGeometry(2,2);this.scene=new THREE.Scene();this.passCamera=new THREE.Camera();
        this.quad=new THREE.Mesh(this.quadGeometry);this.quad.frustumCulled=false;this.scene.add(this.quad);
        this.background=material(quadVertex,backgroundFragment,this.u);
        this.body=material(quadVertex,bodyFragment,{...this.u,bodyDirection:{value:new THREE.Vector3()},axisX:{value:new THREE.Vector3()},axisY:{value:new THREE.Vector3()},light:{value:new THREE.Vector3()},tint:{value:new THREE.Vector3()},angularRadius:{value:0},peak:{value:0},sourceWeight:{value:1},luminous:{value:false},roundAtlasBody:{value:false},bodyNorth:{value:new THREE.Vector3(0,0,-1)},bodySouth:{value:new THREE.Vector3(0,0,-1)},lightNorth:{value:new THREE.Vector3()},lightSouth:{value:new THREE.Vector3()}},{transparent:true});
        this.blur=material(quadVertex,blurFragment,{...this.u,passResolution:{value:new THREE.Vector2()},inputImage:{value:null},axis:{value:new THREE.Vector2()},weights:{value:new Float32Array(25)},radius:{value:0}});
        this.down=material(quadVertex,downFragment,{...this.u,passResolution:{value:new THREE.Vector2()},sourceResolution:{value:new THREE.Vector2()},inputImage:{value:null}});
        this.tone=material(quadVertex,toneFragment,{...this.u,coreImage:{value:null},wingImage:{value:null},coreSize:{value:new THREE.Vector2()},wingSize:{value:new THREE.Vector2()},wingWeight:{value:0}});
        this.output=material(quadVertex,outputFragment,{...this.u,inputImage:{value:null}});
        this.pointScene=new THREE.Scene();this.pointGeometry=new THREE.InstancedBufferGeometry();
        this.pointGeometry.setAttribute('position',this.quadGeometry.attributes.position.clone());this.pointGeometry.setIndex(this.quadGeometry.index.clone());
        const chartImage=pointKernelAtlas();this.chartTexture=new THREE.DataTexture(chartImage.data,chartImage.width,chartImage.height);this.chartTexture.minFilter=this.chartTexture.magFilter=THREE.LinearFilter;this.chartTexture.needsUpdate=true;
        this.pointMaterial=material(pointVertex,pointFragment,{...this.u,mapSymbols:{value:false},kernelAtlas:{value:this.chartTexture},pointSigma:{value:.35},pointWingSigma:{value:.35},pointWingWeight:{value:0},pointGain:{value:1},pointSymbolMix:{value:0},rasterScale:{value:1}},{transparent:true,blending:THREE.AdditiveBlending});
        this.pointMesh=new THREE.Mesh(this.pointGeometry,this.pointMaterial);this.pointMesh.frustumCulled=false;this.pointScene.add(this.pointMesh);
        this.capacity=0;this.count=0;this.targets=[];this.pyramid=[];this.presentation=null;this.width=1;this.height=1;this.dpr=1;this.outputDpr=1;this.resourceGeneration=0;
    }
    setDiffuse(texture){setDiffuseUniforms(this.u,texture);}
    resize(width,height,dpr=1) {
        width=Math.max(1,Math.round(width));height=Math.max(1,Math.round(height));
        // Keep the expensive physical/disk buffers bounded. Render sharp
        // points AFTER upsampling into a native-resolution display target;
        // CSS must never stretch the point sprites with the background image.
        const outputDpr=Math.min(dpr,2,Math.sqrt(8800000/(width*height)));
        dpr=Math.min(outputDpr,Math.sqrt(2200000/(width*height)));
        const w=Math.max(1,Math.round(width*dpr)),h=Math.max(1,Math.round(height*dpr));
        const ow=Math.max(1,Math.round(width*outputDpr)),oh=Math.max(1,Math.round(height*outputDpr));
        // CSS size / DPR may change while backing dimensions stay identical.
        this.width=width;this.height=height;this.dpr=w/width;this.outputDpr=ow/width;
        if(w===this.targets[0]?.width && h===this.targets[0]?.height && ow===this.renderer.domElement.width && oh===this.renderer.domElement.height)return;
        this.renderer.setPixelRatio(1);this.renderer.setSize(ow,oh,false);this.u.resolution.value.set(w,h);
        for(const t of this.targets)t.dispose();
        for(const level of this.pyramid)for(const t of level)t.dispose();this.pyramid=[];
        const filter=this.linearFiltering?THREE.LinearFilter:THREE.NearestFilter;
        this.targets=Array.from({length:4},()=>new THREE.WebGLRenderTarget(w,h,{type:THREE.FloatType,minFilter:filter,magFilter:filter,depthBuffer:false,stencilBuffer:false,colorSpace:THREE.LinearSRGBColorSpace}));
        this.presentation?.dispose();
        this.presentation=w===ow && h===oh?null:new THREE.WebGLRenderTarget(ow,oh,{type:THREE.HalfFloatType,minFilter:THREE.LinearFilter,magFilter:THREE.LinearFilter,depthBuffer:false,stencilBuffer:false,colorSpace:THREE.LinearSRGBColorSpace});
        this.resourceGeneration++;
    }
    configure(camera,sky,{focal,overview=false,atlasTransform={zoom:1,panX:0,panY:0}}={}) {
        this.camera=camera;this.sky=sky;this.focal=focal;this.overview=overview;camera.updateMatrixWorld();
        const u=this.u,env=sky.environment;
        u.focal.value=focal*this.dpr;u.projectionKind.value=overview?2:camera.userData.skyProjection==='stereographic'?1:0;
        u.cameraToWorld.value.setFromMatrix4(camera.matrixWorld);u.toEquatorial.value.set(...sky.frame.matrix.flat()).transpose();
        u.surface.value=sky.frame.surface;u.atmosphere.value=env.atmosphere;u.nightL.value=env.nightL;u.dayL.value=env.dayL;u.extinction.value=env.extinction;
        for(let i=0;i<2;i++){u.moonDirection.value[i].fromArray(env.moons[i]?.direction || [0,0,1]);u.moonStrength.value[i]=env.moons[i]?.strength || 0;}
        const horizontal=Math.min((this.width-90)/4,(this.height-105)/2),vertical=Math.min((this.width-55)/2,(this.height-130)/4);
        const baseRadius=Math.max(10,Math.max(horizontal,vertical)),z=atlasTransform.zoom;
        const centers=horizontal>=vertical?[[this.width/2-baseRadius-18,this.height/2+5],[this.width/2+baseRadius+18,this.height/2+5]]:
            [[this.width/2,this.height/2-baseRadius-20],[this.width/2,this.height/2+baseRadius+20]];
        const [north,south]=centers.map(c=>[(c[0]-this.width/2)*z+this.width/2+atlasTransform.panX,(c[1]-this.height/2)*z+this.height/2+atlasTransform.panY]);
        const radius=baseRadius*z;this.layout={radius,north,south};
        u.atlasNorth.value.set(north[0]*this.dpr,(this.height-north[1])*this.dpr,radius*this.dpr);
        u.atlasSouth.value.set(south[0]*this.dpr,(this.height-south[1])*this.dpr,radius*this.dpr);
        this.orientation=sky.frame.surface?180:13.564125;u.orientation.value=this.orientation*DEG;
        if(overview){this.focal=radius/(Math.PI/2);u.focal.value=this.focal*this.dpr;}
    }
    project(direction) {
        if(this.overview){const p=projectHemisphere(direction,this.layout.radius,this.layout,this.orientation);return {...p,visible:true};}
        const p=projectSkyPosition(new THREE.Vector3(...direction),this.camera);
        return {x:(p.x+1)*this.width/2,y:(1-p.y)*this.height/2,visible:p.visible};
    }
    pixelsPerSteradian(direction) {
        if(this.overview){const t=Math.acos(Math.abs(direction[2]));return (this.focal*this.dpr)**2*(t<1e-8?1:t/Math.sin(t));}
        const v=new THREE.Vector3(...direction).applyMatrix3(new THREE.Matrix3().setFromMatrix4(this.camera.matrixWorldInverse));
        const c=-v.z/v.length();
        return (this.focal*this.dpr)**2*(this.u.projectionKind.value===1?(2/Math.max(1e-8,1+c))**2:1/Math.max(1e-8,c)**3);
    }
    ensurePoints(count) {
        if(count<=this.capacity)return;
        this.capacity=Math.max(count,Math.ceil(this.capacity*1.5),1024);
        // Three caches the instance limit on a geometry after its first draw.
        // The loading frame may contain just a few planets; replacing only
        // attributes later would leave the full star catalog capped at 1024.
        this.pointGeometry.dispose();this.pointGeometry=new THREE.InstancedBufferGeometry();
        // Own the quad buffers: disposing the old instanced geometry must
        // not invalidate the fullscreen pass geometry's cached vertex arrays.
        this.pointGeometry.setAttribute('position',this.quadGeometry.attributes.position.clone());this.pointGeometry.setIndex(this.quadGeometry.index.clone());
        this.pointMesh.geometry=this.pointGeometry;
        this.centers=new Float32Array(this.capacity*2);this.energies=new Float32Array(this.capacity*3);this.axes=new Float32Array(this.capacity*4);this.pointWeights=new Float32Array(this.capacity*4);this.symbols=new Float32Array(this.capacity*2);this.symbolTints=new Float32Array(this.capacity*3);
        this.pointGeometry.setAttribute('chartSymbol',new THREE.InstancedBufferAttribute(this.symbols,2).setUsage(THREE.DynamicDrawUsage));
        this.pointGeometry.setAttribute('chartTint',new THREE.InstancedBufferAttribute(this.symbolTints,3).setUsage(THREE.DynamicDrawUsage));
        this.pointGeometry.setAttribute('sourceAxes',new THREE.InstancedBufferAttribute(this.axes,4).setUsage(THREE.DynamicDrawUsage));
        this.pointGeometry.setAttribute('sourceWeights',new THREE.InstancedBufferAttribute(this.pointWeights,4).setUsage(THREE.DynamicDrawUsage));
        this.pointGeometry.setAttribute('center',new THREE.InstancedBufferAttribute(this.centers,2).setUsage(THREE.DynamicDrawUsage));
        this.pointGeometry.setAttribute('radiance',new THREE.InstancedBufferAttribute(this.energies,3).setUsage(THREE.DynamicDrawUsage));
    }
    beginPoints(count){this.ensurePoints(count);this.count=0;}
    addPoint(direction,flux,hex,{rawWeight=1,displayWeight=1,distance=1e30}={}) {
        const first=this.project(direction);if(!Number.isFinite(first.x+first.y))return;
        const basis=diskBasis(direction),epsilon=1e-6;
        const gain=this.u.displayPass.value?this.displayScale:1;
        const eye=this.sky.eye,sigma=Math.hypot(.35,eye.opticalBlur?gain*eye.acuityArcmin*ARC_MIN*this.focal*this.dpr/2.354820045:0);
        const wingSigma=Math.hypot(.35,eye.opticalBlur?eye.glareArcmin*ARC_MIN*this.focal*this.dpr:0),wingWeight=eye.opticalBlur?eye.glareFraction:0;
        const factor=displayKey(this.sky.environment.adaptation)*2**eye.exposureEV*gain*gain/Math.max(.005,this.sky.environment.adaptation);
        const a=flux*(this.focal*this.dpr)**2*factor/(2*Math.PI*sigma*sigma),b=flux*(this.focal*this.dpr)**2*factor*wingWeight/(2*Math.PI*wingSigma*wingSigma);
        const chartRadius=pointRadius(luxToMagnitude(flux))*this.focal/SYMBOL_REFERENCE.perspectiveFocal*gain*this.dpr;
        const opticalSupport=Math.max(sigma*Math.sqrt(2*Math.max(0,Math.log(Math.max(1,a*8192)))),wingSigma*Math.sqrt(2*Math.max(0,Math.log(Math.max(1,b*8192)))))+1;
        const support=this.overview || gain>=3?chartRadius+3:Math.max(opticalSupport,gain>1?chartRadius+3:0);
        if(!this.overview){
            // A local projection derivative is valid near the source only.
            // Near 90 degrees, its enormous screen ellipse must not scatter
            // a star from far outside the field across the entire picture.
            const view=new THREE.Vector3(...direction).transformDirection(this.camera.matrixWorldInverse);
            const diagonal=Math.hypot(this.width,this.height)/2;
            const cone=this.u.projectionKind.value===1?2*Math.atan(diagonal/(2*this.focal)):Math.atan(diagonal/this.focal);
            if(Math.acos(Math.max(-1,Math.min(1,-view.z)))>cone+support/(this.focal*this.dpr))return;
        }
        const hemispheres=this.overview && Math.abs(Math.asin(direction[2]))<support/(this.focal*this.dpr)?[first.north,!first.north]:[first.north];
        for(let index=0;index<hemispheres.length;index++){
            const north=hemispheres[index];
            const project=v=>this.overview?projectHemisphere(v,this.layout.radius,this.layout,this.orientation,north):this.project(v);
            const p=project(direction),offset=axis=>{const v=direction.map((c,j)=>c+epsilon*axis[j]),n=Math.hypot(...v);return project(v.map(c=>c/n));};
            if(this.u.atlasMap.value && distance===1e30){
                // An occulted background point emits no visible PSF at all;
                // clipping its already-spread wings alone would leave a halo.
                if(this.mapCircles.some(c=>c.north===north && distance>c.distance && Math.hypot(p.x-c.x,p.y-c.y)<=c.radius))continue;
            }
            const x=offset(basis.x),y=offset(basis.y),f=this.focal*epsilon;
            const axes=[(x.x-p.x)/f,(y.x-p.x)/f,-(x.y-p.y)/f,-(y.y-p.y)/f],det=Math.abs(axes[0]*axes[3]-axes[1]*axes[2]);
            if(!axes.every(Number.isFinite) || det<1e-8)continue;
            const pad=(support*Math.max(Math.hypot(...axes.slice(0,2)),Math.hypot(...axes.slice(2)))+2)/this.dpr;
            if(p.x < -pad || p.y < -pad || p.x>this.width+pad || p.y>this.height+pad)continue;
            const i=this.count++,color=luminanceTint(hex),energy=flux*this.pixelsPerSteradian(direction);
            this.centers.set([p.x*this.dpr,(this.height-p.y)*this.dpr],2*i);
            this.energies.set([energy*color.x,energy*color.y,energy*color.z],3*i);this.axes.set(axes,4*i);
            this.pointWeights.set([index===0?rawWeight:0,displayWeight,this.overview?(north?1:-1):0,distance],4*i);
            this.symbols.set([chartRadius,pointOpacity(luxToMagnitude(flux))],2*i);this.symbolTints.set(color.displayRGB,3*i);
        }
    }
    drawPoints() {
        if(!this.count)return;
        this.pointGeometry.instanceCount=this.count;this.pointGeometry.attributes.center.needsUpdate=true;this.pointGeometry.attributes.radiance.needsUpdate=true;this.pointGeometry.attributes.sourceAxes.needsUpdate=true;this.pointGeometry.attributes.sourceWeights.needsUpdate=true;this.pointGeometry.attributes.chartSymbol.needsUpdate=true;this.pointGeometry.attributes.chartTint.needsUpdate=true;
        this.renderer.render(this.pointScene,this.passCamera);
    }
    pass(mat,target) {
        this.quad.material=mat;this.renderer.setRenderTarget(target);this.renderer.render(this.scene,this.passCamera);
    }
    boundsForBody(body) {
        if(this.overview && this.u.displayPass.value){
            const circles=this.bodyCircles(body);if(!circles.length)return null;
            const x0=Math.min(...circles.map(p=>p.x-p.radius))-2/this.dpr,x1=Math.max(...circles.map(p=>p.x+p.radius))+2/this.dpr;
            const y0=Math.min(...circles.map(p=>p.y-p.radius))-2/this.dpr,y1=Math.max(...circles.map(p=>p.y+p.radius))+2/this.dpr;
            return new THREE.Vector4(x0/this.width*2-1,1-y1/this.height*2,x1/this.width*2-1,1-y0/this.height*2);
        }
        const r=body.angularDiameter*DEG/2,basis=diskBasis(body.view),points=[this.project(body.view)];
        // Include the whole disk at a hemisphere seam; otherwise a small
        // screen rectangle keeps supersampling local to the actual object.
        if(this.overview && Math.abs(body.view[2])<Math.sin(r))return new THREE.Vector4(-1,-1,1,1);
        for(let i=0;i<64;i++){
            const a=i*Math.PI/32,v=body.view.map((c,j)=>c+Math.tan(r)*(Math.cos(a)*basis.x[j]+Math.sin(a)*basis.y[j]));
            const n=Math.hypot(...v);points.push(this.project(v.map(c=>c/n)));
        }
        if(points.some(p=>!Number.isFinite(p.x) || !Number.isFinite(p.y)))return points.some(p=>p.visible)?new THREE.Vector4(-1,-1,1,1):null;
        const x0=Math.min(...points.map(p=>p.x))-2/this.dpr,x1=Math.max(...points.map(p=>p.x))+2/this.dpr;
        const y0=Math.min(...points.map(p=>p.y))-2/this.dpr,y1=Math.max(...points.map(p=>p.y))+2/this.dpr;
        if(x1<0 || x0>this.width || y1<0 || y0>this.height)return null;
        return new THREE.Vector4(x0/this.width*2-1,1-y1/this.height*2,x1/this.width*2-1,1-y0/this.height*2);
    }
    bodyCircles(body) {return atlasBodyCircles(body,this.layout,this.orientation).filter(p=>!this.sky.frame.surface || p.north);}
    scaleAtlasMasks(factor){
        for(let i=0;i<this.u.atlasOccluderCount.value;i++){
            const c=this.u.atlasOccluders.value[i];c.x*=factor;c.y*=factor;c.z*=factor;
        }
    }
    blurTo(source,temp,destination,sigma) {
        let level=0,sigmaX=sigma,sigmaY=sigma;
        // Area averaging before convolution prevents the gaps produced by
        // sparse taps when the same angular PSF is magnified on screen.
        while(Math.max(sigmaX,sigmaY)>6 && source.width>2 && source.height>2){
            const w=Math.max(1,Math.ceil(source.width/2)),h=Math.max(1,Math.ceil(source.height/2));
            if(!this.pyramid[level])this.pyramid[level]=Array.from({length:4},()=>new THREE.WebGLRenderTarget(w,h,{type:THREE.FloatType,minFilter:THREE.NearestFilter,magFilter:THREE.NearestFilter,depthBuffer:false,stencilBuffer:false}));
            const targets=this.pyramid[level],d=this.down.uniforms;
            d.passResolution.value.set(w,h);d.sourceResolution.value.set(source.width,source.height);d.inputImage.value=source.texture;this.pass(this.down,targets[0]);
            sigmaX*=w/source.width;sigmaY*=h/source.height;
            source=targets[0];temp=targets[1];destination=targets[this.blurBranch===0?2:3];level++;
        }
        const k=gaussianKernel(Math.min(8,sigmaX)),ky=gaussianKernel(Math.min(8,sigmaY)),u=this.blur.uniforms;
        u.passResolution.value.set(source.width,source.height);
        u.weights.value.fill(0);u.weights.value.set(k.weights);u.radius.value=k.radius;
        u.inputImage.value=source.texture;u.axis.value.set(k.step,0);this.pass(this.blur,temp);
        u.weights.value.fill(0);u.weights.value.set(ky.weights);u.radius.value=ky.radius;
        u.inputImage.value=temp.texture;u.axis.value.set(0,1);this.pass(this.blur,destination);
        return destination;
    }
    render(camera,sky,stars,options={}) {
        this.displayScale=options.displayScale ?? (options.overview?DEFAULT_DISPLAY_SCALE:1);
        this.u.deepSky.value=options.deepSky!==false;
        this.configure(camera,sky,options);this.pointMaterial.uniforms.mapSymbols.value=!!options.overview;
        const [raw,temp,core,wing]=this.targets,u=this.u;
        u.adaptation.value=sky.environment.adaptation;u.displayKey.value=displayKey(sky.environment.adaptation);
        u.exposureEV.value=sky.eye.exposureEV;u.white.value=!!options.white;
        // Visibility and radiometry use the physical sky once. The display
        // scene has enlarged silhouettes, including their opaque dark sides.
        const sources=[];
        for(const star of stars) {
            const direction=backgroundDirection(star,sky.frame);
            if(sky.frame.surface && direction[2]<0)continue;
            const altitude=Math.asin(Math.max(-1,Math.min(1,direction[2])))/DEG;
            const magnitude=observedMagnitude(star.app_mag,altitude,sky.eye,sky.environment.atmosphere);
            const limit=sky.environment.atmosphere?nakedEyeLimit(skyLuminance(direction,sky.environment)):6.5;
            if(magnitude>limit)continue;
            sources.push({direction,flux:magnitudeToLux(magnitude),color:options.white?'#ffffff':star.color_hex});
        }
        const physicalBodies=new Map(sky.bodies.map(b=>[b.id,b]));let inView=0;
        const scene=(sceneSky,target,displayPass)=>{
          this.configure(camera,sceneSky,options);u.displayPass.value=displayPass;
          this.pointMaterial.uniforms.rasterScale.value=1;
          const mapDisplay=displayPass && this.overview;
          this.mapCircles=mapDisplay?sceneSky.bodies.flatMap(b=>this.bodyCircles(b).map(c=>({...c,distance:b.distance}))):[];
          if(this.mapCircles.length>ATLAS_OCCLUDER_LIMIT)throw Error('平面天体遮挡数量超出容量');
          u.atlasMap.value=mapDisplay;u.atlasOccluderCount.value=this.mapCircles.length;
          this.mapCircles.forEach((c,i)=>u.atlasOccluders.value[i].set(c.x*this.dpr,(this.height-c.y)*this.dpr,(c.north?1:-1)*c.radius*this.dpr,c.distance));
          // Map display circles and physical angular silhouettes differ.
          // Use the circle for every display mask; keep HDR geometry intact.
          const angularBodies=mapDisplay?[]:sceneSky.bodies;
          u.bounds.value.set(-1,-1,1,1);this.pass(this.background,target);
          this.beginPoints(2*(stars.length+sky.bodies.length));
          for(const source of sources){
            if(pointHidden(source.direction,angularBodies))continue;const count=this.count;
            this.addPoint(source.direction,source.flux,source.color);if(displayPass && this.count>count)inView++;
          }
          for(const body of [...sceneSky.bodies].sort((a,b)=>b.distance-a.distance)) {
            if((sceneSky.frame.surface && body.altitude < -body.angularDiameter/2 && !mapDisplay) || (!mapDisplay && bodyFullyOcculted(body,sceneSky)))continue;
            const radius=Math.sqrt(this.pixelsPerSteradian(body.view))*body.angularDiameter*DEG/2;
            const physical=physicalBodies.get(body.id),physicalRadius=Math.sqrt(this.pixelsPerSteradian(body.view))*physical.angularDiameter*DEG/2;
            // Changing the display gain does not change point/disk weighting
            // or surface brightness; that would counteract the size change.
            const t=Math.max(0,Math.min(1,(physicalRadius-.6)/1.2)),pointWeight=body.kind==='planet'?(this.overview?unresolvedPointWeight(2*SYMBOL_REFERENCE.perspectiveFocal*Math.tan(physical.angularDiameter*DEG/2)):1-t*t*(3-2*t)):0;
            if((radius<1.2 || pointWeight>0) && body.brightEnough){
                const sample=bodyLightSample(body,mapDisplay?{...sceneSky,bodies:angularBodies}:sceneSky);
                if(sample.fraction>0)this.addPoint(sample.direction,magnitudeToLux(body.observedMagnitude)*sample.fraction,options.white?'#ffffff':body.color,{rawWeight:radius<1.2?1:0,displayWeight:pointWeight,distance:body.distance});
            }
            const bounds=this.boundsForBody(body);if(!bounds)continue;
            const b=this.body.uniforms,basis=diskBasis(body.view);
            u.bounds.value.copy(bounds);b.bodyDirection.value.fromArray(body.view);b.axisX.value.fromArray(basis.x);b.axisY.value.fromArray(basis.y);b.light.value.fromArray(diskLight(body,basis));
            b.angularRadius.value=body.angularDiameter*DEG/2;b.peak.value=body.brightEnough && (displayPass || radius>=1.2)?diskPeakLuminance(physical,body.observedMagnitude):0;
            b.tint.value.copy(luminanceTint(options.white?'#ffffff':body.color));b.luminous.value=body.id==='Sol';b.roundAtlasBody.value=mapDisplay;b.sourceWeight.value=displayPass?1-pointWeight:1;
            b.bodyNorth.value.set(0,0,-1);b.bodySouth.value.set(0,0,-1);
            if(mapDisplay)for(const c of this.bodyCircles(body)){
                (c.north?b.bodyNorth:b.bodySouth).value.set(c.x*this.dpr,(this.height-c.y)*this.dpr,c.radius*this.dpr);
                (c.north?b.lightNorth:b.lightSouth).value.fromArray(c.light);
            }
            this.pass(this.body,target);
          }
        };
        scene(sky,raw,false);
        // Point flux has already been clipped in angular space. Deposit it
        // after the disks so pixel coverage does not occult it a second time.
        this.drawPoints();
        // Draw a separate bounded display scene. Coverage blends already
        // mapped samples, so a 1/144 limb sample cannot turn a whole pixel white.
        scene(chartSky(sky,this.displayScale),temp,true);
        u.bounds.value.set(-1,-1,1,1);
        const sigma=sky.eye.acuityArcmin*ARC_MIN*this.focal*this.dpr/2.354820045*this.displayScale;
        this.blurBranch=0;const coreResult=sky.eye.opticalBlur?this.blurTo(temp,wing,core,sigma):temp;
        const wingSigma=sky.eye.glareArcmin*ARC_MIN*this.focal*this.dpr*this.displayScale;
        this.blurBranch=1;const wingResult=sky.eye.opticalBlur?this.blurTo(temp,wing,temp,wingSigma):temp;
        const t=this.tone.uniforms;t.coreImage.value=coreResult.texture;t.wingImage.value=wingResult.texture;t.wingWeight.value=sky.eye.opticalBlur?sky.eye.glareFraction:0;
        t.coreSize.value.set(coreResult.width,coreResult.height);t.wingSize.value.set(wingResult.width,wingResult.height);
        this.pass(this.tone,wing);
        const samplingDpr=this.dpr,rasterScale=this.outputDpr/samplingDpr;
        let presentation=wing;
        if(this.presentation){
            // Upsample only the sky and extended bodies. Point sources are
            // subsequently integrated into native display pixels, in linear
            // light, so neither CSS interpolation nor sRGB addition blurs them.
            this.dpr=this.outputDpr;this.configure(camera,this.sky,options);
            u.resolution.value.set(this.presentation.width,this.presentation.height);
            this.scaleAtlasMasks(rasterScale);
            t.coreImage.value=t.wingImage.value=wing.texture;t.wingWeight.value=0;
            t.coreSize.value.set(wing.width,wing.height);t.wingSize.value.set(wing.width,wing.height);
            this.pass(this.tone,this.presentation);presentation=this.presentation;
        }
        const p=this.pointMaterial.uniforms;
        // Enhancement uses the same magnitude-dependent angular symbol as
        // the atlas. Pixel antialiasing never receives the 6x/8x size gain.
        p.pointGain.value=this.displayScale;p.rasterScale.value=rasterScale;
        p.pointSigma.value=Math.hypot(.35,sky.eye.opticalBlur?sigma*rasterScale:0);
        p.pointWingSigma.value=Math.hypot(.35,sky.eye.opticalBlur?wingSigma/this.displayScale*rasterScale:0);
        const mix=Math.max(0,Math.min(1,(this.displayScale-1)/2));p.pointSymbolMix.value=mix*mix*(3-2*mix);
        p.pointWingWeight.value=sky.eye.opticalBlur?sky.eye.glareFraction:0;this.drawPoints();
        this.output.uniforms.inputImage.value=presentation.texture;this.pass(this.output,null);
        if(this.presentation){
            this.dpr=samplingDpr;this.configure(camera,this.sky,options);u.resolution.value.set(raw.width,raw.height);
            this.scaleAtlasMasks(1/rasterScale);
        }
        return {eligible:sources.length,inView,focal:this.focal,dpr:this.dpr,outputDpr:this.outputDpr,resourceGeneration:this.resourceGeneration,linearFiltering:this.linearFiltering};
    }
    readLinear() {
        const w=this.targets[0].width,h=this.targets[0].height,data=new Float32Array(w*h*4);
        this.renderer.readRenderTargetPixels(this.targets[0],0,0,w,h,data);return {data,width:w,height:h};
    }
    readPixels() {
        const gl=this.renderer.getContext(),width=gl.drawingBufferWidth,height=gl.drawingBufferHeight,data=new Uint8Array(width*height*4);
        gl.readPixels(0,0,width,height,gl.RGBA,gl.UNSIGNED_BYTE,data);return {data,width,height};
    }
    dispose() {
        for(const t of this.targets)t.dispose();for(const level of this.pyramid)for(const t of level)t.dispose();
        this.presentation?.dispose();
        for(const m of [this.background,this.body,this.blur,this.down,this.tone,this.output,this.pointMaterial])m.dispose();
        this.chartTexture.dispose();this.pointGeometry.dispose();this.quadGeometry.dispose();this.renderer.dispose();
    }
}
