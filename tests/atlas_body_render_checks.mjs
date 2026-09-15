import * as THREE from 'three';
import {AtlasBodyPainter} from '../web/v1/sky_atlas_bodies.mjs';
import {AtlasStarPainter} from '../web/v1/sky_atlas_stars.mjs';
import {EyeSkyRenderer} from '../web/v2/eye_renderer.mjs';
import {observerSky,EYE_DEFAULTS} from '../web/v2/eye_model.mjs';
import {sanitizeState} from '../web/v2/sky_state.mjs';
import {diskBasis,skyBackground,SYMBOL_REFERENCE,srgbToLinear} from '../web/shared/sky_render.mjs';
import {DEG,equatorialDirection,lambertPhase,unit,projectHemisphere} from '../web/shared/solar_system.mjs';
const base=observerSky(0,sanitizeState({mode:'center'}));base.frame.position=[0,0,0];
base.environment={atmosphere:false,extinction:0,nightL:0,dayL:0,moons:[],adaptation:.005};base.eye={...EYE_DEFAULTS,opticalBlur:false};
const canvas=document.createElement('canvas'),eye=new EyeSkyRenderer(canvas),camera=new THREE.PerspectiveCamera(60,1,.01,10);camera.up.set(0,0,1);camera.lookAt(1,0,0);
const c=document.createElement('canvas'),ctx=c.getContext('2d'),painter=new AtlasBodyPainter(ctx),points=new AtlasStarPainter(ctx);
let total=0,failures=0;const metrics={};
const assert=(v,m)=>{if(!v)throw Error(m);};
async function test(name,fn){total++;const li=document.createElement('li');try{await fn();li.className='pass';li.textContent='通过：'+name;}catch(e){failures++;li.className='fail';li.textContent='失败：'+name+' — '+e.message;}document.querySelector('#results').append(li);await new Promise(r=>setTimeout(r,0));}
function body(id='Echo',dec=8,phase=0){
 const view=equatorialDirection(38,dec),basis=diskBasis(view),a=phase*DEG;
 const magnitude=(id==='Echo'?-9.9:-12.7)-2.5*Math.log10(Math.max(1e-20,lambertPhase(a)));
 return {id,kind:id==='Venus-Sol'?'planet':id==='Sol'?'star':'moon',color:'#ffffff',view,distance:.01,angularDiameter:id==='Echo'?.11486:id==='Venus-Sol'?.01:id==='Sol'?.43348:.52747,
  lightDirection:view.map((v,i)=>basis.z[i]*Math.cos(a)+basis.x[i]*Math.sin(a)),phaseAngle:a,magnitude,observedMagnitude:magnitude,brightEnough:phase<180,visible:phase<180,altitude:dec,aboveHorizon:dec>=0};
}
function settings(b,{target=28,gain=6,size=256,dpr=1,surface=false,north=b.view[2]>=0}={}){
 const sky={...base,bodies:[b],frame:{...base.frame,surface}};eye.resize(size,size,dpr);eye.configure(camera,sky,{focal:1500,overview:true});
 const theta=Math.acos(Math.abs(b.view[2])),J=theta<1e-8?1:theta/Math.sin(theta),scale=eye.focal;
 const zoom=target/(scale*gain*Math.tan(b.angularDiameter*DEG/2)*Math.sqrt(J));
 const p=projectHemisphere(b.view,eye.layout.radius,eye.layout,eye.orientation,north);
 return {overview:true,focal:1500,displayScale:gain,size,dpr,surface,atlasTransform:{zoom,panX:(size/2-p.x)*zoom,panY:(size/2-p.y)*zoom},target};
}
function show(version,bodies,options,stars=[],planetGlow=false){
 eye.resize(options.size,options.size,options.dpr);const sky={...base,bodies,frame:{...base.frame,surface:options.surface}};
 eye.render(camera,sky,stars,options);
 if(version==='V2'){
  const im=eye.readPixels(),out=new Uint8ClampedArray(im.data.length);
  for(let y=0;y<im.height;y++)out.set(im.data.subarray((im.height-1-y)*im.width*4,(im.height-y)*im.width*4),y*im.width*4);
  return {...im,data:out};
 }
 c.width=c.height=options.size*options.dpr;ctx.setTransform(options.dpr,0,0,options.dpr,0,0);ctx.fillStyle='rgb('+skyBackground(sky).join(',')+')';ctx.fillRect(0,0,options.size,options.size);
 for(const s of stars){const p=eye.project(s.baseDirection);points.draw(s,p,eye.focal/SYMBOL_REFERENCE.perspectiveFocal*options.displayScale);}
 painter.drawSky(eye.sky,eye.layout,eye.orientation,{planetGlow,solarGlow:false});
 return {width:c.width,height:c.height,data:ctx.getImageData(0,0,c.width,c.height).data};
}
function radii(im,blank,extent){
 const n=im.width,h=n/2,signal=(x,y)=>Math.max(0,srgbToLinear(im.data[4*(y*n+x)]/255)-srgbToLinear(blank.data[4*(y*n+x)]/255));
 const sample=(x,y)=>{const i=Math.floor(x-.5),j=Math.floor(y-.5),fx=x-.5-i,fy=y-.5-j;return (signal(i,j)*(1-fx)+signal(i+1,j)*fx)*(1-fy)+(signal(i,j+1)*(1-fx)+signal(i+1,j+1)*fx)*fy;};
 const threshold=sample(h,h)*.10,rs=[];
 for(let a=0;a<64;a++){let last=0;for(let r=0;r<extent;r+=.1)if(sample(h+r*Math.cos(a*Math.PI/32),h+r*Math.sin(a*Math.PI/32))>threshold)last=r;rs.push(last);}
 return {min:Math.min(...rs),max:Math.max(...rs),average:rs.reduce((a,b)=>a+b)/rs.length};
}
function centroid(im,blank){let weight=0,x=0,y=0;for(let j=0;j<im.height;j++)for(let i=0;i<im.width;i++){const k=4*(j*im.width+i),w=Math.max(0,srgbToLinear(im.data[k]/255)-srgbToLinear(blank.data[k]/255));weight+=w;x+=(i+.5-im.width/2)*w;y+=(j+.5-im.height/2)*w;}return {x:x/weight,y:y/weight,weight};}
await test('V1/V2 日月在离极点位置、南北半球、缩放和 DPR 下保持圆形',()=>{
 const cases=[];let worst=0,scaleError=0;
 for(const version of ['V1','V2'])for(const id of ['Sol','Luna','Echo'])for(const dec of [75,30,5,-5,-30,-75])for(const dpr of [1,2])for(const target of [16,40]){
  const b=body(id,dec),o=settings(b,{target,dpr}),blank=show(version,[],o),im=show(version,[b],o),r=radii(im,blank,target*dpr*1.7);
  worst=Math.max(worst,r.max-r.min);scaleError=Math.max(scaleError,Math.abs(r.average-target*dpr*Math.sqrt(.99)));
  cases.push({version,id,dec,dpr,target,...r});
 }
 metrics.roundness={worst,scaleError,cases};assert(worst<1.4 && scaleError<1.3,JSON.stringify({worst,scaleError}));
});
await test('圆形双月保留正确月相比例、朝向与新月暗面',()=>{
 const cases=[];let minimumAlignment=1;
 for(const version of ['V1','V2'])for(const id of ['Luna','Echo'])for(const dec of [8,-8,80,-80]){
  const b=body(id,dec),o=settings(b,{target:40}),blank=show(version,[],o);let previous=Infinity;
  for(const phase of [0,60,90,120,150,180]){
   const moon=body(id,dec,phase),im=show(version,[moon],o),m=centroid(im,blank);assert(m.weight<previous,`${version} ${id} ${phase} 非单调月相`);previous=m.weight;
   if(phase===90){const p=eye.project(moon.view),q=eye.project(unit(moon.view.map((v,i)=>v+1e-5*moon.lightDirection[i]))),dx=q.x-p.x,dy=q.y-p.y,cos=(m.x*dx+m.y*dy)/Math.hypot(m.x,m.y)/Math.hypot(dx,dy);minimumAlignment=Math.min(minimumAlignment,cos);}
   if(phase===180)assert(m.weight===0,`${version} 新月仍发亮`);cases.push({version,id,dec,phase,weight:m.weight});
  }
 }
 metrics.phases={minimumAlignment,cases};assert(minimumAlignment>.995,String(minimumAlignment));
});
const star=d=>({id:'witness',app_mag:0,color_hex:'#ffffff',baseDirection:d,baseDistanceAU:1e12});
await test('离轴暗月按新的圆形外缘遮星，旧椭圆多出的区域不误挡',()=>{
 const cases=[];
 for(const version of ['V1','V2'])for(const id of ['Luna','Echo'])for(const dec of [8,-8])for(const dpr of [1,2]){
  const b=body(id,dec,180),o=settings(b,{target:32,dpr}),blank=show(version,[b],o),l=eye.layout,p=eye.project(b.view),h=p.north?l.north:l.south,radial=[p.x-h[0],p.y-h[1]],length=Math.hypot(...radial),R=32;
  const dirAt=(x,y)=>{const theta=Math.hypot(x-h[0],y-h[1])/l.radius*Math.PI/2,sign=p.north?1:-1,phi=Math.atan2(-sign*(x-h[0]),y-h[1])+eye.orientation*DEG;return [Math.sin(theta)*Math.cos(phi),Math.sin(theta)*Math.sin(phi),sign*Math.cos(theta)];};
  const inside=dirAt(p.x+radial[0]/length*.94*R,p.y+radial[1]/length*.94*R),outside=dirAt(p.x-radial[1]/length*1.13*R,p.y+radial[0]/length*1.13*R);
  const covered=show(version,[b],o,[star(inside)]),exposed=show(version,[b],o,[star(outside)]);let leak=0,visible=0;
  for(let y=0;y<blank.height;y++)for(let x=0;x<blank.width;x++){const k=4*(y*blank.width+x);if(Math.hypot(x+.5-blank.width/2,y+.5-blank.height/2)<R*dpr-2)leak=Math.max(leak,Math.abs(blank.data[k]-covered.data[k]));visible+=Math.max(0,exposed.data[k]-blank.data[k]);}
  cases.push({version,id,dec,dpr,leak,visible});
 }
 metrics.occlusion=cases;assert(cases.every(c=>c.leak<=1 && c.visible>50),JSON.stringify(cases.filter(c=>c.leak>1 || c.visible<=50)));
});
await test('双月跨半球接缝的两侧都有盘面，地表模式只显示地平线上部分',()=>{
 const cases=[];
 for(const version of ['V1','V2'])for(const id of ['Luna','Echo'])for(const north of [true,false])for(const surface of [false,true]){
  const b=body(id,0),o=settings(b,{target:36,north,surface}),empty=show(version,[],o),im=show(version,[b],o);let signal=0;
  for(let i=0;i<im.data.length;i+=4)signal+=Math.max(0,im.data[i]-empty.data[i]);
  cases.push({version,id,north,surface,signal});assert(surface&&!north?signal===0:signal>100,JSON.stringify(cases.at(-1)));
 }
 metrics.seams=cases;
});
await test('平面圆形显示与增强不改 V2 物理亮度缓冲',()=>{
 let worst=0;
 for(const id of ['Luna','Echo','Venus-Sol'])for(const dec of [8,-8]){
  const b=body(id,dec,60),o=settings(b,{target:32,gain:1});show('V2',[b],o);const before=eye.readLinear();show('V2',[b],{...o,displayScale:6});const after=eye.readLinear();
  for(let i=0;i<before.data.length;i++)worst=Math.max(worst,Math.abs(before.data[i]-after.data[i]));
 }
 metrics.physicalDifference=worst;assert(worst===0,String(worst));
});
await test('平面圆盘与行星光点按距离遮挡，前景行星仍可见',()=>{
 const cases=[];
 for(const version of ['V1','V2'])for(const id of ['Luna','Echo']){
  const moon=body(id,8,180),o=settings(moon,{target:36}),planet={...body('Venus-Sol',8),angularDiameter:.002,magnitude:-4,observedMagnitude:-4,distance:.1};
  const blank=show(version,[moon],o,[],true),back=show(version,[moon,planet],o,[],true),front=show(version,[moon,{...planet,distance:.001}],o,[],true);let leak=0,visible=0;
  for(let i=0;i<blank.data.length;i+=4){leak=Math.max(leak,Math.abs(back.data[i]-blank.data[i]));visible+=Math.max(0,front.data[i]-blank.data[i]);}
  cases.push({version,id,leak,visible});assert(leak<=1 && visible>100,JSON.stringify(cases.at(-1)));
 }
 metrics.distanceOrder=cases;
});
for(const version of ['V1','V2'])for(const phase of [0,60,90]){
 const b=body('Echo',8,phase),o=settings(b,{target:48}),im=show(version,[b],o),figure=document.createElement('figure'),image=document.createElement('canvas'),label=document.createElement('figcaption');
 image.width=im.width;image.height=im.height;image.getContext('2d').putImageData(new ImageData(im.data,im.width,im.height),0,0);label.textContent=`${version} Echo · ${phase===0?'满月':phase===60?'凸月':'半月'}`;figure.append(image,label);document.querySelector('#samples').append(figure);
}
painter.clear();points.clear();eye.dispose();document.querySelector('#status').textContent=`${total-failures}/${total} 通过；${failures} 失败`;window.renderCheckReport={total,failures,metrics};
