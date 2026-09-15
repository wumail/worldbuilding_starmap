import {catalogURL,diffuseInfo} from './catalog_data.mjs';
import {SYSTEM,TERRAX,LOCAL_DAY,MAX_PERIOD,DEG,AU_KM,clamp,unit,subtract,scale,dot,cross,skyState,applyFrame,prepareBackground,projectHemisphere} from './solar_system.mjs';

// 两页共享同一组星点显示参数。半径没有暗星下限，避免把大部分星表压成同尺寸。
// 星等每增加 1，通量约减至 1/2.512；屏幕对这一动态范围作压缩，不表示实体角径。
export const POINT_STYLE=Object.freeze({referenceMagnitude:6.5,referenceRadius:.023,radiusExponent:.09,
    maximumRadius:.17,referenceOpacity:.5,opacityExponent:.14,coreFraction:.2,exposure:64});
// 显示基准固定下来，缩放和窗口适配只改变投影比例，不改变天体的符号配比。
// 基准角尺度取 256 px 高、60° 透视视场；它不等于原来的点像大小。
export const SYMBOL_REFERENCE=Object.freeze({atlasRadius:(Math.PI/2)*128/Math.tan(30*DEG),perspectiveFocal:128/Math.tan(30*DEG)});
// Shared readable preset; angular metrics still report physical values.
export const DEFAULT_DISPLAY_SCALE=6;
export const MAX_DISPLAY_SCALE=8;
export function displaySky(sky,gain=1) {
    if(gain===1)return sky;
    return {...sky,bodies:sky.bodies.map(b=>({...b,physicalAngularDiameter:b.angularDiameter,
        angularDiameter:2*Math.atan(gain*Math.tan(b.angularDiameter*DEG/2))/DEG}))};
}
export const atlasSymbolScale=radius=>radius/(Math.PI/2)/SYMBOL_REFERENCE.perspectiveFocal;
// Circular atlas disks: retain the projected center and local disk area,
// but remove the azimuthal map's radial/tangential aspect-ratio distortion.
// The phase angle is unchanged; the bright limb faces the projected light.
// Both atlas renderers and their opaque masks share these display circles.
export function atlasBodyCircles(body,layout,orientation,angularRadius=body.angularDiameter*DEG/2) {
    const theta=Math.acos(clamp(Math.abs(body.view[2]),0,1));
    const areaScale=theta<1e-8?1:theta/Math.sin(theta);
    const radius=layout.radius/(Math.PI/2)*Math.tan(angularRadius)*Math.sqrt(areaScale);
    const cosine=body.id==='Sol'?1:clamp(-dot(body.view,body.lightDirection),-1,1);
    const tangent=body.id==='Sol'?[0,0,0]:body.lightDirection.map((v,i)=>v+cosine*body.view[i]);
    const amplitude=Math.sqrt(Math.max(0,1-cosine*cosine)),length=Math.hypot(...tangent);
    const toward=length>1e-8?unit(body.view.map((v,i)=>v+1e-5*tangent[i]/length)):null;
    return [true,false].map(north=>{
        const p=projectHemisphere(body.view,layout.radius,layout,orientation,north),center=north?layout.north:layout.south;
        const q=toward?projectHemisphere(toward,layout.radius,layout,orientation,north):p,dx=q.x-p.x,dy=p.y-q.y,n=Math.hypot(dx,dy);
        const light=n>1e-10?[amplitude*dx/n,amplitude*dy/n,cosine]:[0,0,cosine];
        return {...p,north,radius,light,visible:Math.hypot(p.x-center[0],p.y-center[1])<=layout.radius+radius};
    }).filter(p=>p.visible);
}
export function perspectiveSymbolScale(height,fov) {
    return height/(2*Math.tan(fov*DEG/2))/SYMBOL_REFERENCE.perspectiveFocal;
}
export function pointRadius(mag) {
    const p=POINT_STYLE;return Math.min(p.maximumRadius,p.referenceRadius*10**(p.radiusExponent*(p.referenceMagnitude-mag)));
}
export function pointOpacity(mag) {
    const p=POINT_STYLE;return Math.min(1,p.referenceOpacity*10**(p.opacityExponent*(p.referenceMagnitude-mag)));
}
export function pointProfile(distance) {
    const t=clamp((distance-POINT_STYLE.coreFraction)/(1-POINT_STYLE.coreFraction),0,1);
    return 1-t*t*(3-2*t);
}
// 压缩显示曝光；像素面积积分与亚像素对比度过渡由共享采样核处理。
export const pointExposure=coverage=>POINT_STYLE.exposure*coverage/(1+POINT_STYLE.exposure*coverage);
export function pointImage(color,size=64) {
    const image=new ImageData(size,size),hex=color.slice(1),rgb=[0,2,4].map(i=>parseInt(hex.slice(i,i+2),16));
    for(let y=0;y<size;y++)for(let x=0;x<size;x++) {
        const k=4*(y*size+x),r=Math.hypot((x+.5)/size*2-1,(y+.5)/size*2-1);
        for(let c=0;c<3;c++)image.data[k+c]=rgb[c];
        image.data[k+3]=Math.round(255*pointExposure(pointProfile(r)));
    }
    return image;
}
// 仅按固定显示基准下的盘面大小分配增强权重；缩放不改变层的比例。
export function unresolvedPointWeight(referenceDiameter) {
    const t=clamp((referenceDiameter-1)/3,0,1);
    return 1-t*t*(3-2*t);
}
export const srgbToLinear=c=>c<=.04045?c/12.92:((c+.055)/1.055)**2.4;
export const linearToSrgb=c=>c<=.0031308?12.92*c:1.055*c**(1/2.4)-.055;
// 这是共同的示意天空底色；暗面保留遮挡，但不能把前景天空抹成黑洞。
export function skyBackground(sky) {
    const day=1-(sky.night ?? 1);
    return [5+12*day,9+31*day,17+47*day].map(Math.round);
}
export function backgroundVisible(star,direction,sky) {
    return (!sky.frame.surface || direction[2]>=0) && star.app_mag<=Math.min(6.5,sky.limitingMagnitude);
}

// 柔光是显示层；其范围固定为日面角半径的倍数，不用星等制造巨型盘面。
export const SOLAR_GLARE_RADIUS_RATIO=2.2;
export function bodyGlareImage(color,size=128) {
    const image=new ImageData(size,size);
    const hex=color.slice(1),tint=[0,2,4].map(i=>parseInt(hex.slice(i,i+2),16));
    for(let y=0;y<size;y++)for(let x=0;x<size;x++) {
        const r2=((x+.5)/size*2-1)**2+((y+.5)/size*2-1)**2;
        if(r2>=1)continue;
        const k=4*(y*size+x);
        for(let c=0;c<3;c++)image.data[k+c]=tint[c];
        image.data[k+3]=255*clamp(2.4*Math.exp(-5*r2)*(1-r2)**2,0,1);
    }
    return image;
}

export const lunarPhaseCosine=body=>clamp(-dot(body.view,body.lightDirection),-1,1);
export function lunarSurfaceScale(body,enhanced=false) {
    if(!enhanced || body.kind!=='moon')return 1;
    const c=lunarPhaseCosine(body),peak=c>=0?1:Math.sqrt(Math.max(0,1-c*c));
    return peak>1e-8?Math.min(4,1/peak):1;
}
// 遮挡对柔光的影响按可见亮面加权；星等已包含相位，不能再乘一次月相。
// 只在升落或遮挡时采样，前景盘面取并集，避免重复扣减。
export function bodyLightVisibility(body,sky) {
    if(!body || body.brightEnough===false)return 0;
    const radius=body.angularDiameter*DEG/2,altitude=Math.asin(clamp(body.view[2],-1,1));
    if(sky.frame.surface && altitude<=-radius)return 0;
    const blockers=sky.bodies.filter(b=>b.id!==body.id && b.distance<body.distance && dot(b.view,body.view)>Math.cos(radius+b.angularDiameter*DEG/2));
    for(const b of blockers)if(b.angularDiameter*DEG/2>=radius+Math.acos(clamp(dot(b.view,body.view),-1,1)))return 0;
    if(!blockers.length && (!sky.frame.surface || altitude>=radius))return 1;
    const basis=diskBasis(body.view),light=body.id==='Sol'?null:diskLight(body,basis),extent=Math.tan(radius),limits=blockers.map(b=>Math.cos(b.angularDiameter*DEG/2));
    let total=0,visible=0;
    for(let y=0;y<48;y++)for(let x=0;x<48;x++) {
        const u=(x+.5)/24-1,v=(y+.5)/24-1;if(u*u+v*v>1)continue;
        const weight=light?Math.max(0,u*light[0]+v*light[1]+Math.sqrt(1-u*u-v*v)*light[2]):1;total+=weight;
        const direction=unit(body.view.map((c,i)=>c+extent*(u*basis.x[i]+v*basis.y[i])));
        if(sky.frame.surface && direction[2]<0)continue;
        if(!blockers.some((b,i)=>dot(direction,b.view)>=limits[i]))visible+=weight;
    }
    return total>0?visible/total:0;
}
export function bodyFullyOcculted(body,sky) {
    const radius=body.angularDiameter*DEG/2;
    return sky.bodies.some(front=>front.id!==body.id && front.distance<body.distance &&
        front.angularDiameter*DEG/2>=radius+Math.atan2(Math.hypot(...cross(front.view,body.view)),clamp(dot(front.view,body.view),-1,1)));
}
export const solarGlareVisibility=sky=>bodyLightVisibility(sky.bodies.find(b=>b.id==='Sol'),sky);
export const fmt=(value,digits=2)=>Number.isFinite(value)?value.toFixed(digits):'—';

export async function catalogFolders() {
    const response=await fetch(new URL('../../output/folders.json',import.meta.url),{cache:'no-store'});
    if (!response.ok) throw new Error('无法读取星表列表');
    const folders=await response.json();
    if (!Array.isArray(folders) || !folders.length || folders.some(f=>!/^output_[A-Za-z0-9_]+$/.test(f))) throw new Error('星表列表无效');
    return folders;
}
export async function loadCatalog(folder) {
    if (!/^output_[A-Za-z0-9_]+$/.test(folder)) throw new Error('星表名称无效');
    const url=await catalogURL(folder);const response=await fetch(url);
    if (!response.ok) throw new Error(`星表读取失败（${response.status}）`);
    const data=await response.json();
    if (!Array.isArray(data.stars) || (data.neighbors!==undefined && !Array.isArray(data.neighbors))) throw new Error('星表格式无效');
    const stars=prepareBackground([...(data.neighbors || []),...data.stars]);
    return {stars,metadata:data.metadata || {},deepSky:data.deep_sky?.objects || [],diffuse:diffuseInfo(data,url)};
}

// 物体朝向观察者的切平面。x、y、z构成右手系，z朝向观察者。
export function diskBasis(direction) {
    const z=scale(direction,-1), reference=Math.abs(z[2])>.999?[0,1,0]:[0,0,1];
    const x=unit(cross(reference,z)),y=cross(z,x);
    return {x,y,z};
}
export function diskLight(body,basis=diskBasis(body.view)) {
    if (body.id==='Sol') return [0,0,1];
    return [dot(body.lightDirection,basis.x),dot(body.lightDirection,basis.y),dot(body.lightDirection,basis.z)];
}

// 等尺寸相位预览；天空内的盘面用真实角直径，面板预览明确放大。
export function phaseImage(body,light,size=96,{background=[0,0,0],brightEnough=true,lightScale=1}={}) {
    const image=new ImageData(size,size),hex=body.color.slice(1),rgb=[0,2,4].map(i=>srgbToLinear(parseInt(hex.slice(i,i+2),16)/255));
    const backdrop=background.map(c=>srgbToLinear(c/255));
    for(let py=0;py<size;py++) for(let px=0;px<size;px++) {
        const x=(px+.5)/size*2-1,y=1-(py+.5)/size*2,r2=x*x+y*y;
        if(r2>1)continue;
        const z=Math.sqrt(1-r2),illumination=body.id==='Sol'?1:brightEnough?Math.max(0,x*light[0]+y*light[1]+z*light[2]):0;
        const k=4*(py*size+px);
        // Lambert 辐亮度在线性空间计算，两页只在最后转换为 sRGB。
        for(let c=0;c<3;c++) image.data[k+c]=255*linearToSrgb(clamp(backdrop[c]+rgb[c]*illumination*lightScale,0,1));
        image.data[k+3]=255;
    }
    return image;
}

export function trajectory(days,options,id,kind) {
    if(kind==='off')return [];
    const span=kind==='day'?LOCAL_DAY:TERRAX.period;
    const start=clamp(days-span/2,0,MAX_PERIOD-span),frame=skyState(days,options).frame;
    const points=[];
    for(let n=0;n<=256;n++) {
        const time=start+span*n/256,body=skyState(time,options).bodies.find(b=>b.id===id);
        // 年轨迹扣除自转，始终与当前恒星背景对齐；26h轨迹为各时刻的地平位置。
        points.push({time,equatorial:body.equatorial,direction:kind==='year'?applyFrame(body.equatorial,frame):body.view});
    }
    return points;
}
export const trajectoryDirections=(points,frame,kind)=>points.map(p=>kind==='year'?applyFrame(p.equatorial,frame):p.direction);
