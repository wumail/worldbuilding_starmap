import {catalogURL,diffuseInfo} from '../shared/catalog_data.mjs';
import {TERRAX,LOCAL_DAY,MAX_PERIOD,DEG,clamp,unit,scale,dot,cross,skyState,applyFrame,prepareBackground} from '../shared/solar_system.mjs';

export const srgbToLinear=c=>c<=.04045?c/12.92:((c+.055)/1.055)**2.4;
export const linearToSrgb=c=>c<=.0031308?12.92*c:1.055*c**(1/2.4)-.055;
// 遮挡对柔光的影响按可见亮面加权；星等已包含相位，不能再乘一次月相。
// 只在升落或遮挡时采样，前景盘面取并集，避免重复扣减。
export function bodyLightSample(body,sky) {
    const empty={fraction:0,direction:body?.view};
    if(!body || body.brightEnough===false)return empty;
    const radius=body.angularDiameter*DEG/2,altitude=Math.asin(clamp(body.view[2],-1,1));
    if(sky.frame.surface && altitude<=-radius)return empty;
    const blockers=sky.bodies.filter(b=>b.id!==body.id && b.distance<body.distance && dot(b.view,body.view)>Math.cos(radius+b.angularDiameter*DEG/2));
    if(bodyFullyOcculted(body,sky))return empty;
    if(!blockers.length && (!sky.frame.surface || altitude>=radius))return {fraction:1,direction:body.view};
    const basis=diskBasis(body.view),light=body.id==='Sol'?null:diskLight(body,basis),extent=Math.tan(radius),limits=blockers.map(b=>Math.cos(b.angularDiameter*DEG/2));
    let total=0,visible=0,centroid=[0,0,0];
    for(let y=0;y<48;y++)for(let x=0;x<48;x++) {
        const u=(x+.5)/24-1,v=(y+.5)/24-1;if(u*u+v*v>1)continue;
        const weight=light?Math.max(0,u*light[0]+v*light[1]+Math.sqrt(1-u*u-v*v)*light[2]):1;total+=weight;
        const direction=unit(body.view.map((c,i)=>c+extent*(u*basis.x[i]+v*basis.y[i])));
        if(sky.frame.surface && direction[2]<0)continue;
        if(!blockers.some((b,i)=>dot(direction,b.view)>=limits[i])){visible+=weight;centroid=centroid.map((v,i)=>v+weight*direction[i]);}
    }
    return total>0 && visible>0?{fraction:visible/total,direction:unit(centroid)}:empty;
}
export const bodyLightVisibility=(body,sky)=>bodyLightSample(body,sky).fraction;
export function bodyFullyOcculted(body,sky) {
    const radius=body.angularDiameter*DEG/2;
    return sky.bodies.some(front=>front.id!==body.id && front.distance<body.distance &&
        front.angularDiameter*DEG/2>=radius+Math.atan2(Math.hypot(...cross(front.view,body.view)),clamp(dot(front.view,body.view),-1,1)));
}
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

// 等尺寸相位预览；主图的实际角径与显示增益由渲染器处理。
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
