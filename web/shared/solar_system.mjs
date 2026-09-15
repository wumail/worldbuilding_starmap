// 轨道长度单位 AU，时间单位地球日，角度输入为度；输出为赤道惯性坐标。
export const AU_KM = 149597870.7;
export const AU_PER_PC = 648000 / Math.PI;
export const EARTH_YEAR = 365.25;
export const DEG = Math.PI / 180;
export const TAU = 2 * Math.PI;
export const clamp = (x, a, b) => Math.max(a, Math.min(b, x));
export const mod = (x, p = TAU) => ((x % p) + p) % p;
export const norm = v => Math.hypot(...v);
export const dot = (a, b) => a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
export const add = (a, b) => a.map((x, i) => x + b[i]);
export const subtract = (a, b) => a.map((x, i) => x - b[i]);
export const scale = (a, s) => a.map(x => x*s);
export const unit = a => scale(a, 1/norm(a));
export const cross = (a, b) => [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]];
export const rotateX = (v, a) => [v[0], v[1]*Math.cos(a)-v[2]*Math.sin(a), v[1]*Math.sin(a)+v[2]*Math.cos(a)];
export const rotateZ = (v, a) => [v[0]*Math.cos(a)-v[1]*Math.sin(a), v[0]*Math.sin(a)+v[1]*Math.cos(a), v[2]];

// 已给定的周期是本模拟时间基准，不与另一套推导周期混用。
// node/peri/mean 中未见于文档的值统一属于可编辑展示初值。
export const SYSTEM = {
    sources: ['../../reference/恒星系最终设定.md', '../../reference/母星最终设定.md'],
    star: {id:'Sol', name:'Sol', kind:'star', massSolar:1.04, radiusKm:758313, color:'#fff3d6', magnitudeAtReference:-26.493},
    observer: {id:'Terrax', radiusKm:7016.8, massKg:7.60148e24, obliquity:25, spinHours:26, heightMetres:1.75},
    bodies: [
        {id:'Mercury-Sol',name:'Mercury-Sol',kind:'planet',a:.50,e:.18,i:3,period:126.6,radiusKm:2363,albedo:.148,color:'#c4b8a5',node:18,peri:62,mean:20},
        {id:'Venus-Sol',name:'Venus-Sol',kind:'planet',a:.92,e:.007,i:1.2,period:316,radiusKm:6050,albedo:.631,color:'#f5dda6',node:55,peri:110,mean:110},
        {id:'Terrax',name:'Terrax',kind:'planet',a:1.34,e:.0167,i:0,period:555.569,radiusKm:7016.8,albedo:null,color:'#5a9ccd',node:0,peri:283,mean:0,canonicalPeri:true},
        {id:'Mars-Sol',name:'Mars-Sol',kind:'planet',a:2.18,e:.08,i:1.8,period:3.156*EARTH_YEAR,radiusKm:3380,albedo:.158,color:'#eb9270',node:83,peri:145,mean:55},
        {id:'Jupiter-Sol',name:'Jupiter-Sol',kind:'planet',a:7.21736,e:.04,i:1,period:19.0130*EARTH_YEAR,radiusKm:73300,albedo:.487,color:'#dfc3a2',node:118,peri:35,mean:70},
        {id:'Saturn-Sol',name:'Saturn-Sol',kind:'planet',a:13.94,e:.05,i:1.6,period:51.03*EARTH_YEAR,radiusKm:54000,albedo:.521,color:'#dccc9d',node:164,peri:210,mean:240},
        {id:'Neptune-Sol',name:'Neptune-Sol',kind:'planet',a:27.38,e:.015,i:.8,period:140.49*EARTH_YEAR,radiusKm:25500,albedo:.422,color:'#86a9e8',node:223,peri:280,mean:180},
        {id:'Luna',name:'Luna',kind:'moon',a:384825.252/AU_KM,e:.006,i:.4,period:24.253,radiusKm:1771.36,albedo:.15,color:'#ddd3c3',node:40,peri:80,mean:195},
        {id:'Echo',name:'Echo',kind:'moon',a:779659.047/AU_KM,e:.012,i:1.1,period:70.258,radiusKm:781.47,albedo:.23,color:'#dbe0e7',node:190,peri:240,mean:210},
    ],
};
export const TERRAX = SYSTEM.bodies.find(b => b.id === 'Terrax');
export const MAX_PERIOD = Math.max(...SYSTEM.bodies.filter(b => b.kind === 'planet').map(b => b.period));
export const LOCAL_DAY = SYSTEM.observer.spinHours/24;
export const INITIAL_ANGLES = Object.fromEntries(SYSTEM.bodies.map(b => [b.id, {node:b.node, peri:b.peri, mean:b.mean}]));

export function solveKepler(mean, eccentricity) {
    if (!Number.isFinite(mean) || !Number.isFinite(eccentricity) || eccentricity < 0 || eccentricity >= 1) throw new Error('无效椭圆轨道');
    const m = mod(mean + Math.PI) - Math.PI;
    let e = eccentricity < .8 ? m : (m >= 0 ? Math.PI : -Math.PI);
    for (let k=0;k<24;k++) {
        const correction = (e-eccentricity*Math.sin(e)-m)/(1-eccentricity*Math.cos(e));
        e -= correction;
        if (Math.abs(correction)<1e-13) return e;
    }
    throw new Error('轨道求解未收敛');
}

export function orbitPosition(body, days, angles = INITIAL_ANGLES) {
    const a = angles[body.id] || body;
    const eccentricAnomaly = solveKepler(a.mean*DEG + TAU*mod(days, body.period)/body.period, body.e);
    let v = [body.a*(Math.cos(eccentricAnomaly)-body.e), body.a*Math.sqrt(1-body.e*body.e)*Math.sin(eccentricAnomaly), 0];
    v = rotateZ(rotateX(rotateZ(v, a.peri*DEG), body.i*DEG), a.node*DEG);
    // 行星相对黄道；卫星相对母星赤道。不能把两套倾角混用。
    return body.kind === 'moon' ? v : rotateX(v, SYSTEM.observer.obliquity*DEG);
}

export function equatorialDirection(ra, dec) {
    return [Math.cos(dec*DEG)*Math.cos(ra*DEG), Math.cos(dec*DEG)*Math.sin(ra*DEG), Math.sin(dec*DEG)];
}
export function spherical(v) {
    const u = unit(v);
    return {ra:mod(Math.atan2(u[1],u[0]))/DEG, dec:Math.asin(clamp(u[2],-1,1))/DEG};
}

// 与现有星表图片相同的近 J2000 银河朝向；此处不是行星轨道初相位。
const GAL_Z = equatorialDirection(192.85948,27.12825);
const GAL_Y = unit(cross(GAL_Z,equatorialDirection(266.4,-28.9)));
const GAL_X = unit(cross(GAL_Y,GAL_Z));
export function galacticToEquatorial(l,b) {
    const v = equatorialDirection(l,b);
    return add(add(scale(GAL_X,v[0]),scale(GAL_Y,v[1])),scale(GAL_Z,v[2]));
}

export function observerFrame(days, options = {}) {
    const angles = options.angles || INITIAL_ANGLES;
    const home = orbitPosition(TERRAX, days, angles);
    const initialSun = scale(orbitPosition(TERRAX,0,angles),-1);
    // 人为时间零点：默认零经线为午夜。26h按文档的恒星自转周期解释。
    const theta0 = Math.atan2(initialSun[1],initialSun[0]) + Math.PI;
    const latitude = (options.latitude ?? 30)*DEG;
    const theta = theta0 + TAU*mod(days,LOCAL_DAY)/LOCAL_DAY + ((options.longitude ?? 0)+(options.spinPhase ?? 0))*DEG;
    const cp=Math.cos(latitude),sp=Math.sin(latitude),ct=Math.cos(theta),st=Math.sin(theta);
    // 右手坐标：+X北，+Y西，+Z天顶。相机面北时，东在画面右边。
    const north=[-sp*ct,-sp*st,cp],west=[st,-ct,0],up=[cp*ct,cp*st,sp];
    const surface = options.mode === 'surface';
    const offset = surface ? scale(up,(SYSTEM.observer.radiusKm+SYSTEM.observer.heightMetres/1000)/AU_KM) : [0,0,0];
    return {home, position:add(home,offset), north,west,up,surface,theta,
        matrix:surface ? [north,west,up] : [[1,0,0],[0,1,0],[0,0,1]]};
}
export const applyFrame = (v,frame) => frame.matrix.map(row => dot(row,v));
export function apparentDirection(position, frame) {
    const relative=subtract(position,frame.position);
    const direction=unit(relative),view=applyFrame(direction,frame);
    return {distance:norm(relative),equatorial:direction,view,...spherical(direction),
        altitude:Math.asin(clamp(dot(frame.up,direction),-1,1))/DEG,
        azimuth:mod(Math.atan2(-dot(frame.west,direction),dot(frame.north,direction)))/DEG};
}
export function lambertPhase(angle) {
    if (angle <= 0) return 1;
    if (angle >= Math.PI) return 0;
    return Math.max(0,(Math.sin(angle)+(Math.PI-angle)*Math.cos(angle))/Math.PI);
}
export function reflectedMagnitude(body, sunDistance, observerDistance, phaseAngle) {
    const phi=lambertPhase(phaseAngle);
    if (phi<=1e-15) return Infinity;
    return SYSTEM.star.magnitudeAtReference + 5*Math.log10(sunDistance/TERRAX.a)
        -2.5*Math.log10(body.albedo*(body.radiusKm/(observerDistance*AU_KM))**2*phi);
}

export function skyState(days, options = {}) {
    if (!Number.isFinite(days)) throw new Error('时间必须是有限数值');
    const angles=options.angles || INITIAL_ANGLES,frame=observerFrame(days,options);
    const records=[];
    const sol={...SYSTEM.star,position:[0,0,0],...apparentDirection([0,0,0],frame),phaseAngle:0,illuminated:1};
    sol.magnitude=SYSTEM.star.magnitudeAtReference+5*Math.log10(sol.distance/TERRAX.a);
    sol.angularDiameter=2*Math.asin(SYSTEM.star.radiusKm/(sol.distance*AU_KM))/DEG;
    records.push(sol);
    for (const body of SYSTEM.bodies) {
        if (body.id==='Terrax') continue;
        const relative=orbitPosition(body,days,angles);
        const position=body.kind==='moon' ? add(frame.home,relative) : relative;
        const apparent=apparentDirection(position,frame);
        const toSun=scale(position,-1),toObserver=subtract(frame.position,position);
        const phaseAngle=Math.acos(clamp(dot(unit(toSun),unit(toObserver)),-1,1));
        records.push({...body,position,...apparent,phaseAngle,lightDirection:applyFrame(unit(toSun),frame),illuminated:(1+Math.cos(phaseAngle))/2,
            magnitude:reflectedMagnitude(body,norm(position),apparent.distance,phaseAngle),
            angularDiameter:2*Math.asin(body.radiusKm/(apparent.distance*AU_KM))/DEG});
    }
    const night=options.daylight && frame.surface ? clamp((-sol.altitude-2)/16,0,1) : 1;
    const limitingMagnitude=night*10.5-4;
    for (const body of records) {
        body.aboveHorizon=!frame.surface || body.altitude>=0;
        body.brightEnough=body.kind==='star' || body.magnitude <= limitingMagnitude;
        body.visible=body.aboveHorizon && body.brightEnough;
        body.status=!body.aboveHorizon ? '地平线下' : !body.brightEnough ? '亮度不足' : '可见';
    }
    return {days,frame,bodies:records,night,limitingMagnitude};
}

export function prepareBackground(stars) {
    const identifiers=new Set();
    return stars.map(star => {
        if (typeof star.id!=='string' || identifiers.has(star.id) || !Number.isFinite(star.app_mag)
            || !/^#[0-9a-fA-F]{6}$/.test(star.color_hex)) throw new Error('星表的标识、星等或颜色无效');
        identifiers.add(star.id);
        let direction;
        if (Number.isFinite(star.gal_lon) && star.gal_lon>=0 && star.gal_lon<360 && Number.isFinite(star.gal_lat) && Math.abs(star.gal_lat)<=90) direction=galacticToEquatorial(star.gal_lon,star.gal_lat);
        else if (star.gal_lon===undefined && star.gal_lat===undefined && Number.isFinite(star.ra) && star.ra>=0 && star.ra<360 && Number.isFinite(star.dec) && Math.abs(star.dec)<=90) direction=equatorialDirection(star.ra,star.dec);
        else throw new Error(`恒星 ${star.id} 的坐标无效`);
        const distance=star.distance_pc ?? star.dist_ly/3.26156;
        if (!Number.isFinite(distance) || distance<=0) throw new Error(`恒星 ${star.id} 的距离无效`);
        return {...star,baseDirection:direction,baseDistanceAU:distance*AU_PER_PC};
    });
}
export function backgroundDirection(star,frame) {
    // 以恒星系为原点的恒星位置，包含母星公转/地表位移造成的视差；没有编造恒星自行。
    const v=unit(star.baseDirection.map((x,i) => x-frame.position[i]/star.baseDistanceAU));
    return applyFrame(v,frame);
}

// 与PNG同族的方位等距投影。太阳、行星与恒星必须调用同一个函数。
export function projectHemisphere(direction, radius, centers, orientation=13.564125, hemisphere=null) {
    const u=unit(direction),north=hemisphere ?? u[2]>=0;
    const latitude=Math.asin(clamp(u[2],-1,1));
    const longitude=Math.atan2(u[1],u[0]);
    // 强制半球仅用于跨赤道的扩展圆盘；圆外部分由画布裁剪。
    const rho=radius*(1-(north?latitude:-latitude)/(Math.PI/2));
    const angle=longitude-orientation*DEG;
    const center=north?centers.north:centers.south;
    return {x:center[0]+(north?-1:1)*rho*Math.sin(angle),y:center[1]+rho*Math.cos(angle),north};
}
