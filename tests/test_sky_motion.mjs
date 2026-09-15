import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import * as s from '../web/shared/solar_system.mjs';
import {DEFAULT_STATE,sanitizeState,timeAt,changeState,SkyClock,STORAGE_KEY} from '../web/v1/sky_state.mjs';
import {displaySky,DEFAULT_DISPLAY_SCALE,pointRadius,pointOpacity,backgroundVisible,diskBasis,diskLight,solarGlareVisibility,bodyLightVisibility,lunarSurfaceScale,atlasSymbolScale,SYMBOL_REFERENCE,trajectory,trajectoryDirections,perspectiveSymbolScale} from '../web/shared/sky_render.mjs';
const near=(a,b,tolerance=1e-10)=>assert.ok(Math.abs(a-b)<=tolerance,`${a} != ${b} ± ${tolerance}`);
const vnear=(a,b,tolerance=1e-10)=>a.forEach((x,i)=>near(x,b[i],tolerance));
const angle=(a,b)=>Math.atan2(s.norm(s.cross(a,b)),s.dot(a,b));

test('两页在相同角尺度下使用一致符号比例；月面增强不决定几何大小',()=>{
    for(const height of [128,256,512,1000])for(const fov of [1,10,30,60,100]) {
        const focal=height/(2*Math.tan(fov*s.DEG/2));
        near(atlasSymbolScale(focal*Math.PI/2),perspectiveSymbolScale(height,fov));
    }
    const bodies=s.skyState(3928.342761).bodies;
    for(const b of bodies.filter(b=>b.kind==='moon')) {
        const before=JSON.stringify(b);
        for(const magnitude of [-13,-9,7])for(const enhanced of [false,true]) {
            const body={...b,magnitude};assert.ok(lunarSurfaceScale(body,enhanced)>=1);near(body.angularDiameter,b.angularDiameter);
        }
        assert.equal(JSON.stringify(b),before);
    }
    const echo=bodies.find(b=>b.id==='Echo');assert.ok(pointRadius(-4)/SYMBOL_REFERENCE.perspectiveFocal<echo.angularDiameter*s.DEG/2);
    assert.equal(sanitizeState().projection,'stereographic');assert.equal(sanitizeState({projection:'perspective'}).projection,'perspective');
});

test('最大时长取最慢行星的周期，使用地球日',()=>{
    near(s.MAX_PERIOD,51313.9725,1e-8);near(s.MAX_PERIOD/s.TERRAX.period,92.36291532,1e-8);
    near(s.LOCAL_DAY,26/24);near(24/(1/s.LOCAL_DAY-1/s.TERRAX.period),26.050797826,1e-8);
});
test('Kepler 方程在零、负角、高偏心率和多圈后收敛',()=>{
    for(const eccentricity of [0,.006,.0167,.18,.7,.95,.999])for(const mean of [-1000,-Math.PI,-.01,0,.001,2,Math.PI,10000]) {
        const e=s.solveKepler(mean,eccentricity);near(s.mod(e-eccentricity*Math.sin(e)-mean+Math.PI)-Math.PI,0,2e-10);
    }
    for(const args of [[0,NaN],[NaN,0],[0,-1],[0,1]])assert.throws(()=>s.solveKepler(...args));
});
test('近远星点距离与椭圆几何一致，各自一圈闭合',()=>{
    for(const body of s.SYSTEM.bodies) {
        const angles={...s.INITIAL_ANGLES,[body.id]:{node:12,peri:33,mean:0}};
        near(s.norm(s.orbitPosition(body,0,angles)),body.a*(1-body.e));
        near(s.norm(s.orbitPosition(body,body.period/2,angles)),body.a*(1+body.e));
        vnear(s.orbitPosition(body,3.12,angles),s.orbitPosition(body,3.12+body.period,angles),2e-11);
    }
});
test('轨道满足面积定律，近星点处移动更快',()=>{
    const body=s.SYSTEM.bodies[0],angles={...s.INITIAL_ANGLES,[body.id]:{node:0,peri:0,mean:0}},h=1e-4;
    const angularMomentum=t=>{const p=s.orbitPosition(body,t,angles),v=s.scale(s.subtract(s.orbitPosition(body,t+h,angles),s.orbitPosition(body,t-h,angles)),1/(2*h));return s.norm(s.cross(p,v));};
    near(angularMomentum(0),angularMomentum(body.period/2),2e-10);
    assert.ok(s.norm(s.subtract(s.orbitPosition(body,h,angles),s.orbitPosition(body,0,angles))) > s.norm(s.subtract(s.orbitPosition(body,body.period/2+h,angles),s.orbitPosition(body,body.period/2,angles))));
});
test('文档周期与独立引力计算相符；卫星计入自身质量',()=>{
    const G=6.67430e-11,msun=1.98847e30,au=149597870700;
    for(const b of s.SYSTEM.bodies) {
        const mass=b.kind==='moon'?s.SYSTEM.observer.massKg+(b.id==='Luna'?7.56226e22:6.09386e21):1.04*msun;
        const period=2*Math.PI*Math.sqrt((b.a*au)**3/(G*mass))/86400;
        assert.ok(Math.abs(period/b.period-1)<.0005,`${b.id}: ${period}`);
    }
});
test('行星黄道面旋转25度，卫星轨道倾角使用赤道面',()=>{
    const planet=s.orbitPosition({...s.TERRAX,e:0},s.TERRAX.period/4,{Terrax:{node:0,peri:0,mean:0}});
    near(s.spherical(planet).dec,25);
    for(const moon of s.SYSTEM.bodies.filter(b=>b.kind==='moon'))for(let i=0;i<24;i++) {
        assert.ok(Math.abs(s.spherical(s.orbitPosition(moon,moon.period*i/24)).dec)<=moon.i+1e-10);
    }
});
test('观察矩阵正交、右手、极地有效；所有天体保持单位方向',()=>{
    for(const latitude of [-90,-30,0,30,90])for(const days of [0,12.7,s.MAX_PERIOD]) {
        const sky=s.skyState(days,{mode:'surface',latitude,longitude:45});
        for(const row of sky.frame.matrix)near(s.norm(row),1);
        vnear(s.cross(sky.frame.north,sky.frame.west),sky.frame.up);
        near(s.dot(sky.frame.north,sky.frame.west),0);
        for(const b of sky.bodies) {near(s.norm(b.view),1);assert.ok([b.distance,b.angularDiameter,b.magnitude,b.phaseAngle].every(Number.isFinite));}
    }
});
test('午夜约定、向东经度、自转26小时及地平四向',()=>{
    const a=s.observerFrame(0,{mode:'surface',latitude:0}),b=s.observerFrame(s.LOCAL_DAY,{mode:'surface',latitude:0});
    vnear(a.up,b.up);near(s.mod(a.theta-Math.atan2(-a.home[1],-a.home[0])),Math.PI);
    const east=s.observerFrame(0,{mode:'surface',latitude:0,longitude:90});vnear(east.up,s.scale(a.west,-1));
    vnear(s.applyFrame(a.north,a),[1,0,0]);vnear(s.applyFrame(s.scale(a.west,-1),a),[0,-1,0]);vnear(s.applyFrame(a.up,a),[0,0,1]);
});
test('恒星从东升起向西移动，26小时重复自转方位',()=>{
    const frame=s.observerFrame(0,{mode:'surface',latitude:0}),east=s.scale(frame.west,-1);
    const later=s.observerFrame(s.LOCAL_DAY/4,{mode:'surface',latitude:0});assert.ok(s.applyFrame(east,later)[2]>.999);
    const setting=s.observerFrame(s.LOCAL_DAY/2,{mode:'surface',latitude:0});assert.ok(s.applyFrame(east,setting)[1]>.999);
});
test('地表视差包含母星半径，月球的视差显著大于远行星',()=>{
    const c=s.skyState(10,{mode:'center'}),g=s.skyState(10,{mode:'surface'});
    near(s.norm(s.subtract(c.frame.position,g.frame.position))*s.AU_KM,7016.80175,1e-7);
    const parallax=id=>angle(c.bodies.find(b=>b.id===id).equatorial,g.bodies.find(b=>b.id===id).equatorial);
    assert.ok(parallax('Luna')>parallax('Neptune-Sol')*1000);
});
test('背景恒星保留方向和距离，公转视差量级为 AU / pc',()=>{
    const f=s.observerFrame(0,{mode:'center'}),base=s.unit(s.cross(f.position,[0,0,1]));
    const star={baseDirection:base,baseDistanceAU:s.AU_PER_PC};
    near(angle(s.backgroundDirection(star,f),base),Math.atan(s.norm(f.position)/s.AU_PER_PC),1e-14);
});
test('两颗卫星的标准满相星等和角直径匹配参考',()=>{
    for(const [id,mag,diameter] of [['Luna',-12.75,.5275],['Echo',-9.90,.1149]]) {
        const b=s.SYSTEM.bodies.find(b=>b.id===id);
        near(s.reflectedMagnitude(b,s.TERRAX.a,b.a,0),mag,.006);
        near(2*Math.asin(b.radiusKm/(b.a*s.AU_KM))/s.DEG,diameter,.0002);
    }
});
test('所有天体的角径与参考半径和独立切线几何一致，Luna 大于 Sol 大于 Echo',()=>{
    const radii={Sol:758313,'Mercury-Sol':2363,'Venus-Sol':6050,'Mars-Sol':3380,'Jupiter-Sol':73300,'Saturn-Sol':54000,'Neptune-Sol':25500,Luna:1771.36,Echo:781.47};
    for(const mode of ['center','surface'])for(const days of [0,10,3928.342761,25000,s.MAX_PERIOD]) {
        const sky=s.skyState(days,{mode,latitude:65,longitude:85}),diameters={};
        for(const body of sky.bodies) {
            const r=radii[body.id],d=body.distance*149597870.7;
            near(body.angularDiameter,2*Math.atan(r/Math.sqrt(d*d-r*r))*180/Math.PI,1e-10);
            diameters[body.id]=body.angularDiameter;
        }
        assert.ok(diameters.Luna>diameters.Sol && diameters.Sol>diameters.Echo);
        for(const id of Object.keys(radii).filter(id=>id.endsWith('-Sol')))assert.ok(diameters.Echo>diameters[id]*8);
    }
});
test('Neptune 冲日仍超6.5等，辅助标记不能改变光度',()=>{
    const b=s.SYSTEM.bodies.find(b=>b.id==='Neptune-Sol');near(s.reflectedMagnitude(b,b.a,b.a-s.TERRAX.a,0),6.91,.01);
    for(let d=0;d<s.MAX_PERIOD;d+=503)assert.equal(s.skyState(d,{mode:'center',markers:true}).bodies.find(b=>b.id==='Neptune-Sol').visible,false);
});
test('Lambert光通量与照明面积不同，距离平方衰减正确',()=>{
    near(s.lambertPhase(0),1);near(s.lambertPhase(Math.PI/2),1/Math.PI);near(s.lambertPhase(Math.PI),0);
    const b=s.SYSTEM.bodies[0],m=s.reflectedMagnitude(b,1,2,0);
    near(s.reflectedMagnitude(b,2,2,0)-m,5*Math.log10(2));near(s.reflectedMagnitude(b,1,4,0)-m,5*Math.log10(2));
    assert.equal(s.reflectedMagnitude(b,1,2,Math.PI),Infinity);
});
test('Sol 的实际角直径始终远大于 Venus，光晕计算不修改天体物理量',()=>{
    for(let day=0;day<=s.MAX_PERIOD;day+=211) {
        const sky=s.skyState(day),sol=sky.bodies.find(b=>b.id==='Sol'),venus=sky.bodies.find(b=>b.id==='Venus-Sol'),before=JSON.stringify(sky);
        assert.ok(sol.angularDiameter>30*venus.angularDiameter);
        const light=solarGlareVisibility(sky);assert.ok(light>=0 && light<=1);assert.equal(JSON.stringify(sky),before);
    }
});
test('太阳光晕随地平和前景遮挡衰减，重叠遮挡按并集计算',()=>{
    const sol={id:'Sol',view:[1,0,0],distance:1,angularDiameter:.44},moon={id:'Luna',view:[1,0,0],distance:.01,angularDiameter:.5};
    const sky=(bodies,surface=false)=>({frame:{surface},bodies:[sol,...bodies]});
    near(solarGlareVisibility(sky([])),1);near(solarGlareVisibility(sky([],true)),.5);
    near(solarGlareVisibility(sky([moon])),0);
    near(solarGlareVisibility(sky([{...moon,distance:2}])),1);
    near(solarGlareVisibility(sky([{...moon,view:[0,1,0]}])),1);
    const partial={...moon,angularDiameter:.44,view:s.equatorialDirection(.22,0)};
    const fraction=solarGlareVisibility(sky([partial]));near(fraction,1-(2*Math.PI/3-Math.sqrt(3)/2)/Math.PI,.01);
    near(solarGlareVisibility(sky([partial,{...partial,id:'Echo'}])),fraction);
    const below={...sol,view:s.equatorialDirection(0,-1)};
    near(solarGlareVisibility({frame:{surface:true},bodies:[below]}),0);
});
test('双月柔光的遮挡按亮面加权，挡住暗半球不等于挡住亮半球',()=>{
    const moon={id:'Echo',kind:'moon',view:[1,0,0],distance:.5,angularDiameter:.44,lightDirection:[0,-1,0],brightEnough:true};
    const blocker={id:'Luna',view:[1,0,0],distance:.1,angularDiameter:.44};
    const sky=b=>({frame:{surface:false},bodies:[moon,b]});
    near(bodyLightVisibility(moon,sky(blocker)),0);
    const dark=bodyLightVisibility(moon,sky({...blocker,view:s.equatorialDirection(.22,0)}));
    const bright=bodyLightVisibility(moon,sky({...blocker,view:s.equatorialDirection(-.22,0)}));
    assert.ok(dark>bright+.3,`${dark} vs ${bright}`);
    near(bodyLightVisibility({...moon,brightEnough:false},sky(blocker)),0);
});
test('月亮相位亮面方向与太阳方向一致',()=>{
    for(const day of [0,10,55,500])for(const b of s.skyState(day).bodies.filter(b=>b.id!=='Sol')) {
        const basis=diskBasis(b.view),light=diskLight(b,basis);
        vnear(s.cross(basis.x,basis.y),basis.z);near(light[2],Math.cos(b.phaseAngle));near(s.norm(light),1);
    }
});
test('白昼和地平筛选生效；质心全天不受它们影响',()=>{
    const night=s.skyState(0,{mode:'surface',daylight:true}),day=s.skyState(s.LOCAL_DAY/2,{mode:'surface',daylight:true});
    near(night.limitingMagnitude,6.5);near(day.limitingMagnitude,-4);
    const star={app_mag:6.5};assert.equal(backgroundVisible(star,[0,0,1],night),true);assert.equal(backgroundVisible(star,[0,0,-1],night),false);assert.equal(backgroundVisible(star,[0,0,1],day),false);
    const center=s.skyState(s.LOCAL_DAY/2,{mode:'center',daylight:true});assert.equal(backgroundVisible(star,[0,0,-1],center),true);
});
test('双半球投影的极点、圆周和赤道唯一归属',()=>{
    const centers={north:[100,100],south:[350,100]},r=80;
    assert.deepEqual(s.projectHemisphere([0,0,1],r,centers,0),{x:100,y:100,north:true});
    assert.deepEqual(s.projectHemisphere([0,0,-1],r,centers,0),{x:350,y:100,north:false});
    for(let i=0;i<360;i+=15) {const p=s.projectHemisphere(s.equatorialDirection(i,0),r,centers,0);assert.ok(p.north);near(Math.hypot(p.x-100,p.y-100),r);}
});
test('地平仰视图北上、东左，星点符号随星等单调变化',()=>{
    const centers={north:[0,0],south:[300,0]},r=100;
    const n=s.projectHemisphere([1,0,0],r,centers,180),e=s.projectHemisphere([0,-1,0],r,centers,180);
    near(n.x,0);near(n.y,-r);near(e.x,-r);near(e.y,0);
    let previous=Infinity;for(let m=-2;m<6.51;m+=.1) {assert.ok(pointRadius(m)<=previous);previous=pointRadius(m);assert.ok(pointOpacity(m)>0);}
});
test('背景星等在实际星表范围内连续区分大小，暗星没有相同半径平台',()=>{
    let previous=Infinity;
    for(let m=-3;m<=6.5;m+=.1) {
        const radius=pointRadius(m),intensity=radius*radius*pointOpacity(m);
        assert.ok(radius>0 && radius<previous,'星等 '+m+' 落入相同半径平台');
        assert.ok(intensity>0);previous=radius;
    }
    assert.ok(pointRadius(6)/pointRadius(6.5)>1.1);
    assert.ok(pointRadius(2)>pointRadius(6.5)*2);
    assert.ok(pointOpacity(5)>pointOpacity(6.5)*1.5);
});
test('跨赤道盘面可投影到两侧，不能把圆盘另一半丢失',()=>{
    const centers={north:[0,0],south:[300,0]},u=s.equatorialDirection(60,-.1),r=100;
    const n=s.projectHemisphere(u,r,centers,0,true),south=s.projectHemisphere(u,r,centers,0,false);
    assert.ok(Math.hypot(n.x,n.y)>r);assert.ok(Math.hypot(south.x-300,south.y)<r);
});
test('一个最慢周期后仅该行星日心轨道闭合，地心视向不伪闭合',()=>{
    const body=s.SYSTEM.bodies.find(b=>b.id==='Neptune-Sol');vnear(s.orbitPosition(body,0),s.orbitPosition(body,s.MAX_PERIOD));
    const first=s.skyState(0,{mode:'center'}),last=s.skyState(s.MAX_PERIOD,{mode:'center'});
    assert.ok(angle(first.bodies.find(b=>b.id===body.id).view,last.bodies.find(b=>b.id===body.id).view)>1e-3);
});
test('外行星由母星相对运动自然产生逆行',()=>{
    let forward=false,retrograde=false;
    const longitude=d=>{const v=s.skyState(d,{mode:'center'}).bodies.find(b=>b.id==='Mars-Sol').equatorial;return Math.atan2(...s.rotateX(v,-25*s.DEG).slice(0,2).reverse());};
    for(let d=0;d<1500;d+=3) {const change=s.mod(longitude(d+1)-longitude(d)+Math.PI)-Math.PI;forward ||=change>0;retrograde ||=change<0;}
    assert.ok(forward && retrograde);
});
test('轨迹只采样模拟时域，年轨迹扣除自转而日轨迹保留',()=>{
    for(const kind of ['day','year'])for(const day of [0,s.MAX_PERIOD]) {
        const points=trajectory(day,{mode:'surface'},'Luna',kind);assert.equal(points.length,257);
        assert.ok(points.every(p=>p.time>=0 && p.time<=s.MAX_PERIOD+1e-8));
        const sample=points[128],body=s.skyState(sample.time,{mode:'surface'}).bodies.find(b=>b.id==='Luna');
        vnear(sample.direction,kind==='day'?body.view:s.applyFrame(body.equatorial,s.observerFrame(day,{mode:'surface'})));
    }
});
test('两个页面读取相同时间锚点，不重复累加',()=>{
    let state=sanitizeState({days:10,anchor:1000,playing:true,speed:2},1000);
    near(timeAt(state,6000),20);near(timeAt(JSON.parse(JSON.stringify(state)),6000),20);
    state=changeState(state,{speed:4},6000);near(timeAt(state,7000),24);
    state=changeState(state,{playing:false},7000);near(timeAt(state,100000),24);
});
test('终点暂停、数值输入边界、非法状态与参考角保护',()=>{
    const state=sanitizeState({days:s.MAX_PERIOD-1,anchor:0,speed:2,playing:true});near(timeAt(state,2000),s.MAX_PERIOD);
    assert.equal(changeState(state,{},2000).playing,false);
    const bad=sanitizeState({days:-20,latitude:300,longitude:-900,speed:NaN,mode:'invalid',angles:{Terrax:{node:120,peri:0,mean:-90}}});
    assert.equal(bad.days,0);assert.equal(bad.latitude,90);assert.equal(bad.longitude,-180);assert.equal(bad.mode,'center');
    assert.deepEqual(bad.angles.Terrax,{node:0,peri:283,mean:270});
});
test('可编辑轨道角改变起始几何，不改变轨道形状或周期',()=>{
    const state=sanitizeState(),edited=sanitizeState({...state,angles:{...state.angles,'Luna':{node:90,peri:150,mean:280}}});
    const first=s.skyState(0,state),last=s.skyState(0,edited);
    assert.ok(angle(first.bodies.find(b=>b.id==='Luna').view,last.bodies.find(b=>b.id==='Luna').view)>.1);
    assert.equal(s.SYSTEM.bodies.find(b=>b.id==='Luna').period,24.253);assert.equal(s.INITIAL_ANGLES.Luna.mean,195);
});
test('默认全天在起点、中点、终点均显示9356颗，两半球完整且投影落在各自圆内',()=>{
    const data=JSON.parse(readFileSync(new URL('../output/output_20260915_galactic_01/sky_view_20260915_galactic_01.json',import.meta.url)));
    const stars=s.prepareBackground([...(data.neighbors||[]),...data.stars]);assert.equal(stars.length,9356);
    assert.ok(stars.every(star=>Math.abs(s.norm(star.baseDirection)-1)<1e-12 && star.app_mag<=6.5));
    for(const days of [0,s.LOCAL_DAY/2,s.MAX_PERIOD/2,s.MAX_PERIOD])for(const options of [undefined,sanitizeState()]) {
        const sky=s.skyState(days,options),counts={north:0,south:0},centers={north:[0,0],south:[300,0]};
        for(const star of stars) {
            const direction=s.backgroundDirection(star,sky.frame);
            if(!backgroundVisible(star,direction,sky))continue;
            const point=s.projectHemisphere(direction,100,centers),hemisphere=point.north?'north':'south',center=centers[hemisphere];
            counts[hemisphere]++;
            assert.ok(Math.hypot(point.x-center[0],point.y-center[1])<=100+1e-10);
        }
        assert.equal(counts.north+counts.south,9356);assert.ok(counts.north>0 && counts.south>0);
    }
    assert.throws(()=>s.prepareBackground([data.stars[0],data.stars[0]]));assert.throws(()=>s.prepareBackground([{...data.stars[0],gal_lon:500}]));assert.throws(()=>s.prepareBackground([{...data.stars[0],distance_pc:-1}]));
});
test('缓存的年轨迹在半日后仍与当前恒星背景一致',()=>{
    const points=trajectory(100,{mode:'surface'},'Luna','year'),later=s.observerFrame(100.5,{mode:'surface'}),directions=trajectoryDirections(points,later,'year');
    for(let i=0;i<points.length;i+=32) {
        const body=s.skyState(points[i].time,{mode:'surface'}).bodies.find(b=>b.id==='Luna');vnear(directions[i],s.applyFrame(body.equatorial,later));
    }
    assert.ok(angle(directions[128],points[128].direction)>2);
});
test('新用户与无效模式均默认全天，地表遮挡必须显式选择',()=>{
    const saved=new Map(),storage={getItem:key=>saved.get(key)??null,setItem:(key,value)=>saved.set(key,value)};
    const clock=new SkyClock({storage,eventTarget:{addEventListener(){}},id:'new',now:()=>1000});
    assert.equal(DEFAULT_STATE.mode,'center');assert.equal(clock.state.mode,'center');
    for(const options of [undefined,{},sanitizeState(),clock.state,{mode:'invalid'}]) {
        const sky=s.skyState(0,options);
        assert.equal(sky.frame.surface,false);assert.equal(backgroundVisible({app_mag:6.5},[0,0,-1],sky),true);
    }
    clock.set({mode:'surface'});assert.equal(s.skyState(0,clock.state).frame.surface,true);
});
test('旧地表记录只迁移观察方式；保留时间、初值与星表，后续地表选择跨刷新保留',()=>{
    const legacyKey='terrax-sky-motion-v1';
    const state=sanitizeState({mode:'surface',days:1234.5,anchor:1000,playing:true,speed:2,latitude:-45,longitude:87,spinPhase:65,daylight:false,markers:false,grid:false,trail:'year',selected:'Echo',folder:'output_20260915_galactic_01',angles:{...s.INITIAL_ANGLES,Luna:{node:30,peri:75,mean:40}}},1000);
    const versions={time:[800,'legacy'],mode:[801,'legacy'],angles:[802,'legacy']};
    for(const legacy of [state,{state,versions}]) {
        const raw=JSON.stringify(legacy),saved=new Map([[legacyKey,raw]]);
        const storage={getItem:key=>saved.get(key)??null,setItem:(key,value)=>saved.set(key,value)};
        const create=id=>new SkyClock({storage,eventTarget:{addEventListener(){}},id,now:()=>6000});
        const clock=create('first');
        assert.deepEqual(clock.state,{...state,mode:'center'});near(clock.time(),1244.5);
        assert.equal(saved.get(legacyKey),raw);assert.equal(JSON.parse(saved.get(STORAGE_KEY)).state.mode,'center');
        if(legacy.versions)for(const key of Object.keys(versions))assert.deepEqual(clock.record.versions[key],versions[key]);
        assert.deepEqual(create('second').state,clock.state);
        clock.set({mode:'surface'});
        assert.deepEqual(create('reload').state,{...state,mode:'surface'});
    }
});
test('真实 SkyClock 同时改不同字段、旧事件乱序交付后两页收敛',()=>{
    let saved=null,now=1000;const queue=[],listeners={},staleReads=new Map();
    const create=id=>new SkyClock({id,now:()=>now,eventTarget:{addEventListener:(type,fn)=>listeners[id]=fn},storage:{getItem:()=>{if(staleReads.has(id)){const value=staleReads.get(id);staleReads.delete(id);return value;}return saved;},setItem:(key,value)=>{saved=value;for(const other of Object.keys(listeners))if(other!==id)queue.push({id:other,event:{key,newValue:value}});}}});
    const a=create('A'),b=create('B');staleReads.set('B',saved);a.set({latitude:45});b.set({longitude:60});
    assert.equal(JSON.parse(saved).state.latitude,30); // 强制模拟另一页先读后写、写入时丢失 A 的字段。
    let deliveries=0;while(queue.length) {const next=queue.pop();listeners[next.id](next.event);assert.ok(++deliveries<20);}
    assert.equal(a.state.latitude,45);assert.equal(a.state.longitude,60);assert.deepEqual(a.state,b.state);
    a.set({playing:true,speed:2});now+=5000;b.set({markers:false});
    while(queue.length) {const next=queue.shift();listeners[next.id](next.event);}
    near(a.time(),10);near(b.time(),10);assert.equal(a.state.markers,false);
    const old={key:STORAGE_KEY,newValue:saved};now+=1000;b.set({days:42,playing:false});
    listeners.B(old);while(queue.length) {const next=queue.shift();listeners[next.id](next.event);}
    near(a.time(),42);near(b.time(),42);assert.deepEqual(a.state,b.state);
});

test('V1 默认统一增益 6 倍，保持物理数据并缩放同一切平面尺寸',()=>{
    assert.equal(sanitizeState().displayScale,6);assert.equal(DEFAULT_DISPLAY_SCALE,6);
    const original=s.skyState(0),before=JSON.stringify(original);
    for(const gain of [1,1.6,6,8]){const shown=displaySky(original,gain);
        for(let i=0;i<shown.bodies.length;i++){const a=original.bodies[i],b=shown.bodies[i];near(Math.tan(b.angularDiameter*s.DEG/2)/Math.tan(a.angularDiameter*s.DEG/2),gain,1e-12);near(b.magnitude,a.magnitude);}
    }assert.equal(JSON.stringify(original),before);assert.equal(sanitizeState({displayScale:Infinity}).displayScale,6);assert.equal(sanitizeState({displayScale:99}).displayScale,8);
});
