import {DEG} from '../web/shared/solar_system.mjs';
import {kinds,sampleMoon,renderFixture,offsetDirection} from './sky_render_fixture.mjs';

export function runLunarOcclusionRenderAudit({test,assert,renderer}) {
    let samples=0,maxLeak=0;
    function compare(a,b,radius,outer=false) {
        let delta=0,count=0;
        for(let y=0;y<a.width;y++)for(let x=0;x<a.width;x++) {
            const r=Math.hypot(x+.5-a.width/2,y+.5-a.width/2);
            if(outer?r<radius+2 || r>radius*1.5:r>radius-2)continue;
            count++;const k=4*(y*a.width+x);for(let c=0;c<3;c++)delta=Math.max(delta,Math.abs(a.data[k+c]-b.data[k+c]));
        }
        if(!outer)assert(count>0,'未检查到盘面内部');return delta;
    }
    for(const kind of kinds)test(`${kind} 月面在不同星等、盈亏和增强开关下保持同一不透明几何边界`,()=>{
        const f=renderFixture(kind,renderer);
        for(const id of ['Luna','Echo'])for(const phase of [0,90,150,180])for(const magnitude of [-13,-5])for(const enhanced of [false,true])for(const dpr of [1,2]) {
            const b=sampleMoon(id,phase,undefined,magnitude),options={dpr,enhanced},r=b.angularDiameter*DEG/2*24000*dpr;
            const stars=[-.45,0,.45,1.3].map(x=>({view:offsetDirection(b.view,x*b.angularDiameter*DEG/2,0),app_mag:0,color_hex:'#ffffff'}));
            const blank=f.show([],{...options,focus:b.view}),witness=f.show([],{...options,focus:b.view,stars});
            const own=f.show([b],options),covered=f.show([b],{...options,stars});samples+=4;
            assert(compare(blank,witness,r)>20,'见证星不可见');
            const leak=compare(own,covered,r);maxLeak=Math.max(maxLeak,leak);assert(leak<=1,JSON.stringify({kind,id,phase,magnitude,enhanced,dpr,leak}));
            assert(compare(own,covered,r,true)>20,'盘外星被扩大月面遮住');
        }
        f.close();
    });
    for(const kind of kinds)test(`${kind} 双月、行星、太阳与光晕按距离遮挡，地平线下盘面被裁去`,()=>{
        const f=renderFixture(kind,renderer);
        for(const id of ['Luna','Echo'])for(const enhanced of [false,true]) {
            const b=sampleMoon(id,180),r=b.angularDiameter*DEG/2*24000;
            const far={...sampleMoon('Far',0),id:'Sol',kind:'star',distance:1.34,angularDiameter:1,magnitude:-26.5};
            const own=f.show([b],{enhanced}),back=f.show([b,far],{enhanced,solarGlow:true});samples+=2;
            assert(compare(own,back,r)<=1,'太阳或光晕穿透暗月');
            const near={...sampleMoon('Venus-Sol',0),kind:'planet',distance:.0001,angularDiameter:.08,color:'#ff6040'};
            const front=f.show([b,near]),reverse=f.show([near,b]);samples+=2;
            assert(compare(own,front,r)>30,'前方行星不可见');assert(front.data.every((v,i)=>v===reverse.data[i]),'输入顺序影响遮挡');
            const horizon=sampleMoon(id,0,[1,0,0]),sky=f.show([horizon],{surface:true}),empty=f.show([],{focus:horizon.view,surface:true});samples+=2;
            if(kind!=='Canvas')for(let y=sky.width/2+2;y<sky.width;y++)for(let x=0;x<sky.width;x++) {
                const k=4*(y*sky.width+x);assert(sky.data[k]===empty.data[k],'月面越过地平');
            }
            else {
                const below=f.show([horizon],{surface:true,north:false}),blank=f.show([],{focus:horizon.view,surface:true,north:false});samples+=2;
                assert(below.data.every((v,i)=>v===blank.data[i]),'下半球仍有实体盘面');
                const north=f.show([horizon],{north:true}),south=f.show([horizon],{north:false});samples+=2;
                assert(north.data.some((v,i)=>v>blank.data[i]+40) && south.data.some((v,i)=>v>blank.data[i]+40),'赤道两侧盘面不完整');
            }
        }
        f.close();
    });
    Object.assign(document.querySelector('#status').dataset,{occlusionRasterSamples:samples,occlusionMaxInteriorLeak:maxLeak});
    renderer.setPixelRatio(1);renderer.setSize(256,256);
}
