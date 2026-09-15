import {skyState} from '../web/shared/solar_system.mjs';
import {DEFAULT_DISPLAY_SCALE} from '../web/shared/sky_render.mjs';
import {kinds,renderFixture,pixelMetrics} from './sky_render_fixture.mjs';

export function runZoomRenderAudit({test,assert,renderer}) {
    const sources=[...[0,4,6.5].map(app_mag=>({id:'background-'+app_mag,app_mag,color_hex:'#ffffff'})),...skyState(3928.342761).bodies.map(b=>({...b,magnitude:Math.min(b.magnitude,6),visible:true,brightEnough:true}))];
    let samples=0,maxError=0,crossError=0;
    const render=(f,source,focal,dpr)=>{
        const view=[0,0,1],background='app_mag' in source,body={...source,view,lightDirection:[0,0,-1]};
        const p=f.show(background?[]:[body],{focus:view,focal,dpr,enhanced:true,solarGlow:true,planetGlow:true,stars:background?[body]:[]});samples++;return pixelMetrics(p,source.id==='Sol'?16:2);
    };
    for(const kind of kinds)test(`${kind} 背景星、主恒星、六行星与双月随共同角尺度缩放`,()=>{
        const f=renderFixture(kind,renderer);
        for(const source of sources)for(const dpr of [1,2]) {
            const reference=render(f,source,12800,dpr);
            assert(reference.count>0,source.id+' 最大倍率未留下像素');
            for(const focal of [400,1600,6400]) {
                const a=render(f,source,focal,dpr);
                for(const key of ['width','height']) {
                    const error=Math.abs(a[key]-reference[key]*focal/12800);maxError=Math.max(maxError,error);
                    assert(error<=3,JSON.stringify({kind,id:source.id,dpr,focal,key,error,a,reference}));
                }
            }
        }
        f.close();
    });
    test('两页及两种沉浸投影在相同中心角尺度和 DPR 下，直接配对实际尺寸与星光强度',()=>{
        const fixtures=kinds.map(kind=>renderFixture(kind,renderer));
        for(const source of sources)for(const focal of [800,6400,12800])for(const dpr of [1,2]) {
            const a=fixtures.map(f=>render(f,source,focal,dpr));
            for(const b of a.slice(1)) {
                for(const key of ['width','height']) {const error=Math.abs(b[key]-a[0][key]);crossError=Math.max(crossError,error);assert(error<=2,JSON.stringify({id:source.id,focal,dpr,key,a}));}
                if('app_mag' in source)assert(Math.abs(b.sum-a[0].sum)<=Math.max(12,.22*Math.max(a[0].sum,b.sum)),JSON.stringify({id:source.id,focal,dpr,a}));
            }
        }
        fixtures.forEach(f=>f.close());
    });
    test(`默认 ${DEFAULT_DISPLAY_SCALE} 倍显示增益在三种绘制中统一放大日月、行星和背景星`,()=>{
        const fixtures=kinds.map(kind=>renderFixture(kind,renderer));
        for(const source of sources)for(const dpr of [1,2]){
            const view=[0,0,1],background='app_mag' in source,body={...source,view,lightDirection:[0,0,-1]},all=[];
            for(const fixture of fixtures){
                const options={focal:2200,dpr,solarGlow:true,planetGlow:true,stars:background?[body]:[]};
                const physical=b=>JSON.stringify([b.angularDiameter,b.magnitude,b.distance,b.view,b.lightDirection]),original=physical(body),a=pixelMetrics(fixture.show(background?[]:[body],{...options,focal:options.focal*DEFAULT_DISPLAY_SCALE,gain:1}),source.id==='Sol'?16:2),b=pixelMetrics(fixture.show(background?[]:[body],{...options,gain:DEFAULT_DISPLAY_SCALE}),source.id==='Sol'?16:2);
                assert(physical(body)===original,'显示倍率修改了物理天体');
                // Compare to an equivalent camera enlargement, not six times
                // a quantized, subpixel footprint. Both samples fit the frame.
                for(const key of ['width','height']){assert(b[key]<384*dpr-4,'增益样本被视口裁剪');assert(Math.abs(b[key]-a[key])<=2,JSON.stringify({kind:fixture.kind,source:source.id,dpr,a,b}));}
                all.push(b);samples+=2;
            }
            for(const b of all.slice(1))for(const key of ['width','height'])assert(Math.abs(b[key]-all[0][key])<=2,JSON.stringify({source:source.id,dpr,all}));
        }fixtures.forEach(f=>f.close());
    });
    Object.assign(document.querySelector('#status').dataset,{zoomRasterSamples:samples,zoomMaxPixelError:maxError,crossViewMaxPixelError:crossError});
    renderer.setPixelRatio(1);renderer.setSize(256,256);
}
