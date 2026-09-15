import {DEG,equatorialDirection} from '../web/shared/solar_system.mjs';
import {kinds,sampleMoon,renderFixture,pixelMetrics,offsetDirection} from './sky_render_fixture.mjs';

export function runLunarPhaseRenderAudit({test,assert,renderer}) {
    let samples=0;
    for(const kind of kinds)test(`${kind} 双月按实际角径显示盈亏，增强只改变表面明暗`,()=>{
        const f=renderFixture(kind,renderer);
        for(const dpr of [1,2])for(const id of ['Luna','Echo'])for(const enhanced of [false,true]) {
            let previous=Infinity;
            for(const phase of [0,60,90,120,150,180]) {
                const body=sampleMoon(id,phase),p=f.show([body],{dpr,enhanced}),a=pixelMetrics(p);samples++;
                assert(a.sum<previous,JSON.stringify({kind,id,dpr,phase,enhanced,a,previous}));previous=a.sum;
                if(phase===0){const expected=body.angularDiameter*DEG*p.focal*dpr;assert(Math.abs(a.width-expected)<3 && Math.abs(a.height-expected)<3,JSON.stringify({kind,id,a,expected}));}
                if(phase===90) {
                    const axis=p.locate(offsetDirection(body.view,1e-5,0)),dx=axis[0]-p.width/2,dy=axis[1]-p.height/2;
                    const mx=a.x-p.width/2,my=a.y-p.height/2;
                    assert((mx*dx+my*dy)/Math.hypot(mx,my)/Math.hypot(dx,dy)>.99,'亮面背向太阳');
                }
                if(kind!=='Canvas')assert(Math.abs(f.motion.disks.get(id).mesh.scale.x-1800*Math.tan(body.angularDiameter*DEG/2))<1e-10,'真实盘面被增强放大');
                if(phase===180)assert(a.sum===0,'新月出现了亮圈');
            }
        }
        for(const dec of [90,45,-45,-90]) {
            const b=sampleMoon('Echo',90,equatorialDirection(37,dec)),p=f.show([b]),a=pixelMetrics(p),q=p.locate(offsetDirection(b.view,1e-5,0));samples++;
            const dx=q[0]-p.width/2,dy=q[1]-p.height/2,mx=a.x-p.width/2,my=a.y-p.height/2;
            assert((dx*mx+dy*my)/Math.hypot(dx,dy)/Math.hypot(mx,my)>.97,`极点/半球亮面反向 ${kind} ${dec}`);
        }
        f.close();
    });
    const gallery=document.createElement('section');gallery.innerHTML='<h2>实际角径下的双月盈亏</h2><p>上排 Luna，下排 Echo；两排使用同一角尺度。左至右：满月、凸月、半月、月牙、细月牙、新月。暗盘与天空同色，但仍遮挡背景。</p>';
    const canvas=document.createElement('canvas');canvas.width=768;canvas.height=256;const out=canvas.getContext('2d'),source=document.createElement('canvas');source.width=source.height=384;
    const f=renderFixture('Canvas',renderer);
    for(const [row,id] of ['Luna','Echo'].entries())for(const [col,phase] of [0,60,90,120,150,180].entries()) {
        const p=f.show([sampleMoon(id,phase)],{enhanced:true});source.getContext('2d').putImageData(new ImageData(new Uint8ClampedArray(p.data),384,384),0,0);
        out.drawImage(source,72,72,240,240,col*128,row*128,128,128);
    }
    f.close();gallery.append(canvas);document.body.append(gallery);
    document.querySelector('#status').dataset.phaseSamples=samples;
    renderer.setPixelRatio(1);renderer.setSize(256,256);
}
