import {AtlasStarPainter} from '../web/v1/sky_atlas_stars.mjs';
import {POINT_STYLE,pointExposure,pointProfile,pointRadius,pointOpacity} from '../web/shared/sky_render.mjs';

// 对照此前已验收的独立 5×5 纹理，专门检查合并纹理后的裁切与边界采样。
export function runAtlasTextureAudit({test,assert}) {
    test('合并后的恒星纹理保留独立纹理的实际像素，覆盖格边界与亚像素落点',()=>{
        const canvas=()=>{const c=document.createElement('canvas');c.width=c.height=24;return c;};
        const actual=canvas(),reference=canvas(),source=canvas();source.width=source.height=5;
        const ctx=actual.getContext('2d'),ref=reference.getContext('2d'),painter=new AtlasStarPainter(ctx);
        const radii=[.05,.1249,.125,.1251,.2499,.25,.2501,.4999,.5,.5001,.9999,1,1.0001,1.8749,1.875,1.8751,1.992,1.9999];
        let samples=0,maxDelta=0;
        for(const color of ['#ffffff','#ffcc6f','#aabfff'])for(const radius of radii) {
            const index=Math.ceil(radius*128),r=index/128,image=new ImageData(5,5);
            const rgb=[0,2,4].map(i=>parseInt(color.slice(1+i,3+i),16));
            for(let y=0;y<5;y++)for(let x=0;x<5;x++) {
                let coverage=0,resolved=0;
                if(r<=.5) {
                    const c=POINT_STYLE.coreFraction;
                    if(x===2 && y===2)coverage=2*Math.PI*(.15+.2*c+.15*c*c)*r*r;
                } else {
                    for(let sy=0;sy<16;sy++)for(let sx=0;sx<16;sx++){const p=pointProfile(Math.hypot(x+(sx+.5)/16-2.5,y+(sy+.5)/16-2.5)/r);coverage+=p/256;resolved+=pointExposure(p)/256;}
                }
                const k=4*(y*5+x);image.data.set(rgb,k);const t=Math.max(0,Math.min(1,(r-.5)/1.5)),w=t*t*(3-2*t);image.data[k+3]=Math.round(255*((1-w)*pointExposure(coverage)+w*resolved));
            }
            source.getContext('2d').putImageData(image,0,0);
            for(const dpr of [1,2])for(const offset of [0,.25,.5,.75]) {
                const position={x:(12+offset)/dpr,y:(12+.75-offset)/dpr},mag=6.5,size=5/dpr;
                for(const c of [ctx,ref]) {c.resetTransform();c.clearRect(0,0,24,24);c.setTransform(dpr,0,0,dpr,0,0);c.globalAlpha=.83;}
                painter.draw({app_mag:mag,color_hex:color},position,radius/(dpr*pointRadius(mag)));
                ref.imageSmoothingEnabled=true;ref.imageSmoothingQuality='low';ref.globalAlpha=.83*pointOpacity(mag)*(radius/r)**2;
                ref.drawImage(source,position.x-size/2,position.y-size/2,size,size);
                const a=ctx.getImageData(0,0,24,24).data,b=ref.getImageData(0,0,24,24).data;
                let delta=0;for(let k=0;k<a.length;k++)delta=Math.max(delta,Math.abs(a[k]-b[k]));
                maxDelta=Math.max(maxDelta,delta);samples++;
                assert(delta<=1,JSON.stringify({radius,index,color,dpr,offset,delta}));
            }
        }
        document.querySelector('#status').dataset.atlasTextureSamples=String(samples);
        document.querySelector('#status').dataset.atlasTextureMaxDelta=String(maxDelta);
        painter.clear();for(const c of [actual,reference,source])c.width=c.height=0;
    });
}
