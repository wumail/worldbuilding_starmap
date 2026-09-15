import {DEG,projectHemisphere} from '../shared/solar_system.mjs';
import {SYMBOL_REFERENCE,diskLight,phaseImage,unresolvedPointWeight,skyBackground,SOLAR_GLARE_RADIUS_RATIO,bodyGlareImage,solarGlareVisibility,atlasBodyCircles,lunarSurfaceScale} from '../shared/sky_render.mjs';

import {AtlasStarPainter} from './sky_atlas_stars.mjs';

// 天体绘制与页面交互分开，真实盘面与光点使用同一角尺度。
export class AtlasBodyPainter {
    constructor(ctx) {
        this.ctx=ctx;this.disks=new Map();this.glares=new Map();this.points=new AtlasStarPainter(ctx);
    }
    clear() {
        for(const texture of this.disks.values())texture.width=texture.height=0;
        for(const {texture} of this.glares.values())texture.width=texture.height=0;
        this.disks.clear();this.glares.clear();this.points.clear();
        if(this.sampleCanvas)this.sampleCanvas.width=this.sampleCanvas.height=0;
        this.sampleCanvas=null;
    }
    paintTexture(texture) {
        const ctx=this.ctx,m=ctx.getTransform(),rx=Math.hypot(m.a,m.b),ry=Math.hypot(m.c,m.d);
        if(Math.min(rx,ry)>=2 || Math.max(rx,ry)>8) {ctx.drawImage(texture,-1,-1,2,2);return;}
        // 对真实小盘面的像素覆盖积分，不能靠把盘面放大来避免漏采样。
        const extentX=Math.abs(m.a)+Math.abs(m.c),extentY=Math.abs(m.b)+Math.abs(m.d);
        const left=Math.floor(m.e-extentX),top=Math.floor(m.f-extentY),width=Math.ceil(m.e+extentX)-left,height=Math.ceil(m.f+extentY)-top;
        const inv=m.inverse(),src=texture.getContext('2d').getImageData(0,0,texture.width,texture.height),image=new ImageData(width,height);
        for(let y=0;y<height;y++)for(let x=0;x<width;x++) {
            const rgb=[0,0,0];let alpha=0;
            for(let sy=0;sy<16;sy++)for(let sx=0;sx<16;sx++) {
                const dx=left+x+(sx+.5)/16,dy=top+y+(sy+.5)/16,u=inv.a*dx+inv.c*dy+inv.e,v=inv.b*dx+inv.d*dy+inv.f;
                if(u*u+v*v>=1)continue;
                const i=4*(Math.min(src.height-1,Math.floor((v+1)/2*src.height))*src.width+Math.min(src.width-1,Math.floor((u+1)/2*src.width))),a=src.data[i+3]/255;
                alpha+=a;for(let c=0;c<3;c++)rgb[c]+=src.data[i+c]*a;
            }
            const i=4*(y*width+x);for(let c=0;c<3;c++)image.data[i+c]=alpha?rgb[c]/alpha:0;image.data[i+3]=255*alpha/256;
        }
        this.sampleCanvas??=document.createElement('canvas');const tile=this.sampleCanvas;
        if(tile.width!==width)tile.width=width;if(tile.height!==height)tile.height=height;
        tile.getContext('2d').putImageData(image,0,0);
        ctx.save();ctx.resetTransform();ctx.drawImage(tile,left,top);ctx.restore();
    }
    drawSky(sky,l,orientation,{solarGlow=false,...options}={}) {
        const visibility=solarGlow?solarGlareVisibility(sky):0;
        for(const body of [...sky.bodies].sort((a,b)=>b.distance-a.distance)) {
            // Draw all silhouettes in distance order. Angular containment is
            // not a valid early-out for circular map symbols.
            this.draw(body,l,orientation,sky,options);
            // 光晕属于太阳所在的层，近处天体的整个盘面随后遮住它。
            if(body.id==='Sol')this.drawGlare(body,l,orientation,sky,visibility,options);
        }
    }
    disk(body,l,orientation,sky,texture,angularRadius,opacity=1) {
        const ctx=this.ctx,light=diskLight(body),sourceAngle=Math.atan2(-light[1],light[0]);
        for(const p of atlasBodyCircles(body,l,orientation,angularRadius)) {
            if(sky.frame.surface && !p.north)continue;
            const angle=Math.atan2(-p.light[1],p.light[0])-sourceAngle,c=Math.cos(angle)*p.radius,s=Math.sin(angle)*p.radius;
            ctx.save();ctx.beginPath();ctx.arc(...(p.north?l.north:l.south),l.radius,0,Math.PI*2);ctx.clip();ctx.globalAlpha*=opacity;
            ctx.transform(c,s,-s,c,p.x,p.y);
            this.paintTexture(texture);ctx.restore();
        }
    }
    drawGlare(body,l,orientation,sky,visibility,{symbolScale=l.radius/SYMBOL_REFERENCE.atlasRadius}={}) {
        if(visibility<=0 || body.id!=='Sol')return;
        let record=this.glares.get(body.id);
        if(!record) {
            const texture=document.createElement('canvas');texture.width=texture.height=128;
            texture.getContext('2d').putImageData(bodyGlareImage(body.color),0,0);
            record={texture};this.glares.set(body.id,record);
        }
        // 柔光与圆形日面共用显示尺度；前景盘面随后覆盖它。
        this.disk(body,l,orientation,sky,record.texture,Math.atan(SOLAR_GLARE_RADIUS_RATIO*Math.tan(body.angularDiameter*DEG/2)),visibility);
    }
    draw(body,l,orientation,sky,{symbolScale=l.radius/SYMBOL_REFERENCE.atlasRadius,lunarGlow=false,planetGlow=true}={}) {
        const ctx=this.ctx,angularRadius=body.angularDiameter*DEG/2;
        const diameter=l.radius/(Math.PI/2)*2*angularRadius;
        const transform=ctx.getTransform(),pixelScale=Math.hypot(transform.a,transform.b);
        const textureSize=Math.min(1024,Math.max(64,2**Math.ceil(Math.log2(Math.max(1,diameter*pixelScale*Math.PI/2)))));
        let texture=this.disks.get(body.id);
        if(!texture) {texture=document.createElement('canvas');this.disks.set(body.id,texture);}
        if(texture.width!==textureSize)texture.width=texture.height=textureSize;
        texture.getContext('2d').putImageData(phaseImage(body,diskLight(body),textureSize,{background:skyBackground(sky),brightEnough:body.brightEnough,lightScale:lunarSurfaceScale(body,lunarGlow)}),0,0);
        this.disk(body,l,orientation,sky,texture,angularRadius);
        const weight=unresolvedPointWeight(2*SYMBOL_REFERENCE.perspectiveFocal*Math.tan((body.physicalAngularDiameter ?? body.angularDiameter)*DEG/2));
        if(planetGlow && body.visible && body.kind==='planet' && weight>0) {
            // 行星识读光点保持同比；关闭增强即可独立查看真实盘面与盈亏。
            const p=projectHemisphere(body.view,l.radius,l,orientation);
            ctx.save();ctx.beginPath();ctx.arc(...(p.north?l.north:l.south),l.radius,0,Math.PI*2);ctx.clip();
            ctx.globalAlpha*=weight;
            this.points.draw({app_mag:body.magnitude,color_hex:body.color},p,symbolScale);
            ctx.restore();
        }
    }
}
