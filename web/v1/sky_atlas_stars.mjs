import {pointRadius,pointOpacity} from '../shared/sky_render.mjs';

import {TILE_SIZE,TILE_STRIDE,TILE_COLUMNS,pointKernelAtlas,resolvedPointImage} from '../shared/sky_point_kernel.mjs';
const COLOR_CACHE_LIMIT=64;

// 复用颜色纹理，每颗星连续改变大小与强度；不把星等取整成几档同尺寸符号。
export class AtlasStarPainter {
    constructor(ctx) {
        this.ctx=ctx;this.sprites=new Map();this.smallSprites=new Map();
    }
    cacheTexture(cache,color,image) {
        let texture;
        if(cache.size>=COLOR_CACHE_LIMIT) {
            const oldest=cache.keys().next().value;
            texture=cache.get(oldest);cache.delete(oldest);
        } else texture=document.createElement('canvas');
        if(texture.width!==image.width)texture.width=image.width;
        if(texture.height!==image.height)texture.height=image.height;
        texture.getContext('2d').putImageData(image,0,0);cache.set(color,texture);return texture;
    }
    clear() {
        for(const cache of [this.sprites,this.smallSprites]) {
            for(const texture of cache.values())texture.width=texture.height=0;
            cache.clear();
        }
    }
    smallSprite(color) {
        if(this.smallSprites.has(color))return this.smallSprites.get(color);
        return this.cacheTexture(this.smallSprites,color,pointKernelAtlas(color));
    }
    draw(star,p,symbolScale=1,white=false) {
        const color=white?'#ffffff':star.color_hex,radius=pointRadius(star.app_mag)*symbolScale;
        const ctx=this.ctx,transform=ctx.getTransform(),pixelScale=Math.hypot(transform.a,transform.b),pixelRadius=radius*pixelScale;
        if(pixelRadius<=0)return;
        const opacity=ctx.globalAlpha;
        ctx.imageSmoothingEnabled=true;
        if(pixelRadius<2) {
            // 先对像素面积积分，再按原生像素绘制。半径最多向上量化 1/128 像素，
            // 用面积比抵消这点差值；5×5 的透明采样区域不是恒星的实际显示直径。
            const index=Math.ceil(pixelRadius*128),texture=this.smallSprite(color),size=TILE_SIZE/pixelScale;
            ctx.imageSmoothingQuality='low';
            ctx.globalAlpha=opacity*pointOpacity(star.app_mag)*(pixelRadius/(index/128))**2;
            ctx.drawImage(texture,index%TILE_COLUMNS*TILE_STRIDE+1,Math.floor(index/TILE_COLUMNS)*TILE_STRIDE+1,TILE_SIZE,TILE_SIZE,p.x-size/2,p.y-size/2,size,size);ctx.globalAlpha=opacity;return;
        }
        const index=Math.ceil(pixelRadius*128),key=color+':'+index;
        let texture=this.sprites.get(key);
        if(!texture)texture=this.cacheTexture(this.sprites,key,resolvedPointImage(color,index/128));
        const size=texture.width/pixelScale;
        ctx.imageSmoothingQuality='low';ctx.globalAlpha=opacity*pointOpacity(star.app_mag)*(pixelRadius/(index/128))**2;
        ctx.drawImage(texture,p.x-size/2,p.y-size/2,size,size);ctx.globalAlpha=opacity;
    }
}
