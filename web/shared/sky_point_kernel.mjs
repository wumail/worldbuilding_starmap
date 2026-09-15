import {POINT_STYLE,pointProfile,pointExposure} from './sky_render.mjs';

export const TILE_SIZE=5,TILE_STRIDE=7,TILE_COLUMNS=16,RADIUS_STEPS=256;
const kernels=new Map();
export function pointKernel(index) {
    let kernel=kernels.get(index);if(kernel)return kernel;
    const radius=index/128;kernel=new Float64Array(TILE_SIZE*TILE_SIZE);
    if(radius<=.5) {const c=POINT_STYLE.coreFraction;kernel[12]=pointExposure(2*Math.PI*(.15+.2*c+.15*c*c)*radius*radius);}
    else for(let y=0;y<TILE_SIZE;y++)for(let x=0;x<TILE_SIZE;x++) {
        let coverage=0,resolved=0;
        for(let sy=0;sy<16;sy++)for(let sx=0;sx<16;sx++) {
            const p=pointProfile(Math.hypot(x+(sx+.5)/16-TILE_SIZE/2,y+(sy+.5)/16-TILE_SIZE/2)/radius);
            coverage+=p/256;resolved+=pointExposure(p)/256;
        }
        // 亚像素可辨识度增强平滑退出；在 2 px 接口处与完整 PSF 的面积积分相同。
        // 这是显示对比度处理，始终不改变几何半径。
        const t=Math.min(1,(radius-.5)/1.5),w=t*t*(3-2*t);
        kernel[y*TILE_SIZE+x]=(1-w)*pointExposure(coverage)+w*resolved;
    }
    kernels.set(index,kernel);return kernel;
}
export function pointKernelAtlas(color='#ffffff') {
    const width=TILE_COLUMNS*TILE_STRIDE,height=Math.ceil((RADIUS_STEPS+1)/TILE_COLUMNS)*TILE_STRIDE,image=new ImageData(width,height);
    const rgb=[0,2,4].map(i=>parseInt(color.slice(1+i,3+i),16));
    for(let index=1;index<=RADIUS_STEPS;index++) {
        const kernel=pointKernel(index),left=index%TILE_COLUMNS*TILE_STRIDE+1,top=Math.floor(index/TILE_COLUMNS)*TILE_STRIDE+1;
        for(let y=-1;y<=TILE_SIZE;y++)for(let x=-1;x<=TILE_SIZE;x++) {
            const k=4*((top+y)*width+left+x);image.data.set(rgb,k);
            image.data[k+3]=Math.round(255*kernel[Math.max(0,Math.min(TILE_SIZE-1,y))*TILE_SIZE+Math.max(0,Math.min(TILE_SIZE-1,x))]);
        }
    }
    return image;
}

// 大点像也先对像素面积积分再平移采样，避免在小/大纹理交界处换一套滤波。
export function resolvedPointImage(color,radius) {
    const size=2*Math.ceil(radius)+3,image=new ImageData(size,size),rgb=[0,2,4].map(i=>parseInt(color.slice(1+i,3+i),16));
    for(let y=0;y<size;y++)for(let x=0;x<size;x++) {
        let alpha=0;
        for(let sy=0;sy<16;sy++)for(let sx=0;sx<16;sx++)alpha+=pointExposure(pointProfile(Math.hypot(x+(sx+.5)/16-size/2,y+(sy+.5)/16-size/2)/radius))/256;
        const k=4*(y*size+x);image.data.set(rgb,k);image.data[k+3]=Math.round(255*alpha);
    }
    return image;
}
