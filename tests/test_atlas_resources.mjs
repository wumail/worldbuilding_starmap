import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {pathToFileURL} from 'node:url';

// 只代替浏览器的画布分配接口；访问顺序、纹理准备、缓存与绘制路径均使用生产代码。
let allocated=[];
globalThis.ImageData=class {
    constructor(width,height) {this.width=width;this.height=height;this.data=new Uint8ClampedArray(width*height*4);}
};
globalThis.document={createElement(tag) {
    assert.equal(tag,'canvas');
    const canvas={width:300,height:150,getContext:()=>({putImageData(){}})};
    allocated.push(canvas);return canvas;
}};
const moduleURL=process.env.ATLAS_STAR_MODULE?pathToFileURL(process.env.ATLAS_STAR_MODULE):new URL('../web/v1/sky_atlas_stars.mjs',import.meta.url);
const {AtlasStarPainter}=await import(moduleURL);
const data=JSON.parse(await readFile(new URL('../output/output_20260915_galactic_01/sky_view_20260915_galactic_01.json',import.meta.url),'utf8'));
const stars=[...(data.neighbors || []),...data.stars];
function fixture() {
    allocated=[];
    const ctx={globalAlpha:1,getTransform:()=>({a:1,b:0}),drawImage(texture,...args) {
        if(args.length===8) {
            const [x,y,w,h]=args;assert.ok(x>=0 && y>=0 && x+w<=texture.width && y+h<=texture.height,'采样超出纹理');
        }
    }};
    return new AtlasStarPainter(ctx);
}
test('真实 9356 星表预热后，播放与尺寸切换不再逐帧创建画布',()=>{
    const painter=fixture(),position={x:0,y:0},counts=[];
    for(const factor of [.8,1,1.6,2,3]) {
        for(const star of stars)painter.draw(star,position,factor);
        const warm=allocated.length;
        for(let frame=0;frame<3;frame++)for(const star of stars)painter.draw(star,position,factor);
        counts.push({factor,warm,additional:allocated.length-warm});
        assert.equal(allocated.length,warm,JSON.stringify(counts));
    }
    assert.ok(allocated.length<=14,JSON.stringify(counts));
    console.log('真实星表资源计数',JSON.stringify(counts));
});
test('超过缓存容量的颜色循环复用画布，不无限创建绘图资源',()=>{
    const painter=fixture();
    const palette=Array.from({length:80},(_,i)=>({app_mag:6.5,color_hex:'#'+(i*1009+1).toString(16).padStart(6,'0')}));
    for(const scale of [1,128])for(const star of palette)painter.draw(star,{x:0,y:0},scale);
    const warm=allocated.length;assert.ok(warm<=128);
    for(let frame=0;frame<2;frame++)for(const scale of [1,128])for(const star of palette)painter.draw(star,{x:0,y:0},scale);
    assert.equal(allocated.length,warm);
});
test('高倍缩放超过 64 种大点像半径后，缓存复用画布并保持容量上限',()=>{
    const painter=fixture(),star={app_mag:0,color_hex:'#ffffff'};
    const draw=()=>{for(let i=0;i<96;i++)painter.draw(star,{x:0,y:0},30+i/4);};
    draw();const warm=allocated.length;
    assert.equal(painter.sprites.size,64);assert.equal(warm,64);
    draw();assert.equal(allocated.length,warm);assert.equal(painter.sprites.size,64);
    const previous=[...allocated];painter.clear();
    assert.ok(previous.every(c=>c.width===0 && c.height===0));
});
test('清理缓存释放画布像素缓冲区，并可重新绘制',()=>{
    const painter=fixture(),star={app_mag:6.5,color_hex:'#ffffff'};
    painter.draw(star,{x:0,y:0});painter.draw(star,{x:0,y:0},128);
    const previous=[...allocated];painter.clear();
    assert.ok(previous.every(c=>c.width===0 && c.height===0));
    painter.draw(star,{x:0,y:0});assert.equal(allocated.length,previous.length+1);
    assert.ok(allocated.at(-1).width>0);
});
