import {DEG,SYSTEM,rotateX,equatorialDirection,unit,mod,applyFrame} from './solar_system.mjs';
import {SKY_OVERLAYS} from './sky_overlays.mjs';

// Fixed reference-epoch ecliptic: editing orbital starting angles does not
// silently rotate the sky regions. Fifteen equal longitude intervals, 24° each.
export const ZODIAC=Object.freeze({count:15,width:24,beltLatitude:30,epoch:'Terrax 第 0 日参考黄道',origin:0});
export const eclipticDirection=(longitude,latitude=0)=>rotateX(equatorialDirection(longitude,latitude),SYSTEM.observer.obliquity*DEG);
export function eclipticCoordinates(direction){
    const v=rotateX(unit(direction),-SYSTEM.observer.obliquity*DEG);
    const latitude=Math.asin(Math.max(-1,Math.min(1,v[2])))/DEG;
    return {longitude:Math.hypot(v[0],v[1])<1e-12?null:mod(Math.atan2(v[1],v[0])/DEG,360),latitude};
}
export function zodiacRegion(direction){
    const c=eclipticCoordinates(direction);
    // Numerical roundoff at an exact boundary belongs to its new interval.
    const index=c.longitude===null?null:Math.floor(mod(c.longitude+1e-10,360)/24);
    return {...c,index,inBelt:index!==null && Math.abs(c.latitude)<=ZODIAC.beltLatitude+1e-10,
        label:index===null?'黄道极点':`${String(index+1).padStart(2,'0')} 天区`};
}
export const zodiacLines=[
    ...Array.from({length:15},(_,i)=>({boundary:true,points:Array.from({length:61},(_,j)=>eclipticDirection(i*24,j-30))})),
    ...[-30,0,30].map(b=>({boundary:b!==0,points:Array.from({length:361},(_,l)=>eclipticDirection(l,b))})),
];
export const zodiacLabels=Array.from({length:15},(_,i)=>({text:`${String(i+1).padStart(2,'0')} 天区`,direction:eclipticDirection(i*24+12,8)}));
export function drawZodiac(ctx,sky,project,{scale=1,maxJump=200}={}){
    ctx.save();ctx.strokeStyle='#b4a17c';ctx.fillStyle='#c8b68e';ctx.lineWidth=.8/scale;
    for(const line of zodiacLines){
        ctx.globalAlpha=SKY_OVERLAYS.line;ctx.setLineDash(line.boundary?[3/scale,4/scale]:[]);ctx.beginPath();let last=null;
        for(const v of line.points){
            const d=applyFrame(v,sky.frame),p=project(d);
            if((sky.frame.surface && d[2]<0) || p.visible===false || !Number.isFinite(p.x+p.y)){last=null;continue;}
            if(last && p.north===last.north && Math.hypot(p.x-last.x,p.y-last.y)<maxJump)ctx.lineTo(p.x,p.y);else ctx.moveTo(p.x,p.y);
            last=p;
        }ctx.stroke();
    }
    ctx.setLineDash([]);ctx.globalAlpha=.85*SKY_OVERLAYS.text;ctx.font=`${11/scale}px -apple-system,sans-serif`;ctx.textAlign='center';
    for(const label of zodiacLabels){const d=applyFrame(label.direction,sky.frame),p=project(d);if(p.visible!==false && (!sky.frame.surface || d[2]>=0))ctx.fillText(label.text,p.x,p.y);}
    ctx.restore();
}
