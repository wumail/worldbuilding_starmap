import {DEG,SYSTEM,equatorialDirection,galacticToEquatorial,rotateX} from './solar_system.mjs';

// Shared sky directions and unattenuated colors. Opacity is applied once by
// the screen overlay, outside the optical/display response of the sky.
export const referenceLines=[];
for(let latitude=-60;latitude<=60;latitude+=30)referenceLines.push({color:latitude===0?'#00ff88':'#8896a6',points:Array.from({length:181},(_,i)=>equatorialDirection(i*2,latitude))});
for(let ra=0;ra<360;ra+=30)referenceLines.push({color:'#8896a6',points:Array.from({length:91},(_,i)=>equatorialDirection(ra,-90+i*2))});
referenceLines.push({color:'#ffaa00',points:Array.from({length:361},(_,i)=>rotateX(equatorialDirection(i,0),SYSTEM.observer.obliquity*DEG))});
referenceLines.push({color:'#4488ff',points:Array.from({length:361},(_,i)=>galacticToEquatorial(i,0))});
