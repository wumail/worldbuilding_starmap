import test from 'node:test';
import assert from 'node:assert/strict';
import {eclipticDirection,eclipticCoordinates,zodiacRegion,zodiacLabels,ZODIAC} from '../web/shared/zodiac.mjs';
import {applyFrame,observerFrame,skyState,DEG} from '../web/shared/solar_system.mjs';
import {sanitizeState as v1} from '../web/v1/sky_state.mjs';
import {sanitizeState as v2} from '../web/v2/sky_state.mjs';
test('15 fixed sectors cover the ecliptic once, including the 360-degree seam',()=>{
 assert.equal(ZODIAC.count*ZODIAC.width,360);assert.equal(zodiacLabels.length,15);
 for(let i=0;i<15;i++)for(const d of [0,.0001,12,23.9999])assert.equal(zodiacRegion(eclipticDirection(i*24+d)).index,i);
 assert.equal(zodiacRegion(eclipticDirection(360)).index,0);
});
test('coordinates are invertible and independent of surface latitude and time',()=>{
 for(const longitude of [0,23.99,180,359.9])for(const latitude of [-80,-30,0,30,80]){
   const direction=eclipticDirection(longitude,latitude),frame=observerFrame(390,{mode:'surface',latitude:67,longitude:-130});
   const view=applyFrame(direction,frame),back=[0,1,2].map(i=>frame.matrix.reduce((sum,row,j)=>sum+row[i]*view[j],0));
   const c=eclipticCoordinates(back);assert.ok(Math.abs(c.latitude-latitude)<1e-10);
   assert.ok(Math.abs(Math.sin((c.longitude-longitude)*DEG))<1e-10);
 }
 assert.equal(zodiacRegion(eclipticDirection(30,90)).index,null);
 assert.equal(zodiacRegion(eclipticDirection(30,31)).inBelt,false);
});
test('Terrax Sun stays in the reference ecliptic, while actual moon phases are not 15 per year',()=>{
 for(let day=0;day<555.569;day+=7){const sol=skyState(day).bodies[0];assert.ok(Math.abs(zodiacRegion(sol.equatorial).latitude)<1e-10);}
 assert.ok(Math.abs(555.569*(1/24.253-1/70.258)-15)<.001);
 assert.ok(Math.abs((555.569/24.253-1)-15)>6);
});
test('both versions independently persist visibility of the same two new layers',()=>{
 for(const sanitize of [v1,v2]){assert.equal(sanitize().zodiac,true);assert.equal(sanitize().deepSky,true);assert.equal(sanitize({zodiac:false,deepSky:false}).zodiac,false);assert.equal(sanitize({zodiac:false,deepSky:false}).deepSky,false);}
});
