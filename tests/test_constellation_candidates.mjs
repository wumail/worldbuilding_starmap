import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import crypto from 'node:crypto';
import {buildCandidates,catalogueStars} from '../src/build_zodiac_candidates.mjs';
import {dot,cross,vector,separation,tangentFrame,projectLocal} from '../web/constellations/geometry.mjs';

const data=JSON.parse(fs.readFileSync(new URL('../design/zodiac_candidates_v1.json',import.meta.url)));
const sourceBytes=fs.readFileSync(new URL(`../${data.catalogue}`,import.meta.url)),raw=JSON.parse(sourceBytes);
const source=new Map(catalogueStars(raw).map(s=>[s.id,s]));
const sites=[...data.regions.map(r=>r.site),...data.remainderSites];
const normals=data.regions.map(r=>r.polygon.map((v,i)=>{
    const n=cross(v,r.polygon[(i+1)%r.polygon.length]);return dot(n,r.site)<0?n.map(x=>-x):n;
}));
const inPolygon=(v,index)=>normals[index].every(n=>dot(n,v)>=-1e-10);
const nearest=v=>sites.map(c=>dot(v,c)).reduce((best,x,i,a)=>x>a[best]?i:best,0);

test('candidate stars retain exact catalogue identity, coordinates, brightness and colour',()=>{
    assert.equal(data.sha256,crypto.createHash('sha256').update(sourceBytes).digest('hex'));
    assert.equal(data.sourceCount,raw.stars.length);assert.equal(data.regions.length,15);
    const selected=new Set();
    for(const [i,r] of data.regions.entries())for(const s of r.members){
        assert.deepEqual(s,source.get(s.id));assert.ok(!selected.has(s.id));selected.add(s.id);
        assert.ok(s.app_mag<=4.5&&Math.abs(s.latitude)<=30);assert.equal(nearest(s.direction),i);assert.ok(inPolygon(s.direction,i));
    }
    assert.equal(selected.size,data.selectedExtendedCount);
    assert.equal(data.regions.reduce((n,r)=>n+r.variants[0].members.length,0),data.selectedCoreCount);
});

test('both skeletons are connected, preserve core members, and obey angular and magnitude limits',()=>{
    for(const r of data.regions){
        const byId=new Map(r.members.map(s=>[s.id,s]));
        assert.ok(r.variants[0].members.length>=5&&r.variants[0].members.length<=7);
        assert.ok(r.variants[0].members.every(id=>r.variants[1].members.includes(id)));
        for(const v of r.variants){
            const seen=new Set([v.members[0]]);assert.equal(v.edges.length,v.members.length-1);
            for(let pass=0;pass<v.members.length;pass++)for(const e of v.edges){
                assert.ok(v.members.includes(e.from)&&v.members.includes(e.to));
                const d=separation(byId.get(e.from).direction,byId.get(e.to).direction);
                assert.ok(Math.abs(d-e.degrees)<1e-10);assert.ok(d<=13+1e-10);
                if(seen.has(e.from)||seen.has(e.to)){seen.add(e.from);seen.add(e.to);}
            }
            assert.equal(seen.size,v.members.length);
            for(let i=0;i<v.members.length;i++)for(let j=i+1;j<v.members.length;j++){
                const d=separation(byId.get(v.members[i]).direction,byId.get(v.members[j]).direction);
                assert.ok(d>=.25-1e-10&&d<=35+1e-10);
            }
        }
    }
    assert.ok(data.regions.filter(r=>r.oldSectors.length>1).length>0);
});

test('irregular polygons agree with nearest-site partition and never overlap in 100000 sky samples',()=>{
    for(let k=0;k<100000;k++){
        const z=1-2*(k+.5)/100000,phi=k*Math.PI*(3-Math.sqrt(5)),v=[Math.sqrt(1-z*z)*Math.cos(phi),Math.sqrt(1-z*z)*Math.sin(phi),z];
        const contained=data.regions.flatMap((_,i)=>inPolygon(v,i)?[i]:[]),index=nearest(v);
        assert.deepEqual(contained,index<15?[index]:[]);
    }
    for(const [i,r] of data.regions.entries())for(const p of r.polygon){assert.ok(Math.abs(Math.hypot(...p)-1)<1e-10);for(const s of sites)assert.ok(dot(p,r.site)>=dot(p,s)-1e-10);}
});

test('all fifteen ecliptic intervals tile 360 degrees, including the zero-longitude seam',()=>{
    const intervals=data.regions.flatMap((r,i)=>r.intervals.map(s=>({...s,index:i}))).sort((a,b)=>a.start-b.start);
    assert.equal(intervals[0].start,0);assert.equal(intervals.at(-1).end,360);
    for(let i=0;i<intervals.length;i++){
        const s=intervals[i];if(i)assert.ok(Math.abs(s.start-intervals[i-1].end)<1e-9);
        assert.ok(s.end>s.start);for(const t of [.0001,.5,.9999])assert.equal(nearest(vector(s.start+(s.end-s.start)*t,0)),s.index);
    }
    assert.equal(new Set(intervals.map(i=>i.index)).size,15);
    assert.ok(Math.abs(data.regions.reduce((s,r)=>s+r.eclipticSpan,0)-360)<1e-9);
    assert.equal(nearest(vector(0,0)),nearest(vector(360,0)));
    for(let longitude=0;longitude<360;longitude+=.05){const v=vector(longitude,0);assert.ok(nearest(v)<15);assert.ok(inPolygon(v,nearest(v)));}
});

test('local projection has equal axes and all members fit desktop and mobile at the shared scale',()=>{
    for(const [width,height] of [[1156,610],[630,500],[334,460],[294,460]])for(const r of data.regions){
        const frame=tangentFrame(r.center.longitude,r.center.latitude),ppd=Math.min((width-44)/58,(height-44)/52);
        const center=projectLocal(frame.center,frame,width/2,height/2,ppd);assert.ok(Math.hypot(center.x-width/2,center.y-height/2)<1e-9);
        const offset=axis=>frame.center.map((x,i)=>x*Math.cos(.01)+axis[i]*Math.sin(.01));
        const north=projectLocal(offset(frame.north),frame,width/2,height/2,ppd),east=projectLocal(offset(frame.east),frame,width/2,height/2,ppd);
        assert.ok(north.y<center.y&&east.x<center.x);assert.ok(Math.abs(center.y-north.y-(center.x-east.x))<1e-9);
        for(const s of r.members){const p=projectLocal(s.direction,frame,width/2,height/2,ppd);assert.ok(p.visible&&p.x>5&&p.y>5&&p.x<width-5&&p.y<height-5);}
    }
});

test('regeneration is deterministic and leaves the source object unchanged',()=>{
    const before=JSON.stringify(raw),rebuilt=buildCandidates(raw,data.sha256);
    const hash=value=>crypto.createHash('sha256').update(JSON.stringify(value)).digest('hex');
    assert.equal(hash(rebuilt),hash(data));assert.equal(JSON.stringify(raw),before);
});
