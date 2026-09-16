import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import {fileURLToPath} from 'node:url';
import {prepareBackground} from '../web/shared/solar_system.mjs';
import {eclipticCoordinates} from '../web/shared/zodiac.mjs';
import {RAD,vector,coordinates,unit,dot,separation,owner,cellPolygon,eclipticIntervals,minimumTree} from '../web/constellations/geometry.mjs';

const ROOT=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
export const SOURCE='output/output_20260915_nebula_03/sky_view_20260915_nebula_03.json';
export const SETTINGS=Object.freeze({count:15,candidateMagnitude:4.5,preferredMagnitude:4,searchLatitude:30,siteLatitude:8,displayScale:2.5,maxEdge:13,minSeparation:.25,outerSiteLatitude:50,boundaryClearance:2});
const weight=s=>(5-s.app_mag)**1.5; // Perceptual priority, deliberately not physical luminosity.
export const catalogueStars=raw=>prepareBackground(raw.stars).map(s=>{
    const {longitude,latitude}=eclipticCoordinates(s.baseDirection);
    return {id:s.id,app_mag:s.app_mag,color_hex:s.color_hex,distance_pc:s.distance_pc??s.dist_ly/3.26156,longitude,latitude,direction:vector(longitude,latitude)};
});

function sitesFor(stars) {
    let best;
    // Initial grids are only starting guesses; every longitude is free to move.
    for(let phase=0;phase<24;phase+=2) {
        let centers=Array.from({length:SETTINGS.count},(_,i)=>vector(i*24+phase,0));
        for(let iteration=0;iteration<120;iteration++) {
            const sums=centers.map(()=>[0,0,0]);
            for(const s of stars) {
                const i=owner(s.direction,centers),w=weight(s);
                for(let j=0;j<3;j++) sums[i][j]+=s.direction[j]*w;
            }
            const next=sums.map((v,i)=>{
                if(Math.hypot(...v)<1e-12)return centers[i];
                const c=coordinates(v);return vector(c.longitude,Math.max(-SETTINGS.siteLatitude,Math.min(SETTINGS.siteLatitude,c.latitude)));
            });
            const change=Math.max(...centers.map((v,i)=>1-dot(v,next[i])));centers=next;
            if(change<1e-12)break;
        }
        centers.sort((a,b)=>coordinates(a).longitude-coordinates(b).longitude);
        // Two unnamed polar remainder regions close the polygons. They are
        // placeholders for a future all-sky cultural partition, not star groups.
        const all=[...centers,[0,0,1],[0,0,-1]],intervals=eclipticIntervals(all);
        if(new Set(intervals.map(i=>i.index)).size!==SETTINGS.count || intervals.some(i=>i.index>=SETTINGS.count))continue;
        const groups=centers.map(()=>[]);
        for(const s of stars){const i=owner(s.direction,all);if(i<groups.length)groups[i].push(s);}
        const bright=groups.map(g=>g.filter(s=>s.app_mag<=SETTINGS.preferredMagnitude).length);
        const cost=stars.reduce((sum,s)=>sum+weight(s)*(1-dot(s.direction,all[owner(s.direction,all)])),0)+bright.reduce((sum,n)=>sum+.5*Math.max(0,5-n)**2,0);
        if(!best || cost<best.cost)best={centers:all,groups,cost,phase,intervals};
    }
    if(!best)throw Error('No partition gives all fifteen constellations an ecliptic interval.');
    return best;
}

function chooseFigure(pool,center) {
    const sorted=[...pool].sort((a,b)=>a.app_mag-b.app_mag || a.id.localeCompare(b.id));
    const goal=Math.min(7,Math.max(5,Math.round(sorted.filter(s=>s.app_mag<=4).length/3)));
    let best;
    for(const anchor of sorted.filter(s=>s.app_mag<=3.5)) {
        const chosen=[anchor];
        while(chosen.length<goal) {
            const ranked=sorted.filter(s=>!chosen.includes(s)).map(s=>{
                const gaps=chosen.map(p=>separation(s.direction,p.direction)),near=Math.min(...gaps),far=Math.max(...gaps);
                return {s,near,far,score:s.app_mag*.9+near*.13+Math.max(0,s.app_mag-4)*2+Math.max(0,far-24)*.15};
            }).filter(x=>x.near>=SETTINGS.minSeparation&&x.near<=SETTINGS.maxEdge&&x.far<=35).sort((a,b)=>a.score-b.score);
            if(!ranked.length)break;chosen.push(ranked[0].s);
        }
        if(chosen.length<5)continue;
        const tree=minimumTree(chosen),mean=unit(chosen.reduce((sum,s)=>sum.map((v,i)=>v+s.direction[i]),[0,0,0]));
        const score=chosen.reduce((sum,s)=>sum+s.app_mag,0)/chosen.length+tree.reduce((sum,e)=>sum+e.degrees,0)*.018+separation(mean,center)*.025+(goal-chosen.length)*1.5;
        if(!best||score<best.score)best={chosen,tree,score};
    }
    if(!best)throw Error('A proposed region lacks a connected five-star figure; manual review is required.');
    const extended=[...best.chosen],extendedEdges=[...best.tree];
    while(extended.length<best.chosen.length+2) {
        const candidates=sorted.filter(s=>!extended.includes(s)).map(s=>{
            const closest=extended.map(p=>({p,d:separation(p.direction,s.direction)})).sort((a,b)=>a.d-b.d)[0];
            return {s,...closest,score:s.app_mag*.7+closest.d*.2};
        }).filter(c=>c.d>=SETTINGS.minSeparation&&c.d<=9&&extended.every(p=>separation(p.direction,c.s.direction)<=35)).sort((a,b)=>a.score-b.score);
        if(!candidates.length)break;
        const c=candidates[0];extended.push(c.s);extendedEdges.push({from:c.p.id,to:c.s.id,degrees:c.d});
    }
    return {core:best.chosen,coreEdges:best.tree,extended,extendedEdges};
}

export function buildCandidates(raw,sourceHash) {
    const stars=catalogueStars(raw),candidates=stars.filter(s=>Math.abs(s.latitude)<=SETTINGS.searchLatitude&&s.app_mag<=SETTINGS.candidateMagnitude);
    const partition=sitesFor(candidates);
    const figures=partition.groups.map((pool,i)=>chooseFigure(pool,partition.centers[i]));
    // Unnamed exterior sites bound the actual proposal to the neighbourhood
    // of the ecliptic. Move each poleward until every chosen star is retained
    // with a 2-degree distance advantage. These sites are never rendered as stars.
    const remainderSites=[];
    for(const sign of [-1,1])for(let longitude=12;longitude<360;longitude+=24){
        let latitude=SETTINGS.outerSiteLatitude,site;
        while(latitude<=90){
            site=vector(longitude,sign*latitude);
            if(figures.every((f,i)=>f.extended.every(s=>separation(s.direction,site)>=separation(s.direction,partition.centers[i])+SETTINGS.boundaryClearance)))break;
            latitude+=1;
        }
        if(latitude>90)throw Error('Cannot close an exterior boundary while retaining all members.');
        remainderSites.push(site);
    }
    const centers=[...partition.centers.slice(0,SETTINGS.count),...remainderSites],intervals=eclipticIntervals(centers);
    const regions=partition.groups.map((pool,index)=>{
        const figure=figures[index],center=coordinates(unit(figure.extended.reduce((sum,s)=>sum.map((v,i)=>v+s.direction[i]),[0,0,0])));
        const regionIntervals=intervals.filter(i=>i.index===index).map(({start,end})=>({start,end})),inside=pool.filter(s=>owner(s.direction,centers)===index);
        return {id:`Z${String(index+1).padStart(2,'0')}`,label:`候选 ${String(index+1).padStart(2,'0')}`,site:partition.centers[index],center,
            polygon:cellPolygon(index,centers),intervals:regionIntervals,eclipticSpan:regionIntervals.reduce((s,i)=>s+i.end-i.start,0),
            candidateCount:inside.length,brightCount:inside.filter(s=>s.app_mag<=4).length,
            members:figure.extended,variants:[{id:'core',label:'简洁主干',members:figure.core.map(s=>s.id),edges:figure.coreEdges},{id:'extended',label:'补充轮廓',members:figure.extended.map(s=>s.id),edges:figure.extendedEdges}],
            brightest:Math.min(...figure.core.map(s=>s.app_mag)),faintest:Math.max(...figure.core.map(s=>s.app_mag)),
            oldSectors:[...new Set(figure.core.map(s=>Math.floor(s.longitude/24)+1))].sort((a,b)=>a-b)};
    });
    return {schema:1,status:'candidate-review',catalogue:SOURCE,sha256:sourceHash,epoch:'Terrax 第 0 日参考黄道',settings:SETTINGS,
        sourceCount:stars.length,candidateCount:candidates.length,selectedCoreCount:regions.reduce((n,r)=>n+r.variants[0].members.length,0),
        selectedExtendedCount:regions.reduce((n,r)=>n+r.members.length,0),fit:{phase:partition.phase,cost:partition.cost},
        remainderSites,regions};
}

if(process.argv[1]&&path.resolve(process.argv[1])===fileURLToPath(import.meta.url)) {
    const bytes=fs.readFileSync(path.join(ROOT,SOURCE)),hash=crypto.createHash('sha256').update(bytes).digest('hex');
    const result=buildCandidates(JSON.parse(bytes),hash),target=path.join(ROOT,'design/zodiac_candidates_v1.json');
    fs.writeFileSync(target,JSON.stringify(result,null,2)+'\n');
    console.log(JSON.stringify({target,sourceCount:result.sourceCount,candidateCount:result.candidateCount,core:result.selectedCoreCount,extended:result.selectedExtendedCount,
        regions:result.regions.map(r=>({id:r.id,stars:r.variants[0].members.length,magnitudes:[r.brightest,r.faintest],span:r.eclipticSpan,crosses:r.oldSectors,maxEdge:Math.max(...r.variants[0].edges.map(e=>e.degrees))}))},null,2));
}
