// Independent angular measurements of saved figures; never calls the generator's
// quality score. A lower count is evidence of less angular crowding, not taste.
import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
const dot=(a,b)=>a.reduce((n,x,i)=>n+x*b[i],0);
const unit=a=>{const length=Math.hypot(...a);return a.map(x=>x/length);};
const tangent=(p,q)=>unit(q.map((x,i)=>x-dot(p,q)*p[i]));
export function angleMeasures(members,edges,lookup){
    const adj=new Map(members.map(id=>[id,[]])),angles=[];
    for(const [a,b] of edges){adj.get(a).push(b);adj.get(b).push(a);}
    for(const [id,near] of adj){
        const p=unit(lookup.get(id).direction),directions=near.map(n=>tangent(p,unit(lookup.get(n).direction)));
        for(let i=0;i<near.length;i++)for(let j=i+1;j<near.length;j++)angles.push({id,degree:near.length,
            degrees:Math.acos(Math.max(-1,Math.min(1,dot(directions[i],directions[j]))))*180/Math.PI});
    }
    const narrow=threshold=>angles.filter(a=>a.degrees<threshold).length;
    const branches=[...adj.values()].filter(a=>a.length>=3).length;
    return {minimumAngle:angles.length?Math.min(...angles.map(a=>a.degrees)):null,
        narrow20:narrow(20),narrow25:narrow(25),narrow30:narrow(30),
        sharpTurns:angles.filter(a=>a.degree===2&&a.degrees<30).length,
        crowdedJunctionAngles:angles.filter(a=>a.degree>=3&&a.degrees<30).length,branches,
        adjacentBranchEdges:edges.filter(([a,b])=>adj.get(a).length>=3&&adj.get(b).length>=3).length};
}
export function measureReport(report){
    const lookup=new Map(report.sky.stars.map(s=>[s.id,s]));
    return report.rows.map(r=>({index:r.index,style:r.style,seed:r.seed,
        full:angleMeasures(r.figure.members,r.figure.edges,lookup),core:angleMeasures(r.figure.coreMembers,r.figure.coreEdges,lookup)}));
}
export function summarizeAngles(rows){
    const result=[];
    for(const style of [...new Set(rows.map(r=>r.style))])for(const variant of ['full','core']){
        const samples=rows.filter(r=>r.style===style).map(r=>r[variant]),minima=samples.map(r=>r.minimumAngle).filter(n=>n!==null).sort((a,b)=>a-b);
        result.push({style,variant,samples:samples.length,
            ...Object.fromEntries(['narrow20','narrow25','narrow30','sharpTurns','crowdedJunctionAngles','branches','adjacentBranchEdges'].map(k=>[k,samples.reduce((n,r)=>n+r[k],0)])),
            figuresWithNarrowAngles:samples.filter(r=>r.narrow30>0).length,
            medianMinimumAngle:minima[Math.floor(minima.length/2)]??null});
    }
    return result;
}
if(process.argv[1]&&path.resolve(process.argv[1])===fileURLToPath(import.meta.url)){
    const args=process.argv.slice(2),value=key=>args.includes(key)?args[args.indexOf(key)+1]:null;
    const before=value('--before'),after=value('--after'),out=value('--out');
    if(!before||!after||!out)throw Error('Usage: --before OLD.json --after NEW.json --out NEW-METRICS.json');
    const oldRows=measureReport(JSON.parse(fs.readFileSync(before))),newRows=measureReport(JSON.parse(fs.readFileSync(after)));
    const report={definition:'All unordered incident arc pairs measured in the unit-sphere tangent plane. Angles below 30 degrees include degree-two reversals and branch crowding. Counts are not a perceptual experiment.',
        before:{path:before,summary:summarizeAngles(oldRows)},after:{path:after,summary:summarizeAngles(newRows)}};
    fs.mkdirSync(path.dirname(out),{recursive:true});fs.writeFileSync(out,JSON.stringify(report,null,2)+'\n',{flag:'wx'});
    console.log(JSON.stringify(report,null,2));
}
