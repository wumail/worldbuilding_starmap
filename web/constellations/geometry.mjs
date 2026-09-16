// All coordinates in this review are Terrax's fixed day-zero ecliptic.
export const RAD = Math.PI / 180;
export const wrap = x => (x % 360 + 360) % 360;
export const delta = (x, center) => wrap(x - center + 180) - 180;
export const dot = (a, b) => a.reduce((s, v, i) => s + v * b[i], 0);
export const unit = v => v.map(x => x / Math.hypot(...v));
export const cross = (a, b) => [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]];
export const vector = (longitude, latitude) => [Math.cos(latitude*RAD)*Math.cos(longitude*RAD), Math.cos(latitude*RAD)*Math.sin(longitude*RAD), Math.sin(latitude*RAD)];
export const coordinates = v => ({longitude: wrap(Math.atan2(v[1], v[0])/RAD), latitude: Math.asin(Math.max(-1, Math.min(1, unit(v)[2])))/RAD});
export const separation = (a, b) => Math.atan2(Math.hypot(...cross(a, b)), dot(a, b))/RAD;
export const owner = (v, centers) => centers.reduce((best, c, i) => dot(v, c) > dot(v, centers[best]) ? i : best, 0);

// Convex spherical Voronoi cells. Vertices are intersections of bisector
// planes, accepted only if every other site's half-space contains them.
export function cellPolygon(index, centers) {
    const center = centers[index], normals = centers.filter((_, i) => i !== index).map(c => center.map((v, j) => v-c[j]));
    const vertices = [];
    for (let i=0; i<normals.length; i++) for (let j=i+1; j<normals.length; j++) {
        const v = cross(normals[i], normals[j]);
        if (Math.hypot(...v) < 1e-12) continue;
        for (const sign of [1, -1]) {
            const p = unit(v).map(x => sign*x);
            if (normals.every(n => dot(n, p) >= -1e-10) && !vertices.some(q => separation(p, q) < 1e-6)) vertices.push(p);
        }
    }
    const east = unit(cross(Math.abs(center[2]) < .95 ? [0,0,1] : [1,0,0], center)), north = cross(center, east);
    return vertices.sort((a,b) => Math.atan2(dot(a,north),dot(a,east))-Math.atan2(dot(b,north),dot(b,east)));
}

export function arc(a, b, maxStep=1) {
    const angle = separation(a, b)*RAD, count = Math.max(1, Math.ceil(angle/RAD/maxStep));
    return Array.from({length:count+1}, (_,i) => {
        const t=i/count;
        if (angle<1e-10) return a;
        return unit(a.map((v,j) => (v*Math.sin((1-t)*angle)+b[j]*Math.sin(t*angle))/Math.sin(angle)));
    });
}

// Exact ecliptic intersections, rather than assigning by rounded longitude.
export function eclipticIntervals(centers) {
    const cuts = [0, 360];
    for (let i=0;i<centers.length;i++) for (let j=i+1;j<centers.length;j++) {
        const x=centers[i][0]-centers[j][0], y=centers[i][1]-centers[j][1];
        if (Math.hypot(x,y)<1e-12) continue;
        for (const offset of [-90,90]) {
            const longitude=wrap(Math.atan2(y,x)/RAD+offset), p=vector(longitude,0);
            if (dot(p,centers[i]) >= dot(p,centers[owner(p,centers)])-1e-10) cuts.push(longitude);
        }
    }
    const sorted=cuts.sort((a,b)=>a-b).filter((v,i,a)=>i===0 || v-a[i-1]>1e-7), intervals=[];
    for (let i=1;i<sorted.length;i++) {
        const start=sorted[i-1], end=sorted[i], index=owner(vector((start+end)/2,0),centers);
        if (intervals.at(-1)?.index===index) intervals.at(-1).end=end;
        else intervals.push({index,start,end});
    }
    return intervals;
}

export function tangentFrame(longitude, latitude) {
    const center=vector(longitude,latitude), east=[-Math.sin(longitude*RAD),Math.cos(longitude*RAD),0], north=cross(center,east);
    return {center,east,north};
}

// Stereographic local chart, east left. pxPerDegree is the central angular
// scale; all fifteen charts use the same scale and square canvas geometry.
export function projectLocal(v, frame, cx, cy, pxPerDegree) {
    const d=1+dot(v,frame.center);
    if (d<1e-8) return {x:Infinity,y:Infinity,visible:false};
    return {x:cx-2*dot(v,frame.east)/d/RAD*pxPerDegree, y:cy-2*dot(v,frame.north)/d/RAD*pxPerDegree, visible:dot(v,frame.center)>.15};
}

export function unprojectLocal(x,y,frame,cx,cy,pxPerDegree){
    const u=(cx-x)/pxPerDegree*RAD/2,v=(cy-y)/pxPerDegree*RAD/2,r2=u*u+v*v;
    return frame.center.map((c,i)=>((1-r2)*c+2*u*frame.east[i]+2*v*frame.north[i])/(1+r2));
}

export function minimumTree(stars) {
    if (!stars.length) return [];
    const used=new Set([0]), edges=[];
    while(used.size<stars.length) {
        let best;
        for(const i of used) for(let j=0;j<stars.length;j++) if(!used.has(j)) {
            const length=separation(stars[i].direction,stars[j].direction);
            if(!best || length<best.length) best={i,j,length};
        }
        used.add(best.j);edges.push({from:stars[best.i].id,to:stars[best.j].id,degrees:best.length});
    }
    return edges;
}
