import {coordinates,delta,wrap} from './geometry.mjs?revision=candidate-editor-band-1';

export const LAYOUT_RULES=Object.freeze({siteLatitude:.5,minSiteGap:10,minEclipticSpan:12,maxEclipticSpan:40,maxBoundaryDrift:3,maxNeighbourOverlap:.25});

// These cells never reach a pole. Longitude on each minor great-circle edge
// is monotone (or constant), so its extrema are at vertices, not between them.
// Unwrap around each site's longitude to treat the 0/360 seam as an ordinary edge.
export function longitudeEnvelope(points,site) {
    const center=coordinates(site).longitude,values=points.map(p=>center+delta(coordinates(p).longitude,center));
    return {start:Math.min(...values),end:Math.max(...values)};
}
export function regionEnvelopes(regions) {
    return regions.map(r=>{
        const longitude=coordinates(r.site).longitude;
        const sections=r.intervals.map(s=>{const mid=(s.start+s.end)/2,shift=longitude+delta(mid,longitude)-mid;return {start:s.start+shift,end:s.end+shift};});
        return {id:r.id,...longitudeEnvelope(r.polygon,r.site),eclipticStart:Math.min(...sections.map(s=>s.start)),eclipticEnd:Math.max(...sections.map(s=>s.end))};
    });
}

// This is an extra cultural layout constraint. Distinct spherical cells can
// already have disjoint interiors while their one-dimensional longitude spans
// contain one another; a polygon-overlap test alone cannot enforce this rule.
export function layoutIssues(regions) {
    const issues=[],ranges=regionEnvelopes(regions),n=regions.length,epsilon=1e-8;
    for(let i=0;i<n;i++){
        const r=regions[i],range=ranges[i],site=coordinates(r.site),next=coordinates(regions[(i+1)%n].site);
        if(Math.abs(site.latitude)>LAYOUT_RULES.siteLatitude+epsilon)issues.push(`${r.id}: center leaves the ordered ecliptic band`);
        if(wrap(next.longitude-site.longitude)<LAYOUT_RULES.minSiteGap-epsilon)issues.push(`${r.id}: centres crowd in longitude`);
        const span=range.eclipticEnd-range.eclipticStart;
        if(span<LAYOUT_RULES.minEclipticSpan-epsilon||span>LAYOUT_RULES.maxEclipticSpan+epsilon)issues.push(`${r.id}: ecliptic width out of range`);
        if(range.eclipticStart-range.start>LAYOUT_RULES.maxBoundaryDrift+epsilon||range.end-range.eclipticEnd>LAYOUT_RULES.maxBoundaryDrift+epsilon)issues.push(`${r.id}: boundary reaches too far sideways`);
    }
    for(let i=0;i<n;i++)for(let j=i+1;j<n;j++)for(const shift of [-360,0,360]){
        const a=ranges[i],b={...ranges[j],start:ranges[j].start+shift,end:ranges[j].end+shift};
        const overlap=Math.min(a.end,b.end)-Math.max(a.start,b.start);
        if(overlap<=epsilon)continue;
        if((a.start<=b.start+epsilon&&a.end>=b.end-epsilon)||(b.start<=a.start+epsilon&&b.end>=a.end-epsilon))issues.push(`${a.id}/${b.id}: longitude containment`);
        const adjacent=j===i+1||(i===0&&j===n-1),limit=LAYOUT_RULES.maxNeighbourOverlap*Math.min(a.eclipticEnd-a.eclipticStart,b.eclipticEnd-b.eclipticStart);
        if(!adjacent||overlap>limit+epsilon)issues.push(`${a.id}/${b.id}: excessive longitude overlap`);
    }
    return issues;
}
