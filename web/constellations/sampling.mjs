import {RAD,vector} from './geometry.mjs?revision=candidate-editor-band-1';
import {toEquatorial,fromEquatorial} from './territories.mjs';

export const DEFAULT_SAMPLING=40;
export const MIN_SAMPLING=15;
export const MAX_SAMPLING=60;
export function normalizeSampling(value=DEFAULT_SAMPLING){
    if(typeof value!=='number'||!Number.isInteger(value)||value<MIN_SAMPLING||value>MAX_SAMPLING)throw Error('采样范围应为黄道两侧各 15°–60°，精度 1°');
    return value;
}
export function samplingInfo(stars,halfWidth){
    normalizeSampling(halfWidth);
    return {method:'ecliptic-latitude-band-1',halfWidthDegrees:halfWidth,maximumMagnitude:4.5,
        candidateCount:stars.filter(s=>s.app_mag<=4.5&&Math.abs(s.latitude)<=halfWidth).length,
        phase:'initial-partition',finalBrightAudit:'full-catalogue'};
}
export const roundSampling=data=>data.sampling?.halfWidthDegrees??30;

// Solve n·direction = sin(beta). For |beta| <= 60° and obliquity 25°
// each band edge is a single-valued declination at every right ascension.
const north=toEquatorial([0,0,1]);
export function samplingDeclination(ra,beta){
    const a=north[0]*Math.cos(ra*RAD)+north[1]*Math.sin(ra*RAD),b=north[2];
    return (Math.asin(Math.sin(beta*RAD)/Math.hypot(a,b))-Math.atan2(a,b))/RAD;
}
export function samplingCurve(beta){
    return Array.from({length:361},(_,ra)=>fromEquatorial(vector(ra,samplingDeclination(ra,beta))));
}
