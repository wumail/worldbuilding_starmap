import {DEG,clamp,dot,lambertPhase,skyState} from '../shared/solar_system.mjs';

// V-band / photopic approximation. E(m) is illuminance at the observer, in lux.
export const ZERO_MAG_LUX=2.54e-6;
export const ARC_MIN=DEG/60,ARC_SEC=DEG/3600;
export const EYE_DEFAULTS=Object.freeze({screenInches:27,distanceCm:60,pixelsPerCm:0,opticalBlur:true,
    acuityArcmin:1.5,glareFraction:.01,glareArcmin:2,extinction:.20,skyMagnitude:21.7,exposureEV:0});
export function sanitizeEye(value={}) {
    const ranges={screenInches:[8,80],distanceCm:[20,200],pixelsPerCm:[0,200],acuityArcmin:[.5,4],
        glareFraction:[0,.15],glareArcmin:[.5,15],extinction:[0,.8],skyMagnitude:[16,22],exposureEV:[-3,3]};
    return Object.fromEntries(Object.entries(EYE_DEFAULTS).map(([k,v])=>[k,k==='opticalBlur'?value[k]!==false:clamp(Number.isFinite(value[k])?value[k]:v,...ranges[k])]));
}
export const magnitudeToLux=m=>Number.isFinite(m)?ZERO_MAG_LUX*10**(-.4*m):0;
export const luxToMagnitude=e=>e>0?-2.5*Math.log10(e/ZERO_MAG_LUX):Infinity;
export const magnitudeToLuminance=m=>magnitudeToLux(m)/(ARC_SEC*ARC_SEC);
export const luminanceToMagnitude=l=>luxToMagnitude(l*ARC_SEC*ARC_SEC);
export const diskSolidAngle=diameterDegrees=>4*Math.PI*Math.sin(diameterDegrees*DEG/4)**2;

// Kasten & Young (1989), above a flat astronomical horizon. Extinction k remains
// an editable clear-air assumption; pressure alone cannot specify aerosols.
export function airMass(altitudeDegrees) {
    const h=clamp(altitudeDegrees,0,90);
    return 1/(Math.sin(h*DEG)+.50572*(h+6.07995)**-1.6364);
}
export function observedMagnitude(m,altitude,eye,atmosphere=true) {
    return m+(atmosphere?eye.extinction*airMass(altitude):0);
}
export function screenCalibration(eye,screenWidth=1920,screenHeight=1080) {
    const aspect=Math.max(1,screenWidth)/Math.max(1,screenHeight);
    const physicalHeight=eye.screenInches*2.54/Math.hypot(aspect,1);
    const pixelsPerCm=eye.pixelsPerCm>0?eye.pixelsPerCm:screenHeight/physicalHeight;
    return {pixelsPerCm,focal:eye.distanceCm*pixelsPerCm,physicalHeightCm:screenHeight/pixelsPerCm};
}
export function fovForFocal(height,focal,projection='perspective') {
    return (projection==='stereographic'?4*Math.atan(height/(4*focal)):2*Math.atan(height/(2*focal)))/DEG;
}
export function diskPeakLuminance(body,magnitude=body.magnitude) {
    const area=diskSolidAngle(body.angularDiameter);
    if(!(area>0))return 0;
    if(body.id==='Sol')return magnitudeToLux(magnitude)/area;
    const phase=Number.isFinite(body.phaseAngle)?body.phaseAngle:Math.acos(clamp(-dot(body.view,body.lightDirection),-1,1));
    const phi=lambertPhase(phase);
    // Normalized Lambert sphere: integral = (2/3) * solid angle * Phi(alpha).
    // Phase is already in magnitude; dividing by Phi here avoids counting it twice.
    return phi>1e-8?magnitudeToLux(magnitude)/(area*(2/3)*phi):0;
}
export function pointHidden(direction,bodies,distance=Infinity,exceptId='') {
    return bodies.some(b=>b.id!==exceptId && b.distance<distance && Math.hypot(...direction.map((v,i)=>v-b.view[i]))<2*Math.sin(b.angularDiameter*DEG/4));
}
export function daylightLuminance(sunAltitude) {
    return sunAltitude<0?800*10**(.4*sunAltitude):800+4500*Math.sqrt(Math.sin(clamp(sunAltitude,0,90)*DEG));
}
// Approximate dark-sky naked-eye threshold (Schaefer form). Its daytime
// continuation is capped at -4 mag; it is not a daylight visibility experiment.
export function nakedEyeLimit(luminance) {
    const sqm=luminanceToMagnitude(Math.max(1e-9,luminance));
    return clamp(7.93-5*Math.log10(10**(4.316-sqm/5)+1),-4,6.5);
}
export function moonScattering(separationDegrees) {
    const rho=clamp(separationDegrees,10,180);
    return 10**5.36*(1.06+Math.cos(rho*DEG)**2)+10**(6.15-rho/40);
}
export function skyLuminance(direction,environment) {
    if(!environment.atmosphere)return environment.nightL;
    if(direction[2]<0)return 0;
    const x=airMass(Math.asin(clamp(direction[2],0,1))/DEG);
    let l=environment.nightL*(.6+.4*Math.sqrt(x))+environment.dayL*(.65+.35*Math.sqrt(x));
    for(const moon of environment.moons) {
        const rho=Math.acos(clamp(dot(direction,moon.direction),-1,1))/DEG;
        l+=moonScattering(rho)*moon.strength*(1-10**(-.4*environment.extinction*x));
    }
    return l;
}
export function observingEnvironment(sky,eye) {
    const atmosphere=sky.frame.surface && sky.daylight!==false;
    const env={atmosphere,extinction:eye.extinction,nightL:magnitudeToLuminance(eye.skyMagnitude),
        dayL:atmosphere?daylightLuminance(sky.bodies[0].altitude):0,moons:[]};
    if(atmosphere)for(const b of sky.bodies.filter(b=>b.kind==='moon' && b.altitude>0)) {
        // Krisciunas-Schaefer scattering shape, rescaled to each moon's computed
        // integrated flux (Terrax's Lambert phase remains authoritative).
        const e=magnitudeToLux(observedMagnitude(b.magnitude,b.altitude,eye));
        env.moons.push({direction:b.view,strength:3.18309886e-6*10**(-.4*3.84)*e/magnitudeToLux(-12.73)});
    }
    env.zenithL=skyLuminance([0,0,1],env);env.adaptation=Math.max(.005,env.zenithL);
    env.limit=atmosphere?nakedEyeLimit(env.zenithL):6.5;return env;
}
export function observerSky(days,state) {
    const sky=skyState(days,state);sky.daylight=state.daylight;
    const eye=sanitizeEye(state.eye),environment=observingEnvironment(sky,eye);sky.eye=eye;sky.environment=environment;
    sky.limitingMagnitude=environment.limit;
    for(const b of sky.bodies) {
        b.aboveHorizon=!sky.frame.surface || b.altitude+b.angularDiameter/2>0;
        b.observedMagnitude=b.aboveHorizon?observedMagnitude(b.magnitude,b.altitude,eye,environment.atmosphere):Infinity;
        const localLimit=environment.atmosphere?nakedEyeLimit(skyLuminance(b.view,environment)):6.5;
        b.brightEnough=b.id==='Sol' || b.observedMagnitude<=localLimit;
        b.visible=b.aboveHorizon && b.brightEnough;
        b.status=!b.aboveHorizon?'地平线下':b.brightEnough?'亮度达标':'当前天空中偏暗';
    }
    return sky;
}
// Normalized separable Gaussian weights. The final disk display uses these
// AFTER tone/coverage and occultation; the physical HDR diagnostic is separate.
export function gaussianKernel(sigma,maxRadius=24) {
    if(sigma<.12)return {step:1,weights:[1],radius:0};
    if(sigma>maxRadius/3)throw new RangeError('Downsample the image before a wide Gaussian blur');
    const step=1,radius=Math.min(maxRadius,Math.ceil(3*sigma));
    const weights=Array.from({length:radius+1},(_,i)=>Math.exp(-.5*(i/sigma)**2));
    const sum=weights[0]+2*weights.slice(1).reduce((a,b)=>a+b,0);
    return {step,radius,weights:weights.map(x=>x/sum)};
}
// Reinhard's global luminance response, with the adaptation level supplied by
// the local sky. This is a documented display operator, not measured eye data.
export function toneLuminance(l,adaptation,exposureEV=0) {
    const x=displayKey(adaptation)*2**exposureEV*Math.max(0,l)/Math.max(.005,adaptation);return x/(1+x);
}
// A display choice: preserve a dark night instead of exposing every background
// to 18% gray. This mesopic key interpolation is not a calibrated retina model.
export function displayKey(adaptation){
    const t=clamp((Math.log10(Math.max(adaptation,1e-9))+2)/4,0,1);
    return .002+(.18-.002)*t*t*(3-2*t);
}
