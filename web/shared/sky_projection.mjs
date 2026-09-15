// 所有角度均为垂直视场；相机坐标 +X 向右、+Y 向上、-Z 向前。
const DEG=Math.PI/180;
export const projectionMode=value=>value==='stereographic'?'stereographic':'perspective';
export function projectionFocal(height,fov,mode='perspective') {
    return mode==='stereographic'?height/(4*Math.tan(fov*DEG/4)):height/(2*Math.tan(fov*DEG/2));
}
export function projectViewDirection(v,fov,aspect,mode='perspective') {
    const length=Math.hypot(...v),denominator=mode==='stereographic'?length-v[2]:-v[2];
    const k=mode==='stereographic'?1/Math.tan(fov*DEG/4):1/Math.tan(fov*DEG/2);
    if(length===0 || denominator<=length*1e-10)return {x:Infinity,y:Infinity,visible:false};
    const x=k*v[0]/denominator/aspect,y=k*v[1]/denominator;
    return {x,y,visible:Math.abs(x)<=1 && Math.abs(y)<=1};
}
export function unprojectViewDirection(x,y,fov,aspect,mode='perspective') {
    const t=Math.tan(fov*DEG/(mode==='stereographic'?4:2)),u=x*aspect*t,v=y*t;
    if(mode==='stereographic') {const d=1+u*u+v*v;return [2*u/d,2*v/d,(u*u+v*v-1)/d];}
    const length=Math.hypot(u,v,1);return [u/length,v/length,-1/length];
}

// Three 的标准材质和自定义材质共用同一非线性变换。w 保留用于裁剪/插值。
export const projectionVertexGLSL=`
uniform bool skyStereo;
uniform float skyStereoFactor;
uniform float skyFocal;
uniform vec2 skyViewport;
vec4 skyProject(vec3 p) {
    if(!skyStereo)return projectionMatrix*vec4(p,1.0);
    float distance=length(p),w=max(1e-7,distance-p.z);
    vec4 original=projectionMatrix*vec4(0.0,0.0,-distance,1.0);
    return vec4(projectionMatrix[0][0]*skyStereoFactor*p.x,
                projectionMatrix[1][1]*skyStereoFactor*p.y,original.z/original.w*w,w);
}
`;
export const projectionFragmentGLSL=`
uniform bool skyStereo;
uniform vec2 skyViewport;
uniform vec2 skyHalfSpan;
uniform vec3 skyWorldUp;
vec3 skyRay() {
    vec2 q=(gl_FragCoord.xy/skyViewport*2.0-1.0)*skyHalfSpan;
    return skyStereo?vec3(2.0*q,dot(q,q)-1.0)/(1.0+dot(q,q)):normalize(vec3(q,-1.0));
}
bool belowSkyHorizon(){return dot(skyWorldUp,skyRay())<0.0;}
`;
