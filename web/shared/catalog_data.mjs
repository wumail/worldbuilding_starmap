const OUTPUT_BASE=new URL('../../output/',import.meta.url);
let indexPromise;
export async function catalogURL(folder){
    indexPromise??=fetch(new URL('catalog_views.json',OUTPUT_BASE),{cache:'no-store'}).then(r=>r.ok?r.json():{}).catch(()=>({}));
    const index=await indexPromise,name=index[folder];
    if(name && !/^sky_view_[A-Za-z0-9_]+\.json$/.test(name))throw Error('浏览星表索引无效');
    return new URL(`${folder}/${name || `star_map_${folder.slice(7)}.json`}`,OUTPUT_BASE);
}
export function diffuseInfo(data,url){
    const r=data.deep_sky?.raster;if(!r)return null;
    if(!/^diffuse_[A-Za-z0-9_]+\.f32$/.test(r.file) || !Number.isInteger(r.width) || !Number.isInteger(r.height) || r.width*r.height>8388608 || r.width<1 || r.height<1)throw Error('深空光图格式无效');
    return {...r,url:new URL(r.file,url).href};
}
export function preferredCatalog(folders,saved){
    const requested=new URLSearchParams(globalThis.location?.search || '').get('catalog');
    if(folders.includes(requested))return requested;
    // Upgrade a remembered default to its morphology revision. An explicit
    // catalogue URL still opens the unchanged historical realization.
    const successor=['output_20260915_galactic_01','output_20260915_nebula_01','output_20260915_nebula_02'].includes(saved)?'output_20260915_nebula_03':saved;
    return folders.includes(successor)?successor:folders.includes(saved)?saved:folders[0];
}
