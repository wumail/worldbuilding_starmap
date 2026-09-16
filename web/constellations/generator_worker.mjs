import {generateDraw,prepareDraw} from './generator.mjs?revision=favourites-regional-1';
import {sampleRegion} from './regional_figures.mjs';
let stars,meta;
self.onmessage=async({data})=>{
    if(data.type==='init'){stars=data.stars;meta=data.meta;return;}
    if(data.type==='sample-region'){
        try{self.postMessage({id:data.id,result:sampleRegion(data.current,stars,data.index,data.seed,data.style)});}
        catch(error){self.postMessage({id:data.id,error:error.message});}return;
    }
    try{await prepareDraw(data.recipe);self.postMessage({id:data.id,result:generateDraw(stars,meta,data.recipe,{onProgress:progress=>self.postMessage({id:data.id,progress})})});}
    catch(error){self.postMessage({id:data.id,error:error.message});}
};
