import {coordinates,delta} from './geometry.mjs?revision=candidate-editor-band-1';
import {toEquatorial} from './territories.mjs';

// Close on the sphere before unwrapping. A loop winding around a pole ends
// one full turn from its first map point; Canvas.closePath would add a false
// 360-degree chord across the chart. Only filled polygons close in map space.
export function equatorialPath(points,center,closed=false){
    if(!points.length)return [];
    let previous=center;
    return (closed?[...points,points[0]]:points).map((v,i)=>{
        const c=coordinates(toEquatorial(v));previous=i?previous+delta(c.longitude,previous):center+delta(c.longitude,center);
        return {longitude:previous,latitude:c.latitude};
    });
}
