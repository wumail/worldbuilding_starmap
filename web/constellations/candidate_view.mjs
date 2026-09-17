import {coordinates,cross,dot,unprojectLocal} from './geometry.mjs';

const MIN_ZOOM=.2,MAX_ZOOM=6;

// The viewing state belongs to the candidate window, independently of edits.
// Keeping its reference scene fixed also prevents draft changes from refitting it.
export function mountCandidateView({canvas,getLayout,redraw,blocked=()=>false}){
    const $=id=>document.getElementById(id),slider=$('candidate-view-zoom');
    let view=null,home=null,drag=null,suppressClick=false;
    const pointer=e=>{const rect=canvas.getBoundingClientRect();return {x:e.clientX-rect.left,y:e.clientY-rect.top};};
    function refresh(){
        const disabled=!view||blocked(),zoom=view?.zoom??1;
        slider.value=zoom;$('candidate-view-value').textContent=`${zoom.toFixed(2)}×`;
        slider.disabled=disabled;$('candidate-view-reset').disabled=disabled;
        $('candidate-view-minus').disabled=disabled||zoom<=MIN_ZOOM;
        $('candidate-view-plus').disabled=disabled||zoom>=MAX_ZOOM;
        canvas.dataset.viewZoom=String(zoom);
    }
    function stopDrag(){
        const id=drag?.id;drag=null;canvas.classList.remove('panning');
        if(id!==undefined&&canvas.hasPointerCapture(id))canvas.releasePointerCapture(id);
    }
    function zoomTo(value){
        if(!view||blocked()||!Number.isFinite(value))return;
        stopDrag();view.zoom=Math.max(MIN_ZOOM,Math.min(MAX_ZOOM,value));refresh();redraw();
    }
    slider.oninput=e=>zoomTo(Number(e.target.value));
    $('candidate-view-minus').onclick=()=>zoomTo(view.zoom/1.25);
    $('candidate-view-plus').onclick=()=>zoomTo(view.zoom*1.25);
    $('candidate-view-reset').onclick=()=>{
        if(!home||blocked())return;stopDrag();view={...home,center:{...home.center}};refresh();redraw();
    };
    canvas.addEventListener('wheel',e=>{
        if(!view||blocked()||drag)return;e.preventDefault();
        const pixels=e.deltaY*(e.deltaMode===1?16:e.deltaMode===2?canvas.clientHeight:1);
        zoomTo(view.zoom*Math.exp(-pixels*.001));
    },{passive:false});
    // Editor hit targets get first refusal; dragging empty sky works in both modes.
    canvas.addEventListener('pointerdown',e=>{
        if(e.defaultPrevented||!view||blocked()||e.button!==0||!e.isPrimary)return;
        suppressClick=false;drag={id:e.pointerId,start:pointer(e),layout:getLayout(),moved:false};
        canvas.setPointerCapture(e.pointerId);e.preventDefault();
    });
    canvas.addEventListener('pointermove',e=>{
        if(!drag||e.pointerId!==drag.id||blocked())return;
        const p=pointer(e);if(!drag.moved&&Math.hypot(p.x-drag.start.x,p.y-drag.start.y)<4)return;
        drag.moved=true;suppressClick=true;canvas.classList.add('panning');
        const l=drag.layout,at=p=>unprojectLocal(p.x,p.y,l.frame,l.width/2,l.height/2,l.ppd);
        const a=at(p),b=at(drag.start),axis=cross(a,b),c=dot(a,b),v=l.frame.center;
        if(c>-1+1e-10){const first=cross(axis,v),second=cross(axis,first);view.center=coordinates(v.map((x,i)=>x+first[i]+second[i]/(1+c)));}
        redraw();
    });
    canvas.addEventListener('pointerup',e=>{if(drag?.id===e.pointerId)stopDrag();});
    canvas.addEventListener('pointercancel',e=>{if(drag?.id===e.pointerId){suppressClick=true;stopDrag();}});
    canvas.addEventListener('lostpointercapture',e=>{if(drag?.id===e.pointerId)stopDrag();});
    canvas.addEventListener('click',e=>{if(suppressClick){suppressClick=false;e.preventDefault();e.stopImmediatePropagation();}},true);
    return {
        setScene(reference,index,{preserve=false}={}){
            stopDrag();suppressClick=false;
            home={reference,center:{...reference.regions[index].center},zoom:1};
            if(!preserve||!view)view={...home,center:{...home.center}};
            refresh();
        },
        refresh,
        get view(){return view;},
        get status(){return view?{center:{...view.center},zoom:view.zoom}:null;}
    };
}
