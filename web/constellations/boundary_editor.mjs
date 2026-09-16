import {coordinates,delta,wrap,arc} from './geometry.mjs?revision=candidate-editor-band-1';
import {toEquatorial,unpackGrid} from './territories.mjs';
import {MANUAL_ALGORITHM,moveBoundaryEdge,applyEditedGrid,gridEdits} from './boundary_edits.mjs?revision=candidate-editor-band-1';

const $=id=>document.getElementById(id),colors=['#acc9d7','#aaa6ce','#d5b393','#b4ccaa','#cbb9cb'];
export function mountBoundaryEditor({stars,onPreview,onClose,onSave}){
    const panel=$('boundary-editor'),canvas=$('boundary-canvas'),ctx=canvas.getContext('2d');
    let base,history=[],position=0,index=0,selected=-1,drag=null,layout=null,handles=[],saving=false;
    const data=()=>history[position];
    const notice=message=>{$('boundary-message').textContent=message;};
    const recipe=next=>{const root=base.recipe;return {algorithm:MANUAL_ALGORITHM,seed:root.seed,style:root.style,shapeSeeds:root.shapeSeeds,base:root,edits:gridEdits(base,next)};};
    function controls(){
        $('boundary-undo').disabled=saving||position<=0;$('boundary-redo').disabled=saving||position>=history.length-1;
        $('boundary-save').disabled=saving||!gridEdits(history[0],data()).length;
        $('boundary-cancel').disabled=saving;$('boundary-region').disabled=saving;
        for(const id of ['boundary-coordinate','boundary-apply','boundary-minus','boundary-plus'])$(id).disabled=saving||selected<0;
        $('boundary-count').textContent=`${data().regions[index].id} · ${data().regions[index].boundary.length} 个拐点 · 与自动原轮次相差 ${gridEdits(base,data()).length} 格`;
    }
    function draw(){
        if(panel.hidden||!data())return;
        const rect=canvas.getBoundingClientRect(),width=rect.width,height=rect.height,dpr=devicePixelRatio||1;
        canvas.width=Math.round(width*dpr);canvas.height=Math.round(height*dpr);ctx.setTransform(dpr,0,0,dpr,0,0);ctx.fillStyle='#03060d';ctx.fillRect(0,0,width,height);
        const current=data(),r=current.regions[index],center=coordinates(toEquatorial(r.site)).longitude;
        const pts=r.boundary.map(([ra,dec])=>[center+delta(ra,center),dec]);
        const minX=Math.min(...pts.map(p=>p[0]))-5,maxX=Math.max(...pts.map(p=>p[0]))+5,minY=Math.min(...pts.map(p=>p[1]))-5,maxY=Math.max(...pts.map(p=>p[1]))+5;
        const ppd=Math.min((width-70)/(maxX-minX),(height-60)/(maxY-minY)),cx=(minX+maxX)/2,cy=(minY+maxY)/2;
        const project=(ra,dec)=>({x:width/2-(center+delta(ra,center)-cx)*ppd,y:height/2-(dec-cy)*ppd});
        layout={width,height,ppd,project,unproject:(x,y)=>({ra:cx+(width/2-x)/ppd,dec:cy+(height/2-y)/ppd})};
        ctx.save();ctx.beginPath();ctx.rect(32,16,width-48,height-42);ctx.clip();
        ctx.lineWidth=.5;
        for(let ra=Math.floor(minX);ra<=maxX;ra++){const p=project(ra,0);ctx.strokeStyle=ra%5?'rgba(150,170,195,.06)':'rgba(150,170,195,.16)';ctx.beginPath();ctx.moveTo(p.x,0);ctx.lineTo(p.x,height);ctx.stroke();}
        for(let dec=Math.floor(minY);dec<=maxY;dec++){const p=project(cx,dec);ctx.strokeStyle=dec%5?'rgba(150,170,195,.06)':'rgba(150,170,195,.16)';ctx.beginPath();ctx.moveTo(0,p.y);ctx.lineTo(width,p.y);ctx.stroke();}
        for(const [j,region] of current.regions.entries()){
            const points=region.boundary.map(([ra,dec])=>project(ra,dec));
            if(points.some((p,k)=>Math.abs(p.x-points[(k+1)%points.length].x)>ppd*180))continue;
            ctx.beginPath();points.forEach((p,k)=>k?ctx.lineTo(p.x,p.y):ctx.moveTo(p.x,p.y));ctx.closePath();
            ctx.fillStyle=j===index?'rgba(163,187,211,.085)':'rgba(145,164,186,.025)';ctx.fill();ctx.strokeStyle=j===index?'#a6b9ce':'#405167';ctx.lineWidth=j===index?1.3:.7;ctx.stroke();
        }
        const memberIds=new Set(r.members.map(s=>s.id)),byId=new Map(r.members.map(s=>[s.id,s]));
        ctx.strokeStyle='rgba(206,187,140,.44)';ctx.lineWidth=.8;
        for(const e of r.variants[1].edges){const points=arc(byId.get(e.from).direction,byId.get(e.to).direction,.25).map(v=>coordinates(toEquatorial(v))).map(c=>project(c.longitude,c.latitude));ctx.beginPath();points.forEach((p,i)=>i?ctx.lineTo(p.x,p.y):ctx.moveTo(p.x,p.y));ctx.stroke();}
        for(const s of stars)if(s.app_mag<=4.5){const c=coordinates(toEquatorial(s.direction)),p=project(c.longitude,c.latitude);if(p.x<0||p.x>width||p.y<0||p.y>height)continue;ctx.globalAlpha=memberIds.has(s.id)?1:.48;ctx.fillStyle=s.color_hex??'#dae3ed';ctx.beginPath();ctx.arc(p.x,p.y,Math.max(.7,2.5-s.app_mag*.38),0,Math.PI*2);ctx.fill();}ctx.globalAlpha=1;
        handles=[];
        r.boundary.forEach((a,j)=>{
            const b=r.boundary[(j+1)%r.boundary.length],p=project(...a),q=project(...b),horizontal=a[1]===b[1];
            const handle={edge:j,x:(p.x+q.x)/2,y:(p.y+q.y)/2,horizontal,coordinate:horizontal?a[1]:a[0],a:p,b:q};handles.push(handle);
            ctx.fillStyle=j===selected?'#e7c88f':'#738ba6';ctx.fillRect(handle.x-3,handle.y-3,6,6);
            if(j===selected){ctx.strokeStyle='#e7c88f';ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(p.x,p.y);ctx.lineTo(q.x,q.y);ctx.stroke();}
        });
        if(drag){const h=handles[drag.edge],a=r.boundary[drag.edge],b=r.boundary[(drag.edge+1)%r.boundary.length],p=h.horizontal?project(a[0],drag.target):project(drag.target,a[1]),q=h.horizontal?project(b[0],drag.target):project(drag.target,b[1]);ctx.setLineDash([5,4]);ctx.strokeStyle='#f1d8a5';ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(p.x,p.y);ctx.lineTo(q.x,q.y);ctx.stroke();ctx.setLineDash([]);}
        ctx.restore();ctx.font='10px system-ui';ctx.fillStyle='#8496ac';ctx.textAlign='center';
        for(let ra=Math.ceil(minX/5)*5;ra<=maxX;ra+=5){const p=project(ra,cy);ctx.fillText(`${wrap(ra)}°`,p.x,height-8);}
        ctx.textAlign='right';for(let dec=Math.ceil(minY/5)*5;dec<=maxY;dec+=5){const p=project(cx,dec);ctx.fillText(`${dec}°`,28,p.y+3);}
        ctx.textAlign='left';ctx.fillText('参考赤纬 ↑ · 赤经 ← · 1° 吸附',38,29);
        controls();
    }
    function select(edge){
        selected=edge;const h=handles[edge];$('boundary-axis').textContent=h?(h.horizontal?'赤纬位置':'赤经位置'):'边界位置';
        $('boundary-coordinate').value=h?.coordinate??'';draw();
    }
    function move(target){
        if(saving||selected<0)return;
        try{
            const cells=moveBoundaryEdge(data(),index,selected,target);
            const next=applyEditedGrid(base,stars,cells,recipe(data()));next.recipe=recipe(next);
            if(!gridEdits(data(),next).length){notice('边界位置没有变化。');return;}
            history=history.slice(0,position+1);history.push(next);position++;selected=-1;
            onPreview(next);notice('已应用。成员、连线、亮星归属、区域拓扑、黄道及余量检查通过。');draw();
        }catch(error){notice(`未应用：${error.message}`);draw();}
    }
    const pointer=e=>{const r=canvas.getBoundingClientRect();return {x:e.clientX-r.left,y:e.clientY-r.top};};
    canvas.onpointerdown=e=>{
        if(saving)return;const p=pointer(e);
        const closest=handles.map(h=>{const dx=h.b.x-h.a.x,dy=h.b.y-h.a.y,t=Math.max(0,Math.min(1,((p.x-h.a.x)*dx+(p.y-h.a.y)*dy)/(dx*dx+dy*dy)));return {...h,d:Math.hypot(p.x-h.a.x-t*dx,p.y-h.a.y-t*dy)};}).sort((a,b)=>a.d-b.d)[0];
        if(!closest||closest.d>12)return;select(closest.edge);drag={edge:closest.edge,target:closest.coordinate};canvas.setPointerCapture(e.pointerId);
    };
    canvas.onpointermove=e=>{if(!drag)return;const p=pointer(e),c=layout.unproject(p.x,p.y),h=handles[drag.edge];drag.target=Math.round(h.horizontal?c.dec:c.ra);draw();};
    canvas.onpointerup=()=>{if(!drag)return;const target=drag.target;drag=null;move(target);};
    canvas.onpointercancel=()=>{drag=null;draw();};
    $('boundary-region').onchange=e=>{index=Number(e.target.value);selected=-1;drag=null;draw();};
    $('boundary-apply').onclick=()=>move(Number($('boundary-coordinate').value));
    $('boundary-minus').onclick=()=>move(Number($('boundary-coordinate').value)-1);
    $('boundary-plus').onclick=()=>move(Number($('boundary-coordinate').value)+1);
    function travel(step){position+=step;selected=-1;onPreview(data());notice('已恢复编辑记录。');draw();}
    $('boundary-undo').onclick=()=>travel(-1);$('boundary-redo').onclick=()=>travel(1);
    $('boundary-cancel').onclick=()=>{panel.hidden=true;history=[];onClose();};
    $('boundary-save').onclick=async()=>{
        saving=true;controls();notice('正在验证并另存本轮…');
        try{await onSave(recipe(data()));panel.hidden=true;history=[];}
        catch(error){notice(`保存未完成：${error.message}`);}
        finally{saving=false;if(!panel.hidden)controls();}
    };
    new ResizeObserver(()=>draw()).observe(canvas);
    return {
        start(original,automatic,region){base=automatic;history=[structuredClone(original)];position=0;index=region;selected=-1;drag=null;saving=false;panel.hidden=false;
            $('boundary-region').replaceChildren(...original.regions.map((r,i)=>{const o=document.createElement('option');o.value=i;o.textContent=r.id;return o;}));$('boundary-region').value=index;
            notice('拖动亮色边界，或点选一条边后调整坐标。检查通过后才应用；保存会创建新轮次。');draw();panel.scrollIntoView({behavior:'smooth',block:'start'});},
        get active(){return !panel.hidden;},
        get status(){return {active:!panel.hidden,index,selectedEdge:selected,historyPosition:position,historyLength:history.length,
            changedCells:data()?gridEdits(base,data()).length:0,sessionChangedCells:data()?gridEdits(history[0],data()).length:0,handles:handles.map(({edge,x,y,horizontal,coordinate})=>({edge,x,y,horizontal,coordinate}))};}
    };
}
