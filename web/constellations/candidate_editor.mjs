import {vector,delta,arc,wrap} from './geometry.mjs?revision=candidate-editor-band-1';
import {fromEquatorial,unpackGrid} from './territories.mjs';
import {gridEdits} from './boundary_edits.mjs?revision=candidate-editor-band-1';
import {manualRecipe,applyManualFigures,editManualFigure,moveManualBoundary,regionRings,manualBoundaryRings,cornerCount,edgeKey} from './manual_figures.mjs';
import {shiftBoundarySegment,isBoundaryCorner,moveBoundaryCorner,removeBoundaryCorner,boundaryStep} from './boundary_geometry.mjs';

const $=id=>document.getElementById(id);
export function edgePoints(a,b,target=null){
    const horizontal=a[1]===b[1],da=delta(b[0],a[0]),dd=b[1]-a[1],n=Math.max(1,Math.ceil(Math.max(Math.abs(da),Math.abs(dd))/.4));
    return Array.from({length:n+1},(_,k)=>fromEquatorial(vector(horizontal||target===null?a[0]+da*k/n:target,!horizontal||target===null?a[1]+dd*k/n:target)));
}
function pathDistance(points,p){
    let distance=Infinity;
    for(let i=1;i<points.length;i++){
        const a=points[i-1],b=points[i];if(a.visible===false||b.visible===false||!Number.isFinite(a.x+a.y+b.x+b.y))continue;
        const dx=b.x-a.x,dy=b.y-a.y,length=dx*dx+dy*dy,t=length?Math.max(0,Math.min(1,((p.x-a.x)*dx+(p.y-a.y)*dy)/length)):0;
        distance=Math.min(distance,Math.hypot(p.x-a.x-t*dx,p.y-a.y-t*dy));
    }
    return distance;
}
export function mountCandidateEditor({stars,getLayout,getView,redraw,onActivity,onPreview,onClose,onSave,onResample}){
    const panel=$('candidate-editor'),canvas=$('detail'),byId=new Map(stars.map(s=>[s.id,s]));
    let base,baseCells,history=[],position=0,index=0,mode=null,selected=-1,selectedCorner=-1,drag=null,handles=[],vertices=[],edgeHits=[],starHits=[],saving=false,chosen=null;
    const data=()=>history[position],notice=message=>{$('candidate-message').textContent=message;};
    const issues=()=>data()?.manualEdits?.issues??[];
    const changed=()=>JSON.stringify(manualRecipe(base,history[0]))!==JSON.stringify(manualRecipe(base,data()));
    function controls(){
        onActivity?.();
        if(!mode)return;
        const r=data().regions[index],s=byId.get(chosen),member=r.members.some(s=>s.id===chosen),pending=issues(),locked=saving||!!drag;
        $('candidate-undo').disabled=locked||position<=0;$('candidate-redo').disabled=locked||position>=history.length-1;
        $('candidate-confirm').disabled=locked||!changed()||pending.length>0;$('candidate-cancel').disabled=locked;
        for(const id of ['boundary-coordinate','boundary-apply','boundary-minus','boundary-plus'])$(id).disabled=locked||selected<0;
        for(const id of ['boundary-ra','boundary-dec','boundary-point-apply','boundary-remove','boundary-join'])$(id).disabled=locked||selectedCorner<0;
        for(const id of ['candidate-tool','candidate-clear-star','boundary-tool','boundary-width'])$(id).disabled=locked;
        $('boundary-point-controls').hidden=selectedCorner<0;$('boundary-edge-controls').hidden=selectedCorner>=0;
        $('boundary-width-label').hidden=$('boundary-tool').value!=='insert';
        $('boundary-selection').textContent=selectedCorner>=0?'已选拐点 · 可拖动、输入坐标或删除':selected>=0?'已选边 · 可拖动或连续微调':'圆点是拐点，方块是整边；空白处可拖动视野';
        $('regenerate-candidate').disabled=locked;
        $('candidate-add-member').disabled=saving||!s||member;$('candidate-remove-member').disabled=saving||!member;
        $('candidate-selected').textContent=s?`${s.id} · ${s.app_mag.toFixed(2)} 等 · ${member?'当前成员':'背景星'}`:'尚未选星';
        $('candidate-count').textContent=`${r.id} · ${r.members.length} 颗成员 · ${r.variants[1].edges.length} 条连线 · ${cornerCount(r)} 个拐点`;
        $('candidate-validation').textContent=pending.length?`尚不能确认：${pending[0]}${pending.length>1?`（共 ${pending.length} 处）`:''}。可继续修改星形或切换“编辑边界”。`:'包围检查通过；确认后应用，收藏可长期保留。';
        $('candidate-validation').classList.toggle('warning',pending.length>0);
        $('boundary-controls').hidden=mode!=='boundary';$('figure-controls').hidden=mode!=='lines';
        for(const [id,kind] of [['edit-boundary','boundary'],['edit-lines','lines']]){$(id).setAttribute('aria-pressed',String(mode===kind));$(id).disabled=locked;}
    }
    function clearSelection(){selected=-1;selectedCorner=-1;for(const id of ['boundary-coordinate','boundary-ra','boundary-dec'])$(id).value='';$('boundary-axis').textContent='边界位置';}
    function selectEdge(slot){
        clearSelection();selected=slot;const h=handles[slot];$('boundary-axis').textContent=h.horizontal?'赤纬位置':'赤经位置';$('boundary-coordinate').value=h.coordinate;
    }
    function selectCorner(slot){
        clearSelection();selectedCorner=slot;const h=vertices[slot];$('boundary-ra').value=h.ra;$('boundary-dec').value=h.dec;
    }
    function restoreSelection(hint){
        if(!hint)return;
        if(hint.kind==='corner'){const slot=vertices.findIndex(h=>h.ra===wrap(hint.at[0])&&h.dec===hint.at[1]);if(slot>=0)selectCorner(slot);}
        else {const slot=handles.findIndex(h=>h.horizontal===hint.horizontal&&h.coordinate===(h.horizontal?hint.at[1]:wrap(hint.at[0]))&&
            (h.horizontal?Math.abs(delta(h.a[0],hint.at[0]))+Math.abs(delta(h.b[0],hint.at[0]))<=Math.abs(delta(h.b[0],h.a[0]))+1e-8:hint.at[1]>=Math.min(h.a[1],h.b[1])&&hint.at[1]<=Math.max(h.a[1],h.b[1])));if(slot>=0)selectEdge(slot);}
    }
    function publish(next,hint){history=history.slice(0,position+1);history.push(next);position++;clearSelection();drag=null;onPreview(next);restoreSelection(hint);controls();redraw();}
    function applyCells(cells,hint,message='边界草稿已更新；确认前须包住所有星座的成员及连线。'){
        const recipe=manualRecipe(base,data());recipe.edits=[];
        for(let k=0;k<cells.length;k++)if(cells[k]!==baseCells[k])recipe.edits.push([k,cells[k]]);
        const result=applyManualFigures(base,stars,recipe,{draft:true});
        if(!gridEdits(data(),result).length){notice('边界位置没有变化。');controls();redraw();return;}
        publish(result,hint);notice(message);
    }
    const edgeHint=(a,b,target)=>({kind:'edge',horizontal:a[1]===b[1],at:a[1]===b[1]?[wrap(a[0]+delta(b[0],a[0])/2),Math.round(target)]:[wrap(Math.round(target)),(a[1]+b[1])/2]});
    function move(target){
        if(saving||selected<0)return;
        try{
            const h=handles[selected];applyCells(moveManualBoundary(data(),index,h.ring,h.edge,target),edgeHint(h.a,h.b,target));
        }catch(error){notice(`未应用：${error.message}`);controls();redraw();}
    }
    function movePoint(target){
        if(saving||selectedCorner<0)return;
        try{const h=vertices[selectedCorner],cells=moveBoundaryCorner(unpackGrid(data().territories),index,regionRings(data().regions[index])[h.ring],h.point,target);
            applyCells(cells,{kind:'corner',at:target.map(Math.round)},'拐点已移动，相邻边保持横平竖直；重合的折点会自动合并。');
        }catch(error){notice(`未应用：${error.message}`);controls();redraw();}
    }
    function removePoint(){
        if(saving||selectedCorner<0)return;
        try{const h=vertices[selectedCorner];applyCells(removeBoundaryCorner(unpackGrid(data().territories),index,regionRings(data().regions[index])[h.ring],h.point,$('boundary-join').value),null,'已删除这处折角并合并相邻边。可撤销后尝试另一种合并方向。');}
        catch(error){notice(`未应用：${error.message}`);controls();redraw();}
    }
    function edit(action){
        if(saving)return;
        try{publish(editManualFigure(base,data(),stars,index,action));notice(action.type==='remove-member'?'已移除该成员及其所有关联连线；背景恒星仍在原位。':'星形草稿已更新。可继续增删成员、连线，或切换编辑边界。');}
        catch(error){notice(`未应用：${error.message}`);}
    }
    async function regenerate(){
        if(!mode||saving)return;saving=true;chosen=null;drag=null;controls();notice('正在当前星区内重新采样和生成星形…');
        try{
            const result=await onResample(data(),index);
            if(!result.changed){notice('本区暂未找到不同的组合；当前草稿保持原样，可扩大边界或手动修改。');return;}
            const recipe=manualRecipe(base,data());recipe.figures=recipe.figures.filter(f=>f.index!==index);recipe.figures.push(result.figure);
            publish(applyManualFigures(base,stars,recipe,{draft:true}));
            notice(`本区已重新生成：从 ${result.candidateCount} 颗候选中选取 ${result.figure.members.length} 颗成员。边界及其他星座保持原样；可继续重生成、手调、撤销或确认。`);
        }catch(error){notice(`未替换星形：${error.message}`);}
        finally{saving=false;controls();redraw();}
    }
    function stroke(ctx,points){ctx.beginPath();let started=false;for(const p of points){if(p.visible===false||!Number.isFinite(p.x+p.y)){started=false;continue;}if(started)ctx.lineTo(p.x,p.y);else ctx.moveTo(p.x,p.y);started=true;}ctx.stroke();}
    function paint(ctx,layout){
        handles=[];vertices=[];edgeHits=[];starHits=[];if(!mode||!data())return;
        const r=data().regions[index],project=layout.project;ctx.save();
        starHits=(layout.hits??[]).map(({star,x,y})=>({id:star.id,x,y,member:r.members.some(s=>s.id===star.id)}));
        if(mode==='boundary'){
            regionRings(r).forEach((ring,ringIndex)=>ring.forEach((a,j)=>{
                const b=ring[(j+1)%ring.length],horizontal=a[1]===b[1],points=edgePoints(a,b).map(project),mid=points[Math.floor(points.length/2)],slot=handles.length;
                handles.push({ring:ringIndex,edge:j,a,b,x:mid.x,y:mid.y,horizontal,coordinate:horizontal?a[1]:a[0],points});
                ctx.strokeStyle=slot===selected?'#e7c88f':'rgba(169,192,214,.68)';ctx.lineWidth=slot===selected?2:1;stroke(ctx,points);
                if(mid.visible!==false){ctx.fillStyle=slot===selected?'#e7c88f':'#8faac3';ctx.fillRect(mid.x-4,mid.y-4,8,8);}
                if(isBoundaryCorner(ring,j)){const p=project(fromEquatorial(vector(...a))),vertex=vertices.length;vertices.push({ring:ringIndex,point:j,ra:a[0],dec:a[1],x:p.x,y:p.y,visible:p.visible});
                    if(p.visible!==false&&Number.isFinite(p.x+p.y)){ctx.beginPath();ctx.arc(p.x,p.y,vertex===selectedCorner?6:4.5,0,Math.PI*2);ctx.fillStyle=vertex===selectedCorner?'#e7c88f':'#14283a';ctx.fill();ctx.strokeStyle=vertex===selectedCorner?'#f9dfac':'#9fc1d8';ctx.lineWidth=1.3;ctx.stroke();}}
            }));
            if(drag){
                ctx.setLineDash([5,4]);ctx.strokeStyle='#f1d8a5';ctx.lineWidth=2;
                for(const ring of drag.rings??[])stroke(ctx,ring.flatMap((p,i)=>edgePoints(p,ring[(i+1)%ring.length])).map(project));
                if(drag.error||!drag.rings?.length){ctx.setLineDash([]);ctx.font='12px sans-serif';ctx.fillStyle='#edbaa0';ctx.fillText(drag.error?'此位置无法直接移动；松开保持原样。':'此位置会清空本区，确认前仍须包住星座；可撤销。',14,24);}
            }
        }else{
            for(const e of r.variants[1].edges){const points=arc(byId.get(e.from).direction,byId.get(e.to).direction,.25).map(project);edgeHits.push({...e,points});ctx.strokeStyle='rgba(222,199,147,.65)';ctx.lineWidth=1.5;stroke(ctx,points);}
            const original=history[0].regions[index],remaining=new Set(r.variants[1].edges.map(e=>edgeKey(e.from,e.to)));
            ctx.setLineDash([3,5]);ctx.strokeStyle='rgba(220,132,122,.38)';ctx.lineWidth=1;
            for(const e of original.variants[1].edges)if(!remaining.has(edgeKey(e.from,e.to)))stroke(ctx,arc(byId.get(e.from).direction,byId.get(e.to).direction,.25).map(project));
            ctx.setLineDash([]);ctx.strokeStyle='rgba(164,200,218,.65)';ctx.lineWidth=.8;
            for(const s of r.members){const p=project(s.direction);if(p.visible){ctx.beginPath();ctx.arc(p.x,p.y,5,0,Math.PI*2);ctx.stroke();}}
        }
        if(chosen){const p=project(byId.get(chosen).direction);if(p.visible){ctx.setLineDash([]);ctx.strokeStyle='#efd197';ctx.lineWidth=1.4;ctx.beginPath();ctx.arc(p.x,p.y,9,0,Math.PI*2);ctx.stroke();}}
        ctx.restore();
    }
    function choose(id){
        if(mode!=='lines'||saving)return;
        if($('candidate-tool').value==='connect'&&chosen&&chosen!==id){const from=chosen;chosen=id;edit({type:'add-edge',from,to:id});}
        else {chosen=id;notice($('candidate-tool').value==='connect'?'已选起点，再点另一颗星连接；背景星会随连线加入星座。':'已选恒星。可加入星座或移除现有成员。');}
        controls();redraw();
    }
    const pointer=e=>{const r=canvas.getBoundingClientRect();return {x:e.clientX-r.left,y:e.clientY-r.top};};
    canvas.addEventListener('pointerdown',e=>{
        if(!mode||saving||e.button!==0||!e.isPrimary)return;const p=pointer(e);
        const nearest=(items,measure)=>items.map((h,j)=>({...h,slot:j,d:measure(h)})).sort((a,b)=>a.d-b.d)[0];
        if(mode==='lines'){
            if($('candidate-tool').value==='erase'){
                const hit=nearest(edgeHits,h=>pathDistance(h.points,p));if(hit?.d<=9){e.preventDefault();edit({type:'remove-edge',from:hit.from,to:hit.to});return;}
            }else {const hit=nearest(starHits,h=>Math.hypot(h.x-p.x,h.y-p.y));if(hit?.d<=10){e.preventDefault();choose(hit.id);return;}}
        }else {
            const v=nearest(vertices,h=>h.visible===false?Infinity:Math.hypot(h.x-p.x,h.y-p.y));
            if(v?.d<=10){selectCorner(v.slot);e.preventDefault();drag={kind:'corner',vertex:v.slot,target:[v.ra,v.dec],start:p,id:e.pointerId};canvas.focus({preventScroll:true});canvas.setPointerCapture(e.pointerId);previewDrag();controls();redraw();return;}
            const h=nearest(handles,h=>pathDistance(h.points,p));
            if(h?.d<=12){selectEdge(h.slot);e.preventDefault();canvas.focus({preventScroll:true});
                try{
                    if($('boundary-tool').value==='insert'){
                        const c=getLayout().unprojectEquatorial(p.x,p.y),step=boundaryStep(unpackGrid(data().territories),index,h.a,h.b,[c.longitude,c.latitude],Number($('boundary-width').value));
                        drag={kind:'insert',step,target:step.target,start:p,id:e.pointerId};
                    }else drag={kind:'edge',handle:h.slot,target:h.coordinate,start:p,id:e.pointerId};
                    canvas.setPointerCapture(e.pointerId);previewDrag();controls();redraw();
                }catch(error){notice(error.message);controls();redraw();}return;
            }
        }
    });
    // Preview is rebuilt from the exact same cells that pointerup will commit,
    // including merges, holes and vanished components after sweeping past an edge.
    function previewDrag(){
        try{
            const cells=unpackGrid(data().territories);
            if(drag.kind==='corner'){const h=vertices[drag.vertex];drag.cells=moveBoundaryCorner(cells,index,regionRings(data().regions[index])[h.ring],h.point,drag.target);}
            else {const h=drag.kind==='insert'?drag.step:handles[drag.handle];drag.cells=shiftBoundarySegment(cells,index,h.a,h.b,drag.target);}
            drag.rings=manualBoundaryRings(drag.cells,index);drag.error=null;
        }catch(error){drag.cells=null;drag.rings=[];drag.error=error.message;}
    }
    function updateDrag(e){
        if(!drag||saving||e.pointerId!==drag.id)return;const p=pointer(e);if(!drag.moved&&Math.hypot(p.x-drag.start.x,p.y-drag.start.y)<3)return;
        drag.moved=true;
        const c=getLayout().unprojectEquatorial(p.x,p.y);
        if(drag.kind==='corner')drag.target=[wrap(Math.round(c.longitude)),Math.max(-90,Math.min(90,Math.round(c.latitude)))];
        else {const horizontal=drag.kind==='insert'?drag.step.horizontal:handles[drag.handle].horizontal;drag.target=Math.round(horizontal?c.latitude:c.longitude);}
        previewDrag();redraw();
    }
    canvas.addEventListener('pointermove',updateDrag);
    function endDrag(){const d=drag;drag=null;if(d&&canvas.hasPointerCapture(d.id))canvas.releasePointerCapture(d.id);return d;}
    canvas.addEventListener('pointerup',e=>{
        if(!drag||e.pointerId!==drag.id)return;updateDrag(e);const d=endDrag();
        try{
            if(d.error)throw Error(d.error);
            const h=d.kind==='insert'?d.step:d.kind==='edge'?handles[d.handle]:null,hint=h?edgeHint(h.a,h.b,d.target):{kind:'corner',at:d.target};
            applyCells(d.cells,hint,d.kind==='insert'?`已插入一段 ${d.step.width}° 宽的折线。可继续拖动新拐点或边；轮廓保持直角。`:'边界草稿已更新，重合的折点会自动合并；确认前须包住所有星座。');
        }catch(error){notice(`未应用：${error.message}`);}
        controls();redraw();
    });
    canvas.addEventListener('pointercancel',()=>{endDrag();controls();redraw();});canvas.addEventListener('lostpointercapture',()=>{if(drag){drag=null;controls();redraw();}});
    const numeric=()=>{const value=$('boundary-coordinate').value;if(!value.trim())throw Error('请输入边界坐标');return Number(value);};
    function nudge(step){try{move(numeric()+step);}catch(error){notice(error.message);}}
    $('boundary-apply').onclick=()=>nudge(0);$('boundary-minus').onclick=()=>nudge(-1);$('boundary-plus').onclick=()=>nudge(1);
    $('boundary-tool').onchange=()=>{endDrag();clearSelection();controls();notice($('boundary-tool').value==='insert'?'点击边上的空段，插入一段浅折线；也可按住拖出凸起或凹口。宽度不足时会按当前边缩短。圆点仍可直接拖动。':'拖动圆点移动拐点，拖动方块或线段移动整条边；选中后可输入坐标。');redraw();};
    $('boundary-point-apply').onclick=()=>{
        if(!$('boundary-ra').value.trim()||!$('boundary-dec').value.trim()){notice('请输入赤经和赤纬。');return;}
        movePoint([Number($('boundary-ra').value),Number($('boundary-dec').value)]);
    };
    $('boundary-remove').onclick=removePoint;
    canvas.addEventListener('keydown',e=>{
        if(mode!=='boundary'||saving)return;
        if(e.key==='Escape'&&drag){e.preventDefault();endDrag();controls();redraw();return;}
        if(drag)return;
        if((e.ctrlKey||e.metaKey)&&e.key.toLowerCase()==='z'){e.preventDefault();travel(e.shiftKey?1:-1);return;}
        if(selectedCorner>=0){
            if(e.key==='Delete'||e.key==='Backspace'){e.preventDefault();removePoint();return;}
            const offset={ArrowLeft:[1,0],ArrowRight:[-1,0],ArrowUp:[0,1],ArrowDown:[0,-1]}[e.key];
            if(offset){e.preventDefault();const h=vertices[selectedCorner],step=e.shiftKey?10:1;movePoint([h.ra+offset[0]*step,h.dec+offset[1]*step]);}
        }
    });
    $('candidate-tool').onchange=()=>{chosen=null;controls();redraw();};
    $('candidate-add-member').onclick=()=>edit({type:'add-member',id:chosen});
    $('candidate-remove-member').onclick=()=>edit({type:'remove-member',id:chosen});
    $('candidate-clear-star').onclick=()=>{chosen=null;controls();redraw();};
    function travel(step){if(saving||position+step<0||position+step>=history.length)return;position+=step;clearSelection();endDrag();onPreview(data());notice('已恢复草稿记录。');controls();redraw();}
    $('candidate-undo').onclick=()=>travel(-1);$('candidate-redo').onclick=()=>travel(1);
    function finish(saved){endDrag();clearSelection();mode=null;history=[];handles=[];vertices=[];edgeHits=[];starHits=[];panel.hidden=true;$('candidate-count').textContent='';canvas.classList.remove('editing');for(const id of ['edit-boundary','edit-lines'])$(id).setAttribute('aria-pressed','false');onClose(saved);}
    $('candidate-cancel').onclick=()=>{if(!saving)finish(false);};
    $('candidate-confirm').onclick=async()=>{
        if(saving||issues().length||!changed())return;saving=true;controls();notice('正在验证包围关系并应用本轮修改…');
        try{await onSave(manualRecipe(base,data()));finish(true);}catch(error){notice(`应用未完成：${error.message}`);}finally{saving=false;controls();}
    };
    function setMode(kind){if(saving)return;endDrag();mode=kind;clearSelection();chosen=null;controls();notice(kind==='boundary'?'圆点可移动拐点，方块可移动整边；选择“新增拐点”后在边上点击或拖出局部折线。选中圆点后可删除，方向键微调，Ctrl / ⌘ Z 撤销。':'选星后加入或移除成员；选择“连接两颗星”可连续画线，“删除连线”可点线删除。拖动空白处移动视野。');redraw();}
    return {
        start(original,automatic,region,kind){
            base=automatic;baseCells=unpackGrid(base.territories);history=[structuredClone(original)];position=0;index=region;saving=false;
            $('candidate-tool').value='select';$('boundary-tool').value='move';panel.hidden=false;canvas.classList.add('editing');setMode(kind);
        },setMode,choose,paint,regenerate,
        get active(){return !!mode;},get working(){return saving||!!drag;},
        get status(){return {active:!!mode,working:saving||!!drag,mode,index,selectedEdge:selected,selectedCorner,selectedStar:chosen,tool:mode?$('candidate-tool').value:null,boundaryTool:mode?$('boundary-tool').value:null,historyPosition:position,historyLength:history.length,
            changedCells:mode?gridEdits(base,data()).length:0,issues:mode?[...issues()]:[],view:mode?{center:getView().center,zoom:getView().zoom}:null,
            preview:drag?{kind:drag.kind,target:drag.target,rings:drag.rings,error:drag.error}:null,
            handles:handles.map(({ring,edge,a,b,x,y,horizontal,coordinate})=>({ring,edge,a,b,x,y,horizontal,coordinate})),vertices,starHits,
            edgeHits:edgeHits.map(({from,to,points})=>({from,to,points:points.filter(p=>p.visible!==false).map(p=>({x:p.x,y:p.y}))}))};}
    };
}
