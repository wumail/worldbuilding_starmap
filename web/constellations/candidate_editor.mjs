import {vector,delta,arc} from './geometry.mjs?revision=candidate-editor-band-1';
import {fromEquatorial,unpackGrid} from './territories.mjs';
import {gridEdits} from './boundary_edits.mjs?revision=candidate-editor-band-1';
import {manualRecipe,applyManualFigures,editManualFigure,moveManualBoundary,regionRings,cornerCount,edgeKey} from './manual_figures.mjs';

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
    let base,baseCells,history=[],position=0,index=0,mode=null,selected=-1,drag=null,handles=[],edgeHits=[],starHits=[],saving=false,chosen=null;
    const data=()=>history[position],notice=message=>{$('candidate-message').textContent=message;};
    const issues=()=>data()?.manualEdits?.issues??[];
    const changed=()=>JSON.stringify(manualRecipe(base,history[0]))!==JSON.stringify(manualRecipe(base,data()));
    function controls(){
        onActivity?.();
        if(!mode)return;
        const r=data().regions[index],s=byId.get(chosen),member=r.members.some(s=>s.id===chosen),pending=issues();
        $('candidate-undo').disabled=saving||position<=0;$('candidate-redo').disabled=saving||position>=history.length-1;
        $('candidate-confirm').disabled=saving||!changed()||pending.length>0;$('candidate-cancel').disabled=saving;
        for(const id of ['boundary-coordinate','boundary-apply','boundary-minus','boundary-plus'])$(id).disabled=saving||selected<0;
        for(const id of ['candidate-tool','candidate-clear-star'])$(id).disabled=saving;
        $('regenerate-candidate').disabled=saving;
        $('candidate-add-member').disabled=saving||!s||member;$('candidate-remove-member').disabled=saving||!member;
        $('candidate-selected').textContent=s?`${s.id} · ${s.app_mag.toFixed(2)} 等 · ${member?'当前成员':'背景星'}`:'尚未选星';
        $('candidate-count').textContent=`${r.id} · ${r.members.length} 颗成员 · ${r.variants[1].edges.length} 条连线 · ${cornerCount(r)} 个拐点`;
        $('candidate-validation').textContent=pending.length?`尚不能确认：${pending[0]}${pending.length>1?`（共 ${pending.length} 处）`:''}。可继续修改星形或切换“编辑边界”。`:'包围检查通过；确认后应用，收藏可长期保留。';
        $('candidate-validation').classList.toggle('warning',pending.length>0);
        $('boundary-controls').hidden=mode!=='boundary';$('figure-controls').hidden=mode!=='lines';
        for(const [id,kind] of [['edit-boundary','boundary'],['edit-lines','lines']]){$(id).setAttribute('aria-pressed',String(mode===kind));$(id).disabled=saving;}
    }
    function publish(next){history=history.slice(0,position+1);history.push(next);position++;selected=-1;drag=null;onPreview(next);controls();redraw();}
    function move(target){
        if(saving||selected<0)return;
        try{
            const h=handles[selected],cells=moveManualBoundary(data(),index,h.ring,h.edge,target),recipe=manualRecipe(base,data());
            recipe.edits=[];for(let k=0;k<cells.length;k++)if(cells[k]!==baseCells[k])recipe.edits.push([k,cells[k]]);
            const result=applyManualFigures(base,stars,recipe,{draft:true});
            if(!gridEdits(data(),result).length){notice('边界位置没有变化。');redraw();return;}
            publish(result);notice('边界草稿已更新；确认前须包住所有星座的成员及连线。');
        }catch(error){notice(`未应用：${error.message}`);redraw();}
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
        handles=[];edgeHits=[];starHits=[];if(!mode||!data())return;
        const r=data().regions[index],project=layout.project;ctx.save();
        starHits=(layout.hits??[]).map(({star,x,y})=>({id:star.id,x,y,member:r.members.some(s=>s.id===star.id)}));
        if(mode==='boundary'){
            regionRings(r).forEach((ring,ringIndex)=>ring.forEach((a,j)=>{
                const b=ring[(j+1)%ring.length],horizontal=a[1]===b[1],points=edgePoints(a,b).map(project),mid=points[Math.floor(points.length/2)],slot=handles.length;
                handles.push({ring:ringIndex,edge:j,x:mid.x,y:mid.y,horizontal,coordinate:horizontal?a[1]:a[0],points});
                ctx.strokeStyle=slot===selected?'#e7c88f':'rgba(169,192,214,.68)';ctx.lineWidth=slot===selected?2:1;stroke(ctx,points);
                if(mid.visible!==false){ctx.fillStyle=slot===selected?'#e7c88f':'#8faac3';ctx.fillRect(mid.x-4,mid.y-4,8,8);}
            }));
            if(drag?.kind==='edge'){const h=handles[drag.handle],ring=regionRings(r)[h.ring];ctx.setLineDash([5,4]);ctx.strokeStyle='#f1d8a5';ctx.lineWidth=2;stroke(ctx,edgePoints(ring[h.edge],ring[(h.edge+1)%ring.length],drag.target).map(project));}
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
            const h=nearest(handles,h=>pathDistance(h.points,p));
            if(h?.d<=12){selected=h.slot;$('boundary-axis').textContent=h.horizontal?'赤纬位置':'赤经位置';$('boundary-coordinate').value=h.coordinate;
                e.preventDefault();drag={kind:'edge',handle:h.slot,target:h.coordinate};canvas.setPointerCapture(e.pointerId);controls();redraw();return;}
        }
    });
    canvas.addEventListener('pointermove',e=>{
        if(!drag||saving)return;const p=pointer(e);
        const c=getLayout().unprojectEquatorial(p.x,p.y),h=handles[drag.handle];drag.target=Math.round(h.horizontal?c.latitude:c.longitude);
        redraw();
    });
    canvas.addEventListener('pointerup',()=>{if(!drag)return;const d=drag;drag=null;if(d.kind==='edge')move(d.target);});
    canvas.addEventListener('pointercancel',()=>{drag=null;redraw();});canvas.addEventListener('lostpointercapture',()=>{drag=null;});
    const numeric=()=>{const value=$('boundary-coordinate').value;if(!value.trim())throw Error('请输入边界坐标');return Number(value);};
    function nudge(step){try{move(numeric()+step);}catch(error){notice(error.message);}}
    $('boundary-apply').onclick=()=>nudge(0);$('boundary-minus').onclick=()=>nudge(-1);$('boundary-plus').onclick=()=>nudge(1);
    $('candidate-tool').onchange=()=>{chosen=null;controls();redraw();};
    $('candidate-add-member').onclick=()=>edit({type:'add-member',id:chosen});
    $('candidate-remove-member').onclick=()=>edit({type:'remove-member',id:chosen});
    $('candidate-clear-star').onclick=()=>{chosen=null;controls();redraw();};
    function travel(step){if(saving||position+step<0||position+step>=history.length)return;position+=step;selected=-1;drag=null;onPreview(data());notice('已恢复草稿记录。');controls();redraw();}
    $('candidate-undo').onclick=()=>travel(-1);$('candidate-redo').onclick=()=>travel(1);
    function finish(saved){mode=null;history=[];handles=[];edgeHits=[];starHits=[];drag=null;panel.hidden=true;$('candidate-count').textContent='';canvas.classList.remove('editing');for(const id of ['edit-boundary','edit-lines'])$(id).setAttribute('aria-pressed','false');onClose(saved);}
    $('candidate-cancel').onclick=()=>{if(!saving)finish(false);};
    $('candidate-confirm').onclick=async()=>{
        if(saving||issues().length||!changed())return;saving=true;controls();notice('正在验证包围关系并应用本轮修改…');
        try{await onSave(manualRecipe(base,data()));finish(true);}catch(error){notice(`应用未完成：${error.message}`);}finally{saving=false;controls();}
    };
    function setMode(kind){if(saving)return;mode=kind;selected=-1;drag=null;chosen=null;controls();notice(kind==='boundary'?'拖动边界方块，或输入参考坐标。拖动空白处移动视野；当前草稿不受自动余量、黄道宽度限制。':'选星后加入或移除成员；选择“连接两颗星”可连续画线，“删除连线”可点线删除。拖动空白处移动视野。');redraw();}
    return {
        start(original,automatic,region,kind){
            base=automatic;baseCells=unpackGrid(base.territories);history=[structuredClone(original)];position=0;index=region;saving=false;
            $('candidate-tool').value='select';panel.hidden=false;canvas.classList.add('editing');setMode(kind);
        },setMode,choose,paint,regenerate,
        get active(){return !!mode;},get working(){return saving;},
        get status(){return {active:!!mode,working:saving,mode,index,selectedEdge:selected,selectedStar:chosen,tool:mode?$('candidate-tool').value:null,historyPosition:position,historyLength:history.length,
            changedCells:mode?gridEdits(base,data()).length:0,issues:mode?[...issues()]:[],view:mode?{center:getView().center,zoom:getView().zoom}:null,
            handles:handles.map(({ring,edge,x,y,horizontal,coordinate})=>({ring,edge,x,y,horizontal,coordinate})),starHits,
            edgeHits:edgeHits.map(({from,to,points})=>({from,to,points:points.filter(p=>p.visible!==false).map(p=>({x:p.x,y:p.y}))}))};}
    };
}
