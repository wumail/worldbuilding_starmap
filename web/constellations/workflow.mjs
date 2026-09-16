import {ALGORITHM, isManual, SUPPORTED_ALGORITHMS, DEFAULT_SEED, normalizeRecipe, redrawRecipe, equivalentDraw} from './generator.mjs?revision=favourites-regional-1';
import {mountCandidateEditor} from './candidate_editor.mjs?revision=favourites-regional-1';
import {FREE_EDIT_ALGORITHM} from './manual_figures.mjs';
import {DEFAULT_SAMPLING} from './sampling.mjs';

export const STORAGE_KEY='terrax-constellation-draws-v1';
const $=id=>document.getElementById(id);
const token=()=>crypto.getRandomValues(new Uint32Array(2)).join('-');
const notesOf=value=>Object.fromEntries(Array.from({length:15},(_,i)=>`Z${String(i+1).padStart(2,'0')}`).filter(id=>typeof value?.[id]==='string').map(id=>[id,value[id].slice(0,2000)]));
const locksOf=value=>Array.isArray(value)?[...new Set(value.filter(i=>Number.isInteger(i)&&i>=0&&i<15))].sort((a,b)=>a-b):[];

export function mountWorkflow({stars,baseline,onChange,onMetadata,getLayout,redraw,onEditing}) {
    const worker=new Worker(new URL('./generator_worker.mjs?revision=favourites-regional-1',import.meta.url),{type:'module'}),pending=new Map();
    worker.postMessage({type:'init',stars,meta:{catalogue:baseline.catalogue,sha256:baseline.sha256}});
    let records=[],current,region=0,currentData=baseline,busy=false,request=0,canStore=true,lastRegionKey='',editing=false,preview=null,nextRound=1,lastFavouriteId=null;
    const message=text=>{$('draw-status').textContent=text;};
    worker.onmessage=({data})=>{const p=pending.get(data.id);if(!p)return;if(data.progress){message(data.progress);return;}clearTimeout(p.timer);pending.delete(data.id);if(data.error)p.reject(Error(data.error));else p.resolve(data.result);};
    worker.onerror=()=>{for(const p of pending.values()){clearTimeout(p.timer);p.reject(Error('抽卡计算未能完成，请刷新页面后重试。'));}pending.clear();};
    function requestWorker(payload){return new Promise((resolve,reject)=>{const id=++request,timer=setTimeout(()=>{pending.delete(id);reject(Error('计算超时，当前方案已保留。请刷新页面后重试。'));},180000);pending.set(id,{resolve,reject,timer});worker.postMessage({id,...payload});});}
    const compute=recipe=>requestWorker({recipe});
    const initialRecord=()=>({id:'initial',title:'初稿 · 简洁与补充',recipe:null,locks:[],notes:{},favourite:false});
    function persist(){
        if(!canStore)return;
        // Only explicit favourites are persisted. The current unstarred draw
        // lives in memory, never as a hidden record or a dangling saved ID.
        const favourites=records.filter(r=>r.favourite);
        if(!favourites.some(r=>r.id===lastFavouriteId))lastFavouriteId=favourites.at(-1)?.id??null;
        try{localStorage.setItem(STORAGE_KEY,JSON.stringify({schema:2,algorithm:ALGORITHM,sha256:baseline.sha256,current:lastFavouriteId,nextRound,records:favourites}));
            $('persistence-status').textContent=`本地仅保留 ${favourites.length} 个收藏方案。${current?.favourite?'当前方案已收藏。':'当前方案未收藏，仅在本页临时保留；刷新前请收藏或导出。'}`;}
        catch{canStore=false;$('persistence-status').textContent='浏览器留存不可用；本次仍可操作，请导出要保留的整轮文件。';}
    }
    function normalizeRecord(r){
        if(typeof r.id!=='string'||r.id.length>150)throw Error('留存编号无效。');
        return {id:r.id,title:String(r.title??'留存方案').slice(0,120),recipe:r.recipe?normalizeRecipe(r.recipe):null,locks:locksOf(r.locks),notes:notesOf(r.notes),favourite:r.favourite===true};
    }
    function setBusy(on){
        busy=on;document.body.dataset.drawing=String(on);
        for(const id of ['new-round','reproduce','refine-round','rounds','original','import-round','import-file','seed','complexity','sampling-width','favourite','export-round','imagery','export','variant','limit','compare','lines','boundaries','previous','next'])$(id).disabled=on||editing;
        $('rounds').disabled=on||editing||!records.length;
        for(const b of $('region-tabs').children)b.disabled=on||editing;
        $('reroll').disabled=on||editing||!current?.recipe||isManual(current.recipe)||current.locks.length===15;
        $('lock-region').disabled=on||editing||!current?.recipe;
        for(const id of ['edit-boundary','edit-lines','regenerate-candidate'])$(id).disabled=on||!currentData.territories||!current?.recipe;
    }
    function refresh(){
        const placeholder=document.createElement('option');placeholder.value='';placeholder.disabled=true;placeholder.textContent=records.length?'选择收藏方案':'暂无收藏方案';
        $('rounds').replaceChildren(placeholder,...records.map(r=>{const option=document.createElement('option');option.value=r.id;option.textContent=`★ ${isManual(r.recipe)?'手动编辑 · ':r.recipe&&r.recipe.algorithm!==ALGORITHM?'旧分区 · ':''}${r.title}`;return option;}));
        $('rounds').value=current?.favourite?current.id:'';
        $('favourite').textContent=current?.favourite?'★ 已收藏本轮':'☆ 收藏本轮';$('favourite').setAttribute('aria-pressed',String(!!current?.favourite));
        const fine=current?.recipe?.algorithm==='terrax-zodiac-draw-3',clean=current?.recipe?.algorithm==='terrax-zodiac-draw-4',expanded=current?.recipe?.algorithm==='terrax-zodiac-draw-5',fitted=current?.recipe?.algorithm==='terrax-zodiac-draw-6';
        $('round-description').textContent=current?.recipe?`${current.recipe.seed} · ${isManual(current.recipe)?'手动编辑 · 已验证':[ALGORITHM,'terrax-zodiac-draw-8'].includes(current.recipe.algorithm)?'亮星已纳入 · 局部优先协调':current.recipe.algorithm==='terrax-zodiac-draw-7'?'总拐点最少 · 旧选星规则':fitted?'贴合星形 · 尚未求最少拐点':fine?'先选星形，再划天区 · 细格边界':clean?'简洁边界 · 旧黄道宽度规则':expanded?'旧扩张边界 · 黄道宽度已检查':'旧分区'} · ${current.recipe.style==='rich'?'丰富':'适中'} · 已锁定 ${current.locks.length} / 15 座`:'保留的初稿 · 88 颗主干星 / 117 颗扩展成员';
        $('layout-note').hidden=!current?.recipe||current.recipe.algorithm===ALGORITHM;
        $('layout-note').textContent=fine?'本轮保留了细格边界。“整理本轮边界”只合并小台阶，保留星形、黄道宽度、锁定与笔记。点击“按种子生成”可另建贴合星形的新版。':clean?'本轮保留旧黄道宽度，可能有多个窄区。点击“按种子生成”可另建同时检查宽度与星形余量的新版；已收藏的原轮次、锁定和笔记仍可回看。':expanded?'本轮保留旧扩张边界，部分边界可能远离星形。点击“按种子生成”可另建贴合星形的新版；成员可能改变，已收藏原轮中的锁定和笔记保留。':'本轮按旧规则保留。点击“按种子生成”可创建新版分区；已收藏的旧轮次、锁定和笔记仍可回看。';
        if(fitted)$('layout-note').textContent='本轮保留原有贴合边界，尚未求最少拐点。点击“按种子生成”可另建已证明最优的正交边界；已收藏的原轮次、锁定和笔记仍可回看。';
        if(current?.recipe?.algorithm==='terrax-zodiac-draw-7')$('layout-note').textContent='本轮保留旧版总数最优边界，尚未强制纳入区域重要亮星。点击“按种子生成”另建新版，或手动调整当前边界。';
        if(current?.recipe?.algorithm==='terrax-zodiac-draw-8')$('layout-note').textContent='本轮保留原有 ±30° 初始采样。拖动滑块可预览更宽采样带，点击“按种子生成”另建新版；也可以直接在候选窗口编辑本轮。';
        if(isManual(current?.recipe))$('layout-note').textContent='手动结果可以继续修改，或在候选窗口内重新生成本区；不沿用自动最优标记。满意后收藏，才会保留在本地列表。';
        $('refine-round').hidden=!fine;
        setBusy(busy);updateRegion(region);onMetadata?.();
    }
    function updateRegion(index){
        region=index;if(!current)return;
        const id=`Z${String(region+1).padStart(2,'0')}`,key=`${current.id}/${id}`;
        if(key!==lastRegionKey){$('imagery').value=current.notes[id]??'';lastRegionKey=key;}
        $('lock-region').checked=current.locks.includes(region);$('lock-region').disabled=busy||editing||!current.recipe;
        $('imagery-label').textContent=`${id} 的意象与修改笔记`;
    }
    async function activate(record,data){
        const result=data??(record.recipe?await compute(record.recipe):baseline);
        current=record;currentData=result;lastRegionKey='';if(record.favourite)lastFavouriteId=record.id;
        if(record.recipe){$('seed').value=record.recipe.seed;$('complexity').value=record.recipe.style;}
        $('sampling-width').value=result.sampling?.halfWidthDegrees??DEFAULT_SAMPLING;
        onChange(result);refresh();persist();
    }
    async function action(fn){
        if(busy||editing)return;setBusy(true);message('正在检查亮星与星形，随后协调各区边界…');
        try{await fn();}catch(error){message(error.message);}
        finally{setBusy(false);}
    }
    async function create(recipe,{locks=[],notes={},local=false,refined=false,manual=false}={}){
        recipe=normalizeRecipe(recipe);
        const result=await compute(recipe),round=nextRound++;
        const record={id:token(),title:`第 ${round} 轮 · ${recipe.seed}${manual?' · 手动编辑':refined?' · 边界整理':local?' · 区域内重抽':''}`,recipe,locks:[...locks],notes:{...notes},favourite:false};
        await activate(record,result);
        const loops=result.regions.filter(r=>r.structure.loops>0).length;
        message(manual&&recipe.algorithm===FREE_EDIT_ALGORITHM?'手动星形已应用；包围检查通过。点击收藏可保留本轮。':refined?'边界已整理，原成员、连线、锁定和笔记保持一致；满意后请收藏新结果。':`本轮 ${result.selectedExtendedCount} 颗成员，${loops} 座含闭合轮廓；生成检查通过。${local?' 分区与锁定星形保持一致。':''}满意后请收藏。`);
    }
    async function refine(record){
        if(record?.recipe?.algorithm!=='terrax-zodiac-draw-3')return;
        await create({...record.recipe,algorithm:'terrax-zodiac-draw-4'},{locks:record.locks,notes:record.notes,refined:true});
    }
    const editor=mountCandidateEditor({stars,getLayout,redraw,
        onPreview(next){preview=next;onChange(next);},
        onResample:(current,index)=>requestWorker({type:'sample-region',current,index,seed:`region-${index}-${token()}`,style:current.recipe?.style??'rich'}),
        onClose(saved){editing=false;preview=null;onEditing(null);onChange(currentData);refresh();message(saved?'修改已应用为未收藏的新结果，锁定与笔记已复制；请收藏要保留的方案。':'已取消编辑，恢复进入编辑前的候选。');},
        async onSave(recipe){
            setBusy(true);
            try{await create(recipe,{locks:current.locks,notes:current.notes,manual:true});}
            finally{setBusy(false);}
        }
    });
    function download(value,name){
        const url=URL.createObjectURL(new Blob([JSON.stringify(value,null,2)+'\n'],{type:'application/json'})),link=document.createElement('a');link.href=url;link.download=name;link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
    }
    function wire(){
        const startEdit=mode=>{if(editing){editor.setMode(mode);return;}return action(async()=>{
            const automatic=isManual(current.recipe)?await compute(current.recipe.base):currentData;
            editing=true;preview=currentData;onEditing(mode);editor.start(currentData,automatic,region,mode);setBusy(false);
        });};
        $('edit-boundary').onclick=()=>startEdit('boundary');$('edit-lines').onclick=()=>startEdit('lines');
        $('regenerate-candidate').onclick=async()=>{if(busy)return;if(!editing)await startEdit('lines');if(editing){editor.setMode('lines');await editor.regenerate();}};
        const parameters=()=>({style:$('complexity').value,samplingHalfWidthDegrees:Number($('sampling-width').value)});
        $('new-round').onclick=()=>action(()=>create({seed:`terrax-${crypto.getRandomValues(new Uint32Array(1))[0].toString(36)}`,...parameters()}));
        $('reproduce').onclick=()=>action(()=>create({seed:$('seed').value,...parameters()}));
        $('refine-round').onclick=()=>action(()=>refine(current));
        $('rounds').onchange=e=>action(async()=>{const record=records.find(r=>r.id===e.target.value);if(!record)return;await activate(record);message('已恢复收藏方案，可继续比较、编辑或填写意象笔记。');});
        $('original').onclick=()=>action(async()=>{await activate(records.find(r=>r.id==='initial')??initialRecord(),baseline);message('已打开内置初稿，可在收藏列表返回保留的方案。');});
        $('reroll').onclick=()=>action(async()=>{
            const recipe=redrawRecipe(current.recipe,current.locks,token()),notes=Object.fromEntries(current.locks.map(i=>{const id=`Z${String(i+1).padStart(2,'0')}`;return [id,current.notes[id]??''];}));
            await create(recipe,{locks:current.locks,notes,local:true});
        });
        $('lock-region').onchange=e=>{if(!current?.recipe)return;const set=new Set(current.locks);if(e.target.checked)set.add(region);else set.delete(region);current.locks=[...set].sort((a,b)=>a-b);refresh();persist();};
        $('imagery').oninput=e=>{const id=`Z${String(region+1).padStart(2,'0')}`;current.notes[id]=e.target.value.slice(0,2000);persist();};
        $('favourite').onclick=()=>{
            current.favourite=!current.favourite;records=records.filter(r=>r.id!==current.id);
            if(current.favourite){records.push(current);lastFavouriteId=current.id;}
            refresh();persist();message(current.favourite?'已加入收藏，本地会保留本轮及笔记。':'已取消收藏；当前画面仍可使用，刷新后不保留此临时方案。');
        };
        $('export-round').onclick=()=>{
            download({format:'terrax-constellation-round',formatVersion:1,record:current,data:currentData},`Terrax-${current.recipe?.seed.replace(/[^a-zA-Z0-9_-]/g,'_')||'initial'}-${current.id.replace(/[^a-zA-Z0-9_-]/g,'_')}.json`);
            message('已导出整轮：种子、成员、连线、边界、锁定状态和意象笔记。');
        };
        $('import-round').onclick=()=>$('import-file').click();
        $('import-file').onchange=e=>{
            const file=e.target.files[0];e.target.value='';if(!file)return;
            action(async()=>{
                if(file.size>15000000)throw Error('文件过大，请选择本页导出的单轮候选文件。');
                const value=JSON.parse(await file.text());
                if(value.format!=='terrax-constellation-round'||value.formatVersion!==1||value.data?.sha256!==baseline.sha256)throw Error('文件格式或源星表不匹配；当前记录未更改。');
                const record=normalizeRecord(value.record),rebuilt=record.recipe?await compute(record.recipe):baseline;
                if(!equivalentDraw(rebuilt,value.data))throw Error('文件中的成员或边界与种子复现结果不符，未导入。');
                record.id=token();record.title=`导入 · ${record.title}`.slice(0,120);if(record.favourite)records.push(record);await activate(record,rebuilt);message('已按种子复核并导入这一轮，锁定与笔记已恢复。');
            });
        };
    }
    return {
        async start(){
            let saved,previousCurrent;
            try{
                const text=localStorage.getItem(STORAGE_KEY);
                if(text){
                    saved=JSON.parse(text);if(![1,2].includes(saved.schema)||!SUPPORTED_ALGORITHMS.includes(saved.algorithm)||saved.sha256!==baseline.sha256||!Array.isArray(saved.records))throw Error('历史留存格式或星表已变化。');
                    // Filter before decoding obsolete unstarred recipes, so a
                    // discarded bad record cannot block valid favourites.
                    records=saved.records.filter(r=>r?.favourite===true).map(normalizeRecord);
                    const numbers=saved.records.map(r=>Number(String(r?.title??'').match(/第 (\d+) 轮/)?.[1]??0)).filter(n=>Number.isSafeInteger(n)&&n<Number.MAX_SAFE_INTEGER);
                    nextRound=Math.max(1,Number.isSafeInteger(saved.nextRound)?saved.nextRound:1,...numbers.map(n=>n+1));
                    if(saved.schema===1){const old=saved.records.find(r=>r?.id===saved.current);if(old&&!old.favourite)try{previousCurrent=normalizeRecord(old);}catch{}}
                    lastFavouriteId=records.find(r=>r.id===saved.current)?.id??records.at(-1)?.id??null;
                }
            }catch{canStore=false;records=[];$('persistence-status').textContent='历史收藏暂不能读取，原存储未覆盖；请将本次满意结果导出备份。';}
            wire();setBusy(true);persist();
            const selected=records.find(r=>r.id===saved?.current)??previousCurrent??records.find(r=>r.id===lastFavouriteId);
            try{
                const url=new URL(location.href),requestedSeed=url.searchParams.get('seed'),requestedRefine=url.searchParams.get('refine')==='1';
                if(requestedSeed!==null){await create({seed:requestedSeed,style:'rich',samplingHalfWidthDegrees:url.searchParams.has('sampling')?Number(url.searchParams.get('sampling')):DEFAULT_SAMPLING});url.searchParams.delete('seed');url.searchParams.delete('sampling');history.replaceState(null,'',url);}
                else if(requestedRefine&&selected?.recipe?.algorithm==='terrax-zodiac-draw-3')await refine(selected);
                else if(selected){await activate(selected);message('已恢复上次查看的一轮。');}
                else await create({seed:DEFAULT_SEED,style:'rich'});
                if(requestedRefine){url.searchParams.delete('refine');history.replaceState(null,'',url);}
            }catch(error){await activate(initialRecord(),baseline);message(error.message);}
            finally{setBusy(false);}
        },
        regionChanged:updateRegion,
        chooseStar:id=>editor.choose(id),
        paintEditor:(ctx,layout)=>editor.paint(ctx,layout),
        get editView(){return editor.view;},
        get status(){return {busy,editing,roundCount:records.length,favouriteCount:records.length,nextRound,canStore,currentId:current?.id,recipe:current?.recipe,locks:[...(current?.locks??[])],favourite:!!current?.favourite};},
        get editor(){return editor.status;},
        get record(){return current?structuredClone(current):null;},
        get data(){return structuredClone(preview??currentData);},
    };
}
