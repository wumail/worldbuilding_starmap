import {zodiacRegion} from '../shared/zodiac.mjs';
import {SYSTEM,TERRAX,MAX_PERIOD,EARTH_YEAR,LOCAL_DAY,INITIAL_ANGLES} from '../shared/solar_system.mjs';
import {observerSky} from './eye_model.mjs';
import {SkyClock} from './sky_state.mjs';
import {fmt,diskLight,phaseImage} from './sky_render.mjs';

export class SkyControls {
    constructor(page) {
        this.clock=new SkyClock(); this.lastPanel=0; this.lastSky=null;
        const host=document.createElement('div');host.id='sky-interface';
        host.innerHTML=`
        <header class="sky-header">
          <a class="sky-brand" href="../../index.html" title="选择星图版本"><span class="brand-star">✧</span><span>TERRAX V2 <small>肉眼观察 · 独立版本</small></span></a>
          <nav aria-label="星图视图"><a href="./sky_atlas.html" ${page===1?'aria-current="page"':''}>01 平面星图</a><a href="./star_map.html" ${page===2?'aria-current="page"':''}>02 沉浸天球</a></nav>
          <div class="header-actions"><a class="v1-link" href="../v1/${page===1?'sky_atlas':'star_map'}.html" target="_blank">V1 对照 ↗</a><button id="motion-config">初始配置</button><button id="motion-about">观测说明</button></div>
        </header>
        <aside class="sky-sidebar" aria-label="天体与观察设置">
          <div class="sidebar-heading"><span class="eyebrow">SOL SYSTEM</span><h1>此刻的天空</h1><p id="sky-condition">正在计算…</p></div>
          <div class="observer-settings">
            <label>观察方式<select id="observer-mode" aria-describedby="observer-note"><option value="center">全天星图 · 完整两半球</option><option value="surface">地表天空 · 地平线下遮挡</option></select></label>
            <p class="small-note" id="observer-note"></p>
            <div class="coordinate-fields"><label>纬度 °N<input id="observer-lat" type="number" min="-90" max="90" step="any" value="30"></label><label>经度 °E<input id="observer-lon" type="number" min="-180" max="180" step="any" value="0"></label></div>
            ${page===2?'<label>投影<select id="sky-projection"><option value="stereographic">立体投影 · 圆形参考线保持圆形</option><option value="perspective">透视投影 · 相机式视角</option></select></label>':''}
            <label class="check"><input id="daylight-toggle" type="checkbox" checked>大气与昼夜</label>
          </div>
          <div class="body-heading"><span>恒星系天体</span><small>点击定位并暂停</small></div>
          <div id="system-bodies"></div>
          <section id="body-card" aria-label="所选天体">
            <div class="body-title"><h2 id="body-name">Luna</h2><span id="body-state"></span></div>
            <div class="phase-row"><canvas id="phase-preview" width="96" height="96" aria-label="放大相位示意"></canvas><div><span id="body-phase"></span><small>放大相位示意<br>主图按角径与显示倍率绘制</small></div></div>
            <button id="body-inspect">放大细看</button><dl id="body-metrics"></dl><p class="small-note">地面星等含近似大气消光；相位示意单独放大。未计算月食。</p>
          </section>
          <figure class="angular-comparison">
            <figcaption>盘面大小对照 · 同倍放大</figcaption>
            <svg id="angular-comparison" viewBox="0 0 240 104" role="img" aria-label="Sol、Luna、Echo 的实际角直径对照，不含光晕">
              ${['Sol','Luna','Echo'].map((id,i)=>`<g data-scale-body="${id}"><circle cx="${40+i*80}" cy="33" r="0"></circle><text x="${40+i*80}" y="79">${id}</text><text class="angle-value" x="${40+i*80}" y="96"></text></g>`).join('')}
            </svg>
            <p class="small-note">比较完整盘面轮廓，不含光晕。全天图压缩了整个天空；放大后查看月相。</p>
          </figure>
          <details class="view-settings" open><summary>图层与轨迹</summary>
            <p class="small-note">星等控制光量，角径与显示倍率控制轮廓。光扩散在盘面遮挡之后计算；所有天体一起缩放。</p>
            <label class="check"><input id="markers-toggle" type="checkbox" checked>天体辅助标记与名称</label>
            <label class="check"><input id="grid-toggle" type="checkbox" checked>坐标网格与参考线</label>
            <label class="check"><input id="zodiac-toggle" type="checkbox" checked>黄道 15 天区</label>
            <label class="check"><input id="deep-sky-toggle" type="checkbox" checked>星团与星云弥散光</label>
            <p class="small-note">黄道带 ±30°，每区 24°，边界固定于参考历元。星云按面亮度绘制；关闭弥散光仍保留恒星所受的尘埃消光。</p>
            <label>所选天体轨迹<select id="trail-select"><option value="off">关闭</option><option value="day">26 小时 · 查看日周运动</option><option value="year">一个 Terrax 年 · 查看长期运动</option></select></label>
            <p class="small-note">虚线与空心圈是位置辅助，不表示肉眼可见。年轨迹扣除自转，与当前恒星背景对齐。</p>
          </details>
          <div id="page-tools"></div>
          <p class="small-note" id="catalog-status" role="status">正在加载背景恒星…</p>
        </aside>
        <footer class="sky-timeline" aria-label="模拟时间">
          <div class="time-heading"><div><span class="eyebrow">NEPTUNE-SOL 的一个公转周期</span><span class="time-number" id="elapsed-years">0.000</span><span class="time-unit"> / 140.49 地球年</span></div><span class="local-time" id="elapsed-local"></span></div>
          <input id="time-slider" aria-label="最大周期内的时间" type="range" min="0" max="${MAX_PERIOD}" step="0.00001" value="0">
          <div class="transport"><div class="transport-buttons"><button id="time-start" title="回到起点">↤ 起点</button><button id="time-back" title="后退 26 小时">−26h</button><button id="time-play" class="primary">播放</button><button id="time-forward" title="前进 26 小时">+26h</button><button id="time-end" title="到达终点">终点 ↦</button></div>
            <label class="speed-label">速度<select id="time-speed"><option value="${LOCAL_DAY/20}">26 小时 / 20 秒</option><option value="${LOCAL_DAY}">26 小时 / 秒</option><option value="${TERRAX.period/30}">1 Terrax 年 / 30 秒</option><option value="${EARTH_YEAR*5}">5 地球年 / 秒</option></select></label>
            <label class="day-label">第 <input id="time-days" type="number" min="0" max="${MAX_PERIOD}" step="any" value="0"> 地球日</label>
          </div><p class="time-note" id="time-note">第 0 日为可编辑的示例历元。终点自动暂停。</p>
        </footer>
        <dialog id="initial-dialog"><form id="initial-form"><div class="dialog-heading"><h2>可编辑初始配置</h2><button type="button" data-close="initial-dialog" aria-label="关闭初始配置">×</button></div>
          <p>参考文件确定了轨道大小、形状、倾角和周期。下列起始位置与轨道朝向采用展示初值，保存后同时应用于两页。</p>
          <p class="small-note">第 0 日没有绑定现实日期。默认 Terrax 位于近星点，零经线处于当地午夜；自转角偏移可以调整这一约定。升交点和近星点角均在各天体自己的参考面内定义。</p>
          <div class="table-scroll"><table><thead><tr><th>天体 / 轨道参考面</th><th>升交点 Ω (°)</th><th>近星点角 ω (°)</th><th>起始平近点角 M₀ (°)</th></tr></thead><tbody id="angle-rows"></tbody></table></div>
          <label>初始自转角偏移 (°)<input id="initial-spin" type="number" step="any" required value="0"></label>
          <p class="small-note">Terrax 的 Ω = 0 为参考面约定；ω = 283° 来自设定，保持固定。其余可编辑值均为展示假设，尚未成为世界观定稿。</p>
          <div class="dialog-actions"><button type="button" id="initial-reset">填入默认初值</button><button type="button" id="initial-export">导出已保存配置</button><button type="submit" class="primary">应用并保存</button></div>
        </form></dialog>
        <dialog id="about-dialog"><div class="dialog-heading"><h2>从 Terrax 看出去</h2><button data-close="about-dialog" aria-label="关闭观测说明">×</button></div>
          <p>V2 从 Terrax 地表观察，保留原版本的星表、轨道、月相、时间与初值。V2 与 V1 的观察设置各自保存。</p>
          <p>默认启用天体观看增强，便于在整幅星图中识别背景星、日月与行星。侧栏“天体显示大小”统一调整它们的大小；这是显示增强，侧栏角径和星等仍为物理值。增强后的完整暗面也会遮挡后方天体；查看实际角尺度或食象几何时，点击“肉眼尺度 1×”关闭增强。</p>
          <p>“肉眼尺度”依据屏幕尺寸、观看距离和校准标尺决定每一度占多少像素。默认 27 英寸、60 厘米只是可编辑的示例；用实物尺校准 5 厘米标尺后，投影中心的日月角大小才与当前观看条件对应。总览把整片天空压进窗口，明确属于缩略星图。</p>
          <p>恒星和行星先按视星等计算到达观察者的总光量，日月再由总光量、角面积及相位分配表面亮度。完整月面始终参与遮挡；暗面不会透出后方星体。月面先保留圆形覆盖与暗面遮挡，再做显示层的轻微扩散；背景星用星等控制点像。夜空保留暗蓝显示基调。亮点的光感范围不等于实体直径。</p>
          <p>地表模式加入空气质量路径消光、太阳高度驱动的近似天光，以及双月照亮天空的估计。较亮的天空会淹没暗星，完整点源数量以当前加载的星表为准。默认银河星表已采用 30,712 光年银心距（9,416.342 pc）。假设眼睛已经适应当前天空；没有模拟数十分钟的适应过程。</p>
          <p>肉眼尺度的透视投影对应平面屏幕观看几何；立体投影保留天球圆形，但离轴尺度仍有差异。平面页默认双半球地图，拖动平移、滚轮缩放；也可进入局部立体视图。两页均可拖动、缩放、点击天体，也可以输入完整 140.49 地球年内的时刻。</p>
          <p>这是具有物理量依据的观察近似。气溶胶、视力与屏幕参数可调整；眼睛光扩散采用归一化的近似核，尚未对具体观察者做心理物理实验。天空亮度模型借鉴地球观测，不能当作 Terrax 的精密气象或食象预测。</p>
          <p><a href="../../reference/母星最终设定.md" target="_blank">母星设定 ↗</a> · <a href="../../reference/恒星系最终设定.md" target="_blank">恒星系设定 ↗</a> · <a href="./data/validation.md" target="_blank">V2 算法、来源与验证 ↗</a></p>
        </dialog>`;
        document.body.append(host);
        this.$=id=>document.getElementById(id);
        for(const body of [SYSTEM.star,...SYSTEM.bodies.filter(b=>b.id!=='Terrax')]) {
            const button=document.createElement('button');button.className='body-row';button.dataset.body=body.id;
            button.innerHTML=`<span class="body-dot" style="background:${body.color}"></span><span>${body.name}</span><span class="body-reading"></span>`;
            button.onclick=()=>this.choose(body.id);this.$('system-bodies').append(button);
        }
        this.bind();this.clock.subscribe(()=>this.sync());this.sync();
    }
    choose(id) { this.clock.set({selected:id,playing:false}); window.dispatchEvent(new CustomEvent('sky-focus',{detail:id})); }
    bind() {
        const $=this.$,set=patch=>this.clock.set(patch);
        $('time-play').onclick=()=>set({playing:!this.clock.state.playing, ...(this.clock.time()>=MAX_PERIOD?{days:0}:{})});
        $('time-start').onclick=()=>set({days:0,playing:false});$('time-end').onclick=()=>set({days:MAX_PERIOD,playing:false});
        $('time-back').onclick=()=>set({days:this.clock.time()-LOCAL_DAY,playing:false});$('time-forward').onclick=()=>set({days:this.clock.time()+LOCAL_DAY,playing:false});
        $('time-slider').oninput=e=>set({days:Number(e.target.value),playing:false});
        $('time-days').oninput=e=>{ if(e.target.value!=='' && e.target.checkValidity())set({days:Number(e.target.value),playing:false}); };
        $('time-days').onchange=e=>{ if(e.target.value==='' || !e.target.checkValidity())e.target.value=fmt(this.clock.time(),6); };
        $('time-speed').onchange=e=>set({speed:Number(e.target.value)});
        $('observer-mode').onchange=e=>set({mode:e.target.value});
        if($('sky-projection'))$('sky-projection').onchange=e=>set({projection:e.target.value});
        $('body-inspect').onclick=()=>{set({playing:false});window.dispatchEvent(new CustomEvent('sky-inspect',{detail:this.clock.state.selected}));};
        for(const [id,key] of [['observer-lat','latitude'],['observer-lon','longitude']]) {
            $(id).oninput=e=>{if(e.target.value!=='' && e.target.checkValidity())set({[key]:Number(e.target.value)});};
            $(id).onchange=e=>{if(e.target.value==='' || !e.target.checkValidity())e.target.value=this.clock.state[key];};
        }
        for(const [id,key] of [['daylight-toggle','daylight'],['markers-toggle','markers'],['grid-toggle','grid'],['zodiac-toggle','zodiac'],['deep-sky-toggle','deepSky']]) $(id).onchange=e=>set({[key]:e.target.checked});
        $('trail-select').onchange=e=>set({trail:e.target.value});
        $('motion-config').onclick=()=>{ this.fillAngles(this.clock.state.angles,this.clock.state.spinPhase);$('initial-dialog').showModal(); };
        $('motion-about').onclick=()=>$('about-dialog').showModal();
        document.querySelectorAll('[data-close]').forEach(b=>b.onclick=()=>$(b.dataset.close).close());
        $('initial-reset').onclick=()=>this.fillAngles(INITIAL_ANGLES,0);
        $('initial-form').onsubmit=e=>{
            e.preventDefault();const angles={};
            for(const body of SYSTEM.bodies) { angles[body.id]={};for(const key of ['node','peri','mean']) angles[body.id][key]=Number(document.querySelector(`[data-angle-body="${body.id}"][data-angle-key="${key}"]`).value); }
            set({angles,spinPhase:Number($('initial-spin').value),playing:false});$('initial-dialog').close();
        };
        $('initial-export').onclick=()=>{
            const blob=new Blob([JSON.stringify({note:'角度为展示初值，轨道硬参数来自 reference；时间单位为地球日。',...this.clock.state,days:this.clock.time(),playing:false},null,2)],{type:'application/json'});
            const url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download='terrax-initial-config.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
        };
    }
    fillAngles(angles,spin) {
        const rows=this.$('angle-rows');rows.replaceChildren();
        for(const body of SYSTEM.bodies) {
            const tr=document.createElement('tr'),name=document.createElement('td');name.textContent=body.name;
            const small=document.createElement('small');small.textContent=body.kind==='moon'?'Terrax 赤道面':'Terrax 黄道面';name.append(small);tr.append(name);
            for(const key of ['node','peri','mean']) {
                const td=document.createElement('td'),input=document.createElement('input');input.type='number';input.step='any';input.required=true;input.value=angles[body.id][key];input.dataset.angleBody=body.id;input.dataset.angleKey=key;
                input.setAttribute('aria-label',`${body.id} ${key}`);input.disabled=body.canonicalPeri && key!=='mean';td.append(input);tr.append(td);
            } rows.append(tr);
        } this.$('initial-spin').value=spin;
    }
    sync() {
        const s=this.clock.state,$=this.$;
        $('observer-mode').value=s.mode;
        if($('sky-projection'))$('sky-projection').value=s.projection;
        $('observer-note').textContent=s.mode==='surface'?'只显示地平线上的半个天空。查看完整天球请选择“全天星图”。':'从 Terrax 质心看完整天球，无地面遮挡；查看日升月落可切换“地表天空”。';
        if(document.activeElement!==$('observer-lat'))$('observer-lat').value=s.latitude;
        if(document.activeElement!==$('observer-lon'))$('observer-lon').value=s.longitude;
        $('observer-lat').disabled=$('observer-lon').disabled=$('daylight-toggle').disabled=s.mode==='center';
        for(const [id,key] of [['daylight-toggle','daylight'],['markers-toggle','markers'],['grid-toggle','grid'],['zodiac-toggle','zodiac'],['deep-sky-toggle','deepSky']]) $(id).checked=s[key];
        $('trail-select').value=s.trail;$('time-speed').value=String(s.speed);
        if(!$('time-speed').value) { const option=new Option(`${fmt(s.speed,4)} 地球日 / 秒`,String(s.speed));$('time-speed').add(option);$('time-speed').value=String(s.speed); }
        this.lastPanel=0;
    }
    frame(now=performance.now()) {
        if(this.clock.time()>=MAX_PERIOD && this.clock.state.playing)this.clock.stopAtEnd();
        const days=this.clock.time();
        const sky=observerSky(days,this.clock.state);this.lastSky=sky;
        if(now-this.lastPanel>=100 || this.lastPanel===0) { this.updatePanel(sky);this.lastPanel=now; }
        return sky;
    }
    updatePanel(sky) {
        const s=this.clock.state,$=this.$,body=sky.bodies.find(b=>b.id===s.selected),sol=sky.bodies[0];
        $('elapsed-years').textContent=fmt(sky.days/EARTH_YEAR,3);
        $('elapsed-local').textContent=`${fmt(sky.days/TERRAX.period,3)} Terrax 年 · ${fmt(sky.days/LOCAL_DAY,2)} 个 26h 自转日`;
        $('time-slider').value=sky.days;if(document.activeElement!==$('time-days'))$('time-days').value=fmt(sky.days,6);
        $('time-play').textContent=s.playing?'暂停':sky.days>=MAX_PERIOD?'从头播放':'播放';
        $('time-note').textContent=sky.days>=MAX_PERIOD?'已到终点，自动暂停。此时并非所有天体共同复位。':s.playing && s.speed>LOCAL_DAY*3?'高速播放会跳过短暂天象；观察日升月落请降低速度。':'第 0 日为可编辑的示例历元。终点自动暂停。';
        $('sky-condition').textContent=s.mode==='center'?'质心全天 · 无地面遮挡':`Sol 高度 ${fmt(sol.altitude,1)}° · ${s.daylight?(sky.night===1?'夜空':sky.night===0?'白昼':'晨昏'):'已关闭昼夜明暗'}`;
        document.querySelectorAll('.body-row').forEach(row=>{
            const b=sky.bodies.find(b=>b.id===row.dataset.body);row.classList.toggle('active',b.id===s.selected);row.setAttribute('aria-pressed',String(b.id===s.selected));
            row.classList.toggle('below',!b.visible);row.querySelector('.body-reading').textContent=`${fmt(b.observedMagnitude,1)} 等 · ${b.aboveHorizon?(b.brightEnough?'达标':'偏暗'):'地平下'}`;
        });
        $('body-name').textContent=body.name;$('body-state').textContent=body.visible?'亮度达标':body.status;
        $('body-phase').textContent=body.id==='Sol'?'自发光恒星':`照明面积 ${fmt(body.illuminated*100,1)}%`;
        $('phase-preview').getContext('2d').putImageData(phaseImage(body,diskLight(body)),0,0);
        const coords=s.mode==='surface'?`<dt>方位 / 高度</dt><dd>${fmt(body.azimuth,2)}° / ${fmt(body.altitude,2)}°</dd>`:`<dt>赤经 / 赤纬</dt><dd>${fmt(body.ra,2)}° / ${fmt(body.dec,2)}°</dd>`;
        const region=zodiacRegion(body.equatorial);
        $('body-metrics').innerHTML=`<dt>大气外视星等</dt><dd>${fmt(body.magnitude,3)}</dd><dt>当前观测估计</dt><dd>${fmt(body.observedMagnitude,3)} 等</dd><dt>角直径</dt><dd>${fmt(body.angularDiameter*60,3)}′</dd><dt>距离</dt><dd>${body.kind==='moon'?`${Math.round(body.distance*149597870.7).toLocaleString()} km`:`${fmt(body.distance,3)} AU`}</dd>${coords}<dt>黄道天区</dt><dd>${region.label}${region.inBelt?'':' · 黄道带外'}</dd><dt>黄经 / 黄纬</dt><dd>${fmt(region.longitude,2)}° / ${fmt(region.latitude,2)}°</dd>`;
        for(const item of document.querySelectorAll('[data-scale-body]')) {
            const b=sky.bodies.find(b=>b.id===item.dataset.scaleBody),circle=item.querySelector('circle');
            circle.setAttribute('r',b.angularDiameter*55);circle.setAttribute('fill',b.color);
            circle.dataset.angularDiameter=String(b.angularDiameter);item.querySelector('.angle-value').textContent=`${fmt(b.angularDiameter*60,1)}′`;
        }
        if(document.documentElement.dataset.storageUnavailable)this.$('time-note').textContent='浏览器未允许保存设置；当前页仍可使用，跨页同步暂不可用。';
    }
    catalogStatus(stars,sky) { if(!this.catalogError)this.$('catalog-status').textContent=`背景星表 ${stars.toLocaleString()} 颗 · 当前达标 ${sky.toLocaleString()} 颗（随地平与当前天光筛选）。`; }
    clearError() {this.catalogError=null;this.$('catalog-status').classList.remove('error');}
    error(message) { this.catalogError=message;this.$('catalog-status').textContent=message;this.$('catalog-status').classList.add('error'); }
}
