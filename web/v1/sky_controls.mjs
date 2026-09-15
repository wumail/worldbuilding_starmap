import {zodiacRegion} from '../shared/zodiac.mjs';
import {SYSTEM,TERRAX,MAX_PERIOD,EARTH_YEAR,LOCAL_DAY,INITIAL_ANGLES,skyState} from '../shared/solar_system.mjs';
import {SkyClock} from './sky_state.mjs';
import {fmt,diskLight,phaseImage,DEFAULT_DISPLAY_SCALE,MAX_DISPLAY_SCALE} from '../shared/sky_render.mjs';

export class SkyControls {
    constructor(page) {
        this.clock=new SkyClock(); this.lastPanel=0; this.lastSky=null;
        const host=document.createElement('div');host.id='sky-interface';
        host.innerHTML=`
        <header class="sky-header">
          <a class="sky-brand" href="../../index.html" title="选择星图版本"><span class="brand-star">✧</span><span>TERRAX V1 <small>标准星图 · 从母星仰望</small></span></a>
          <nav aria-label="星图视图"><a href="./sky_atlas.html" ${page===1?'aria-current="page"':''}>01 平面星图</a><a href="./star_map.html" ${page===2?'aria-current="page"':''}>02 沉浸天球</a></nav>
          <div class="header-actions"><a href="../v2/${page===1?'sky_atlas':'star_map'}.html" target="_blank">V2 观看 ↗</a><button id="motion-config">初始配置</button><button id="motion-about">观测说明</button></div>
        </header>
        <aside class="sky-sidebar" aria-label="天体与观察设置">
          <div class="sidebar-heading"><span class="eyebrow">SOL SYSTEM</span><h1>此刻的天空</h1><p id="sky-condition">正在计算…</p></div>
          <div class="observer-settings">
            <label>观察方式<select id="observer-mode" aria-describedby="observer-note"><option value="center">全天星图 · 完整两半球</option><option value="surface">地表天空 · 地平线下遮挡</option></select></label>
            <p class="small-note" id="observer-note"></p>
            <div class="coordinate-fields"><label>纬度 °N<input id="observer-lat" type="number" min="-90" max="90" step="any" value="30"></label><label>经度 °E<input id="observer-lon" type="number" min="-180" max="180" step="any" value="0"></label></div>
            ${page===2?'<label>投影<select id="sky-projection"><option value="stereographic">立体投影 · 圆形参考线保持圆形</option><option value="perspective">透视投影 · 相机式视角</option></select></label>':''}
            <label class="check"><input id="daylight-toggle" type="checkbox" checked>近似昼夜明暗</label>
          </div>
          <div class="body-heading"><span>恒星系天体</span><small>点击定位并暂停</small></div>
          <div id="system-bodies"></div>
          <section id="body-card" aria-label="所选天体">
            <div class="body-title"><h2 id="body-name">Luna</h2><span id="body-state"></span></div>
            <div class="phase-row"><canvas id="phase-preview" width="96" height="96" aria-label="放大相位示意"></canvas><div><span id="body-phase"></span><small>放大相位示意<br>主图按角径与显示倍率绘制</small></div></div>
            <button id="body-inspect">放大观察</button><dl id="body-metrics"></dl><p class="small-note">星等与亮面未计遮挡、月食和大气消光。</p>
          </section>
          <figure class="angular-comparison">
            <figcaption>盘面大小对照 · 同倍放大</figcaption>
            <svg id="angular-comparison" viewBox="0 0 240 104" role="img" aria-label="Sol、Luna、Echo 的实际角直径对照，不含光晕">
              ${['Sol','Luna','Echo'].map((id,i)=>`<g data-scale-body="${id}"><circle cx="${40+i*80}" cy="33" r="0"></circle><text x="${40+i*80}" y="79">${id}</text><text class="angle-value" x="${40+i*80}" y="96"></text></g>`).join('')}
            </svg>
            <p class="small-note">比较完整盘面轮廓，不含光晕。全天图压缩了整个天空；放大后查看月相。</p>
          </figure>
          <details class="view-settings" open><summary>图层与轨迹</summary>
            <label>天体显示大小 <output id="display-scale-value"></output><input id="display-scale" type="range" min="1" max="${MAX_DISPLAY_SCALE}" step=".1"></label><p class="small-note">默认统一放大至 ${DEFAULT_DISPLAY_SCALE} 倍，便于整图观看；恒星光点、日月和行星同步调整。设为 1 倍可恢复原尺度，侧栏角径始终是真实值。</p>
            <label class="check"><input id="solar-glow-toggle" type="checkbox" checked>太阳光晕（亮光示意）</label>
            <label class="check"><input id="lunar-glow-toggle" type="checkbox" checked>月面明暗增强（不改变大小）</label>
            <label class="check"><input id="planet-glow-toggle" type="checkbox" checked>行星亮度增强（示意）</label>
            <p class="small-note">所有天体随画面同比缩放。月面按实际角径乘统一显示倍率；光点和太阳柔光属于亮度示意。</p>
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
          <p>两页共用恒星表、轨道初值、时间和观察地点。平面图将全天展开成两个半球；沉浸天球支持拖动转头、滚轮改变视场。观察者始终位于 Terrax。</p>
          <p>默认“全天星图”从 Terrax 质心观察完整两半球，不加地面遮挡和昼夜明暗，适合查看完整星空与行星逆行。“地表天空”包含 26 小时恒星自转、25° 轴倾、年周与地表视差，只显示地平线上的星点。地表初始地点 30°N、0°E 可编辑。</p>
          <p>总长 <strong>140.49 地球年 = 51,313.9725 地球日 ≈ 92.3629 Terrax 年</strong>，取最慢行星 Neptune-Sol 的公转周期。所有天体不会在这个时刻一起回到原位。</p>
          <p>轨道采用固定 Kepler 椭圆，行星绕 Sol、双月绕 Terrax；按参考文件的周期推进。星等用距离和 Lambert 相位计算。恒星和行星亮度光点用于识读，和日月轮廓一起随画面同比缩放，缩放不会改变它们之间的大小比例。关闭“行星亮度增强”可查看行星的盘面与盈亏。太阳、行星、月亮以实际角径为基准，再应用统一显示倍率。</p>
          <p>默认的 ${DEFAULT_DISPLAY_SCALE} 倍天体显示大小是识读增强，设为 1 倍可恢复原尺度。两页的背景星、行星和日月统一放大；面板角径始终是真实值。增强模式下的遮挡也使用放大轮廓，不能据此读取真实食分。</p>
          <p>Sol、Luna、Echo 与行星实体均按半径和距离得到实际角径，完整月面不随盈亏、星等或明暗增强开关改变；暗面同样遮挡后方星体。背景恒星和未分辨行星采用同一套星等光点，亮星较大、暗星较小。这些光点是经压缩的显示点像，不是恒星实体直径，也不是经过人眼标定的曝光。</p>
          <p>平面图是方位等距展开，实体盘面在边缘会随投影变形；沉浸页可选普通透视或立体投影。两者都从观察者原点向天球观察。立体投影让球面圆保持圆形，但离轴区域的比例仍会变化。天空盒采用普通透视时同样会出现离轴椭圆；更换球壳形状不会消除它。</p>
          <p>两页使用一致的角尺度基准。全天图把 360° 的天空压入窗口，Echo 可能不足一个像素；使用“放大观察”可检查盘面与盈亏。放大整个天空会同步放大背景星。太阳光晕只在实际日面附近叠加柔光，可关闭；月面明暗增强仅提高月牙亮面可辨识度，不扩大月面及遮挡边界。相位预览单独放大，大小对照图使用同一个放大比例。</p>
          <p>“亮度达标”只表示通过地平线与近似星等阈值。晴暗夜阈值为 6.5 等；昼夜明暗是可关闭的示意，未模拟当地天气、折射、月光与恒星眩光。天体相互遮挡仅按盘面覆盖绘制，星等不作食分修正；未模拟月食。不能用于精确食象预测。</p>
          <p>默认银河星表已采用设定的 30,712 光年银心距（9,416.342 pc），包含星团与星云。具体参数以所选星表为准。进动、章动正按双月倾角为第 0 日瞬时值的方案独立核验；本页仍用固定 Kepler 轨道，未加入多体扰动、光行时或恒星自行，不能视为高精度长期星历。</p>
          <p><a href="../../reference/母星最终设定.md" target="_blank">母星设定 ↗</a> · <a href="../../reference/恒星系最终设定.md" target="_blank">恒星系设定 ↗</a> · <a href="../../data/sky_motion_validation.md" target="_blank">计算与验证说明 ↗</a></p>
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
        for(const [id,key] of [['daylight-toggle','daylight'],['solar-glow-toggle','solarGlow'],['lunar-glow-toggle','lunarGlow'],['planet-glow-toggle','planetGlow'],['markers-toggle','markers'],['grid-toggle','grid'],['zodiac-toggle','zodiac'],['deep-sky-toggle','deepSky']]) $(id).onchange=e=>set({[key]:e.target.checked});
        $('display-scale').oninput=e=>set({displayScale:Number(e.target.value)});
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
        $('display-scale').value=s.displayScale;$('display-scale-value').textContent=`${s.displayScale.toFixed(1)}×`;
        if($('sky-projection'))$('sky-projection').value=s.projection;
        $('observer-note').textContent=s.mode==='surface'?'只显示地平线上的半个天空。查看完整天球请选择“全天星图”。':'从 Terrax 质心看完整天球，无地面遮挡；查看日升月落可切换“地表天空”。';
        if(document.activeElement!==$('observer-lat'))$('observer-lat').value=s.latitude;
        if(document.activeElement!==$('observer-lon'))$('observer-lon').value=s.longitude;
        $('observer-lat').disabled=$('observer-lon').disabled=$('daylight-toggle').disabled=s.mode==='center';
        for(const [id,key] of [['daylight-toggle','daylight'],['solar-glow-toggle','solarGlow'],['lunar-glow-toggle','lunarGlow'],['planet-glow-toggle','planetGlow'],['markers-toggle','markers'],['grid-toggle','grid'],['zodiac-toggle','zodiac'],['deep-sky-toggle','deepSky']]) $(id).checked=s[key];
        $('trail-select').value=s.trail;$('time-speed').value=String(s.speed);
        if(!$('time-speed').value) { const option=new Option(`${fmt(s.speed,4)} 地球日 / 秒`,String(s.speed));$('time-speed').add(option);$('time-speed').value=String(s.speed); }
        this.lastPanel=0;
    }
    frame(now=performance.now()) {
        if(this.clock.time()>=MAX_PERIOD && this.clock.state.playing)this.clock.stopAtEnd();
        const days=this.clock.time();
        const sky=skyState(days,this.clock.state);this.lastSky=sky;
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
            row.classList.toggle('below',!b.visible);row.querySelector('.body-reading').textContent=`${fmt(b.magnitude,1)} 等 · ${b.aboveHorizon?(b.brightEnough?'达标':'偏暗'):'地平下'}`;
        });
        $('body-name').textContent=body.name;$('body-state').textContent=body.visible?'亮度达标':body.status;
        $('body-phase').textContent=body.id==='Sol'?'自发光恒星':`照明面积 ${fmt(body.illuminated*100,1)}%`;
        $('phase-preview').getContext('2d').putImageData(phaseImage(body,diskLight(body)),0,0);
        const coords=s.mode==='surface'?`<dt>方位 / 高度</dt><dd>${fmt(body.azimuth,2)}° / ${fmt(body.altitude,2)}°</dd>`:`<dt>赤经 / 赤纬</dt><dd>${fmt(body.ra,2)}° / ${fmt(body.dec,2)}°</dd>`;
        const region=zodiacRegion(body.equatorial);
        $('body-metrics').innerHTML=`<dt>视星等</dt><dd>${fmt(body.magnitude,3)}</dd><dt>角直径</dt><dd>${fmt(body.angularDiameter*60,3)}′</dd><dt>距离</dt><dd>${body.kind==='moon'?`${Math.round(body.distance*149597870.7).toLocaleString()} km`:`${fmt(body.distance,3)} AU`}</dd>${coords}<dt>黄道天区</dt><dd>${region.label}${region.inBelt?'':' · 黄道带外'}</dd><dt>黄经 / 黄纬</dt><dd>${fmt(region.longitude,2)}° / ${fmt(region.latitude,2)}°</dd>`;
        for(const item of document.querySelectorAll('[data-scale-body]')) {
            const b=sky.bodies.find(b=>b.id===item.dataset.scaleBody),circle=item.querySelector('circle');
            circle.setAttribute('r',b.angularDiameter*55);circle.setAttribute('fill',b.color);
            circle.dataset.angularDiameter=String(b.angularDiameter);item.querySelector('.angle-value').textContent=`${fmt(b.angularDiameter*60,1)}′`;
        }
        if(document.documentElement.dataset.storageUnavailable)this.$('time-note').textContent='浏览器未允许保存设置；当前页仍可使用，跨页同步暂不可用。';
    }
    catalogStatus(stars,sky) { if(!this.catalogError)this.$('catalog-status').textContent=`背景星表 ${stars.toLocaleString()} 颗 · 当前达标 ${sky.toLocaleString()} 颗（全天阈值 6.5 等）。`; }
    clearError() {this.catalogError=null;this.$('catalog-status').classList.remove('error');}
    error(message) { this.catalogError=message;this.$('catalog-status').textContent=message;this.$('catalog-status').classList.add('error'); }
}
