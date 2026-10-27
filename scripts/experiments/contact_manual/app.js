/* No external libraries, fetch, training or Dataset writes. */
'use strict';
const C=window.ContactCore,M=window.CONTACT_MANIFEST,$=id=>document.getElementById(id);
const qa=new URLSearchParams(location.search).get('qa')==='1';
const storageKey='touch4d.manual-contact.001.v1'+(qa?'.qa':'');
let state=Object.fromEntries(M.episodes.map(e=>[e.episode,C.empty()]));
let episode=0,data=null,frame=0,loading=0,timer=null,yaw=.4,pitch=.35,rotation=null;
let geometryBounds=null;const loaded=[];
function notice(message){$('notice').textContent=message;}
try{const saved=localStorage.getItem(storageKey);if(saved)state=C.restore(JSON.parse(saved),M);}
catch(error){$('storage').textContent='无法读取浏览器保存记录，请用 JSON 恢复：'+error.message;}
if(qa)$('storage').textContent='浏览器验收模式：测试存储与正式标注完全隔离。';
function persist(){try{localStorage.setItem(storageKey,JSON.stringify(C.serialize(state,M)));if(!qa)$('storage').textContent='已自动保存到当前浏览器；建议下载 JSON 备份。';}
catch(error){$('storage').textContent='自动保存失败，请立即下载 JSON 备份：'+error.message;}}
function exportsUpdate(){
  const value=C.serialize(state,M);$('exportText').value=C.text(value);$('jsonPreview').textContent=JSON.stringify(value,null,2);$('csvPreview').textContent=C.csv(value);
  const n=Object.values(value.episodes).filter(a=>a.review_status==='MANUAL_CONFIRMED').length;
  $('progress').textContent=`${n}/10 Episodes Completed`;
  for(const item of M.episodes){const b=$('ep'+item.episode),done=value.episodes[item.episode].review_status==='MANUAL_CONFIRMED';b.classList.toggle('done',done);b.classList.toggle('active',item.episode===episode);b.textContent=`ep${item.episode} · ${done?'已确认':'未确认'}`;}
}
for(const item of M.episodes){const b=document.createElement('button');b.id='ep'+item.episode;b.onclick=()=>selectEpisode(item.episode);$('episodes').append(b);}
function stop(){if(timer!==null)clearTimeout(timer);timer=null;$('play').textContent='播放';}
function setReady(ok){for(const id of ['frame','frameSlider','start','end','setStart','setEnd','setEndN','play','prev','next','prev10','next10','signalStart','signalEnd'])$(id).disabled=!ok;if(!ok)$('confirm').disabled=true;}
async function selectEpisode(ep){
  stop();const request=++loading;episode=ep;data=null;setReady(false);exportsUpdate();$('episodeTitle').textContent=`ep${ep} · 加载中…`;notice('');
  const meta=M.episodes.find(x=>x.episode===ep);
  if(meta.status!=='READY'){notice('本 Episode 证据不完整：'+(meta.warnings||[]).join('; '));return;}
  try{
    if(!window.CONTACT_EPISODES[ep])await new Promise((resolve,reject)=>{const s=document.createElement('script');s.src=`episodes/ep${ep}/data.js`;s.onload=resolve;s.onerror=()=>reject(Error('无法加载本地证据；请参考 README 使用本地HTTP服务'));document.body.append(s);});
    if(request!==loading)return;
    data=window.CONTACT_EPISODES[ep];loaded.push(ep);
    while(loaded.length>3){const old=loaded.shift();if(old!==ep&&!loaded.includes(old))delete window.CONTACT_EPISODES[old];}
    if(data.rows.length!==meta.frames||data.timestamps.length!==meta.frames)throw Error('时间轴长度与manifest不一致');
    $('episodeTitle').textContent=`ep${ep} · ${data.frames} 帧 · ${(meta.span_s).toFixed(3)} s · ${ep<5?'单右手':'双手'}`;
    $('frame').max=data.frames-1;$('frameSlider').max=data.frames-1;$('start').max=data.frames-1;$('end').max=data.frames;
    $('start').value=state[ep].start??'';$('end').value=state[ep].end??'';$('notes').value=state[ep].notes;
    $('rgb').replaceChildren();
    data.cameras.forEach((camera,i)=>{const figure=document.createElement('figure'),img=document.createElement('img'),cap=document.createElement('figcaption');img.id='rgb'+i;img.alt=camera+' 当前真实RGB';img.onerror=()=>notice(`RGB图片缺失：ep${episode} f${frame} ${camera}`);cap.textContent=camera;figure.append(img,cap);$('rgb').append(figure);});
    const lo=[Infinity,Infinity,Infinity],hi=[-Infinity,-Infinity,-Infinity];
    for(const g of data.geometry)for(const p of g.xyz.concat(g.controllers.flatMap(c=>c.xyz.filter((_,i)=>c.mask[i]))))for(let i=0;i<3;i++){lo[i]=Math.min(lo[i],p[i]);hi[i]=Math.max(hi[i],p[i]);}
    geometryBounds={center:lo.map((v,i)=>(v+hi[i])/2),scale:290/Math.max(...hi.map((v,i)=>v-lo[i]))};
    if(data.warnings.length)notice(data.warnings.join('; '));
    setReady(true);showFrame(0);annotationUpdate(false);
  }catch(error){if(request===loading)notice(error.message);}
}
const imagePath=(f,camera=data.cameras[0])=>`episodes/ep${episode}/rgb/${camera}/${String(f).padStart(6,'0')}.jpg`;
const fmt=(x,d=2)=>x===null||x===undefined?'缺测':Number(x).toFixed(d);
function showFrame(f){if(!data)return;frame=Math.max(0,Math.min(data.frames-1,Math.round(f)));$('frame').value=frame;$('frameSlider').value=frame;
  const r=data.rows[frame];$('frameTitle').textContent=`原始帧 ${frame} / ${data.frames-1} · ${r.time_s.toFixed(3)}s · ${r.timestamp_us}μs`;
  data.cameras.forEach((c,i)=>$('rgb'+i).src=imagePath(frame,c));drawGeometry();drawHeat();drawTimeline();
  $('numbers').replaceChildren();
  for(const h of r.hands){const a=document.createElement('article');a.textContent=`${h.hand===0?'左手':'右手'} · opening ${fmt(h.opening_mm)} mm · speed ${fmt(h.speed_mm_s)} mm/s · Controller step RMS ${fmt(h.controller_step_rms_mm)} mm · 最近/中位 GS center 距离 ${fmt(h.min_mm)} / ${fmt(h.median_mm)} mm`;
    for(let j=0;j<2;j++){const s=h.fingers[j],p=document.createElement('p');p.textContent=`指${j}: mean ${fmt(s.mean,3)} · max ${fmt(s.maximum)} · 正响应 ${s.positive_count} · 有效 ${s.valid_count}/384 · Age ${fmt(s.age_ms,3)} ms`;a.append(p);} $('numbers').append(a);}
  if($('neighbors').parentElement.open)drawNeighbors();
}
function imageCard(f,title){const figure=document.createElement('figure'),img=document.createElement('img'),cap=document.createElement('figcaption');
  cap.textContent=title;if(f>=0&&f<data.frames){img.src=imagePath(f);img.alt=`ep${episode} frame ${f}`;img.onclick=()=>showFrame(f);figure.append(img);}else{const p=document.createElement('p');p.textContent='无该帧（Episode 边界）';figure.append(p);}figure.append(cap);return figure;}
function drawNeighbors(){if(!data)return;$('neighbors').replaceChildren();for(let f=Math.max(0,frame-5);f<=Math.min(data.frames-1,frame+5);f++)$('neighbors').append(imageCard(f,`f${f}`));}
function drawComparison(){if(!data)return;$('comparison').replaceChildren();const a=state[episode];for(const [title,v] of [['Start',a.start],['End',a.end]])if(v!==null){$('comparison').append(imageCard(v-1,`${title} 前：f${v-1}`),imageCard(v,`${title} 后：f${v}${v===data.frames?'（末端之外）':''}`));}}
function annotationUpdate(save=true){if(!data)return;const a=state[episode],ok=C.valid(a.start,a.end,data.frames),s=C.stats(a.start,a.end,data.timestamps);
  $('reviewStatus').textContent=a.confirmed&&ok?'MANUAL_CONFIRMED':'UNREVIEWED（草稿不等于确认）';$('confirm').disabled=!ok;
  $('counts').textContent=s?`接触前 ${s.before} 帧 · 接触操作 ${s.contact} 帧 [${a.start}, ${a.end}) · 接触后 ${s.after} 帧 · 完整 H10/Δ1 窗口 ${s.h10} 个 · `+(s.boundaryDuration===null?`末边界无时间戳；首末接触观测跨度 ${s.observedSpan.toFixed(3)}s`:`半开边界时长 ${s.boundaryDuration.toFixed(3)}s，首末接触观测跨度 ${s.observedSpan.toFixed(3)}s`):'尚未选择合法完整区间：必须为整数 0 ≤ Start < End ≤ 总帧数。';
  drawTimeline();drawComparison();exportsUpdate();if(save)persist();}
function editCut(which,v){if(!data)return;const a=state[episode];a[which]=v;a.confirmed=false;$(which).value=v??'';annotationUpdate();}
for(const id of ['start','end'])$(id).oninput=()=>{const raw=$(id).value;editCut(id,raw===''?null:Number(raw));};
$('setStart').onclick=()=>editCut('start',frame);$('setEnd').onclick=()=>editCut('end',frame);$('setEndN').onclick=()=>editCut('end',data.frames);
$('confirm').onclick=()=>{if(data&&C.valid(state[episode].start,state[episode].end,data.frames)){state[episode].confirmed=true;annotationUpdate();}};
$('unreview').onclick=()=>{state[episode].confirmed=false;annotationUpdate();};
$('clear').onclick=()=>{state[episode]=C.empty();$('start').value='';$('end').value='';$('notes').value='';annotationUpdate();};
$('notes').oninput=()=>{state[episode].notes=$('notes').value;exportsUpdate();persist();};
function drawTimeline(){if(!data)return;const cv=$('timeline'),ctx=cv.getContext('2d'),W=cv.width,H=cv.height,left=68,right=16,top=30,band=70;
  ctx.clearRect(0,0,W,H);const x=f=>left+f/data.frames*(W-left-right),a=state[episode];
  if(C.valid(a.start,a.end,data.frames)){ctx.fillStyle='#deeee7';ctx.fillRect(x(a.start),top,x(a.end)-x(a.start),H-top-24);}
  const colors=['#ab4a8e','#d18722','#216ca7','#2b947b'];
  const specs=[['压力均值','mean','finger'],['压力峰值','maximum','finger'],['正响应数','positive_count','finger'],['有效Taxels','valid_count','finger'],['Age ms','age_ms','finger'],['开口 mm','opening_mm','hand'],['速度 mm/s','speed_mm_s','hand'],['最近距离 mm','min_mm','hand']];
  for(let b=0;b<specs.length;b++){
    const [name,key,kind]=specs[b],base=top+b*band+band-12;
    const series=[];
    for(let h=0;h<data.hands.length;h++){if(kind==='finger')for(let j=0;j<2;j++)series.push(data.rows.map(r=>r.hands[h].fingers[j][key]));else series.push(data.rows.map(r=>r.hands[h][key]));}
    const max=Math.max(1,...series.flat().filter(v=>v!==null));ctx.strokeStyle='#d8e2e8';ctx.beginPath();ctx.moveTo(left,base);ctx.lineTo(W-right,base);ctx.stroke();
    ctx.fillStyle='#435b6b';ctx.font='12px system-ui';ctx.fillText(name,2,base-28);ctx.fillText(fmt(max,1),2,base-12);
    series.forEach((s,i)=>{ctx.strokeStyle=colors[i];ctx.beginPath();let active=false;for(let f=0;f<s.length;f++){if(s[f]===null){active=false;continue;}const xx=x(f),yy=base-s[f]/max*(band-20);if(active)ctx.lineTo(xx,yy);else ctx.moveTo(xx,yy);active=true;}ctx.stroke();});
  }
  ctx.setLineDash([4,5]);ctx.strokeStyle='#87939a';for(const r of data.references)for(const [s,e] of r.intervals)for(const f of [s,e]){ctx.beginPath();ctx.moveTo(x(f),top);ctx.lineTo(x(f),H-25);ctx.stroke();}ctx.setLineDash([]);
  ctx.strokeStyle='#3073be';ctx.beginPath();ctx.moveTo(x(frame),top);ctx.lineTo(x(frame),H-25);ctx.stroke();ctx.fillStyle='#223d50';
  for(let f=0;f<=data.frames;f+=Math.max(1,Math.round(data.frames/10)))ctx.fillText(String(f),x(f)-6,H-6);ctx.fillText(String(data.frames),W-35,H-6);
  ctx.fillText(data.hands.length===1?'右指0 / 右指1；手级曲线为右手':'左指0 / 左指1 / 右指0 / 右指1；手级曲线按左、右顺序',220,16);
  for(const id of ['start','end']){const el=$(id+'Handle'),v=a[id];el.hidden=v===null||!Number.isFinite(v);if(!el.hidden){el.style.left=(x(v)/W*100)+'%';el.setAttribute('aria-valuenow',v);el.setAttribute('aria-valuemin','0');el.setAttribute('aria-valuemax',data.frames);}}
}
function eventFrame(e){const rect=$('timeline').getBoundingClientRect(),px=(e.clientX-rect.left)/rect.width*$('timeline').width;return Math.max(0,Math.min(data.frames,Math.round((px-68)/($('timeline').width-84)*data.frames)));}
$('timeline').onpointerdown=e=>{if(data)showFrame(Math.min(data.frames-1,eventFrame(e)));};
for(const id of ['start','end']){const el=$(id+'Handle');let dragging=false;
  el.onpointerdown=e=>{dragging=true;el.setPointerCapture(e.pointerId);e.preventDefault();e.stopPropagation();};
  el.onpointermove=e=>{if(dragging&&data)editCut(id,Math.min(id==='start'?data.frames-1:data.frames,eventFrame(e)));};
  el.onpointerup=()=>dragging=false;el.onpointercancel=()=>dragging=false;
  el.onkeydown=e=>{if(data&&['ArrowLeft','ArrowRight'].includes(e.key)){e.stopPropagation();e.preventDefault();editCut(id,Math.max(0,Math.min(id==='start'?data.frames-1:data.frames,state[episode][id]+(e.key==='ArrowLeft'?-1:1))));}};
}
function drawGeometry(){if(!data)return;const cv=$('geometry'),ctx=cv.getContext('2d');ctx.clearRect(0,0,cv.width,cv.height);
  const {center,scale}=geometryBounds;function xy(p){let[x,y,z]=p.map((v,i)=>v-center[i]);return[320+(x*Math.cos(yaw)-y*Math.sin(yaw))*scale,195-(z*Math.cos(pitch)-(x*Math.sin(yaw)+y*Math.cos(yaw))*Math.sin(pitch))*scale];}
  ctx.fillStyle='#7b91a5';for(const p of data.geometry[frame].xyz){const[x,y]=xy(p);ctx.fillRect(x,y,2.5,2.5);}
  for(const c of data.geometry[frame].controllers)c.xyz.forEach((p,i)=>{if(!c.mask[i])return;const[x,y]=xy(p);ctx.fillStyle=c.hand===0?'#be4a88':'#167e9d';ctx.beginPath();ctx.arc(x,y,3,0,7);ctx.fill();});
  ctx.fillStyle='#274553';ctx.font='13px system-ui';ctx.fillText('左手=紫红 / 右手=青蓝 · world metres · 固定显示范围',10,20);
}
$('geometry').onpointerdown=e=>{rotation=[e.clientX,e.clientY];e.target.setPointerCapture(e.pointerId);};
$('geometry').onpointermove=e=>{if(rotation){yaw+=(e.clientX-rotation[0])*.01;pitch+=(e.clientY-rotation[1])*.01;rotation=[e.clientX,e.clientY];drawGeometry();}};
$('geometry').onpointerup=()=>rotation=null;
function drawHeat(){const cv=$('heat'),ctx=cv.getContext('2d');ctx.clearRect(0,0,cv.width,cv.height);const n=data.streams.length;
  for(let j=0;j<n;j++){const x0=(j%2)*340,y0=Math.floor(j/2)*170+28;ctx.fillStyle='#274553';ctx.font='13px system-ui';ctx.fillText(`${data.streams[j]<2?'左':'右'}手 指${data.streams[j]%2} · max色标 ${data.pressure_max}`,x0+4,y0-10);
    for(let k=0;k<384;k++){const v=data.tactile[frame][j][k],ok=data.valid[frame][j][k],r=Math.max(0,Math.min(1,v/Math.max(1,data.pressure_max)));ctx.fillStyle=ok?`rgb(${Math.round(18+237*Math.sqrt(r))},${Math.round(12+200*r*r)},${Math.round(30+40*r)})`:'#b5bbc0';ctx.fillRect(x0+(k%32)*10,y0+Math.floor(k/32)*10,10,10);}}
}
$('frame').onchange=()=>{const n=Number($('frame').value);if(Number.isInteger(n))showFrame(n);};$('frameSlider').oninput=()=>showFrame(Number($('frameSlider').value));
$('prev').onclick=()=>showFrame(frame-1);$('next').onclick=()=>showFrame(frame+1);$('prev10').onclick=()=>showFrame(frame-10);$('next10').onclick=()=>showFrame(frame+10);
$('neighbors').parentElement.ontoggle=drawNeighbors;
document.addEventListener('keydown',e=>{if(['INPUT','TEXTAREA'].includes(e.target.tagName)||e.ctrlKey||e.metaKey||e.altKey)return;if(['ArrowLeft','ArrowRight'].includes(e.key)){e.preventDefault();showFrame(frame+(e.key==='ArrowLeft'?-1:1)*(e.shiftKey?10:1));}});
$('play').onclick=()=>{if(timer!==null){stop();return;}function tick(){if(!data||frame>=data.frames-1){stop();return;}const delay=Math.max(20,(data.timestamps[frame+1]-data.timestamps[frame])/1000);showFrame(frame+1);timer=setTimeout(tick,delay);}$('play').textContent='暂停';tick();};
function referenceJump(end){if(!data)return;const r=data.references.flatMap(x=>x.intervals);if(!r.length){notice('没有可用自动参考，请直接浏览证据。');return;}showFrame(end?Math.max(...r.map(x=>x[1])):Math.min(...r.map(x=>x[0])));}
$('signalStart').onclick=()=>referenceJump(false);$('signalEnd').onclick=()=>referenceJump(true);
$('copy').onclick=async()=>{try{await navigator.clipboard.writeText($('exportText').value);notice('已复制实际标注。请粘贴发送给 Codex。');}catch(error){$('exportText').focus();$('exportText').select();const ok=document.execCommand('copy');notice(ok?'已复制标注结果。':'请手动复制已选中的文本。');}};
function download(name,body,type){const url=URL.createObjectURL(new Blob([body],{type}));const a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
$('downloadJSON').onclick=()=>download('contact_boundaries.json',JSON.stringify(C.serialize(state,M),null,2),'application/json');
$('downloadCSV').onclick=()=>download('contact_boundaries.csv',C.csv(C.serialize(state,M)),'text/csv;charset=utf-8');
$('import').onchange=async()=>{const file=$('import').files[0];if(!file)return;try{const candidate=C.restore(JSON.parse(await file.text()),M);state=candidate;persist();exportsUpdate();await selectEpisode(episode);notice('已恢复 JSON 中的标注与草稿，请检查。');}catch(error){notice('未导入：'+error.message);}finally{$('import').value='';}};
exportsUpdate();selectEpisode(0);
