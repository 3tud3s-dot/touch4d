'use strict';

function initRGB(figure) {
  const cases = window.DEMO_RGB.cases;
  const windowSelect = figure.querySelector('.rgb-window');
  const cameraSelect = figure.querySelector('.rgb-camera');
  const slider = figure.querySelector('input[type="range"]');
  const output = figure.querySelector('output');
  const image = figure.querySelector('img');
  const viewport = figure.querySelector('.rgb-viewport');
  const previous = figure.querySelector('.rgb-prev');
  const next = figure.querySelector('.rgb-next');
  for (const item of cases) windowSelect.add(new Option(item.label, item.key));
  window.DEMO_RGB.cameras.forEach((name, i) => cameraSelect.add(new Option(`${i + 1} · ${name.replace('brics-odroid-', '')}`, String(i))));
  windowSelect.value = figure.dataset.defaultCase;
  cameraSelect.value = '0';
  let request = 0;
  const cache = new Map();
  function cachedImage(path) {
    if (!cache.has(path)) {
      const task = new Promise((resolve, reject) => {
        const preview = new Image();
        preview.onload = () => resolve(preview);
        preview.onerror = () => reject(new Error('RGB frame could not be loaded.'));
        preview.src = path;
      });
      cache.set(path, task.catch(error => {cache.delete(path); throw error;}));
      if (cache.size > 6) cache.delete(cache.keys().next().value);
    }
    return cache.get(path);
  }
  async function render() {
    const ticket = ++request;
    const chosen = cases.find(c => c.key === windowSelect.value);
    const h = Number(slider.value), camera = Number(cameraSelect.value);
    const frame = chosen.views[camera].frames[h - 1];
    const frameId = chosen.start + h;
    previous.disabled = h === 1; next.disabled = h === 15;
    slider.setAttribute('aria-valuetext', `H${h}, frame ${frameId}`);
    output.textContent = `Loading H${h}…`;
    figure.setAttribute('aria-busy', 'true');
    try {
      const ready = await cachedImage(frame.path);
      if (ticket !== request) return;
      viewport.style.aspectRatio = `${frame.width} / ${frame.rgb_height}`;
      image.style.top = `${-100 * frame.top / frame.rgb_height}%`;
      image.src = ready.src;
      image.width = frame.width; image.height = frame.height;
      image.alt = `Ground truth, SoMA and G5-Direct: episode ${chosen.episode}, frame ${frameId}, ${window.DEMO_RGB.cameras[camera]}.`;
      output.textContent = `H${h} / 15 · Frame ${frameId}`;
      const caption = figure.querySelector('figcaption');
      caption.replaceChildren();
      const label = document.createElement('strong'); label.textContent = `${chosen.group}. `;
      caption.append(label, `Episode ${chosen.episode} · ${chosen.start} → ${frameId} · ${window.DEMO_RGB.cameras[camera].replace('brics-odroid-', '')}`);
      figure.dataset.renderedCase = chosen.key;
      figure.dataset.renderedCamera = String(camera);
      figure.dataset.renderedHorizon = String(h);
      figure.setAttribute('aria-busy', 'false');
      for (const adjacent of [h - 2, h]) {
        const item = chosen.views[camera].frames[adjacent];
        if (item) cachedImage(item.path).catch(() => {});
      }
    } catch (error) {if (ticket === request) {output.textContent = error.message; figure.setAttribute('aria-busy','false');}}
  }
  windowSelect.addEventListener('change', render);
  cameraSelect.addEventListener('change', render);
  slider.addEventListener('input', render);
  previous.onclick = () => {slider.value = Math.max(1, Number(slider.value) - 1); render();};
  next.onclick = () => {slider.value = Math.min(15, Number(slider.value) + 1); render();};
  render();
}
document.querySelectorAll('.rgb-comparison').forEach(initRGB);

for (const split of ['resimulation', 'generalization']) {
  const body = document.querySelector(`#${split}-table tbody`);
  const rows = window.DEMO_METRICS[split];
  const keys = ['mean_cd_mm', 'h15_cd_mm', 'rgb_mae', 'alpha_mae'];
  for (const row of rows) {
    const tr = document.createElement('tr');
    if (row.model === 'g5_e8') tr.className = 'ours';
    const name = document.createElement('td');
    name.textContent = row.model === 'g5_e8' ? 'Ours (G5-Direct)' : 'SoMA';
    tr.append(name);
    for (const [i, key] of keys.entries()) {
      const cell = document.createElement('td');
      const value = row[key].toFixed(i < 2 ? 3 : 5);
      if (row[key] === Math.min(...rows.map(r => r[key]))) {
        const strong = document.createElement('strong'); strong.textContent = value; cell.append(strong);
      } else cell.textContent = value;
      tr.append(cell);
    }
    body.append(tr);
  }
}

function loadScript(path) {
  return new Promise((resolve, reject) => {
    const script = document.createElement('script'); script.src = path;
    script.onload = resolve; script.onerror = () => reject(new Error('Scene data could not be loaded.'));
    document.head.append(script);
  });
}

async function initScene(section) {
  const status = section.querySelector('.loading');
  const slider = section.querySelector('input');
  const output = section.querySelector('output');
  const play = section.querySelector('.play');
  const canvases = [...section.querySelectorAll('canvas')];
  try {
    const embedded = document.getElementById(`scene-data-${section.dataset.key}`);
    let data;
    if (embedded) data = JSON.parse(embedded.textContent);
    else {await loadScript(section.dataset.source); data = window.DEMO_SCENES[section.dataset.key];}
    const panels = canvases.map(createSplatPanel);
    const initial = {yaw: -.20, pitch: .42, zoom: 1.12};
    let {yaw, pitch, zoom} = initial;
    let matrix = orbitMatrix(yaw, pitch), pending = false, loaded = false, token = 0, timer = null;
    let drag = null;
    const cache = new Map();
    function frame(id) {
      if (!cache.has(id)) cache.set(id, decodeSplat(data.assets[id]).catch(error => {cache.delete(id); throw error;}));
      return cache.get(id);
    }
    function schedule() {
      if (pending) return;
      pending = true;
      requestAnimationFrame(() => {pending = false; if (loaded) panels.forEach(p => p.draw(matrix, data.pivot, zoom));});
    }
    function stop() {if (timer !== null) clearInterval(timer); timer = null; play.textContent = 'Play';}
    async function show() {
      const current = ++token, h = Number(slider.value), index = h - 1;
      const frameId = data.frame_ids[index];
      output.textContent = `H${h} / 15 · Frame ${frameId}`;
      slider.setAttribute('aria-valuetext', `H${h}, frame ${frameId}`);
      try {
        const pair = await Promise.all(data.frames[index].map(frame));
        if (current !== token) return;
        panels.forEach((p, i) => p.setFrame(pair[i]));
        loaded = true; status.hidden = true; schedule();
        section.dataset.renderedHorizon = String(h);
        const next = data.frames[index + 1];
        if (next) next.forEach(id => frame(id).catch(() => {}));
      } catch (error) {stop(); status.hidden = false; status.classList.add('error'); status.textContent = error.message;}
    }
    function rotate(dx, dy) {yaw += dx * .009; pitch = Math.max(-1.55, Math.min(1.55, pitch + dy * .009)); matrix = orbitMatrix(yaw, pitch); schedule();}
    const resize = new ResizeObserver(schedule);
    for (const canvas of canvases) {
      resize.observe(canvas);
      canvas.addEventListener('pointerdown', e => {if (e.button !== 0 || drag) return; canvas.setPointerCapture(e.pointerId); canvas.focus({preventScroll:true}); drag = {id:e.pointerId, x:e.clientX, y:e.clientY}; canvas.classList.add('dragging');});
      canvas.addEventListener('pointermove', e => {if (drag?.id !== e.pointerId) return; rotate(e.clientX-drag.x, e.clientY-drag.y); drag.x=e.clientX; drag.y=e.clientY;});
      for (const event of ['pointerup','pointercancel','lostpointercapture']) canvas.addEventListener(event, e => {if (drag?.id === e.pointerId) {drag=null; canvas.classList.remove('dragging');}});
      canvas.addEventListener('wheel', e => {e.preventDefault(); const delta=e.deltaY*(e.deltaMode===1?16:e.deltaMode===2?180:1); zoom=Math.max(.45, Math.min(4.2,zoom*Math.exp(-delta*.0018))); schedule();}, {passive:false});
      canvas.addEventListener('keydown', e => {const change={ArrowLeft:[-12,0],ArrowRight:[12,0],ArrowUp:[0,-12],ArrowDown:[0,12]}[e.key]; if (change) {e.preventDefault(); rotate(...change);}});
      canvas.addEventListener('webglcontextlost', e => {e.preventDefault(); stop(); status.hidden=false; status.textContent='3D rendering was interrupted. Please reload the page.';});
    }
    section.querySelector('.reset').onclick = () => {({yaw,pitch,zoom}=initial); matrix=orbitMatrix(yaw,pitch); schedule();};
    slider.addEventListener('input', () => {stop(); show();});
    play.onclick = () => {
      if (timer !== null) {stop(); return;}
      if (Number(slider.value) === 15) {slider.value=1; show();}
      play.textContent='Pause';
      timer=setInterval(() => {slider.value=Number(slider.value)+1; show(); if (Number(slider.value)>=15) stop();}, 300);
    };
    document.addEventListener('visibilitychange', () => {if (document.hidden) stop();});
    await show();
  } catch (error) {
    status.classList.add('error'); status.textContent='Interactive 3D requires WebGL 2 and a recent browser. '+error.message;
    slider.disabled=true; play.disabled=true;
  }
}

const observer = new IntersectionObserver(entries => {
  for (const entry of entries) if (entry.isIntersecting) {observer.unobserve(entry.target); initScene(entry.target);}
}, {rootMargin:'300px'});
document.querySelectorAll('.scene').forEach(section => observer.observe(section));
