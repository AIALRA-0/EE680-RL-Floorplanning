(() => {
  const get = id => document.getElementById(id);
  const make = (tag, text, cls) => {
    const node = document.createElement(tag);
    if (text !== undefined) node.textContent = text;
    if (cls) node.className = cls;
    return node;
  };
  let state, input, busy = false, playing = false, timer, chapter = '01', page = 0;
  let inspected = null, manualPage = false;
  let detailFrames = [], detailIndex = 0, detailState = null, restoreDetail = true;
  const pageSize = 20;
  const phaseNames = {observe: "Read the current state", candidate: "Propose the next candidate", bounds: "Check all four rectangle boundaries",
    overlap: "Compare with each placed macro", mask: "Write this action-mask entry", reject: "Reject and advance the candidate index",
    choose: "Choose the first legal anchor", apply: "Execute the selected action", metrics: "Calculate Area and each net's HPWL",
    reward: "Explain this step's actual reward", next_observation: "Return the next observation", done: "Episode finished"};
  const pending = {
    '05': ["How can PPO and SA-SP use one evaluation protocol", [
      "Lucas owns the environment and PPO; Alarik owns Python SA-SP, scoring verification, baseline experiments and CPU profiling. Shared Python evaluation connects their exported layouts.",
      "Both solvers need the same rotation policy, lower-left endpoints, canvas constraints and Area/HPWL definitions. The comparison adapter disables rotation and uses the PPO canvas.",
      "Chapters 02-04 retain actual PPO training and replay. Chapter 05 uses independently scored exports from the frozen PPO model and the adapted SA-SP baseline."]]
  };

  function storage(key, value) {
    try {
      if (value === undefined) return localStorage.getItem('ee680:learning:' + key);
      localStorage.setItem('ee680:learning:' + key, value);
      return true;
    } catch (_) { return false; }
  }
  function pane(name) {
    if (name === 'source' || name === 'notes') inspector(name);
    document.querySelectorAll('[data-reading-pane]').forEach(node => node.classList.toggle('is-open', node.dataset.readingPane === name));
    document.querySelectorAll('[data-pane]').forEach(node => node.setAttribute('aria-pressed', String(node.dataset.pane === name)));
  }
  function inspector(name) {
    const selected = name === 'notes' ? 'notes' : 'source';
    document.documentElement.dataset.inspector = selected;
    document.querySelectorAll('button[data-inspector]').forEach(button => button.setAttribute('aria-pressed', String(button.dataset.inspector === selected)));
    storage('inspector', selected);
  }
  function showInput(circuit, name, raw) {
    get('input-name').textContent = `${name} · ${Object.keys(circuit.blocks).length} macros / ${circuit.nets.length} nets`;
    get('blocks').replaceChildren(...Object.entries(circuit.blocks).map(([name, size]) => make('p', `${name}: width ${size[0]}, height ${size[1]}`)));
    get('nets').replaceChildren(...circuit.nets.map((net, index) => make('p', `Net ${index}: ${net.join(', ')}`)));
    get('raw-input').closest('details').hidden = raw === undefined;
    if (raw !== undefined) get('raw-input').textContent = raw;
    get('parsed-input').textContent = JSON.stringify(circuit, null, 2);
  }
  window.addEventListener('ee680:ppo-input', event => showInput(event.detail.circuit, event.detail.name));
  window.addEventListener('ee680:ppo-source', event => {
    if (!input || !['02', '03', '04'].includes(chapter)) return;
    const {name, text} = event.detail;
    if (!Object.hasOwn(input.code, name)) { const option = make('option', name); option.value = name; get('code-file').append(option); }
    input.code[name] = text; get('code-file').value = name; renderCode();
  });
  function selectChapter(value, save = true) {
    chapter = value;
    document.querySelectorAll('[data-chapter]').forEach(node => {
      if (node.dataset.chapter === value) node.setAttribute('aria-current', 'page');
      else node.removeAttribute('aria-current');
    });
    get('chapter-00').hidden = value !== '00';
    get('chapter-01').hidden = value !== '01';
    const ppoChapter = ['02', '03', '04'].includes(value);
    get('phase-heading').hidden = ppoChapter; get('phase-list').hidden = ppoChapter;
    get('lesson-context').textContent = ppoChapter ? "PyTorch / actual training and inference" : "Python environment / individual checks";
    if (!ppoChapter && state) get('connection-status').textContent = `Actual Python environment / ${state.event_count} recorded events / reading progress is saved`;
    get('chapter-ppo').hidden = !ppoChapter;
    get('chapter-future').hidden = ppoChapter || !pending[value];
    if (ppoChapter) window.dispatchEvent(new CustomEvent('ee680:chapter', {detail: value}));
    if (!ppoChapter && pending[value]) {
      get('future-number').textContent = value + "  / integration";
      get('future-title').textContent = pending[value][0];
      get('future-body').replaceChildren(...pending[value][1].map(text => make('p', text)));
      const frame = make('iframe');
      frame.src = 'comparison.html'; frame.title = 'Actual PPO and SA-SP comparison';
      frame.style.cssText = 'width:100%;height:1400px;border:1px solid var(--line);border-radius:8px';
      get('future-body').append(frame);
    }
    get('note').value = storage('note:' + chapter) || '';
    get('note-status').textContent = "Notes stay in this browser and are stored separately for each chapter";
    if (input) {
      if (!ppoChapter) showInput(input.circuit, input.source, input.raw);
      get('code-file').value = value === '00' ? 'src/data/parse_blknet.py' : 'src/environment/placement_env.py';
      renderCode();
    }
    if (save) storage('chapter', value);
    pane('lesson');
  }
  async function api(url, command) {
    const response = await fetch(url, command ? {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({command})} : {});
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || String(response.status));
    return data;
  }
  function stop() { playing = false; clearTimeout(timer); get('play').textContent = "Play individual explanations"; }
  async function command(name) {
    if (busy) return;
    busy = true; controls(); get('operation-error').hidden = true;
    try {
      state = await api('/api/command', name);
      restoreDetail = false;
      inspected = null; manualPage = false; render(); rememberDetail();
      if (['choose', 'done'].includes(state.phase)) stop();
    } catch (error) {
      stop(); get('operation-error').hidden = false;
      get('operation-error').textContent = "The operation did not finish: " + error.message;
    } finally { busy = false; controls(); }
  }
  function controls() {
    if (!state) return;
    const ended = state.phase === 'done';
    const moreDetail = detailIndex < detailFrames.length - 1;
    const searching = ['observe', 'candidate', 'bounds', 'overlap', 'mask', 'reject'].includes(state.phase);
    ['advance', 'candidate', 'choose', 'module', 'play'].forEach(id => get(id).disabled = busy || (ended && (!moreDetail || !['advance', 'play'].includes(id))) || (['choose', 'candidate'].includes(id) && !searching));
    get('reset').disabled = busy;
    get('advance').textContent = moreDetail ? "Continue this explanation" : state.phase === 'choose' ? "Execute the selected position" : ended ? "Episode explanations completed" : "Open the next check";
    get('detail-prev').disabled = detailIndex === 0 || busy;
  }
  function errorText(error) {
    if (!error) return "Legal";
    if (error.includes('overlap')) return "Overlaps " + error.split(' ').at(-1) + " ";
    if (error.includes('outside')) return "The full rectangle extends outside the canvas";
    return error;
  }
  function currentFrame() { return detailFrames[detailIndex]; }
  function visibleState() {
    const transition = state.transitions.at(-1);
    return currentFrame()?.view === 'before' && transition
      ? {observation: transition.observation, info: transition.before_info, details: transition.before_details, reward: 0}
      : state;
  }
  function codeExcerpt(reference) {
    const wrapper = make('div', undefined, 'lesson-code');
    const link = make('a', reference.file); link.href = '/' + reference.file; link.target = '_blank';
    wrapper.append(link);
    const pre = make('pre');
    const lines = input.code[reference.file].split('\n');
    const methodStart = lines.findIndex(line => line.startsWith('    def ' + reference.method + '('));
    let methodEnd = methodStart + 1;
    while (methodStart >= 0 && methodEnd < lines.length && !/^    (def |@)/.test(lines[methodEnd])) methodEnd++;
    const found = new Set();
    reference.focus.forEach(token => {
      lines.forEach((line, index) => { if (index >= methodStart && index < methodEnd && line.includes(token)) found.add(index); });
    });
    [...found].sort((a, b) => a - b).forEach(index => {
      const row = make('span', undefined, 'code-line');
      row.append(make('span', String(index + 1).padStart(3) + '  ', 'line-number'));
      const line = lines[index];
      const matching = reference.focus.filter(token => line.includes(token)).sort((a, b) => b.length - a.length)[0];
      const offset = line.indexOf(matching);
      row.append(document.createTextNode(line.slice(0, offset)), make('mark', matching), document.createTextNode(line.slice(offset + matching.length)));
      pre.append(row);
    });
    wrapper.append(make('p', "Actual source code; numbers identify lines and highlighting identifies the expression being explained", 'small'), pre);
    return wrapper;
  }
  function rememberDetail() { storage('detail', JSON.stringify({event: state.current.number, index: detailIndex})); }
  function explain() {
    if (detailState !== state) {
      detailFrames = window.EE680PlacementLesson.build(state);
      detailIndex = 0;
      if (restoreDetail) {
        try {
          const saved = JSON.parse(storage('detail'));
          if (saved?.event === state.current.number && Number.isInteger(saved.index)) detailIndex = Math.max(0, Math.min(detailFrames.length - 1, saved.index));
        } catch (_) { /* Keep the first explanation when no saved reading position exists. */ }
      }
      restoreDetail = false; detailState = state;
    }
    const selected = currentFrame();
    get('stage-caption').textContent = `Python event ${state.current.number}  / current macro ${state.active_macro || "Finished"}`;
    get('stage-title').textContent = phaseNames[state.phase];
    get('detail-progress').textContent = `Explanation ${detailIndex + 1} / ${detailFrames.length} `;
    get('detail-prev').disabled = detailIndex === 0 || busy;
    get('stage-explanation').replaceChildren(make('h4', selected.title), ...selected.paragraphs.map(text => make('p', text)), codeExcerpt(selected.code));
    if (selected.mapping && state.attempt) {
      const a = state.attempt, w = state.observation.board.width;
      const strip = make('div', undefined, 'coordinate-mapping');
      for (const [label, value] of [["Candidate index i", a.index], ["Integer division gives y", Math.floor(a.index / w)], ["Remainder gives x", a.index % w]]) {
        const box = make('div'); box.append(make('span', label), make('strong', String(value))); strip.append(box);
      }
      get('stage-explanation').append(strip);
    }
    if (selected.mask) {
      const group = make('div', undefined, 'mask-strip');
      const index = state.attempt?.index || 0, start = Math.max(0, index - 3);
      for (let i = start; i < Math.min(state.mask.length, start + 8); i++) {
        const value = state.mask[i];
        const cell = make('div'); cell.append(make('span', `mask[${i}]`), make('strong', value === null ? "Unknown" : value ? 'True' : 'False'));
        if (i === index) cell.setAttribute('data-current', 'true'); group.append(cell);
      }
      get('stage-explanation').append(group);
    }
    get('observation-json').textContent = JSON.stringify(visibleState().observation, null, 2);
    if (chapter === '01') { get('code-file').value = selected.code.file; renderCode(); }
    get('score-details').open = state.phase === 'metrics' || state.phase === 'reward';
  }
  async function advanceReading() {
    if (busy) return;
    inspected = null; get('attempt-inspection').hidden = true;
    if (detailIndex < detailFrames.length - 1) {
      detailIndex++; explain(); board(); renderScore(); rememberDetail(); controls();
      if (state.phase === 'choose' && detailIndex === detailFrames.length - 1) stop();
    } else if (state.phase !== 'done') await command('advance');
    else stop();
  }
  function board() {
    if (!state) return;
    const obs = inspected?.observation || visibleState().observation;
    const candidate = inspected || (['candidate', 'bounds', 'overlap', 'mask', 'reject', 'choose'].includes(state.phase) || (state.phase === 'apply' && currentFrame()?.view === 'before') ? state.attempt : null);
    const svg = get('board'), width = svg.getBoundingClientRect().width;
    if (width < 50) return;
    const {board: size, circuit, placements} = obs;
    const scale = Math.min((width - 48) / size.width, 350 / size.height);
    const left = Math.max(32, (width - scale * size.width) / 2), top = 16, height = scale * size.height + 60;
    svg.setAttribute('viewBox', `0 0 ${width} ${height}`); svg.replaceChildren();
    const px = x => left + x * scale, py = y => top + (size.height - y) * scale;
    function node(tag, attrs, text) {
      const element = document.createElementNS('http://www.w3.org/2000/svg', tag);
      Object.entries(attrs).forEach(([key, value]) => element.setAttribute(key, value));
      if (text !== undefined) element.textContent = text;
      svg.append(element); return element;
    }
    node('title', {}, `Placed ${Object.keys(placements).length} macros${candidate ? `; checking candidate ( ${candidate.x}, ${candidate.y})` : ''}`);
    node('rect', {x: left, y: top, width: size.width * scale, height: size.height * scale, class: 'frame'});
    for (const net of circuit.nets) {
      const points = net.filter(name => placements[name]).map(name => placements[name]);
      if (points.length < 2) continue;
      const xs = points.map(p => p.x), ys = points.map(p => p.y);
      node('path', {d: `M${px(Math.min(...xs))} ${py(Math.min(...ys))} H${px(Math.max(...xs))} V${py(Math.max(...ys))} H${px(Math.min(...xs))} Z`, class: 'net'});
    }
    for (const [name, p] of Object.entries(placements)) {
      const [w, h] = circuit.blocks[name];
      node('rect', {x: px(p.x), y: py(p.y + h), width: w * scale, height: h * scale, class: 'macro'});
      node('text', {x: px(p.x + w / 2), y: py(p.y + h / 2) + 4, 'text-anchor': 'middle'}, name);
      node('circle', {cx: px(p.x), cy: py(p.y), r: 3, class: 'endpoint'});
    }
    if (candidate && obs.current_macro) {
      const [w, h] = circuit.blocks[obs.current_macro];
      const rejected = candidate.resolved && !candidate.report.legal;
      const clip = node('clipPath', {id: 'board-clip'});
      const region = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
      Object.entries({x: left, y: top, width: size.width * scale, height: size.height * scale}).forEach(([k,v]) => region.setAttribute(k,v)); clip.append(region);
      node('rect', {x: px(candidate.x), y: py(candidate.y + h), width: w * scale, height: h * scale, class: 'candidate' + (rejected ? ' rejected' : ''), 'clip-path': 'url(#board-clip)'});
    }
    node('text', {x: left, y: height - 16}, '0');
    node('text', {x: px(size.width), y: height - 16, 'text-anchor': 'end'}, `x = ${size.width}`);
    node('text', {x: 2, y: top + 12}, String(size.height));
    node('text', {x: 2, y: top + 28}, 'y');
    if (candidate) node('circle', {cx: px(candidate.x), cy: py(candidate.y), r: 4, class: 'candidate-anchor'});
    if (candidate && currentFrame()?.edge && obs.current_macro) {
      const [w, h] = circuit.blocks[obs.current_macro], edge = currentFrame().edge;
      const vertical = edge === 'left' || edge === 'right';
      const at = vertical ? candidate.x + (edge === 'right' ? w : 0) : candidate.y + (edge === 'top' ? h : 0);
      node('line', vertical ? {x1: px(at), x2: px(at), y1: py(candidate.y), y2: py(candidate.y + h), class: 'focus-edge'}
        : {x1: px(candidate.x), x2: px(candidate.x + w), y1: py(at), y2: py(at), class: 'focus-edge'});
    }
    get('board-focus').textContent = currentFrame()?.edge ? `The bold segment identifies the ${({left: "left edge / x", right: "right edge", bottom: "bottom edge / y", top: "top edge"})[currentFrame().edge]}; the dot marks the candidate's lower-left corner.` : '';
    get('board-status').textContent = inspected ? "State before this historical candidate was checked" : currentFrame()?.view === 'before' ? "State before this step call" : `Actually placed ${Object.keys(placements).length} / ${Object.keys(circuit.blocks).length}`;
  }
  function renderScore() {
    const shown = visibleState(), m = shown.info.metrics, details = shown.details;
    get('metrics').replaceChildren(...[["Bounding-box Area", m.area], [m.complete ? "Complete HPWL" : "HPWL of placed endpoints", m.hpwl], ["Actual reward", shown.reward]].map(([name, value]) => {
      const cell = make('div'); cell.append(make('span', name), make('strong', String(value))); return cell;
    }));
    const nodes = [], b = details.bounds;
    nodes.push(make('p', b ? `Bounding box: x from ${b.left} to ${b.right}, y from ${b.bottom} to ${b.top}; Area = ( ${b.right} − ${b.left}) × (${b.top} − ${b.bottom}) = ${m.area}.` : "The canvas is empty, so the occupied bounding-box Area is 0."));
    for (const net of details.nets) {
      const points = Object.entries(net.placed).map(([name, p]) => `${name}(${p.x}, ${p.y})`).join(', ') || "None";
      nodes.push(make('p', `Net ${net.index} placed endpoints: ${points}; x span ${net.dx} + y span ${net.dy} = ${net.hpwl}.${net.missing.length ? `Missing endpoints: ${net.missing.join(', ')}.` : "All endpoints are placed."}`));
    }
    nodes.push(make('p', `Total HPWL =  ${details.nets.map(net => net.hpwl).join(' + ') || '0'} = ${m.hpwl}; ${m.complete ? "All macros are placed." : "This is still a partial-circuit quantity."}`));
    get('score-calculation').replaceChildren(...nodes);
  }
  async function inspectAttempt(module, index) {
    stop();
    try {
      inspected = await api(`/api/attempt?module=${module}&index=${index}`);
      get('attempt-inspection').hidden = false;
      const checks = inspected.report.checks.slice(0, inspected.shown_checks);
      const list = make('div', undefined, 'check-list');
      for (const check of checks) {
        for (const detail of window.EE680PlacementLesson.checkFrames(check, inspected)) {
          const section = make('section');
          section.append(make('h4', detail.title), ...detail.paragraphs.map(text => make('p', text)), codeExcerpt(detail.code));
          list.append(section);
        }
      }
      get('attempt-detail').replaceChildren(make('p', `${inspected.observation.current_macro} @ (${inspected.x}, ${inspected.y}); index ${index}; ${inspected.resolved ? errorText(inspected.report.error) : "Some checks have not yet been unfolded"}`), list);
      board();
      get('attempt-inspection').scrollIntoView({block: 'nearest'});
    } catch (error) { get('operation-error').hidden = false; get('operation-error').textContent = "That candidate record is unavailable: " + error.message; }
  }
  function renderAttempts() {
    const attempts = state.attempts, max = Math.max(0, Math.ceil(attempts.length / pageSize) - 1);
    if (!manualPage) page = max;
    page = Math.max(0, Math.min(page, max));
    const rows = attempts.slice(page * pageSize, (page + 1) * pageSize).map(attempt => {
      const row = make('tr', undefined, attempt.legal === true ? 'selected' : '');
      const result = attempt.legal === null ? "Checks are not fully unfolded" : attempt.legal ? `mask[${attempt.index}] = true` : `false · ${errorText(attempt.error)}`;
      const cell = make('td'); const button = make('button', "Inspect checks"); button.type = 'button';
      button.addEventListener('click', () => inspectAttempt(state.module_index, attempt.index)); cell.append(button);
      row.append(make('td', `${attempt.index} / (${attempt.x}, ${attempt.y})`), make('td', result), cell); return row;
    });
    get('attempts').replaceChildren(...rows);
    const resolved = attempts.filter(a => a.legal !== null).length;
    const total = state.observation.board.width * state.observation.board.height;
    get('attempt-summary').textContent = `Current macro ${state.active_macro || "None"}: proposed ${attempts.length} candidates; ${attempts.filter(a => a.legal === false).length} rejected; ${attempts.filter(a => a.legal === true).length} passed. The reading trace has unfolded ${total} mask entries out of ${resolved} ; the remaining ${total - resolved} entries have not been unfolded.`;
    get('attempt-page').textContent = `${page + 1} / ${max + 1} pages / ${pageSize} entries per page`;
    get('previous-page').disabled = page === 0; get('next-page').disabled = page === max;
    const history = get('history-module'), previous = history.value;
    history.replaceChildren(...state.modules.map((module, index) => { const option = make('option', `${module.name || "Finished"} · ${module.attempts} candidates`); option.value = index; return option; }));
    if (Number(previous) < state.modules.length) history.value = previous;
  }
  function renderCode() {
    const name = get('code-file').value;
    const functions = {observe: 'observation', candidate: '_candidate', bounds: 'inspect_action', overlap: 'inspect_action',
      mask: 'advance', reject: 'advance', choose: 'advance', apply: 'step', metrics: 'metric_details',
      reward: '_finish_if_needed', next_observation: 'observation', done: 'advance'};
    const wanted = chapter === '01' && state ? (currentFrame()?.code.method || functions[state.phase]) : null;
    const lines = input.code[name].split('\n');
    const start = wanted ? lines.findIndex(line => line.startsWith('    def ' + wanted + '(')) : -1;
    let end = start + 1;
    while (start >= 0 && end < lines.length && !/^    (def |@)/.test(lines[end])) end++;
    get('code-source').textContent = start >= 0 ? lines.slice(start, end).map((line, i) => `${start + i + 1}  ${line}`).join('\n') : input.code[name];
    get('code-context').textContent = start >= 0 ? `This explanation refers to ${wanted}; the excerpt shows actual file line numbers. The link opens the full source.` : "Full contents of the selected source file";
    get('code-link').href = '/' + name;
  }
  function render() {
    get('connection-status').textContent = `Actual Python environment / ${state.event_count} recorded events / reading progress is saved`;
    get('attempt-inspection').hidden = true;
    explain(); renderScore(); renderAttempts(); board();
    get('phase-list').replaceChildren(...Object.entries(phaseNames).map(([key, name]) => {
      const node = make('li', name); if (key === state.phase) node.setAttribute('aria-current', 'step'); return node;
    }));
    const transition = state.transitions.at(-1);
    get('transition-summary').textContent = transition ? `Executed ${state.transitions.length} actions; the latest placed ${transition.observation.current_macro} at ( ${transition.action.join(', ')}), reward = ${transition.reward}. This chapter's fixed policy does not produce network log probabilities or value estimates.` : "No action has executed. Checking a candidate alone does not create an executed transition.";
    get('transition-json').textContent = JSON.stringify(transition || [], null, 2);
    get('events').replaceChildren(...state.recent_events.map(event => make('li', `Event ${event.number}  / ${event.module || "Finished"} · ${phaseNames[event.kind]}${event.index !== undefined ? ` · i = ${event.index}` : ''}`)));
    controls();
  }
  document.querySelectorAll('[data-pane]').forEach(button => button.addEventListener('click', () => pane(button.dataset.pane)));
  document.querySelectorAll('button[data-inspector]').forEach(button => button.addEventListener('click', () => inspector(button.dataset.inspector)));
  document.querySelectorAll('[data-chapter], [data-open-chapter]').forEach(button => button.addEventListener('click', () => { stop(); selectChapter(button.dataset.chapter || button.dataset.openChapter); if (state) board(); }));
  get('advance').addEventListener('click', () => { stop(); advanceReading(); });
  get('detail-prev').addEventListener('click', () => { if (detailIndex > 0 && !busy) { stop(); detailIndex--; explain(); board(); renderScore(); rememberDetail(); controls(); } });
  ['candidate', 'choose', 'module', 'reset'].forEach(name => get(name).addEventListener('click', () => { stop(); command(name); }));
  get('play').addEventListener('click', async () => {
    if (playing) { stop(); return; }
    playing = true; get('play').textContent = "Pause explanations";
    const tick = async () => { if (!playing) return; await advanceReading(); if (playing) timer = setTimeout(tick, Number(get('speed').value)); };
    tick();
  });
  get('previous-page').addEventListener('click', () => { manualPage = true; page--; renderAttempts(); });
  get('next-page').addEventListener('click', () => { manualPage = true; page++; renderAttempts(); });
  get('history-show').addEventListener('click', () => inspectAttempt(Number(get('history-module').value), Number(get('history-index').value)));
  get('close-inspection').addEventListener('click', () => {
    inspected = null; get('attempt-inspection').hidden = true; board();
    get('stage-title').scrollIntoView({block: 'start'});
  });
  get('code-file').addEventListener('change', renderCode);
  get('note').addEventListener('input', () => { get('note-status').textContent = storage('note:' + chapter, get('note').value) ? "Saved in this browser" : "Browser storage is unavailable; copy this note before leaving"; });
  get('theme').value = storage('theme') || 'dark';
  function theme() { document.documentElement.dataset.theme = get('theme').value; }
  get('theme').addEventListener('change', () => { theme(); storage('theme', get('theme').value); }); theme();
  new ResizeObserver(board).observe(get('board'));
  inspector(storage('inspector') || 'source');
  const requestedChapter = new URL(location.href).searchParams.get('chapter');
  selectChapter(['00', '01', '02', '03', '04', '05'].includes(requestedChapter) ? requestedChapter : storage('chapter') || '01', false);
  if (location.protocol === 'file:') {
    get('connection-status').textContent = "Run python steps/serve.py from the repository root, then open http://127.0.0.1:8868/steps/index.html";
    document.querySelectorAll('#chapter-01 button').forEach(button => button.disabled = true);
    return;
  }
  Promise.all([api('/api/input'), api('/api/session')]).then(([loadedInput, snapshot]) => {
    input = loadedInput; state = snapshot;
    get('input-name').textContent = `${input.source} · ${Object.keys(input.circuit.blocks).length} macros / ${input.circuit.nets.length} nets`;
    get('blocks').replaceChildren(...Object.entries(input.circuit.blocks).map(([name, size]) => make('p', `${name}: width ${size[0]}, height ${size[1]}`)));
    get('nets').replaceChildren(...input.circuit.nets.map((net, index) => make('p', `Net ${index}: ${net.join(', ')}`)));
    get('raw-input').textContent = input.raw; get('parsed-input').textContent = JSON.stringify(input.circuit, null, 2);
    get('code-file').replaceChildren(...Object.keys(input.code).map(name => { const option = make('option', name); option.value = name; return option; }));
    selectChapter(chapter, false); render();
  }).catch(error => { get('connection-status').textContent = "The local Python service is not connected: " + error.message; document.querySelectorAll('#chapter-01 button').forEach(button => button.disabled = true); });
})();
