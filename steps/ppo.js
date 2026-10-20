(() => {
  const get = id => document.getElementById(id);
  const make = (tag, text) => { const node = document.createElement(tag); if (text !== undefined) node.textContent = text; return node; };
  const svgNode = (tag, attrs, text) => { const node = document.createElementNS('http://www.w3.org/2000/svg', tag); for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v); if (text !== undefined) node.textContent = text; return node; };
  let data, replay, chapter = '02', explanation = 0, circuitIndex = 0, timer, request = 0, sourceRequest = 0, sampleIndex = 0;
  let activeRun = new URL(location.href).searchParams.get('run') || '';
  const lessonBuilders = window.EE680PPOLessons;
  function table(target, headings, rows) {
    const table = make('table'), head = make('thead'), body = make('tbody'), tr = make('tr');
    headings.forEach(h => tr.append(make('th', h))); head.append(tr);
    rows.forEach(row => { const tr = make('tr'); row.forEach(value => tr.append(make('td', value))); body.append(tr); });
    table.append(head, body); get(target).replaceChildren(table);
  }
  const number = value => typeof value === 'number' ? value.toFixed(4) : '—';
  function lessonContext() {
    const selectedUpdate = data?.selected_update;
    const teachingMode = Boolean(data?.teaching && data.teaching.source_update === selectedUpdate);
    const lessonData = (teachingMode ? data.teaching : data?.rollout) || {};
    const samples = lessonData.transitions || [];
    sampleIndex = Math.min(sampleIndex, Math.max(0, samples.length - 1));
    const sample = samples[sampleIndex];
    const config = lessonData.config || data?.progress.config || {};
    const updateStats = teachingMode ? lessonData.update_stats : data?.progress.history.find(row => row.update === selectedUpdate) || {};
    return {data, replay, step: replay?.steps[Math.min(Number(get('ppo-step').value), replay.steps.length-1)],
      lessonData, teachingMode, samples, sampleIndex, sample, config, updateStats,
      updateExample: updateStats?.example, board: lessonData.board || replay?.board,
      sampleCircuit: lessonData.circuits?.[sample?.circuit_index || 0] || replay?.circuit};
  }
  function numericExplanation(frame) {
    const context = lessonContext();
    get('ppo-numeric-context').textContent = chapter === '02' ? "These numbers describe the selected evaluation action. Move the action slider to inspect another actual decision." : context.teachingMode ? `These numbers come from an isolated teaching update starting at checkpoint ${context.lessonData.source_update}. It samples separately and saves a separate model, preserving the official checkpoint.` : "These numbers come from the selected checkpoint's retained training records. Older records may contain fewer fields.";
    get('ppo-numeric-explanation').replaceChildren(make('pre', (frame[4]?.(context) || []).join('\n')));
  }
  async function json(url) { const response = await fetch(url, {cache: 'no-store'}); const result = await response.json(); if (!response.ok) throw new Error(result.error); return result; }
  async function explain() {
    const generation = ++sourceRequest, frames = lessonBuilders[chapter](), frame = frames[explanation];
    get('ppo-number').textContent = chapter + ' / PPO';
    get('ppo-title').textContent = {'02': "How does the model read the circuit and select legal actions", '03': "How does real interaction become PPO training data", '04': "How does environment feedback change network parameters"}[chapter];
    get('ppo-explanation-title').textContent = frame[0]; get('ppo-explanation').textContent = frame[1];
    get('ppo-explanation-progress').textContent = `Explanation ${explanation + 1} / ${frames.length} `;
    get('ppo-prev').disabled = explanation === 0; get('ppo-next').disabled = explanation === frames.length - 1;
    numericExplanation(frame);
    get('ppo-source-link').href = '/' + frame[2]; get('ppo-source-link').textContent = "Source: " + frame[2];
    try {
      const response = await fetch('/' + frame[2]); if (!response.ok) throw new Error("Could not load source");
      const text = await response.text(), lines = text.split('\n'); let start = lines.findIndex(line => line.startsWith(frame[3])), end = start + 1;
      if (start < 0) throw new Error("The referenced source definition was not found");
      if (generation !== sourceRequest) return;
      while (end < lines.length && !/^(def |class |    def |@torch)/.test(lines[end])) end++;
      get('ppo-code').textContent = lines.slice(start, end).map((line, i) => `${start + i + 1}  ${line}`).join('\n');
      window.dispatchEvent(new CustomEvent('ee680:ppo-source', {detail: {name: frame[2], text}}));
    } catch (error) { get('ppo-code').textContent = error.message; }
  }
  async function refresh() {
    clearTimeout(timer); get('ppo-error').hidden = true;
    const generation = ++request, selectedCircuit = circuitIndex, checkpoint = get('ppo-checkpoint').value;
    try {
      const runQuery = activeRun ? '&run=' + encodeURIComponent(activeRun) : '';
      const incoming = await json('/api/ppo?view=learning' + runQuery + (checkpoint === 'latest' ? '' : '&update=' + checkpoint));
      if (generation !== request) return;
      renderRuns(incoming.runs || []);
      if (!incoming.available) { get('ppo-status').textContent = incoming.message; get('ppo-record-panels').hidden = true; timer = setTimeout(refresh,5000); return; }
      const selectedReplay = await json('/api/ppo/replay?index=' + selectedCircuit + runQuery + (checkpoint === 'latest' ? '' : '&update=' + checkpoint));
      if (generation !== request) return;
      data = incoming; replay = selectedReplay;
      get('ppo-record-link').href = '/api/ppo?view=learning' + runQuery + (checkpoint === 'latest' ? '' : '&update=' + checkpoint);
      activeRun = incoming.directory; get('ppo-run').value = activeRun; get('ppo-record-panels').hidden = false;
      const p = data.progress;
      const checkpoints = get('ppo-checkpoint'); checkpoints.replaceChildren();
      [['latest', "Latest saved evaluation"], ...data.checkpoints.map(update => [String(update), "Update " + update])].forEach(([value, label]) => { const option = make('option', label); option.value = value; checkpoints.append(option); });
      checkpoints.value = checkpoint;
      window.dispatchEvent(new CustomEvent('ee680:ppo-input', {detail: {circuit: replay.circuit, name: "PPO replay: " + p.evaluations[selectedCircuit].input}}));
      get('ppo-status').textContent = `${p.status === 'completed' ? "Training completed" : "Training is running"}  / completed updates: ${p.update}  / replay checkpoint: ${replay.update ?? incoming.selected_update} · seed ${p.seed} · CPU · ${number(p.training_seconds)} seconds, including periodic evaluation and saving`;
      get('connection-status').textContent = `Reading actual PPO records / completed updates: ${p.update}  / seed ${p.seed}`;
      const selector = get('ppo-circuit'); selector.replaceChildren();
      p.evaluations.forEach((item, i) => { const option = make('option', item.input); option.value = i; selector.append(option); });
      selector.value = circuitIndex;
      get('ppo-step').max = replay.steps.length; get('ppo-step').value = Math.min(Number(get('ppo-step').value), replay.steps.length);
      render();
      explain();
      if ((p.status === 'running' || incoming.runs?.some(r=>r.status!=='completed')) && !get('chapter-ppo').hidden) timer = setTimeout(refresh, 5000);
    } catch (error) { if (generation === request) { get('ppo-error').hidden = false; get('ppo-error').textContent = "Could not read this record: " + error.message; } }
  }
  function renderRuns(runs) {
    const select = get('ppo-run'); select.replaceChildren();
    runs.forEach(run => { const option = make('option', run.label + (run.status === 'preparing' ? " (preparing)" : run.status === 'running' ? " (training)" : '')); option.value=run.directory; select.append(option); });
    if (activeRun) select.value=activeRun;
  }
  function render() {
    const progress = data.progress, last = progress.history.find(row=>row.update===data.selected_update) || {}, count = Number(get('ppo-step').value);
    const selected = replay.steps[Math.min(count, replay.steps.length - 1)];
    get('ppo-metrics').replaceChildren(...[["Mean training return", last.mean_reward], ["Training completion rate", last.legal_completion_rate], ["Parameter change L2", last.parameter_delta_l2], ["Approximate KL", last.approx_kl], ["Policy loss", last.policy_loss], ["Value loss", last.value_loss], ["Gradient norm", last.gradient_norm]].map(([name, value]) => { const box = make('div'); box.append(make('span', name), make('strong', number(value))); return box; }));
    get('ppo-config').textContent = JSON.stringify({config: progress.config, runtime: progress.runtime, evaluations: progress.evaluations}, null, 2);
    get('ppo-figure-section').hidden = !data.figures?.length;
    get('ppo-figures').replaceChildren(...(data.figures || []).map(path => { const link = make('a'); link.href = '/' + path; link.target = '_blank'; const img = make('img'); img.src = '/' + path; img.alt = {'ablation-curves.png': "Actual training means and sample standard deviations across three seeds and three reward weights", 'training-curve.png': "Actual training returns and legal completion rates for this experiment", 'ppo-floorplans.png': "Actual PPO layouts, Area and HPWL for this experiment"}[path.split('/').at(-1)] || path; img.loading = 'lazy'; link.append(img); return link; }));
    if (selected) {
      const names = Object.keys(replay.circuit.blocks);
      table('ppo-nodes', ["Macro", 'w/W', 'h/H', 'area/WH', 'x/W', 'y/H', 'placed', 'current', 'nets/M'], (selected.nodes || []).map((row, i) => [names[i], ...row.map(number)]));
      get('ppo-incidence').textContent = JSON.stringify({columns: names, rows: selected.incidence}, null, 2);
      get('ppo-features').textContent = `The displayed inputs precede the action for ${selected.observation.current_macro}; legal anchors: ${selected.legal_count}; critic V(s)=${number(selected.value)}; selected action log probability=${number(selected.log_prob)}.`;
      table('ppo-actions', ["Candidate i", '(x, y)', "Probability"], (selected.top_actions || []).map(item => [item.index, `(${item.index % replay.board.width}, ${Math.floor(item.index / replay.board.width)})`, number(item.probability)]));
    }
    drawBoard(count); drawCurve(progress.history);
    const rollout = data.rollout;
    get('ppo-rollout-caption').textContent = rollout ? `First 30 of ${rollout.transitions.length} training transitions used in update ${rollout.update}; latest saved update: ${progress.update}.` : "This checkpoint has no archived official rollout. Any isolated teaching data below is labeled separately.";
    table('ppo-rollout', ['episode', "Circuit", "Macro", "Action i", 'reward', 'old logp', 'old V', 'return', "Normalized A", "Terminal"], (rollout?.transitions || []).slice(0, 30).map(r => [r.episode, r.circuit_index, r.macro, r.action, number(r.reward), number(r.log_prob), number(r.value), number(r.return), number(r.advantage_normalized), r.terminated ? "Yes" : "No"]));
    renderSample();
    renderEvaluations();
    numericExplanation(lessonBuilders[chapter]()[explanation]);
  }
  function renderEvaluations() {
    const studied=Boolean(data.study), before=studied?data.study.before:data['initial-evaluation']?.circuits, after=studied?data.study.after:data.progress.evaluations;
    const rows=[["Experiment start",before?.[circuitIndex]], [data.progress.status==='completed'?"Final experiment model":"Latest saved evaluation",after?.[circuitIndex]]].filter(([,row])=>row);
    get('ppo-evaluation-section').hidden=!rows.length;
    get('ppo-evaluation-origin').textContent=studied?"Sources: study-before-evaluation.json and study-after-evaluation.json. Each official model is loaded separately; evaluation seed=100000.":"Sources: initial-evaluation.json and progress.json. Training uses a separately preserved, fixed evaluation RNG stream.";
    table('ppo-evaluation',["Stage","Episodes","Legal complete","Mean reward","Mean Area","Mean HPWL"],rows.map(([label,r])=>[label,r.episodes,`${Math.round(r.legal_completion_rate*r.episodes)}/${r.episodes}`,number(r.mean_reward),number(r.mean_area_complete),number(r.mean_hpwl_complete)]));
  }
  function drawBoard(count) {
    const board = get('ppo-board'), {width, height} = replay.board;
    let view = [-5,-5,width+10,height+10];
    const focus = replay.steps[Math.min(count,replay.steps.length-1)];
    if (get('ppo-focus').checked && focus) {
      const [w,h] = replay.circuit.blocks[focus.observation.current_macro], size=Math.max(w,h)*3;
      view=[focus.action[0]+w/2-size/2,height-focus.action[1]-h/2-size/2,size,size];
    }
    board.setAttribute('viewBox', view.join(' ')); board.replaceChildren();
    board.append(svgNode('rect', {x: 0, y: 0, width, height, fill: 'var(--surface)', stroke: 'var(--line)'}));
    const current = replay.steps[count];
    if (get('ppo-show-mask').checked && (current?.legal_mask || current?.legal_mask_runs)) {
      const indices = current.legal_mask ? current.legal_mask.flatMap((valid,i)=>valid?[i]:[]) : current.legal_mask_runs.flatMap(([start,end])=>Array.from({length:end-start},(_,i)=>start+i));
      const path = indices.map(i=>`M${i % width} ${height - Math.floor(i / width) - 1}h1v1h-1Z`).join(' ');
      board.append(svgNode('path', {d: path, fill: 'var(--secondary)', 'fill-opacity': .16}));
    }
    const positions = {};
    replay.steps.slice(0, count).forEach(step => positions[step.observation.current_macro] = {x: step.action[0], y: step.action[1]});
    for (const net of replay.circuit.nets) for (let i = 0; i < net.length; i++) for (let j = i + 1; j < net.length; j++) {
      const a = positions[net[i]], b = positions[net[j]];
      if (a && b) board.append(svgNode('line', {x1: a.x, y1: height - a.y, x2: b.x, y2: height - b.y, stroke: 'var(--secondary)', 'stroke-opacity': .4, 'stroke-width': .25}));
    }
    Object.entries(positions).forEach(([name, p]) => {
      const [w, h] = replay.circuit.blocks[name];
      board.append(svgNode('rect', {x: p.x, y: height - p.y - h, width: w, height: h, fill: 'var(--selected)', 'fill-opacity': .5, stroke: 'var(--text)', 'stroke-width': .25}));
      if (Object.keys(positions).length<=20 || name===focus?.observation.current_macro || get('ppo-focus').checked) board.append(svgNode('text', {x: p.x + w / 2, y: height - p.y - h / 2, 'font-size': view[2]/40, 'text-anchor': 'middle', fill: 'var(--text)'}, name));
    });
    const next = replay.steps[count];
    if (next) {
      const [w, h] = replay.circuit.blocks[next.observation.current_macro];
      board.append(svgNode('rect', {x: next.action[0], y: height - next.action[1] - h, width: w, height: h, fill: 'none', stroke: 'var(--text)', 'stroke-width': .4, 'stroke-dasharray': '1 1'}));
    }
    const metrics = count ? replay.steps[count - 1].info.metrics : {area: 0, hpwl: 0};
    const total=Object.keys(replay.circuit.blocks).length;
    get('ppo-step-caption').textContent = `Placed ${count} / ${total} macros / Area ${metrics.area} · HPWL ${metrics.hpwl}${count < replay.steps.length ? " (partial endpoints)" : ' · ' + (replay.validation.legal ? "Complete layout passed independent validation" : "No legal continuation; this episode ended before placing every macro")}  / inference time ${number(replay.inference_seconds)} seconds`;
  }
  function renderSample() {
    const c=lessonContext(), r=c.sample, selector=get('ppo-sample'); selector.replaceChildren();
    c.samples.forEach((sample,i)=>{const option=make('option',`Sample ${i+1}  / episode ${sample.episode} · ${sample.macro} · reward ${number(sample.reward)}`);option.value=i;selector.append(option);});
    selector.value=sampleIndex;
    get('ppo-sample-origin').textContent=(c.teachingMode ? `Isolated teaching update: starting from official checkpoint ${c.lessonData.source_update}, seed=${c.lessonData.seed}, sampling ${c.lessonData.episodes.length} episodes and applying one PPO update. The official model stays unchanged.` : `Official training samples from update ${c.lessonData.update ?? '—'}, stored separately from the argmax evaluation above.`)+(r?` The selected sample belongs to circuit ${r.circuit_index}, containing ${Object.keys(c.sampleCircuit.blocks).length} macros.`:'');
    if(!r){get('ppo-sample-math').textContent="This checkpoint has no sampling or update record yet.";get('ppo-sample-board').replaceChildren();get('ppo-update-example').textContent='';get('ppo-sample-caption').textContent='';return;}
    const n=x=>Number.isFinite(x)?x.toFixed(6):'—';
    get('ppo-sample-math').textContent=[`Current macro ${r.macro}; action i=${r.action}`,`x=${r.action % c.board.width}, y=${Math.floor(r.action/c.board.width)}`,`old logp=${n(r.log_prob)}; old V=${n(r.value)}`,`reward=${n(r.reward)}; terminated=${r.terminated}`,`next V=${n(r.next_value)}; delta=${n(r.delta)}`,`A_raw=${n(r.advantage_raw)}; return=${n(r.return)}`,`A_norm=${n(r.advantage_normalized)}`,`Next macro=${r.next_macro ?? 'None'}`].join('\n');
    const e=c.updateExample;
    get('ppo-update-example').textContent=e?`A separate actual optimizer example uses episode ${e.episode}, macro ${e.macro}, action ${e.action}: old logp=${n(e.old_log_prob)}; after one Adam step its log probability is ${n(e.log_prob_after_step)}. Chapter 04 explains each calculation.`:"No detailed optimizer example was saved for this checkpoint.";
    drawSampleBoard(c);
  }
  function drawSampleBoard(c) {
    const svg=get('ppo-sample-board'), r=c.sample, {width,height}=c.board;
    svg.replaceChildren(); svg.setAttribute('viewBox',`-5 -12 ${2*width+20} ${height+22}`);
    if(!r.placements_before || !r.placements_after){get('ppo-sample-caption').textContent="This older record has no complete before/after coordinates. Select a checkpoint with an isolated teaching update.";return;}
    const x=r.action%width, y=Math.floor(r.action/width), [mw,mh]=c.sampleCircuit.blocks[r.macro];
    let view=[0,0,width,height];
    if(get('ppo-focus').checked){const size=Math.max(mw,mh)*3;view=[x+mw/2-size/2,height-y-mh/2-size/2,size,size];}
    [r.placements_before,r.placements_after].forEach((positions,side)=>{
      const inner=svgNode('svg',{x:side*(width+10),y:0,width,height,viewBox:view.join(' ')});
      inner.append(svgNode('rect',{x:0,y:0,width,height,fill:'var(--surface)',stroke:'var(--line)'}));
      Object.entries(positions).forEach(([name,p])=>{const [w,h]=c.sampleCircuit.blocks[name];inner.append(svgNode('rect',{x:p.x,y:height-p.y-h,width:w,height:h,fill:name===r.macro?'var(--secondary)':'var(--selected)','fill-opacity':.5,stroke:'var(--text)','stroke-width':view[2]/300}));if(Object.keys(positions).length<=10||name===r.macro)inner.append(svgNode('text',{x:p.x+w/2,y:height-p.y-h/2,'font-size':view[2]/20,'text-anchor':'middle',fill:'var(--text)'},name));});
      if(side===0)inner.append(svgNode('rect',{x,y:height-y-mh,width:mw,height:mh,fill:'none',stroke:'var(--text)','stroke-width':view[2]/150,'stroke-dasharray':`${view[2]/80} ${view[2]/80}`}));
      svg.append(inner);svg.append(svgNode('text',{x:side*(width+10)+width/2,y:-4,'font-size':width/16,'text-anchor':'middle',fill:'var(--text)'},side?"After execution":"Before execution"));
    });
    get('ppo-sample-caption').textContent=`One actual training transition: the left view shows the proposed position for ${r.macro}; the right view shows coordinates after env.step. reward=${number(r.reward)}, terminated=${r.terminated}.${get('ppo-focus').checked?"Zoomed around the selected macro.":"Enable local zoom above to inspect this region."}`;
  }
  function drawCurve(history) {
    const svg = get('ppo-curve'); svg.setAttribute('viewBox', '0 0 640 220'); svg.replaceChildren();
    svg.append(svgNode('path', {d: 'M45 15 V185 H625', stroke: 'var(--line)', fill: 'none'}));
    const points = (key, transform) => history.map((row, i) => `${45 + i * 580 / Math.max(1, history.length - 1)},${185 - transform(row[key]) * 160}`).join(' ');
    svg.append(svgNode('polyline', {points: points('mean_reward', r => (r + 2) / 2), fill: 'none', stroke: 'var(--text)', 'stroke-width': 2}));
    svg.append(svgNode('polyline', {points: points('legal_completion_rate', r => r), fill: 'none', stroke: 'var(--secondary)', 'stroke-width': 2, 'stroke-dasharray': '5 3'}));
    [['0 / 1', 3, 20], ['−2 / 0', 3, 187], ["PPO update count ->", 245, 212]].forEach(([text, x, y]) => svg.append(svgNode('text', {x, y, fill: 'var(--text)', 'font-size': 11}, text)));
    get('ppo-curve-caption').textContent = "The solid line is mean sampled return (range -2 to 0); the dashed line is legal completion rate (0 to 1). Each point is one actual PPO update. Full samples and initial/final evaluations remain in the saved records.";
  }
  window.addEventListener('ee680:chapter', event => { chapter = event.detail; explanation = 0; explain(); refresh(); });
  get('ppo-prev').addEventListener('click', () => { explanation--; explain(); });
  get('ppo-next').addEventListener('click', () => { explanation++; explain(); });
  get('ppo-refresh').addEventListener('click', refresh);
  get('ppo-circuit').addEventListener('change', () => { circuitIndex = Number(get('ppo-circuit').value); get('ppo-step').value = 0; refresh(); });
  get('ppo-run').addEventListener('change',()=>{activeRun=get('ppo-run').value;circuitIndex=0;sampleIndex=0;get('ppo-step').value=0;get('ppo-checkpoint').value='latest';refresh();});
  get('ppo-checkpoint').addEventListener('change', () => { get('ppo-step').value = 0; refresh(); });
  get('ppo-step').addEventListener('input', () => { if (replay) render(); });
  get('ppo-show-mask').addEventListener('change', () => { if (replay) drawBoard(Number(get('ppo-step').value)); });
  get('ppo-focus').addEventListener('change',()=>{if(replay){drawBoard(Number(get('ppo-step').value));renderSample();}});
  get('ppo-sample').addEventListener('change',()=>{sampleIndex=Number(get('ppo-sample').value);renderSample();numericExplanation(lessonBuilders[chapter]()[explanation]);});
})();
