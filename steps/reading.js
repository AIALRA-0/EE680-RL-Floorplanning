/* Render authored course artifacts; never synthesize experimental values. */
(() => {
  const get = id => document.getElementById(id), el = (tag,text) => {const n=document.createElement(tag);if(text!==undefined)n.textContent=text;return n;};
  let book, index=0, sourceGeneration=0;
  const store = (key,value) => {try {const k='ee680:course:'+document.documentElement.lang+':'+key;if(value===undefined)return localStorage.getItem(k);localStorage.setItem(k,value);} catch (_) {return null;}};
  function navigation() {
    get('contents').replaceChildren(...book.chapters.map(ch => {
      const detail=el('details'),summary=el('summary',ch.id+' / '+ch.title);detail.open=ch.id===book.questions[index].chapter;
      detail.append(summary,...ch.question_ids.map(id=>{const q=book.questions.find(q=>q.id===id),a=el('a',q.question);a.href='#'+id;
        if(id===book.questions[index].id)a.setAttribute('aria-current','page');return a;}));return detail;
    }));
  }
  async function source(q,container) {
    const generation=++sourceGeneration,record=book.materials.find(s=>s.id===q.source_ids[0]);
    const details=el('details'),summary=el('summary',book.ui.code+' / '+record.symbol),link=el('a',record.path+':'+record.start+'-'+record.end),pre=el('pre');
    link.href='/'+record.path;link.target='_blank';details.className='source';details.append(summary,link,pre);container.append(details);
    try {const response=await fetch('/'+record.path);if(!response.ok)throw new Error(response.status+' '+record.path);const lines=(await response.text()).split('\n');
      if(generation!==sourceGeneration)return;pre.textContent=lines.slice(record.start-1,record.end).map((s,i)=>(record.start+i)+'  '+s).join('\n');
    }catch(error){pre.textContent=error.message;}
  }
  function quiz(ch) {
    const form=get('quiz');form.replaceChildren(...ch.quiz.map((q,i)=>{
      const field=el('fieldset');field.append(el('legend',(i+1)+'. '+q.prompt));q.options.forEach((text,j)=>{const label=el('label'),input=el('input');input.type='radio';input.name=q.id;input.value=j;label.append(input,document.createTextNode(text));field.append(label);});return field;
    }));
    const short=el('fieldset'),answer=el('textarea');answer.rows=4;answer.name='short';answer.value=store('answer:'+ch.id)||'';
    answer.addEventListener('input',()=>store('answer:'+ch.id,answer.value));short.append(el('legend','5. '+ch.short_answer.prompt),answer);form.append(short);
    const button=el('button',book.ui.check);button.type='submit';form.append(button);get('quiz-results').replaceChildren();
    form.onsubmit=event=>{event.preventDefault();const result=get('quiz-results');result.replaceChildren(...ch.quiz.map((q,i)=>{
      const selected=form.elements[q.id].value,ok=selected!==''&&Number(selected)===q.correct;
      const p=el('p',(i+1)+'. '+(ok?book.ui.correct:book.ui.incorrect)+' — '+q.rationale);p.className='quiz-result';return p;
    }),el('p',book.ui.expected+': '+ch.short_answer.expected));};
  }
  function render() {
    const q=book.questions[index],ch=book.chapters.find(c=>c.id===q.chapter),ui=book.ui;
    store('position',q.id);get('chapter').textContent=ch.id+' / '+ch.title;get('progress').textContent=(index+1)+' / '+book.questions.length;
    get('question-position').textContent=(index+1)+' / '+book.questions.length;get('previous').disabled=index===0;get('next').disabled=index===book.questions.length-1;
    const container=get('question');container.replaceChildren(el('h2',q.question));const premise=el('p',ui.premise+': '+q.premise);premise.className='premise';container.append(premise,el('h3',ui.steps));
    const list=el('ol');list.className='derivation';q.explanation.forEach(s=>{const li=el('li'),ground=el('p',s.grounding);ground.className='grounding';li.append(el('h3',s.title),el('p',s.statement),ground);list.append(li);});container.append(list);
    const result=el('div');result.className='result';result.append(el('h3',ui.answer),el('p',q.result.answer),el('h3',ui.pitfall),el('p',q.appendix.pitfall));container.append(result);source(q,container);
    get('lab-guide').textContent=ch.media.reading_guide;get('media-takeaway').textContent=ch.media.takeaway;
    get('media-components').replaceChildren(...ch.media.components.map(c=>{const p=el('p');p.append(el('b',c.label+' / '),document.createTextNode(c.text));return p;}));
    const url=new URL(ch.media.src,location.origin);url.searchParams.set('theme',get('theme').value);
    if(get('lab').dataset.chapter!==ch.id){get('lab').src=url.href;get('lab').dataset.chapter=ch.id;}
    get('lab-link').href=ch.media.src;get('lab-link').textContent=ui.open_lab;
    get('notes').value=store('note:'+q.id)||'';get('notes-status').textContent='';quiz(ch);navigation();
  }
  function go() {const id=location.hash.slice(1);const next=book.questions.findIndex(q=>q.id===id);if(next>=0)index=next;render();}
  get('notes').addEventListener('input',()=>{store('note:'+book.questions[index].id,get('notes').value);get('notes-status').textContent=book.ui.save;});
  get('previous').onclick=()=>{location.hash=book.questions[--index].id;};get('next').onclick=()=>{location.hash=book.questions[++index].id;};
  function theme(value){document.documentElement.dataset.theme=value;store('theme',value);const frame=get('lab');if(frame.contentDocument)frame.contentDocument.documentElement.dataset.theme=value;}
  get('theme').onchange=()=>theme(get('theme').value);get('lab').onload=()=>theme(get('theme').value);
  fetch('course/book.json').then(r=>{if(!r.ok)throw new Error(r.status);return r.json();}).then(data=>{
    book=data;get('book-title').textContent=book.title;document.title=book.title;get('introduction').textContent=book.introduction;
    for(const [id,key] of [['contents-title','contents'],['lab-title','lab'],['quiz-title','quiz'],['notes-title','notes'],['previous','previous'],['next','next'],['terms-title','terms']])get(id).textContent=book.ui[key];
    get('terms').replaceChildren(...book.glossary.map(t=>{const a=el('article');a.append(el('h3',t.term),el('p',t.definition),el('p',t.analogy),el('p',t.example));return a;}));
    get('theme').value=store('theme')||'auto';theme(get('theme').value);window.addEventListener('hashchange',go);
    if(!location.hash&&store('position'))location.hash=store('position');else go();
  }).catch(error=>get('error').textContent=error.message);
})();
