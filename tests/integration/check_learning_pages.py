"""Exercise question dependencies, embedded lab views, notes and real source references."""
import json,sys,tempfile
from pathlib import Path
from threading import Thread
from urllib.request import urlopen
from playwright.sync_api import sync_playwright, expect
import argparse
parser=argparse.ArgumentParser(description='Validate the question-led reader and embedded actual visualizations using Playwright.')
parser.add_argument('--output',type=Path,default=Path('steps/build/page-validation'))
parser.add_argument('--include-local-zh',action='store_true')
args=parser.parse_args()
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));OUT=args.output.resolve();OUT.mkdir(parents=True,exist_ok=True)
from steps.serve import make_server
from integration.visualization.learning_session import LearningSession
from src.data.parse_blknet import load_circuit

views=[];tested_trials=[];book=json.loads((ROOT/'steps/course/book.json').read_text(encoding='utf-8'))
with tempfile.TemporaryDirectory() as folder:
 server=make_server(0,output=folder,ppo_pointer=ROOT/'steps/evidence/ppo-latest.json')
 server.RequestHandlerClass.log_message=lambda *_:None
 thread=Thread(target=server.serve_forever,daemon=True);thread.start();base=f'http://127.0.0.1:{server.server_port}'
 before=json.load(urlopen(base+'/api/session'))
 try:
  with sync_playwright() as pw:
   browser=pw.chromium.launch()
   for width in [1440,736,320]:
    for theme in ['light','dark']:
     print('Checking reader viewport',width,theme,flush=True);page=browser.new_page(viewport={'width':width,'height':1000});errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
     page.goto(base+'/steps/reading.html');page.locator('#question h2').wait_for();page.locator('#theme').select_option(theme)
     questions=book['questions'] if width==1440 else [book['questions'][0],book['questions'][2],book['questions'][-1]]
     for q in questions:
      page.evaluate('(id)=>location.hash=id',q['id']);page.wait_for_function('(title)=>document.querySelector("#question h2").textContent===title',arg=q['question'])
      page.wait_for_function('document.querySelector("#question pre").textContent.length>30')
      assert page.locator('#question .derivation>li').count()==3
      assert 'undefined' not in page.locator('#question').inner_text()
      assert not page.evaluate('document.documentElement.scrollWidth>innerWidth'),(width,theme,q['id'])
     page.locator('#lab').scroll_into_view_if_needed();frame=page.frame_locator('#lab');frame.locator('#comparison-table table').wait_for(timeout=15000);assert frame.locator('#comparison-ppo-board rect').count()>1
     frame.locator('#comparison-sa-step').fill('1');assert 'undefined' not in frame.locator('#sa-event').inner_text()
     frame.locator('#sa-snapshot').select_option('candidate');frame.locator('#sa-best').click()
     if width==1440 and theme=='light':
      comparison=json.load(urlopen(base+'/api/comparison'))
      for case_index,case in enumerate(comparison['cases']):
       frame.locator('#comparison-case').select_option(str(case_index))
       for trial in case['trials']:
        side='sa' if trial['algorithm']=='sa_sp_aligned' else 'ppo'
        frame.locator('#comparison-'+side).select_option(trial['file'])
        expect(frame.locator('#'+side+'-raw')).to_have_attribute('href','/'+trial['file'],timeout=15000)
        assert not frame.locator('#comparison-error').inner_text()
        if trial['validation']['complete']:assert frame.locator('#comparison-'+side+'-board rect').count()==case['macros']+1
        tested_trials.append(trial['file'])
     assert not frame.locator('body').evaluate('(n)=>n.scrollWidth>innerWidth'),(width,theme,'comparison iframe')
     ch=book['chapters'][-1]
     for q in ch['quiz']:page.locator(f'input[name="{q["id"]}"][value="{q["correct"]}"]').check()
     page.locator('#quiz button').click();assert page.locator('#quiz-results').inner_text().count('Correct')==4
     page.locator('#notes').fill('Local browser note retained after reload.');page.reload();page.locator('#question h2').wait_for()
     assert page.locator('#notes').input_value()=='Local browser note retained after reload.'
     page.evaluate('scrollTo(0,0)');page.screenshot(path=str(OUT/f'reader-{width}-{theme}.png'))
     page.locator('#lab').scroll_into_view_if_needed();page.screenshot(path=str(OUT/f'reader-lab-{width}-{theme}.png'))
     assert not errors,errors;views.append(dict(width=width,theme=theme,questions=len(questions),overflow=False,errors=errors));page.close()
   if args.include_local_zh:
    page=browser.new_page();page.goto(base+'/steps/build/local-zh/reading.html');page.locator('#question h2').wait_for()
    assert page.locator('html').get_attribute('lang')=='zh-CN'
    assert any(0x4e00<=ord(c)<=0x9fff for c in page.locator('#question').inner_text())
    page.goto(base+'/steps/build/local-zh/comparison.html');page.locator('#comparison-table table').wait_for();page.close()
   page=browser.new_page();page.goto(base+'/steps/index.html?chapter=01');page.locator('#stage-explanation .code-line').first.wait_for()
   circuit=load_circuit(ROOT/'benchmarks/alarik-synthetic/blknet_5.txt');walk=LearningSession(circuit);examples=[];seen=set()
   while True:
    key=walk.phase+(':'+walk.module['name'] if walk.phase in ('bounds','overlap','mask','choose','metrics','reward','apply') else '')
    if key not in seen:examples.append(walk.snapshot());seen.add(key)
    if walk.phase=='done':break
    walk.advance()
   failed=LearningSession({'blocks':{'A':[2,2],'B':[2,2]},'nets':[]},2,2);failed.command('choose')
   for _ in range(3):failed.advance()
   examples.append(failed.snapshot());sources=json.load(urlopen(base+'/api/input'))['code']
   contract=page.evaluate(r'''({examples,sources})=>{let frames=0,references=0;for(const state of examples)for(const f of window.EE680PlacementLesson.build(state)){frames++;if(!f.title||f.paragraphs.length<2)throw Error('Incomplete explanation');const lines=sources[f.code.file].split('\n');const start=lines.findIndex(l=>l.startsWith('    def '+f.code.method+'('));let end=start+1;while(end<lines.length&&!/^    (def |@)/.test(lines[end]))end++;const text=lines.slice(start,end).join('\n');for(const token of f.code.focus){if(start<0||!text.includes(token))throw Error(f.title+': '+token);references++;}}return {examples:examples.length,frames,references};}''',dict(examples=examples,sources=sources))
   page.close();browser.close()
  assert json.load(urlopen(base+'/api/session'))==before
  result=dict(status='passed',views=views,comparison_trials_viewed=len(tested_trials),local_chinese=args.include_local_zh,notes_preserved=True,reading_preserves_01=True,placement_contract=contract)
  (OUT/'reader-validation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
 finally:server.shutdown();server.server_close();thread.join()
