"""Prepare checkpoint-matched learning traces and repeated before/after evaluations."""

import argparse
import json
from pathlib import Path
import shutil
from types import SimpleNamespace
import torch

from src.data.parse_blknet import load_circuit
from src.experiments.evaluate import evaluate_many, run_episode, write_json
from src.experiments.explain import record_teaching
from src.experiments.plot import floorplans, training_curve
from src.ppo.algorithm import load_checkpoint


def checkpoint_study(directory, checkpoint, inputs, episodes):
    model, _, _, saved = load_checkpoint(checkpoint)
    meta = saved['metadata']
    update = meta['update']
    for index,path in enumerate(inputs):
        replay=run_episode(load_circuit(path),model,meta['width'],meta['height'],meta['area_weight'])
        replay['update']=update
        write_json(directory/f'replay-{index}-update-{update}.json',replay)
    record_teaching(SimpleNamespace(checkpoint=checkpoint,input=inputs,
                                    output=directory/f'teaching-update-{update}.json',episodes=episodes,seed=20000))
    return update


def repeated_evaluation(directory, checkpoint, inputs, episodes, tag):
    model,_,_,saved=load_checkpoint(checkpoint)
    meta=saved['metadata']
    torch.manual_seed(100000)
    results=[]
    for index,path in enumerate(inputs):
        summary=evaluate_many(load_circuit(path),model,episodes,meta['width'],meta['height'],meta['area_weight'])
        write_json(directory/f'{tag}-samples-{index}.json',summary.pop('samples'))
        results.append({'input':str(path),'checkpoint_update':meta['update'],**summary})
    write_json(directory/f'{tag}-evaluation.json',{'seed':100000,'circuits':results})
    return results


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--teaching-episodes',type=int,default=2)
    parser.add_argument('--eval-episodes',type=int,default=32)
    args=parser.parse_args()
    torch.set_num_threads(1)
    progress=json.loads((args.run/'progress.json').read_text(encoding='utf-8'))
    inputs=list(map(Path,progress['inputs']))
    checkpoints=sorted(args.run.glob('checkpoint-[0-9]*.pt'),key=lambda p:int(p.stem.rsplit('-',1)[1]))
    updates=[checkpoint_study(args.run,path,inputs,args.teaching_episodes) for path in checkpoints]
    last=updates[-1]
    shutil.copy2(args.run/f'teaching-update-{last}.json',args.run/'teaching.json')
    shutil.copy2(args.run/f'teaching-update-{last}.pt',args.run/'teaching.pt')
    for index in range(len(inputs)):
        shutil.copy2(args.run/f'replay-{index}-update-{last}.json',args.run/f'replay-{index}.json')
    before=repeated_evaluation(args.run,checkpoints[0],inputs,args.eval_episodes,'study-before')
    after=repeated_evaluation(args.run,checkpoints[-1],inputs,args.eval_episodes,'study-after')
    figures=args.run/'figures';figures.mkdir(exist_ok=True)
    training_curve(args.run,figures);floorplans(args.run,figures)
    write_json(args.run/'study.json',{'status':'completed','checkpoints':updates,'before':before,'after':after})
    print(json.dumps({'run':str(args.run),'checkpoints':updates,'before':before,'after':after}),flush=True)


if __name__=='__main__':
    main()
