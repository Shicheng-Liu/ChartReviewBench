"""Offline release integration: A/matplotlib, B/seaborn, C/plotly + C/clean.

Uses oracle replay to test plumbing, NOT model capability. Requires the external
release and pinned rendering dependencies. Does not modify source data or runs.
"""
import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
import subprocess

SANDBOX = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SANDBOX / 'src'))
from chartsandbox.agent import ScriptedAgent
from chartsandbox.runner import run_episode
from prepare_release import build, WORKSPACE_FILE


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--release',type=Path,required=True)
    ap.add_argument('--case',choices=('all','matplotlib','seaborn','plotly','clean'),default='all')
    args = ap.parse_args()
    release=args.release.resolve()
    with tempfile.TemporaryDirectory(prefix='release-integration-') as tmp:
        root=Path(tmp)
        for track, library, clean in [('A','matplotlib',False),('B','seaborn',False),
                                       ('C','plotly',False),('C','matplotlib',True)]:
            if args.case != 'all' and args.case != ('clean' if clean else library):
                continue
            rows=[json.loads(l) for l in (release/'data'/f'{track}.jsonl').open()]
            row=next(r for r in rows if r['library']==library and bool(r['clean'])==clean)
            task=build(row,root/'tasks'/track,release,tuple(WORKSPACE_FILE),10,6,300,90)
            script=row['ground_truth_code']
            code="exec(compile(open('chart.py').read(), 'chart.py', 'exec'))\n"
            code+="\nimport matplotlib.pyplot as plt\nif plt.get_fignums(): plt.gcf().savefig('output.png')\n"
            if library=='plotly':
                code+="\nimport plotly.graph_objects as go\nfor v in list(globals().values()):\n    if isinstance(v, go.Figure): v.write_image('output.png')\n"
            agent=ScriptedAgent([{'tool':'write_file','args':{'path':'chart.py','content':script}},
                                 {'tool':'execute_python','args':{'code':code}},
                                 {'tool':'finish','args':{}}])
            out=root/'runs'/track/row['id']
            r=run_episode(task,agent,out)
            assert r['subgoal_results'][0]['passed'],(out/'trajectory.jsonl').read_text()
            assert (out/'trajectory.jsonl').is_file() and (out/'task_config.json').is_file()
            env={k:v for k,v in os.environ.items() if k not in ('PYTHONPATH','PYTHONHOME','PYTHONSTARTUP')}
            env['CRB_PYTHON']=sys.executable
            cmd=[sys.executable,str(SANDBOX/'scripts/score_rules.py'),str(out.parent),
                 '--release',str(release),'--tasks',str(task.parent),'--python',sys.executable]
            p=subprocess.run(cmd,env=env,capture_output=True,text=True,timeout=300)
            assert p.returncode==0,p.stdout+p.stderr
            rb=json.loads((out/'result.json').read_text())['rule_based']
            assert 'error' not in rb,rb
            assert rb['executability']==1,rb
            assert rb['recovery'] is None if clean else rb['recovery']==1,rb
            assert rb['preservation']==1,rb
            assert rb['data_fidelity'] in (None,1),rb
            print(f'PASS Track {track} {library} clean={clean}: replay → durable trace → independent rule scoring',flush=True)
    return 0


if __name__=='__main__': raise SystemExit(main())
