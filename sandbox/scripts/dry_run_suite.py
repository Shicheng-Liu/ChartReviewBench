"""Run real subprocess episodes to verify durable output and subset/full resume."""
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

SANDBOX = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SANDBOX / 'src'))
from chartsandbox.agent import Agent, ToolCall
from chartsandbox.runner import run_episode


def task(root, track, name, fail=False):
    d = root / track / name
    (d / 'workspace').mkdir(parents=True)
    (d / 'oracle').mkdir()
    config = {'id':name, 'family':'debug_repair', 'instruction':'Render a chart.',
              'max_turns':2,'step_budget':4,'step_timeout_s':20,'wall_time_s':90,
              'subgoals':[{'id':'code_valid','verifiers':[{'type':'execution','params':{'produces':'output.png'}}]}]}
    (d / 'task.yaml').write_text(yaml.safe_dump(config))
    code = "import matplotlib.pyplot as plt\nplt.plot([1,2],[3,4])\nplt.savefig('output.png')\n"
    (d / 'sim_agent.json').write_text(json.dumps([
        {'tool':'execute_python','args':{'code':"raise ValueError('model failure')" if fail else code}},
        {'tool':'finish','args':{}}]))
    return d


def main():
    with tempfile.TemporaryDirectory(prefix='suite-test-') as temp:
        root = Path(temp)
        tasks, out = root/'tasks', root/'results'
        task(tasks,'trackA','same-id')
        task(tasks,'trackB','same-id',fail=True)
        task(tasks,'trackC','same-id')
        task(tasks,'trackC','clean-reference')
        cmd = [sys.executable,str(SANDBOX/'scripts/run_suite.py'),str(tasks),
               '--out',str(out),'--agent','scripted','--model','openrouter:test/model',
               '--mock-judge','--workers','2']
        def run(extra=(), expected=0):
            p=subprocess.run(cmd+list(extra),capture_output=True,text=True)
            assert p.returncode==expected, p.stdout+p.stderr
            return p.stdout
        run(['--limit','1'])
        a=out/'trackA/same-id/result.json'
        stamp=a.stat().st_mtime_ns
        run()
        assert len(list(out.glob('*/*/result.json')))==4
        assert a.stat().st_mtime_ns==stamp
        stamps={str(p):p.stat().st_mtime_ns for p in out.glob('*/*/result.json')}
        run()
        assert stamps=={str(p):p.stat().st_mtime_ns for p in out.glob('*/*/result.json')}
        # A valid zero score is complete, so resume must not cherry-pick it away.
        b=out/'trackB/same-id/result.json'
        assert json.loads(b.read_text())['all_passed'] is False
        # A partial JSON cannot masquerade as a committed episode.
        b.write_text('{')
        run()
        assert json.loads(b.read_text())['final_score']==0
        assert list((out/'.attempts/trackB/same-id').glob('*/result.json'))
        # Changing the model or the input is rejected before launching any work.
        run(['--model','openrouter:test/other'],expected=2)
        (tasks/'trackA/same-id/workspace/new.txt').write_text('changed')
        run(expected=2)
        assert a.stat().st_mtime_ns==stamp
        # Crash after one action: step trace survives without a false completion.
        class Crash(Agent):
            def reset(self,task,sandbox): self.n=0
            def act(self,obs):
                self.n+=1
                if self.n==2: raise RuntimeError('injected outage')
                return ToolCall('execute_python',{'code':"print('durable step')"})
        partial=root/'partial'
        try:
            run_episode(tasks/'trackC/same-id',Crash(),partial)
        except RuntimeError as e:
            assert 'injected outage' in str(e)
        else:
            raise AssertionError('expected outage')
        assert not (partial/'result.json').exists()
        trace=[json.loads(x) for x in (partial/'trajectory.jsonl').read_text().splitlines()]
        assert len(trace)==1 and 'durable step' in str(trace[0])
        assert (partial/'agent_checkpoint.json').exists()
        assert list((out/'trackA/same-id/renders').glob('*/*.png'))
        summary=json.loads((out/'summary.json').read_text())
        assert summary['scored']==4
    print('PASS: A/B/C mapping, sample→full, zero-score resume, corrupt-result retry/archive, config/input guard, incremental traces')


if __name__=='__main__': main()
