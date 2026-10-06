"""Real shell stages with controlled model commands, failure and restart paths."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import unittest


PROJECT = Path(__file__).resolve().parents[1]
FAKE_PYTHON = r'''#!/usr/bin/env python3
import json, os, pathlib, sys, time, zipfile
args = sys.argv[1:]
if args[0] == '-u': args = args[1:]
if args[0] == '-': os.execv(sys.executable, [sys.executable, *args])
script, args = args[0], args[1:]
root = pathlib.Path(os.environ['AIC_OUTPUT_ROOT'])
def value(name): return args[args.index(name)+1]
def log(entry):
    fd=os.open(root/'calls.jsonl', os.O_WRONLY|os.O_CREAT|os.O_APPEND, 0o600)
    os.write(fd, (json.dumps(entry)+'\n').encode()); os.close(fd)
if script == 'v15_pipeline_support.py':
    action=args[0]
    if action == 'state':
        name=value('--state-file') if '--state-file' in args else 'status.json'
        (root/'v15_pipeline'/name).write_text(json.dumps({'stage':value('--stage'),'status':value('--status')}))
    elif action.startswith('wait-'):
        if os.environ.get('FAKE_DEPENDENCY_FAIL') == '1': raise SystemExit(8)
    elif action == 'accepted': print(int(json.loads((root/'v15_selection.json').read_text())['candidate_accepted']))
    elif action == 'baseline-evaluation': print(root/'v14_baseline_expanded')
    elif action == 'checkpoint':
        kind=value('--kind'); out=root/'sources'/kind; out.mkdir(parents=True,exist_ok=True)
        path=out/'model.pt'; path.write_text('model'); print(path)
    elif action == 'export':
        out=root/'v15_delivery'/value('--kind'); out.mkdir(parents=True,exist_ok=True)
        if '--collect' in args:
            for name in ('pred_results.csv','pred_results.zip'): (out/name).write_text('collected')
        (out/'provenance.json').write_text(json.dumps({'online_score':None}))
    else: raise AssertionError(action)
    raise SystemExit(0)
stage=os.environ.get('AIC_V15_STAGE','benchmark')
log({'script':script,'stage':stage,'args':args,'gpu':os.environ.get('CUDA_VISIBLE_DEVICES')})
if script == 'train_v15.py':
    out=pathlib.Path(value('--output-dir')); out.mkdir(parents=True,exist_ok=True)
    (out/'resume_latest.pt').write_text('resume')
if os.environ.get('FAKE_BLOCK_STAGE') == stage:
    (root/'blocked.started').touch()
    deadline=time.monotonic()+20
    while not (root/'release').exists():
        if time.monotonic()>deadline: raise SystemExit(90)
        time.sleep(.01)
if os.environ.get('FAKE_FAIL_STAGE') == stage: raise SystemExit(7)
if script == 'benchmark_v15.py':
    pathlib.Path(value('--output')).write_text(json.dumps({
      'format_version':15,'recipe':'expanded_mlp','success':True,'initialization':'official_base_only',
      'trained_checkpoint_weights_loaded':False,'chosen_workers':8,'batch_size':256,
      'gradient_accumulation':1,'pin_memory':False,'prefetch_factor':1,'configurations':[]}))
elif script == 'train_v15.py': (out/'model.pt').write_text('model')
elif script == 'evaluate_tta_v15.py':
    (pathlib.Path(value('--output-dir'))/'strict_eval.json').write_text('{}')
elif script == 'select_v15.py':
    pathlib.Path(value('--output')).write_text(json.dumps({'candidate_accepted':os.environ.get('FAKE_ACCEPT','1')=='1'}))
elif script == 'compare_v15.py': pathlib.Path(value('--output')).write_text('{}')
elif script == 'predict_v15.py':
    out=pathlib.Path(value('--output')); out.write_text('example.jpg, 0000\n')
    with zipfile.ZipFile(value('--zip-output'),'w') as z: z.write(out,'pred_results.csv')
else: raise AssertionError(script)
'''


class Fixture:
    def __init__(self, root):
        self.project, self.output, self.bin = root/'project space', root/'output space', root/'bin'
        self.project.mkdir(); self.output.mkdir(); self.bin.mkdir()
        (self.project/'scripts').mkdir()
        for name in ('run_v15_pipeline.sh','run_v15_train.sh','run_v15_evaluate_and_predict.sh'):
            shutil.copyfile(PROJECT/'scripts'/name, self.project/'scripts'/name)
        self.executable('fake-python', FAKE_PYTHON)
        self.executable('nvidia-smi', "#!/usr/bin/env bash\nprintf '81920\\n'\n")
        if shutil.which('flock') is None:
            self.executable('flock', '#!/usr/bin/env python3\nimport fcntl,sys\n'
                            'try: fcntl.flock(int(sys.argv[-1]),fcntl.LOCK_EX|fcntl.LOCK_NB)\n'
                            'except BlockingIOError: raise SystemExit(1)\n')
        if shutil.which('setsid') is None:
            self.executable('setsid', '#!/usr/bin/env python3\nimport os,sys\nos.setsid()\nos.execvp(sys.argv[1],sys.argv[1:])\n')
        self.env={**os.environ, 'AIC_PROJECT_DIR':str(self.project),'AIC_OUTPUT_ROOT':str(self.output),
                  'AIC_PYTHON':str(self.bin/'fake-python'),'AIC_GPU':'4',
                  'PATH':str(self.bin)+os.pathsep+str(Path(sys.executable).parent)+os.pathsep+os.environ['PATH']}
        for key in ('AIC_DELIVERY_GPU','AIC_BENCHMARK','AIC_V15_STAGE','AIC_SELECTION','FAKE_FAIL_STAGE','FAKE_BLOCK_STAGE'):
            self.env.pop(key,None)
        self.command=['bash',str(self.project/'scripts/run_v15_pipeline.sh')]

    def executable(self,name,content):
        path=self.bin/name; path.write_text(content); path.chmod(0o755)

    def run(self,**env):
        return subprocess.run(self.command,env={**self.env,**env},text=True,stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT,timeout=25)

    def calls(self):
        return [json.loads(row) for row in (self.output/'calls.jsonl').read_text().splitlines()]

    def status(self):
        return json.loads((self.output/'v15_pipeline/status.json').read_text())


class PipelineTests(unittest.TestCase):
    def test_acceptance_and_fallback_produce_separate_packages_and_resume(self):
        for accepted in ('1','0'):
            with self.subTest(accepted=accepted), tempfile.TemporaryDirectory() as directory:
                f=Fixture(Path(directory)); result=f.run(FAKE_ACCEPT=accepted)
                self.assertEqual(result.returncode,0,result.stdout)
                calls=f.calls(); stages=[row['stage'] for row in calls]
                self.assertEqual(stages.count('benchmark'),1)
                self.assertEqual(stages.count('validate'),1)
                self.assertEqual('refit' in stages,accepted=='1')
                kinds={p.name for p in (f.output/'v15_delivery').iterdir()}
                self.assertEqual(kinds,{'v14_validation','v14_refit'} | ({'v15_validation','v15_refit'} if accepted=='1' else set()))
                for row in calls:
                    self.assertNotIn('train_v14.py',row['script'])
                self.assertEqual(f.run(FAKE_ACCEPT=accepted).returncode,0)
                self.assertEqual(calls,f.calls())
                self.assertEqual(f.status()['status'],'complete')

    def test_refit_failure_resumes_without_reselecting_frozen_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            f=Fixture(Path(directory)); result=f.run(FAKE_FAIL_STAGE='refit')
            self.assertNotEqual(result.returncode,0)
            self.assertEqual(f.status(),{'stage':'full_data_refit','status':'failed'})
            original=(f.output/'v15_selection.json').read_bytes()
            (f.output/'v15_pipeline/selection.done').unlink()
            result=f.run(FAKE_ACCEPT='0')
            self.assertEqual(result.returncode,0,result.stdout)
            self.assertEqual(original,(f.output/'v15_selection.json').read_bytes())
            refits=[row for row in f.calls() if row['stage']=='refit']
            self.assertIn('--resume',refits[-1]['args'])
            self.assertEqual(sum(row['script']=='select_v15.py' for row in f.calls()),1)

    def test_delivery_runs_on_distinct_gpu_and_reports_dependency_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            f=Fixture(Path(directory)); result=f.run(AIC_DELIVERY_GPU='5')
            self.assertEqual(result.returncode,0,result.stdout)
            self.assertTrue(any(row['stage']=='v14-pair' and row['gpu']=='5' for row in f.calls()))
        with tempfile.TemporaryDirectory() as directory:
            f=Fixture(Path(directory)); result=f.run(AIC_DELIVERY_GPU='5',FAKE_DEPENDENCY_FAIL='1')
            self.assertNotEqual(result.returncode,0)
            state=json.loads((f.output/'v15_pipeline/v14_delivery_status.json').read_text())
            self.assertEqual(state['status'],'failed')

    def test_pipeline_lock_rejects_duplicate_and_training_can_finish(self):
        with tempfile.TemporaryDirectory() as directory:
            f=Fixture(Path(directory))
            process=subprocess.Popen(f.command,env={**f.env,'FAKE_BLOCK_STAGE':'validate'},
                                     stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
            try:
                deadline=time.monotonic()+12
                while not (f.output/'blocked.started').exists() and time.monotonic()<deadline: time.sleep(.01)
                self.assertTrue((f.output/'blocked.started').exists())
                duplicate=f.run()
                self.assertEqual(duplicate.returncode,2,duplicate.stdout)
                self.assertIn('already active',duplicate.stdout)
            finally:
                (f.output/'release').touch()
                output,_=process.communicate(timeout=25)
            self.assertEqual(process.returncode,0,output)


if __name__ == '__main__': unittest.main()
