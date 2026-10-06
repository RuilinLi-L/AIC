"""Exercise the actual V19 supervisor shell scripts with small fake GPU jobs."""
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
import json,os,pathlib,sys,time,zipfile
args=sys.argv[1:]
if args and args[0]=='-u': args=args[1:]
if args[0] in ('-','-c'): os.execv(sys.executable,[sys.executable,*args])
script,args=args[0],args[1:]
root=pathlib.Path(os.environ['AIC_OUTPUT_ROOT'])
def value(key): return args[args.index(key)+1]
def write(path,data):
    path=pathlib.Path(path); path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(data))
def log(item):
    fd=os.open(root/'calls.jsonl',os.O_WRONLY|os.O_CREAT|os.O_APPEND,0o600)
    os.write(fd,(json.dumps(item)+'\n').encode()); os.close(fd)
if script=='v19_pipeline_support.py':
    action=args[0]
    if action=='init': print(100)
    elif action=='state':
        name=value('--state-file') if '--state-file' in args else 'status.json'
        write(root/'v19_pipeline'/name,{'stage':value('--stage'),'status':value('--status')})
    elif action=='wait-gpu':
        if os.environ.get('FAKE_GPU_FAIL')=='1': raise SystemExit(8)
    elif action=='wait-baselines':
        if os.environ.get('FAKE_DEPENDENCY_FAIL')=='1': raise SystemExit(9)
    elif action=='audit-check':
        assert pathlib.Path(value('--audit-json')).exists()
        if os.environ.get('FAKE_AUDIT_FAIL')=='1': raise SystemExit(11)
    elif action=='benchmark-needed':
        path=pathlib.Path(value('--benchmark'))
        if path.exists():
            data=json.loads(path.read_text()); assert data.get('success') is not False or data.get('status')=='cuda_oom'
        print(int(not path.exists()))
    elif action=='benchmark-status':
        data=json.loads(pathlib.Path(value('--benchmark')).read_text()); print(data.get('resource_status','runnable'))
    elif action=='benchmark-settings':
        data=json.loads(pathlib.Path(value('--benchmark')).read_text()); print(8,data['batch_size'])
    elif action=='official-check':
        assert pathlib.Path(value('--report')).is_file()
    elif action=='freeze-resources':
        entries={}
        for key,recipe,suffix in [('candidate_a','resolution384_rank32','a'),('candidate_b','rank32_dropout','b')]:
            report=json.loads(pathlib.Path(value('--benchmark-'+suffix)).read_text())
            entries[key]={'recipe':recipe,'run_dir':value('--candidate-'+suffix),'workers':8,'batch_size':256,
                'status':report.get('resource_status','runnable'),'gradient_accumulation':1,
                'benchmark':{'path':value('--benchmark-'+suffix)},'official_check':{'path':value('--official-'+suffix)}}
        write(value('--resource-json'),{'branch':'standard','candidates':entries})
    elif action.startswith('resource-'):
        path=pathlib.Path(os.environ['AIC_RESOURCE_JSON']); data=json.loads(path.read_text())
        if action=='resource-branch': print(data['branch'])
        elif action in ('resource-field','resource-settings'):
            if '--candidate' in args: entry=data['candidates'][value('--candidate')]
            else: entry=next(e for e in data['candidates'].values() if e['recipe']==value('--recipe'))
            if action=='resource-settings': print(entry['workers'],entry['batch_size'])
            else:
                item=entry[value('--field')]; print(item['path'] if isinstance(item,dict) else item)
    elif action=='accepted': print(int(json.loads((root/'v19_selection.json').read_text())['refit_required']))
    elif action=='recipe': print(json.loads((root/'v19_selection.json').read_text())['recipe'])
    elif action=='bind':
        assert (root/'v19_selection.json').is_file()
    elif action=='checkpoint':
        path=root/'sources'/value('--kind')/'model.pt'; path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text('model'); print(path)
    elif action=='reusable': print('null')
    elif action=='export':
        write(root/'v19_delivery'/value('--kind')/'provenance.json',{'online_score':None})
    else: raise AssertionError(action)
    raise SystemExit(0)
if script=='audit_generalization_v19.py':
    write(pathlib.Path(value('--output-dir'))/'audit.json',{'status':'passed'}); raise SystemExit(0)
recipe=os.environ.get('AIC_V19_RECIPE')
stage=os.environ.get('AIC_V19_STAGE','official_check' if script=='scripts/check_v19_official.py' else 'benchmark')
log({'script':script,'stage':stage,'recipe':recipe,'args':args,'gpu':os.environ.get('CUDA_VISIBLE_DEVICES')})
if script=='train_v19.py':
    resource=json.loads(pathlib.Path(value('--resource-json')).read_text())
    entry=next(e for e in resource['candidates'].values() if e['recipe']==recipe)
    assert int(value('--batch-size'))==entry['batch_size']
    assert int(value('--gradient-accumulation'))==entry['gradient_accumulation']
    out=pathlib.Path(value('--output-dir')); out.mkdir(parents=True,exist_ok=True); (out/'resume_latest.pt').touch()
    if os.environ.get('FAKE_BARRIER')=='1' and stage=='validate':
        (root/(recipe+'.started')).touch()
        stop=time.monotonic()+8
        while not all((root/(r['recipe']+'.started')).exists() for r in resource['candidates'].values() if r['status']=='runnable'):
            if time.monotonic()>stop: raise SystemExit(90)
            time.sleep(.01)
    if os.environ.get('FAKE_BLOCK_STAGE')==stage:
        (root/'blocked.started').touch()
        stop=time.monotonic()+20
        while not (root/'release').exists():
            if time.monotonic()>stop: raise SystemExit(90)
            time.sleep(.01)
if os.environ.get('FAKE_FAIL_STAGE')==stage and os.environ.get('FAKE_FAIL_RECIPE',recipe)==recipe:
    if script=='benchmark_v19.py': write(value('--output'),{'success':False})
    raise SystemExit(7)
if script=='scripts/check_v19_official.py': write(value('--output'),{'success':True})
elif script=='benchmark_v19.py':
    oom=(recipe=='resolution384_rank32' and os.environ.get('FAKE_OOM_A')=='1') or os.environ.get('FAKE_OOM_ALL')=='1'
    write(value('--output'),{'batch_size':256,'success':not oom,'status':'cuda_oom' if oom else 'ok', 'resource_status':'resource_infeasible' if oom else 'runnable'})
elif script=='train_v19.py': (out/('best_model.pt' if stage=='validate' else 'model.pt')).touch()
elif script=='evaluate_tta_v19.py':
    out=pathlib.Path(value('--output-dir')); write(out/'strict_eval.json',{}); (out/'model.pt').touch()
elif script=='compare_v19.py': write(value('--output'),{})
elif script=='select_v19.py':
    resource=json.loads(pathlib.Path(value('--resource-json')).read_text())
    write(value('--output'),{'refit_required':os.environ.get('FAKE_ACCEPT','1')=='1' and any(e['status']=='runnable' for e in resource['candidates'].values()),
                            'recipe':os.environ.get('FAKE_WINNER','rank32_dropout' if resource['branch']=='standard' else 'resolution384_rank32')})
elif script=='predict_v19.py':
    out=pathlib.Path(value('--output')); out.write_text('a.jpg, 0000\n')
    with zipfile.ZipFile(value('--zip-output'),'w') as z: z.write(out,'pred_results.csv')
else: raise AssertionError(script)
'''


class Fixture:
    def __init__(self, root):
        self.project,self.output,self.bin=root/'project space',root/'output space',root/'bin'
        self.project.mkdir(); self.output.mkdir(); self.bin.mkdir(); (self.project/'scripts').mkdir()
        for name in ('pipeline','train','evaluate_and_predict'):
            shutil.copyfile(PROJECT/f'scripts/run_v19_{name}.sh',self.project/f'scripts/run_v19_{name}.sh')
        self.executable('fake-python',FAKE_PYTHON)
        if shutil.which('flock') is None:
            self.executable('flock','#!/usr/bin/env python3\nimport fcntl,sys\n'
                            'flags=fcntl.LOCK_EX | (fcntl.LOCK_NB if "-n" in sys.argv else 0)\n'
                            'try: fcntl.flock(int(sys.argv[-1]),flags)\n'
                            'except BlockingIOError: raise SystemExit(1)\n')
        self.env={k:v for k,v in os.environ.items() if not k.startswith(('AIC_','FAKE_'))}
        self.env.update(AIC_PROJECT_DIR=str(self.project),AIC_OUTPUT_ROOT=str(self.output),
                        AIC_PYTHON=str(self.bin/'fake-python'),AIC_GPU_A='0',
                        PATH=str(self.bin)+os.pathsep+str(Path(sys.executable).parent)+os.pathsep+os.environ['PATH'])
        self.command=['bash',str(self.project/'scripts/run_v19_pipeline.sh')]
    def executable(self,name,text):
        path=self.bin/name; path.write_text(text); path.chmod(0o755)
    def run(self,**env):
        return subprocess.run(self.command,env={**self.env,**env},text=True,stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT,timeout=30)
    def calls(self):
        return [json.loads(line) for line in (self.output/'calls.jsonl').read_text().splitlines()]
    def status(self): return json.loads((self.output/'v19_pipeline/status.json').read_text())


class PipelineTests(unittest.TestCase):
    def test_two_gpus_overlap_and_only_winner_refits(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=Fixture(Path(tmp)); result=f.run(AIC_GPU_B='3',FAKE_BARRIER='1')
            self.assertEqual(result.returncode,0,result.stdout)
            trains=[c for c in f.calls() if c['script']=='train_v19.py']
            self.assertEqual([(c['recipe'],c['gpu']) for c in trains if c['stage']=='refit'],[('rank32_dropout','0')])
            self.assertEqual({c['gpu'] for c in trains if c['stage']=='validate'},{'0','3'})
            self.assertEqual({p.name for p in (f.output/'v19_delivery').iterdir()},{'validation','refit'})
            calls=f.calls(); self.assertEqual(f.run(AIC_GPU_B='3').returncode,0)
            self.assertEqual(f.calls(),calls)
    def test_one_gpu_serial_fallback_predicts_without_retraining(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=Fixture(Path(tmp)); result=f.run(FAKE_ACCEPT='0')
            self.assertEqual(result.returncode,0,result.stdout)
            trains=[c for c in f.calls() if c['script']=='train_v19.py']
            self.assertEqual([c['recipe'] for c in trains],['resolution384_rank32','rank32_dropout'])
            self.assertTrue(all(c['stage']=='validate' and c['gpu']=='0' for c in trains))
            self.assertEqual({p.name for p in (f.output/'v19_delivery').iterdir()},{'validation','fallback_refit'})
            self.assertEqual(f.status()['status'],'complete')
    def test_refit_failure_resume_preserves_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=Fixture(Path(tmp)); result=f.run(FAKE_FAIL_STAGE='refit')
            self.assertNotEqual(result.returncode,0)
            self.assertEqual(f.status(),{'stage':'full_data_refit','status':'failed'})
            original=(f.output/'v19_selection.json').read_bytes()
            (f.output/'v19_pipeline/selection.done').unlink()
            result=f.run(FAKE_ACCEPT='0')
            self.assertEqual(result.returncode,0,result.stdout)
            self.assertEqual((f.output/'v19_selection.json').read_bytes(),original)
            self.assertEqual(sum(c['script']=='select_v19.py' for c in f.calls()),1)
            self.assertIn('--resume',[c for c in f.calls() if c['stage']=='refit'][-1]['args'])
    def test_dependency_failure_and_duplicate_lock_are_explicit(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=Fixture(Path(tmp)); result=f.run(FAKE_DEPENDENCY_FAIL='1')
            self.assertNotEqual(result.returncode,0)
            self.assertEqual(f.status(),{'stage':'baseline_binding','status':'failed'})
        with tempfile.TemporaryDirectory() as tmp:
            f=Fixture(Path(tmp)); proc=subprocess.Popen(f.command,env={**f.env,'FAKE_BLOCK_STAGE':'validate'},
                stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
            try:
                deadline=time.monotonic()+10
                while not (f.output/'blocked.started').exists() and time.monotonic()<deadline: time.sleep(.01)
                self.assertTrue((f.output/'blocked.started').exists())
                duplicate=f.run(); self.assertEqual(duplicate.returncode,2,duplicate.stdout)
                self.assertIn('already active',duplicate.stdout)
            finally:
                (f.output/'release').touch(); output,_=proc.communicate(timeout=30)
            self.assertEqual(proc.returncode,0,output)
    def test_failed_benchmark_does_not_automatically_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=Fixture(Path(tmp)); result=f.run(FAKE_FAIL_STAGE='benchmark')
            self.assertNotEqual(result.returncode,0)
            calls=f.calls(); result=f.run()
            self.assertNotEqual(result.returncode,0,result.stdout)
            self.assertEqual(f.calls(),calls)
    def test_data_overlap_gate_precedes_gpu_work(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=Fixture(Path(tmp)); result=f.run(FAKE_AUDIT_FAIL='1')
            self.assertNotEqual(result.returncode,0)
            self.assertEqual(f.status()['stage'],'data_audit')
            self.assertFalse((f.output/'calls.jsonl').exists())
    def test_failed_candidate_has_no_done_marker_or_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=Fixture(Path(tmp)); result=f.run(AIC_GPU_B='3',FAKE_FAIL_STAGE='validate')
            self.assertNotEqual(result.returncode,0)
            self.assertFalse((f.output/'v19_resolution384_rank32/validation.done').exists())
            self.assertFalse((f.output/'v19_selection.json').exists())
            child=json.loads((f.output/'v19_pipeline/resolution384_rank32_status.json').read_text())
            self.assertEqual(child['status'],'failed')
            self.assertEqual(f.status()['status'],'failed')

    def test_official_checks_precede_training_and_failure_never_trains(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=Fixture(Path(tmp)); result=f.run()
            self.assertEqual(result.returncode,0,result.stdout)
            calls=f.calls()
            for recipe in ('resolution384_rank32','rank32_dropout'):
                relevant=[call['script'] for call in calls if call['recipe']==recipe]
                self.assertLess(relevant.index('scripts/check_v19_official.py'),relevant.index('benchmark_v19.py'))
                self.assertLess(relevant.index('benchmark_v19.py'),relevant.index('train_v19.py'))
            selection=next(call for call in calls if call['script']=='select_v19.py')
            self.assertIn('--baseline',selection['args'])
            self.assertIn('--resource-json',selection['args'])
            self.assertNotIn('--baseline-expanded',selection['args'])
        with tempfile.TemporaryDirectory() as tmp:
            f=Fixture(Path(tmp)); result=f.run(FAKE_FAIL_STAGE='official_check')
            self.assertNotEqual(result.returncode,0,result.stdout)
            self.assertFalse(any(call['script']=='train_v19.py' for call in f.calls()))

    def test_actual_oom_skips_only_affected_candidate_without_batch_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=Fixture(Path(tmp)); result=f.run(AIC_GPU_B='3',FAKE_OOM_A='1')
            self.assertEqual(result.returncode,0,result.stdout)
            trains=[c for c in f.calls() if c['script']=='train_v19.py']
            self.assertTrue(trains)
            self.assertEqual({c['recipe'] for c in trains},{'rank32_dropout'})
            for call in trains:
                args=call['args']; self.assertEqual(args[args.index('--batch-size')+1],'256')
                self.assertEqual(args[args.index('--gradient-accumulation')+1],'1')
            before=f.calls(); self.assertEqual(f.run(AIC_GPU_B='3').returncode,0)
            self.assertEqual(f.calls(),before)
    def test_both_oom_reuses_baseline_without_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=Fixture(Path(tmp)); result=f.run(FAKE_OOM_ALL='1')
            self.assertEqual(result.returncode,0,result.stdout)
            self.assertFalse(any(c['script']=='train_v19.py' for c in f.calls()))
            self.assertEqual(f.status()['status'],'complete')
    def test_second_probe_failure_never_starts_training_or_fakes_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=Fixture(Path(tmp)); result=f.run(FAKE_FAIL_STAGE='benchmark',FAKE_FAIL_RECIPE='rank32_dropout')
            self.assertNotEqual(result.returncode,0,result.stdout)
            self.assertFalse(any(call['script']=='train_v19.py' for call in f.calls()))
            self.assertFalse((f.output/'v19_pipeline/resource_plan.json').exists())
            self.assertFalse((f.output/'v19_selection.json').exists())

    def test_invalid_recipe_cannot_refit_even_if_selection_claims_it_won(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=Fixture(Path(tmp)); result=f.run(FAKE_OOM_A='1',FAKE_WINNER='matched_control')
            self.assertNotEqual(result.returncode,0,result.stdout)
            self.assertIn('Only a runnable V19 winner may refit',result.stdout)
            self.assertFalse(any(call['stage']=='refit' for call in f.calls()))

if __name__=='__main__': unittest.main()
