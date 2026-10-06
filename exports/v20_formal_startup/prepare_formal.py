import os, signal, time, shutil, subprocess
from pathlib import Path
import json
old_pid=3444202
snapshot=Path('/home/mcxu/lrl/AIC/.runs/v20_formal_shared_20261006_150101')
work=Path('/data/mcxu/AIC/outputs/v20_joint_20261006')
tests=(snapshot/'v20_final_tests.log').read_text()
assert tests.rstrip().endswith('OK') and 'Ran 58 tests' in tests
assert not (work/'pipeline/resource_plan.json').exists()
cmd=Path(f'/proc/{old_pid}/cmdline').read_bytes().replace(b'\0',b' ').decode()
assert '/.runs/v20_joint_20261006_094716/v20_pipeline.py' in cmd
children=Path(f'/proc/{old_pid}/task/{old_pid}/children').read_text().strip()
assert not children, f'old supervisor has children: {children}'
os.kill(old_pid,signal.SIGTERM)
for _ in range(50):
    if not Path(f'/proc/{old_pid}').exists(): break
    time.sleep(.2)
assert not Path(f'/proc/{old_pid}').exists()
probe=work/'gpu3_shared_probe_20261006_143539'
context=json.loads((probe/'context.json').read_text())
assert context['status']=='passed'
run=work/'loraplus_rank32'; run.mkdir(exist_ok=True)
for name in ('official_check.json','benchmark.json'):
    dest=run/name
    if dest.exists(): assert dest.read_bytes()==(probe/name).read_bytes()
    else: shutil.copy2(probe/name,dest)
shutil.copy2('/home/mcxu/lrl/AIC/.runs/v20_joint_20261006_094716/cpu_regression.log',snapshot/'cpu_regression.log')
env=dict(os.environ,AIC_V20_GPU_POLICY='gpu3_shared_measured_v1',AIC_V20_SHARED_PROBE=str(probe/'context.json'),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
subprocess.run(['/data/mcxu/conda-envs/aic/bin/python','-B',str(snapshot/'v20_pipeline_support.py'),'freeze-resources','--root','/data/mcxu/AIC/outputs','--v19-resource','/data/mcxu/AIC/outputs/v19_dedup_20261005/pipeline/resource_plan.json','--candidate-a',str(work/'dora_rank32'),'--candidate-b',str(run),'--model-dir','/home/mcxu/lrl/AIC/clip-ViT-B-32','--source-manifest',str(snapshot/'source_manifest.json'),'--resource-json',str(work/'pipeline/resource_plan.json')],env=env,cwd=snapshot,check=True)
subprocess.run(['tmux','new-session','-d','-s','v20_formal_shared_20261006',f'bash {snapshot}/launch_v20.sh >> {work}/pipeline/formal_supervisor.log 2>&1'],check=True)
print('formal supervisor dispatched',flush=True)
