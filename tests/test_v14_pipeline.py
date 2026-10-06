"""Exercise the real V14 shell orchestration with fake model commands and GPUs."""

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
if args and args[0] == '-u':
    args = args[1:]
if args[0] == '-':
    os.execv(sys.executable, [sys.executable, *args])
script, args = args[0], args[1:]
def value(name):
    return args[args.index(name) + 1]
stage = os.environ['AIC_V14_STAGE']
root = pathlib.Path(os.environ['AIC_OUTPUT_ROOT'])
entry = {'script': script, 'stage': stage, 'args': args,
         'gpu': os.environ.get('CUDA_VISIBLE_DEVICES'),
         'pin': os.environ.get('AIC_PIN_MEMORY')}
if script == 'train_v14.py':
    out = pathlib.Path(value('--output-dir')); out.mkdir(parents=True, exist_ok=True)
    if stage == 'refit':
        selection = json.loads(pathlib.Path(value('--selection-json')).read_text())
        entry['selected_recipe'] = selection['recipe']
    (out / 'resume_latest.pt').write_text('completed epoch checkpoint')
fd = os.open(root / 'calls.jsonl', os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
os.write(fd, (json.dumps(entry) + '\n').encode()); os.close(fd)
if os.environ.get('AIC_FAKE_BLOCK_STAGE') == stage:
    (root / 'blocked.started').touch()
    deadline = time.monotonic() + 25
    while not (root / 'release').exists():
        if time.monotonic() > deadline: raise SystemExit(91)
        time.sleep(.01)
if os.environ.get('AIC_FAKE_FAIL_STAGE') == stage:
    raise SystemExit(7)
if script == 'train_v14.py':
    (out / 'model.pt').write_text('model')
elif script == 'evaluate_tta_v14.py':
    out = pathlib.Path(value('--output-dir')); out.mkdir(parents=True, exist_ok=True)
    (out / 'strict_eval.json').write_text(json.dumps({'source': value('--run-dir')}))
elif script == 'select_v14.py':
    assert (pathlib.Path(value('--baseline')) / 'strict_eval.json').is_file()
    assert (pathlib.Path(value('--candidate')) / 'strict_eval.json').is_file()
    out = pathlib.Path(value('--output'))
    out.write_text(json.dumps({'recipe': os.environ.get('AIC_FAKE_WINNER', 'expanded_robust'),
                               'selected_epoch': 18, 'frozen': 'keep these exact bytes'}))
elif script == 'predict_v14.py':
    assert pathlib.Path(value('--checkpoint')).is_file()
    assert value('--expected-rows') == '37444'
    out = pathlib.Path(value('--output')); out.write_text('example.jpg, 0000\n')
    with zipfile.ZipFile(value('--zip-output'), 'w') as archive:
        archive.write(out, 'pred_results.csv')
else:
    raise AssertionError(script)
'''


class PipelineFixture:
    def __init__(self, root):
        self.project = root / "project with spaces"
        self.output = root / "outputs with spaces"
        self.bin = root / "bin"
        self.project.mkdir(); self.output.mkdir(); self.bin.mkdir()
        (self.project / "scripts").mkdir()
        for name in ("run_v14_pipeline.sh", "run_v14_train.sh", "run_v14_evaluate_and_predict.sh"):
            shutil.copyfile(PROJECT / "scripts" / name, self.project / "scripts" / name)
        self._executable("fake-python", FAKE_PYTHON)
        self._executable("nvidia-smi", "#!/usr/bin/env bash\nprintf '81920\\n'\n")
        # macOS has flock(2) but no flock CLI. This preserves the inherited
        # descriptor's lock semantics while Linux tests use the real utility.
        if shutil.which("flock") is None:
            self._executable("flock", "#!/usr/bin/env python3\nimport fcntl, sys\n"
                             "try: fcntl.flock(int(sys.argv[-1]), fcntl.LOCK_EX | fcntl.LOCK_NB)\n"
                             "except BlockingIOError: raise SystemExit(1)\n")
        (self.output / "v14_benchmark.json").write_text(json.dumps({
            "chosen_workers": 4, "batch_size": 128, "gradient_accumulation": 2,
            "pin_memory": False, "prefetch_factor": 1,
        }))
        self.environment = {
            **os.environ,
            "PATH": str(self.bin) + os.pathsep + str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"],
            "AIC_PROJECT_DIR": str(self.project), "AIC_OUTPUT_ROOT": str(self.output),
            "AIC_PYTHON": str(self.bin / "fake-python"), "AIC_GPU": "3", "AIC_BASELINE_GPU": "6",
        }
        for key in ("AIC_BENCHMARK", "AIC_SELECTION", "AIC_V14_STAGE", "AIC_FAKE_FAIL_STAGE", "AIC_FAKE_BLOCK_STAGE"):
            self.environment.pop(key, None)
        self.command = ["bash", str(self.project / "scripts" / "run_v14_pipeline.sh")]

    def _executable(self, name, text):
        destination = self.bin / name
        destination.write_text(text)
        destination.chmod(0o755)

    def run(self, **env):
        return subprocess.run(self.command, env={**self.environment, **env}, text=True,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=20)

    def calls(self):
        return [json.loads(line) for line in (self.output / "calls.jsonl").read_text().splitlines()]

    def status(self):
        return json.loads((self.output / "v14_pipeline" / "status.json").read_text())


class V14PipelineTests(unittest.TestCase):
    def test_complete_both_winner_paths_and_idempotent_restart(self):
        for winner in ("expanded_robust", "expanded"):
            with self.subTest(winner=winner), tempfile.TemporaryDirectory() as directory:
                fixture = PipelineFixture(Path(directory))
                result = fixture.run(AIC_FAKE_WINNER=winner)
                self.assertEqual(result.returncode, 0, result.stdout)
                self.assertEqual(fixture.status()["status"], "complete")
                calls = fixture.calls()
                self.assertEqual(len(calls), 6)
                by_stage = {entry["stage"]: entry for entry in calls}
                self.assertEqual(set(by_stage), {"baseline", "validate", "evaluate", "select", "refit", "predict"})
                self.assertEqual(by_stage["baseline"]["gpu"], "6")
                self.assertEqual(by_stage["select"]["gpu"], "")
                self.assertEqual(by_stage["refit"]["selected_recipe"], winner)
                for stage in ("validate", "evaluate", "refit", "predict"):
                    self.assertEqual(by_stage[stage]["gpu"], "3")
                    self.assertEqual(by_stage[stage]["pin"], "0")
                args = by_stage["validate"]["args"]
                self.assertEqual(args[args.index("--batch-size") + 1], "128")
                self.assertEqual(args[args.index("--gradient-accumulation") + 1], "2")
                self.assertEqual(args[args.index("--workers") + 1], "4")
                self.assertIn(str(fixture.output / "v13_expanded/cache/frozen.npy"), args)
                self.assertNotIn("--neighbor-cache", by_stage["refit"]["args"])
                self.assertTrue((fixture.output / "v14_refit/pred_results.zip").is_file())
                repeated = fixture.run(AIC_FAKE_WINNER=winner)
                self.assertEqual(repeated.returncode, 0, repeated.stdout)
                self.assertEqual(fixture.calls(), calls)

    def test_training_failure_records_state_and_restart_uses_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = PipelineFixture(Path(directory))
            failed = fixture.run(AIC_FAKE_FAIL_STAGE="validate")
            self.assertNotEqual(failed.returncode, 0)
            self.assertEqual(fixture.status()["stage"], "baseline_and_validation")
            self.assertEqual(fixture.status()["status"], "failed")
            self.assertFalse((fixture.output / "v14_pipeline/validation.done").exists())
            resumed = fixture.run()
            self.assertEqual(resumed.returncode, 0, resumed.stdout)
            calls = [entry for entry in fixture.calls() if entry["stage"] == "validate"]
            self.assertEqual(len(calls), 2)
            self.assertNotIn("--resume", calls[0]["args"])
            self.assertIn(str(fixture.output / "v14_expanded_robust/resume_latest.pt"), calls[1]["args"])
            self.assertEqual(fixture.status()["status"], "complete")

    def test_refit_restart_preserves_selection_even_if_marker_is_missing(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = PipelineFixture(Path(directory))
            failed = fixture.run(AIC_FAKE_FAIL_STAGE="refit", AIC_FAKE_WINNER="expanded")
            self.assertNotEqual(failed.returncode, 0)
            self.assertEqual(fixture.status()["stage"], "full_data_refit")
            selection = fixture.output / "v14_selection.json"
            frozen = selection.read_bytes()
            (fixture.output / "v14_pipeline/selection.done").unlink()
            resumed = fixture.run(AIC_FAKE_WINNER="expanded_robust")
            self.assertEqual(resumed.returncode, 0, resumed.stdout)
            self.assertEqual(selection.read_bytes(), frozen)
            calls = fixture.calls()
            self.assertEqual(sum(entry["stage"] == "select" for entry in calls), 1)
            refits = [entry for entry in calls if entry["stage"] == "refit"]
            self.assertEqual(len(refits), 2)
            self.assertEqual(refits[-1]["selected_recipe"], "expanded")
            self.assertIn("--resume", refits[-1]["args"])

    def test_active_pipeline_lock_rejects_a_second_launch(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = PipelineFixture(Path(directory))
            process = subprocess.Popen(fixture.command,
                                       env={**fixture.environment, "AIC_FAKE_BLOCK_STAGE": "validate"},
                                       text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            try:
                deadline = time.monotonic() + 12
                while not (fixture.output / "blocked.started").exists() and time.monotonic() < deadline:
                    time.sleep(.01)
                self.assertTrue((fixture.output / "blocked.started").is_file())
                duplicate = fixture.run()
                self.assertEqual(duplicate.returncode, 2, duplicate.stdout)
                self.assertIn("already active", duplicate.stdout)
            finally:
                (fixture.output / "release").touch()
                output, _ = process.communicate(timeout=20)
            self.assertEqual(process.returncode, 0, output)


if __name__ == "__main__":
    unittest.main()
