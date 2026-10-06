"""V16 dependency, resource, and submission contracts do not silently change sources."""
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import torch

from v16_pipeline_support import (benchmark_needed, benchmark_settings, checkpoint_path, decision, export_submission,
                                  initialize, reusable_package, sha256, state, wait_baselines, wait_gpu)


class SupportTests(unittest.TestCase):
    def fixture(self, root, accepted=True):
        selection_path=root/'v16_selection.json'; selection_path.write_text('{"selected":true}')
        calibration={'view_weights':{'center':1.},'alpha':0.,'precision':'fp32'}
        cp={'training_stage':'validate','selected_epoch':18,'config':{'recipe':'mlp_light'},
            'class_names':['0000','0001'],'dataset_signature':'dataset','class_bias':torch.zeros(2),
            'calibration':calibration}
        model=root/'validation.pt'; torch.save(cp,model)
        selected={'recipe':'mlp_light','refit_required':accepted,'source_checkpoint':str(model),
            'source_checkpoint_sha256':sha256(model),'selected_epoch':18,'class_names':['0000','0001'],
            'dataset_signature':'dataset','class_bias':[0.,0.],'calibration':calibration}
        test=root/'test'; test.mkdir()
        for name in ('a.jpg','b.jpg'): (test/name).touch()
        return selected,cp,model,test
    def package(self,directory,model):
        directory.mkdir(parents=True)
        csv=directory/'pred_results.csv'; csv.write_text('a.jpg, 0000\nb.jpg, 0001\n')
        with zipfile.ZipFile(directory/'pred_results.zip','w') as archive: archive.write(csv,'pred_results.csv')
        (directory/'provenance.json').write_text(json.dumps({'checkpoint':str(model),'checkpoint_sha256':sha256(model)}))
    def test_budget_start_survives_restart_and_counts_queue_time(self):
        with tempfile.TemporaryDirectory() as tmp,patch.dict(os.environ,{},clear=True):
            root=Path(tmp)
            with patch('v16_pipeline_support.time.time',return_value=100): first=initialize(root)
            with patch('v16_pipeline_support.time.time',return_value=200):
                self.assertEqual(first,initialize(root))
                status=state(root,'gpu_queue','waiting')
            self.assertEqual(status['elapsed_wall_clock_seconds'],100)
            self.assertFalse(status['automatic_short_training'])
    def test_gpu_low_memory_is_bounded_and_does_not_launch_or_kill(self):
        with tempfile.TemporaryDirectory() as tmp,patch.dict(os.environ,{},clear=True),patch(
                'v16_pipeline_support.subprocess.run',return_value=subprocess.CompletedProcess([],0,'100\n','')) as run:
            root=Path(tmp)
            with self.assertRaises(TimeoutError): wait_gpu(root,'4',32768,0,0)
            self.assertEqual(run.call_args.args[0][0],'nvidia-smi')
            status=json.loads((root/'v16_pipeline/gpu_status.json').read_text())
            self.assertEqual(status['status'],'waiting')
    def test_baseline_binding_detects_change_and_wait_timeout(self):
        with tempfile.TemporaryDirectory() as tmp,patch.dict(os.environ,{},clear=True):
            root=Path(tmp); directories=[root/'v14_baseline_expanded',root/'v15_expanded_mlp']
            with self.assertRaises(TimeoutError): wait_baselines(root,directories,0,0)
            for directory in directories:
                directory.mkdir()
                for name in ('model.pt','strict_eval.json'): (directory/name).write_text('{}')
            wait_baselines(root,directories,0,0)
            (directories[1]/'model.pt').write_text('changed')
            with self.assertRaisesRegex(ValueError,'changed'): wait_baselines(root,directories,0,0)
    def test_refit_requires_frozen_selection_and_matching_bias(self):
        with tempfile.TemporaryDirectory() as tmp,patch.dict(os.environ,{},clear=True):
            root=Path(tmp); selected,cp,model,test=self.fixture(root)
            with patch('select_v16.read_decision',return_value=selected):
                self.assertEqual(checkpoint_path(root,'validation'),model)
                cp.update(training_stage='refit',refit_selection_sha256=sha256(root/'v16_selection.json'))
                refit=root/'v16_refit/model.pt'; refit.parent.mkdir(); torch.save(cp,refit)
                self.assertEqual(checkpoint_path(root,'refit'),refit)
                cp['class_bias']=torch.ones(2); torch.save(cp,refit)
                with self.assertRaisesRegex(ValueError,'differs'): checkpoint_path(root,'refit')
                (root/'v16_selection.json').write_text('{"changed":true}')
                with self.assertRaisesRegex(ValueError,'changed'): decision(root)
    def test_fallback_reuses_only_matching_provenance_and_validates_filenames(self):
        with tempfile.TemporaryDirectory() as tmp,patch.dict(os.environ,{},clear=True):
            root=Path(tmp); selected,cp,model,test=self.fixture(root,False)
            source=root/'v15_delivery/v14_validation'; self.package(source,model)
            before={p.name:p.read_bytes() for p in source.iterdir()}
            with patch('select_v16.read_decision',return_value=selected):
                self.assertEqual(reusable_package(root,'validation')['checkpoint'],str(model))
                result=export_submission(root,'validation',model,test,2,source)
                self.assertIsNone(result['online_score']); self.assertEqual(result['linecount'],2)
                self.assertEqual(before,{p.name:p.read_bytes() for p in source.iterdir()})
                self.assertIsNone(reusable_package(root,'refit'))
                with self.assertRaisesRegex(ValueError,'accepted'): checkpoint_path(root,'refit')
                (test/'b.jpg').rename(test/'c.jpg')
                with self.assertRaisesRegex(ValueError,'filenames'): export_submission(root,'validation',model,test,2)
    def test_benchmark_requires_actual_oom_before_smaller_microbatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'benchmark.json'
            self.assertTrue(benchmark_needed(path,'mlp_light'))
            path.write_text(json.dumps({'format_version':16,'recipe':'mlp_light','success':False}))
            self.assertTrue(benchmark_needed(path,'mlp_light'))
            report={'format_version':16,'recipe':'mlp_light','success':True,'initialization':'official_base_only',
                    'trained_checkpoint_weights_loaded':False,'pin_memory':False,'prefetch_factor':1,
                    'chosen_workers':8,'batch_size':128,'gradient_accumulation':2,'configurations':[]}
            path.write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError,'actual'): benchmark_settings(path,'mlp_light')
            report['configurations']=[{'batch_size':256,'status':'out_of_memory'}]
            path.write_text(json.dumps(report)); self.assertEqual(benchmark_settings(path,'mlp_light'),(8,128))
            with self.assertRaisesRegex(ValueError,'recipe'): benchmark_settings(path,'mlp_light_ln')

if __name__=='__main__': unittest.main()
