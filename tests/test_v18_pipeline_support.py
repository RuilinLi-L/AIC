"""V18 dependency, resource, and submission contracts do not silently change sources."""
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

from v18_pipeline_support import (benchmark_needed, benchmark_settings, checkpoint_path, decision, export_submission,
                                  freeze_resources, initialize, official_check, read_resource_plan, resource_for_recipe,
                                  reusable_package, sha256, state, wait_baselines, wait_gpu)


class SupportTests(unittest.TestCase):
    def resources(self, root, branch='standard'):
        baseline=root/'v15_expanded_mlp'; baseline.mkdir(exist_ok=True)
        for name in ('model.pt','strict_eval.json'): (baseline/name).write_text('{}')
        batch=256 if branch=='standard' else 128
        recipes=('resolution384','rank32' if branch=='standard' else 'matched_control')
        runs=[]; reports=[]; checks=[]
        for recipe in recipes:
            run=root/('v18_'+recipe); run.mkdir(exist_ok=True); runs.append(run)
            size=384 if recipe=='resolution384' else 320
            rank=32 if recipe=='rank32' else 16
            report={'format_version':18,'recipe':recipe,'success':True,'initialization':'official_base_only',
                    'trained_checkpoint_weights_loaded':False,'pin_memory':False,'prefetch_factor':1,
                    'chosen_workers':8,'batch_size':batch,'gradient_accumulation':256//batch,
                    'image_size':size,'lora_rank':rank,'base_model_identity':'official',
                    'dataset_signature':'dataset','configurations':([{'batch_size':256,'status':'out_of_memory'}]
                         if recipe=='resolution384' and batch==128 else [])}
            probe=run/'benchmark.json'; probe.write_text(json.dumps(report)); reports.append(probe)
            official={'format_version':18,'recipe':recipe,'success':True,'lora_modules':72,
                'visual_trainable_parameters':5308416 if rank==32 else 2654208,'optimizer_steps':2,
                'base_model_identity':'official','image_size':size,'lora_rank':rank,
                'frozen_backbone_unchanged':True,'roundtrip_logits_exact':True,
                'all_attention_and_mlp_gradients_nonzero':True,'missing_mlp_weight_rejected':True}
            check=run/'official_check.json'; check.write_text(json.dumps(official)); checks.append(check)
        with patch('v18_core.model_identity',return_value='official'):
            resource=freeze_resources(root,*reports,*checks,*runs,baseline,'model')
        return resource,root/'v18_pipeline/resource_plan.json'

    def fixture(self, root, accepted=True):
        selection_path=root/'v18_selection.json'; selection_path.write_text('{"selected":true}')
        calibration={'view_weights':{'center':1.},'alpha':0.,'precision':'fp32'}
        recipe='resolution384' if accepted else 'expanded_mlp'
        version=18 if accepted else 15
        cp={'format_version':version,'training_stage':'validate','selected_epoch':18,
            'config':{'recipe':recipe,'base_model_identity':'official'},
            'class_names':['0000','0001'],'dataset_signature':'dataset','class_bias':torch.zeros(2),
            'calibration':calibration}
        model=root/'validation.pt'; torch.save(cp,model)
        selected={'recipe':recipe,'source_format_version':version,'base_model_identity':'official',
            'refit_required':accepted,'source_checkpoint':str(model),
            'source_checkpoint_sha256':sha256(model),'selected_epoch':18,'class_names':['0000','0001'],
            'dataset_signature':'dataset','class_bias':[0.,0.],'calibration':calibration}
        resource,path=self.resources(root)
        selected.update(resource_sha256=sha256(path),resource_branch=resource['branch'])
        test=root/'test'; test.mkdir()
        for name in ('a.jpg','b.jpg'): (test/name).touch()
        return selected,cp,model,test
    def package(self,directory,model):
        directory.mkdir(parents=True)
        csv=directory/'pred_results.csv'; csv.write_text('a.jpg, 0000\nb.jpg, 0001\n')
        with zipfile.ZipFile(directory/'pred_results.zip','w') as archive: archive.write(csv,'pred_results.csv')
        (directory/'provenance.json').write_text(json.dumps({'checkpoint':str(model),'checkpoint_sha256':sha256(model),
            'csv_sha256':sha256(csv),'zip_sha256':sha256(directory/'pred_results.zip')}))
    def test_budget_start_survives_restart_and_counts_queue_time(self):
        with tempfile.TemporaryDirectory() as tmp,patch.dict(os.environ,{},clear=True):
            root=Path(tmp)
            with patch('v18_pipeline_support.time.time',return_value=100): first=initialize(root)
            with patch('v18_pipeline_support.time.time',return_value=200):
                self.assertEqual(first,initialize(root))
                status=state(root,'gpu_queue','waiting')
            self.assertEqual(status['elapsed_wall_clock_seconds'],100)
            self.assertFalse(status['automatic_short_training'])
    def test_gpu_low_memory_is_bounded_and_does_not_launch_or_kill(self):
        with tempfile.TemporaryDirectory() as tmp,patch.dict(os.environ,{},clear=True),patch(
                'v18_pipeline_support.subprocess.run',return_value=subprocess.CompletedProcess([],0,'100\n','')) as run:
            root=Path(tmp)
            with self.assertRaises(TimeoutError): wait_gpu(root,'4',32768,0,0)
            self.assertEqual(run.call_args.args[0][0],'nvidia-smi')
            status=json.loads((root/'v18_pipeline/gpu_status.json').read_text())
            self.assertEqual(status['status'],'waiting')
    def test_baseline_binding_detects_change_and_wait_timeout(self):
        with tempfile.TemporaryDirectory() as tmp,patch.dict(os.environ,{},clear=True):
            root=Path(tmp); directories=[root/'v15_expanded_mlp']
            with self.assertRaises(TimeoutError): wait_baselines(root,directories,0,0)
            for directory in directories:
                directory.mkdir()
                for name in ('model.pt','strict_eval.json'): (directory/name).write_text('{}')
            wait_baselines(root,directories,0,0)
            (directories[0]/'model.pt').write_text('changed')
            with self.assertRaisesRegex(ValueError,'changed'): wait_baselines(root,directories,0,0)
    def test_refit_requires_frozen_selection_and_matching_bias(self):
        with tempfile.TemporaryDirectory() as tmp,patch.dict(os.environ,{},clear=True):
            root=Path(tmp); selected,cp,model,test=self.fixture(root)
            with patch('select_v18.read_decision',return_value=selected):
                self.assertEqual(checkpoint_path(root,'validation'),model)
                cp.update(training_stage='refit',refit_selection_sha256=sha256(root/'v18_selection.json'))
                refit=root/'v18_refit/model.pt'; refit.parent.mkdir(); torch.save(cp,refit)
                self.assertEqual(checkpoint_path(root,'refit'),refit)
                cp['class_bias']=torch.ones(2); torch.save(cp,refit)
                with self.assertRaisesRegex(ValueError,'differs'): checkpoint_path(root,'refit')
                (root/'v18_selection.json').write_text('{"changed":true}')
                with self.assertRaisesRegex(ValueError,'changed'): decision(root)
    def test_fallback_reuses_only_matching_provenance_and_validates_filenames(self):
        with tempfile.TemporaryDirectory() as tmp,patch.dict(os.environ,{},clear=True):
            root=Path(tmp); selected,cp,model,test=self.fixture(root,False)
            source=root/'v15_delivery/v15_validation'; self.package(source,model)
            before={p.name:p.read_bytes() for p in source.iterdir()}
            with patch('select_v18.read_decision',return_value=selected):
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
            self.assertTrue(benchmark_needed(path,'resolution384'))
            path.write_text(json.dumps({'format_version':18,'recipe':'resolution384','success':False}))
            self.assertTrue(benchmark_needed(path,'resolution384'))
            report={'format_version':18,'recipe':'resolution384','success':True,'initialization':'official_base_only',
                    'trained_checkpoint_weights_loaded':False,'pin_memory':False,'prefetch_factor':1,
                    'chosen_workers':8,'batch_size':128,'gradient_accumulation':2,'configurations':[],
                    'image_size':384,'lora_rank':16}
            path.write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError,'actual'): benchmark_settings(path,'resolution384')
            report['configurations']=[{'batch_size':256,'status':'out_of_memory'}]
            path.write_text(json.dumps(report)); self.assertEqual(benchmark_settings(path,'resolution384'),(8,128))
            with self.assertRaisesRegex(ValueError,'recipe'): benchmark_settings(path,'rank32')

    def test_official_check_binds_recipe_and_actual_official_weights(self):
        with tempfile.TemporaryDirectory() as tmp,patch('v18_core.model_identity',return_value='official'):
            path=Path(tmp)/'official_check.json'
            record={'format_version':18,'recipe':'resolution384','success':True,'lora_modules':72,
                'visual_trainable_parameters':2654208,'optimizer_steps':2,'base_model_identity':'official',
                'image_size':384,'lora_rank':16,
                'frozen_backbone_unchanged':True,'roundtrip_logits_exact':True,
                'all_attention_and_mlp_gradients_nonzero':True,'missing_mlp_weight_rejected':True}
            path.write_text(json.dumps(record))
            self.assertEqual(official_check(path,'resolution384','model'),record)
            with self.assertRaisesRegex(ValueError,'recipe'): official_check(path,'rank32','model')
            record['base_model_identity']='other'; path.write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError,'identity'): official_check(path,'resolution384','model')

    def test_fallback_refit_requires_matching_v15_model_and_untampered_package(self):
        with tempfile.TemporaryDirectory() as tmp,patch.dict(os.environ,{},clear=True):
            root=Path(tmp); selected,cp,model,test=self.fixture(root,False)
            cp['training_stage']='refit'; refit=root/'refit.pt'; torch.save(cp,refit)
            source=root/'v15_delivery/v15_refit'; self.package(source,refit)
            with patch('select_v18.read_decision',return_value=selected):
                self.assertEqual(reusable_package(root,'refit')['checkpoint'],str(refit))
                result=export_submission(root,'fallback_refit',refit,test,2,source)
                self.assertEqual(result['recipe'],'expanded_mlp')
                self.assertEqual(result['source_format_version'],15)
                # Repacking an altered CSV must not bypass its existing provenance.
                csv=source/'pred_results.csv'; csv.write_text('a.jpg, 0001\nb.jpg, 0000\n')
                with zipfile.ZipFile(source/'pred_results.zip','w') as archive: archive.write(csv,'pred_results.csv')
                self.assertIsNone(reusable_package(root,'refit'))
                with self.assertRaisesRegex(ValueError,'hash'): export_submission(root,'fallback_refit',refit,test,2,source)

    def test_resource_branch_freezes_both_probes_before_training_and_detects_mutation(self):
        with tempfile.TemporaryDirectory() as tmp,patch.dict(os.environ,{},clear=True):
            root=Path(tmp); resource,path=self.resources(root)
            self.assertEqual(read_resource_plan(path),resource)
            self.assertEqual(resource_for_recipe(resource,'rank32')['batch_size'],256)
            with self.assertRaisesRegex(ValueError,'not active'): resource_for_recipe(resource,'matched_control')
            probe=Path(resource['candidates']['candidate_b']['benchmark']['path'])
            data=json.loads(probe.read_text()); data['batch_size']=128; data['gradient_accumulation']=2
            probe.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError,'source changed'): read_resource_plan(path)

    def test_matched_branch_requires_actual_384_oom_and_only_128_control(self):
        with tempfile.TemporaryDirectory() as tmp,patch.dict(os.environ,{},clear=True):
            root=Path(tmp); resource,path=self.resources(root,'matched_control')
            self.assertEqual(read_resource_plan(path)['branch'],'matched_control')
            self.assertEqual(resource_for_recipe(resource,'matched_control')['batch_size'],128)
            a=resource['candidates']['candidate_a']; probe=Path(a['benchmark']['path'])
            report=json.loads(probe.read_text()); report['configurations']=[]; probe.write_text(json.dumps(report))
            a['benchmark']['sha256']=sha256(probe); path.write_text(json.dumps(resource))
            with self.assertRaisesRegex(ValueError,'actual'): read_resource_plan(path)

    def test_failed_rank_probe_cannot_be_recast_as_128_and_resource_freeze_cannot_be_late(self):
        with tempfile.TemporaryDirectory() as tmp,patch.dict(os.environ,{},clear=True):
            root=Path(tmp); resource,path=self.resources(root)
            a,b=(resource['candidates'][key] for key in ('candidate_a','candidate_b'))
            probe=Path(b['benchmark']['path']); report=json.loads(probe.read_text())
            report.update(batch_size=128,gradient_accumulation=2); probe.write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError,'rank32 requires'): benchmark_settings(probe,'rank32')
            report.update(batch_size=256,gradient_accumulation=1); probe.write_text(json.dumps(report))
            path.unlink(); (Path(a['run_dir'])/'resume_latest.pt').touch()
            with patch('v18_core.model_identity',return_value='official'),self.assertRaisesRegex(ValueError,'before either'):
                freeze_resources(root,a['benchmark']['path'],b['benchmark']['path'],
                    a['official_check']['path'],b['official_check']['path'],a['run_dir'],b['run_dir'],
                    resource['baseline']['directory'],'model')

if __name__=='__main__': unittest.main()
