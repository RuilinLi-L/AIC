from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from audit_generalization_v19 import source_record
from v19_pipeline_support import (RESOURCE_SPECS, benchmark_needed, benchmark_settings,
    benchmark_status, freeze_resources, read_resource_plan, validate_audit, sha256)


class ResourceTests(unittest.TestCase):
    def fixture(self, root, oom=()):
        audit_dir=root/'audit'; audit_dir.mkdir()
        sources={};artifacts={}
        for mapping,names in ((sources,('manifest','feature_cache','feature_metadata')),
                              (artifacts,('decoded_overlap','decoded_images','validation_neighbors'))):
            for name in names:
                p=audit_dir/name;p.write_text('unchanged');mapping[name]=source_record(p)
        audit=audit_dir/'audit.json'
        audit.write_text(json.dumps({'format_version':19,'kind':'v19_generalization_audit',
            'status':'passed','decoded_cross_split_groups':0,'decoded_cross_split_pairs':0,
            'dataset_signature':'dataset','base_model_identity':'official','sources':sources,'artifacts':artifacts}))
        baseline=root/'v18_rank32';baseline.mkdir()
        for name in ('model.pt','strict_eval.json'): (baseline/name).write_text('{}')
        runs=[];probes=[];checks=[]
        for recipe,spec in RESOURCE_SPECS.items():
            run=root/recipe;run.mkdir();runs.append(run)
            report={'format_version':19,'recipe':recipe,'success':recipe not in oom,
                'status':'cuda_oom' if recipe in oom else 'ok', 'initialization':'official_base_only',
                'trained_checkpoint_weights_loaded':False,'pin_memory':False,'prefetch_factor':1,
                'chosen_workers':8,'batch_size':256,'gradient_accumulation':1,'dataset_signature':'dataset',
                'base_model_identity':'official',**spec,
                'configurations':[{'batch_size':256,'status':'out_of_memory'}] if recipe in oom else []}
            probe=run/'benchmark.json';probe.write_text(json.dumps(report));probes.append(probe)
            check=run/'official.json';check.write_text(json.dumps({'format_version':19,'recipe':recipe,
                'success':True,'lora_modules':72,'optimizer_steps':2,'base_model_identity':'official',**spec,
                'frozen_backbone_unchanged':True,'roundtrip_logits_exact':True,
                'all_attention_and_mlp_gradients_nonzero':True,'missing_mlp_weight_rejected':True}));checks.append(check)
        with patch('v19_core.model_identity',return_value='official'):
            result=freeze_resources(root,*probes,*checks,*runs,baseline,'model',audit_path=audit)
        return result,root/'v19_pipeline/resource_plan.json',audit

    def test_freeze_records_science_and_rejects_mutated_sources(self):
        with tempfile.TemporaryDirectory() as tmp,patch.dict(os.environ,{},clear=True):
            resource,path,audit=self.fixture(Path(tmp))
            self.assertEqual(read_resource_plan(path),resource)
            a=resource['candidates']['candidate_a']
            self.assertEqual((a['zoom_shortest_edge'],a['lora_alpha'],a['batch_size']),(439,64.,256))
            Path(a['benchmark']['path']).write_text('{}')
            with self.assertRaisesRegex(ValueError,'source changed'): read_resource_plan(path)

    def test_only_actual_recorded_cuda_oom_allows_candidate_absence(self):
        with tempfile.TemporaryDirectory() as tmp,patch.dict(os.environ,{},clear=True):
            resource,path,audit=self.fixture(Path(tmp),oom=('resolution384_rank32',))
            a=resource['candidates']['candidate_a'];p=Path(a['benchmark']['path'])
            self.assertEqual(a['status'],'resource_infeasible')
            self.assertFalse(benchmark_needed(p,a['recipe']))
            with self.assertRaisesRegex(ValueError,'cannot train'): benchmark_settings(p,a['recipe'])
            report=json.loads(p.read_text());report['configurations']=[];p.write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError,'actual CUDA OOM'): benchmark_status(p,a['recipe'])

    def test_128_microbatch_and_generic_failures_never_become_resource_fallback(self):
        with tempfile.TemporaryDirectory() as tmp,patch.dict(os.environ,{},clear=True):
            resource,path,audit=self.fixture(Path(tmp))
            b=resource['candidates']['candidate_b'];p=Path(b['benchmark']['path'])
            original=json.loads(p.read_text())
            for changes in ({'batch_size':128,'gradient_accumulation':2},
                            {'success':False,'status':'runtime_error'}):
                p.write_text(json.dumps({**original,**changes}))
                with self.assertRaises(ValueError): benchmark_needed(p,b['recipe'])

    def test_data_overlap_and_modified_artifacts_fail_audit_gate(self):
        with tempfile.TemporaryDirectory() as tmp,patch.dict(os.environ,{},clear=True):
            resource,path,audit=self.fixture(Path(tmp));original=json.loads(audit.read_text())
            for changes in ({'status':'failed'},{'decoded_cross_split_groups':1},{'decoded_cross_split_pairs':1}):
                audit.write_text(json.dumps({**original,**changes}))
                with self.assertRaisesRegex(ValueError,'audit failed'): validate_audit(audit)
            audit.write_text(json.dumps(original))
            Path(original['artifacts']['validation_neighbors']['path']).write_text('altered')
            with self.assertRaisesRegex(ValueError,'artifact changed'): validate_audit(audit)

    def test_resource_skip_status_cannot_be_forged(self):
        with tempfile.TemporaryDirectory() as tmp,patch.dict(os.environ,{},clear=True):
            resource,path,audit=self.fixture(Path(tmp))
            resource['candidates']['candidate_a']['status']='resource_infeasible';path.write_text(json.dumps(resource))
            with self.assertRaisesRegex(ValueError,'status differs'): read_resource_plan(path)


if __name__ == '__main__': unittest.main()
