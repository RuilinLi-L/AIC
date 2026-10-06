import os
import unittest
from unittest.mock import patch
from v16_runtime import pipeline_eta, scientific_config


class RuntimeTests(unittest.TestCase):
    def test_pipeline_eta_counts_preparation_and_waits_without_shortening_schedule(self):
        history=[{'epoch':epoch,'train_seconds':100,'validation_seconds':10,'audit_seconds':20} for epoch in (1,2)]
        with patch.dict(os.environ,{'AIC_PIPELINE_STARTED_AT':'100'}),patch('v16_runtime.time.time',return_value=181000):
            result=pipeline_eta(history,stage='validate',stop_epoch=24,train_rows=80,final_rows=100)
        self.assertEqual(result['elapsed_wall_clock_seconds'],180900)
        self.assertTrue(result['external_dependency_wait_included'])
        self.assertTrue(result['budget_exceeded'])
        self.assertEqual(result['stop_epoch'],24)
        self.assertIsNone(result['future_dependency_wait_seconds'])
    def test_effective_batch_contract_and_runtime_only_fields(self):
        result=scientific_config({'recipe':'mlp_light','batch_size':128,'gradient_accumulation':2,'workers':8})
        self.assertEqual(result,{'recipe':'mlp_light'})
        with self.assertRaises(ValueError): scientific_config({'batch_size':128,'gradient_accumulation':1})

if __name__=='__main__': unittest.main()
