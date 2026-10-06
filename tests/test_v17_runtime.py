import os
import unittest
from unittest.mock import patch
from v17_runtime import pipeline_eta, scientific_config, training_history_seconds


class RuntimeTests(unittest.TestCase):
    def test_pipeline_eta_counts_preparation_and_waits_without_shortening_schedule(self):
        history=[{'epoch':epoch,'train_seconds':100,'validation_seconds':10,'audit_seconds':20} for epoch in (1,2)]
        with patch.dict(os.environ,{'AIC_PIPELINE_STARTED_AT':'100'}),patch('v17_runtime.time.time',return_value=181000):
            result=pipeline_eta(history,stage='validate',stop_epoch=24,train_rows=80,final_rows=100)
        self.assertEqual(result['elapsed_wall_clock_seconds'],180900)
        self.assertTrue(result['external_dependency_wait_included'])
        self.assertTrue(result['budget_exceeded'])
        self.assertEqual(result['stop_epoch'],24)
        self.assertIsNone(result['future_dependency_wait_seconds'])
    def test_effective_batch_contract_and_runtime_only_fields(self):
        result=scientific_config({'recipe':'agreement_recovery','batch_size':128,'gradient_accumulation':2,'workers':8})
        self.assertEqual(result,{'recipe':'agreement_recovery'})
        with self.assertRaises(ValueError): scientific_config({'batch_size':128,'gradient_accumulation':1})

    def test_supervision_preparation_counts_in_elapsed_and_future_eta(self):
        history = [{'epoch': epoch, 'train_seconds': 100, 'validation_seconds': 10,
                    'audit_seconds': 20, 'supervision_seconds': 50} for epoch in (1, 2)]
        without = [{**record, 'supervision_seconds': 0} for record in history]
        self.assertEqual(training_history_seconds(history), 360)
        args = dict(stage='validate', stop_epoch=24, train_rows=100, final_rows=100)
        with patch.dict(os.environ, {}, clear=True):
            actual = pipeline_eta(history, **args)
            reference = pipeline_eta(without, **args)
        self.assertEqual(actual['measured_history_seconds'] - reference['measured_history_seconds'], 100)
        self.assertEqual(actual['remaining_stage_seconds'] - reference['remaining_stage_seconds'], 22 * 50)
        for i, epochs in enumerate((18, 24)):
            self.assertEqual(actual['refit_seconds_range'][i] - reference['refit_seconds_range'][i], epochs * 50)
        self.assertEqual(actual['stop_epoch'], 24)

if __name__=='__main__': unittest.main()
