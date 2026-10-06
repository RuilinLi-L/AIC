import os
import unittest
from unittest.mock import patch
from v19_runtime import pipeline_eta, scientific_config, training_history_seconds


class RuntimeTests(unittest.TestCase):
    def test_pipeline_eta_counts_preparation_and_waits_without_shortening_schedule(self):
        history=[{'epoch':epoch,'train_seconds':100,'validation_seconds':10,'audit_seconds':20} for epoch in (1,2)]
        with patch.dict(os.environ,{'AIC_PIPELINE_STARTED_AT':'100'}),patch('v19_runtime.time.time',return_value=181000):
            result=pipeline_eta(history,stage='validate',stop_epoch=24,train_rows=80,final_rows=100)
        self.assertEqual(result['elapsed_wall_clock_seconds'],180900)
        self.assertTrue(result['external_dependency_wait_included'])
        self.assertTrue(result['budget_exceeded'])
        self.assertEqual(result['stop_epoch'],24)
        self.assertIsNone(result['future_dependency_wait_seconds'])
    def test_actual_batch_contract_and_runtime_only_fields(self):
        original = {'recipe': 'resolution384_rank32', 'batch_size': 256,
                    'gradient_accumulation': 1, 'workers': 8}
        self.assertEqual(scientific_config(original), original)
        self.assertEqual(scientific_config({**original, 'prefetch_factor': 2, 'eval_batch_size': 32}), original)
        for batch, accumulation in ((128, 2), (128, 1), (256.0, 1), (256, True), (True, 256)):
            with self.subTest(batch=batch, accumulation=accumulation), self.assertRaises(ValueError):
                scientific_config({**original, 'batch_size': batch, 'gradient_accumulation': accumulation})
        self.assertNotEqual(scientific_config({**original, 'lora_dropout': 0.0}),
                            scientific_config({**original, 'lora_dropout': 0.05}))

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
