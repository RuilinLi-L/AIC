"""Submission collection rejects changed contracts and mismatched test filenames."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import torch

from v15_pipeline_support import baseline_selection, checkpoint_path, export_submission, sha256, wait_v14


class PipelineSupportTests(unittest.TestCase):
    def fixture(self, root):
        (root/'v14_pipeline').mkdir()
        (root/'v14_pipeline/selection.done').touch()
        (root/'v14_pipeline/prediction.done').touch()
        selection_path=root/'v14_selection.json'; selection_path.write_text('{"frozen":true}')
        model=root/'v14_refit/model.pt'; model.parent.mkdir()
        selection={'source_checkpoint':str(root/'validation.pt'),'selected_epoch':18,
                   'recipe':'expanded','class_names':['0000','0001'],'dataset_signature':'dataset',
                   'class_bias':[0.,0.], 'calibration':{'view_weights':{'center':1.},'alpha':0.,'precision':'fp32'}}
        checkpoint={'training_stage':'refit','selected_epoch':18,'config':{'recipe':'expanded'},
                    'refit_selection_sha256':sha256(selection_path),'dataset_signature':'dataset',
                    'class_names':['0000','0001'],'class_bias':torch.zeros(2),'calibration':selection['calibration']}
        torch.save(checkpoint,model)
        test=root/'test'; test.mkdir()
        for name in ('a.jpg','b.jpg'): (test/name).touch()
        csv=root/'v14_refit/pred_results.csv'; csv.write_text('a.jpg, 0000\nb.jpg, 0001\n')
        with zipfile.ZipFile(root/'v14_refit/pred_results.zip','w') as archive:
            archive.write(csv,'pred_results.csv')
        return selection, checkpoint, model, test

    def test_collect_validates_contract_and_writes_read_only_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); selection,checkpoint,model,test=self.fixture(root)
            original={p.name:p.read_bytes() for p in (root/'v14_refit').iterdir()}
            with patch('select_v14.load_selection',return_value=selection):
                result=export_submission(root,'v14_refit',model,2,True,test)
                self.assertEqual(result['recipe'],'expanded')
                self.assertEqual(result['checkpoint_sha256'],sha256(model))
                self.assertIsNone(result['online_score'])
                self.assertEqual(result['target_online_score'],72)
                self.assertEqual(result,export_submission(root,'v14_refit',model,2,False,test))
            self.assertEqual(original,{p.name:p.read_bytes() for p in (root/'v14_refit').iterdir()})

    def test_refit_rejects_changed_bias_classes_and_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); selection,original,model,_=self.fixture(root)
            for key,value in (('class_bias',torch.ones(2)),('class_names',['0001','0000']),('dataset_signature','other')):
                with self.subTest(key=key),patch('select_v14.load_selection',return_value=selection):
                    changed=deepcopy(original); changed[key]=value; torch.save(changed,model)
                    with self.assertRaises(ValueError): checkpoint_path(root,'v14_refit')

    def test_collection_rejects_wrong_test_filenames_even_when_zip_matches(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); selection,_,model,test=self.fixture(root)
            (test/'b.jpg').rename(test/'c.jpg')
            with patch('select_v14.load_selection',return_value=selection),self.assertRaisesRegex(ValueError,'filenames'):
                export_submission(root,'v14_refit',model,2,True,test)
            self.assertFalse((root/'v15_delivery/v14_refit/provenance.json').exists())

    def test_consumed_selection_is_immutable_and_waits_are_bounded(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); selection,_,_,_=self.fixture(root)
            with patch('select_v14.load_selection',return_value=selection):
                baseline_selection(root)
                (root/'v14_selection.json').write_text('{"frozen":false}')
                with self.assertRaisesRegex(ValueError,'changed'): baseline_selection(root)
            (root/'v14_pipeline/selection.done').unlink()
            with self.assertRaises(TimeoutError): wait_v14(root,'selection',0,0)
            (root/'v14_pipeline/status.json').write_text(json.dumps({'stage':'train','status':'failed'}))
            with self.assertRaisesRegex(RuntimeError,'dependency failed'): wait_v14(root,'selection',10,0)


if __name__ == '__main__': unittest.main()
