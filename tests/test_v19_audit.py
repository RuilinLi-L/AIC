import hashlib
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image, PngImagePlugin

from audit_generalization_v19 import decoded_hash, decoded_overlaps, nearest_training


class AuditTests(unittest.TestCase):
    def test_distinct_files_with_equal_decoded_pixels_are_cross_split_overlap(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = Image.new('RGB', (12, 13), (17, 42, 119))
            metadata = PngImagePlugin.PngInfo(); metadata.add_text('source','different encoding')
            a,b = root/'a.png',root/'b.png'
            image.save(a); image.save(b,pnginfo=metadata)
            hashes = [hashlib.sha256(p.read_bytes()).hexdigest() for p in (a,b)]
            self.assertNotEqual(*hashes)
            pixels = [decoded_hash((i,str(p),h))[1] for i,(p,h) in enumerate(zip((a,b),hashes))]
            self.assertEqual(*pixels)
            self.assertEqual(len(decoded_overlaps(pixels,[0],[1])),1)
            with self.assertRaisesRegex(ValueError,'changed'):
                decoded_hash((0,str(a),hashes[1]))

    def test_dimensions_participate_in_pixel_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths=[]
            for i,size in enumerate(((2,6),(3,4))):
                p=Path(tmp)/f'{i}.png'; Image.new('RGB',size,(1,2,3)).save(p); paths.append(p)
            hashes=[decoded_hash((i,str(p),hashlib.sha256(p.read_bytes()).hexdigest()))[1]
                    for i,p in enumerate(paths)]
            self.assertNotEqual(*hashes)

    def test_neighbours_use_only_training_pool_and_deterministic_ties(self):
        f=np.array([[1,0],[1,0],[0,1],[1,0]],np.float32)
        values=list(nearest_training(f,[2,1,0],[3],batch_size=1))
        self.assertEqual(values,[(3,0,1.0)])
        with self.assertRaisesRegex(ValueError,'disjoint'):
            list(nearest_training(f,[0,1],[1]))
        f[0]=np.nan
        with self.assertRaisesRegex(ValueError,'nonfinite'):
            list(nearest_training(f,[0,1],[3]))


if __name__ == '__main__': unittest.main()
