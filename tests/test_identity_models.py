"""Opt-in real-model check: RUN_MODEL_TESTS=1 python -m unittest discover -s tests -v."""
import os
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from core import config
from core.face_analyser import get_face_analyser
from core.engine import face_swapper


@unittest.skipUnless(os.environ.get('RUN_MODEL_TESTS') == '1', 'requires installed models')
class IdentityModelTests(unittest.TestCase):
    def test_standard_swap_preserves_selected_source_identity(self):
        settings = dict(EXECUTION_PROVIDERS=['CPUExecutionProvider'],
                        MOUTH_MASK_ENABLED=False, EYES_MASK_ENABLED=False,
                        EYEBROWS_MASK_ENABLED=False, POISSON_BLEND_ENABLED=False,
                        OPACITY=1.0)
        with patch.multiple(config, **settings), patch.object(face_swapper, 'swapper_', None):
            analyser = get_face_analyser()
            source_image = cv2.imread(str(config.BASE_DIR / 'assets/elon_musk.jpg'))
            target_image = cv2.imread(str(config.BASE_DIR / 'assets/vijay.jpg'))
            source_faces = analyser.get(source_image)
            target_faces = analyser.get(target_image)
            self.assertEqual(len(source_faces), 1)
            self.assertEqual(len(target_faces), 1)
            source, target = source_faces[0], target_faces[0]
            output = face_swapper.swap_face(source, target, target_image)
            faces = analyser.get(output)
            self.assertEqual(len(faces), 1)
            identity = faces[0].normed_embedding
            source_similarity = float(np.dot(identity, source.normed_embedding))
            target_similarity = float(np.dot(identity, target.normed_embedding))
            self.assertGreater(source_similarity, 0.5)
            self.assertGreater(source_similarity, target_similarity + 0.2)


if __name__ == '__main__':
    unittest.main()
