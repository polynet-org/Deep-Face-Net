import unittest
from types import SimpleNamespace
from unittest.mock import patch, Mock

import cv2
import numpy as np
from core import config
from core.engine import face_masking as masking
from core.engine import face_swapper as swapping
from core.engine import face_enhancer as enhancer
from core import face_analyser


def face_points():
    points = np.zeros((106, 2), np.float32)
    for start, end, center in [(0, 33, (64, 65)), (33, 43, (42, 48)),
                               (43, 52, (42, 35)), (52, 72, (64, 85)),
                               (87, 97, (86, 48)), (97, 106, (86, 35))]:
        angles = np.linspace(0, 2*np.pi, end-start, endpoint=False)
        radius = (48, 55) if start == 0 else (12, 6)
        points[start:end] = np.column_stack((np.cos(angles)*radius[0],
                                             np.sin(angles)*radius[1])) + center
    return points


class MaskTests(unittest.TestCase):
    def setUp(self):
        self.frame = np.full((128, 128, 3), 200, np.uint8)
        self.creators = [masking.create_lower_mouth_mask, masking.create_eyes_mask,
                         masking.create_eyebrows_mask]

    def test_missing_malformed_and_nonfinite_landmarks(self):
        for points in [None, np.zeros((5, 2)), np.full((106, 2), np.nan)]:
            for creator in self.creators:
                with self.subTest(points_shape=getattr(points, 'shape', None), creator=creator):
                    mask, cutout, box, polygon = creator(SimpleNamespace(landmark_2d_106=points), self.frame)
                    self.assertFalse(mask.any())
                    self.assertIsNone(cutout)
                    self.assertEqual(box, (0, 0, 0, 0))

    def test_offscreen_and_degenerate_features(self):
        for points in [face_points()+1000, face_points()-1000, np.zeros((106, 2))]:
            for creator in self.creators:
                self.assertIsNone(creator(SimpleNamespace(landmark_2d_106=points), self.frame)[1])

    def test_clipped_masks_and_restoration(self):
        for offset in [0, -45, 50]:
            for creator in self.creators:
                face = SimpleNamespace(landmark_2d_106=face_points()+offset)
                mask, cutout, box, polygon = creator(face, self.frame)
                result = masking.apply_mask_area(np.zeros_like(self.frame), cutout, box,
                                                 mask, polygon, region_mask=mask)
                self.assertEqual(result.shape, self.frame.shape)
                self.assertTrue(np.all(result[mask == 0] == 0))
                if np.any(mask == 255):
                    self.assertTrue(np.all(result[mask == 255] == 200))

    def test_separate_eyes_and_nonempty_eyebrows(self):
        face = SimpleNamespace(landmark_2d_106=face_points())
        mask = masking.create_eyes_mask(face, self.frame)[0]
        self.assertEqual(mask[48, 64], 0)
        self.assertGreater(mask[48, 42], 200)
        self.assertGreater(masking.create_eyebrows_mask(face, self.frame)[0].max(), 200)

    def test_zero_feather_ratio(self):
        with patch.object(config, 'MASK_FEATHER_RATIO', 0):
            for creator in self.creators:
                creator(SimpleNamespace(landmark_2d_106=face_points()), self.frame)


class PipelineTests(unittest.TestCase):
    def test_cuda_does_not_silently_select_fp16_model(self):
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        with patch.object(config, 'EXECUTION_PROVIDERS', providers), \
             patch.object(swapping, 'swapper_', None), \
             patch.object(swapping.insightface.model_zoo, 'get_model') as load:
            result = swapping.get_face_swapper()
            load.assert_called_once_with(str(config.SWAPPER_MODEL), providers=providers)
            self.assertIs(result, load.return_value)

    def test_selected_source_is_passed_as_identity_input(self):
        source, target = object(), object()
        frame = np.zeros((16, 16, 3), np.uint8)
        with patch.object(swapping, 'get_face_swapper') as factory, \
             patch.object(config, 'MOUTH_MASK_ENABLED', False), \
             patch.object(config, 'EYES_MASK_ENABLED', False), \
             patch.object(config, 'EYEBROWS_MASK_ENABLED', False), \
             patch.object(config, 'POISSON_BLEND_ENABLED', False):
            factory.return_value.get.return_value = frame.copy()
            swapping.swap_face(source, target, frame)
            factory.return_value.get.assert_called_once_with(
                frame, target, source, paste_back=True)

    def test_supplied_analyser_is_used(self):
        analyser = Mock()
        analyser.get.return_value = []
        frame = np.zeros((16, 16, 3), np.uint8)
        result, count = swapping.detect_and_swap(None, frame, analyser)
        analyser.get.assert_called_once_with(frame)
        self.assertIs(result, frame)
        self.assertEqual(count, 0)

    def test_masks_trigger_lazy_landmarks(self):
        face = SimpleNamespace(landmark_2d_106=face_points())
        model = Mock()
        model.get.side_effect = lambda frame, *args, **kwargs: np.zeros_like(frame)
        with patch.object(swapping, 'get_face_swapper', return_value=model), \
             patch.object(face_analyser, 'ensure_face_landmarks') as landmarks, \
             patch.object(config, 'MOUTH_MASK_ENABLED', True):
            frame = np.full((128, 128, 3), 200, np.uint8)
            result = swapping.swap_face(object(), face, frame)
            landmarks.assert_called_once()
            self.assertGreater(result[85, 64, 0], 150)
            self.assertEqual(result[0, 0, 0], 0)

    def test_no_landmark_inference_without_masks(self):
        with patch.object(config, 'MOUTH_MASK_ENABLED', False), \
             patch.object(config, 'EYES_MASK_ENABLED', False), \
             patch.object(config, 'EYEBROWS_MASK_ENABLED', False), \
             patch.object(config, 'POISSON_BLEND_ENABLED', False), \
             patch.object(swapping, 'get_face_swapper') as factory, \
             patch.object(face_analyser, 'ensure_face_landmarks') as landmarks:
            factory.return_value.get.return_value = np.zeros((16, 16, 3), np.uint8)
            swapping.swap_face(object(), object(), np.zeros((16, 16, 3), np.uint8))
            landmarks.assert_not_called()

    def test_same_size_enhancer_output_is_applied(self):
        session = Mock()
        session.get_inputs.return_value = [SimpleNamespace(name='input', shape=[1, 3, 16, 16])]
        session.run.return_value = [np.ones((1, 3, 16, 16), np.float32)]
        frame = np.zeros((16, 16, 3), np.uint8)
        face = SimpleNamespace(kps=np.zeros((5, 2), np.float32))
        with patch.object(enhancer, 'get_face_enhancer', return_value=session), \
             patch.object(enhancer, '_align_face', return_value=(frame, np.eye(2, 3))), \
             patch.object(config, 'ENHANCE_WEIGHT', 0.6):
            result = enhancer.enhance_faces(frame, [face], (16, 16))
            self.assertAlmostEqual(int(result[8, 8, 0]), 102, delta=1)

    def test_landmark_model_is_reused(self):
        model = Mock()
        with patch.object(face_analyser, '_LANDMARK_MODEL', model):
            face = SimpleNamespace(landmark_2d_106=None)
            face_analyser.ensure_face_landmarks(None, face)
            model.get.assert_called_once_with(None, face)
            face.landmark_2d_106 = face_points()
            face_analyser.ensure_face_landmarks(None, face)
            self.assertEqual(model.get.call_count, 1)


class WorkerTests(unittest.TestCase):
    def test_enhancer_loading_is_deferred(self):
        from app.file_processing_thread import FileProcessingThread, MODE_SWAP_ENHANCE
        with patch.object(enhancer, 'get_face_enhancer') as load:
            worker = FileProcessingThread(None, 'input.jpg', mode=MODE_SWAP_ENHANCE)
            load.assert_not_called()
            worker._load_enhancer()
            load.assert_called_once()

    def test_video_resources_released_on_processing_failure(self):
        from app.file_processing_thread import FileProcessingThread
        cap = Mock()
        cap.isOpened.return_value = True
        cap.get.side_effect = [0, 30, 128, 128]
        cap.read.return_value = (True, np.zeros((128, 128, 3), np.uint8))
        writer = Mock()
        writer.isOpened.return_value = True
        worker = FileProcessingThread(None, 'input.mp4', '/tmp/output.mp4')
        with patch('app.file_processing_thread.cv2.VideoCapture', return_value=cap), \
             patch('app.file_processing_thread.cv2.VideoWriter', return_value=writer), \
             patch.object(worker, '_process_frame', side_effect=RuntimeError('inference failed')):
            with self.assertRaisesRegex(RuntimeError, 'inference failed'):
                worker.process_video(Mock())
        cap.release.assert_called_once()
        writer.release.assert_called_once()

    def test_unknown_video_frame_count(self):
        from app.file_processing_thread import FileProcessingThread
        frame = np.zeros((16, 16, 3), np.uint8)
        cap = Mock()
        cap.isOpened.return_value = True
        cap.get.side_effect = [0, 30, 16, 16]
        cap.read.side_effect = [(True, frame)] * 5 + [(False, None)]
        writer = Mock()
        writer.isOpened.return_value = True
        worker = FileProcessingThread(None, 'input.mp4', '/tmp/output.mp4')
        with patch('app.file_processing_thread.cv2.VideoCapture', return_value=cap), \
             patch('app.file_processing_thread.cv2.VideoWriter', return_value=writer), \
             patch.object(worker, '_process_frame', return_value=frame):
            worker.process_video(Mock())
        self.assertEqual(writer.write.call_count, 5)
        cap.release.assert_called_once()
        writer.release.assert_called_once()


if __name__ == '__main__':
    unittest.main()
