"""Video capture and processing thread for Qt application"""

import cv2
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal
from core.face_analyser import get_face_analyser
from core.engine.face_swapper import swap_face, detect_and_swap
from concurrent.futures import ThreadPoolExecutor
from core import config


class VideoThread(QThread):
    """Worker thread for video capture and face swapping"""

    frame_ready = pyqtSignal(np.ndarray)
    fps_update = pyqtSignal(float)
    face_count_update = pyqtSignal(int)
    error_occurred = pyqtSignal(str)

    def __init__(self, camera_index=0):
        super().__init__()
        self.camera_index = camera_index
        self.running = False
        self.swap_enabled = False
        self.mouth_mask_enabled = False
        self.source_face = None
        self.cap = None
        self.face_analyser = None

        self._executor = ThreadPoolExecutor(max_workers=1)
        self._pending_future = None
        self._last_result = None
        self._last_face_count = 0

    def set_source_face(self, source_face):
        """Set the source face for swapping"""
        self.source_face = source_face

    def load_enhancer(self):
        """Pre-load the selected enhancer model for live mode."""
        if config.ENHANCER_MODE == "gfpgan":
            from core.engine.face_enhancer import get_face_enhancer
            get_face_enhancer()
        elif config.ENHANCER_MODE == "gpen256":
            from core.engine.face_enhancer_gpen256 import get_enhancer_session
            get_enhancer_session()
        elif config.ENHANCER_MODE == "gpen512":
            from core.engine.face_enhancer_gpen512 import get_enhancer_session
            get_enhancer_session()

    def enable_swap(self, enabled):
        """Enable or disable face swapping"""
        self.swap_enabled = enabled

    def enable_mouth_mask(self, enabled):
        """Enable or disable mouth masking"""
        self.mouth_mask_enabled = enabled

    def run(self):
        """Main thread loop for video capture and processing"""
        try:
            # Initialize camera
            self.cap = cv2.VideoCapture(self.camera_index)
            if not self.cap.isOpened():
                self.error_occurred.emit("Failed to open camera")
                return

            # Set camera properties for better performance
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)

            # Initialize face analyser if available (graceful degradation)
            try:
                self.face_analyser = get_face_analyser(live_mode=True)
            except Exception:
                self.face_analyser = None

            self.running = True
            fps_counter = FPSCounter()

            while self.running:
                ret, frame = self.cap.read()
                if not ret:
                    self.error_occurred.emit("Failed to read frame from camera")
                    break

                try:
                    if self.swap_enabled and self.source_face is not None:
                        if self._pending_future is None or self._pending_future.done():
                            if (
                                self._pending_future is not None
                                and self._pending_future.done()
                            ):
                                completed = self._pending_future
                                self._pending_future = None
                                result = completed.result()
                                if result is not None:
                                    self._last_result, self._last_face_count = result

                            self._pending_future = self._executor.submit(
                                detect_and_swap,
                                self.source_face,
                                frame.copy(),
                                self.face_analyser,
                            )

                        if self._last_result is not None:
                            frame = self._last_result
                        self.face_count_update.emit(self._last_face_count)

                    else:
                        if self.face_analyser is not None:
                            faces = self.face_analyser.get(frame)
                            self.face_count_update.emit(len(faces))
                            for face in faces:
                                bbox = face.bbox.astype(int)
                                cv2.rectangle(
                                    frame,
                                    (bbox[0], bbox[1]),
                                    (bbox[2], bbox[3]),
                                    (0, 255, 0),
                                    2,
                                )

                except Exception as e:
                    import traceback

                    print(f"[Face Swap Error] {e}")
                    traceback.print_exc()

                # Enhancer
                if self.swap_enabled and config.ENHANCER_MODE != "none":
                    try:
                        from core.face_analyser import get_many_faces
                        enh_faces = get_many_faces(frame, live_mode=True)
                        if enh_faces:
                            h, w = frame.shape[:2]
                            if config.ENHANCER_MODE == "gfpgan":
                                from core.engine.face_enhancer import enhance_faces
                            elif config.ENHANCER_MODE == "gpen256":
                                from core.engine.face_enhancer_gpen256 import enhance_faces
                            elif config.ENHANCER_MODE == "gpen512":
                                from core.engine.face_enhancer_gpen512 import enhance_faces
                            frame = enhance_faces(frame, enh_faces, (w, h))
                    except Exception as e:
                        print(f"Error applying enhancer: {e}")

                # Mirror
                if getattr(config, "LIVE_MIRROR", False):
                    frame = cv2.flip(frame, 1)

                # FPS
                fps = fps_counter.update()

                self.frame_ready.emit(frame)

                if fps > 0:
                    self.fps_update.emit(fps)

        except Exception as e:
            self.error_occurred.emit(f"Video thread error: {str(e)}")
        finally:
            self.cleanup()

    def stop(self):
        """Stop the video thread"""
        self.running = False
        self.wait()

    def cleanup(self):
        """Release resources"""
        if self.cap is not None:
            self.cap.release()
        self._executor.shutdown(wait=True, cancel_futures=True)


class FPSCounter:
    """Simple FPS counter"""

    def __init__(self, avg_frames=30):
        self.avg_frames = avg_frames
        self.frame_times = []
        self.last_time = cv2.getTickCount()

    def update(self):
        """Update and return current FPS"""
        current_time = cv2.getTickCount()
        time_diff = (current_time - self.last_time) / cv2.getTickFrequency()
        self.last_time = current_time

        self.frame_times.append(time_diff)
        if len(self.frame_times) > self.avg_frames:
            self.frame_times.pop(0)

        if len(self.frame_times) > 0:
            avg_time = sum(self.frame_times) / len(self.frame_times)
            return 1.0 / avg_time if avg_time > 0 else 0
        return 0
