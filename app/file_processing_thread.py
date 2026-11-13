from PyQt6.QtCore import QThread, pyqtSignal
import sys
import cv2
from pathlib import Path
import time
from core.engine.face_swapper import detect_and_swap
from core.face_analyser import get_face_analyser

# Processing mode constants
MODE_SWAP          = "swap"
MODE_ENHANCE       = "enhance"
MODE_SWAP_ENHANCE  = "swap_enhance"


def _detect_and_swap_map(frame, face_analyser, face_map):
    """
    Multi-source swap: for each detected face in the frame, find the
    closest matching target embedding in face_map and swap it with the
    corresponding source face.

    face_map: list of (source_face, target_normed_embedding)
    Returns: (processed_frame, face_count)
    """
    import numpy as np
    from core.face_analyser import get_many_faces
    from core.engine.face_swapper import swap_face

    faces = get_many_faces(frame, live_mode=False)
    if not faces:
        return frame, 0

    MATCH_THRESHOLD = 0.40
    target_embeddings = [emb for _, emb in face_map]

    result = frame.copy()
    for detected_face in faces:
        if detected_face.normed_embedding is None:
            continue
        sims = [float(np.dot(detected_face.normed_embedding, tgt_emb))
                for tgt_emb in target_embeddings]
        best_idx = int(np.argmax(sims))
        if sims[best_idx] >= MATCH_THRESHOLD:
            src_face = face_map[best_idx][0]
            result = swap_face(src_face, detected_face, result)

    return result, len(faces)


class _StdoutCapture:
    """Context manager that intercepts stdout and forwards lines to a callback.
    Used to capture facexlib/GFPGAN auxiliary-model download progress messages.
    """
    def __init__(self, callback):
        self._callback = callback
        self._original = None
        self._buf = ""

    def __enter__(self):
        self._original = sys.stdout
        sys.stdout = self
        return self

    def __exit__(self, *_):
        sys.stdout = self._original

    def write(self, text):
        self._original.write(text)      # still show in terminal
        self._buf += text
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            line = line.strip()
            if line:
                self._callback(line)

    def flush(self):
        if self._original:
            self._original.flush()


class FileProcessingThread(QThread):
    """
    Background thread for processing video/image files.

    mode:
      'swap'         - face swap only (source face required)
      'enhance'      - GFPGAN enhancement only, no swap (source face not needed)
      'swap_enhance' - swap then GFPGAN (source face required)
    """
    progress_update     = pyqtSignal(int)
    status_update       = pyqtSignal(str)
    finished_processing = pyqtSignal(str)
    error_occurred      = pyqtSignal(str)

    def __init__(self, source_face, target_path, output_path=None, mode=MODE_SWAP,
                 target_embeddings=None, face_map=None):
        """
        face_map: list of (source_face, target_normed_embedding) pairs.
                  When provided, each target face in the video is matched to the
                  closest source face in the map and swapped accordingly.
                  Overrides source_face + target_embeddings when set.
        target_embeddings: legacy single-source filter (list of normed embeddings).
        """
        super().__init__()
        self.source_face       = source_face
        self.target_path       = Path(target_path)
        self.output_path       = output_path
        self.mode              = mode
        self.target_embeddings = target_embeddings
        self.face_map          = face_map   # list of (src_face, tgt_emb)
        self._is_running       = True

        self._needs_enhance = mode in (MODE_ENHANCE, MODE_SWAP_ENHANCE)
        self._needs_swap    = mode in (MODE_SWAP,    MODE_SWAP_ENHANCE)
        self._enhance_fn    = None

    def _load_enhancer(self):
        if self._needs_enhance:
            # Warm up GFPGAN - capture any facexlib auxiliary downloads and
            # relay them to the UI so the user knows something is happening.
            self.status_update.emit(
                "Loading GFPGAN enhancer… (first run may download auxiliary models)"
            )
            self.progress_update.emit(-1)   # -1 → indeterminate progress bar
            try:
                from core.engine.face_enhancer import enhance_faces, get_face_enhancer

                def _relay(line):
                    # Forward only meaningful lines (skip OpenCV noise)
                    low = line.lower()
                    if any(k in low for k in ("download", "loading", "error", "%", "mb", "kb")):
                        self.status_update.emit(f"↓ {line}")

                with _StdoutCapture(_relay):
                    get_face_enhancer()      # validates model + triggers facexlib DL

                self._enhance_fn = enhance_faces
                self.status_update.emit("GFPGAN ready ✓")
                self.progress_update.emit(0)  # reset to normal
            except FileNotFoundError as e:
                self.progress_update.emit(0)
                print(f"Warning: Enhancement disabled - {e}")
                self._needs_enhance = False
                if self.mode == MODE_ENHANCE:
                    raise

    def run(self):
        try:
            if not self.target_path.exists():
                raise FileNotFoundError(f"Target file not found: {self.target_path}")

            image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
            video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
            target_ext = self.target_path.suffix.lower()

            if not self.output_path:
                suffix_map = {
                    MODE_SWAP:         "_swapped",
                    MODE_ENHANCE:      "_enhanced",
                    MODE_SWAP_ENHANCE: "_swapped_enhanced",
                }
                suffix = suffix_map.get(self.mode, "_swapped")
                self.output_path = str(
                    self.target_path.parent
                    / f"{self.target_path.stem}{suffix}{self.target_path.suffix}"
                )

            self._load_enhancer()
            self.status_update.emit("Initializing face analyser...")
            face_analyser = get_face_analyser()

            if target_ext in image_extensions:
                self.process_image(face_analyser)
            elif target_ext in video_extensions:
                self.process_video(face_analyser)
            else:
                raise ValueError(f"Unsupported file format: {target_ext}")

        except Exception as e:
            self.error_occurred.emit(str(e))

    def _process_frame(self, frame, face_analyser, w, h):
        """Apply swap and/or enhance to a single frame. Returns processed frame."""
        if self._needs_swap:
            if self.face_map:
                # Multi-source mode: match each detected face to the closest source
                frame, _ = _detect_and_swap_map(frame, face_analyser, self.face_map)
            else:
                frame, _ = detect_and_swap(
                    self.source_face, frame, face_analyser,
                    target_embeddings=self.target_embeddings,
                )

        if self._needs_enhance and self._enhance_fn:
            from core.face_analyser import get_many_faces
            detect_faces = get_many_faces(frame)
            if detect_faces:
                frame = self._enhance_fn(frame, detect_faces, (w, h))

        return frame

    def process_image(self, face_analyser):
        self.status_update.emit("Processing image...")
        img = cv2.imread(str(self.target_path))
        if img is None:
            raise ValueError("Failed to load target image")

        h, w = img.shape[:2]
        img = self._process_frame(img, face_analyser, w, h)

        if not cv2.imwrite(str(self.output_path), img):
            raise IOError(f"Failed to save image: {self.output_path}")
        self.progress_update.emit(100)
        self.status_update.emit("Image processing complete")
        self.finished_processing.emit(str(self.output_path))

    def process_video(self, face_analyser):
        self.status_update.emit("Opening video...")
        cap = cv2.VideoCapture(str(self.target_path))
        if not cap.isOpened():
            raise ValueError("Failed to open target video")

        out = None
        try:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps    = cap.get(cv2.CAP_PROP_FPS)
            width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            out_path_obj = Path(self.output_path)
            if out_path_obj.suffix.lower() != ".mp4":
                out_path_obj = out_path_obj.with_suffix(".mp4")
                self.output_path = str(out_path_obj)

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(self.output_path, fourcc, fps, (width, height))

            if not out.isOpened():
                self.status_update.emit("Warning: 'mp4v' codec failed. Trying 'avc1'...")
                fourcc = cv2.VideoWriter_fourcc(*"avc1")
                out = cv2.VideoWriter(self.output_path, fourcc, fps, (width, height))
                if not out.isOpened():
                    raise ValueError(f"Failed to initialize video writer for {self.output_path}")

            mode_labels = {
                MODE_SWAP:         "Swapping",
                MODE_ENHANCE:      "Enhancing",
                MODE_SWAP_ENHANCE: "Swapping + Enhancing",
            }
            self.status_update.emit(f"{mode_labels.get(self.mode, 'Processing')} video frames...")

            frame_count = 0
            start_time  = time.time()

            while self._is_running:
                ret, frame = cap.read()
                if not ret:
                    break

                frame = self._process_frame(frame, face_analyser, width, height)
                out.write(frame)
                frame_count += 1

                if frame_count % 5 == 0:
                    progress = min(99, int(frame_count / total_frames * 100)) if total_frames > 0 else -1
                    self.progress_update.emit(progress)

                    if frame_count % 30 == 0:
                        elapsed = time.time() - start_time
                        fps_proc = frame_count / elapsed if elapsed > 0 else 0
                        self.status_update.emit(
                            f"{mode_labels.get(self.mode, 'Processing')}: {progress}% - {fps_proc:.1f} FPS"
                        )

        finally:
            cap.release()
            if out is not None:
                out.release()

        if self._is_running:
            self.progress_update.emit(100)
            self.status_update.emit("Processing complete")
            self.finished_processing.emit(str(self.output_path))
        else:
            self.status_update.emit("Processing cancelled")

    def stop(self):
        self._is_running = False
        self.wait()
