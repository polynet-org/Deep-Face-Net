"""
face_selector_widget.py
========================
Scans target video/image, shows detected faces, lets user assign a
different source image to each person. All pairs swap in one run.
"""

import cv2
import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QScrollArea, QFileDialog, QFrame, QMessageBox, QSizePolicy,
)
from PyQt6.QtCore import Qt, pyqtSignal, QThread
from PyQt6.QtGui import QPixmap, QImage


# ── Background scan thread ────────────────────────────────────────────────────

class _ScanThread(QThread):
    faces_found = pyqtSignal(list)
    status      = pyqtSignal(str)

    def __init__(self, target_path, max_frames=120, sample_every=10):
        super().__init__()
        self.target_path  = target_path
        self.max_frames   = max_frames
        self.sample_every = sample_every

    def run(self):
        from core.face_analyser import get_face_analyser
        self.status.emit("Scanning...")
        try:
            analyser = get_face_analyser(live_mode=False)
        except Exception as e:
            self.status.emit(f"Error: {e}")
            self.faces_found.emit([])
            return

        ext      = self.target_path.lower().rsplit(".", 1)[-1]
        is_video = ext in ("mp4", "avi", "mov", "mkv", "webm")
        results, seen = [], []

        def _try_add(face, frame):
            emb = face.normed_embedding
            if emb is None:
                return
            if any(float(np.dot(emb, s)) > 0.55 for s in seen):
                return
            b = face.bbox.astype(int)
            h, w = frame.shape[:2]
            p = int((b[2] - b[0]) * 0.2)
            x1, y1 = max(0, b[0]-p), max(0, b[1]-p)
            x2, y2 = min(w, b[2]+p), min(h, b[3]+p)
            seen.append(emb)
            results.append((frame[y1:y2, x1:x2].copy(), emb))

        if is_video:
            cap = cv2.VideoCapture(self.target_path)
            fi = sc = 0
            while sc < self.max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                if fi % self.sample_every == 0:
                    for face in (analyser.get(frame) or []):
                        _try_add(face, frame)
                    sc += 1
                fi += 1
            cap.release()
        else:
            frame = cv2.imread(self.target_path)
            if frame is not None:
                for face in (analyser.get(frame) or []):
                    _try_add(face, frame)

        self.status.emit(f"Found {len(results)} face(s)")
        self.faces_found.emit(results)


# ── Helper ────────────────────────────────────────────────────────────────────

def _to_pixmap(bgr: np.ndarray, size: int) -> QPixmap:
    rgb  = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb  = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
    h, w = rgb.shape[:2]
    img  = QImage(rgb.data, w, h, 3*w, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(img)


# ── One row per detected face ─────────────────────────────────────────────────

class _FacePairRow(QFrame):
    source_changed = pyqtSignal()

    def __init__(self, idx: int, target_crop: np.ndarray, working_dir: str = "", parent=None):
        super().__init__(parent)
        self.idx         = idx
        self.source_face = None
        self.enabled     = True
        self.working_dir = working_dir

        self.setMinimumHeight(140)
        self.setStyleSheet("""
            QFrame {
                background: #252525;
                border: 1px solid #3a3a3a;
                border-radius: 12px;
            }
        """)
        self._build(target_crop)

    def _build(self, crop):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # Header strip
        header_widget = QWidget()
        header_widget.setFixedHeight(26)
        header_widget.setStyleSheet("background: #1e1e1e; border-radius: 11px 11px 0 0; border: none;")
        header_layout = QHBoxLayout(header_widget)
        header_layout.setContentsMargins(12, 0, 12, 0)
        header_layout.setSpacing(0)

        person_lbl = QLabel(f"Person {self.idx + 1}")
        person_lbl.setStyleSheet("color: #aaa; font-size: 11px; font-weight: bold; border: none; background: transparent;")
        header_layout.addWidget(person_lbl)
        header_layout.addStretch()

        outer.addWidget(header_widget)

        # Main body row
        row = QHBoxLayout()
        row.setContentsMargins(14, 10, 14, 10)
        row.setSpacing(16)

        # ── Detected face column ──────────────────────────────────────────────
        det_col = QVBoxLayout()
        det_col.setSpacing(4)
        det_col.setAlignment(Qt.AlignmentFlag.AlignCenter)

        det_label = QLabel("Detected Face")
        det_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        det_label.setStyleSheet("color: #666; font-size: 10px; border: none; background: transparent;")
        det_col.addWidget(det_label)

        tgt = QLabel()
        tgt.setFixedSize(100, 100)
        tgt.setAlignment(Qt.AlignmentFlag.AlignCenter)
        tgt.setPixmap(_to_pixmap(crop, 98))
        tgt.setStyleSheet(
            "border: 2px solid #4a4a4a; border-radius: 8px; background: #1a1a1a;"
        )
        det_col.addWidget(tgt)
        row.addLayout(det_col)

        # ── flex gap ──────────────────────────────────────────────────────────
        row.addStretch(1)

        # ── Arrow ─────────────────────────────────────────────────────────────
        arr = QLabel("→")
        arr.setFixedWidth(32)
        arr.setAlignment(Qt.AlignmentFlag.AlignCenter)
        arr.setStyleSheet("color: #3a3a3a; font-size: 26px; border: none; background: transparent;")
        row.addWidget(arr)

        # ── flex gap ──────────────────────────────────────────────────────────
        row.addStretch(1)

        # ── Swap-with (source) column ─────────────────────────────────────────
        src_col = QVBoxLayout()
        src_col.setSpacing(4)
        src_col.setAlignment(Qt.AlignmentFlag.AlignCenter)

        src_label = QLabel("Swap With")
        src_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        src_label.setStyleSheet("color: #666; font-size: 10px; border: none; background: transparent;")
        src_col.addWidget(src_label)

        self._src_thumb = QLabel()
        self._src_thumb.setFixedSize(110, 110)
        self._src_thumb.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._src_thumb.setStyleSheet("""
            border: 2px dashed #3a3a3a;
            border-radius: 8px;
            color: #444;
            font-size: 10px;
            background: #1a1a1a;
        """)
        self._src_thumb.setText("None assigned")
        self._src_thumb.setWordWrap(True)
        src_col.addWidget(self._src_thumb)
        row.addLayout(src_col)

        # ── flex gap (wider before buttons) ───────────────────────────────────
        row.addStretch(2)

        # ── Action buttons column ─────────────────────────────────────────────
        action_col = QVBoxLayout()
        action_col.setSpacing(8)
        action_col.setAlignment(Qt.AlignmentFlag.AlignVCenter)

        self._btn_assign = QPushButton("+ Assign")
        self._btn_assign.setFixedHeight(36)
        self._btn_assign.setMinimumWidth(90)
        self._btn_assign.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_assign.setStyleSheet("""
            QPushButton {
                background: #1a3a1a;
                color: #4CAF50;
                border: 1px solid #4CAF50;
                border-radius: 6px;
                font-size: 11px;
                font-weight: bold;
            }
            QPushButton:hover { background: #2a5a2a; border-color: #66bb6a; color: #66bb6a; }
        """)
        self._btn_assign.clicked.connect(self._browse)
        action_col.addWidget(self._btn_assign)

        self._btn_clear = QPushButton("✕ Clear")
        self._btn_clear.setFixedHeight(30)
        self._btn_clear.setMinimumWidth(90)
        self._btn_clear.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_clear.setVisible(False)
        self._btn_clear.setStyleSheet("""
            QPushButton {
                background: #2a1a1a;
                color: #f44336;
                border: 1px solid #f44336;
                border-radius: 6px;
                font-size: 10px;
                font-weight: bold;
            }
            QPushButton:hover { background: #3a2a2a; }
        """)
        self._btn_clear.clicked.connect(self._clear)
        action_col.addWidget(self._btn_clear)

        self._skip = QPushButton("Skip")
        self._skip.setCheckable(True)
        self._skip.setFixedHeight(30)
        self._skip.setMinimumWidth(90)
        self._skip.setCursor(Qt.CursorShape.PointingHandCursor)
        self._skip.setStyleSheet(self._skip_off_style())
        self._skip.toggled.connect(self._on_skip)
        action_col.addWidget(self._skip)

        row.addLayout(action_col)

        outer.addLayout(row)

    def _browse(self):
        path, _ = QFileDialog.getOpenFileName(
            self, f"Source image for Person {self.idx+1}",
            self.working_dir or "",
            "Images (*.png *.jpg *.jpeg *.bmp *.webp)"
        )
        if not path:
            return
        try:
            img = cv2.imread(path)
            if img is None:
                return
            from core.face_analyser import get_face_analyser
            faces = get_face_analyser(live_mode=False).get(img)
            if not faces:
                QMessageBox.warning(self, "No Face", "No face detected in that image.")
                return
            self.source_face = faces[0]
            self._src_thumb.setPixmap(_to_pixmap(img, 98))
            self._src_thumb.setStyleSheet(
                "border: 2px solid #4CAF50; border-radius: 8px; background: #1a1a1a;"
            )
            self._btn_clear.setVisible(True)
            self._btn_assign.setText("Change")
            self._btn_assign.setStyleSheet("""
                QPushButton {
                    background: #1a3a1a; color: #4CAF50;
                    border: 1px solid #4CAF50; border-radius: 6px;
                    font-size: 10px; font-weight: bold;
                }
                QPushButton:hover { background: #2a5a2a; }
            """)
            self.setStyleSheet("""
                QFrame {
                    background: #1c2a1c;
                    border: 1px solid #4CAF50;
                    border-radius: 12px;
                }
            """)
            self.source_changed.emit()
        except Exception as e:
            print(f"Error loading source: {e}")

    def _clear(self):
        self.source_face = None
        self._src_thumb.clear()
        self._src_thumb.setText("None assigned")
        self._src_thumb.setStyleSheet("""
            border: 2px dashed #3a3a3a; border-radius: 8px;
            color: #444; font-size: 10px; background: #1a1a1a;
        """)
        self._btn_clear.setVisible(False)
        self._btn_assign.setText("+ Assign")
        self._btn_assign.setStyleSheet("""
            QPushButton {
                background: #1a3a1a; color: #4CAF50;
                border: 1px solid #4CAF50; border-radius: 6px;
                font-size: 11px; font-weight: bold;
            }
            QPushButton:hover { background: #2a5a2a; border-color: #66bb6a; color: #66bb6a; }
        """)
        self.setStyleSheet("""
            QFrame {
                background: #252525;
                border: 1px solid #3a3a3a;
                border-radius: 12px;
            }
        """)
        self.source_changed.emit()

    def _on_skip(self, checked):
        self.enabled = not checked
        self._skip.setStyleSheet(
            self._skip_on_style() if checked else self._skip_off_style()
        )
        self.setStyleSheet(f"""
            QFrame {{
                background: {'#1c1c1c' if checked else '#252525'};
                border: 1px solid {'#282828' if checked else '#3a3a3a'};
                border-radius: 12px;
            }}
        """)
        # Dim the content when skipped
        self.setEnabled(not checked)
        self._skip.setEnabled(True)  # Always keep skip button active
        self.source_changed.emit()

    @staticmethod
    def _skip_off_style():
        return """
            QPushButton {
                background: #2a0f0f; color: #f55; border: 1px solid #f44;
                border-radius: 6px; font-size: 10px; font-weight: bold;
            }
            QPushButton:hover { background: #3a1a1a; }
        """

    @staticmethod
    def _skip_on_style():
        return """
            QPushButton {
                background: #2a2a2a; color: #666; border: 1px solid #444;
                border-radius: 6px; font-size: 10px; font-weight: bold;
            }
            QPushButton:hover { background: #333; }
        """


# ── Public widget ─────────────────────────────────────────────────────────────

class FaceSelectorWidget(QWidget):
    mapping_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._face_data   = []
        self._rows        = []
        self._scan_thread = None
        self._working_dir = ""
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(6)

        # Status bar with icon
        status_row = QHBoxLayout()
        self._scan_spinner = QLabel("")
        self._scan_spinner.setFixedWidth(16)
        self._scan_spinner.setStyleSheet("color: #4CAF50; font-size: 12px; border: none;")
        status_row.addWidget(self._scan_spinner)

        self._status_lbl = QLabel("Load a target file to detect faces")
        self._status_lbl.setStyleSheet(
            "color: #666; font-size: 11px; font-style: italic; border: none;"
        )
        status_row.addWidget(self._status_lbl)
        status_row.addStretch()
        root.addLayout(status_row)

        # Scroll area — no max height, expands to fill available space
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._scroll.setMinimumHeight(200)
        self._scroll.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self._scroll.setStyleSheet("""
            QScrollArea {
                border: 1px solid #2e2e2e;
                border-radius: 10px;
                background: #1a1a1a;
            }
            QScrollBar:vertical {
                background: #1a1a1a; width: 6px; border-radius: 3px;
            }
            QScrollBar::handle:vertical {
                background: #3a3a3a; border-radius: 3px; min-height: 20px;
            }
            QScrollBar::handle:vertical:hover { background: #4a4a4a; }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; }
        """)

        self._rows_widget = QWidget()
        self._rows_widget.setStyleSheet("background: transparent;")
        self._rows_layout = QVBoxLayout(self._rows_widget)
        self._rows_layout.setSpacing(10)
        self._rows_layout.setContentsMargins(10, 10, 10, 10)
        self._rows_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self._scroll.setWidget(self._rows_widget)
        root.addWidget(self._scroll)

        # Placeholder inside scroll
        self._placeholder = QLabel("No faces detected yet")
        self._placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._placeholder.setStyleSheet(
            "color: #3a3a3a; font-size: 13px; padding: 30px;"
        )
        self._rows_layout.addWidget(self._placeholder)

    # ── Public API ────────────────────────────────────────────────────────────

    def scan(self, target_path: str, working_dir: str = ""):
        self._working_dir = working_dir
        self._clear_rows()
        self._face_data.clear()
        self._status_lbl.setText("Scanning for faces...")
        self._placeholder.setText("Scanning...")
        self._placeholder.show()

        if self._scan_thread and self._scan_thread.isRunning():
            self._scan_thread.terminate()

        self._scan_thread = _ScanThread(target_path)
        self._scan_thread.status.connect(self._on_status)
        self._scan_thread.faces_found.connect(self._on_faces_found)
        self._scan_thread.start()

    @property
    def face_pairs(self) -> list:
        """Returns [(source_face, target_normed_embedding), ...] for assigned enabled rows."""
        return [
            (row.source_face, self._face_data[row.idx][1])
            for row in self._rows
            if row.enabled and row.source_face is not None
        ]

    @property
    def selected_embeddings(self) -> list:
        return [self._face_data[row.idx][1] for row in self._rows if row.enabled]

    @property
    def has_any_source(self) -> bool:
        return any(r.source_face is not None and r.enabled for r in self._rows)

    # ── Slots ─────────────────────────────────────────────────────────────────

    def _on_status(self, msg):
        self._status_lbl.setText(msg)
        is_scanning = "scanning" in msg.lower()
        self._scan_spinner.setText("⟳" if is_scanning else "✓" if "found" in msg.lower() else "")

    def _on_faces_found(self, face_data):
        self._face_data = face_data
        self._clear_rows()
        if not face_data:
            self._placeholder.setText("No faces detected in this file.")
            self._placeholder.show()
            self._status_lbl.setText("No faces found")
            self._scan_spinner.setText("")
            return
        self._placeholder.hide()
        for idx, (crop, _emb) in enumerate(face_data):
            row = _FacePairRow(idx, crop, working_dir=self._working_dir)
            row.source_changed.connect(self.mapping_changed.emit)
            self._rows_layout.addWidget(row)
            self._rows.append(row)
        n = len(face_data)
        self._status_lbl.setText(
            f"Found {n} person(s) — click '+ Assign' to pick who to swap"
        )
        self._scan_spinner.setText("✓")
        self.mapping_changed.emit()

    def _clear_rows(self):
        self._rows.clear()
        while self._rows_layout.count():
            item = self._rows_layout.takeAt(0)
            w = item.widget()
            # Never delete the placeholder — just remove it from the layout
            if w and w is not self._placeholder:
                w.deleteLater()
        # Re-add placeholder (it was just removed, not deleted)
        self._rows_layout.addWidget(self._placeholder)
        self._placeholder.show()
