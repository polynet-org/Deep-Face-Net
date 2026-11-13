"""
Qt Deepfake Application - Real-time Face Swapping
A professional PyQt6 application for live deepfake face swapping
"""

import sys
import os
import cv2
import numpy as np
import json
import zipfile
from pathlib import Path

# Try to import pyvirtualcam (optional dependency)
try:
    import pyvirtualcam

    VIRTUAL_CAM_AVAILABLE = True
except ImportError:
    VIRTUAL_CAM_AVAILABLE = False
    print("pyvirtualcam not available - virtual camera feature disabled")
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QFileDialog,
    QMessageBox,
    QGroupBox,
    QStatusBar,
    QFrame,
    QCheckBox,
    QRadioButton,
    QButtonGroup,
    QTabWidget,
    QProgressBar,
    QDialog,
    QLineEdit,
    QComboBox,
    QScrollArea,
    QSlider,
    QSizePolicy,
)
from PyQt6.QtCore import (
    Qt,
    QTimer,
    QUrl,
    QThread,
    pyqtSignal,
    PYQT_VERSION_STR,
    qVersion,
)
from PyQt6.QtGui import QImage, QPixmap, QFont, QDesktopServices, QIcon, QColor

from app.video_thread import VideoThread
from app.drag_drop_widget import DragDropWidget
from app.file_processing_thread import FileProcessingThread
from app.face_selector_widget import FaceSelectorWidget
from core.face_analyser import get_face_analyser
from download_models import MODELS, check_model_status, get_model_path, format_size


class ModelDownloadThread(QThread):
    """Background thread for downloading a model file"""

    progress_update = pyqtSignal(int)  # percentage 0-100
    download_complete = pyqtSignal(str)  # model_name
    download_error = pyqtSignal(str, str)  # model_name, error_msg
    status_update = pyqtSignal(str)  # status text

    def __init__(self, model_name, model_info):
        super().__init__()
        self.model_name = model_name
        self.model_info = model_info
        self._cancelled = False

    def run(self):
        try:
            import requests

            url = self.model_info["url"]
            dest_path = get_model_path(self.model_name)

            if self.model_info["location"] == "insightface":
                # Download zip and extract
                dest_path.mkdir(parents=True, exist_ok=True)
                zip_path = dest_path.parent / f"{self.model_name}.zip"
                self._download_file(requests, url, zip_path)
                if self._cancelled:
                    return
                self.status_update.emit("Extracting...")
                with zipfile.ZipFile(zip_path, "r") as z:
                    z.extractall(dest_path)
                zip_path.unlink(missing_ok=True)
            else:
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                self._download_file(requests, url, dest_path)

            if not self._cancelled:
                self.download_complete.emit(self.model_name)
        except Exception as e:
            self.download_error.emit(self.model_name, str(e))

    def _download_file(self, requests, url, dest_path):
        response = requests.get(url, stream=True)
        response.raise_for_status()
        total_size = int(response.headers.get("content-length", 0))
        downloaded = 0
        block_size = 8192

        with open(dest_path, "wb") as f:
            for chunk in response.iter_content(block_size):
                if self._cancelled:
                    return
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    self.progress_update.emit(int(downloaded * 100 / total_size))

    def cancel(self):
        self._cancelled = True


class CameraDetectionThread(QThread):
    """Background thread for detecting available cameras"""

    cameras_detected = pyqtSignal(list)  # Emits list of camera indices

    def __init__(self, max_cameras=6):
        super().__init__()
        self.max_cameras = max_cameras

    def run(self):
        """Detect available cameras in background"""
        available = []
        # Test camera indices 0-5 (covers most cases, faster than 0-10)
        for i in range(self.max_cameras):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                # Try to read a frame to verify camera works
                # Set a short timeout to avoid long waits
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                ret, _ = cap.read()
                if ret:
                    available.append(i)
                cap.release()

        self.cameras_detected.emit(available)


class SettingsDialog(QDialog):
    """Settings dialog for configuring app preferences"""

    def __init__(self, current_working_dir, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setModal(True)
        self.setMinimumWidth(500)

        self.working_dir = current_working_dir

        layout = QVBoxLayout()

        # Working Directory Section
        dir_group = QGroupBox("Working Directory")
        dir_layout = QVBoxLayout()

        info_label = QLabel("All file browsers will start from this directory:")
        info_label.setStyleSheet("color: #aaa; font-size: 11px;")
        dir_layout.addWidget(info_label)

        # Directory display and browse
        dir_row = QHBoxLayout()
        self.dir_input = QLineEdit(self.working_dir)
        self.dir_input.setReadOnly(True)
        dir_row.addWidget(self.dir_input)

        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self.browse_directory)
        dir_row.addWidget(browse_btn)

        dir_layout.addLayout(dir_row)
        dir_group.setLayout(dir_layout)
        layout.addWidget(dir_group)

        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        save_btn = QPushButton("Save")
        save_btn.clicked.connect(self.accept)
        save_btn.setStyleSheet("background-color: #4CAF50; min-width: 80px;")
        button_layout.addWidget(save_btn)

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        cancel_btn.setStyleSheet("background-color: #555; min-width: 80px;")
        button_layout.addWidget(cancel_btn)

        layout.addLayout(button_layout)
        self.setLayout(layout)

        # Apply dark theme
        self.setStyleSheet("""
            QDialog {
                background-color: #1e1e1e;
            }
            QGroupBox {
                font-weight: bold;
                color: #ffffff;
                border: 2px solid #444;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
                background-color: #2a2a2a;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
            QLabel { color: #ffffff; }
            QLineEdit {
                background-color: #2a2a2a;
                color: #ffffff;
                border: 1px solid #444;
                border-radius: 4px;
                padding: 5px;
            }
        """)

    def browse_directory(self):
        """Open directory browser"""
        directory = QFileDialog.getExistingDirectory(
            self, "Select Working Directory", self.working_dir
        )
        if directory:
            self.working_dir = directory
            self.dir_input.setText(directory)

    def get_working_dir(self):
        """Return the selected working directory"""
        return self.working_dir


class DeepfakeApp(QMainWindow):
    """Main application window for deepfake face swapping"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Deep Face Net - Advanced Face Swapping")
        self.setGeometry(100, 100, 1200, 800)

        # Set app icon
        icon_path = Path(__file__).parent.parent / "assets" / "icon.png"
        if icon_path.exists():
            self.setWindowIcon(QIcon(str(icon_path)))

        # Application state
        self.source_image = None
        self.source_face = None
        self.video_thread = None
        self.is_capturing = False
        self.selected_camera_index = 0  # Default camera
        self.available_cameras = []
        self.virtual_cam = None  # Virtual camera instance
        self.virtual_cam_enabled = False  # Virtual camera state

        # Offline processing state
        self.offline_source_image = None
        self.offline_source_face = None
        self.target_file_path = None
        self.processing_thread = None
        self.last_output_path = None

        # Model download state
        self._active_downloads = {}  # model_name -> ModelDownloadThread
        self._model_cards = {}  # model_name -> dict of widgets

        # Settings
        self.settings_file = Path.home() / ".deepfacenet_settings.json"
        self.working_dir = self.load_settings()

        # Initialize UI
        self.init_ui()

        # Apply modern styling
        self.apply_styles()

    def init_ui(self):
        """Initialize the user interface"""
        # Central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)

        # Create Tab Widget
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)

        # Tab 1: Live Camera
        self.live_tab = QWidget()
        self.setup_live_tab()
        self.tabs.addTab(self.live_tab, "Live Camera")

        # Tab 2: Offline Processing
        self.offline_tab = QWidget()
        self.setup_offline_tab()
        self.tabs.addTab(self.offline_tab, "Offline Processing")

        # Tab 3: Models
        self.models_tab = QWidget()
        self.setup_models_tab()
        self.models_tab_index = self.tabs.addTab(self.models_tab, "Models")

        # Tab 4: About
        self.about_tab = QWidget()
        self.setup_about_tab()
        self.tabs.addTab(self.about_tab, "About")

        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_label = QLabel("Ready")
        self.fps_label = QLabel("FPS: 0")
        self.face_count_label = QLabel("Faces: 0")
        self.status_bar.addWidget(self.status_label, stretch=1)
        self.status_bar.addPermanentWidget(self.face_count_label)
        self.status_bar.addPermanentWidget(self.fps_label)

        # Check required models after status bar is initialized
        self.check_models_on_startup()

        # Add Settings button to status bar
        settings_btn = QPushButton("⚙ Settings")
        settings_btn.clicked.connect(self.open_settings)
        settings_btn.setStyleSheet("""
            QPushButton {
                background-color: #555;
                padding: 5px 15px;
                margin-left: 10px;
            }
            QPushButton:hover { background-color: #666; }
        """)
        self.status_bar.addPermanentWidget(settings_btn)

        # Detect and populate available cameras
        self.detect_and_populate_cameras()

    def setup_live_tab(self):
        """Setup the Live Camera tab"""
        layout = QHBoxLayout(self.live_tab)
        layout.setSpacing(15)
        layout.setContentsMargins(15, 15, 15, 15)

        # Left panel - Video display
        left_panel = self.create_video_panel()
        layout.addWidget(left_panel, stretch=3)

        # Right panel - Controls
        right_panel = self.create_control_panel()
        layout.addWidget(right_panel, stretch=1)

    def setup_offline_tab(self):
        """Setup the Offline Processing tab"""
        layout = QHBoxLayout(self.offline_tab)
        layout.setSpacing(16)
        layout.setContentsMargins(16, 16, 16, 16)

        # ── Left column: scrollable inputs ───────────────────────────────────
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        left_scroll.setStyleSheet(
            "QScrollArea { border: none; background: transparent; }"
        )

        left_container = QWidget()
        left_container.setStyleSheet("background: transparent;")
        input_layout = QVBoxLayout(left_container)
        input_layout.setSpacing(12)
        input_layout.setContentsMargins(0, 0, 6, 0)

        # ── Section style helper ──────────────────────────────────────────────
        _GRP = """
            QGroupBox {
                font-weight: bold; font-size: 12px; color: #fff;
                border: 1px solid #3a3a3a; border-radius: 10px;
                margin-top: 10px; padding-top: 14px;
                background-color: #252525;
            }
            QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 6px; }
        """

        # ── 1. Target File ────────────────────────────────────────────────────
        target_group = QGroupBox("1. Select Target Video / Image")
        target_group.setStyleSheet(_GRP)
        target_inner = QVBoxLayout()
        target_inner.setContentsMargins(10, 6, 10, 10)
        target_inner.setSpacing(8)

        # Clear button row
        target_header = QHBoxLayout()
        target_header.addStretch()
        self.btn_clear_target = QPushButton("✕ Clear")
        self.btn_clear_target.setFixedHeight(22)
        self.btn_clear_target.setEnabled(False)
        self.btn_clear_target.clicked.connect(self.clear_target_file)
        self.btn_clear_target.setStyleSheet("""
            QPushButton { background:#444; color:#fff; border-radius:4px; font-size:11px; padding:0 8px; border:none; }
            QPushButton:hover { background:#f44336; }
            QPushButton:disabled { background:#2a2a2a; color:#555; }
        """)
        target_header.addWidget(self.btn_clear_target)
        target_inner.addLayout(target_header)

        self.drop_widget = DragDropWidget()
        self.drop_widget.setMinimumHeight(100)
        self.drop_widget.file_dropped.connect(self.handle_file_drop)
        target_inner.addWidget(self.drop_widget)

        btn_browse_target = QPushButton("Browse Target File...")
        btn_browse_target.setFixedHeight(34)
        btn_browse_target.clicked.connect(self.browse_target_file)
        btn_browse_target.setStyleSheet("""
            QPushButton { background:#333; color:#ccc; border:1px solid #555; border-radius:6px; font-size:12px; }
            QPushButton:hover { background:#444; color:#fff; }
        """)
        target_inner.addWidget(btn_browse_target)
        target_group.setLayout(target_inner)
        input_layout.addWidget(target_group)

        # ── 2. Face Source (mode toggle) ─────────────────────────────────────
        face_source_group = QGroupBox("2. Face Source")
        face_source_group.setStyleSheet(_GRP)
        fs_outer = QVBoxLayout()
        fs_outer.setContentsMargins(10, 8, 10, 12)
        fs_outer.setSpacing(10)

        # ── Mode toggle bar ───────────────────────────────────────────────────
        _toggle_base = """
            QRadioButton {{
                color: {fg}; font-size: 12px; font-weight: bold;
                padding: 7px 18px;
                border: 1px solid {bd}; border-radius: {br};
                background: {bg};
            }}
            QRadioButton:checked {{
                color: {cfg}; background: {cbg}; border-color: {cbd};
            }}
            QRadioButton::indicator {{ width: 0; height: 0; }}
        """
        self._src_mode_group = QButtonGroup(self)

        self.radio_single_face = QRadioButton("⬡  Single Face Swap")
        self.radio_single_face.setChecked(True)
        self.radio_single_face.setToolTip(
            "Swap ONE source face onto every face detected in the target."
        )
        self.radio_single_face.setStyleSheet(
            _toggle_base.format(
                fg="#aaa",
                bg="#1e1e1e",
                bd="#3a3a3a",
                br="8px 0 0 8px",
                cfg="#fff",
                cbg="#1b3a1b",
                cbd="#4CAF50",
            )
        )
        self._src_mode_group.addButton(self.radio_single_face, 0)

        self.radio_multi_face = QRadioButton("⬡  Multi-Face Mapping")
        self.radio_multi_face.setToolTip(
            "Assign a DIFFERENT source face to each detected person."
        )
        self.radio_multi_face.setStyleSheet(
            _toggle_base.format(
                fg="#aaa",
                bg="#1e1e1e",
                bd="#3a3a3a",
                br="0 8px 8px 0",
                cfg="#fff",
                cbg="#1a2a3a",
                cbd="#2196F3",
            )
        )
        self._src_mode_group.addButton(self.radio_multi_face, 1)

        toggle_row = QHBoxLayout()
        toggle_row.setSpacing(0)
        toggle_row.addWidget(self.radio_single_face)
        toggle_row.addWidget(self.radio_multi_face)
        toggle_row.addStretch()
        fs_outer.addLayout(toggle_row)

        # ── Single-Face Panel ─────────────────────────────────────────────────
        self.single_face_panel = QWidget()
        self.single_face_panel.setStyleSheet("background: transparent;")
        self.single_face_panel.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        sf_layout = QVBoxLayout(self.single_face_panel)
        sf_layout.setContentsMargins(0, 4, 0, 4)
        sf_layout.setSpacing(8)

        sf_hint = QLabel(
            "Select one source face image — it will be swapped onto every "
            "face detected in the target video / image."
        )
        sf_hint.setWordWrap(True)
        sf_hint.setStyleSheet(
            "color:#666; font-size:10px; font-style:italic; border:none;"
        )
        sf_layout.addWidget(sf_hint)

        # ── Body: [preview 150x150] [status + browse + clear] ────────────────
        sf_body = QHBoxLayout()
        sf_body.setSpacing(16)
        sf_body.setAlignment(Qt.AlignmentFlag.AlignTop)

        # Preview thumbnail — fixed size, left side
        self.offline_source_preview = QLabel()
        self.offline_source_preview.setFixedSize(150, 150)
        self.offline_source_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.offline_source_preview.setStyleSheet(
            "background:#1a1a1a; border:2px dashed #3a3a3a;"
            "border-radius:10px; color:#444; font-size:11px;"
        )
        self.offline_source_preview.setText("No face\nselected")
        sf_body.addWidget(self.offline_source_preview, 0, Qt.AlignmentFlag.AlignTop)

        # Actions: status label + buttons, right side
        sf_actions = QVBoxLayout()
        sf_actions.setSpacing(8)
        sf_actions.setContentsMargins(0, 4, 0, 0)

        self._single_src_status = QLabel("No source loaded")
        self._single_src_status.setStyleSheet(
            "color:#555; font-size:11px; border:none; background:transparent;"
        )
        self._single_src_status.setWordWrap(True)
        sf_actions.addWidget(self._single_src_status)

        btn_browse_single = QPushButton("Browse Source Image…")
        btn_browse_single.setFixedHeight(38)
        btn_browse_single.setMaximumWidth(260)
        btn_browse_single.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_browse_single.clicked.connect(self.select_offline_source_image)
        btn_browse_single.setStyleSheet("""
            QPushButton {
                background: #1b3a1b; color: #4CAF50;
                border: 1px solid #4CAF50; border-radius: 7px;
                font-size: 12px; font-weight: bold;
            }
            QPushButton:hover { background: #2a4a2a; border-color: #66bb6a; color: #66bb6a; }
        """)
        sf_actions.addWidget(btn_browse_single)

        self.btn_clear_source = QPushButton("✕  Clear Source")
        self.btn_clear_source.setFixedHeight(30)
        self.btn_clear_source.setMaximumWidth(260)
        self.btn_clear_source.setEnabled(False)
        self.btn_clear_source.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_clear_source.clicked.connect(self.clear_offline_source)
        self.btn_clear_source.setStyleSheet("""
            QPushButton {
                background: #2a1a1a; color: #f44336;
                border: 1px solid #f44336; border-radius: 7px; font-size: 11px;
            }
            QPushButton:hover { background: #3a2a2a; }
            QPushButton:disabled {
                background: #1e1e1e; color: #3a3a3a; border-color: #2a2a2a;
            }
        """)
        sf_actions.addWidget(self.btn_clear_source)
        sf_actions.addStretch()

        sf_body.addLayout(sf_actions)
        sf_layout.addLayout(sf_body)
        fs_outer.addWidget(self.single_face_panel)

        # ── Multi-Face Panel ──────────────────────────────────────────────────
        self.multi_face_panel = QWidget()
        self.multi_face_panel.setStyleSheet("background: transparent;")
        mf_layout = QVBoxLayout(self.multi_face_panel)
        mf_layout.setContentsMargins(0, 0, 0, 0)
        mf_layout.setSpacing(4)

        mf_hint = QLabel(
            "Each detected person from the target appears below. "
            "Click '+ Assign' to pick which source face to swap them with."
        )
        mf_hint.setWordWrap(True)
        mf_hint.setStyleSheet(
            "color:#666; font-size:10px; font-style:italic; border:none;"
        )
        mf_layout.addWidget(mf_hint)

        self.face_selector = FaceSelectorWidget()
        self.face_selector.mapping_changed.connect(self.check_offline_readiness)
        self.face_selector.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        mf_layout.addWidget(self.face_selector)

        fs_outer.addWidget(self.multi_face_panel, stretch=1)
        self.multi_face_panel.setVisible(False)

        # Connect mode toggle
        self.radio_single_face.toggled.connect(self._on_face_source_mode_changed)

        face_source_group.setLayout(fs_outer)
        face_source_group.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        input_layout.addWidget(face_source_group, stretch=1)

        left_scroll.setWidget(left_container)
        layout.addWidget(left_scroll, stretch=3)

        # ── Right column: process panel ───────────────────────────────────────
        process_layout = QVBoxLayout()
        process_layout.setSpacing(12)

        # ── Processing Mode (section 3 of right side) ─────────────────────────
        from core.config import ENHANCER_MODEL, SWAPPER_MODEL, ANALYSIS_MODEL

        _gfpgan_ok = ENHANCER_MODEL.exists()
        _swapper_ok = SWAPPER_MODEL.exists()
        _analyser_ok, _, _ = check_model_status(ANALYSIS_MODEL)

        mode_group_box = QGroupBox("3. Processing Mode")
        mode_group_box.setStyleSheet(_GRP)
        mode_inner = QVBoxLayout()
        mode_inner.setSpacing(4)
        mode_inner.setContentsMargins(12, 8, 12, 12)

        radio_style = "QRadioButton {{ color: {c}; font-size: 12px; padding: 3px 0; }}"

        self._mode_btn_group = QButtonGroup(self)

        self.radio_swap = QRadioButton("Swap Only")
        self.radio_swap.setChecked(True)
        self.radio_swap.setToolTip("Replace faces in target with your source face.")
        self.radio_swap.setStyleSheet(radio_style.format(c="#ccc"))
        self.radio_swap.toggled.connect(self.check_offline_readiness)
        self._mode_btn_group.addButton(self.radio_swap)

        self.radio_enhance = QRadioButton("Enhance Only  (no swap)")
        self.radio_enhance.setEnabled(_gfpgan_ok)
        self.radio_enhance.setToolTip(
            "Sharpen faces in target using GFPGAN - no source face needed."
            if _gfpgan_ok
            else "Download GFPGANv1.4.pth from the Models tab first."
        )
        self.radio_enhance.setStyleSheet(
            radio_style.format(c="#ccc" if _gfpgan_ok else "#555")
        )
        self.radio_enhance.toggled.connect(self.check_offline_readiness)
        self._mode_btn_group.addButton(self.radio_enhance)

        self.radio_swap_enhance = QRadioButton("Swap + Enhance")
        self.radio_swap_enhance.setEnabled(_gfpgan_ok)
        self.radio_swap_enhance.setToolTip(
            "Swap faces, then run GFPGAN for sharper results."
            if _gfpgan_ok
            else "Download GFPGANv1.4.pth from the Models tab first."
        )
        self.radio_swap_enhance.setStyleSheet(
            radio_style.format(c="#ccc" if _gfpgan_ok else "#555")
        )
        self.radio_swap_enhance.toggled.connect(self.check_offline_readiness)
        self._mode_btn_group.addButton(self.radio_swap_enhance)

        mode_inner.addWidget(self.radio_swap)
        mode_inner.addWidget(self.radio_enhance)
        mode_inner.addWidget(self.radio_swap_enhance)

        # ── Model status pills ─────────────────────────────────────────────────
        pills_row = QHBoxLayout()
        pills_row.setSpacing(6)

        def _make_pill(label, ok):
            p = QLabel(f"{'✓' if ok else '✗'}  {label}")
            p.setStyleSheet(
                f"background: {'#1b3a1b' if ok else '#3a1b1b'}; "
                f"color: {'#4CAF50' if ok else '#f44336'}; "
                "border-radius: 4px; padding: 2px 8px; font-size: 11px; font-weight: bold;"
            )
            return p

        pills_row.addWidget(_make_pill("Detect", _analyser_ok))
        pills_row.addWidget(_make_pill("Swap model", _swapper_ok))
        pills_row.addWidget(_make_pill("GFPGAN", _gfpgan_ok))
        pills_row.addStretch()
        mode_inner.addSpacing(6)
        mode_inner.addLayout(pills_row)

        if not _gfpgan_ok:
            missing_lbl = QLabel("GFPGAN not downloaded — go to Models tab")
            missing_lbl.setStyleSheet(
                "color: #888; font-size: 10px; padding-top: 2px;"
                "border: none; background: transparent;"
            )
            mode_inner.addWidget(missing_lbl)

        mode_group_box.setLayout(mode_inner)
        process_layout.addWidget(mode_group_box)

        # ── 4. Process (status + progress + actions) ───────────────────────────
        status_group = QGroupBox("4. Process")
        status_group.setStyleSheet(_GRP)
        status_inner = QVBoxLayout()
        status_inner.setSpacing(8)
        status_inner.setContentsMargins(12, 8, 12, 12)

        self.process_info_label = QLabel(
            "Select a source face and a target file to begin."
        )
        self.process_info_label.setWordWrap(True)
        self.process_info_label.setStyleSheet(
            "color: #888; font-style: italic; font-size: 11px;"
        )
        status_inner.addWidget(self.process_info_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFixedHeight(22)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #3a3a3a;
                border-radius: 6px;
                text-align: center;
                background: #1e1e1e;
                color: #fff;
                font-size: 11px;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #2e7d32, stop:1 #4CAF50);
                border-radius: 5px;
            }
        """)
        status_inner.addWidget(self.progress_bar)

        # Download status indicator
        self.download_status_label = QLabel()
        self.download_status_label.setWordWrap(True)
        self.download_status_label.setStyleSheet(
            "color: #FF9800; font-size: 10px; padding: 4px 6px;"
            "background: #1f1500; border-radius: 4px; border: 1px solid #3a2800;"
        )
        self.download_status_label.setVisible(False)
        status_inner.addWidget(self.download_status_label)

        # Start + Cancel button row
        proc_btn_row = QHBoxLayout()
        proc_btn_row.setSpacing(8)

        self.btn_start_process = QPushButton("▶  Start Processing")
        self.btn_start_process.setEnabled(False)
        self.btn_start_process.setMinimumHeight(46)
        self.btn_start_process.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #1565C0, stop:1 #2196F3);
                color: white;
                font-size: 15px;
                font-weight: bold;
                border: none;
                border-radius: 8px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #1976D2, stop:1 #42A5F5);
            }
            QPushButton:disabled { background: #333; color: #666; }
        """)
        self.btn_start_process.clicked.connect(self.start_offline_processing)
        proc_btn_row.addWidget(self.btn_start_process, stretch=1)

        self.btn_cancel_process = QPushButton("✕")
        self.btn_cancel_process.setFixedSize(46, 46)
        self.btn_cancel_process.setEnabled(False)
        self.btn_cancel_process.setToolTip("Cancel processing")
        self.btn_cancel_process.setStyleSheet("""
            QPushButton {
                background: #3a1a1a; color: #f44336;
                border: 1px solid #f44336; border-radius: 8px;
                font-size: 16px; font-weight: bold;
            }
            QPushButton:hover { background: #4a2a2a; }
            QPushButton:disabled { background: #222; color: #444; border-color: #333; }
        """)
        self.btn_cancel_process.clicked.connect(self.cancel_offline_processing)
        proc_btn_row.addWidget(self.btn_cancel_process)

        status_inner.addLayout(proc_btn_row)

        # Result Actions
        actions_layout = QHBoxLayout()
        actions_layout.setSpacing(8)

        self.btn_open_file = QPushButton("Open File")
        self.btn_open_file.setEnabled(False)
        self.btn_open_file.setFixedHeight(34)
        self.btn_open_file.clicked.connect(self.open_output_file)
        self.btn_open_file.setStyleSheet("""
            QPushButton { background: #1a2a3a; color: #58a6ff; border: 1px solid #2a4a6a;
                          border-radius: 6px; font-size: 12px; }
            QPushButton:hover { background: #1e3a5a; }
            QPushButton:disabled { background: #222; color: #444; border-color: #333; }
        """)

        self.btn_open_folder = QPushButton("Show in Folder")
        self.btn_open_folder.setEnabled(False)
        self.btn_open_folder.setFixedHeight(34)
        self.btn_open_folder.clicked.connect(self.open_output_folder)
        self.btn_open_folder.setStyleSheet("""
            QPushButton { background: #2a1f00; color: #FF9800; border: 1px solid #5a3a00;
                          border-radius: 6px; font-size: 12px; }
            QPushButton:hover { background: #3a2900; }
            QPushButton:disabled { background: #222; color: #444; border-color: #333; }
        """)

        actions_layout.addWidget(self.btn_open_file)
        actions_layout.addWidget(self.btn_open_folder)
        status_inner.addLayout(actions_layout)

        # Result Preview (Hidden by default)
        self.result_preview_label = QLabel()
        self.result_preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.result_preview_label.setVisible(False)
        self.result_preview_label.setStyleSheet(
            "border: 2px solid #4caf50; border-radius: 8px;"
        )
        status_inner.addWidget(self.result_preview_label)

        status_inner.addStretch()
        status_group.setLayout(status_inner)
        process_layout.addWidget(status_group)

        layout.addLayout(process_layout, stretch=1)

    # --- Models Tab Methods ---

    def setup_models_tab(self):
        """Setup the Models download tab"""
        layout = QVBoxLayout(self.models_tab)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)

        # Header
        header = QLabel("Model Manager")
        header.setStyleSheet(
            "font-size: 18px; font-weight: bold; color: #fff; margin-bottom: 5px;"
        )
        layout.addWidget(header)

        subtitle = QLabel(
            "Download and manage the AI models required for face swapping."
        )
        subtitle.setStyleSheet("color: #aaa; font-size: 12px; margin-bottom: 10px;")
        layout.addWidget(subtitle)

        # Scroll area for model cards
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")

        cards_widget = QWidget()
        cards_layout = QVBoxLayout(cards_widget)
        cards_layout.setSpacing(12)

        # Show all models
        for model_name, model_info in MODELS.items():
            card = self._create_model_card(model_name, model_info)
            cards_layout.addWidget(card)

        cards_layout.addStretch()
        scroll.setWidget(cards_widget)
        layout.addWidget(scroll)

        # Refresh button
        refresh_btn = QPushButton("Refresh Status")
        refresh_btn.setFixedHeight(36)
        refresh_btn.setStyleSheet("background-color: #555; max-width: 200px;")
        refresh_btn.clicked.connect(self.refresh_model_status)
        layout.addWidget(refresh_btn, alignment=Qt.AlignmentFlag.AlignLeft)

    def _create_model_card(self, model_name, model_info):
        """Create a single model card widget"""
        is_downloaded, path, current_size = check_model_status(model_name)
        expected_size = model_info.get("size", 0)
        is_required = model_info.get("required", False)

        card = QFrame()
        card.setStyleSheet("""
            QFrame {
                background-color: #2a2a2a;
                border: 2px solid #444;
                border-radius: 8px;
                padding: 4px;
            }
        """)
        card_layout = QVBoxLayout(card)
        card_layout.setSpacing(2)
        card_layout.setContentsMargins(10, 8, 10, 8)

        # Row 1: Name + status icon + required badge
        top_row = QHBoxLayout()

        if is_downloaded:
            status_icon = QLabel("[OK]")
            status_icon.setStyleSheet(
                "font-size: 12px; font-weight: bold; color: #4CAF50; border: none;"
            )
        else:
            status_icon = QLabel("[!!]")
            status_icon.setStyleSheet(
                "font-size: 12px; font-weight: bold; color: #FF9800; border: none;"
            )
        status_icon.setFixedWidth(30)
        top_row.addWidget(status_icon)

        name_label = QLabel(model_name)
        name_label.setStyleSheet(
            "font-size: 14px; font-weight: bold; color: #fff; border: none;"
        )
        top_row.addWidget(name_label)

        if is_required:
            req_badge = QLabel("REQUIRED")
            req_badge.setStyleSheet("""
                background-color: #f44336;
                color: white;
                font-size: 10px;
                font-weight: bold;
                padding: 2px 8px;
                border-radius: 4px;
                border: none;
            """)
            req_badge.setFixedHeight(20)
            top_row.addWidget(req_badge)
        else:
            opt_badge = QLabel("OPTIONAL")
            opt_badge.setStyleSheet("""
                background-color: #555;
                color: #ccc;
                font-size: 10px;
                font-weight: bold;
                padding: 2px 8px;
                border-radius: 4px;
                border: none;
            """)
            opt_badge.setFixedHeight(20)
            top_row.addWidget(opt_badge)

        top_row.addStretch()
        card_layout.addLayout(top_row)

        # Row 2: Description + size
        desc_label = QLabel(model_info.get("description", ""))
        desc_label.setStyleSheet("color: #aaa; font-size: 11px; border: none;")
        card_layout.addWidget(desc_label)

        size_text = f"Size: {format_size(expected_size)}"
        if is_downloaded:
            size_text += f"  |  Downloaded: {format_size(current_size)}"
            if path:
                size_text += f"  |  {path}"
        size_label = QLabel(size_text)
        size_label.setStyleSheet("color: #777; font-size: 10px; border: none;")
        size_label.setWordWrap(True)
        card_layout.addWidget(size_label)

        # Row 3: Progress bar (hidden by default)
        progress = QProgressBar()
        progress.setValue(0)
        progress.setVisible(False)
        progress.setFixedHeight(18)
        progress.setStyleSheet("""
            QProgressBar {
                border: 1px solid #555;
                border-radius: 4px;
                text-align: center;
                font-size: 10px;
                color: #fff;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
            }
        """)
        card_layout.addWidget(progress)

        # Row 4: Status text (for download progress)
        status_text = QLabel("")
        status_text.setStyleSheet("color: #4CAF50; font-size: 11px; border: none;")
        status_text.setVisible(False)
        card_layout.addWidget(status_text)

        # Row 5: Buttons
        btn_row = QHBoxLayout()
        btn_row.addStretch()

        if is_downloaded:
            dl_btn = QPushButton("Re-download")
            dl_btn.setStyleSheet("background-color: #555; max-width: 150px;")
        else:
            dl_btn = QPushButton("Download")
            dl_btn.setStyleSheet("background-color: #2196F3; max-width: 150px;")

        dl_btn.setFixedHeight(32)
        dl_btn.clicked.connect(
            lambda checked, mn=model_name: self.start_model_download(mn)
        )
        btn_row.addWidget(dl_btn)

        card_layout.addLayout(btn_row)

        # Store widget references for later updates
        self._model_cards[model_name] = {
            "card": card,
            "status_icon": status_icon,
            "size_label": size_label,
            "progress": progress,
            "status_text": status_text,
            "dl_btn": dl_btn,
        }

        return card

    def check_models_on_startup(self):
        """Check required models and show warning on tab if missing"""
        missing_required = []
        for model_name, model_info in MODELS.items():
            if model_info.get("required", False):
                is_downloaded, _, _ = check_model_status(model_name)
                if not is_downloaded:
                    missing_required.append(model_name)

        if missing_required:
            self.tabs.setTabText(self.models_tab_index, "⚠️ Models (Download Required)")
            self.tabs.tabBar().setTabTextColor(
                self.models_tab_index,
                QColor("#FF9800"),
            )
            if hasattr(self, "status_label"):
                self.status_label.setText("⚠️ Warning: Required AI models are missing. Face swapping will not work until downloaded from the Models tab.")
                self.status_label.setStyleSheet("color: #FF9800; font-weight: bold;")

    def start_model_download(self, model_name):
        """Start downloading a model in the background"""
        if model_name in self._active_downloads:
            return  # Already downloading

        model_info = MODELS.get(model_name)
        if not model_info:
            return

        widgets = self._model_cards.get(model_name)
        if not widgets:
            return

        # Update UI
        widgets["progress"].setValue(0)
        widgets["progress"].setVisible(True)
        widgets["status_text"].setText("Downloading...")
        widgets["status_text"].setVisible(True)
        widgets["status_text"].setStyleSheet(
            "color: #2196F3; font-size: 11px; border: none;"
        )
        widgets["dl_btn"].setEnabled(False)
        widgets["dl_btn"].setText("Downloading...")

        # Start download thread
        thread = ModelDownloadThread(model_name, model_info)
        thread.progress_update.connect(
            lambda pct, mn=model_name: self._on_download_progress(mn, pct)
        )
        thread.download_complete.connect(self._on_download_complete)
        thread.download_error.connect(self._on_download_error)
        thread.status_update.connect(
            lambda status, mn=model_name: self._on_download_status(mn, status)
        )
        thread.start()

        self._active_downloads[model_name] = thread

    def _on_download_progress(self, model_name, pct):
        widgets = self._model_cards.get(model_name)
        if widgets:
            widgets["progress"].setValue(pct)

    def _on_download_status(self, model_name, status):
        widgets = self._model_cards.get(model_name)
        if widgets:
            widgets["status_text"].setText(status)

    def _on_download_complete(self, model_name):
        widgets = self._model_cards.get(model_name)
        if widgets:
            widgets["progress"].setValue(100)
            widgets["status_text"].setText("Download complete!")
            widgets["status_text"].setStyleSheet(
                "color: #4CAF50; font-size: 11px; border: none;"
            )
            widgets["status_icon"].setText("[OK]")
            widgets["status_icon"].setStyleSheet(
                "font-size: 12px; font-weight: bold; color: #4CAF50; border: none;"
            )
            widgets["dl_btn"].setEnabled(True)
            widgets["dl_btn"].setText("Re-download")
            widgets["dl_btn"].setStyleSheet("background-color: #555; max-width: 150px;")

            # Update size label
            is_downloaded, path, current_size = check_model_status(model_name)
            model_info = MODELS.get(model_name, {})
            expected_size = model_info.get("size", 0)
            size_text = f"Size: {format_size(expected_size)}  |  Downloaded: {format_size(current_size)}"
            if path:
                size_text += f"  |  {path}"
            widgets["size_label"].setText(size_text)

        self._active_downloads.pop(model_name, None)

        # Update tab warning
        self._update_models_tab_badge()

    def _on_download_error(self, model_name, error_msg):
        widgets = self._model_cards.get(model_name)
        if widgets:
            widgets["progress"].setVisible(False)
            widgets["status_text"].setText(f"Error: {error_msg}")
            widgets["status_text"].setStyleSheet(
                "color: #f44336; font-size: 11px; border: none;"
            )
            widgets["dl_btn"].setEnabled(True)
            widgets["dl_btn"].setText("Retry")
            widgets["dl_btn"].setStyleSheet(
                "background-color: #f44336; max-width: 150px;"
            )

        self._active_downloads.pop(model_name, None)

    def _update_models_tab_badge(self):
        """Update the models tab title based on current model status"""
        has_missing = False
        for model_name, model_info in MODELS.items():
            if model_info.get("required", False):
                is_downloaded, _, _ = check_model_status(model_name)
                if not is_downloaded:
                    has_missing = True
                    break

        self.tabs.setTabText(
            self.models_tab_index, "[!] Models" if has_missing else "Models"
        )

    def refresh_model_status(self):
        """Refresh status of all model cards"""
        for model_name in list(self._model_cards.keys()):
            if model_name in self._active_downloads:
                continue  # Don't refresh while downloading
            widgets = self._model_cards[model_name]
            model_info = MODELS.get(model_name, {})
            is_downloaded, path, current_size = check_model_status(model_name)
            expected_size = model_info.get("size", 0)

            if is_downloaded:
                widgets["status_icon"].setText("[OK]")
                widgets["status_icon"].setStyleSheet(
                    "font-size: 12px; font-weight: bold; color: #4CAF50; border: none;"
                )
            else:
                widgets["status_icon"].setText("[!!]")
                widgets["status_icon"].setStyleSheet(
                    "font-size: 12px; font-weight: bold; color: #FF9800; border: none;"
                )

            size_text = f"Size: {format_size(expected_size)}"
            if is_downloaded:
                size_text += f"  |  Downloaded: {format_size(current_size)}"
                if path:
                    size_text += f"  |  {path}"
            widgets["size_label"].setText(size_text)

            if is_downloaded:
                widgets["dl_btn"].setText("Re-download")
                widgets["dl_btn"].setStyleSheet(
                    "background-color: #555; max-width: 150px;"
                )
            else:
                widgets["dl_btn"].setText("Download")
                widgets["dl_btn"].setStyleSheet(
                    "background-color: #2196F3; max-width: 150px;"
                )

            widgets["progress"].setVisible(False)
            widgets["status_text"].setVisible(False)

        self._update_models_tab_badge()
        self.status_label.setText("Model status refreshed")

    # --- Live Tab Methods ---

    def create_video_panel(self):
        """Create the video display panel"""
        group = QGroupBox("Live Video Feed")
        layout = QVBoxLayout()

        # Minimal clear button in top-right
        header = QHBoxLayout()
        header.addStretch()

        # Clear button
        self.clear_feed_btn = QPushButton("✕")
        self.clear_feed_btn.setFixedSize(20, 20)
        self.clear_feed_btn.setToolTip("Clear feed")
        self.clear_feed_btn.clicked.connect(self.clear_video_feed)
        self.clear_feed_btn.setStyleSheet("""
            QPushButton {
                background-color: #444;
                color: #fff;
                border-radius: 10px;
                font-weight: bold;
                padding: 0;
            }
            QPushButton:hover { background-color: #666; }
        """)
        header.addWidget(self.clear_feed_btn)
        layout.addLayout(header)

        # Alert banner (hidden by default)
        self.missing_model_banner = QLabel(
            "⚠️ Face swapping disabled: 'buffalo_l' detection model is missing. Download it in the Models tab."
        )
        self.missing_model_banner.setStyleSheet("""
            QLabel {
                background-color: #3a2e00;
                color: #ffcc00;
                border: 1px solid #cca300;
                border-radius: 4px;
                padding: 8px;
                font-weight: bold;
            }
        """)
        self.missing_model_banner.setVisible(False)
        self.missing_model_banner.setWordWrap(True)
        layout.addWidget(self.missing_model_banner)

        # Video display label
        self.video_label = QLabel()
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet(
            "QLabel { background-color: #1a1a1a; border: 2px solid #333; border-radius: 8px; }"
        )
        self.video_label.setText("Camera feed will appear here")
        layout.addWidget(self.video_label)

        group.setLayout(layout)
        return group

    def create_control_panel(self):
        """Create the control panel"""
        # Wrap in scroll area for cross-platform compatibility
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll_area.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: transparent;
            }
            QScrollArea > QWidget > QWidget {
                background-color: transparent;
            }
        """)

        widget = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(15)
        layout.setContentsMargins(5, 5, 5, 5)

        # Source image section
        source_group = QGroupBox("Source Image")
        source_layout = QVBoxLayout()

        # Header with Clear Button
        source_header = QHBoxLayout()
        source_header.addStretch()
        self.btn_clear_live_source = QPushButton("✕")
        self.btn_clear_live_source.setFixedSize(24, 24)
        self.btn_clear_live_source.setToolTip("Clear source selection")
        self.btn_clear_live_source.setEnabled(False)
        self.btn_clear_live_source.clicked.connect(self.clear_live_source)
        self.btn_clear_live_source.setStyleSheet("""
            QPushButton {
                background-color: #444; color: #fff; border-radius: 12px; font-weight: bold; padding: 0;
            }
            QPushButton:hover { background-color: #f44336; }
            QPushButton:disabled { background-color: #333; color: #555; }
        """)
        source_header.addWidget(self.btn_clear_live_source)
        source_layout.addLayout(source_header)

        self.source_preview = QLabel()
        self.source_preview.setFixedSize(200, 200)
        self.source_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.source_preview.setStyleSheet(
            "QLabel { background-color: #2a2a2a; border: 2px solid #444; border-radius: 8px; }"
        )
        self.source_preview.setText("No image selected")
        source_layout.addWidget(
            self.source_preview, alignment=Qt.AlignmentFlag.AlignCenter
        )

        self.select_source_btn = QPushButton("Select Source Image")
        self.select_source_btn.clicked.connect(self.select_source_image)
        source_layout.addWidget(self.select_source_btn)

        self.source_status = QLabel("No source loaded")
        self.source_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.source_status.setStyleSheet("color: #888;")
        source_layout.addWidget(self.source_status)

        source_layout.setContentsMargins(10, 25, 10, 10)
        source_group.setLayout(source_layout)
        layout.addWidget(source_group)

        # Camera controls section
        camera_group = QGroupBox("Camera Controls")
        camera_layout = QVBoxLayout()
        camera_layout.setContentsMargins(10, 25, 10, 10)
        camera_layout.setSpacing(8)

        # Camera selection
        camera_select_label = QLabel("Select Camera:")
        camera_select_label.setStyleSheet(
            "color: #ffffff; font-weight: bold; margin-top: 5px;"
        )
        camera_layout.addWidget(camera_select_label)

        camera_select_row = QHBoxLayout()
        self.camera_combo = QComboBox()
        self.camera_combo.setMinimumHeight(30)
        self.camera_combo.currentIndexChanged.connect(self.on_camera_selection_changed)
        self.camera_combo.setStyleSheet("""
            QComboBox {
                background-color: #2a2a2a;
                color: #ffffff;
                border: 2px solid #444;
                border-radius: 5px;
                padding: 5px;
            }
            QComboBox:hover {
                border-color: #4CAF50;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox QAbstractItemView {
                background-color: #2a2a2a;
                color: #ffffff;
                selection-background-color: #4CAF50;
            }
        """)
        camera_select_row.addWidget(self.camera_combo)

        self.refresh_cameras_btn = QPushButton("🔄")
        self.refresh_cameras_btn.setFixedSize(30, 30)
        self.refresh_cameras_btn.setToolTip("Refresh camera list")
        self.refresh_cameras_btn.clicked.connect(self.refresh_cameras)
        self.refresh_cameras_btn.setStyleSheet("""
            QPushButton {
                background-color: #555;
                border-radius: 5px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #666;
            }
        """)
        camera_select_row.addWidget(self.refresh_cameras_btn)
        camera_layout.addLayout(camera_select_row)

        self.start_btn = QPushButton("Start Camera")
        self.start_btn.clicked.connect(self.toggle_camera)
        self.start_btn.setFixedHeight(40)
        camera_layout.addWidget(self.start_btn)

        self.swap_btn = QPushButton("Enable Face Swap")
        self.swap_btn.clicked.connect(self.toggle_swap)
        self.swap_btn.setEnabled(False)
        self.swap_btn.setFixedHeight(40)
        camera_layout.addWidget(self.swap_btn)

        # Advanced masking section
        self.advanced_group = QGroupBox("Advanced Settings")
        advanced_layout = QVBoxLayout()
        advanced_layout.setSpacing(6)

        # Opacity slider
        opacity_layout = QHBoxLayout()
        self.opacity_label = QLabel("Opacity: 100%")
        self.opacity_label.setStyleSheet("color: white;")
        self.opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self.opacity_slider.setRange(0, 100)
        self.opacity_slider.setValue(100)
        self.opacity_slider.setEnabled(False)

        def _update_opacity(val):
            self.opacity_label.setText(f"Opacity: {val}%")
            from core import config

            config.OPACITY = val / 100.0

        self.opacity_slider.valueChanged.connect(_update_opacity)
        opacity_layout.addWidget(self.opacity_label)
        opacity_layout.addWidget(self.opacity_slider)
        advanced_layout.addLayout(opacity_layout)

        # Mouth mask checkbox
        self.mouth_mask_checkbox = QCheckBox("Enable Mouth Mask")
        self.mouth_mask_checkbox.setEnabled(False)

        def _toggle_mouth_mask(state):
            from core import config

            config.MOUTH_MASK_ENABLED = state == Qt.CheckState.Checked.value

        self.mouth_mask_checkbox.stateChanged.connect(_toggle_mouth_mask)
        self.mouth_mask_checkbox.setStyleSheet("QCheckBox { color: #ffffff; }")
        advanced_layout.addWidget(self.mouth_mask_checkbox)

        # Eyes mask checkbox
        self.eyes_mask_checkbox = QCheckBox("Enable Eyes Mask")
        self.eyes_mask_checkbox.setEnabled(False)

        def _toggle_eyes_mask(state):
            from core import config

            config.EYES_MASK_ENABLED = state == Qt.CheckState.Checked.value

        self.eyes_mask_checkbox.stateChanged.connect(_toggle_eyes_mask)
        self.eyes_mask_checkbox.setStyleSheet("QCheckBox { color: #ffffff; }")
        advanced_layout.addWidget(self.eyes_mask_checkbox)

        # Eyebrows mask checkbox
        self.eyebrows_mask_checkbox = QCheckBox("Enable Eyebrows Mask")
        self.eyebrows_mask_checkbox.setEnabled(False)

        def _toggle_eyebrows_mask(state):
            from core import config

            config.EYEBROWS_MASK_ENABLED = state == Qt.CheckState.Checked.value

        self.eyebrows_mask_checkbox.stateChanged.connect(_toggle_eyebrows_mask)
        self.eyebrows_mask_checkbox.setStyleSheet("QCheckBox { color: #ffffff; }")
        advanced_layout.addWidget(self.eyebrows_mask_checkbox)

        # Many Faces checkbox
        self.many_faces_checkbox = QCheckBox("Enable Many Faces")
        self.many_faces_checkbox.setEnabled(False)

        def _toggle_many_faces(state):
            from core import config

            config.MANY_FACES = state == Qt.CheckState.Checked.value

        self.many_faces_checkbox.stateChanged.connect(_toggle_many_faces)
        self.many_faces_checkbox.setStyleSheet("QCheckBox { color: #ffffff; }")
        advanced_layout.addWidget(self.many_faces_checkbox)

        # Mirror Camera checkbox
        self.mirror_camera_checkbox = QCheckBox("Mirror Camera")
        self.mirror_camera_checkbox.setEnabled(False)

        def _toggle_mirror(state):
            from core import config

            config.LIVE_MIRROR = state == Qt.CheckState.Checked.value

        self.mirror_camera_checkbox.stateChanged.connect(_toggle_mirror)
        self.mirror_camera_checkbox.setStyleSheet("QCheckBox { color: #ffffff; }")
        advanced_layout.addWidget(self.mirror_camera_checkbox)

        # Poisson Blending checkbox
        self.poisson_blend_checkbox = QCheckBox("Enable Seamless Blending")
        self.poisson_blend_checkbox.setEnabled(False)

        def _toggle_poisson(state):
            from core import config

            config.POISSON_BLEND_ENABLED = state == Qt.CheckState.Checked.value

        self.poisson_blend_checkbox.stateChanged.connect(_toggle_poisson)
        self.poisson_blend_checkbox.setStyleSheet("QCheckBox { color: #ffffff; }")
        advanced_layout.addWidget(self.poisson_blend_checkbox)

        # Enhancer Mode Combo
        enhancer_layout = QHBoxLayout()
        enhancer_label = QLabel("Live Enhancer:")
        enhancer_label.setStyleSheet("color: white;")
        self.enhancer_combo = QComboBox()
        self.enhancer_combo.addItems(["None", "GPEN-256", "GPEN-512", "GFPGAN"])
        self.enhancer_combo.setEnabled(False)

        def _enhancer_changed(idx):
            from core import config

            mapping = {0: "none", 1: "gpen256", 2: "gpen512", 3: "gfpgan"}
            config.ENHANCER_MODE = mapping.get(idx, "none")

            # Request video thread to reload enhancer logic if necessary
            if (
                config.ENHANCER_MODE != "none"
                and hasattr(self, "video_thread")
                and self.video_thread is not None
            ):
                self.video_thread.load_enhancer()

        self.enhancer_combo.currentIndexChanged.connect(_enhancer_changed)
        enhancer_layout.addWidget(enhancer_label)
        enhancer_layout.addWidget(self.enhancer_combo)
        advanced_layout.addLayout(enhancer_layout)

        self.advanced_group.setLayout(advanced_layout)
        camera_layout.addWidget(self.advanced_group)

        # Virtual camera checkbox (only if available)
        if VIRTUAL_CAM_AVAILABLE:
            self.virtual_cam_checkbox = QCheckBox("Enable Virtual Camera")
            self.virtual_cam_checkbox.setEnabled(False)
            self.virtual_cam_checkbox.stateChanged.connect(self.toggle_virtual_camera)
            self.virtual_cam_checkbox.setStyleSheet(
                "QCheckBox { color: #ffffff; padding: 8px 5px; }"
                "QCheckBox::indicator { width: 18px; height: 18px; }"
            )
            self.virtual_cam_checkbox.setToolTip(
                "Stream to virtual webcam for OBS/Zoom/Discord"
            )
            camera_layout.addWidget(self.virtual_cam_checkbox)

        camera_group.setLayout(camera_layout)
        layout.addWidget(camera_group)

        # ── Model status (Live tab) ───────────────────────────────────────────
        from core.config import ENHANCER_MODEL, SWAPPER_MODEL, ANALYSIS_MODEL

        live_models_group = QGroupBox("Model Status")
        live_models_layout = QVBoxLayout()
        live_models_layout.setContentsMargins(10, 20, 10, 10)
        live_models_layout.setSpacing(5)

        analyser_ok, _, _ = check_model_status(ANALYSIS_MODEL)

        _models_to_show = [
            ("Detect (buffalo_l)", analyser_ok),
            ("Swap (inswapper)", SWAPPER_MODEL.exists()),
            ("Enhance (GFPGAN)", ENHANCER_MODEL.exists()),
        ]
        for name, ok in _models_to_show:
            row = QHBoxLayout()
            dot = QLabel("●")
            dot.setFixedWidth(16)
            dot.setStyleSheet(
                f"color: {'#4CAF50' if ok else '#f44336'}; font-size: 14px; border: none;"
            )
            lbl = QLabel(name)
            lbl.setStyleSheet(
                f"color: {'#ccc' if ok else '#888'}; font-size: 11px; border: none;"
            )
            status = QLabel("Ready" if ok else "Not downloaded")
            status.setStyleSheet(
                f"color: {'#4CAF50' if ok else '#f44336'}; font-size: 11px; border: none;"
            )
            row.addWidget(dot)
            row.addWidget(lbl)
            row.addStretch()
            row.addWidget(status)
            live_models_layout.addLayout(row)

        goto_btn = QPushButton("→ Go to Models Tab")
        goto_btn.setStyleSheet("""
            QPushButton { background-color: #333; color: #58a6ff; font-size: 11px;
                          border: 1px solid #444; border-radius: 4px; padding: 4px 8px; }
            QPushButton:hover { background-color: #444; }
        """)
        goto_btn.clicked.connect(
            lambda: self.tabs.setCurrentIndex(self.models_tab_index)
        )
        live_models_layout.addWidget(goto_btn)
        live_models_group.setLayout(live_models_layout)
        layout.addWidget(live_models_group)

        # Info section
        info_group = QGroupBox("Information")
        info_layout = QVBoxLayout()
        info_layout.setContentsMargins(10, 25, 10, 10)

        info_text = QLabel(
            "1. Select a source image\n"
            "2. Start the camera\n"
            "3. Enable face swap\n\n"
            "The source face will be\n"
            "swapped onto detected faces\n"
            "in the live video feed."
        )
        info_text.setWordWrap(True)
        info_text.setStyleSheet("color: #aaa; font-size: 11px;")
        info_layout.addWidget(info_text)

        info_group.setLayout(info_layout)
        layout.addWidget(info_group)

        layout.addStretch()
        widget.setLayout(layout)
        scroll_area.setWidget(widget)
        return scroll_area

    def detect_and_populate_cameras(self):
        """Detect cameras in background and populate the combo box"""
        self.status_label.setText("Detecting cameras...")
        self.camera_combo.setEnabled(False)
        self.refresh_cameras_btn.setEnabled(False)

        # Start background detection thread
        self.camera_detection_thread = CameraDetectionThread(max_cameras=6)
        self.camera_detection_thread.cameras_detected.connect(self.on_cameras_detected)
        self.camera_detection_thread.start()

    def on_cameras_detected(self, available_cameras):
        """Handle camera detection results from background thread"""
        self.available_cameras = available_cameras

        self.camera_combo.clear()
        if self.available_cameras:
            for cam_idx in self.available_cameras:
                self.camera_combo.addItem(f"Camera {cam_idx}", cam_idx)
            self.status_label.setText(f"Found {len(self.available_cameras)} camera(s)")
            # Set default selection
            if self.selected_camera_index in self.available_cameras:
                idx = self.available_cameras.index(self.selected_camera_index)
                self.camera_combo.setCurrentIndex(idx)
        else:
            self.camera_combo.addItem("No cameras found", -1)
            self.selected_camera_index = -1
            self.status_label.setText("No cameras detected")

        self.camera_combo.setEnabled(True)
        self.refresh_cameras_btn.setEnabled(True)

    def refresh_cameras(self):
        """Refresh the camera list"""
        if self.is_capturing:
            QMessageBox.warning(
                self,
                "Camera Active",
                "Please stop the camera before refreshing the camera list.",
            )
            return
        self.detect_and_populate_cameras()

    def on_camera_selection_changed(self, index):
        """Handle camera selection change"""
        if index >= 0:
            self.selected_camera_index = self.camera_combo.itemData(index)
            if self.selected_camera_index >= 0:
                self.status_label.setText(
                    f"Selected Camera {self.selected_camera_index}"
                )

    def select_source_image(self):
        """Open file dialog to select source image for live mode"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Source Image",
            self.working_dir,
            "Image Files (*.png *.jpg *.jpeg *.bmp)",
        )

        if file_path:
            try:
                # Show loading indicator
                self.status_label.setText("Loading source image...")
                QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
                QApplication.processEvents()  # Keep UI responsive

                # Load image
                self.source_image = cv2.imread(file_path)
                if self.source_image is None:
                    raise ValueError("Failed to load image")

                # Extract face from source image
                self.status_label.setText("Analyzing face...")
                QApplication.processEvents()
                face_analyser = get_face_analyser()
                faces = face_analyser.get(self.source_image)

                if len(faces) == 0:
                    QApplication.restoreOverrideCursor()
                    QMessageBox.warning(
                        self,
                        "No Face Detected",
                        "No face was detected in the selected image. Please choose an image with a clear face.",
                    )
                    self.status_label.setText("Ready")
                    return

                # Use the first detected face
                self.source_face = faces[0]

                # Display preview
                self.display_source_preview(self.source_image, self.source_preview)
                self.source_status.setText(f"✓ Face detected")
                self.source_status.setStyleSheet("color: #4CAF50;")
                self.btn_clear_live_source.setEnabled(True)

                # Update video thread if running
                if self.video_thread is not None:
                    self.video_thread.set_source_face(self.source_face)
                    self.swap_btn.setEnabled(True)

                self.status_label.setText("Source image loaded successfully")
                QApplication.restoreOverrideCursor()

            except Exception as e:
                QApplication.restoreOverrideCursor()
                QMessageBox.critical(
                    self, "Error", f"Failed to process image: {str(e)}"
                )
                self.status_label.setText("Ready")

    def clear_live_source(self):
        """Clear the live source image selection"""
        self.source_image = None
        self.source_face = None
        self.source_preview.clear()
        self.source_preview.setText("No image selected")
        self.source_status.setText("No source loaded")
        self.source_status.setStyleSheet("color: #888;")
        self.btn_clear_live_source.setEnabled(False)

        # Update video thread if running
        if self.video_thread is not None:
            self.video_thread.set_source_face(None)
            self.swap_btn.setEnabled(False)
            # If swap was enabled, disable it
            if self.video_thread.swap_enabled:
                self.toggle_swap()

        self.status_label.setText("Source image cleared")

    def display_source_preview(self, image, label_widget):
        """Display source image preview on a specific label"""
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize to fit preview
        h, w = rgb_image.shape[:2]
        aspect = w / h
        if aspect > 1:
            new_w = 200
            new_h = int(200 / aspect)
        else:
            new_h = 200
            new_w = int(200 * aspect)

        resized = cv2.resize(rgb_image, (new_w, new_h))

        # Convert to QPixmap
        q_image = QImage(
            resized.data, new_w, new_h, new_w * 3, QImage.Format.Format_RGB888
        )
        pixmap = QPixmap.fromImage(q_image)

        label_widget.setPixmap(pixmap)

    def toggle_camera(self):
        """Start or stop camera capture"""
        if not self.is_capturing:
            self.start_camera()
        else:
            self.stop_camera()

    def start_camera(self):
        """Start video capture"""
        try:
            # Validate camera selection
            if self.selected_camera_index < 0:
                QMessageBox.warning(
                    self,
                    "No Camera",
                    "No camera selected. Please select a valid camera from the dropdown.",
                )
                return

            # Create and start video thread with selected camera
            self.video_thread = VideoThread(camera_index=self.selected_camera_index)
            self.video_thread.frame_ready.connect(self.update_frame)
            self.video_thread.fps_update.connect(self.update_fps)
            self.video_thread.face_count_update.connect(self.update_face_count)
            self.video_thread.error_occurred.connect(self.handle_error)

            # Set source face if available
            if self.source_face is not None:
                self.video_thread.set_source_face(self.source_face)
                self.swap_btn.setEnabled(True)

            self.video_thread.start()

            self.is_capturing = True
            self.start_btn.setText("Stop Camera")
            self.start_btn.setStyleSheet("background-color: #f44336;")
            self.camera_combo.setEnabled(
                False
            )  # Disable camera selection while running
            self.refresh_cameras_btn.setEnabled(False)

            # Enable virtual camera checkbox if available
            if VIRTUAL_CAM_AVAILABLE:
                self.virtual_cam_checkbox.setEnabled(True)

            # Show a subtle banner if the analyser model is missing
            from core.config import ANALYSIS_MODEL
            from download_models import check_model_status

            is_downloaded, _, _ = check_model_status(ANALYSIS_MODEL)
            self.missing_model_banner.setVisible(not is_downloaded)

            self.status_label.setText(f"Camera {self.selected_camera_index} started")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to start camera: {str(e)}")

    def stop_camera(self):
        """Stop video capture"""
        self.missing_model_banner.setVisible(False)
        if self.video_thread is not None:
            self.video_thread.stop()
            self.video_thread = None

        self.is_capturing = False
        self.start_btn.setText("Start Camera")
        self.start_btn.setStyleSheet(
            "QPushButton { background-color: #4CAF50; } QPushButton:hover { background-color: #45a049; } QPushButton:disabled { background-color: #555; color: #888; }"
        )
        self.camera_combo.setEnabled(True)  # Re-enable camera selection
        self.refresh_cameras_btn.setEnabled(True)
        self.swap_btn.setEnabled(False)
        self.swap_btn.setText("Enable Face Swap")
        self.swap_btn.setStyleSheet(
            "QPushButton { background-color: #4CAF50; } QPushButton:hover { background-color: #45a049; } QPushButton:disabled { background-color: #555; color: #888; }"
        )
        self.mouth_mask_checkbox.setEnabled(False)
        self.eyes_mask_checkbox.setEnabled(False)
        self.eyebrows_mask_checkbox.setEnabled(False)
        self.poisson_blend_checkbox.setEnabled(False)
        self.opacity_slider.setEnabled(False)
        self.mouth_mask_checkbox.setChecked(False)
        self.eyes_mask_checkbox.setChecked(False)
        self.eyebrows_mask_checkbox.setChecked(False)
        self.poisson_blend_checkbox.setChecked(False)
        self.many_faces_checkbox.setChecked(False)
        self.mirror_camera_checkbox.setChecked(False)
        self.enhancer_combo.setCurrentIndex(0)
        self.many_faces_checkbox.setEnabled(False)
        self.mirror_camera_checkbox.setEnabled(False)
        self.enhancer_combo.setEnabled(False)

        # Stop and disable virtual camera
        if VIRTUAL_CAM_AVAILABLE:
            if self.virtual_cam_enabled:
                self.virtual_cam_checkbox.setChecked(False)
            self.virtual_cam_checkbox.setEnabled(False)

        # Automatically clear the video feed
        self.clear_video_feed()

        self.status_label.setText("Camera stopped")
        self.fps_label.setText("FPS: 0")
        self.face_count_label.setText("Faces: 0")

    def clear_video_feed(self):
        """Clear the video feed display"""
        # Remove any pixmap and force text display
        self.video_label.setPixmap(QPixmap())
        self.video_label.clear()
        self.video_label.setText("Camera feed will appear here")
        self.video_label.repaint()  # Force immediate repaint

    def toggle_swap(self):
        """Toggle face swapping on/off"""
        if self.video_thread is not None:
            current_state = self.video_thread.swap_enabled
            new_state = not current_state

            if new_state:
                # Validate required models before enabling face swap
                from core.config import SWAPPER_MODEL, ANALYSIS_MODEL
                from download_models import check_model_status

                analyser_ok, _, _ = check_model_status(ANALYSIS_MODEL)
                swapper_ok = SWAPPER_MODEL.exists()

                if not swapper_ok or not analyser_ok:
                    missing = []
                    if not swapper_ok:
                        missing.append("Face Swap Model (inswapper_128.onnx)")
                    if not analyser_ok:
                        missing.append("Face Detection Model Pack (buffalo_l)")

                    reply = QMessageBox.warning(
                        self,
                        "Models Required",
                        "Face swapping cannot function without required AI models:\n\n"
                        + "\n".join(f"• {m}" for m in missing)
                        + "\n\nFace swap is disabled until these are downloaded.\n"
                        + "Would you like to open the Models tab to download them now?",
                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                        QMessageBox.StandardButton.Yes,
                    )
                    if reply == QMessageBox.StandardButton.Yes:
                        self.tabs.setCurrentIndex(self.models_tab_index)
                    return

                self.video_thread.enable_swap(True)
                self.swap_btn.setText("Disable Face Swap")
                self.swap_btn.setStyleSheet("background-color: #FF9800;")
                self.status_label.setText("Face swapping enabled")
                self.mouth_mask_checkbox.setEnabled(True)
                self.eyes_mask_checkbox.setEnabled(True)
                self.eyebrows_mask_checkbox.setEnabled(True)
                self.poisson_blend_checkbox.setEnabled(True)
                self.many_faces_checkbox.setEnabled(True)
                self.mirror_camera_checkbox.setEnabled(True)
                self.enhancer_combo.setEnabled(True)
                self.opacity_slider.setEnabled(True)
            else:
                self.swap_btn.setText("Enable Face Swap")
                self.swap_btn.setStyleSheet(
                    "QPushButton { background-color: #4CAF50; } QPushButton:hover { background-color: #45a049; } QPushButton:disabled { background-color: #555; color: #888; }"
                )
                self.status_label.setText("Face swapping disabled")
                self.mouth_mask_checkbox.setEnabled(False)
                self.eyes_mask_checkbox.setEnabled(False)
                self.eyebrows_mask_checkbox.setEnabled(False)
                self.poisson_blend_checkbox.setEnabled(False)
                self.many_faces_checkbox.setEnabled(False)
                self.mirror_camera_checkbox.setEnabled(False)
                self.enhancer_combo.setEnabled(False)
                self.opacity_slider.setEnabled(False)
                self.mouth_mask_checkbox.setChecked(False)
                self.eyes_mask_checkbox.setChecked(False)
                self.eyebrows_mask_checkbox.setChecked(False)
                self.poisson_blend_checkbox.setChecked(False)
                self.many_faces_checkbox.setChecked(False)
                self.mirror_camera_checkbox.setChecked(False)
                self.enhancer_combo.setCurrentIndex(0)

    def toggle_mouth_mask(self, state):
        """Toggle mouth masking on/off"""
        if self.video_thread is not None:
            enabled = state == Qt.CheckState.Checked.value
            self.video_thread.enable_mouth_mask(enabled)

            if enabled:
                self.status_label.setText("Mouth masking enabled")
            else:
                self.status_label.setText("Mouth masking disabled")

    def toggle_virtual_camera(self, state):
        """Toggle virtual camera output"""
        if not VIRTUAL_CAM_AVAILABLE:
            return

        enabled = state == Qt.CheckState.Checked.value

        if enabled:
            # Check if camera is running
            if not self.is_capturing or self.video_thread is None:
                self.virtual_cam_checkbox.setChecked(False)
                QMessageBox.warning(
                    self,
                    "Camera Not Running",
                    "Please start the camera before enabling virtual camera.",
                )
                return

            try:
                # Get actual camera resolution from video thread
                if (
                    hasattr(self.video_thread, "cap")
                    and self.video_thread.cap is not None
                ):
                    width = int(self.video_thread.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(self.video_thread.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                else:
                    # Fallback to common resolution
                    width, height = 640, 480

                # Create virtual camera with actual camera resolution
                self.virtual_cam = pyvirtualcam.Camera(
                    width=width, height=height, fps=30, fmt=pyvirtualcam.PixelFormat.BGR
                )
                self.virtual_cam_enabled = True
                self.status_label.setText(
                    f"Virtual camera started: {self.virtual_cam.device} ({width}x{height})"
                )
            except Exception as e:
                self.virtual_cam_enabled = False
                self.virtual_cam_checkbox.setChecked(False)
                QMessageBox.critical(
                    self,
                    "Virtual Camera Error",
                    f"Failed to start virtual camera:\n{str(e)}\n\nOn Linux, install v4l2loopback:\nsudo modprobe v4l2loopback",
                )
        else:
            if self.virtual_cam is not None:
                self.virtual_cam.close()
                self.virtual_cam = None
            self.virtual_cam_enabled = False
            self.status_label.setText("Virtual camera stopped")

    def update_frame(self, frame):
        """Update video display with new frame"""
        # Convert BGR to RGB for display
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert to QImage
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        q_image = QImage(
            rgb_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888
        )

        # Scale to fit label while maintaining aspect ratio
        pixmap = QPixmap.fromImage(q_image)
        scaled_pixmap = pixmap.scaled(
            self.video_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

        self.video_label.setPixmap(scaled_pixmap)

        # Send to virtual camera if enabled
        if self.virtual_cam_enabled and self.virtual_cam is not None:
            try:
                # Send frame at original resolution (matches virtual camera size)
                self.virtual_cam.send(frame)
            except Exception as e:
                print(f"Virtual camera error: {e}")
                self.virtual_cam_enabled = False
                if VIRTUAL_CAM_AVAILABLE:
                    self.virtual_cam_checkbox.setChecked(False)

    def update_fps(self, fps):
        """Update FPS display"""
        self.fps_label.setText(f"FPS: {fps:.1f}")

    def update_face_count(self, count):
        """Update face count display"""
        self.face_count_label.setText(f"Faces: {count}")

    def handle_error(self, error_msg):
        """Handle errors from video thread"""
        QMessageBox.critical(self, "Error", error_msg)
        self.stop_camera()

    # --- Offline Tab Methods ---

    def select_offline_source_image(self):
        """Select source image for offline mode (single-face)"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Source Face Image",
            self.working_dir,
            "Image Files (*.png *.jpg *.jpeg *.bmp *.webp)",
        )

        if file_path:
            try:
                QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
                self._single_src_status.setText("Loading…")
                QApplication.processEvents()

                image = cv2.imread(file_path)
                if image is None:
                    raise ValueError("Failed to load image")

                self._single_src_status.setText("Detecting face…")
                QApplication.processEvents()
                face_analyser = get_face_analyser()
                faces = face_analyser.get(image)

                if not faces:
                    QApplication.restoreOverrideCursor()
                    QMessageBox.warning(
                        self, "No Face", "No face was detected in the selected image."
                    )
                    self._single_src_status.setText("No source loaded")
                    return

                self.offline_source_face = faces[0]
                self.offline_source_image = image

                # Update preview
                self.display_source_preview(image, self.offline_source_preview)
                self.offline_source_preview.setStyleSheet(
                    "background:#1a1a1a; border:2px solid #4CAF50; border-radius:10px;"
                )
                self._single_src_status.setText(
                    f"✓ Face detected\n{Path(file_path).name}"
                )
                self._single_src_status.setStyleSheet(
                    "color:#4CAF50; font-size:11px; border:none; background:transparent;"
                )
                self.btn_clear_source.setEnabled(True)
                self.check_offline_readiness()
                QApplication.restoreOverrideCursor()

            except Exception as e:
                QApplication.restoreOverrideCursor()
                self._single_src_status.setText("Error loading image")
                QMessageBox.critical(self, "Error", str(e))

    def clear_offline_source(self):
        """Clear the offline source selection"""
        self.offline_source_image = None
        self.offline_source_face = None
        self.offline_source_preview.clear()
        self.offline_source_preview.setText("No face\nselected")
        self.offline_source_preview.setStyleSheet(
            "background:#1a1a1a; border:2px dashed #3a3a3a;"
            "border-radius:10px; color:#444; font-size:11px;"
        )
        self._single_src_status.setText("No source loaded")
        self._single_src_status.setStyleSheet(
            "color:#555; font-size:11px; border:none; background:transparent;"
        )
        self.btn_clear_source.setEnabled(False)
        self.check_offline_readiness()

    def _on_face_source_mode_changed(self, single_checked: bool):
        """Toggle between Single-Face and Multi-Face source modes."""
        self.single_face_panel.setVisible(single_checked)
        self.multi_face_panel.setVisible(not single_checked)
        self.check_offline_readiness()

    def handle_file_drop(self, file_path):
        """Handle dropped file for offline processing"""
        self.target_file_path = file_path
        self.last_target_dir = str(Path(file_path).parent)
        self.btn_clear_target.setEnabled(True)
        self.process_info_label.setText(f"Target: {Path(file_path).name}")
        self.face_selector.scan(file_path, working_dir=self.working_dir)
        self.check_offline_readiness()

    def browse_target_file(self):
        """Browse for target video/image"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Target File",
            self.working_dir,
            "Video/Image Files (*.mp4 *.avi *.mov *.jpg *.png)",
        )
        if file_path:
            self.last_target_dir = str(Path(file_path).parent)
            self.handle_file_drop(file_path)
            # Update drag widget text manually if browsed
            self.drop_widget.setText(f"Selected: {Path(file_path).name}")
            self.drop_widget.setStyleSheet("""
                QLabel {
                    border: 2px solid #4CAF50;
                    border-radius: 10px;
                    background-color: #252525;
                    color: #4CAF50;
                    font-weight: bold;
                    font-size: 14px;
                    padding: 20px;
                }
            """)
            self.btn_clear_target.setEnabled(True)

    def clear_target_file(self):
        """Clear the target file selection"""
        self.target_file_path = None
        self.drop_widget.setText("Drag & Drop Video/Image Here")
        self.drop_widget.setStyleSheet("""
            QLabel {
                border: 2px dashed #666;
                border-radius: 10px;
                background-color: #252525;
                color: #aaa;
                font-size: 14px;
                padding: 20px;
            }
            QLabel:hover {
                border-color: #4CAF50;
                background-color: #2d2d2d;
                color: #fff;
            }
        """)
        self.btn_clear_target.setEnabled(False)
        self.check_offline_readiness()
        self.process_info_label.setText(
            "Select a source face and a target file to begin."
        )

    def check_offline_readiness(self):
        """Enable start button if all required inputs are ready."""
        enhance_only = self.radio_enhance.isChecked()
        has_target = self.target_file_path is not None
        is_single = (
            hasattr(self, "radio_single_face") and self.radio_single_face.isChecked()
        )

        if enhance_only:
            ready = has_target
        elif is_single:
            ready = has_target and (self.offline_source_face is not None)
        else:
            has_face_map = (
                hasattr(self, "face_selector") and self.face_selector.has_any_source
            )
            ready = has_target and has_face_map

        self.btn_start_process.setEnabled(ready)

        if ready:
            if enhance_only:
                self.process_info_label.setText("Ready to enhance.")
            elif is_single:
                self.process_info_label.setText("✓ Ready — single face swap.")
                self.process_info_label.setStyleSheet("color:#4CAF50; font-size:11px;")
            else:
                n = len(self.face_selector.face_pairs)
                self.process_info_label.setText(
                    f"✓ Ready — {n} face assignment(s) set."
                )
                self.process_info_label.setStyleSheet("color:#4CAF50; font-size:11px;")
        else:
            self.process_info_label.setStyleSheet(
                "color:#888; font-style:italic; font-size:11px;"
            )
            if enhance_only:
                self.process_info_label.setText("Select a target file to begin.")
            elif is_single:
                self.process_info_label.setText(
                    "Browse a source face image and a target file."
                )
            else:
                self.process_info_label.setText(
                    "Switch to Multi-Face mode, load a target, then assign source faces."
                )

    def start_offline_processing(self):
        """Start the offline processing thread"""
        from app.file_processing_thread import (
            MODE_SWAP,
            MODE_ENHANCE,
            MODE_SWAP_ENHANCE,
        )
        from core.config import ENHANCER_MODEL

        if self.radio_enhance.isChecked():
            mode = MODE_ENHANCE
        elif self.radio_swap_enhance.isChecked():
            mode = MODE_SWAP_ENHANCE
        else:
            mode = MODE_SWAP

        if not self.target_file_path:
            return

        # Guard: Swapper and Analyser models must exist for face swapping
        if mode in (MODE_SWAP, MODE_SWAP_ENHANCE):
            from core.config import SWAPPER_MODEL, ANALYSIS_MODEL
            from download_models import check_model_status

            analyser_ok, _, _ = check_model_status(ANALYSIS_MODEL)
            swapper_ok = SWAPPER_MODEL.exists()

            if not swapper_ok or not analyser_ok:
                missing = []
                if not swapper_ok:
                    missing.append("Face Swap Model (inswapper_128.onnx)")
                if not analyser_ok:
                    missing.append("Face Detection Model Pack (buffalo_l)")

                reply = QMessageBox.warning(
                    self,
                    "Models Required",
                    "Offline face swapping cannot perform without required AI models:\n\n"
                    + "\n".join(f"• {m}" for m in missing)
                    + "\n\nWould you like to open the Models tab to download them now?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.Yes,
                )
                if reply == QMessageBox.StandardButton.Yes:
                    self.tabs.setCurrentIndex(self.models_tab_index)
                return

        # Guard: GFPGAN model must exist for enhance modes
        if mode in (MODE_ENHANCE, MODE_SWAP_ENHANCE) and not ENHANCER_MODEL.exists():
            QMessageBox.warning(
                self,
                "GFPGAN Model Missing",
                f"Enhancement model not found:\n{ENHANCER_MODEL}\n\n"
                "Please download it from the Models tab first.",
            )
            return

        is_single = self.radio_single_face.isChecked()
        face_map = []

        if mode != MODE_ENHANCE:
            if is_single:
                if not self.offline_source_face:
                    QMessageBox.warning(
                        self,
                        "No Source Face",
                        "Please browse and select a source face image.",
                    )
                    return
                # face_map stays [] — uses source_face below
            else:
                face_map = (
                    self.face_selector.face_pairs
                    if hasattr(self, "face_selector")
                    else []
                )
                if not face_map:
                    QMessageBox.warning(
                        self,
                        "No Source Faces",
                        "Please assign a source face to at least one person.",
                    )
                    return

        self.btn_start_process.setEnabled(False)
        self.btn_cancel_process.setEnabled(True)
        self.progress_bar.setValue(0)

        if face_map:
            # Multi-face mode
            self.processing_thread = FileProcessingThread(
                source_face=None,
                target_path=self.target_file_path,
                mode=mode,
                face_map=face_map,
            )
        else:
            # Single-face or enhance-only mode
            self.processing_thread = FileProcessingThread(
                source_face=self.offline_source_face,
                target_path=self.target_file_path,
                mode=mode,
            )

        self.processing_thread.progress_update.connect(self._handle_offline_progress)
        self.processing_thread.status_update.connect(self._handle_offline_status)
        self.processing_thread.finished_processing.connect(self.on_processing_finished)
        self.processing_thread.error_occurred.connect(self.on_processing_error)
        self.processing_thread.start()

    def _handle_offline_progress(self, value: int):
        """Handle progress signal: -1 means indeterminate (GFPGAN init/download)."""
        if value == -1:
            self.progress_bar.setRange(0, 0)  # pulsing / indeterminate
        else:
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(value)
            if value > 0:
                self.download_status_label.setVisible(False)

    def _handle_offline_status(self, text: str):
        """Route status updates: download lines go to the orange indicator label."""
        low = text.lower()
        is_download = text.startswith("\u2193 ") or any(
            k in low for k in ("download", "loading gfpgan", "auxiliary")
        )
        if is_download:
            self.download_status_label.setText(text)
            self.download_status_label.setVisible(True)
        else:
            # For non-download messages, update the main info label
            self.process_info_label.setText(text)
            if "ready" in low or "complete" in low:
                self.download_status_label.setVisible(False)

    def cancel_offline_processing(self):
        """Cancel an in-progress processing job"""
        if self.processing_thread and self.processing_thread.isRunning():
            self.processing_thread.stop()
            self.process_info_label.setText("Processing cancelled.")
        self.btn_cancel_process.setEnabled(False)
        self.btn_start_process.setEnabled(True)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)

    def on_processing_finished(self, output_path):
        """Handle processing completion"""
        self.btn_start_process.setEnabled(True)
        self.btn_cancel_process.setEnabled(False)
        self.last_output_path = output_path

        self.process_info_label.setText(f"✓ Done! Saved to: {Path(output_path).name}")
        self.process_info_label.setStyleSheet("color: #4CAF50; font-size: 11px;")

        # Enable actions
        self.btn_open_file.setEnabled(True)
        self.btn_open_folder.setEnabled(True)

        # Show preview if image
        if Path(output_path).suffix.lower() in {
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp",
            ".webp",
        }:
            self.show_result_preview(output_path)

    def show_result_preview(self, path):
        """Show preview of processed image"""
        try:
            image = cv2.imread(path)
            if image is not None:
                # Resize for preview (max 300px height)
                h, w = image.shape[:2]
                scale = min(1.0, 300 / h)
                new_w = int(w * scale)
                new_h = int(h * scale)
                resized = cv2.resize(image, (new_w, new_h))

                rgb_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                q_image = QImage(
                    rgb_image.data, new_w, new_h, new_w * 3, QImage.Format.Format_RGB888
                )
                self.result_preview_label.setPixmap(QPixmap.fromImage(q_image))
                self.result_preview_label.setVisible(True)
        except Exception as e:
            print(f"Preview error: {e}")

    def open_output_file(self):
        """Open the processed file"""
        if self.last_output_path:
            QDesktopServices.openUrl(QUrl.fromLocalFile(self.last_output_path))

    def open_output_folder(self):
        """Open the folder containing the processed file"""
        if self.last_output_path:
            folder = str(Path(self.last_output_path).parent)
            QDesktopServices.openUrl(QUrl.fromLocalFile(folder))

    def on_processing_error(self, error_msg):
        """Handle processing error"""
        self.btn_start_process.setEnabled(True)
        self.btn_cancel_process.setEnabled(False)
        self.process_info_label.setText("⚠ Error occurred.")
        self.process_info_label.setStyleSheet("color: #f44336; font-size: 11px;")
        QMessageBox.critical(self, "Error", f"Processing failed: {error_msg}")

    def apply_styles(self):
        """Apply modern styling to the application"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1e1e1e;
            }
            QTabWidget::pane {
                border: 1px solid #444;
                background-color: #1e1e1e;
            }
            QTabBar::tab {
                background-color: #2a2a2a;
                color: #aaa;
                padding: 10px 20px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background-color: #3a3a3a;
                color: #fff;
                font-weight: bold;
                border-bottom: 2px solid #4CAF50;
            }
            QGroupBox {
                font-weight: bold;
                font-size: 13px;
                color: #ffffff;
                border: 2px solid #444;
                border-radius: 8px;
                margin-top: 12px;
                padding-top: 18px;
                background-color: #2a2a2a;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                left: 10px;
                padding: 0 5px;
                top: 2px;
            }
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 10px;
                font-size: 13px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
            QPushButton:disabled {
                background-color: #555;
                color: #888;
            }
            QLabel {
                color: #ffffff;
            }
            QStatusBar {
                background-color: #2a2a2a;
                color: #ffffff;
            }
            QStatusBar QLabel {
                color: #ffffff;
                padding: 2px 8px;
            }
            QCheckBox {
                color: white;
            }
        """)

    def load_settings(self):
        """Load settings from JSON file"""
        try:
            if self.settings_file.exists():
                with open(self.settings_file, "r") as f:
                    settings = json.load(f)
                    return settings.get("working_dir", str(Path.home()))
        except Exception as e:
            print(f"Error loading settings: {e}")
        return str(Path.home())

    def save_settings(self):
        """Save settings to JSON file"""
        try:
            settings = {"working_dir": self.working_dir}
            with open(self.settings_file, "w") as f:
                json.dump(settings, f, indent=2)
        except Exception as e:
            print(f"Error saving settings: {e}")

    def open_settings(self):
        """Open settings dialog"""
        dialog = SettingsDialog(self.working_dir, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.working_dir = dialog.get_working_dir()
            self.save_settings()
            self.status_label.setText(
                f"Working directory updated: {Path(self.working_dir).name}"
            )

    def setup_about_tab(self):
        """Setup the About tab with app information"""
        layout = QVBoxLayout(self.about_tab)
        layout.setContentsMargins(0, 0, 0, 0)

        # Scroll area so it works on small screens
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")

        content = QWidget()
        inner = QVBoxLayout(content)
        inner.setSpacing(16)
        inner.setContentsMargins(40, 40, 40, 40)

        # Centre everything
        inner.addStretch()

        # App icon
        icon_path = Path(__file__).parent.parent / "assets" / "icon.png"
        if icon_path.exists():
            icon_label = QLabel()
            pixmap = QPixmap(str(icon_path)).scaled(
                96,
                96,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            icon_label.setPixmap(pixmap)
            icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            icon_label.setStyleSheet("border: none;")
            inner.addWidget(icon_label)

        # App name
        name_label = QLabel("Deep Face Net")
        name_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        name_label.setStyleSheet("font-size: 28px; font-weight: bold; color: #ffffff;")
        inner.addWidget(name_label)

        # Version badge
        version_label = QLabel("v2.0.0")
        version_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        version_label.setStyleSheet(
            "font-size: 14px; color: #4CAF50; font-weight: bold;"
        )
        inner.addWidget(version_label)

        # Description
        desc_label = QLabel("Advanced Real-time Face Swapping Application")
        desc_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        desc_label.setWordWrap(True)
        desc_label.setStyleSheet("font-size: 13px; color: #aaa;")
        inner.addWidget(desc_label)

        # Separator
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setFixedWidth(300)
        sep.setStyleSheet("background-color: #444; max-height: 1px;")
        inner.addWidget(sep, alignment=Qt.AlignmentFlag.AlignCenter)

        # Info card
        info_card = QFrame()
        info_card.setFixedWidth(360)
        info_card.setStyleSheet("""
            QFrame {
                background-color: #2a2a2a;
                border: 1px solid #444;
                border-radius: 10px;
                padding: 6px;
            }
        """)
        info_layout = QVBoxLayout(info_card)
        info_layout.setSpacing(8)
        info_layout.setContentsMargins(20, 16, 20, 16)

        for key_text, val_text in [
            ("Developed", "PolyNet Org"),
            (
                "Python",
                f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            ),
            ("Qt", qVersion()),
            ("PyQt6", PYQT_VERSION_STR),
        ]:
            row = QHBoxLayout()
            key = QLabel(key_text)
            key.setStyleSheet("color: #888; font-size: 12px; border: none;")
            key.setFixedWidth(80)
            val = QLabel(val_text)
            val.setStyleSheet(
                "color: #fff; font-size: 12px; font-weight: bold; border: none;"
            )
            row.addWidget(key)
            row.addWidget(val)
            row.addStretch()
            info_layout.addLayout(row)

        inner.addWidget(info_card, alignment=Qt.AlignmentFlag.AlignCenter)

        # GitHub button
        github_btn = QPushButton("  POLYNET")
        github_btn.setFixedWidth(260)
        github_btn.setFixedHeight(40)
        github_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        github_btn.clicked.connect(
            lambda: QDesktopServices.openUrl(QUrl("https://www.polynet.online"))
        )
        github_btn.setStyleSheet("""
            QPushButton {
                background-color: #333;
                color: #58a6ff;
                border: 1px solid #444;
                border-radius: 8px;
                font-size: 13px;
            }
            QPushButton:hover {
                background-color: #444;
                border-color: #58a6ff;
            }
        """)
        inner.addWidget(github_btn, alignment=Qt.AlignmentFlag.AlignCenter)

        inner.addStretch()

        scroll.setWidget(content)
        layout.addWidget(scroll)

    def closeEvent(self, event):
        """Handle application close event"""
        if self.is_capturing:
            self.stop_camera()
        if self.processing_thread and self.processing_thread.isRunning():
            self.processing_thread.stop()
        # Cancel active model downloads
        for thread in self._active_downloads.values():
            thread.cancel()
        event.accept()


def main():
    """Main application entry point"""
    # Enable High DPI scaling for Windows/multi-monitor setups
    try:
        from PyQt6.QtCore import Qt as QtCore_Qt

        QApplication.setHighDpiScaleFactorRoundingPolicy(
            QtCore_Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
        )
    except AttributeError:
        pass  # Older PyQt6 versions handle this automatically

    app = QApplication(sys.argv)

    # Set application-wide font with cross-platform fallbacks
    font = QFont()
    font.setFamilies(
        ["Segoe UI", "SF Pro Display", "Ubuntu", "Noto Sans", "sans-serif"]
    )
    font.setPointSize(10)
    app.setFont(font)

    window = DeepfakeApp()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
