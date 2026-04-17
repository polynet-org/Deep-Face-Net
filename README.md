# Deep Face Net

<div align="center">
<img width="150" height="150" alt="icon" src="https://github.com/user-attachments/assets/8fa8358b-3cb4-4cf9-aa63-15004191cd26" />

</div>

<div align="center">


![Deep Face Net](https://img.shields.io/badge/Deep%20Face%20Net-Advanced%20Face%20Swapping-blue?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.8+-green?style=for-the-badge&logo=python)
![PyQt6](https://img.shields.io/badge/PyQt6-GUI-orange?style=for-the-badge&logo=qt)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-red?style=for-the-badge&logo=opencv)

**Real-time face swapping application with both GUI and CLI support**

[Features](#features) • [Installation](#installation) • [Usage](#usage) • [Screenshots](#screenshots) • [License](#license)


</div>


---

## Features

- GUI (PyQt6 dark theme) and CLI support
- Live webcam face swapping with multiple camera support
- Virtual camera output for OBS, Zoom, Discord (optional)
- Mouth mask to preserve natural mouth movements
- Live model-status indicators

---

## 💎 Premium Version

This is the free version. Deep Face Net Premium unlocks advanced capabilities designed for professionals.

### Premium Features
- 🔁 Multi-face swapping in videos
- 🎬 High-quality studio-level output
- ⚡ Optimized performance pipeline
- 🧠 Advanced face tracking and consistency

👉 Get Premium: https://www.polynet.online

---

## Requirements

- **Python**: 3.8 or higher
- **Operating System**: Linux, macOS, or Windows
- **GPU**: CUDA-compatible GPU recommended for better performance (optional)
- **Webcam**: Required for live face swapping features

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/MIDHUNGRAJ/Deep-Face-Net.git
cd Deep-Face-Net
```

### 2. Create Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

> [!TIP]
> For **GPU acceleration**, install `onnxruntime-gpu` instead of `onnxruntime`.
> If you're on **CUDA 13** (common on Arch Linux / CachyOS), see the [Troubleshooting](#-troubleshooting) section below.

### 4. Download Models

The application requires several pre-trained models to function. Due to their large size, they are not included in the git repository.

You can **automatically download** all required models using the included script:

```bash
python download_models.py
```

This will download:
- **InSwapper 128** (Face Swapping)
- **GFPGAN** (Face Enhancement)

If the automated download fails, please refer to [models/README.md](models/README.md) for manual download links.

### 5. Optional: Virtual Camera Support

For virtual camera output (to use with OBS, Zoom, etc.):

```bash
pip install pyvirtualcam
```

**Linux users** also need to install v4l2loopback:
```bash
sudo apt-get install v4l2loopback-dkms
sudo modprobe v4l2loopback devices=1
```

---

## Usage

### GUI Mode (Default)

Launch the graphical interface:

```bash
python run.py
```

#### Live Camera Tab
1. Select a source image containing the face you want to use
2. Choose your camera from the dropdown
3. Click "Start Camera"
4. Click "Enable Face Swap" to begin swapping
5. Optional: Enable "Mouth Mask" or "Virtual Camera"

#### Offline Processing Tab
1. Select a source face image
2. Drag & drop or browse for a target video/image
3. Click "Start Processing"
4. Wait for completion and view/open the result

### CLI Mode

The CLI mode is activated when you provide command-line arguments.


#### Process an Image File

```bash
python run.py --source face.jpg --target photo.jpg --output result.jpg
```

#### Live Webcam Mode (CLI)

```bash
python run.py --source face.jpg --webcam
```

Use a specific camera:
```bash
python run.py --source face.jpg --webcam --camera-index 1
```

**Available Options:**
- `--source, -s`: Path to source image (face to swap from)
- `--target, -t`: Path to target image file
- `--output, -o`: Path to output file
- `--webcam`: Use webcam for live face swapping
- `--camera-index`: Camera index for webcam mode (default: 0)

---

## Configuration

Settings are stored in `core/config.py`:

- **Model Paths**: Configure paths to ONNX and enhancement models
- **Camera Settings**: Default camera index, resolution, and FPS
- **Face Detection**: Detection size and confidence threshold

User preferences (working directory, etc.) are saved in `~/.deepfacenet_settings.json`

---

## Troubleshooting

### GPU / CUDA 13 Support

The default `onnxruntime-gpu` package on PyPI may not support **CUDA 13** yet. If face swapping fails or falls back to CPU on systems running CUDA 13 (e.g., **Arch Linux / CachyOS**), install the nightly build instead:

```bash
# Install required dependencies
pip install coloredlogs flatbuffers numpy packaging protobuf sympy

# Install onnxruntime-gpu nightly with CUDA 13 support
pip install --pre --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ort-cuda-13-nightly/pypi/simple/ onnxruntime-gpu
```

## License

This project is licensed under the AGPL-3.0 License - see the [LICENSE](LICENSE) file for details.

---

## Ethical Use

Only use this software with proper consent. Do not create deceptive, harmful, or non-consensual content. Users are solely responsible for how they use this tool.

---

## Acknowledgments

This project uses the following open-source technologies:

- [InsightFace](https://github.com/deepinsight/insightface) - Face analysis and recognition
- [InSwapper](https://github.com/haofanwang/inswapper) - Face swapping model
- [GFPGAN](https://github.com/TencentARC/GFPGAN) - Face enhancement model
- [PyQt6](https://www.riverbankcomputing.com/software/pyqt/) - GUI framework
- [OpenCV](https://opencv.org/) - Computer vision library
- [ONNX Runtime](https://onnxruntime.ai/) - Model inference

---

## Contact

For questions, suggestions, or issues, please open an issue on GitHub.

