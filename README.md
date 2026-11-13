# Deep Face Net

<div align="center">
<img width="150" height="150" alt="logo" src="https://github.com/user-attachments/assets/53e79f4c-ba3d-4589-b8b6-5f33fbcbd92d" />
</div>

<div align="center">


![Deep Face Net](https://img.shields.io/badge/Deep%20Face%20Net-Advanced%20Face%20Swapping-blue?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.8+-green?style=for-the-badge&logo=python)
![PyQt6](https://img.shields.io/badge/PyQt6-GUI-orange?style=for-the-badge&logo=qt)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-red?style=for-the-badge&logo=opencv)

**A professional real-time face swapping application with both GUI and CLI support**

[Features](#features) • [Installation](#installation) • [Usage](#usage) • [Screenshots](#screenshots) • [License](#license)


</div>

<img width="1321" height="1004" alt="image" src="https://github.com/user-attachments/assets/67496f67-4671-4a2b-b58d-707095f49295" />
<img width="1320" height="1003" alt="image" src="https://github.com/user-attachments/assets/feed4b8a-a8bd-4172-819e-5d7350b73b31" />

---

## Features

- GUI (PyQt6 dark theme) and CLI support
- Live webcam face swapping with multiple camera support
- Offline face swap for images and videos
- Three processing modes: **Swap Only**, **Enhance Only** (GFPGAN, no source needed), **Swap + Enhance**
- Virtual camera output for OBS, Zoom, Discord (optional)
- Mouth mask to preserve natural mouth movements
- Model status indicators in Live and Offline tabs

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

#### Process a Video File

```bash
python run.py --source face.jpg --target video.mp4 --output result.mp4
```

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

#### Additional Options

```bash
python run.py --source face.jpg --target video.mp4 --output result.mp4 --mode swap_enhance
```

**Available Options:**
- `--source, -s`: Path to source image (face to swap from)
- `--target, -t`: Path to target image or video file
- `--output, -o`: Path to output file
- `--webcam`: Use webcam for live face swapping
- `--camera-index`: Camera index for webcam mode (default: 0)
- `--mode`: Processing mode — `swap` (default), `enhance` (no source needed), `swap_enhance`
- `--fps`: Target FPS for video processing (default: 30)

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

---

### BasicSR / GFPGAN Installation Issues

`gfpgan` requires **BasicSR**, which can fail to build from PyPI on many systems (missing headers, PEP 517 errors, CUDA mismatch). The recommended approach is to clone both from source.

**Step 1 — Install BasicSR from the maintained fork** (fixes common build failures):

```bash
git clone https://github.com/MIDHUNGRAJ/BasicSR.git
cd BasicSR
pip install -e .
```

> This fork uses `pyproject.toml` instead of the legacy `setup.py`, avoiding most build errors.

**Step 2 — Install GFPGAN from source:**

```bash
git clone https://github.com/TencentARC/GFPGAN.git
cd GFPGAN
pip install facexlib
pip install -r requirements.txt
python setup.py develop
```

After this, GFPGAN will be available system-wide and the Enhance modes in Deep Face Net will work correctly.

---

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Ethical Considerations

**IMPORTANT**: This software is intended for educational and entertainment purposes only.

- **DO**: Use for learning, research, and creative projects
- **DO**: Obtain consent from individuals whose faces you use
- **DO**: Clearly label deepfake content when sharing
- **DON'T**: Create misleading or deceptive content
- **DON'T**: Use for harassment, fraud, or illegal activities
- **DON'T**: Violate anyone's privacy or rights

**Users are solely responsible for how they use this software. The developers assume no liability for misuse.**

---

## Acknowledgments

This project uses the following open-source technologies:

- [InsightFace](https://github.com/deepinsight/insightface) - Face analysis and recognition
- [InSwapper](https://github.com/haofanwang/inswapper) - Face swapping model
- [GFPGAN](https://github.com/TencentARC/GFPGAN) - Face enhancement model
- [BasicSR](https://github.com/MIDHUNGRAJ/BasicSR) - Super-resolution backbone (maintained fork)
- [PyQt6](https://www.riverbankcomputing.com/software/pyqt/) - GUI framework
- [OpenCV](https://opencv.org/) - Computer vision library
- [ONNX Runtime](https://onnxruntime.ai/) - Model inference

---

## Contact

For questions, suggestions, or issues, please open an issue on GitHub.

