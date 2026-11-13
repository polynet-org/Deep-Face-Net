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

<img width="1202" height="831" alt="image" src="https://github.com/user-attachments/assets/2fdbd6cd-dcfe-484d-a2a4-6c869eb4a728" />
<img width="1202" height="834" alt="image" src="https://github.com/user-attachments/assets/0b0d92c1-b4a2-4716-bcc4-690170489dc2" />


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

> [!IMPORTANT]
> **Pre-Trained AI Models Required**:
> Deep Face Net needs model weights to perform face swapping (`inswapper_128.onnx` and `buffalo_l`). Due to file size limits, they are not bundled in the download.
>
> When you launch the app, open the **Models** tab to download them with one click, or run `python download_models.py` in your terminal. Face swapping will remain disabled until these models are downloaded.

### Standalone Apps (No Python Setup Needed)

Grab the ready-to-use version for your operating system from the [Releases](https://github.com/polynet-org/Deep-Face-Net/releases) page:

* **Linux**: Download `DeepFaceNet-x86_64.AppImage`, make it executable, and run:
  ```bash
  chmod +x DeepFaceNet-x86_64.AppImage
  ./DeepFaceNet-x86_64.AppImage
  ```
* **Windows**: Download `DeepFaceNet_Setup.exe` and follow the install wizard.
* **macOS**: Download `DeepFaceNet.dmg`, open it, and drag DeepFaceNet into your Applications folder.

---

### Run from Source

#### 1. Clone the Repository

```bash
git clone https://github.com/polynet-org/Deep-Face-Net.git
cd Deep-Face-Net
```

#### 2. Create Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

#### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

> [!TIP]
> `onnxruntime-gpu` is installed by default and automatically utilizes NVIDIA CUDA when available, seamlessly falling back to CPU execution if no compatible GPU is detected.

#### 4. Download Models

The application requires pre-trained models to function. Due to their size, they are not included in the git repository.

You can **automatically download** all required models using the included script:

```bash
python download_models.py
```

This will download:
- **InSwapper 128** (Face Swapping)
- **GFPGAN** (Face Enhancement, ONNX)

If the automated download fails, please refer to [models/README.md](models/README.md) for manual download links.

#### 5. Optional: Virtual Camera Support

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
- `--mode`: Processing mode (`swap` by default, `enhance`, or `swap_enhance`)
- `--fps`: Target FPS for video processing (default: 30)

---

## Configuration

Settings are stored in `core/config.py`:

- **Model Paths**: Configure paths to ONNX and enhancement models
- **Camera Settings**: Default camera index, resolution, and FPS
- **Face Detection**: Detection size and confidence threshold

User preferences (working directory, etc.) are saved in `~/.deepfacenet_settings.json`

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

