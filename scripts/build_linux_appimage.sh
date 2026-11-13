#!/bin/bash
# Script to build an AppImage for Linux, producing a single standalone executable.

set -e

# Setup directories
cd "$(dirname "$0")/.."
PROJECT_ROOT=$(pwd)
BUILD_DIR="${PROJECT_ROOT}/build"
DIST_DIR="${PROJECT_ROOT}/dist"
APP_DIR="${DIST_DIR}/DeepFaceNet.AppDir"

echo "[+] Building Python executable with PyInstaller..."
python3 scripts/build_executable.py

echo "[+] Preparing AppDir..."
rm -rf "$APP_DIR"
mkdir -p "$APP_DIR/usr/bin"
mkdir -p "$APP_DIR/usr/share/applications"
mkdir -p "$APP_DIR/usr/share/icons/hicolor/256x256/apps"

# Move the dist output into the AppDir
cp -r "${DIST_DIR}/DeepFaceNet/"* "$APP_DIR/usr/bin/"

# Copy icon
if [ -f "assets/icon.png" ]; then
    cp assets/icon.png "$APP_DIR/deepfacenet.png"
    cp assets/icon.png "$APP_DIR/usr/share/icons/hicolor/256x256/apps/deepfacenet.png"
else
    # Fallback if no specific png icon is available
    echo "Warning: assets/icon.png not found for AppImage icon!"
    touch "$APP_DIR/deepfacenet.png"
fi

# Create .desktop file
cat > "$APP_DIR/deepfacenet.desktop" << EOF
[Desktop Entry]
Name=DeepFaceNet
Exec=DeepFaceNet
Icon=deepfacenet
Type=Application
Categories=Graphics;Video;
Comment=Real-time face swapping application
EOF

# Create AppRun script
cat > "$APP_DIR/AppRun" << 'EOF'
#!/bin/bash
HERE="$(dirname "$(readlink -f "${0}")")"
export PATH="${HERE}/usr/bin:${PATH}"
export LD_LIBRARY_PATH="${HERE}/usr/lib:${LD_LIBRARY_PATH}"
exec "${HERE}/usr/bin/DeepFaceNet" "$@"
EOF

chmod +x "$APP_DIR/AppRun"

# Download linuxdeploy plugin for AppImage if we don't have python-appimage tools
# Using predefined appimagetool 
export ARCH=x86_64
if ! command -v appimagetool &> /dev/null; then
    echo "[!] appimagetool not found. Downloading..."
    wget -c "https://github.com/AppImage/AppImageKit/releases/download/continuous/appimagetool-x86_64.AppImage" -O appimagetool
    chmod +x appimagetool
    echo "[+] Running appimagetool..."
    ./appimagetool --appimage-extract-and-run "$APP_DIR" "${DIST_DIR}/DeepFaceNet-x86_64.AppImage"
    rm -f appimagetool
else
    echo "[+] Running appimagetool..."
    appimagetool "$APP_DIR" "${DIST_DIR}/DeepFaceNet-x86_64.AppImage"
fi

echo "[+] AppImage created successfully in dist/DeepFaceNet-x86_64.AppImage"
