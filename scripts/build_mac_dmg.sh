#!/bin/bash
# Script to build a DMG installer for macOS.

set -e

# Setup directories
cd "$(dirname "$0")/.."
PROJECT_ROOT=$(pwd)
DIST_DIR="${PROJECT_ROOT}/dist"

echo "[+] Building Python executable with PyInstaller..."
python3 scripts/build_executable.py

# Check for the macOS .app bundle format (often created when --windowed is used on macOS)
APP_BUNDLE="${DIST_DIR}/DeepFaceNet.app"
APP_DIR="${DIST_DIR}/DeepFaceNet"

TARGET_DIR="$APP_DIR"
if [ -d "$APP_BUNDLE" ]; then
    TARGET_DIR="$APP_BUNDLE"
    echo "[+] Found macOS application bundle: $APP_BUNDLE"
else
    echo "[+] Found standalone directory: $APP_DIR"
fi

echo "[+] Packaging into DMG..."

if command -v create-dmg &> /dev/null; then
    echo "[+] Using create-dmg utility..."
    create-dmg \
      --volname "DeepFaceNet" \
      --window-pos 200 120 \
      --window-size 600 400 \
      --icon-size 100 \
      --app-drop-link 400 185 \
      "${DIST_DIR}/DeepFaceNet.dmg" \
      "$TARGET_DIR"
else
    echo "[!] 'create-dmg' not found (can be installed via 'brew install create-dmg')."
    echo "[+] Falling back to native 'hdiutil'..."
    hdiutil create -volname "DeepFaceNet" \
                   -srcfolder "$TARGET_DIR" \
                   -ov -format UDZO \
                   "${DIST_DIR}/DeepFaceNet.dmg"
fi

echo "[+] Process completed! macOS DMG should be available in: ${DIST_DIR}/DeepFaceNet.dmg"
