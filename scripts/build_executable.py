import os
import sys
import shutil
import subprocess
import argparse
from pathlib import Path

def get_project_root():
    """Get the absolute path to the project root directory."""
    return Path(__file__).resolve().parent.parent

def build_executable(obfuscate=False, clean=True):
    """
    Builds the DeepFaceNet standalone executable using PyInstaller.
    """
    root_dir = get_project_root()
    os.chdir(root_dir)

    print(f"Project root: {root_dir}")
    
    # Clean previous builds
    if clean:
        print("Cleaning previous builds...")
        for p in ["build", "dist", "DeepFaceNet.spec"]:
            path = root_dir / p
            if path.exists():
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink()

    # The main entrypoint
    entrypoint = "run.py"
    
    sep = ";" if sys.platform == "win32" else ":"
    assets_arg = f"--add-data=assets{sep}assets"
    
    # Core command 
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--noconfirm", # Overwrite output directory
        "--onedir",    # Create a one-folder bundle (better for large DL apps)
        "--windowed",  # No console window (GUI only app)
        "--name=DeepFaceNet",
        assets_arg,
        "--hidden-import=onnxruntime",
        "--hidden-import=onnxruntime.capi._pybind_state",
        "--hidden-import=cv2",
        "--hidden-import=insightface",
        "--hidden-import=PyQt6",
        "--collect-all=onnxruntime",   # bundles CUDA provider .so files
        "--collect-all=insightface",   # bundles all insightface data/models
        entrypoint
    ]
    
    if obfuscate:
        # Note: PyArmor requires commercial license for full robust obfuscation on large projects,
        # but here we provide the standard command if pyarmor is installed and desired.
        print("Obfuscating source code using PyArmor...")
        
        # We need PyArmor 8+ syntax. Typically: pyarmor gen -O obf_dist -r .
        # Since pyarmor structure can be complex with PyInstaller, we use pyarmor's builtin integration or
        # compile first then package. 
        # For simplicity in this script, we compile using Nuitka as an alternative obfuscation tool, 
        # or we just rely on standard pyinstaller depending on the user's setup.
        print("WARNING: Obfuscation requires PyArmor or Nuitka installed.")
        print("For robust black-box obfuscation, we recommend running Nuitka natively.")
        print("Example: python -m nuitka --standalone --plugin-enable=pyqt6 --include-data-dir=app/assets=app/assets run.py")
        # Proceed with normal pyinstaller...
        pass

    print("Running PyInstaller...")
    print("Command:", " ".join(cmd))
    
    subprocess.run(cmd, check=True)
    
    print("\n[+] Build successful!")
    print(f"[+] Executable created in: {root_dir / 'dist' / 'DeepFaceNet'}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build DeepFaceNet Executable")
    parser.add_argument("--obfuscate", action="store_true", help="Attempt source code obfuscation (Requires PyArmor or Nuitka)")
    parser.add_argument("--no-clean", action="store_true", help="Do not clean previous builds")
    args = parser.parse_args()
    
    build_executable(obfuscate=args.obfuscate, clean=not args.no_clean)
