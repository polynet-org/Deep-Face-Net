"""
Setup script for Deep Face Net
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_file.exists():
    with open(requirements_file, "r", encoding="utf-8") as f:
        requirements = [
            line.strip() 
            for line in f 
            if line.strip() and not line.startswith("#")
        ]

setup(
    name="deep-face-net",
    version="2.2.0",
    author="Deep Face Net Contributors",
    author_email="",
    description="A professional real-time face swapping application with GUI and CLI support",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MIDHUNGRAJ/Deep-Face-Net",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Multimedia :: Graphics",
        "Topic :: Multimedia :: Video",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "gpu": ["onnxruntime-gpu>=1.22.0"],
        "virtualcam": ["pyvirtualcam>=0.10.0"],
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ],
    },
    entry_points={
        "console_scripts": [
            "deepfacenet=core.main:main",
        ],
    },
    include_package_data=True,
    keywords="deepfake face-swap face-swapping insightface onnx computer-vision ai",
    project_urls={
        "Bug Reports": "https://github.com/polynet-org/Deep-Face-Net/issues",
        "Source": "https://github.com/polynet-org/Deep-Face-Net",
    },
)
