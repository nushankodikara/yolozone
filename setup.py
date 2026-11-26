from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="yolozone",
    version="0.1.0",
    author="Nushan Kodikara",
    author_email="nushankodi@gmail.com",
    description="A powerful computer vision toolkit built on YOLOv8 for object detection, pose estimation, and tracking",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/nushankodikara/yolozone",
    project_urls={
        "Documentation": "https://nushankodikara.github.io/yolozone/",
        "Source Code": "https://github.com/nushankodikara/yolozone",
        "Bug Tracker": "https://github.com/nushankodikara/yolozone/issues",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.26.4",
        "opencv-python>=4.10.0.84",
        "ultralytics>=8.3.11"
    ],
)
