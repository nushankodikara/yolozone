from setuptools import setup, find_packages

setup(
    name="yolozone",
    version="0.0.1",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.26.4",
        "opencv-python>=4.10.0.84",
        "ultralytics>=8.3.11"
    ],
)
