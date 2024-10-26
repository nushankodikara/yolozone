from setuptools import setup, find_packages

with open("README.md", "r") as f:
    description = f.read()

setup(
    name="yolozone",
    version="0.0.2",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.26.4",
        "opencv-python>=4.10.0.84",
        "ultralytics>=8.3.11"
    ],
    long_description=description,
    long_description_content_type="text/markdown",
)
