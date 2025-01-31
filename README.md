<!-- Image Banner -->
![YoloZone](https://github.com/nushankodikara/yolozone/blob/main/YoloZone.png?raw=true)

# YoloZone

YoloZone is a powerful computer vision toolkit built on YOLOv8, providing intuitive interfaces for object detection, pose estimation, and object tracking. It simplifies complex computer vision tasks with easy-to-use APIs and comprehensive documentation.

[![Documentation](https://img.shields.io/badge/docs-visit%20now-blue)](https://nushankodikara.github.io/yolozone/)
[![GitHub](https://img.shields.io/badge/github-follow-black)](https://github.com/nushankodikara/)
[![LinkedIn](https://img.shields.io/badge/linkedin-connect-blue)](https://www.linkedin.com/in/nushan-kodikara/)

## Features

### Object Detection
- Detect and classify objects in images and videos
- Support for custom models and multiple detection strategies
- Real-time processing capabilities
- Configurable confidence thresholds

### Pose Estimation
- Advanced human pose detection and keypoint analysis
- Real-time pose tracking
- 17-point keypoint detection
- Angle and distance measurements between keypoints
- Support for multiple people in frame

### Object Tracking
- Robust object tracking across video frames
- Motion pattern analysis
- Trajectory data generation
- Line crossing detection
- Multi-object tracking

## Installation

```bash
pip install yolozone
```

## Quick Start

### Object Detection
```python
from yolozone import Objects

# Initialize detector
detector = Objects()

# Detect objects in an image
results = detector.detect('image.jpg')

# Process video stream
detector.process_video('video.mp4', output='output.mp4')
```

### Pose Estimation
```python
from yolozone import Pose

# Initialize pose estimator
pose = Pose()

# Detect poses in an image
results = pose.detect('image.jpg')

# Get keypoints
for detection in results:
    keypoints = pose.get_keypoints(detection)
    print(f"Found person with {len(keypoints)} keypoints")
```

### Object Tracking
```python
from yolozone import Tracker

# Initialize tracker
tracker = Tracker()

# Track objects in video
tracks = tracker.track_video('video.mp4')

# Analyze motion patterns
for track in tracks:
    motion = tracker.analyze_motion(track)
    print(f"Track {track.id}: {motion.pattern}")
```

## Keypoint Reference

The pose estimation module uses the following 17 keypoints:

| ID | Keypoint | ID | Keypoint |
|----|----------|----|----------|
| 0 | Nose | 9 | Left Wrist |
| 1 | Left Eye | 10 | Right Wrist |
| 2 | Right Eye | 11 | Left Hip |
| 3 | Left Ear | 12 | Right Hip |
| 4 | Right Ear | 13 | Left Knee |
| 5 | Left Shoulder | 14 | Right Knee |
| 6 | Right Shoulder | 15 | Left Ankle |
| 7 | Left Elbow | 16 | Right Ankle |
| 8 | Right Elbow | | |

## Documentation

Visit our [comprehensive documentation](https://nushankodikara.github.io/yolozone/) for:
- Detailed API references
- Code examples
- Implementation guides
- Best practices
- Troubleshooting tips

## Requirements

- Python 3.7+
- ultralytics (YOLOv8)
- opencv-python
- numpy

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Author

- **Nushan Kodikara**
  - GitHub: [@nushankodikara](https://github.com/nushankodikara)
  - LinkedIn: [Nushan Kodikara](https://www.linkedin.com/in/nushan-kodikara/)
  - Email: nushankodi@gmail.com

## References

- [Ultralytics YOLOv8](https://docs.ultralytics.com/)
- [YOLOv8 Pose Estimation](https://docs.ultralytics.com/tasks/pose/)