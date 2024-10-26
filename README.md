# YoloZone

YoloZone is a helper library for detecting and analyzing human poses in images and videos using the YOLO models. It provides simple functions for finding keypoints, angles, and distances between points in a pose.

## Installation

```bash
pip install yolozone
```

## Keypoints / Landmarks

We use the YOLO model for pose landmarks without any changes. The model returns 17 keypoints / landmarks.

| Keypoint | Description |
| -------- | -------- |
| 0 | Nose |
| 1 | Left Eye |
| 2 | Right Eye |
| 3 | Left Ear |
| 4 | Right Ear |
| 5 | Left Shoulder |
| 6 | Right Shoulder |
| 7 | Left Elbow |
| 8 | Right Elbow |
| 9 | Left Wrist |
| 10 | Right Wrist |
| 11 | Left Hip |
| 12 | Right Hip |
| 13 | Left Knee |
| 14 | Right Knee |
| 15 | Left Ankle |
| 16 | Right Ankle |

## Usage

```python
from yolozone import PoseDetector
import cv2

model = PoseDetector(model="yolov8n-pose.pt") # Initialize model
cap = cv2.VideoCapture(0) # Initialize video capture for webcam / cameras
# cap = cv2.VideoCapture("test/video.mp4") # Initialize video capture for video

while True:
    try:
        ret, frame = cap.read() # Read frame
        
        # Find keypoints
        keypoints = model.find_keypoints(frame, device="mps")

        # Get Points and Lines to Draw pose
        points, lines = model.draw_pose(keypoints)

        # Draw points
        for point in points:
            cv2.circle(frame, point, 5, (255, 255, 255), 2)
        
        # Draw lines
        for line in lines:
            cv2.line(frame, line[0], line[1], (255, 255, 255), 2) # line[0] and line[1] are starting and ending points of the line
        
        # Find angle between Left Shoulder (5) , Left Elbow (7) and Left Wrist (9)
        angle, text, text_position = model.angle_between_3_points(keypoints, 5, 7, 9)
        cv2.putText(frame, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.circle(frame, text_position, 5, (255, 0, 0), -1)

        # Find distance between Left Shoulder (5) and Left Wrist (9)
        distance, text, text_position, pointOutA, pointOutB = model.distance_between_2_points(keypoints, 5, 9)
        cv2.putText(frame, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.line(frame, pointOutA, pointOutB, (0, 255, 0), 2)

        # Find distance between Right Shoulder (6) and Right Wrist (10)
        distance, text, text_position, pointOutC, pointOutD = model.distance_between_2_points(keypoints, 6, 10)
        cv2.putText(frame, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.line(frame, pointOutC, pointOutD, (0, 255, 0), 2)

        # Find the angle between Left Shoulder (5), Left Wrist (9) and Right Shoulder (6), Right Wrist (10)
        angle, text, text_position = model.angle_between_2_lines(keypoints, 5, 9, 6, 10)
        cv2.putText(frame, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    except Exception as e:
        print(e)
        pass

    cv2.imshow("frame", frame)
    cv2.waitKey(1)
```

## Features

### Pose Detection (WIP)
- [x] Keypoints / Landmarks
- [x] Angles between 3 points
- [x] Distance between 2 points
- [x] Angle between 2 lines

### Object Detection (WIP)

### Face Detection (WIP)

## References

- [Ultralytics YOLOv8](https://docs.ultralytics.com/tasks/pose/)