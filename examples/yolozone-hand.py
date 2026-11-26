import sys
import os
import cv2

# Add parent directory to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from yolozone import HandTracker, PoseDetector


def process_hand_video(video_path: str):
    """Process video with 21-point hand & finger tracking."""
    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        return

    cap = cv2.VideoCapture(video_path)
    tracker = HandTracker()
    pose = PoseDetector("yolov8s-pose.pt")

    frame_count = 0
    print(f"\nProcessing Hand & Finger Tracking: {video_path}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        keypoints = pose.find_keypoints(frame)
        tracked_hands = tracker.update(frame, keypoints=keypoints)

        # Draw 21-point finger skeleton, gesture HUD, and trails
        frame = tracker.draw_tracks(frame, tracked_hands, draw_skeleton=True, draw_trails=True, draw_gestures=True)

        cv2.putText(frame, f"Frame: {frame_count}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow("Hand & Finger Tracking - YoloZone", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    process_hand_video("videos/hand-gesture-01.mp4")
