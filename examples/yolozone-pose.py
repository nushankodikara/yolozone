import sys
import os

# Add the parent directory to Python path to find the local yolozone module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from yolozone.pose import PoseDetector
import cv2

def process_video(video_path, model):
    """Process a video file with pose detection"""
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties for display
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    print(f"\nProcessing video: {video_path}")
    print(f"Resolution: {frame_width}x{frame_height}")
    print(f"FPS: {fps}")
    
    frame_count = 0
    
    while True:
        try:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            print(f"\rProcessing frame: {frame_count}", end="")
                
            # Find keypoints in the frame
            keypoints = model.find_keypoints(frame)
            
            # Get points and lines for pose visualization
            points, lines = model.draw_pose(keypoints)  # Returns only 2 values
            
            # Draw the basic pose skeleton
            for point in points:
                cv2.circle(frame, point, 5, (255, 255, 255), 2)
            
            for line in lines:
                cv2.line(frame, line[0], line[1], (255, 255, 255), 2)
            
            # Calculate and display angle between shoulder, elbow and wrist (useful for exercise form)
            angle, text, text_pos = model.angle_between_3_points(keypoints, 5, 7, 9)  # Left arm
            cv2.putText(frame, f"L.Arm: {text}", text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # Calculate distance between hands (wrists) - useful for tracking arm spread
            dist, text, text_pos, p1, p2 = model.distance_between_2_points(keypoints, 9, 10)  # Left to Right wrist
            cv2.putText(frame, f"Hand dist: {text}", text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            cv2.line(frame, p1, p2, (255, 0, 0), 2)
            
            # Calculate angle between arms (useful for symmetry checking)
            angle, text, text_pos = model.angle_between_2_lines(keypoints, 5, 9, 6, 10)  # Angle between arms
            cv2.putText(frame, f"Arms angle: {text}", text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            # Add frame counter to the image
            cv2.putText(frame, f"Frame: {frame_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Display the frame
            cv2.imshow(f"Exercise Analysis - {video_path}", frame)
            
            # Break if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\nProcessing stopped by user")
                break
                
        except Exception as e:
            print(f"\nError processing frame {frame_count}: {str(e)}")
            continue
    
    print(f"\nProcessed {frame_count} frames")
    cap.release()
    cv2.destroyAllWindows()

def main():
    # Initialize the pose detector with a small model for faster processing
    print("Initializing YOLOv8 pose detector...")
    model = PoseDetector(model="yolov8l-pose.pt")
    
    # Process both exercise videos
    process_video("videos/exercise-01.mp4", model)
    process_video("videos/exercise-02.mp4", model)

if __name__ == "__main__":
    main()
