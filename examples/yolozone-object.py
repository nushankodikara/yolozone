import sys
import os

# Add the parent directory to Python path to find the local yolozone module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from yolozone.objects import ObjectDetector
import cv2

def process_traffic_video(video_path, model):
    """Process a traffic video with object detection"""
    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        return
        
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video: {video_path}")
        return
    
    # Get video properties for display
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    print(f"\nProcessing video: {video_path}")
    print(f"Resolution: {frame_width}x{frame_height}")
    print(f"FPS: {fps}")
    
    frame_count = 0
    vehicle_classes = ['car', 'truck', 'bus', 'motorcycle']
    
    while True:
        try:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            print(f"\rProcessing frame: {frame_count}", end="")
            
            # Detect objects in the frame
            results = model.detect_objects(frame, conf=0.3)
            
            # Draw detections
            frame, detections = model.draw_detections(frame, results, classes=vehicle_classes)
            
            # Count vehicles
            counts = model.count_objects(results, classes=vehicle_classes)
            
            # Display counts on frame
            y_pos = 30
            cv2.putText(frame, f"Frame: {frame_count}", (10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            y_pos += 30
            for vehicle, count in counts.items():
                text = f"{vehicle}: {count}"
                cv2.putText(frame, text, (10, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                y_pos += 30
            
            # Draw center points for all detections
            for _, _, box in detections:
                x1, y1, x2, y2 = box
                center = (int((x1 + x2)/2), int((y1 + y2)/2))
                cv2.circle(frame, center, 4, (0, 0, 255), -1)
            
            # Display the frame
            cv2.imshow(f"Traffic Analysis - {video_path}", frame)
            
            # Break if 'q' is pressed
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\nProcessing stopped by user")
                break
            
        except Exception as e:
            print(f"\nError processing frame {frame_count}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\nProcessed {frame_count} frames")
    cap.release()
    cv2.destroyAllWindows()

def main():
    # Initialize the object detector with a larger model for better accuracy
    print("Initializing YOLOv8 object detector...")
    model = ObjectDetector(model="yolov8n.pt")
    
    # Process traffic videos
    process_traffic_video("videos/traffic-01.mp4", model)
    process_traffic_video("videos/traffic-02.mp4", model)

if __name__ == "__main__":
    main() 