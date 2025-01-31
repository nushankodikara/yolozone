import sys
import os
from collections import defaultdict

# Add the parent directory to Python path to find the local yolozone module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from yolozone.objects import ObjectDetector
import cv2

def process_tracking_video(video_path, model):
    """Process a video with object tracking"""
    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        return
        
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video: {video_path}")
        return
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Define counting line (horizontal, in middle of frame)
    line_start = (0, frame_height // 2)
    line_end = (frame_width, frame_height // 2)
    
    # Initialize cumulative counts
    total_counts = {
        'up': defaultdict(int),
        'down': defaultdict(int)
    }
    
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
            
            # Detect and track objects
            results = model.detect_objects(frame, conf=0.3, track=True)
            
            # Update tracks
            tracks = model.tracker.update(results)
            
            # Update line crossings
            crossings = model.tracker.update_line_crossings(tracks, line_start, line_end)
            
            # Update cumulative counts
            for direction in ['up', 'down']:
                for cls, count in crossings[direction].items():
                    total_counts[direction][cls] += count
            
            # Draw detections
            frame, detections = model.draw_detections(frame, results, classes=vehicle_classes)
            
            # Draw tracks with trails
            frame = model.tracker.draw_tracks(frame, tracks, draw_trails=True)
            
            # Draw counting line and statistics
            frame = model.tracker.draw_counting_line(frame, line_start, line_end, total_counts)
            
            # Get all tracks as array
            all_tracks = model.tracker.get_all_tracks()
            
            # Display tracking stats
            y_pos = 30
            cv2.putText(frame, f"Frame: {frame_count}", (10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            y_pos += 30
            active_tracks = len([t for t in all_tracks if t['track_id'] not in model.tracker.inactive_tracks])
            cv2.putText(frame, f"Active Tracks: {active_tracks}", (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            y_pos += 30
            cv2.putText(frame, f"Total Tracked: {len(all_tracks)}", (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Display total crossing counts
            y_pos += 40
            cv2.putText(frame, "Total Crossings:", (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            y_pos += 30
            
            # Show up/down totals by class
            for cls in vehicle_classes:
                up_count = total_counts['up'][cls]
                down_count = total_counts['down'][cls]
                if up_count > 0 or down_count > 0:
                    text = f"{cls}: ↑{up_count} ↓{down_count}"
                    cv2.putText(frame, text, (10, y_pos),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    y_pos += 25
            
            # Display the frame
            cv2.imshow(f"Object Tracking - {video_path}", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\nProcessing stopped by user")
                break
            elif key == ord('i'):  # Press 'i' to print track info
                print("\nCurrent Tracks:")
                for track in all_tracks:
                    print(f"ID: {track['track_id']}, Class: {track['class']}, "
                          f"Length: {track['length']}, Displacement: {track['displacement']:.2f}")
                print("\nCumulative Crossing Counts:")
                print("Upward/Rightward:", dict(total_counts['up']))
                print("Downward/Leftward:", dict(total_counts['down']))
            
        except Exception as e:
            print(f"\nError processing frame {frame_count}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    # Print final counts
    print("\nFinal Crossing Counts:")
    print("Upward/Rightward:", dict(total_counts['up']))
    print("Downward/Leftward:", dict(total_counts['down']))
    
    print(f"\nProcessed {frame_count} frames")
    cap.release()
    cv2.destroyAllWindows()

def main():
    print("Initializing YOLOv8 object detector with tracking...")
    model = ObjectDetector(model="yolov8n.pt")
    
    # Process videos
    process_tracking_video("videos/traffic-03.mp4", model)
    process_tracking_video("videos/traffic-04.mp4", model)

if __name__ == "__main__":
    main() 