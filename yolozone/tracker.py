'''
YoloZone Tracker Module
Developed by:
    - Nushan Kodikara
Contact:
    - nushankodi@gmail.com
'''
import numpy as np
import cv2
from collections import defaultdict, Counter

class ObjectTracker:
    def __init__(self):
        """Initialize the object tracker"""
        self.tracks = defaultdict(list)  # Dictionary to store track history
        self.max_track_length = 30  # Maximum number of points to keep in track history
        self.line_crossings = defaultdict(set)  # Track IDs that have crossed each line
        self.inactive_tracks = set()  # Track IDs that are no longer active
        self.max_inactive_frames = 30  # Number of frames before removing inactive tracks
        self.next_track_id = 0  # Counter for assigning new track IDs
        self.frame_count = 0  # Counter for frames processed
        
        # Tracking stability parameters
        self.max_movement_threshold = 100  # Maximum pixel movement between frames
        self.min_detection_confidence = 0.3  # Minimum confidence for detection
        self.min_track_hits = 3  # Minimum consecutive detections to confirm track
        self.max_size_change = 0.5  # Maximum allowed size change ratio between frames
        self.max_frames_to_match = 30  # Maximum frame difference for matching
        self.iou_threshold = 0.3  # Minimum IOU for box matching
        
    def assign_track_id(self):
        """Get next available track ID"""
        track_id = self.next_track_id
        self.next_track_id += 1
        return track_id
    
    def calculate_box_size(self, box):
        """Calculate box area"""
        x1, y1, x2, y2 = box
        return (x2 - x1) * (y2 - y1)
    
    def is_valid_movement(self, prev_center, current_center, prev_box, current_box):
        """Check if movement between frames is valid"""
        # Check distance movement
        distance = np.linalg.norm(np.array(current_center) - np.array(prev_center))
        if distance > self.max_movement_threshold:
            return False
            
        # Check size change
        prev_size = self.calculate_box_size(prev_box)
        current_size = self.calculate_box_size(current_box)
        if prev_size > 0:
            size_change = abs(current_size - prev_size) / prev_size
            if size_change > self.max_size_change:
                return False
                
        return True
    
    def calculate_iou(self, box1, box2):
        """Calculate Intersection over Union between two boxes"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculate intersection coordinates
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
            
        # Calculate areas
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def find_best_match(self, current_box, current_center, class_name, tracks):
        """Find best matching track based on position, time, and class"""
        best_score = float('-inf')
        best_track_id = None
        
        for track_id, track in tracks.items():
            if not track or track[-1]['last_seen'] > self.max_frames_to_match:
                continue
                
            prev_entry = track[-1]
            
            # Only match same class
            if prev_entry['class'] != class_name:
                continue
            
            # Calculate position score
            prev_center = prev_entry['center']
            distance = np.linalg.norm(np.array(current_center) - np.array(prev_center))
            position_score = 1.0 / (1.0 + distance)
            
            # Calculate temporal score
            frames_diff = self.frame_count - prev_entry['frame']
            temporal_score = 1.0 / (1.0 + frames_diff)
            
            # Calculate IOU score
            iou_score = self.calculate_iou(current_box, prev_entry['box'])
            
            # Calculate final score (weighted combination)
            score = (0.4 * position_score + 
                    0.3 * temporal_score + 
                    0.3 * iou_score)
            
            if score > best_score and iou_score > self.iou_threshold:
                best_score = score
                best_track_id = track_id
        
        return best_track_id, best_score
    
    def update(self, results, min_hits=3):
        """Update tracks with new detections"""
        if not hasattr(results, 'boxes') or not hasattr(results.boxes, 'id'):
            raise ValueError("No tracking information found. Make sure tracking is enabled in detection.")
        
        self.frame_count += 1
        current_tracks = {}
        boxes = results.boxes.cpu().numpy()
        active_track_ids = set()
        
        # Filter detections by confidence
        valid_boxes = [box for box in boxes if float(box.conf) >= self.min_detection_confidence]
        
        # Process each detection
        for box in valid_boxes:
            class_id = int(box.cls)
            class_name = results.names[class_id]
            conf = float(box.conf)
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            current_center = (int((x1 + x2)/2), int((y1 + y2)/2))
            current_box = (x1, y1, x2, y2)
            
            # Find best matching track
            matched_track_id, match_score = self.find_best_match(
                current_box, current_center, class_name, self.tracks)
            
            # If no good match found, create new track
            if matched_track_id is None:
                matched_track_id = self.assign_track_id()
            
            active_track_ids.add(matched_track_id)
            
            # Update track history
            self.tracks[matched_track_id].append({
                'center': current_center,
                'box': current_box,
                'class': class_name,
                'conf': conf,
                'last_seen': 0,
                'frame': self.frame_count,
                'match_score': match_score if matched_track_id else 1.0
            })
            
            # Limit track history length
            if len(self.tracks[matched_track_id]) > self.max_track_length:
                self.tracks[matched_track_id].pop(0)
            
            # Add to current tracks if it has enough hits
            if len(self.tracks[matched_track_id]) >= min_hits:
                current_tracks[matched_track_id] = {
                    'center': current_center,
                    'box': current_box,
                    'class': class_name,
                    'conf': conf,
                    'history': self.tracks[matched_track_id]
                }
        
        # Update inactive tracks
        for track_id in list(self.tracks.keys()):
            if track_id not in active_track_ids:
                if track_id not in self.inactive_tracks:
                    self.inactive_tracks.add(track_id)
                if self.tracks[track_id]:
                    last_entry = self.tracks[track_id][-1]
                    last_entry['last_seen'] += 1
        
        return current_tracks
    
    def draw_tracks(self, frame, tracks, draw_labels=True, draw_trails=True):
        """Draw tracked objects and their trails"""
        # Draw trails first
        if draw_trails:
            for track_id, track_info in tracks.items():
                points = [p['center'] for p in track_info['history']]
                if len(points) > 1:
                    # Draw trail as connected dots with decreasing size
                    for i in range(len(points) - 1):
                        # Calculate dot size (larger for more recent points)
                        size = max(2, int(5 * (i + 1) / len(points)))
                        cv2.circle(frame, points[i], size, (0, 0, 255), -1)
        
        # Draw current boxes and labels
        for track_id, track_info in tracks.items():
            x1, y1, x2, y2 = track_info['box']
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw current position as a larger dot
            cv2.circle(frame, track_info['center'], 6, (0, 0, 255), -1)
            
            if draw_labels:
                # Draw class and confidence at top
                label = f"{track_info['class']} {track_info['conf']:.2f}"
                text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                cv2.rectangle(frame, (x1, y1-text_size[1]-10), 
                            (x1+text_size[0], y1), (0, 255, 0), -1)
                cv2.putText(frame, label, (x1, y1-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                
                # Draw track ID at bottom
                id_label = f"ID: {track_id}"
                id_text_size = cv2.getTextSize(id_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                cv2.rectangle(frame, (x1, y2), 
                            (x1+id_text_size[0], y2+id_text_size[1]+10), (0, 255, 0), -1)
                cv2.putText(frame, id_label, (x1, y2+id_text_size[1]+5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                
                # Draw YOLO ID if available (for debugging)
                if 'yolo_id' in track_info['history'][-1]:
                    yolo_id = track_info['history'][-1]['yolo_id']
                    yolo_label = f"YOLO: {yolo_id}"
                    yolo_text_size = cv2.getTextSize(yolo_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                    cv2.rectangle(frame, (x2-yolo_text_size[0], y2), 
                                (x2, y2+yolo_text_size[1]+10), (0, 255, 0), -1)
                    cv2.putText(frame, yolo_label, (x2-yolo_text_size[0], y2+yolo_text_size[1]+5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        return frame
    
    def get_all_tracks(self):
        """Get all tracked objects as an array
        
        Returns:
            tracks_array: List of dictionaries containing track information
        """
        tracks_array = []
        for track_id, track in self.tracks.items():
            if track:  # If track has any points
                track_info = self.get_track_info(track_id)
                if track_info:
                    tracks_array.append(track_info)
        
        return tracks_array
    
    def get_track_info(self, track_id):
        """Get information about a specific track
        
        Args:
            track_id: ID of the track to get info for
            
        Returns:
            track_info: Dictionary containing track history and statistics
        """
        if track_id not in self.tracks:
            return None
            
        track = self.tracks[track_id]
        if not track:
            return None
            
        # Calculate basic statistics
        centers = [p['center'] for p in track]
        boxes = [p['box'] for p in track]
        classes = [p['class'] for p in track]
        
        # Calculate displacement
        if len(centers) >= 2:
            start = np.array(centers[0])
            end = np.array(centers[-1])
            displacement = np.linalg.norm(end - start)
        else:
            displacement = 0
            
        return {
            'track_id': track_id,
            'length': len(track),
            'current_pos': centers[-1] if centers else None,
            'start_pos': centers[0] if centers else None,
            'displacement': displacement,
            'class': max(set(classes), key=classes.count),  # most common class
            'history': track
        }
    
    def detect_line_crossing(self, line_start, line_end, track_info):
        """Detect if a track has crossed a line
        
        Args:
            line_start: (x, y) tuple of line start point
            line_end: (x, y) tuple of line end point
            track_info: Track information dictionary
            
        Returns:
            crossed: Boolean indicating if line was crossed in this update
            direction: 1 for upward/rightward crossing, -1 for downward/leftward, 0 for no crossing
        """
        if len(track_info['history']) < 2:
            return False, 0
            
        # Get last two points of track
        p1 = track_info['history'][-2]['center']
        p2 = track_info['history'][-1]['center']
        
        # Line segments
        x1, y1 = line_start
        x2, y2 = line_end
        x3, y3 = p1
        x4, y4 = p2
        
        # Calculate intersection
        denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if denominator == 0:  # Lines are parallel
            return False, 0
            
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denominator
        u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denominator
        
        if 0 <= t <= 1 and 0 <= u <= 1:  # Lines intersect
            # Determine crossing direction (positive = upward/rightward, negative = downward/leftward)
            direction = 1 if (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1) < 0 else -1
            return True, direction
            
        return False, 0
    
    def update_line_crossings(self, tracks, line_start, line_end):
        """Update and count line crossings
        
        Args:
            tracks: Dictionary of current tracks
            line_start: (x, y) tuple of line start point
            line_end: (x, y) tuple of line end point
            
        Returns:
            crossings: Dictionary with counts of crossings by direction and class
        """
        line_id = (line_start, line_end)
        crossings = {
            'up': Counter(),    # or rightward
            'down': Counter(),  # or leftward
            'total': Counter()
        }
        
        for track_id, track_info in tracks.items():
            crossed, direction = self.detect_line_crossing(line_start, line_end, track_info)
            
            if crossed and track_id not in self.line_crossings[line_id]:
                self.line_crossings[line_id].add(track_id)
                
                if direction > 0:
                    crossings['up'][track_info['class']] += 1
                else:
                    crossings['down'][track_info['class']] += 1
                crossings['total'][track_info['class']] += 1
        
        return crossings
    
    def draw_counting_line(self, frame, line_start, line_end, counts):
        """Draw counting line and crossing statistics
        
        Args:
            frame: Image to draw on
            line_start: (x, y) tuple of line start point
            line_end: (x, y) tuple of line end point
            counts: Dictionary of crossing counts
            
        Returns:
            frame: Frame with line and statistics drawn
        """
        # Draw the counting line
        cv2.line(frame, line_start, line_end, (0, 255, 255), 2)
        
        # Draw arrows indicating directions
        mid_x = (line_start[0] + line_end[0]) // 2
        mid_y = (line_start[1] + line_end[1]) // 2
        
        # Draw direction arrows and labels
        arrow_length = 30
        
        # Draw up/down arrows
        cv2.arrowedLine(frame, (mid_x - 20, mid_y - arrow_length), 
                       (mid_x - 20, mid_y), (0, 255, 255), 2)
        cv2.arrowedLine(frame, (mid_x + 20, mid_y), 
                       (mid_x + 20, mid_y - arrow_length), (0, 255, 255), 2)
        
        # Draw counts with better positioning and labels
        y_offset = -100  # Start above the line
        cv2.putText(frame, "Crossing Counts:", (mid_x + 50, mid_y + y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        y_offset += 30
        
        # Show upward counts (bottom to top)
        for cls, count in counts['up'].items():
            text = f"{cls} ↑: {count}"
            cv2.putText(frame, text, (mid_x + 50, mid_y + y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            y_offset += 25
        
        y_offset += 10  # Add space between up and down counts
        
        # Show downward counts (top to bottom)
        for cls, count in counts['down'].items():
            text = f"{cls} ↓: {count}"
            cv2.putText(frame, text, (mid_x + 50, mid_y + y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            y_offset += 25
        
        return frame 