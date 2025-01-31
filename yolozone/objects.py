'''
YoloZone Objects Module
Developed by:
    - Nushan Kodikara
Contact:
    - nushankodi@gmail.com
'''
import numpy as np
import cv2
from ultralytics import YOLO
from .tracker import ObjectTracker

class ObjectDetector:
    def __init__(self, model="yolov8s.pt"):
        """Initialize the object detector with a YOLO model"""
        self.model = model
        self.detector = YOLO(self.model)
        self.tracker = ObjectTracker()
        
    def detect_objects(self, img, device="cpu", conf=0.25, track=False):
        """Detect objects in an image
        
        Args:
            img: Input image
            device: Device to run inference on ('cpu', 'cuda', 'mps')
            conf: Confidence threshold (0-1)
            track: Enable object tracking
            
        Returns:
            results: Detection results containing boxes, classes, and confidence scores
        """
        results = self.detector.track(img, device=device, conf=conf) if track else self.detector(img, device=device, conf=conf)
        return results[0]
    
    def get_boxes(self, results):
        """Get bounding boxes from results
        
        Returns:
            boxes: List of [x1, y1, x2, y2, confidence, class_id]
        """
        return results.boxes.cpu().numpy()
    
    def draw_detections(self, img, results, classes=None, color=(0, 255, 0), thickness=2):
        """Draw detection boxes and labels on image
        
        Args:
            img: Image to draw on
            results: Detection results from detect_objects()
            classes: List of class names to filter (optional)
            color: Box and text color (B,G,R)
            thickness: Line thickness
            
        Returns:
            img: Image with detections drawn
            detections: List of (class_name, confidence, box) tuples
        """
        detections = []
        
        if results.boxes is None or len(results.boxes) == 0:
            return img, detections
            
        boxes = results.boxes.cpu().numpy()
        
        for box in boxes:
            # Get box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Get confidence and class
            conf = float(box.conf)
            class_id = int(box.cls)
            class_name = results.names[class_id]
            
            # Filter by class if specified
            if classes and class_name.lower() not in classes:
                continue
                
            # Draw box
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
            
            # Draw label
            label = f"{class_name} {conf:.2f}"
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, thickness)[0]
            cv2.rectangle(img, (x1, y1-text_size[1]-10), (x1+text_size[0], y1), color, -1)
            cv2.putText(img, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), thickness)
            
            detections.append((class_name, conf, (x1, y1, x2, y2)))
            
        return img, detections
    
    def count_objects(self, results, classes=None):
        """Count number of detected objects by class
        
        Args:
            results: Detection results
            classes: List of class names to filter (optional)
            
        Returns:
            counts: Dictionary of {class_name: count}
        """
        counts = {}
        
        if results.boxes is None or len(results.boxes) == 0:
            return counts
            
        boxes = results.boxes.cpu().numpy()
        
        for box in boxes:
            class_id = int(box.cls)
            class_name = results.names[class_id]
            
            if classes and class_name.lower() not in classes:
                continue
                
            counts[class_name] = counts.get(class_name, 0) + 1
            
        return counts
    
    def get_object_centers(self, results):
        """Get center points of all detected objects
        
        Returns:
            centers: Dictionary of {class_name: [(x,y), ...]}
        """
        centers = {}
        boxes = self.get_boxes(results)
        names = results.names
        
        for box in boxes:
            x1, y1, x2, y2 = map(int, box[:4])
            class_id = int(box[5])
            class_name = names[class_id]
            
            center = (int((x1 + x2)/2), int((y1 + y2)/2))
            
            if class_name not in centers:
                centers[class_name] = []
            centers[class_name].append(center)
            
        return centers 