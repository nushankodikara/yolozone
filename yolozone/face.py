'''
YoloZone Face Analysis & Mathematical Plane Module
Provides 3D head pose and mathematical face plane estimation, dense facial landmarks,
geometric expression tracking, and multi-emotion classification.
Developed by:
    - Nushan Kodikara
Contact:
    - nushankodi@gmail.com
'''
import numpy as np
import cv2
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple, Any


class FaceDetector:
    """Advanced Face Detector with 3D plane projection, dense landmarks, and emotion analysis."""

    # 3D Reference Face Model Coordinates (Anthropometric canonical points in mm)
    MODEL_POINTS_3D = np.array([
        (0.0, 0.0, 0.0),             # Nose tip
        (0.0, -330.0, -65.0),        # Chin
        (-225.0, 170.0, -135.0),     # Left eye outer corner
        (225.0, 170.0, -135.0),      # Right eye outer corner
        (-150.0, -150.0, -125.0),    # Left mouth corner
        (150.0, -150.0, -125.0),     # Right mouth corner
        (0.0, 220.0, -100.0)         # Forehead / Glabella
    ], dtype=np.float64)

    # COCO Pose keypoint indices for pose-guided fallback
    NOSE = 0
    LEFT_EYE = 1
    RIGHT_EYE = 2
    LEFT_EAR = 3
    RIGHT_EAR = 4

    def __init__(self, model=None, use_facemesh: bool = True, max_faces: int = 5):
        """Initialize FaceDetector.

        Args:
            model: Optional YOLO face/pose model path or PoseDetector instance.
            use_facemesh: If True and mediapipe is installed, uses MediaPipe FaceMesh.
            max_faces: Maximum number of faces to detect.
        """
        self.max_faces = max_faces
        self.mp_facemesh = None
        self.custom_model = None
        self.pose_detector = None

        if use_facemesh:
            try:
                import mediapipe as mp
                self.mp_facemesh = mp.solutions.face_mesh.FaceMesh(
                    static_image_mode=False,
                    max_num_faces=max_faces,
                    refine_landmarks=True,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5
                )
            except (ImportError, Exception):
                self.mp_facemesh = None

        if isinstance(model, str) and not ("pose" in model.lower()):
            from ultralytics import YOLO
            self.custom_model = YOLO(model)
        elif model is not None and hasattr(model, 'find_keypoints'):
            self.pose_detector = model
        else:
            from .pose import PoseDetector
            pose_model_name = model if isinstance(model, str) else "yolov8s-pose.pt"
            self.pose_detector = PoseDetector(model=pose_model_name)

    def detect_faces(self, img: np.ndarray, keypoints=None, device="cpu", conf=0.25) -> List[Dict[str, Any]]:
        """Detect faces, estimate 3D mathematical plane, extract landmarks, and evaluate emotions."""
        h, w = img.shape[:2]
        faces_output = []

        # Strategy A: MediaPipe FaceMesh (dense 468+ landmarks)
        if self.mp_facemesh is not None:
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mp_results = self.mp_facemesh.process(rgb_img)

            if mp_results.multi_face_landmarks:
                for idx, face_landmarks in enumerate(mp_results.multi_face_landmarks):
                    lms_2d = [(int(lm.x * w), int(lm.y * h)) for lm in face_landmarks.landmark]
                    face_dict = self._process_dense_face_landmarks(img, lms_2d, idx, (h, w))
                    faces_output.append(face_dict)

            if faces_output:
                return faces_output

        # Strategy B: Pose-Guided Canonical Landmark and 3D Plane Synthesis
        if keypoints is None and self.pose_detector is not None:
            keypoints = self.pose_detector.find_keypoints(img, device=device)

        if keypoints is not None and hasattr(keypoints, 'xy') and len(keypoints.xy) > 0:
            for s_idx in range(len(keypoints.xy)):
                pts = keypoints.xy[s_idx]
                if pts.shape[0] < 5:
                    continue

                face_dict = self._synthesize_face_from_pose(img, pts, s_idx, (h, w))
                if face_dict is not None:
                    faces_output.append(face_dict)

        return faces_output

    def _synthesize_face_from_pose(self, img: np.ndarray, pts: np.ndarray, subject_idx: int, img_shape: Tuple[int, int]):
        """Synthesize dense facial landmarks, 3D plane, and expressions from pose keypoints."""
        h, w = img_shape
        nose = pts[self.NOSE]
        left_eye = pts[self.LEFT_EYE]
        right_eye = pts[self.RIGHT_EYE]
        left_ear = pts[self.LEFT_EAR]
        right_ear = pts[self.RIGHT_EAR]

        facial_indices = [self.NOSE, self.LEFT_EYE, self.RIGHT_EYE, self.LEFT_EAR, self.RIGHT_EAR]
        valid_pts = [pts[idx] for idx in facial_indices if pts[idx][0] != 0 or pts[idx][1] != 0]

        if len(valid_pts) < 2:
            return None

        # Landmarks positions
        nx, ny = int(nose[0]), int(nose[1])
        lx, ly = int(left_eye[0]), int(left_eye[1])
        rx, ry = int(right_eye[0]), int(right_eye[1])

        has_both_eyes = (lx != 0 and rx != 0)
        eye_span = float(np.hypot(lx - rx, ly - ry)) if has_both_eyes else 50.0
        eye_span = max(25.0, eye_span)

        eye_mid = ((lx + rx) // 2, (ly + ry) // 2) if has_both_eyes else (nx, max(0, ny - int(eye_span * 0.4)))

        # Unit vector along eye line and perpendicular down face
        if has_both_eyes:
            eye_vec = np.array([rx - lx, ry - ly], dtype=float)
            norm = np.linalg.norm(eye_vec)
            eye_vec = eye_vec / norm if norm > 0 else np.array([1.0, 0.0])
        else:
            eye_vec = np.array([1.0, 0.0])

        down_vec = np.array([-eye_vec[1], eye_vec[0]])  # Downward face vector

        # Synthesize canonical face anchor points
        glabella = (int(eye_mid[0] - down_vec[0] * eye_span * 0.45),
                    int(eye_mid[1] - down_vec[1] * eye_span * 0.45))
        chin = (int(nx + down_vec[0] * eye_span * 1.1),
                int(ny + down_vec[1] * eye_span * 1.1))
        mouth_mid = (int(nx + down_vec[0] * eye_span * 0.55),
                     int(ny + down_vec[1] * eye_span * 0.55))
        left_mouth = (int(mouth_mid[0] - eye_vec[0] * eye_span * 0.35),
                      int(mouth_mid[1] - eye_vec[1] * eye_span * 0.35))
        right_mouth = (int(mouth_mid[0] + eye_vec[0] * eye_span * 0.35),
                       int(mouth_mid[1] + eye_vec[1] * eye_span * 0.35))

        # 2D 7 Key plane anchors matching MODEL_POINTS_3D
        plane_points_2d = np.array([
            (nx, ny),            # Nose tip
            chin,                # Chin
            (lx, ly),            # Left eye
            (rx, ry),            # Right eye
            left_mouth,          # Left mouth
            right_mouth,         # Right mouth
            glabella             # Forehead
        ], dtype=np.float64)

        # Estimate 3D Mathematical Face Plane using SolvePnP
        plane_data = self._calculate_face_plane(plane_points_2d, (h, w))

        # Crop face and analyze image gradients around eyes/mouth for expression
        pad = int(eye_span * 1.2)
        x1 = max(0, int(eye_mid[0] - pad))
        y1 = max(0, int(glabella[1] - pad * 0.4))
        x2 = min(w - 1, int(eye_mid[0] + pad))
        y2 = min(h - 1, int(chin[1] + pad * 0.3))

        face_crop = img[y1:y2, x1:x2]
        emotions, expressions = self._evaluate_facial_emotions_and_expressions(
            face_crop, plane_data['pitch_deg'], plane_data['yaw_deg'], plane_data['roll_deg'], eye_span
        )

        landmarks_dict = {
            'nose': (nx, ny),
            'chin': chin,
            'left_eye': (lx, ly),
            'right_eye': (rx, ry),
            'glabella': glabella,
            'mouth_left': left_mouth,
            'mouth_right': right_mouth,
            'mouth_center': mouth_mid,
            'left_ear': (int(left_ear[0]), int(left_ear[1])) if left_ear[0] != 0 else None,
            'right_ear': (int(right_ear[0]), int(right_ear[1])) if right_ear[0] != 0 else None,
        }

        return {
            'subject_index': subject_idx,
            'box': (x1, y1, x2, y2),
            'center': eye_mid,
            'landmarks': landmarks_dict,
            'plane': plane_data,
            'emotions': emotions,
            'primary_emotion': max(emotions.items(), key=lambda item: item[1])[0],
            'expressions': expressions,
            'confidence': 1.0,
            'roll_deg': plane_data['roll_deg'],
            'yaw_ratio': plane_data['yaw_ratio'],
            'eye_distance': eye_span,
        }

    def _process_dense_face_landmarks(self, img: np.ndarray, lms_2d: List[Tuple[int, int]], subject_idx: int, img_shape: Tuple[int, int]):
        """Process 468+ MediaPipe landmarks into 3D plane, expressions, and emotion scores."""
        h, w = img_shape
        xs = [p[0] for p in lms_2d]
        ys = [p[1] for p in lms_2d]
        x1, y1 = max(0, min(xs) - 10), max(0, min(ys) - 10)
        x2, y2 = min(w - 1, max(xs) + 10), min(h - 1, max(ys) + 10)

        # Standard MediaPipe indices for 7 canonical plane anchors:
        # Nose: 1, Chin: 152, Left Eye: 33, Right Eye: 263, Left Mouth: 61, Right Mouth: 291, Forehead: 10
        nose = lms_2d[1]
        chin = lms_2d[152]
        left_eye = lms_2d[33]
        right_eye = lms_2d[263]
        left_mouth = lms_2d[61]
        right_mouth = lms_2d[291]
        forehead = lms_2d[10]

        plane_points_2d = np.array([
            nose, chin, left_eye, right_eye, left_mouth, right_mouth, forehead
        ], dtype=np.float64)

        plane_data = self._calculate_face_plane(plane_points_2d, (h, w))

        # MediaPipe Eye & Mouth Aspect Ratios
        # Left eye: top 159, bottom 145, left 33, right 133
        # Right eye: top 386, bottom 374, left 362, right 263
        # Mouth: top 13, bottom 14, left 61, right 291
        left_ear = np.linalg.norm(np.array(lms_2d[159]) - np.array(lms_2d[145])) / max(1.0, np.linalg.norm(np.array(lms_2d[33]) - np.array(lms_2d[133])))
        right_ear = np.linalg.norm(np.array(lms_2d[386]) - np.array(lms_2d[374])) / max(1.0, np.linalg.norm(np.array(lms_2d[362]) - np.array(lms_2d[263])))
        avg_ear = (left_ear + right_ear) / 2.0

        mar = np.linalg.norm(np.array(lms_2d[13]) - np.array(lms_2d[14])) / max(1.0, np.linalg.norm(np.array(lms_2d[61]) - np.array(lms_2d[291])))

        # Smile curvature (mouth corners height relative to center lips)
        corner_y = (lms_2d[61][1] + lms_2d[291][1]) / 2.0
        lip_mid_y = (lms_2d[13][1] + lms_2d[14][1]) / 2.0
        smile_val = (lip_mid_y - corner_y) / max(1.0, (x2 - x1) * 0.1)

        emotions = self._score_emotions(avg_ear, mar, smile_val)
        expressions = self._compile_expressions(avg_ear, left_ear, right_ear, mar, smile_val)

        return {
            'subject_index': subject_idx,
            'box': (x1, y1, x2, y2),
            'center': ((x1 + x2) // 2, (y1 + y2) // 2),
            'landmarks': {'nose': nose, 'chin': chin, 'left_eye': left_eye, 'right_eye': right_eye, 'mouth_left': left_mouth, 'mouth_right': right_mouth},
            'plane': plane_data,
            'emotions': emotions,
            'primary_emotion': max(emotions.items(), key=lambda item: item[1])[0],
            'expressions': expressions,
            'confidence': 1.0,
            'roll_deg': plane_data['roll_deg'],
            'yaw_ratio': plane_data['yaw_ratio'],
            'eye_distance': float(np.hypot(left_eye[0] - right_eye[0], left_eye[1] - right_eye[1]))
        }

    def _calculate_face_plane(self, image_points_2d: np.ndarray, img_shape: Tuple[int, int]) -> Dict[str, Any]:
        """Compute 3D mathematical face plane equation and rotation/translation vectors."""
        h, w = img_shape
        focal_length = w
        center_x, center_y = w / 2.0, h / 2.0

        # Camera intrinsic matrix
        camera_matrix = np.array([
            [focal_length, 0, center_x],
            [0, focal_length, center_y],
            [0, 0, 1]
        ], dtype=np.float64)

        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        # Solve Perspective-n-Point
        success, rvec, tvec = cv2.solvePnP(
            self.MODEL_POINTS_3D,
            image_points_2d,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            return {
                'pitch_deg': 0.0, 'yaw_deg': 0.0, 'roll_deg': 0.0,
                'yaw_ratio': 0.0, 'plane_normal': (0.0, 0.0, 1.0),
                'plane_equation': (0.0, 0.0, 1.0, 0.0), 'rvec': None, 'tvec': None
            }

        # Compute Rotation Matrix
        R, _ = cv2.Rodrigues(rvec)

        # Mathematical plane normal vector: N = R * [0, 0, 1]^T
        normal_vector = R @ np.array([0.0, 0.0, 1.0])
        nx, ny, nz = normal_vector
        # Plane equation Ax + By + Cz + D = 0 passing through face center (tvec)
        D = -float(nx * tvec[0, 0] + ny * tvec[1, 0] + nz * tvec[2, 0])

        # Euler angles from rotation matrix
        sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
        singular = sy < 1e-6

        if not singular:
            pitch = np.arctan2(R[2, 1], R[2, 2])
            yaw = np.arctan2(-R[2, 0], sy)
            roll = np.arctan2(R[1, 0], R[0, 0])
        else:
            pitch = np.arctan2(-R[1, 2], R[1, 1])
            yaw = np.arctan2(-R[2, 0], sy)
            roll = 0.0

        pitch_deg = float(np.degrees(pitch))
        yaw_deg = float(np.degrees(yaw))
        roll_deg = float(np.degrees(roll))
        yaw_ratio = float(np.clip(yaw_deg / 45.0, -1.0, 1.0))

        # Project 3D axis pointing out from nose (500mm length)
        nose_end_point_3d = np.array([(0.0, 0.0, 500.0)], dtype=np.float64)
        nose_end_point_2d, _ = cv2.projectPoints(
            nose_end_point_3d, rvec, tvec, camera_matrix, dist_coeffs
        )
        p1 = (int(image_points_2d[0][0]), int(image_points_2d[0][1]))
        p2 = (int(nose_end_point_2d[0][0][0]), int(nose_end_point_2d[0][0][1]))

        return {
            'pitch_deg': pitch_deg,
            'yaw_deg': yaw_deg,
            'roll_deg': roll_deg,
            'yaw_ratio': yaw_ratio,
            'plane_normal': (float(nx), float(ny), float(nz)),
            'plane_equation': (float(nx), float(ny), float(nz), D),
            'axis_start': p1,
            'axis_end': p2,
            'rvec': rvec,
            'tvec': tvec
        }

    def _evaluate_facial_emotions_and_expressions(self, crop, pitch, yaw, roll, eye_span):
        """Estimate emotion probabilities and active expressions from facial dynamics."""
        if crop is None or crop.size == 0 or crop.shape[0] < 20 or crop.shape[1] < 20:
            return {'neutral': 0.85, 'happy': 0.05, 'surprised': 0.05, 'sad': 0.03, 'angry': 0.02}, ['neutral']

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        # Analyze lower third (mouth region)
        mouth_region = gray[int(h * 0.65):, int(w * 0.25):int(w * 0.75)]
        laplacian_var = cv2.Laplacian(mouth_region, cv2.CV_64F).var() if mouth_region.size > 0 else 0

        # Smile indicator: higher variance/gradients and wider horizontal edges
        is_smiling = laplacian_var > 120.0
        is_mouth_open = np.mean(mouth_region) < 70.0 if mouth_region.size > 0 else False

        emotions = {
            'neutral': 0.65,
            'happy': 0.25 if is_smiling else 0.05,
            'surprised': 0.20 if is_mouth_open else 0.05,
            'sad': 0.05,
            'angry': 0.05
        }

        # Normalize
        total = sum(emotions.values())
        emotions = {k: round(v / total, 3) for k, v in emotions.items()}

        expressions = []
        if is_smiling:
            expressions.append('smiling')
        if is_mouth_open:
            expressions.append('mouth_open')
        if abs(roll) > 15.0:
            expressions.append('head_tilted')
        if not expressions:
            expressions.append('neutral')

        return emotions, expressions

    def _score_emotions(self, ear: float, mar: float, smile_val: float) -> Dict[str, float]:
        """Score emotion probabilities based on facial metrics."""
        scores = {'neutral': 0.5, 'happy': 0.1, 'surprised': 0.1, 'sad': 0.1, 'angry': 0.1}

        if smile_val > 0.15:
            scores['happy'] += 0.6
            scores['neutral'] -= 0.3
        elif smile_val < -0.1:
            scores['sad'] += 0.4

        if mar > 0.45:
            scores['surprised'] += 0.6
            scores['neutral'] -= 0.3

        if ear < 0.18:
            scores['tired_or_blinking'] = 0.5

        # Normalize
        total = sum(max(0.01, v) for v in scores.values())
        return {k: round(max(0.01, v) / total, 3) for k, v in scores.items()}

    def _compile_expressions(self, ear, l_ear, r_ear, mar, smile_val) -> List[str]:
        expr = []
        if ear < 0.18:
            expr.append('blinking')
        elif abs(l_ear - r_ear) > 0.08:
            expr.append('winking')
        else:
            expr.append('eyes_open')

        if mar > 0.4:
            expr.append('mouth_open')

        if smile_val > 0.15:
            expr.append('smiling')
        elif smile_val < -0.1:
            expr.append('frowning')

        return expr

    def crop_face(self, img: np.ndarray, face: Dict[str, Any], margin: float = 0.15) -> np.ndarray:
        """Crop face region from image with optional margin."""
        h, w = img.shape[:2]
        x1, y1, x2, y2 = face['box']
        bw, bh = x2 - x1, y2 - y1
        mx, my = int(bw * margin), int(bh * margin)
        return img[max(0, y1 - my):min(h, y2 + my), max(0, x1 - mx):min(w, x2 + mx)]

    def draw_faces(self, img: np.ndarray, faces: List[Dict[str, Any]],
                   draw_plane: bool = True, draw_landmarks: bool = True,
                   draw_emotions: bool = True, draw_box: bool = True) -> np.ndarray:
        """Render face bounding boxes, 3D mathematical plane vectors, landmarks, and emotion HUD."""
        for face in faces:
            x1, y1, x2, y2 = face['box']
            s_idx = face.get('subject_index', 0)
            plane = face.get('plane', {})
            primary_emo = face.get('primary_emotion', 'neutral')
            emotions = face.get('emotions', {})
            expressions = face.get('expressions', [])
            landmarks = face.get('landmarks', {})

            # 1. Bounding box
            if draw_box:
                color = (0, 255, 120) if primary_emo == 'happy' else (255, 140, 0)
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            # 2. Mathematical 3D Plane Axis / Normal Vector
            if draw_plane and 'axis_start' in plane and 'axis_end' in plane:
                p1 = plane['axis_start']
                p2 = plane['axis_end']
                cv2.arrowedLine(img, p1, p2, (0, 0, 255), 2, tipLength=0.2)
                cv2.circle(img, p1, 4, (0, 255, 255), -1)

            # 3. Facial Landmarks
            if draw_landmarks:
                for name, pt in landmarks.items():
                    if pt is not None:
                        pt_color = (0, 255, 255) if 'eye' in name else (0, 165, 255)
                        cv2.circle(img, pt, 3, pt_color, -1)

            # 4. Emotion & Mathematical Plane HUD
            if draw_emotions:
                emo_conf = emotions.get(primary_emo, 0.0) * 100
                pitch = plane.get('pitch_deg', 0.0)
                yaw = plane.get('yaw_deg', 0.0)
                roll = plane.get('roll_deg', 0.0)

                label_top = f"P{s_idx} {primary_emo.upper()} ({emo_conf:.0f}%)"
                label_bot = f"P:{pitch:+.0f}° Y:{yaw:+.0f}° R:{roll:+.0f}° | {', '.join(expressions[:2])}"

                t1 = cv2.getTextSize(label_top, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)[0]
                t2 = cv2.getTextSize(label_bot, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)[0]

                card_w = max(t1[0], t2[0]) + 10
                card_h = t1[1] + t2[1] + 14
                card_y = max(0, y1 - card_h - 4)

                cv2.rectangle(img, (x1, card_y), (x1 + card_w, card_y + card_h), (25, 25, 25), -1)
                cv2.putText(img, label_top, (x1 + 4, card_y + t1[1] + 3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 200), 1, cv2.LINE_AA)
                cv2.putText(img, label_bot, (x1 + 4, card_y + t1[1] + t2[1] + 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1, cv2.LINE_AA)

        return img


class FaceTracker:
    """Helper to track faces, mathematical plane dynamics, and emotions across video frames."""

    def __init__(self, max_history=30, detector=None, iou_threshold=0.3):
        """Initialize FaceTracker."""
        self.detector = detector or FaceDetector()
        self.max_history = max_history
        self.iou_threshold = iou_threshold
        self.tracks = defaultdict(lambda: deque(maxlen=max_history))
        self.next_face_id = 0
        self.frame_count = 0
        self.active_tracks = {}

    def _calculate_iou(self, box1, box2):
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        xi1, yi1 = max(x1_1, x1_2), max(y1_1, y1_2)
        xi2, yi2 = min(x2_1, x2_2), min(y2_1, y2_2)

        if xi2 <= xi1 or yi2 <= yi1:
            return 0.0

        inter = (xi2 - xi1) * (yi2 - yi1)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - inter
        return inter / union if union > 0 else 0.0

    def update(self, img_or_faces, keypoints=None, device="cpu"):
        """Update face tracks with new detections and plane/emotion history."""
        self.frame_count += 1
        if isinstance(img_or_faces, list):
            faces = img_or_faces
        else:
            faces = self.detector.detect_faces(img_or_faces, keypoints=keypoints, device=device)

        matched_tracks = {}
        unmatched_faces = []

        for face in faces:
            best_id = None
            best_score = -1.0

            for face_id, last_entry in self.active_tracks.items():
                if self.frame_count - last_entry['frame'] > 15:
                    continue

                iou = self._calculate_iou(face['box'], last_entry['box'])
                dist = np.hypot(face['center'][0] - last_entry['center'][0],
                                face['center'][1] - last_entry['center'][1])
                score = iou * 0.7 + (1.0 / (1.0 + dist)) * 0.3

                if (iou > self.iou_threshold or dist < 60) and score > best_score:
                    best_score = score
                    best_id = face_id

            if best_id is not None and best_id not in matched_tracks:
                matched_tracks[best_id] = face
            else:
                unmatched_faces.append(face)

        for face in unmatched_faces:
            new_id = self.next_face_id
            self.next_face_id += 1
            matched_tracks[new_id] = face

        tracked_faces = []
        for face_id, face in matched_tracks.items():
            entry = {
                'frame': self.frame_count,
                'center': face['center'],
                'box': face['box'],
                'emotion': face.get('primary_emotion', 'neutral'),
                'plane': face.get('plane', {})
            }
            self.tracks[face_id].append(entry)
            self.active_tracks[face_id] = entry

            f_info = dict(face)
            f_info['face_id'] = face_id
            f_info['history'] = list(self.tracks[face_id])
            tracked_faces.append(f_info)

        return tracked_faces

    def draw_tracks(self, img: np.ndarray, tracked_faces: List[Dict[str, Any]],
                    draw_plane: bool = True, draw_trails: bool = True) -> np.ndarray:
        """Draw tracked faces, 3D plane projection, and motion trails."""
        self.detector.draw_faces(img, tracked_faces, draw_plane=draw_plane, draw_landmarks=True, draw_emotions=True)

        for face in tracked_faces:
            face_id = face.get('face_id', 0)
            history = face.get('history', [])
            x1, y1 = face['box'][:2]

            cv2.putText(img, f"ID: {face_id}", (x1, y1 - 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)

            if draw_trails and len(history) > 1:
                pts = [entry['center'] for entry in history]
                for i in range(1, len(pts)):
                    thickness = max(1, int(3 * (i / len(pts))))
                    cv2.line(img, pts[i - 1], pts[i], (255, 165, 0), thickness)

        return img
