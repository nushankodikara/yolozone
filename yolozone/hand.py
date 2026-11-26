'''
YoloZone Hand & Finger Tracking Module
Provides 21-landmark hand kinematic tracking, individual finger pose analysis,
joint angle measurement, and gesture recognition.
Developed by:
    - Nushan Kodikara
Contact:
    - nushankodi@gmail.com
'''
import numpy as np
import cv2
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple, Any


# 21 Hand Landmark indices (Standard Hand Topology)
WRIST = 0
THUMB_CMC = 1
THUMB_MCP = 2
THUMB_IP = 3
THUMB_TIP = 4
INDEX_MCP = 5
INDEX_PIP = 6
INDEX_DIP = 7
INDEX_TIP = 8
MIDDLE_MCP = 9
MIDDLE_PIP = 10
MIDDLE_DIP = 11
MIDDLE_TIP = 12
RING_MCP = 13
RING_PIP = 14
RING_DIP = 15
RING_TIP = 16
PINKY_MCP = 17
PINKY_PIP = 18
PINKY_DIP = 19
PINKY_TIP = 20

FINGER_CHAINS = {
    'thumb': [WRIST, THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP],
    'index': [WRIST, INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP],
    'middle': [WRIST, MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP],
    'ring': [WRIST, RING_MCP, RING_PIP, RING_DIP, RING_TIP],
    'pinky': [WRIST, PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP],
}

FINGER_COLORS = {
    'thumb': (0, 220, 100),    # Emerald
    'index': (255, 200, 0),    # Cyan/Yellow
    'middle': (0, 165, 255),   # Orange
    'ring': (255, 100, 255),   # Magenta
    'pinky': (50, 50, 255),    # Red
}


class HandDetector:
    """Hand and finger detector capable of extracting 21 landmarks, finger poses, and gestures."""

    # COCO Pose keypoint indices for pose-guided fallback
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_ELBOW = 7
    RIGHT_ELBOW = 8
    LEFT_WRIST = 9
    RIGHT_WRIST = 10
    NOSE = 0

    def __init__(self, model=None, use_mediapipe: bool = True, max_hands: int = 2):
        """Initialize HandDetector.

        Args:
            model: Optional YOLO hand/pose model path or PoseDetector instance.
            use_mediapipe: If True and mediapipe is installed, utilizes MediaPipe Hands
                           for direct landmark tracking.
            max_hands: Maximum number of hands to detect.
        """
        self.max_hands = max_hands
        self.mp_hands = None
        self.custom_model = None
        self.pose_detector = None

        # 1. Attempt MediaPipe initialization if requested
        if use_mediapipe:
            try:
                import mediapipe as mp
                self.mp_hands = mp.solutions.hands.Hands(
                    static_image_mode=False,
                    max_num_hands=max_hands,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5
                )
            except (ImportError, Exception):
                self.mp_hands = None

        # 2. YOLO or Pose detector fallback
        if isinstance(model, str) and not ("pose" in model.lower()):
            from ultralytics import YOLO
            self.custom_model = YOLO(model)
        elif model is not None and hasattr(model, 'find_keypoints'):
            self.pose_detector = model
        else:
            from .pose import PoseDetector
            pose_model_name = model if isinstance(model, str) else "yolov8s-pose.pt"
            self.pose_detector = PoseDetector(model=pose_model_name)

    def detect_hands(self, img: np.ndarray, keypoints=None, device="cpu", conf=0.25) -> List[Dict[str, Any]]:
        """Detect hands and extract 21 finger landmarks and pose states.

        Args:
            img: BGR image numpy array
            keypoints: Optional pre-computed pose keypoints
            device: Device for YOLO inference
            conf: Confidence threshold

        Returns:
            List of hand dictionaries containing 21 landmarks, finger poses, and gesture classifications.
        """
        h, w = img.shape[:2]
        hands_output = []

        # Strategy A: MediaPipe Hands (if available)
        if self.mp_hands is not None:
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mp_results = self.mp_hands.process(rgb_img)

            if mp_results.multi_hand_landmarks:
                for idx, hand_landmarks in enumerate(mp_results.multi_hand_landmarks):
                    handedness_label = "right"
                    if mp_results.multi_handedness and len(mp_results.multi_handedness) > idx:
                        handedness_label = mp_results.multi_handedness[idx].classification[0].label.lower()

                    # Convert normalized coords to pixels
                    landmarks_21 = []
                    xs, ys = [], []
                    for lm in hand_landmarks.landmark:
                        px, py = int(lm.x * w), int(lm.y * h)
                        landmarks_21.append((px, py))
                        xs.append(px)
                        ys.append(py)

                    x1, y1 = max(0, min(xs) - 10), max(0, min(ys) - 10)
                    x2, y2 = min(w - 1, max(xs) + 10), min(h - 1, max(ys) + 10)
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                    hand_data = self._analyze_hand_landmarks(
                        landmarks=landmarks_21,
                        hand_type=handedness_label,
                        box=(x1, y1, x2, y2),
                        center=(cx, cy),
                        wrist=landmarks_21[WRIST],
                        subject_idx=idx,
                        img_shape=(h, w)
                    )
                    hands_output.append(hand_data)

            if hands_output:
                return hands_output

        # Strategy B: Pose-Guided Kinematic Finger Synthesis
        if keypoints is None and self.pose_detector is not None:
            keypoints = self.pose_detector.find_keypoints(img, device=device)

        if keypoints is not None and hasattr(keypoints, 'xy') and len(keypoints.xy) > 0:
            for s_idx in range(len(keypoints.xy)):
                pts = keypoints.xy[s_idx]
                if pts.shape[0] < 11:
                    continue

                for h_type, w_idx, e_idx, s_shoulder in [
                    ('left', self.LEFT_WRIST, self.LEFT_ELBOW, self.LEFT_SHOULDER),
                    ('right', self.RIGHT_WRIST, self.RIGHT_ELBOW, self.RIGHT_SHOULDER)
                ]:
                    wrist = pts[w_idx]
                    elbow = pts[e_idx]
                    shoulder = pts[s_shoulder]

                    if wrist[0] == 0 and wrist[1] == 0:
                        continue

                    hand_data = self._synthesize_hand_from_pose(
                        img, pts, s_idx, h_type, wrist, elbow, shoulder, (h, w)
                    )
                    if hand_data is not None:
                        hands_output.append(hand_data)

        return hands_output

    def _synthesize_hand_from_pose(self, img, pts, s_idx, h_type, wrist, elbow, shoulder, img_shape):
        """Construct full 21 landmarks using pose arm vector and image contour analysis."""
        h, w = img_shape
        wx, wy = int(wrist[0]), int(wrist[1])
        ex, ey = int(elbow[0]), int(elbow[1])
        sx, sy = int(shoulder[0]), int(shoulder[1])

        # Compute arm directional unit vector
        if ex != 0 or ey != 0:
            arm_vec = np.array([wx - ex, wy - ey], dtype=float)
            norm = np.linalg.norm(arm_vec)
            arm_vec = arm_vec / norm if norm > 0 else np.array([0.0, -1.0])
            forearm_len = norm
        else:
            arm_vec = np.array([0.0, -1.0], dtype=float)
            forearm_len = 100.0

        perp_vec = np.array([-arm_vec[1], arm_vec[0]])
        if h_type == 'right':
            perp_vec = -perp_vec

        scale = max(35.0, forearm_len * 0.42)
        cx = int(wx + arm_vec[0] * scale * 0.5)
        cy = int(wy + arm_vec[1] * scale * 0.5)

        half_box = int(scale * 0.8)
        x1 = max(0, cx - half_box)
        y1 = max(0, cy - half_box)
        x2 = min(w - 1, cx + half_box)
        y2 = min(h - 1, cy + half_box)

        # Crop hand region and check skin/contour convexity to detect finger extension
        crop = img[y1:y2, x1:x2]
        crop_fingers_extended = self._estimate_fingers_from_crop(crop)

        # Synthesize 21 standard landmarks relative to wrist and forearm orientation
        landmarks_21 = self._generate_canonical_21_landmarks(
            (wx, wy), arm_vec, perp_vec, scale, crop_fingers_extended
        )

        is_raised = (sy != 0 and wy < sy)
        hand_data = self._analyze_hand_landmarks(
            landmarks=landmarks_21,
            hand_type=h_type,
            box=(x1, y1, x2, y2),
            center=(cx, cy),
            wrist=(wx, wy),
            subject_idx=s_idx,
            img_shape=(h, w)
        )
        hand_data['is_raised'] = is_raised
        return hand_data

    def _estimate_fingers_from_crop(self, crop: np.ndarray) -> Dict[str, bool]:
        """Estimate which fingers are extended using contour analysis and convexity defects."""
        defaults = {'thumb': True, 'index': True, 'middle': True, 'ring': False, 'pinky': False}
        if crop is None or crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
            return defaults

        # Convert to grayscale and threshold
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return defaults

        largest = max(contours, key=cv2.contourArea)
        hull = cv2.convexHull(largest, returnPoints=False)
        if hull is not None and len(hull) > 3 and len(largest) > 3:
            try:
                defects = cv2.convexityDefects(largest, hull)
                defect_count = 0
                if defects is not None:
                    for i in range(defects.shape[0]):
                        s, e, f, d = defects[i, 0]
                        if d > 1000:  # significant valley between fingers
                            defect_count += 1
                if defect_count >= 3:
                    return {'thumb': True, 'index': True, 'middle': True, 'ring': True, 'pinky': True}
                elif defect_count == 2:
                    return {'thumb': True, 'index': True, 'middle': True, 'ring': False, 'pinky': False}
                elif defect_count == 1:
                    return {'thumb': False, 'index': True, 'middle': False, 'ring': False, 'pinky': False}
            except Exception:
                pass

        return defaults

    def _generate_canonical_21_landmarks(self, wrist, arm_vec, perp_vec, scale, finger_states):
        """Generate 21 geometric landmark positions along finger kinematic chains."""
        wx, wy = wrist
        lms = [wrist] * 21

        # Finger base offsets from wrist across MCP arch
        mcp_offsets = {
            'thumb': (-0.35, 0.25),
            'index': (-0.2, 0.6),
            'middle': (0.0, 0.65),
            'ring': (0.2, 0.6),
            'pinky': (0.35, 0.5),
        }

        mcp_indices = {'thumb': THUMB_MCP, 'index': INDEX_MCP, 'middle': MIDDLE_MCP, 'ring': RING_MCP, 'pinky': PINKY_MCP}
        pip_indices = {'thumb': THUMB_IP, 'index': INDEX_PIP, 'middle': MIDDLE_PIP, 'ring': RING_PIP, 'pinky': PINKY_PIP}
        dip_indices = {'thumb': THUMB_IP, 'index': INDEX_DIP, 'middle': MIDDLE_DIP, 'ring': RING_DIP, 'pinky': PINKY_DIP}
        tip_indices = {'thumb': THUMB_TIP, 'index': INDEX_TIP, 'middle': MIDDLE_TIP, 'ring': RING_TIP, 'pinky': PINKY_TIP}

        # Thumb CMC
        cmc_pos = (int(wx + perp_vec[0] * scale * (-0.2) + arm_vec[0] * scale * 0.1),
                   int(wy + perp_vec[1] * scale * (-0.2) + arm_vec[1] * scale * 0.1))
        lms[THUMB_CMC] = cmc_pos

        for fname in ['thumb', 'index', 'middle', 'ring', 'pinky']:
            lat_off, fwd_off = mcp_offsets[fname]
            mcp_x = int(wx + perp_vec[0] * scale * lat_off + arm_vec[0] * scale * fwd_off)
            mcp_y = int(wy + perp_vec[1] * scale * lat_off + arm_vec[1] * scale * fwd_off)
            lms[mcp_indices[fname]] = (mcp_x, mcp_y)

            # Determine extension direction
            is_ext = finger_states.get(fname, True)
            ext_len = scale * (0.4 if is_ext else 0.15)
            curl_dir = arm_vec if is_ext else -arm_vec * 0.5

            pip_x = int(mcp_x + curl_dir[0] * ext_len * 0.4 + perp_vec[0] * scale * lat_off * 0.2)
            pip_y = int(mcp_y + curl_dir[1] * ext_len * 0.4 + perp_vec[1] * scale * lat_off * 0.2)
            lms[pip_indices[fname]] = (pip_x, pip_y)

            dip_x = int(pip_x + curl_dir[0] * ext_len * 0.3)
            dip_y = int(pip_y + curl_dir[1] * ext_len * 0.3)
            lms[dip_indices[fname]] = (dip_x, dip_y)

            tip_x = int(dip_x + curl_dir[0] * ext_len * 0.3)
            tip_y = int(dip_y + curl_dir[1] * ext_len * 0.3)
            lms[tip_indices[fname]] = (tip_x, tip_y)

        return lms

    def _analyze_hand_landmarks(self, landmarks, hand_type, box, center, wrist, subject_idx, img_shape):
        """Analyze 21 hand landmarks to evaluate pose of each finger and overall gesture."""
        fingers_data = {}

        # Analyze each finger's extension, flex angle, and tip position
        for fname, indices in FINGER_CHAINS.items():
            pts = [landmarks[i] for i in indices]
            wrist_pt = np.array(landmarks[WRIST])
            mcp_pt = np.array(pts[1]) if fname == 'thumb' else np.array(pts[1])
            pip_pt = np.array(pts[2])
            tip_pt = np.array(pts[-1])

            # Flex angle at PIP joint (MCP -> PIP -> TIP)
            v1 = mcp_pt - pip_pt
            v2 = tip_pt - pip_pt
            n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)

            if n1 > 0 and n2 > 0:
                cosine = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
                angle_deg = float(np.degrees(np.arccos(cosine)))
            else:
                angle_deg = 180.0

            # Extension check: distance from wrist to tip vs distance from wrist to PIP
            dist_wrist_tip = np.linalg.norm(tip_pt - wrist_pt)
            dist_wrist_pip = np.linalg.norm(pip_pt - wrist_pt)

            if fname == 'thumb':
                # Thumb extends sideways
                ip_pt = np.array(pts[3])
                is_ext = np.linalg.norm(tip_pt - wrist_pt) > np.linalg.norm(ip_pt - wrist_pt)
            else:
                is_ext = dist_wrist_tip > dist_wrist_pip * 1.15

            if is_ext:
                state = 'extended'
            elif angle_deg < 90.0:
                state = 'curled'
            else:
                state = 'half_curled'

            fingers_data[fname] = {
                'state': state,
                'is_extended': bool(is_ext),
                'flex_angle': angle_deg,
                'tip': tuple(map(int, tip_pt)),
                'points': [tuple(map(int, p)) for p in pts]
            }

        # Pinch detection (Thumb TIP to Index TIP)
        thumb_tip = np.array(landmarks[THUMB_TIP])
        index_tip = np.array(landmarks[INDEX_TIP])
        pinch_dist = float(np.linalg.norm(thumb_tip - index_tip))
        is_pinching = pinch_dist < 35.0

        # Classify gesture based on finger poses
        gesture = self._classify_gesture(fingers_data, is_pinching)

        return {
            'subject_index': subject_idx,
            'type': hand_type,
            'wrist': wrist,
            'center': center,
            'box': box,
            'landmarks': landmarks,
            'fingers': fingers_data,
            'gesture': gesture,
            'pinch_distance': pinch_dist,
            'is_pinching': is_pinching,
            'is_raised': False,
            'confidence': 1.0,
        }

    def _classify_gesture(self, fingers: Dict[str, Dict], is_pinching: bool) -> str:
        """Classify discrete hand gesture from individual finger states."""
        ext_thumb = fingers['thumb']['is_extended']
        ext_index = fingers['index']['is_extended']
        ext_middle = fingers['middle']['is_extended']
        ext_ring = fingers['ring']['is_extended']
        ext_pinky = fingers['pinky']['is_extended']

        num_extended = sum([ext_thumb, ext_index, ext_middle, ext_ring, ext_pinky])

        if is_pinching:
            return "pinch"
        if num_extended == 5:
            return "open_palm"
        if num_extended == 0:
            return "fist"
        if ext_index and ext_middle and not ext_ring and not ext_pinky:
            return "peace"
        if ext_index and not ext_middle and not ext_ring and not ext_pinky:
            return "pointing"
        if ext_thumb and not ext_index and not ext_middle and not ext_ring and not ext_pinky:
            return "thumbs_up"
        if ext_index and ext_pinky and not ext_middle and not ext_ring:
            return "rock_horns"
        if is_pinching and ext_middle and ext_ring and ext_pinky:
            return "ok_sign"

        return f"{num_extended}_fingers"

    def get_hands_distance(self, keypoints, subject_index=0) -> Tuple[float, str, Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
        """Calculate Euclidean distance between left and right wrists."""
        pts = keypoints.xy[subject_index]
        lw = pts[self.LEFT_WRIST]
        rw = pts[self.RIGHT_WRIST]

        if (lw[0] == 0 and lw[1] == 0) or (rw[0] == 0 and rw[1] == 0):
            return 0.0, "N/A", (0, 0), (0, 0), (0, 0)

        p1 = np.array([int(lw[0]), int(lw[1])])
        p2 = np.array([int(rw[0]), int(rw[1])])
        dist = float(np.linalg.norm(p1 - p2))
        center_pos = tuple(map(int, (p1 + p2) // 2))

        return dist, f"{dist:.1f}px", center_pos, tuple(p1), tuple(p2)

    def is_clapping(self, keypoints, threshold_px=50, subject_index=0) -> bool:
        """Check if person is clapping or hands are touching."""
        dist, _, _, _, _ = self.get_hands_distance(keypoints, subject_index=subject_index)
        return dist > 0 and dist <= threshold_px

    def is_hand_raised(self, keypoints, hand_type="left", subject_index=0) -> bool:
        """Check if specified hand is raised above shoulder."""
        pts = keypoints.xy[subject_index]
        w_idx = self.LEFT_WRIST if hand_type == "left" else self.RIGHT_WRIST
        s_idx = self.LEFT_SHOULDER if hand_type == "left" else self.RIGHT_SHOULDER

        wrist = pts[w_idx]
        shoulder = pts[s_idx]
        if (wrist[0] == 0 and wrist[1] == 0) or (shoulder[0] == 0 and shoulder[1] == 0):
            return False

        return bool(wrist[1] < shoulder[1])

    def draw_hands(self, img: np.ndarray, hands: List[Dict[str, Any]],
                   draw_skeleton: bool = True, draw_box: bool = True,
                   draw_finger_labels: bool = True, draw_gestures: bool = True) -> np.ndarray:
        """Draw hand bounding boxes, 21-point finger skeletons, and finger pose status."""
        for hand in hands:
            box = hand['box']
            h_type = hand['type']
            gesture = hand.get('gesture', '')
            landmarks = hand.get('landmarks', [])

            # 1. Draw 21-point skeleton
            if draw_skeleton and len(landmarks) == 21:
                # Draw palm connections (MCP arch)
                mcp_pts = [landmarks[WRIST], landmarks[INDEX_MCP], landmarks[MIDDLE_MCP],
                           landmarks[RING_MCP], landmarks[PINKY_MCP], landmarks[WRIST]]
                for i in range(len(mcp_pts) - 1):
                    cv2.line(img, mcp_pts[i], mcp_pts[i+1], (180, 180, 180), 2)

                # Draw finger chains
                for fname, indices in FINGER_CHAINS.items():
                    f_color = FINGER_COLORS.get(fname, (255, 255, 255))
                    for i in range(len(indices) - 1):
                        p1 = landmarks[indices[i]]
                        p2 = landmarks[indices[i+1]]
                        cv2.line(img, p1, p2, f_color, 2)

                # Draw landmark dots
                for idx, pt in enumerate(landmarks):
                    color = (0, 0, 255) if idx in [THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP] else (255, 255, 255)
                    radius = 4 if idx in [THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP] else 3
                    cv2.circle(img, pt, radius, color, -1)

            # 2. Draw Bounding Box
            if draw_box:
                color = (0, 220, 100) if h_type == 'left' else (255, 140, 0)
                cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color, 2)

            # 3. Draw Gesture & Finger Pose HUD
            if draw_gestures and gesture:
                label = f"{h_type.upper()}: {gesture.replace('_', ' ').upper()}"
                t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                ly = max(0, box[1] - 8)
                cv2.rectangle(img, (box[0], ly - t_size[1] - 6), (box[0] + t_size[0] + 8, ly + 2), (20, 20, 20), -1)
                cv2.putText(img, label, (box[0] + 4, ly - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

            # 4. Draw Individual Finger States
            if draw_finger_labels and 'fingers' in hand:
                for fname, fdata in hand['fingers'].items():
                    tip = fdata['tip']
                    state_char = "E" if fdata['is_extended'] else "C"
                    cv2.putText(img, state_char, (tip[0] + 4, tip[1] - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.38, FINGER_COLORS.get(fname, (255, 255, 255)), 1)

        return img


class HandTracker:
    """Helper to track hands, finger pose trajectories, and waving gestures across frames."""

    def __init__(self, max_history=30, detector=None):
        """Initialize HandTracker."""
        self.detector = detector or HandDetector()
        self.max_history = max_history
        self.histories = defaultdict(lambda: deque(maxlen=max_history))
        self.frame_count = 0

    def update(self, img_or_hands, keypoints=None, device="cpu"):
        """Update hand tracks with frame or pre-detected hands."""
        self.frame_count += 1
        if isinstance(img_or_hands, list):
            hands = img_or_hands
        else:
            hands = self.detector.detect_hands(img_or_hands, keypoints=keypoints, device=device)

        tracked = []
        for hand in hands:
            s_idx = hand.get('subject_index', 0)
            h_type = hand.get('type', 'hand')
            track_key = f"{s_idx}_{h_type}"

            center = hand['center']
            box = hand['box']

            self.histories[track_key].append({
                'frame': self.frame_count,
                'center': center,
                'box': box,
                'gesture': hand.get('gesture', ''),
                'is_raised': hand.get('is_raised', False)
            })

            # Calculate velocity
            vel_x, vel_y, speed = self._calculate_velocity(track_key)
            is_waving = self._detect_waving(track_key)

            hand_info = dict(hand)
            hand_info['track_id'] = track_key
            hand_info['history'] = list(self.histories[track_key])
            hand_info['velocity'] = (vel_x, vel_y)
            hand_info['speed'] = speed
            hand_info['is_waving'] = is_waving
            tracked.append(hand_info)

        return tracked

    def _calculate_velocity(self, track_key):
        hist = self.histories[track_key]
        if len(hist) < 2:
            return 0.0, 0.0, 0.0
        dt = max(1, hist[-1]['frame'] - hist[-2]['frame'])
        vx = (hist[-1]['center'][0] - hist[-2]['center'][0]) / dt
        vy = (hist[-1]['center'][1] - hist[-2]['center'][1]) / dt
        return vx, vy, float(np.hypot(vx, vy))

    def _detect_waving(self, track_key, min_reversals=3, min_amplitude=15):
        hist = self.histories[track_key]
        if len(hist) < 10 or not hist[-1].get('is_raised', False):
            return False

        xs = [entry['center'][0] for entry in hist]
        diffs = [xs[i+1] - xs[i] for i in range(len(xs) - 1)]

        direction_changes = 0
        last_dir = 0
        for d in diffs:
            if abs(d) >= min_amplitude / 3:
                curr_dir = 1 if d > 0 else -1
                if last_dir != 0 and curr_dir != last_dir:
                    direction_changes += 1
                last_dir = curr_dir

        return direction_changes >= min_reversals

    def draw_tracks(self, img: np.ndarray, tracked_hands: List[Dict[str, Any]],
                    draw_skeleton: bool = True, draw_trails: bool = True, draw_gestures: bool = True) -> np.ndarray:
        """Render tracked hands, finger pose skeletons, trails, and waving indicators."""
        self.detector.draw_hands(
            img, tracked_hands,
            draw_skeleton=draw_skeleton,
            draw_box=True,
            draw_finger_labels=True,
            draw_gestures=draw_gestures
        )

        for hand in tracked_hands:
            track_key = hand.get('track_id', '')
            history = self.histories[track_key]

            if draw_trails and len(history) > 1:
                points = [entry['center'] for entry in history]
                h_type = hand.get('type', 'hand')
                trail_color = (0, 200, 100) if h_type == 'left' else (255, 140, 0)
                for i in range(1, len(points)):
                    thickness = max(1, int(4 * (i / len(points))))
                    cv2.line(img, points[i - 1], points[i], trail_color, thickness)

            if hand.get('is_waving', False):
                cx, cy = hand['center']
                cv2.putText(img, "WAVING!", (cx - 30, cy - 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

        return img
