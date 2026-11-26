'''
YoloZone Tracking Helpers Module
Developed by:
    - Nushan Kodikara
Contact:
    - nushankodi@gmail.com
'''
import numpy as np
import cv2
from collections import defaultdict, deque


class ZoneTracker:
    """Helper for tracking objects, occupancy, entries, exits, and dwell time within a defined Zone / ROI."""

    def __init__(self, zone_polygon, name="Zone", on_enter=None, on_exit=None):
        """Initialize ZoneTracker.

        Args:
            zone_polygon: List of (x, y) tuples defining the polygon vertices, or (x1, y1, x2, y2) rect.
            name: Label name for the zone.
            on_enter: Optional callback function fn(track_id, track_info, zone_name).
            on_exit: Optional callback function fn(track_id, track_info, dwell_frames, zone_name).
        """
        self.name = name
        self.on_enter = on_enter
        self.on_exit = on_exit

        # Convert to numpy contour
        if isinstance(zone_polygon, (list, tuple)) and len(zone_polygon) == 4 and isinstance(zone_polygon[0], (int, float)):
            x1, y1, x2, y2 = map(int, zone_polygon)
            self.polygon = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32)
        else:
            self.polygon = np.array(zone_polygon, dtype=np.int32)

        self.current_tracks_in_zone = set()
        self.last_known_tracks = {}
        self.dwell_times = defaultdict(int)  # track_id -> frames inside
        self.total_entered = 0
        self.total_exited = 0
        self.class_counts = defaultdict(int)  # class_name -> total entries
        self.frame_count = 0

    def is_inside(self, point_or_box):
        """Check if a point or box center/bottom is inside the polygon.

        Args:
            point_or_box: Either (x, y) tuple or (x1, y1, x2, y2) box tuple.
        """
        if len(point_or_box) == 4:
            x1, y1, x2, y2 = point_or_box
            # Use bottom center (ground contact point) or center
            test_point = (int((x1 + x2) / 2), int(y2))
        else:
            test_point = (int(point_or_box[0]), int(point_or_box[1]))

        # cv2.pointPolygonTest returns positive if inside, 0 on edge, negative outside
        return cv2.pointPolygonTest(self.polygon, test_point, False) >= 0

    def update(self, tracks):
        """Update zone tracking with current active tracks.

        Args:
            tracks: Dictionary of tracks {track_id: track_info} or list of track dicts.

        Returns:
            summary: Dictionary containing current occupancy, entered, exited, and active IDs.
        """
        self.frame_count += 1
        active_in_zone = set()

        track_dict = tracks if isinstance(tracks, dict) else {t['track_id']: t for t in tracks}

        for track_id, track_info in track_dict.items():
            pos = track_info.get('center') or (track_info.get('box')[:2] if 'box' in track_info else None)
            if pos is None:
                continue

            if self.is_inside(track_info.get('box', pos)):
                active_in_zone.add(track_id)
                self.dwell_times[track_id] += 1

                # Check if newly entered
                if track_id not in self.current_tracks_in_zone:
                    self.total_entered += 1
                    cls_name = track_info.get('class', 'object')
                    self.class_counts[cls_name] += 1
                    if self.on_enter:
                        self.on_enter(track_id, track_info, self.name)

        # Check for exits
        exited = self.current_tracks_in_zone - active_in_zone
        for exit_id in exited:
            self.total_exited += 1
            if self.on_exit:
                t_info = track_dict.get(exit_id) or self.last_known_tracks.get(exit_id, {'track_id': exit_id})
                self.on_exit(exit_id, t_info, self.dwell_times[exit_id], self.name)

        self.last_known_tracks.update(track_dict)
        self.current_tracks_in_zone = active_in_zone

        return {
            'zone': self.name,
            'occupancy': len(self.current_tracks_in_zone),
            'total_entered': self.total_entered,
            'total_exited': self.total_exited,
            'active_tracks': list(self.current_tracks_in_zone),
            'dwell_times': {tid: self.dwell_times[tid] for tid in self.current_tracks_in_zone}
        }

    def draw(self, img, color=(0, 255, 200), alpha=0.25, thickness=2, draw_stats=True):
        """Render the zone polygon and occupancy overlay on image.

        Args:
            img: Image to draw on
            color: (B, G, R) color for zone
            alpha: Transparency for zone fill (0.0 to 1.0)
            thickness: Border thickness
            draw_stats: Whether to render statistics badge
        """
        # Draw translucent polygon fill
        overlay = img.copy()
        cv2.fillPoly(overlay, [self.polygon], color)
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

        # Draw polygon border
        cv2.polylines(img, [self.polygon], isClosed=True, color=color, thickness=thickness)

        if draw_stats:
            # Place stats badge at top-left vertex of polygon
            min_pt = self.polygon[np.argmin(self.polygon[:, 1])]
            bx, by = int(min_pt[0]), max(35, int(min_pt[1]) - 10)

            label = f"{self.name} | Occupancy: {len(self.current_tracks_in_zone)} (In: {self.total_entered}, Out: {self.total_exited})"
            t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(img, (bx, by - t_size[1] - 6), (bx + t_size[0] + 8, by + 4), (20, 20, 20), -1)
            cv2.rectangle(img, (bx, by - t_size[1] - 6), (bx + t_size[0] + 8, by + 4), color, 1)
            cv2.putText(img, label, (bx + 4, by - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        return img


class LineCrossingHelper:
    """Helper for managing multiple line crossing counters with directional classification."""

    def __init__(self):
        """Initialize LineCrossingHelper."""
        self.lines = {}  # name -> {'start': (x,y), 'end': (x,y), 'crossings': {'in': Counter(), 'out': Counter()}}
        self.crossed_ids = defaultdict(set)  # line_name -> set(track_ids)

    def add_line(self, name, start_point, end_point):
        """Add a counting line with a unique name."""
        self.lines[name] = {
            'start': tuple(map(int, start_point)),
            'end': tuple(map(int, end_point)),
            'in_count': defaultdict(int),
            'out_count': defaultdict(int),
            'total': 0
        }

    def _segments_intersect(self, p1, p2, p3, p4):
        """Check if segment p1-p2 intersects p3-p4 and return direction."""
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4

        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if denom == 0:
            return False, 0

        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom

        if 0 <= t <= 1 and 0 <= u <= 1:
            # Cross product to determine crossing direction
            cross = (x4 - x3) * (y2 - y1) - (y4 - y3) * (x2 - x1)
            direction = 1 if cross > 0 else -1
            return True, direction

        return False, 0

    def update(self, tracks):
        """Update crossings for all configured lines using active tracks.

        Args:
            tracks: Dictionary of tracks or list of tracks from ObjectTracker
        """
        track_dict = tracks if isinstance(tracks, dict) else {t['track_id']: t for t in tracks}
        new_crossings = []

        for line_name, line_info in self.lines.items():
            l_start = line_info['start']
            l_end = line_info['end']

            for track_id, track in track_dict.items():
                history = track.get('history', [])
                if len(history) < 2:
                    continue

                if track_id in self.crossed_ids[line_name]:
                    continue

                p_prev = history[-2]['center']
                p_curr = history[-1]['center']

                crossed, direction = self._segments_intersect(l_start, l_end, p_prev, p_curr)
                if crossed:
                    self.crossed_ids[line_name].add(track_id)
                    cls_name = track.get('class', 'object')
                    line_info['total'] += 1

                    dir_key = 'in' if direction > 0 else 'out'
                    if dir_key == 'in':
                        line_info['in_count'][cls_name] += 1
                    else:
                        line_info['out_count'][cls_name] += 1

                    new_crossings.append({
                        'line': line_name,
                        'track_id': track_id,
                        'direction': dir_key,
                        'class': cls_name
                    })

        return new_crossings

    def draw(self, img, color=(0, 255, 255), thickness=2):
        """Draw all counting lines and stats on the image."""
        for line_name, line_info in self.lines.items():
            p1 = line_info['start']
            p2 = line_info['end']
            in_sum = sum(line_info['in_count'].values())
            out_sum = sum(line_info['out_count'].values())

            # Draw line
            cv2.line(img, p1, p2, color, thickness)

            # Midpoint text
            mx, my = (p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2
            text = f"{line_name} [In: {in_sum} | Out: {out_sum}]"
            cv2.putText(img, text, (mx + 10, my - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, thickness, cv2.LINE_AA)

        return img


class SpeedEstimator:
    """Helper to calculate and display real-world or pixel velocity of tracked objects."""

    def __init__(self, pixels_per_meter=15.0, fps=30.0, smoothing_window=5):
        """Initialize SpeedEstimator.

        Args:
            pixels_per_meter: Calibration factor (pixels per meter).
            fps: Video frame rate.
            smoothing_window: Number of frames to average speed over.
        """
        self.ppm = max(0.001, pixels_per_meter)
        self.fps = max(1.0, fps)
        self.window = smoothing_window
        self.speed_histories = defaultdict(lambda: deque(maxlen=smoothing_window))

    def estimate_speed(self, track_id, track_history):
        """Calculate smoothed speed for a track.

        Args:
            track_id: Track ID
            track_history: List of position dicts [{'center': (x,y), ...}, ...]

        Returns:
            speed_kmh: Speed in km/h
            speed_mps: Speed in m/s
        """
        if len(track_history) < 2:
            return 0.0, 0.0

        p_curr = track_history[-1]['center']
        p_prev = track_history[-2]['center']
        pixel_dist = np.hypot(p_curr[0] - p_prev[0], p_curr[1] - p_prev[1])

        # Distance in meters
        meters = pixel_dist / self.ppm
        # Speed in m/s
        mps = meters * self.fps
        # Speed in km/h
        kmh = mps * 3.6

        self.speed_histories[track_id].append(kmh)
        avg_kmh = float(np.mean(self.speed_histories[track_id]))
        avg_mps = avg_kmh / 3.6

        return avg_kmh, avg_mps

    def draw_speeds(self, img, tracks, unit="km/h", color=(0, 255, 0)):
        """Draw speed tags above each tracked object."""
        track_dict = tracks if isinstance(tracks, dict) else {t['track_id']: t for t in tracks}

        for track_id, track in track_dict.items():
            history = track.get('history', [])
            if len(history) < 2:
                continue

            kmh, mps = self.estimate_speed(track_id, history)
            speed_val = kmh if unit == "km/h" else mps
            text = f"{speed_val:.1f} {unit}"

            cx, cy = history[-1]['center']
            cv2.putText(img, text, (cx - 20, cy - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

        return img


class TrajectoryVisualizer:
    """Helper for generating path trails, directional vectors, and spatial heatmaps."""

    def __init__(self, frame_shape=(720, 1280), decay=0.99):
        """Initialize TrajectoryVisualizer.

        Args:
            frame_shape: (height, width) of video frame for heatmap accumulator
            decay: Heatmap persistence decay factor (0.0 to 1.0)
        """
        self.frame_shape = frame_shape
        self.decay = decay
        self.density_map = np.zeros(frame_shape[:2], dtype=np.float32)

    def update(self, tracks):
        """Update trajectory accumulator with current track positions."""
        self.density_map *= self.decay

        track_dict = tracks if isinstance(tracks, dict) else {t['track_id']: t for t in tracks}
        h, w = self.frame_shape[:2]

        for _, track in track_dict.items():
            cx, cy = track.get('center', (0, 0))
            if 0 <= cx < w and 0 <= cy < h:
                # Add gaussian splat around position
                cv2.circle(self.density_map, (cx, cy), 15, 1.0, -1)

    def get_heatmap(self, base_img=None, colormap=cv2.COLORMAP_JET, alpha=0.5):
        """Generate a color heatmap overlay.

        Args:
            base_img: Optional base frame to overlay heatmap onto
            colormap: OpenCV colormap
            alpha: Heatmap blend weight

        Returns:
            heatmap_img: Rendered heatmap (or blended frame)
        """
        # Normalize density map to 0-255
        norm_map = cv2.normalize(self.density_map, None, 0, 255, cv2.NORM_MINMAX)
        norm_map = np.uint8(norm_map)
        colored_map = cv2.applyColorMap(norm_map, colormap)

        if base_img is not None:
            if base_img.shape[:2] != self.density_map.shape[:2]:
                colored_map = cv2.resize(colored_map, (base_img.shape[1], base_img.shape[0]))
            return cv2.addWeighted(base_img, 1.0 - alpha, colored_map, alpha, 0)

        return colored_map

    def draw_trails(self, img, tracks, max_length=30, color=(0, 140, 255)):
        """Draw smooth fading motion trails for all tracks."""
        track_dict = tracks if isinstance(tracks, dict) else {t['track_id']: t for t in tracks}

        for _, track in track_dict.items():
            history = track.get('history', [])
            pts = [p['center'] for p in history[-max_length:]]
            if len(pts) < 2:
                continue

            for i in range(1, len(pts)):
                thickness = max(1, int(4 * (i / len(pts))))
                cv2.line(img, pts[i - 1], pts[i], color, thickness, cv2.LINE_AA)

        return img
