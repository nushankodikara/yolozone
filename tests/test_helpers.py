import unittest
import os
import sys
import json
import shutil
import numpy as np
import cv2

# Add parent directory to sys.path so yolozone can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from yolozone import (
    HandDetector,
    HandTracker,
    FaceDetector,
    FaceTracker,
    ZoneTracker,
    LineCrossingHelper,
    SpeedEstimator,
    TrajectoryVisualizer,
    DatasetHelper,
    ObjectTrainer,
    normalize_model_name,
    identify_model_family,
    list_supported_models,
    __version__,
)


class MockKeypoints:
    """Mock keypoints structure resembling Ultralytics Keypoints output."""
    def __init__(self, keypoints_array):
        # shape: (num_subjects, 17, 2)
        self.xy = np.array(keypoints_array, dtype=float)


class TestModelRegistry(unittest.TestCase):
    def test_normalize_model_name(self):
        self.assertEqual(normalize_model_name("yolo11s"), "yolo11s.pt")
        self.assertEqual(normalize_model_name("yolov10n.pt"), "yolov10n.pt")
        self.assertEqual(normalize_model_name("custom.onnx"), "custom.onnx")

    def test_identify_model_family(self):
        info_11 = identify_model_family("yolo11m-pose.pt")
        self.assertEqual(info_11['family'], "yolo11")
        self.assertEqual(info_11['task'], "pose")
        self.assertEqual(info_11['size'], "medium")

        info_10 = identify_model_family("yolov10n.pt")
        self.assertEqual(info_10['family'], "yolov10")
        self.assertEqual(info_10['task'], "detect")
        self.assertEqual(info_10['size'], "nano")

        info_world = identify_model_family("yolov8s-world.pt")
        self.assertTrue(info_world['is_world_model'])

    def test_list_supported_models(self):
        yolo11_models = list_supported_models("yolo11")
        self.assertIn("yolo11n.pt", yolo11_models)
        self.assertIn("yolo11n-pose.pt", yolo11_models)

        all_models = list_supported_models()
        self.assertGreater(len(all_models), 20)


class TestHandAndFingerHelpers(unittest.TestCase):
    def setUp(self):
        pts = np.zeros((1, 17, 2))
        pts[0, 5] = [200, 200]  # Left shoulder
        pts[0, 7] = [200, 150]  # Left elbow
        pts[0, 9] = [200, 100]  # Left wrist (raised)

        pts[0, 6] = [300, 200]  # Right shoulder
        pts[0, 8] = [300, 250]  # Right elbow
        pts[0, 10] = [300, 300] # Right wrist (down)

        self.keypoints = MockKeypoints(pts)
        self.detector = HandDetector(model=None, use_mediapipe=False)
        self.detector.pose_detector = None

    def test_detect_hands_and_21_landmarks(self):
        dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
        hands = self.detector.detect_hands(dummy_img, keypoints=self.keypoints)
        self.assertEqual(len(hands), 2)

        left_hand = next(h for h in hands if h['type'] == 'left')
        right_hand = next(h for h in hands if h['type'] == 'right')

        self.assertTrue(left_hand['is_raised'])
        self.assertFalse(right_hand['is_raised'])

        # Check that 21 landmarks exist
        self.assertEqual(len(left_hand['landmarks']), 21)
        self.assertEqual(len(right_hand['landmarks']), 21)

        # Check individual finger poses
        self.assertIn('fingers', left_hand)
        for fname in ['thumb', 'index', 'middle', 'ring', 'pinky']:
            self.assertIn(fname, left_hand['fingers'])
            finger = left_hand['fingers'][fname]
            self.assertIn('state', finger)
            self.assertIn('flex_angle', finger)
            self.assertIn('tip', finger)

    def test_hand_gestures(self):
        dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
        hands = self.detector.detect_hands(dummy_img, keypoints=self.keypoints)
        for h in hands:
            self.assertIn('gesture', h)
            self.assertIn('is_pinching', h)


class TestFacePlaneAndEmotionHelpers(unittest.TestCase):
    def setUp(self):
        pts = np.zeros((1, 17, 2))
        pts[0, 0] = [320, 180]  # Nose
        pts[0, 1] = [300, 160]  # Left eye
        pts[0, 2] = [340, 160]  # Right eye
        pts[0, 3] = [280, 165]  # Left ear
        pts[0, 4] = [360, 165]  # Right ear
        self.keypoints = MockKeypoints(pts)
        self.detector = FaceDetector(model=None, use_facemesh=False)
        self.detector.pose_detector = None

    def test_detect_face_3d_plane_and_emotions(self):
        dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
        faces = self.detector.detect_faces(dummy_img, keypoints=self.keypoints)
        self.assertEqual(len(faces), 1)

        face = faces[0]
        # Mathematical 3D Plane verification
        plane = face['plane']
        self.assertIn('plane_normal', plane)
        self.assertEqual(len(plane['plane_normal']), 3)
        self.assertIn('plane_equation', plane)
        self.assertEqual(len(plane['plane_equation']), 4)
        self.assertIn('pitch_deg', plane)
        self.assertIn('yaw_deg', plane)
        self.assertIn('roll_deg', plane)

        # Emotion & Expressions verification
        self.assertIn('emotions', face)
        self.assertIn('primary_emotion', face)
        self.assertIn('expressions', face)
        self.assertIsInstance(face['emotions'], dict)
        self.assertIn('neutral', face['emotions'])


class TestObjectTrainingHelper(unittest.TestCase):
    def setUp(self):
        self.test_dir = "test_yolozone_dataset"
        self.classes = ["drone", "package"]
        self.dataset = DatasetHelper(self.test_dir, self.classes)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_dataset_generation_and_samples(self):
        # Verify structure
        self.assertTrue(os.path.exists(self.dataset.images_train))
        self.assertTrue(os.path.exists(self.dataset.labels_train))
        self.assertTrue(os.path.exists(self.dataset.data_yaml_path))

        # Add sample
        dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
        self.dataset.add_sample(dummy_img, [
            ("drone", (100, 100, 200, 200)),
            ("package", (300, 300, 450, 400))
        ], split="train", sample_id="test_01")

        summary = self.dataset.get_summary()
        self.assertEqual(summary['train_samples'], 1)
        self.assertEqual(summary['num_classes'], 2)

        # Check label file content
        lbl_file = os.path.join(self.dataset.labels_train, "test_01.txt")
        self.assertTrue(os.path.exists(lbl_file))
        with open(lbl_file, "r") as f:
            lines = f.readlines()
        self.assertEqual(len(lines), 2)
        # Verify class 0 and class 1
        self.assertTrue(lines[0].startswith("0 "))
        self.assertTrue(lines[1].startswith("1 "))

    def test_object_trainer_init(self):
        trainer = ObjectTrainer(base_model="yolov8n.pt", dataset=self.dataset)
        self.assertEqual(trainer.base_model_name, "yolov8n.pt")
        self.assertEqual(trainer.data_yaml_path, self.dataset.data_yaml_path)


class TestSpatialTrackingHelpers(unittest.TestCase):
    def test_zone_tracker(self):
        zone_polygon = [(100, 100), (300, 100), (300, 300), (100, 300)]
        entered_events = []
        exited_events = []

        def on_enter(tid, info, name):
            entered_events.append((tid, name))

        def on_exit(tid, info, dwell, name):
            exited_events.append((tid, dwell, name))

        zone = ZoneTracker(zone_polygon, name="TestZone", on_enter=on_enter, on_exit=on_exit)

        tracks_f1 = {
            1: {'track_id': 1, 'center': (200, 200), 'box': (180, 180, 220, 220), 'class': 'person'},
            2: {'track_id': 2, 'center': (400, 400), 'box': (390, 390, 410, 410), 'class': 'car'}
        }
        res1 = zone.update(tracks_f1)
        self.assertEqual(res1['occupancy'], 1)
        self.assertEqual(res1['total_entered'], 1)
        self.assertEqual(len(entered_events), 1)

        tracks_f2 = {
            1: {'track_id': 1, 'center': (210, 210), 'box': (190, 190, 230, 230), 'class': 'person'},
            2: {'track_id': 2, 'center': (250, 250), 'box': (240, 240, 260, 260), 'class': 'car'}
        }
        res2 = zone.update(tracks_f2)
        self.assertEqual(res2['occupancy'], 2)
        self.assertEqual(res2['total_entered'], 2)

        tracks_f3 = {
            2: {'track_id': 2, 'center': (250, 250), 'box': (240, 240, 260, 260), 'class': 'car'}
        }
        res3 = zone.update(tracks_f3)
        self.assertEqual(res3['occupancy'], 1)
        self.assertEqual(res3['total_exited'], 1)
        self.assertEqual(len(exited_events), 1)

    def test_line_crossing_helper(self):
        line_helper = LineCrossingHelper()
        line_helper.add_line("MidLine", start_point=(0, 200), end_point=(500, 200))

        tracks = {
            1: {
                'track_id': 1,
                'class': 'car',
                'history': [
                    {'center': (250, 180)},
                    {'center': (250, 220)}
                ]
            }
        }
        crossings = line_helper.update(tracks)
        self.assertEqual(len(crossings), 1)
        self.assertEqual(crossings[0]['track_id'], 1)
        self.assertEqual(line_helper.lines['MidLine']['total'], 1)

    def test_speed_estimator(self):
        estimator = SpeedEstimator(pixels_per_meter=10.0, fps=30.0)
        history = [
            {'center': (100, 100)},
            {'center': (200, 100)}
        ]
        kmh, mps = estimator.estimate_speed(1, history)
        self.assertAlmostEqual(mps, 300.0, places=1)
        self.assertAlmostEqual(kmh, 1080.0, places=1)

    def test_trajectory_visualizer(self):
        viz = TrajectoryVisualizer(frame_shape=(480, 640))
        tracks = {
            1: {'track_id': 1, 'center': (100, 100), 'history': [{'center': (90, 90)}, {'center': (100, 100)}]}
        }
        viz.update(tracks)
        heatmap = viz.get_heatmap()
        self.assertEqual(heatmap.shape, (480, 640, 3))

    def test_simpleversion_json(self):
        self.assertEqual(__version__, "0.1.0")
        with open("simpleversion.json", "r") as f:
            v_data = json.load(f)
        self.assertEqual(v_data['version'], "0.1.0")
        self.assertIn("3.12", v_data['python_versions'])
        self.assertIn("3.13", v_data['python_versions'])
        self.assertIn("yolo11n", v_data['supported_models'])


if __name__ == "__main__":
    unittest.main()
