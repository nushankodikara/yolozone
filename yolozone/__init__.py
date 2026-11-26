from .pose import PoseDetector
from .objects import ObjectDetector
from .tracker import ObjectTracker
from .hand import HandDetector, HandTracker
from .face import FaceDetector, FaceTracker
from .helpers import (
    ZoneTracker,
    LineCrossingHelper,
    SpeedEstimator,
    TrajectoryVisualizer,
)
from .training import ObjectTrainer, DatasetHelper
from .models import (
    SUPPORTED_MODELS,
    normalize_model_name,
    identify_model_family,
    list_supported_models,
)

# Convenient aliases
Pose = PoseDetector
Objects = ObjectDetector
Tracker = ObjectTracker

__all__ = [
    "PoseDetector",
    "ObjectDetector",
    "ObjectTracker",
    "HandDetector",
    "HandTracker",
    "FaceDetector",
    "FaceTracker",
    "ZoneTracker",
    "LineCrossingHelper",
    "SpeedEstimator",
    "TrajectoryVisualizer",
    "ObjectTrainer",
    "DatasetHelper",
    "SUPPORTED_MODELS",
    "normalize_model_name",
    "identify_model_family",
    "list_supported_models",
    "Pose",
    "Objects",
    "Tracker",
]

__version__ = "0.1.0"
