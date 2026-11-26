'''
YoloZone Models Registry and Loader
Supports YOLOv8, YOLOv9, YOLOv10, YOLO11, and YOLO-World models.
Developed by:
    - Nushan Kodikara
Contact:
    - nushankodi@gmail.com
'''
import os
from typing import Dict, List, Optional


SUPPORTED_MODELS = {
    "yolov8": [
        "yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt",
        "yolov8n-pose.pt", "yolov8s-pose.pt", "yolov8m-pose.pt", "yolov8l-pose.pt", "yolov8x-pose.pt",
    ],
    "yolov9": [
        "yolov9t.pt", "yolov9s.pt", "yolov9m.pt", "yolov9c.pt", "yolov9e.pt",
    ],
    "yolov10": [
        "yolov10n.pt", "yolov10s.pt", "yolov10m.pt", "yolov10b.pt", "yolov10l.pt", "yolov10x.pt",
    ],
    "yolo11": [
        "yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt", "yolo11x.pt",
        "yolo11n-pose.pt", "yolo11s-pose.pt", "yolo11m-pose.pt", "yolo11l-pose.pt", "yolo11x-pose.pt",
    ],
    "yolo_world": [
        "yolov8s-world.pt", "yolov8m-world.pt", "yolov8l-world.pt", "yolov8x-world.pt",
        "yolov8s-worldv2.pt", "yolov8m-worldv2.pt", "yolov8l-worldv2.pt", "yolov8x-worldv2.pt",
    ],
}


def normalize_model_name(model_name: str) -> str:
    """Normalize model name string by ensuring .pt suffix when not an explicit path."""
    if not model_name:
        return "yolov8s.pt"
    if not (model_name.endswith(".pt") or model_name.endswith(".onnx") or model_name.endswith(".engine") or model_name.endswith(".torchscript")):
        return f"{model_name}.pt"
    return model_name


def identify_model_family(model_name: str) -> Dict[str, str]:
    """Identify the family, task, and size of a YOLO model."""
    name_lower = os.path.basename(model_name).lower()

    # Determine family
    family = "custom"
    if "yolo11" in name_lower or "yolov11" in name_lower:
        family = "yolo11"
    elif "yolov10" in name_lower:
        family = "yolov10"
    elif "yolov9" in name_lower:
        family = "yolov9"
    elif "world" in name_lower:
        family = "yolo_world"
    elif "yolov8" in name_lower:
        family = "yolov8"

    # Determine task
    task = "detect"
    if "pose" in name_lower:
        task = "pose"
    elif "seg" in name_lower:
        task = "segment"
    elif "cls" in name_lower:
        task = "classify"
    elif "obb" in name_lower:
        task = "obb"

    # Determine scale/size
    size = "unknown"
    for s, label in [("n", "nano"), ("s", "small"), ("m", "medium"), ("b", "balanced"),
                     ("l", "large"), ("x", "extra-large"), ("t", "tiny"), ("c", "compact"), ("e", "extended")]:
        if f"{s}." in name_lower or f"{s}-" in name_lower or f"{s}_" in name_lower:
            size = label
            break

    return {
        "model": model_name,
        "family": family,
        "task": task,
        "size": size,
        "is_world_model": family == "yolo_world"
    }


def list_supported_models(family: Optional[str] = None) -> List[str]:
    """Get list of officially supported YOLO models."""
    if family:
        return SUPPORTED_MODELS.get(family.lower(), [])
    all_models = []
    for models in SUPPORTED_MODELS.values():
        all_models.extend(models)
    return all_models
