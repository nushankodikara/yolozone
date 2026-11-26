'''
YoloZone Object Training Helper Module
Simplifies dataset creation, annotation, fine-tuning, and deployment for adding custom items to YOLO models.
Developed by:
    - Nushan Kodikara
Contact:
    - nushankodi@gmail.com
'''
import os
import shutil
import random
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import cv2
from ultralytics import YOLO
from .models import normalize_model_name


class DatasetHelper:
    """Helper to initialize and manage YOLO dataset directory structure and data.yaml."""

    def __init__(self, dataset_dir: str, classes: List[str]):
        """Initialize DatasetHelper.

        Args:
            dataset_dir: Root directory for the dataset.
            classes: List of class names to detect (e.g. ['drone', 'package', 'forklift']).
        """
        self.dataset_dir = os.path.abspath(dataset_dir)
        self.classes = classes
        self.class_to_id = {cls_name: idx for idx, cls_name in enumerate(classes)}

        # Define subdirectories
        self.images_train = os.path.join(self.dataset_dir, "images", "train")
        self.images_val = os.path.join(self.dataset_dir, "images", "val")
        self.labels_train = os.path.join(self.dataset_dir, "labels", "train")
        self.labels_val = os.path.join(self.dataset_dir, "labels", "val")
        self.data_yaml_path = os.path.join(self.dataset_dir, "data.yaml")

        self.setup_directories()
        self.generate_yaml()

    def setup_directories(self):
        """Create necessary dataset folder structure."""
        for path in [self.images_train, self.images_val, self.labels_train, self.labels_val]:
            os.makedirs(path, exist_ok=True)

    def generate_yaml(self) -> str:
        """Write YOLO standard data.yaml configuration file."""
        names_yaml = "\n".join([f"  {idx}: {name}" for idx, name in enumerate(self.classes)])
        yaml_content = f"""# YoloZone Generated Dataset Configuration
path: {self.dataset_dir}
train: images/train
val: images/val

names:
{names_yaml}
"""
        with open(self.data_yaml_path, "w", encoding="utf-8") as f:
            f.write(yaml_content)
        return self.data_yaml_path

    def add_sample(self, image: np.ndarray, bboxes: List[Tuple[str, Tuple[int, int, int, int]]],
                   split: Optional[str] = None, val_ratio: float = 0.2, sample_id: Optional[str] = None) -> str:
        """Add an image sample with bounding boxes to the dataset.

        Args:
            image: BGR image numpy array.
            bboxes: List of (class_name, (x1, y1, x2, y2)) in pixel coordinates.
            split: 'train', 'val', or None (random split based on val_ratio).
            val_ratio: Probability of assigning to validation set if split is None.
            sample_id: Optional custom filename prefix.

        Returns:
            Saved image path.
        """
        h, w = image.shape[:2]
        if split is None:
            split = "val" if random.random() < val_ratio else "train"

        img_dir = self.images_val if split == "val" else self.images_train
        lbl_dir = self.labels_val if split == "val" else self.labels_train

        sample_name = sample_id or f"sample_{len(os.listdir(img_dir)):05d}_{random.randint(1000, 9999)}"
        img_filename = f"{sample_name}.jpg"
        lbl_filename = f"{sample_name}.txt"

        img_path = os.path.join(img_dir, img_filename)
        lbl_path = os.path.join(lbl_dir, lbl_filename)

        # Save image
        cv2.imwrite(img_path, image)

        # Convert bounding boxes to normalized YOLO format (class_id x_center y_center width height)
        label_lines = []
        for cls_name, (x1, y1, x2, y2) in bboxes:
            cls_id = self.class_to_id.get(cls_name)
            if cls_id is None:
                continue

            cx = ((x1 + x2) / 2.0) / w
            cy = ((y1 + y2) / 2.0) / h
            bw = abs(x2 - x1) / w
            bh = abs(y2 - y1) / h

            # Clamp coordinates to [0, 1]
            cx, cy = max(0.0, min(1.0, cx)), max(0.0, min(1.0, cy))
            bw, bh = max(0.0, min(1.0, bw)), max(0.0, min(1.0, bh))

            label_lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        with open(lbl_path, "w", encoding="utf-8") as f:
            f.write("\n".join(label_lines) + "\n")

        return img_path

    def get_summary(self) -> Dict[str, Any]:
        """Get dataset statistics (counts of train/val images and labels)."""
        train_count = len([f for f in os.listdir(self.images_train) if f.endswith(('.jpg', '.png'))])
        val_count = len([f for f in os.listdir(self.images_val) if f.endswith(('.jpg', '.png'))])
        return {
            "dataset_dir": self.dataset_dir,
            "classes": self.classes,
            "num_classes": len(self.classes),
            "train_samples": train_count,
            "val_samples": val_count,
            "total_samples": train_count + val_count,
            "yaml_path": self.data_yaml_path
        }


class ObjectTrainer:
    """Helper to train and fine-tune YOLO models with custom datasets effortlessly."""

    def __init__(self, base_model: str = "yolov8s.pt", dataset: Optional[Union[str, DatasetHelper]] = None):
        """Initialize ObjectTrainer.

        Args:
            base_model: Pretrained base weights (e.g. 'yolov8n.pt', 'yolo11s.pt', 'yolov9c.pt').
            dataset: DatasetHelper instance or path to data.yaml.
        """
        self.base_model_name = normalize_model_name(base_model)
        self.model = YOLO(self.base_model_name)
        self.dataset = dataset
        self.data_yaml_path = dataset.data_yaml_path if isinstance(dataset, DatasetHelper) else dataset
        self.training_results = None
        self.best_model_path = None

    def create_dataset(self, dataset_dir: str, classes: List[str]) -> DatasetHelper:
        """Create and link a new dataset for training."""
        self.dataset = DatasetHelper(dataset_dir, classes)
        self.data_yaml_path = self.dataset.data_yaml_path
        return self.dataset

    def train(self, epochs: int = 30, imgsz: int = 640, batch: int = 16,
              device: str = "cpu", project: str = "yolozone_runs", name: str = "train_custom",
              lr0: float = 0.01, workers: int = 2, **kwargs) -> Dict[str, Any]:
        """Start model training / fine-tuning.

        Args:
            epochs: Number of training epochs.
            imgsz: Input image resolution.
            batch: Batch size.
            device: 'cpu', 'cuda', 'mps', or specific GPU ID.
            project: Directory to save run logs and weights.
            name: Run name.
            lr0: Initial learning rate.
            workers: DataLoader workers.
            **kwargs: Extra arguments passed directly to ultralytics YOLO.train().

        Returns:
            Dictionary containing best weights path, metrics, and training summary.
        """
        if not self.data_yaml_path or not os.path.exists(self.data_yaml_path):
            raise FileNotFoundError(f"data.yaml not found at '{self.data_yaml_path}'. Please create or specify a dataset.")

        results = self.model.train(
            data=self.data_yaml_path,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=device,
            project=project,
            name=name,
            lr0=lr0,
            workers=workers,
            **kwargs
        )

        self.training_results = results
        # Determine best weights path
        save_dir = getattr(results, 'save_dir', os.path.join(project, name))
        best_pt = os.path.join(save_dir, "weights", "best.pt")
        last_pt = os.path.join(save_dir, "weights", "last.pt")

        self.best_model_path = best_pt if os.path.exists(best_pt) else (last_pt if os.path.exists(last_pt) else None)

        return {
            "best_weights": self.best_model_path,
            "save_dir": str(save_dir),
            "epochs": epochs,
            "device": device
        }

    def export(self, format: str = "onnx", imgsz: int = 640, **kwargs) -> str:
        """Export the trained model to production formats (onnx, torchscript, engine, etc.)."""
        model_to_export = YOLO(self.best_model_path) if self.best_model_path else self.model
        exported_path = model_to_export.export(format=format, imgsz=imgsz, **kwargs)
        return str(exported_path)

    def test(self, image_or_video: Union[str, np.ndarray], conf: float = 0.25, device: str = "cpu"):
        """Test the trained or current model on a sample image or frame."""
        eval_model = YOLO(self.best_model_path) if self.best_model_path else self.model
        return eval_model(image_or_video, conf=conf, device=device)[0]
