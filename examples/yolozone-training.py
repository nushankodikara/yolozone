import sys
import os
import cv2

# Add parent directory to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from yolozone import DatasetHelper, ObjectTrainer


def main():
    print("=== YoloZone Object Training Helper Demo ===")
    dataset_dir = "custom_drone_dataset"

    # 1. Initialize dataset structure and data.yaml
    classes = ["drone", "package"]
    dataset = DatasetHelper(dataset_dir, classes=classes)
    print(f"Initialized dataset at: {dataset.dataset_dir}")
    print(f"Generated YAML: {dataset.data_yaml_path}")

    # 2. Add sample images and bounding box annotations
    cap = cv2.VideoCapture("videos/training-sample-01.mp4")
    frame_idx = 0
    while cap.isOpened() and frame_idx < 20:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if frame_idx % 4 == 0:
            dataset.add_sample(frame, [
                ("drone", (150, 100, 230, 160)),
                ("package", (400, 300, 520, 420))
            ])
    cap.release()

    summary = dataset.get_summary()
    print("Dataset Summary:", summary)

    # 3. Initialize Trainer with a lightweight base model
    trainer = ObjectTrainer(base_model="yolov8n.pt", dataset=dataset)
    print(f"Trainer ready with model: {trainer.base_model_name}")
    print("To start training, run: trainer.train(epochs=30, imgsz=640, device='cpu')")

    # Clean up generated demo dataset folder
    import shutil
    shutil.rmtree(dataset_dir, ignore_errors=True)
    print("Demo completed successfully!")


if __name__ == "__main__":
    main()
