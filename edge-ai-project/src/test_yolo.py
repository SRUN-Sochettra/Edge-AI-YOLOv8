from pathlib import Path

from ultralytics import YOLO
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "yolov8n.pt"


def main():
    if not MODEL_PATH.exists():
        print(f"ERROR: Model not found: {MODEL_PATH}")
        print("Fix: put yolov8n.pt inside the models folder.")
        return

    model = YOLO(str(MODEL_PATH))

    print(f"Model loaded: {len(model.names)} classes")

    test_img = np.random.randint(
        0,
        255,
        (480, 640, 3),
        dtype=np.uint8
    )

    results = model(test_img, verbose=False)

    print(f"Inference OK — {len(results[0].boxes)} detections on random noise")
    print("\nEnvironment verification complete. All systems operational.")


if __name__ == "__main__":
    main()