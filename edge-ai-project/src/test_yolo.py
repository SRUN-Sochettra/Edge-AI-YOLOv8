from ultralytics import YOLO
import numpy as np

model = YOLO("yolov8n.pt")

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
