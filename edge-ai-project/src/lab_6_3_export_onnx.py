from ultralytics import YOLO
import os

print("Loading YOLOv8n model...")
model = YOLO("yolov8n.pt")

print("Exporting to ONNX format...")

export_path = model.export(
    format="onnx",
    imgsz=640,
    simplify=True,
    opset=17,
    dynamic=False
)

if os.path.exists("yolov8n.onnx"):
    size_mb = os.path.getsize("yolov8n.onnx") / (1024 * 1024)
    print(f"Export successful: yolov8n.onnx ({size_mb:.1f} MB)")
else:
    print("Export may have failed. Check for error messages above.")