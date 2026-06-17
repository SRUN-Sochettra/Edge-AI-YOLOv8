from pathlib import Path
import shutil

from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models"
PT_MODEL_PATH = MODEL_DIR / "yolov8n.pt"
ONNX_MODEL_PATH = MODEL_DIR / "yolov8n.onnx"


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    if not PT_MODEL_PATH.exists():
        print(f"ERROR: Model not found: {PT_MODEL_PATH}")
        print("Fix: put yolov8n.pt inside the models folder.")
        return

    print("Loading YOLOv8n model...")
    model = YOLO(str(PT_MODEL_PATH))

    print("Exporting to ONNX format...")

    export_path = model.export(
        format="onnx",
        imgsz=640,
        simplify=True,
        opset=17,
        dynamic=False
    )

    export_path = Path(export_path)

    if export_path.exists():
        if export_path.resolve() != ONNX_MODEL_PATH.resolve():
            shutil.move(str(export_path), str(ONNX_MODEL_PATH))

        size_mb = ONNX_MODEL_PATH.stat().st_size / (1024 * 1024)
        print(f"Export successful: {ONNX_MODEL_PATH} ({size_mb:.1f} MB)")
    else:
        print("Export may have failed. Check for error messages above.")


if __name__ == "__main__":
    main()