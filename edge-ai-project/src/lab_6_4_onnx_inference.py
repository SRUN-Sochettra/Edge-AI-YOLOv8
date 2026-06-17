from pathlib import Path

import onnxruntime as ort


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ONNX_MODEL_PATH = PROJECT_ROOT / "models" / "yolov8n.onnx"


def main():
    if not ONNX_MODEL_PATH.exists():
        print(f"ERROR: ONNX model not found: {ONNX_MODEL_PATH}")
        print("Fix: run python src\\lab_6_3_export_onnx.py first.")
        return

    print("ONNX Runtime version:", ort.__version__)
    print("Available providers:", ort.get_available_providers())

    available_providers = ort.get_available_providers()

    preferred_providers = []

    if "CUDAExecutionProvider" in available_providers:
        preferred_providers.append("CUDAExecutionProvider")

    preferred_providers.append("CPUExecutionProvider")

    session = ort.InferenceSession(
        str(ONNX_MODEL_PATH),
        providers=preferred_providers
    )

    print(f"Active provider: {session.get_providers()[0]}")

    for inp in session.get_inputs():
        print(f"Input : {inp.name}, Shape: {inp.shape}, Type: {inp.type}")

    for out in session.get_outputs():
        print(f"Output: {out.name}, Shape: {out.shape}, Type: {out.type}")


if __name__ == "__main__":
    main()