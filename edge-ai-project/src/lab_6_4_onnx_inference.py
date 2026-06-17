import onnxruntime as ort

print("ONNX Runtime version:", ort.__version__)
print("Available providers:", ort.get_available_providers())

session = ort.InferenceSession(
    "yolov8n.onnx",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
)

print(f"Active provider: {session.get_providers()[0]}")

for inp in session.get_inputs():
    print(f"Input : {inp.name}, Shape: {inp.shape}, Type: {inp.type}")

for out in session.get_outputs():
    print(f"Output: {out.name}, Shape: {out.shape}, Type: {out.type}")
