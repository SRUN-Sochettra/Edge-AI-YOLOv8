from pathlib import Path
import time

from ultralytics import YOLO
import cv2


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "yolov8n.pt"

CAMERA_INDEX = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
CONFIDENCE = 0.50


def main():
    if not MODEL_PATH.exists():
        print(f"ERROR: Model not found: {MODEL_PATH}")
        print("Fix: put yolov8n.pt inside the models folder.")
        return

    model = YOLO(str(MODEL_PATH))

    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print("ERROR: Cannot open webcam. Check connection and drivers.")
        print("Fix: close apps using the webcam or change CAMERA_INDEX from 0 to 1.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    print("Webcam active. Press Q to quit.")

    fps_list = []

    while True:
        t0 = time.perf_counter()

        ret, frame = cap.read()

        if not ret:
            print("Frame capture failed.")
            break

        results = model(frame, conf=CONFIDENCE, verbose=False)
        annotated = results[0].plot()

        fps = 1.0 / (time.perf_counter() - t0)
        fps_list.append(fps)

        cv2.putText(
            annotated,
            f"FPS: {fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            2
        )

        cv2.putText(
            annotated,
            f"Objects: {len(results[0].boxes)}",
            (10, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.imshow("YOLOv8 Real-Time Detection", annotated)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    if fps_list:
        avg_fps = sum(fps_list) / len(fps_list)
        print(f"\nSession: {len(fps_list)} frames, Avg FPS: {avg_fps:.1f}")


if __name__ == "__main__":
    main()