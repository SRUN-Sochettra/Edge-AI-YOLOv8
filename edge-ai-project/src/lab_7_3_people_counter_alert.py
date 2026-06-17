from pathlib import Path
import time
import csv
from datetime import datetime

from ultralytics import YOLO
import cv2


PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "models" / "yolov8n.pt"
LOG_DIR = PROJECT_ROOT / "logs"
LOG_FILE = LOG_DIR / "people_counter_alert_log.csv"

LOG_DIR.mkdir(parents=True, exist_ok=True)

WINDOW_NAME = "People Counter with Alert System"

CAMERA_INDEX = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

CONFIDENCE = 0.50
TARGET_CLASS_ID = 0  # COCO class 0 = person
ALERT_THRESHOLD = 3


def initialize_log_file():
    if not LOG_FILE.exists():
        with open(LOG_FILE, "w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow([
                "Timestamp",
                "Frame ID",
                "People Count",
                "Alert Threshold",
                "Alert Active",
                "FPS"
            ])


def log_status(frame_id, people_count, alert_active, fps):
    with open(LOG_FILE, "a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            frame_id,
            people_count,
            ALERT_THRESHOLD,
            alert_active,
            f"{fps:.2f}"
        ])


def draw_text(frame, text, position, color=(255, 255, 255), scale=0.8, thickness=2):
    cv2.putText(
        frame,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA
    )


def main():
    initialize_log_file()

    if not MODEL_PATH.exists():
        print(f"ERROR: Model not found: {MODEL_PATH}")
        print("Fix: put yolov8n.pt inside the models folder.")
        return

    print("Loading YOLOv8 model...")
    model = YOLO(str(MODEL_PATH))

    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        print("Fix: close apps using the camera or change CAMERA_INDEX from 0 to 1.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    print("\nPeople Counter Active")
    print("-----------------------------------")
    print(f"Alert threshold: {ALERT_THRESHOLD}")
    print("Q = Quit")
    print("-----------------------------------\n")

    frame_id = 0
    fps_history = []
    last_log_time = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Frame capture failed.")
            break

        frame_id += 1
        start = time.perf_counter()

        results = model(frame, conf=CONFIDENCE, verbose=False)

        inference_time = time.perf_counter() - start
        fps = 1.0 / inference_time if inference_time > 0 else 0

        fps_history.append(fps)
        if len(fps_history) > 10:
            fps_history.pop(0)

        smooth_fps = sum(fps_history) / len(fps_history)

        display = frame.copy()
        people_count = 0

        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])

            if cls_id != TARGET_CLASS_ID:
                continue

            people_count += 1

            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].cpu().numpy()]
            label = f"person {conf:.2f}"

            cv2.rectangle(
                display,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            draw_text(
                display,
                label,
                (x1, max(y1 - 10, 25)),
                (0, 255, 0),
                0.6
            )

        alert_active = people_count > ALERT_THRESHOLD

        if alert_active:
            status_text = "ALERT: OVER CAPACITY"
            status_color = (0, 0, 255)

            # Flashing red border
            if int(time.time() * 2) % 2 == 0:
                cv2.rectangle(
                    display,
                    (5, 5),
                    (display.shape[1] - 5, display.shape[0] - 5),
                    (0, 0, 255),
                    10
                )
        else:
            status_text = "Normal"
            status_color = (0, 255, 0)

        # Header panel
        cv2.rectangle(display, (0, 0), (520, 145), (20, 20, 20), -1)

        draw_text(display, f"People Count: {people_count}", (15, 35), (255, 255, 255), 0.9)
        draw_text(display, f"Threshold: {ALERT_THRESHOLD}", (15, 70), (255, 255, 255), 0.8)
        draw_text(display, f"Status: {status_text}", (15, 105), status_color, 0.8)
        draw_text(display, f"FPS: {smooth_fps:.1f}", (15, 135), (0, 255, 255), 0.7)

        cv2.imshow(WINDOW_NAME, display)

        # Log once per second
        current_time = time.time()
        if current_time - last_log_time >= 1.0:
            log_status(frame_id, people_count, alert_active, smooth_fps)
            last_log_time = current_time

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    print("\nProgram terminated.")
    print(f"Log file: {LOG_FILE}")


if __name__ == "__main__":
    main()