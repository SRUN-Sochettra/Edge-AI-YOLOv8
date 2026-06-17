from pathlib import Path
import time
import csv
from collections import defaultdict
from datetime import datetime

from ultralytics import YOLO
import cv2


PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "models" / "yolov8n.pt"
LOG_DIR = PROJECT_ROOT / "logs"
LOG_FILE = LOG_DIR / "line_crossing_log.csv"

LOG_DIR.mkdir(parents=True, exist_ok=True)

WINDOW_NAME = "Bidirectional Line-Crossing Counter"

CAMERA_INDEX = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

CONFIDENCE = 0.50
OFFSET = 30

# Set to None to track all classes.
# Set to [0] to count only people.
TARGET_CLASS_IDS = None


track_last_side = {}
track_last_center = {}
count_up = 0
count_down = 0


def initialize_log_file():
    if not LOG_FILE.exists():
        with open(LOG_FILE, "w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow([
                "Timestamp",
                "Track ID",
                "Class",
                "Direction",
                "Total Up",
                "Total Down"
            ])


def log_crossing(track_id, class_name, direction):
    with open(LOG_FILE, "a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            track_id,
            class_name,
            direction,
            count_up,
            count_down
        ])


def get_side(y, line_y):
    if y < line_y - OFFSET:
        return -1  # above line

    if y > line_y + OFFSET:
        return 1  # below line

    return 0  # inside tolerance zone


def get_track_color(track_id):
    red = (track_id * 37) % 255
    green = (track_id * 17) % 255
    blue = (track_id * 29) % 255
    return int(blue), int(green), int(red)


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
    global count_up
    global count_down

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

    print("\nLine-Crossing Counter Active")
    print("-----------------------------------")
    print("Q = Quit")
    print("R = Reset counts")
    print("-----------------------------------\n")

    fps_history = []

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Frame capture failed.")
            break

        frame_height, frame_width = frame.shape[:2]
        line_y = frame_height // 2

        start = time.perf_counter()

        results = model.track(
            frame,
            persist=True,
            conf=CONFIDENCE,
            verbose=False
        )

        inference_time = time.perf_counter() - start
        fps = 1.0 / inference_time if inference_time > 0 else 0

        fps_history.append(fps)
        if len(fps_history) > 10:
            fps_history.pop(0)

        smooth_fps = sum(fps_history) / len(fps_history)

        display = frame.copy()

        # Draw counting line and tolerance zone
        cv2.line(display, (0, line_y), (frame_width, line_y), (0, 255, 255), 3)
        cv2.line(display, (0, line_y - OFFSET), (frame_width, line_y - OFFSET), (80, 80, 80), 1)
        cv2.line(display, (0, line_y + OFFSET), (frame_width, line_y + OFFSET), (80, 80, 80), 1)

        draw_text(display, "COUNTING LINE", (20, line_y - 12), (0, 255, 255), 0.7)

        boxes = results[0].boxes
        active_tracks = 0

        if boxes is not None and boxes.id is not None:
            for box in boxes:
                track_id = int(box.id[0])
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = model.names[cls_id]

                if TARGET_CLASS_IDS is not None and cls_id not in TARGET_CLASS_IDS:
                    continue

                active_tracks += 1

                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].cpu().numpy()]

                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                current_side = get_side(cy, line_y)

                if track_id not in track_last_side:
                    track_last_side[track_id] = current_side

                previous_side = track_last_side[track_id]

                if previous_side == -1 and current_side == 1:
                    count_down += 1
                    log_crossing(track_id, class_name, "DOWN")
                    print(f"Track ID {track_id} crossed DOWN")

                elif previous_side == 1 and current_side == -1:
                    count_up += 1
                    log_crossing(track_id, class_name, "UP")
                    print(f"Track ID {track_id} crossed UP")

                if current_side != 0:
                    track_last_side[track_id] = current_side

                color = get_track_color(track_id)

                cv2.rectangle(
                    display,
                    (x1, y1),
                    (x2, y2),
                    color,
                    2
                )

                label = f"ID {track_id} | {class_name} {conf:.2f}"

                draw_text(
                    display,
                    label,
                    (x1, max(y1 - 10, 25)),
                    color,
                    0.6
                )

                cv2.circle(display, (cx, cy), 5, color, -1)

                if track_id in track_last_center:
                    cv2.line(
                        display,
                        track_last_center[track_id],
                        (cx, cy),
                        color,
                        2
                    )

                track_last_center[track_id] = (cx, cy)

        # Header panel
        cv2.rectangle(display, (0, 0), (500, 155), (20, 20, 20), -1)

        draw_text(display, f"UP Count: {count_up}", (15, 35), (0, 255, 0), 0.9)
        draw_text(display, f"DOWN Count: {count_down}", (15, 70), (0, 180, 255), 0.9)
        draw_text(display, f"Active Tracks: {active_tracks}", (15, 105), (255, 255, 255), 0.8)
        draw_text(display, f"FPS: {smooth_fps:.1f}", (15, 140), (0, 255, 255), 0.8)

        cv2.imshow(WINDOW_NAME, display)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        if key == ord("r"):
            count_up = 0
            count_down = 0
            track_last_side.clear()
            track_last_center.clear()
            print("Counts reset.")

    cap.release()
    cv2.destroyAllWindows()

    print("\nProgram terminated.")
    print(f"UP Count: {count_up}")
    print(f"DOWN Count: {count_down}")
    print(f"Log file: {LOG_FILE}")


if __name__ == "__main__":
    main()