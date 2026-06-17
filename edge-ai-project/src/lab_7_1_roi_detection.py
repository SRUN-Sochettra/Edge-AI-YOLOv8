from ultralytics import YOLO
import cv2
import time
import csv
from pathlib import Path
from collections import Counter
from datetime import datetime


# ==========================================================
# Paths
# ==========================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name == "src" else SCRIPT_DIR

MODEL_PATH = PROJECT_ROOT / "models" / "yolov8n.pt"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
SNAPSHOT_DIR = OUTPUT_DIR / "snapshots"
LOG_DIR = PROJECT_ROOT / "logs"
LOG_FILE = LOG_DIR / "roi_detection_log.csv"

SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================================
# Configuration
# ==========================================================

WINDOW_NAME = "ROI Analytics System"
CONFIDENCE = 0.50
MIN_CONFIDENCE = 0.10
MAX_CONFIDENCE = 0.95
CONFIDENCE_STEP = 0.05

CAMERA_INDEX = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720


# ==========================================================
# ROI State
# ==========================================================

roi_pts = []
drawing = False
roi_set = False
current_mouse_pos = None


# ==========================================================
# Mouse Callback
# ==========================================================

def mouse_callback(event, x, y, flags, param):
    global roi_pts
    global drawing
    global roi_set
    global current_mouse_pos

    current_mouse_pos = (x, y)

    if event == cv2.EVENT_LBUTTONDOWN:
        roi_pts = [(x, y)]
        drawing = True
        roi_set = False

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            current_mouse_pos = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        roi_pts.append((x, y))
        drawing = False
        roi_set = True


# ==========================================================
# Helpers
# ==========================================================

def get_roi_bounds(roi):
    x1 = min(roi[0][0], roi[1][0])
    y1 = min(roi[0][1], roi[1][1])
    x2 = max(roi[0][0], roi[1][0])
    y2 = max(roi[0][1], roi[1][1])

    return x1, y1, x2, y2


def is_inside_roi(box_xyxy, roi):
    x1, y1, x2, y2 = get_roi_bounds(roi)

    cx = int((box_xyxy[0] + box_xyxy[2]) / 2)
    cy = int((box_xyxy[1] + box_xyxy[3]) / 2)

    return x1 <= cx <= x2 and y1 <= cy <= y2


def draw_text(frame, text, position, color=(255, 255, 255), scale=0.7, thickness=2):
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


def draw_roi_overlay(frame):
    if roi_set and len(roi_pts) == 2:
        x1, y1, x2, y2 = get_roi_bounds(roi_pts)

        overlay = frame.copy()

        cv2.rectangle(
            overlay,
            (x1, y1),
            (x2, y2),
            (0, 255, 255),
            -1
        )

        cv2.addWeighted(
            overlay,
            0.20,
            frame,
            0.80,
            0,
            frame
        )

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 255),
            2
        )

        width = x2 - x1
        height = y2 - y1
        area = width * height

        draw_text(frame, f"ROI: {width}x{height} | Area: {area}", (x1, max(y1 - 10, 30)), (0, 255, 255), 0.6)

    elif drawing and len(roi_pts) == 1 and current_mouse_pos is not None:
        cv2.rectangle(
            frame,
            roi_pts[0],
            current_mouse_pos,
            (0, 255, 255),
            2
        )


def initialize_log_file():
    if not LOG_FILE.exists():
        with open(LOG_FILE, "w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow([
                "Timestamp",
                "Frame ID",
                "Class",
                "Confidence",
                "BBox X1",
                "BBox Y1",
                "BBox X2",
                "BBox Y2",
                "ROI X1",
                "ROI Y1",
                "ROI X2",
                "ROI Y2"
            ])


def log_detection(frame_id, class_name, confidence, box_xyxy):
    if roi_set and len(roi_pts) == 2:
        rx1, ry1, rx2, ry2 = get_roi_bounds(roi_pts)
    else:
        rx1, ry1, rx2, ry2 = "", "", "", ""

    x1, y1, x2, y2 = [int(v) for v in box_xyxy]

    with open(LOG_FILE, "a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            frame_id,
            class_name,
            f"{confidence:.2f}",
            x1,
            y1,
            x2,
            y2,
            rx1,
            ry1,
            rx2,
            ry2
        ])


# ==========================================================
# Main Program
# ==========================================================

def main():
    global CONFIDENCE
    global roi_pts
    global roi_set
    global drawing

    initialize_log_file()

    if not MODEL_PATH.exists():
        print(f"ERROR: Model not found: {MODEL_PATH}")
        print("Fix: move yolov8n.pt into the models folder.")
        return

    print("Loading YOLOv8 model...")
    model = YOLO(str(MODEL_PATH))

    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        print("Fix: close Zoom/Teams/OBS/Camera app or change CAMERA_INDEX from 0 to 1.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    cv2.namedWindow(WINDOW_NAME)
    cv2.setMouseCallback(WINDOW_NAME, mouse_callback)

    print("\nControls")
    print("-----------------------------------")
    print("Mouse : Draw ROI")
    print("R     : Reset ROI")
    print("S     : Save Snapshot")
    print("+     : Increase Confidence")
    print("-     : Decrease Confidence")
    print("UP    : Increase Confidence")
    print("DOWN  : Decrease Confidence")
    print("Q     : Quit")
    print("-----------------------------------\n")

    frame_id = 0
    fps_history = []

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Frame capture failed.")
            break

        frame_id += 1
        start = time.perf_counter()

        results = model(
            frame,
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

        draw_roi_overlay(display)

        class_counter = Counter()
        roi_count = 0

        boxes = results[0].boxes

        for box in boxes:
            xyxy = box.xyxy[0].cpu().numpy()
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = model.names[cls_id]

            if roi_set:
                inside = is_inside_roi(xyxy, roi_pts)
            else:
                inside = True

            if not inside:
                continue

            x1, y1, x2, y2 = [int(v) for v in xyxy]

            label = f"{class_name} {conf:.2f}"

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

            roi_count += 1
            class_counter[class_name] += 1

            log_detection(
                frame_id=frame_id,
                class_name=class_name,
                confidence=conf,
                box_xyxy=xyxy
            )

        # Dashboard text
        draw_text(display, f"FPS: {smooth_fps:.1f}", (10, 30), (0, 255, 255))
        draw_text(display, f"Confidence: {CONFIDENCE:.2f}", (10, 60), (255, 255, 255))
        draw_text(display, f"Objects in ROI: {roi_count}", (10, 90), (0, 255, 0))

        if roi_set:
            draw_text(display, "ROI: Active", (10, 120), (0, 255, 255))
        else:
            draw_text(display, "ROI: Not set - detecting full frame", (10, 120), (0, 180, 255))

        y = 155
        draw_text(display, "Class Statistics:", (10, y), (255, 255, 255), 0.6)

        for class_name, count in class_counter.most_common(5):
            y += 25
            draw_text(display, f"- {class_name}: {count}", (10, y), (255, 255, 255), 0.6)

        cv2.imshow(WINDOW_NAME, display)

        key = cv2.waitKeyEx(1)

        # Q or q
        if key in [ord("q"), ord("Q")]:
            break

        # R or r
        elif key in [ord("r"), ord("R")]:
            roi_pts = []
            roi_set = False
            drawing = False
            print("ROI reset.")

        # S or s
        elif key in [ord("s"), ord("S")]:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            snapshot_path = SNAPSHOT_DIR / f"snapshot_{timestamp}.jpg"
            cv2.imwrite(str(snapshot_path), display)
            print(f"Snapshot saved: {snapshot_path}")

        # + key or Up arrow
        elif key in [ord("+"), ord("="), 82, 2490368]:
            CONFIDENCE = min(MAX_CONFIDENCE, CONFIDENCE + CONFIDENCE_STEP)
            print(f"Confidence increased: {CONFIDENCE:.2f}")

        # - key or Down arrow
        elif key in [ord("-"), ord("_"), 84, 2621440]:
            CONFIDENCE = max(MIN_CONFIDENCE, CONFIDENCE - CONFIDENCE_STEP)
            print(f"Confidence decreased: {CONFIDENCE:.2f}")

    cap.release()
    cv2.destroyAllWindows()

    print("\nProgram terminated.")
    print(f"Log file: {LOG_FILE}")
    print(f"Snapshots folder: {SNAPSHOT_DIR}")


if __name__ == "__main__":
    main()