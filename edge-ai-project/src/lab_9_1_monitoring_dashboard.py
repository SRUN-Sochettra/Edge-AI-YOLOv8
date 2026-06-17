from pathlib import Path
import time
import csv
from collections import Counter, deque
from datetime import datetime

from ultralytics import YOLO
import cv2
import numpy as np
import psutil


PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "models" / "yolov8n.pt"
LOG_DIR = PROJECT_ROOT / "logs"
LOG_FILE = LOG_DIR / "dashboard_metrics_log.csv"

LOG_DIR.mkdir(parents=True, exist_ok=True)

WINDOW_NAME = "Edge AI Dashboard"

CAMERA_INDEX = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

CONFIDENCE = 0.50

VIDEO_PANEL_WIDTH = 960
VIDEO_PANEL_HEIGHT = 540
INFO_PANEL_WIDTH = 420

FPS_HISTORY_SIZE = 10
LOG_INTERVAL_SECONDS = 1.0


def initialize_log_file():
    if not LOG_FILE.exists():
        with open(LOG_FILE, "w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow([
                "Timestamp",
                "Frame ID",
                "FPS",
                "CPU Percent",
                "RAM Used MB",
                "RAM Percent",
                "Detection Count",
                "Top Class",
                "Uptime Seconds"
            ])


def log_metrics(frame_id, fps, cpu_percent, ram_used_mb, ram_percent, detection_count, top_class, uptime):
    with open(LOG_FILE, "a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            frame_id,
            f"{fps:.2f}",
            f"{cpu_percent:.1f}",
            f"{ram_used_mb:.1f}",
            f"{ram_percent:.1f}",
            detection_count,
            top_class,
            f"{uptime:.1f}"
        ])


def draw_text(frame, text, position, color=(255, 255, 255), scale=0.65, thickness=2):
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


def draw_metric_bar(panel, label, value, max_value, y, color):
    x = 25
    width = INFO_PANEL_WIDTH - 50
    height = 22

    percentage = 0 if max_value == 0 else min(value / max_value, 1.0)
    filled_width = int(width * percentage)

    draw_text(panel, label, (x, y - 8), (220, 220, 220), 0.55, 1)

    cv2.rectangle(panel, (x, y), (x + width, y + height), (60, 60, 60), -1)
    cv2.rectangle(panel, (x, y), (x + filled_width, y + height), color, -1)
    cv2.rectangle(panel, (x, y), (x + width, y + height), (120, 120, 120), 1)

    draw_text(
        panel,
        f"{value:.1f}",
        (x + width - 75, y + 17),
        (255, 255, 255),
        0.5,
        1
    )


def create_info_panel(
    fps,
    cpu_percent,
    ram_used_mb,
    ram_percent,
    detection_count,
    class_counter,
    frame_id,
    uptime,
    session_total_detections
):
    panel = np.zeros((VIDEO_PANEL_HEIGHT, INFO_PANEL_WIDTH, 3), dtype=np.uint8)
    panel[:] = (28, 28, 28)

    # Header
    cv2.rectangle(panel, (0, 0), (INFO_PANEL_WIDTH, 70), (15, 15, 15), -1)
    draw_text(panel, "EDGE AI DASHBOARD", (25, 32), (0, 255, 255), 0.8, 2)
    draw_text(panel, "Real-Time Monitoring", (25, 58), (180, 180, 180), 0.5, 1)

    # Performance metrics
    draw_text(panel, "Performance", (25, 105), (255, 255, 255), 0.75, 2)

    fps_color = (0, 255, 0) if fps >= 20 else (0, 200, 255) if fps >= 10 else (0, 0, 255)
    draw_metric_bar(panel, "FPS", fps, 60, 130, fps_color)

    cpu_color = (0, 255, 0) if cpu_percent < 60 else (0, 200, 255) if cpu_percent < 85 else (0, 0, 255)
    draw_metric_bar(panel, "CPU %", cpu_percent, 100, 185, cpu_color)

    ram_color = (0, 255, 0) if ram_percent < 60 else (0, 200, 255) if ram_percent < 85 else (0, 0, 255)
    draw_metric_bar(panel, "RAM %", ram_percent, 100, 240, ram_color)

    # Detection metrics
    draw_text(panel, "Detection Stats", (25, 305), (255, 255, 255), 0.75, 2)
    draw_text(panel, f"Current Objects : {detection_count}", (25, 338), (0, 255, 0), 0.6, 1)
    draw_text(panel, f"Total Detections: {session_total_detections}", (25, 366), (0, 255, 255), 0.6, 1)

    y = 400
    draw_text(panel, "Top Classes:", (25, y), (220, 220, 220), 0.6, 1)

    if class_counter:
        for class_name, count in class_counter.most_common(4):
            y += 25
            draw_text(panel, f"- {class_name}: {count}", (35, y), (255, 255, 255), 0.55, 1)
    else:
        y += 25
        draw_text(panel, "- No detections", (35, y), (160, 160, 160), 0.55, 1)

    # Session info
    draw_text(panel, "Session", (25, 500), (255, 255, 255), 0.7, 2)
    draw_text(panel, f"Frame: {frame_id}", (25, 525), (200, 200, 200), 0.5, 1)
    draw_text(panel, f"Uptime: {uptime:.1f}s", (180, 525), (200, 200, 200), 0.5, 1)

    return panel


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

    print("\nDashboard Active")
    print("-----------------------------------")
    print("Q = Quit")
    print("-----------------------------------\n")

    frame_id = 0
    session_total_detections = 0

    fps_history = deque(maxlen=FPS_HISTORY_SIZE)

    session_start = time.time()
    last_log_time = 0

    process = psutil.Process()

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Frame capture failed.")
            break

        frame_id += 1
        frame_start = time.perf_counter()

        results = model(frame, conf=CONFIDENCE, verbose=False)

        inference_time = time.perf_counter() - frame_start
        fps = 1.0 / inference_time if inference_time > 0 else 0

        fps_history.append(fps)
        smooth_fps = sum(fps_history) / len(fps_history)

        display = frame.copy()
        class_counter = Counter()

        boxes = results[0].boxes
        detection_count = len(boxes)

        session_total_detections += detection_count

        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = model.names[cls_id]
            class_counter[class_name] += 1

            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].cpu().numpy()]
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
                0.55,
                1
            )

        # Resize video panel
        video_panel = cv2.resize(display, (VIDEO_PANEL_WIDTH, VIDEO_PANEL_HEIGHT))

        # Small overlay on video
        cv2.rectangle(video_panel, (0, 0), (250, 75), (20, 20, 20), -1)
        draw_text(video_panel, f"FPS: {smooth_fps:.1f}", (15, 30), (0, 255, 255), 0.65, 2)
        draw_text(video_panel, f"Objects: {detection_count}", (15, 60), (0, 255, 0), 0.65, 2)

        cpu_percent = psutil.cpu_percent(interval=None)
        memory = psutil.virtual_memory()
        ram_percent = memory.percent
        ram_used_mb = process.memory_info().rss / (1024 ** 2)

        uptime = time.time() - session_start

        top_class = class_counter.most_common(1)[0][0] if class_counter else "None"

        info_panel = create_info_panel(
            fps=smooth_fps,
            cpu_percent=cpu_percent,
            ram_used_mb=ram_used_mb,
            ram_percent=ram_percent,
            detection_count=detection_count,
            class_counter=class_counter,
            frame_id=frame_id,
            uptime=uptime,
            session_total_detections=session_total_detections
        )

        dashboard = np.hstack([video_panel, info_panel])

        cv2.imshow(WINDOW_NAME, dashboard)

        current_time = time.time()
        if current_time - last_log_time >= LOG_INTERVAL_SECONDS:
            log_metrics(
                frame_id=frame_id,
                fps=smooth_fps,
                cpu_percent=cpu_percent,
                ram_used_mb=ram_used_mb,
                ram_percent=ram_percent,
                detection_count=detection_count,
                top_class=top_class,
                uptime=uptime
            )
            last_log_time = current_time

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    print("\nProgram terminated.")
    print(f"Log file: {LOG_FILE}")


if __name__ == "__main__":
    main()