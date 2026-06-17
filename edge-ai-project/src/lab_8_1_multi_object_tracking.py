from pathlib import Path
import time
from collections import defaultdict, deque

from ultralytics import YOLO
import cv2


PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "models" / "yolov8n.pt"

WINDOW_NAME = "Real-Time Multi-Object Tracking"

CAMERA_INDEX = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

CONFIDENCE = 0.50
MAX_TRAIL_LENGTH = 50


track_history = defaultdict(lambda: deque(maxlen=MAX_TRAIL_LENGTH))
unique_track_ids_seen = set()


def get_track_color(track_id):
    # Deterministic color per track ID
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

    print("\nMulti-Object Tracking Active")
    print("-----------------------------------")
    print("Q = Quit")
    print("-----------------------------------\n")

    fps_history = []

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Frame capture failed.")
            break

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
        active_tracks = 0

        boxes = results[0].boxes

        if boxes is not None and boxes.id is not None:
            for box in boxes:
                track_id = int(box.id[0])
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = model.names[cls_id]

                unique_track_ids_seen.add(track_id)
                active_tracks += 1

                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].cpu().numpy()]

                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                track_history[track_id].append((cx, cy))

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

                points = list(track_history[track_id])

                for i in range(1, len(points)):
                    cv2.line(
                        display,
                        points[i - 1],
                        points[i],
                        color,
                        2
                    )

                cv2.circle(display, (cx, cy), 4, color, -1)

        # Header panel
        cv2.rectangle(display, (0, 0), (520, 130), (20, 20, 20), -1)

        draw_text(display, f"FPS: {smooth_fps:.1f}", (15, 35), (0, 255, 255), 0.8)
        draw_text(display, f"Active Tracks: {active_tracks}", (15, 70), (255, 255, 255), 0.8)
        draw_text(display, f"Unique Objects Seen: {len(unique_track_ids_seen)}", (15, 105), (0, 255, 0), 0.8)

        cv2.imshow(WINDOW_NAME, display)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    print("\nProgram terminated.")
    print(f"Unique objects seen: {len(unique_track_ids_seen)}")


if __name__ == "__main__":
    main()