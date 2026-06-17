from ultralytics import YOLO
import cv2
import time

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("ERROR: Cannot open webcam. Check connection and drivers.")
    exit(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("Webcam active. Press Q to quit.")

fps_list = []

while True:
    t0 = time.perf_counter()

    ret, frame = cap.read()

    if not ret:
        print("Frame capture failed.")
        break

    results = model(frame, conf=0.5, verbose=False)
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
    print(f"\nSession: {len(fps_list)} frames, Avg FPS: {sum(fps_list) / len(fps_list):.1f}")