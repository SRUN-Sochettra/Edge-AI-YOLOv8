import cv2

print(f"OpenCV version: {cv2.__version__}")

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if cap.isOpened():
    ret, frame = cap.read()

    if ret:
        print(f"Webcam OK — Frame size: {frame.shape[1]}x{frame.shape[0]}")
    else:
        print("Webcam opened but failed to read a frame.")

    cap.release()
else:
    print("Could not open webcam. Check device connection.")