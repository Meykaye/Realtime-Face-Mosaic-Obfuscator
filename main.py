import cv2
import numpy as np

def mosaic_face(frame, x, y, w, h, mosaic_scale=0.04):
    face = frame[y:y+h, x:x+w]
    small = cv2.resize(face, (0, 0), fx=mosaic_scale, fy=mosaic_scale, interpolation=cv2.INTER_LINEAR)
    mosaic = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    frame[y:y+h, x:x+w] = mosaic
    return frame

# Load the pre-trained DNN face detector
net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "res10_300x300_ssd_iter_140000.caffemodel")

# Start video capture
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Webcam not accessible")
    exit()

# Set window to be resizable
cv2.namedWindow("Realtime Mosaic Face Blur", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Realtime Mosaic Face Blur", 1000, 800)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    (h, w) = frame.shape[:2]

    # Detect faces
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104, 177, 123), False, False)
    net.setInput(blob)
    detections = net.forward()

    # Apply mosaic to detected faces
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * [w, h, w, h]
            (x1, y1, x2, y2) = box.astype("int")
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)
            frame = mosaic_face(frame, x1, y1, x2 - x1, y2 - y1)

    # Show result
    cv2.imshow("Realtime Mosaic Face Blur", frame)

    # Exit on any key press
    if cv2.waitKey(1) != -1:
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
