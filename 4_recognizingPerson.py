import os
import numpy as np
import imutils
import pickle
import time
import cv2

# ===== CONFIGURATION =====
basePath = os.getcwd()
embeddingModelPath = os.path.join(basePath, "openface_nn4.small2.v1.t7")
embeddingFilePath = os.path.join(basePath, "output", "embeddings.pickle")
recognizerFilePath = os.path.join(basePath, "output", "recognizer.pickle")
labelEncoderPath = os.path.join(basePath, "output", "le.pickle")
prototxtPath = os.path.join(basePath, "model", "deploy.prototxt")
modelPath = os.path.join(basePath, "model", "res10_300x300_ssd_iter_140000.caffemodel")
confidenceThreshold = 0.5

# ===== LOADING MODELS =====
print("[INFO] Loading face detector...")
try:
    detector = cv2.dnn.readNetFromCaffe(prototxtPath, modelPath)
except Exception as e:
    print(f"[ERROR] Could not load face detector: {e}")
    exit()

print("[INFO] Loading face embedding model...")
try:
    embedder = cv2.dnn.readNetFromTorch(embeddingModelPath)
except Exception as e:
    print(f"[ERROR] Could not load face embedding model: {e}")
    exit()

print("[INFO] Loading recognizer and label encoder...")
try:
    with open(recognizerFilePath, "rb") as f:
        recognizer = pickle.load(f)
    with open(labelEncoderPath, "rb") as f:
        le = pickle.load(f)
except Exception as e:
    print(f"[ERROR] Could not load recognizer or label encoder: {e}")
    exit()

# ===== VIDEO STREAM =====
print("[INFO] Starting video stream...")
cam = cv2.VideoCapture(0)  # Change to 1 if using an external webcam
if not cam.isOpened():
    print("[ERROR] Could not access webcam.")
    exit()
time.sleep(2.0)

# ===== MAIN LOOP =====
while True:
    ret, frame = cam.read()
    if not ret:
        print("[WARNING] Failed to grab frame.")
        break

    frame = imutils.resize(frame, width=600)
    (h, w) = frame.shape[:2]

    imageBlob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)), 1.0, (300, 300),
        (104.0, 177.0, 123.0), swapRB=False, crop=False
    )

    detector.setInput(imageBlob)
    detections = detector.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > confidenceThreshold:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            face = frame[startY:endY, startX:endX]
            (fH, fW) = face.shape[:2]

            if fW < 20 or fH < 20:
                continue

            faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96),
                                             (0, 0, 0), swapRB=True, crop=False)
            embedder.setInput(faceBlob)
            vec = embedder.forward()

            preds = recognizer.predict_proba(vec)[0]
            j = np.argmax(preds)
            proba = preds[j]
            name = le.classes_[j]

            text = f"{name} : {proba * 100:.2f}%"
            y = startY - 10 if startY - 10 > 10 else startY + 10

            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
            cv2.putText(frame, text, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

    cv2.imshow("Face Recognition", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC key
        break

# ===== CLEANUP =====
cam.release()
cv2.destroyAllWindows()
