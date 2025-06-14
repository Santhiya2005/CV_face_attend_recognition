import os
import sys
import subprocess
import numpy as np
import imutils
import pickle
import time
import cv2
import csv
from collections.abc import Iterable

# === Install scikit-learn if missing ===
try:
    import sklearn
except ImportError:
    print("[INFO] Installing required module: scikit-learn")
    subprocess.check_call([sys.executable.replace('w', ''), "-m", "pip", "install", "scikit-learn"])

# === CONFIGURATION ===
embeddingModel = os.path.join(os.getcwd(), "openface_nn4.small2.v1.t7")
embeddingFile = "output/embeddings.pickle"
recognizerFile = "output/recognizer.pickle"
labelEncFile = "output/le.pickle"
csvFilePath = "student.csv"
confidenceThreshold = 0.5

# === VERIFY FILES EXIST ===
for file_path in [embeddingModel, embeddingFile, recognizerFile, labelEncFile, csvFilePath]:
    if not os.path.exists(file_path):
        print(f"[ERROR] File not found: {file_path}")
        sys.exit(1)

# === LOAD MODELS ===
print("[INFO] Loading face detector...")
prototxt = "model/deploy.prototxt"
model = "model/res10_300x300_ssd_iter_140000.caffemodel"
detector = cv2.dnn.readNetFromCaffe(prototxt, model)

print("[INFO] Loading face embedder...")
embedder = cv2.dnn.readNetFromTorch(embeddingModel)

print("[INFO] Loading recognizer and label encoder...")
with open(recognizerFile, "rb") as f:
    recognizer = pickle.load(f)
with open(labelEncFile, "rb") as f:
    le = pickle.load(f)

# === LOAD STUDENT DATA ===
print("[INFO] Loading student records...")
students = {}
with open(csvFilePath, "r") as f:
    reader = csv.reader(f)
    for row in reader:
        if len(row) >= 2:
            students[row[0]] = row[1]  # name -> roll number

# === START VIDEO STREAM ===
print("[INFO] Starting video stream...")
cam = cv2.VideoCapture(0)
time.sleep(2.0)

while True:
    ret, frame = cam.read()
    if not ret:
        print("[WARNING] Frame capture failed.")
        break

    frame = imutils.resize(frame, width=600)
    (h, w) = frame.shape[:2]

    imageBlob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)), 1.0, (300, 300),
        (104.0, 177.0, 123.0), swapRB=False, crop=False
    )

    detector.setInput(imageBlob)
    detections = detector.forward()

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > confidenceThreshold:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            face = frame[startY:endY, startX:endX]
            (fH, fW) = face.shape[:2]
            if fW < 20 or fH < 20:
                continue

            faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), (0, 0, 0),
                                             swapRB=True, crop=False)
            embedder.setInput(faceBlob)
            vec = embedder.forward()

            preds = recognizer.predict_proba(vec)[0]
            j = np.argmax(preds)
            proba = preds[j]
            name = le.classes_[j]
            roll = students.get(name, "Unknown")

            text = f"{name} : {roll} : {proba * 100:.2f}%"
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
            cv2.putText(frame, text, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

    cv2.imshow("Face Recognition", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

cam.release()
cv2.destroyAllWindows()
