from flask import Flask, render_template, Response
import cv2
import numpy as np
import pickle
import time
import os

app = Flask(__name__)

# Load Models
print("[INFO] Loading models...")
detector = cv2.dnn.readNetFromCaffe("model/deploy.prototxt", "model/res10_300x300_ssd_iter_140000.caffemodel")
embedder = cv2.dnn.readNetFromTorch("openface_nn4.small2.v1.t7")

with open("output/recognizer.pickle", "rb") as f:
    recognizer = pickle.load(f)
with open("output/le.pickle", "rb") as f:
    le = pickle.load(f)

# Camera
cam = cv2.VideoCapture(0)

def generate_frames():
    while True:
        success, frame = cam.read()
        if not success:
            break
        else:
            frame = cv2.resize(frame, (600, 400))
            (h, w) = frame.shape[:2]

            imageBlob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                              (300, 300), (104.0, 177.0, 123.0),
                                              swapRB=False, crop=False)

            detector.setInput(imageBlob)
            detections = detector.forward()

            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.5:
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")

                    face = frame[startY:endY, startX:endX]
                    if face.shape[0] < 20 or face.shape[1] < 20:
                        continue

                    faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
                                                     (96, 96), (0, 0, 0),
                                                     swapRB=True, crop=False)
                    embedder.setInput(faceBlob)
                    vec = embedder.forward()

                    preds = recognizer.predict_proba(vec)[0]
                    j = np.argmax(preds)
                    proba = preds[j]
                    name = le.classes_[j]

                    text = f"{name}: {proba*100:.2f}%"
                    y = startY - 10 if startY - 10 > 10 else startY + 10
                    cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                    cv2.putText(frame, text, (startX, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return '''
    <h2>Face Recognition Live Feed</h2>
    <img src="/video_feed">
    '''

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
