from imutils import paths
import numpy as np
import imutils
import pickle
import cv2
import os

# Define paths
dataset = "dataset"
embeddingFile = "output/embeddings.pickle"
embeddingModel = "model/openface_nn4.small2.v1.t7"  # Updated to local model path
prototxt = "model/deploy.prototxt"
model = "model/res10_300x300_ssd_iter_140000.caffemodel"

# Ensure output directory exists
output_dir = os.path.dirname(embeddingFile)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Load face detector model
print("Loading face detector model...")
if not os.path.exists(prototxt):
    raise Exception(f"Prototxt file not found at: {prototxt}")
if not os.path.exists(model):
    raise Exception(f"Caffemodel file not found at: {model}")
detector = cv2.dnn.readNetFromCaffe(prototxt, model)

# Load embedding model
print("Loading embedding model...")
if not os.path.exists(embeddingModel):
    raise Exception(f"Embedding model file not found at: {embeddingModel}")
embedder = cv2.dnn.readNetFromTorch(embeddingModel)

# Get image paths
imagePaths = list(paths.list_images(dataset))

# Initialize lists
knownEmbeddings = []
knownNames = []
total = 0
conf = 0.5

# Process each image
for (i, imagePath) in enumerate(imagePaths):
    print(f"[INFO] Processing image {i + 1}/{len(imagePaths)}")
    name = imagePath.split(os.path.sep)[-2]

    image = cv2.imread(imagePath)
    image = imutils.resize(image, width=600)
    (h, w) = image.shape[:2]

    # Convert image to blob for face detection
    imageBlob = cv2.dnn.blobFromImage(
        cv2.resize(image, (300, 300)), 1.0, (300, 300),
        (104.0, 177.0, 123.0), swapRB=False, crop=False)

    detector.setInput(imageBlob)
    detections = detector.forward()

    if len(detections) > 0:
        i = np.argmax(detections[0, 0, :, 2])
        confidence = detections[0, 0, i, 2]

        if confidence > conf:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            face = image[startY:endY, startX:endX]
            (fH, fW) = face.shape[:2]

            if fW < 20 or fH < 20:
                continue

            faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
                                             (96, 96), (0, 0, 0),
                                             swapRB=True, crop=False)
            embedder.setInput(faceBlob)
            vec = embedder.forward()

            knownNames.append(name)
            knownEmbeddings.append(vec.flatten())
            total += 1

print(f"[INFO] Processed {total} faces.")

# Save embeddings to file
data = {"embeddings": knownEmbeddings, "names": knownNames}
with open(embeddingFile, "wb") as f:
    f.write(pickle.dumps(data))

print("[INFO] Embedding process completed successfully.")
