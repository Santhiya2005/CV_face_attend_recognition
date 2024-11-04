from imutils import paths
import numpy as np
import imutils
import pickle
import cv2
import os

# Define paths
dataset = "dataset"
embeddingFile = "output/embeddings.pickle"  # Initial name for embedding file
embeddingModel = "C:/Users/SANTHIYA/Downloads/openface_nn4.small2.v1.t7"  # Path to embedding model

# Caffe model for face detection
prototxt = "C:/Users/SANTHIYA/Downloads/model/model/deploy.prototxt"
model = "C:/Users/SANTHIYA/Downloads/model/model/res10_300x300_ssd_iter_140000.caffemodel"

# Ensure output directory exists
output_dir = os.path.dirname(embeddingFile)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Loading Caffe model for face detection
print("Loading face detector model...")
if not os.path.exists(prototxt):
    raise Exception(f"Prototxt file not found at: {prototxt}")
if not os.path.exists(model):
    raise Exception(f"Caffemodel file not found at: {model}")
detector = cv2.dnn.readNetFromCaffe(prototxt, model)

# Loading PyTorch model for facial embeddings
print("Loading embedding model...")
if not os.path.exists(embeddingModel):
    raise Exception(f"Embedding model file not found at: {embeddingModel}")
embedder = cv2.dnn.readNetFromTorch(embeddingModel)

# Getting image paths
imagePaths = list(paths.list_images(dataset))

# Initialization
knownEmbeddings = []
knownNames = []
total = 0
conf = 0.5

# Processing images
for (i, imagePath) in enumerate(imagePaths):
    print(f"Processing image {i + 1}/{len(imagePaths)}")
    name = imagePath.split(os.path.sep)[-2]
    image = cv2.imread(imagePath)
    image = imutils.resize(image, width=600)
    (h, w) = image.shape[:2]

    # Convert image to blob for DNN face detection
    imageBlob = cv2.dnn.blobFromImage(
        cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=False, crop=False)

    # Set input blob image
    detector.setInput(imageBlob)
    # Predict faces
    detections = detector.forward()

    # Process detections with confidence greater than threshold
    if len(detections) > 0:
        i = np.argmax(detections[0, 0, :, 2])
        confidence = detections[0, 0, i, 2]

        if confidence > conf:
            # Extract the face ROI
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            face = image[startY:endY, startX:endX]
            (fH, fW) = face.shape[:2]

            # Ignore small faces
            if fW < 20 or fH < 20:
                continue

            # Convert face to blob for embedding extraction
            faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
            embedder.setInput(faceBlob)
            vec = embedder.forward()

            # Append name and embedding
            knownNames.append(name)
            knownEmbeddings.append(vec.flatten())
            total += 1

print(f"Processed {total} faces.")

# Save the embeddings
data = {"embeddings": knownEmbeddings, "names": knownNames}
with open(embeddingFile, "wb") as f:
    f.write(pickle.dumps(data))

print("Embedding process completed.")
