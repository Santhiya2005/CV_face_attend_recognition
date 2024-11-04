import imutils
import time
import cv2
import csv
import os

# Use the correct path for the haarcascade file from OpenCV data
cascade = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
detector = cv2.CascadeClassifier(cascade)

Name = str(input("Enter your Name : "))
Roll_Number = int(input("Enter your Roll_Number : "))
dataset = 'dataset'
sub_data = Name
path = os.path.join(dataset, sub_data)

# Create directory for the dataset if it doesn't exist
if not os.path.exists(path):
    os.makedirs(path)
    print(f"Directory created for {sub_data}")

# Save student info in CSV
info = [str(Name), str(Roll_Number)]
with open('student.csv', 'a', newline='') as csvFile:
    write = csv.writer(csvFile)
    write.writerow(info)

print("Starting video stream...")
# Use camera 0 (default camera). If this doesn’t work, try cv2.VideoCapture(1)
cam = cv2.VideoCapture(0)  
time.sleep(2.0)  # Give time for the camera to warm up
total = 0

# Capture 50 images
while total < 50:
    print(total)
    ret, frame = cam.read()
    
    if not ret:
        print("Failed to grab frame")
        break
    
    img = imutils.resize(frame, width=400)
    
    # Detect faces
    rects = detector.detectMultiScale(
        cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), scaleFactor=1.1,
        minNeighbors=5, minSize=(30, 30))

    # Draw rectangle around faces and save images
    for (x, y, w, h) in rects:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        p = os.path.sep.join([path, "{}.png".format(str(total).zfill(5))])
        cv2.imwrite(p, img)
        total += 1

    # Display frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# Release camera and close all windows
cam.release()
cv2.destroyAllWindows()
