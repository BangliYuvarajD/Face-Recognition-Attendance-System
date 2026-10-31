import cv2
import os
import numpy as np

image_path = "images"

faces = []
labels = []

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

for image_name in os.listdir(image_path):

    file_path = os.path.join(image_path, image_name)

    img = cv2.imread(file_path)

    if img is None:
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    detected_faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(100, 100)
    )

    for (x, y, w, h) in detected_faces:

        face = gray[y:y+h, x:x+w]

        faces.append(face)
        labels.append(0)

print("Faces found for training:", len(faces))

if len(faces) == 0:
    print("ERROR: No faces found in images folder.")
    exit()

recognizer = cv2.face.LBPHFaceRecognizer_create()

recognizer.train(faces, np.array(labels))

recognizer.save("trainer.yml")

print("Training completed successfully!")