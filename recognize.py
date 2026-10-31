import cv2
import os
import csv
from datetime import datetime

# Load face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Load trained model
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")

# Person names
names = ["Yuva"]

# Attendance file
attendance_file = "attendance.csv"

# Create CSV file if it doesn't exist
if not os.path.exists(attendance_file):
    with open(attendance_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Name", "Date", "Time", "Status"])

marked = set()

# Open camera
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not camera.isOpened():
    print("Error: Cannot open camera.")
    exit()

print("Camera started!")
print("Look at the camera.")
print("Press Q to quit.")

while True:
    ret, frame = camera.read()

    if not ret:
        print("Error: Cannot read camera frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(100, 100)
    )

    for (x, y, w, h) in faces:

        face = gray[y:y+h, x:x+w]

        # Recognize face
        label, confidence = recognizer.predict(face)

        if confidence < 80:
            name = names[label]

            # Mark attendance only once
            if name not in marked:
                now = datetime.now()

                with open(attendance_file, "a", newline="") as f:
                    writer = csv.writer(f)

                    writer.writerow([
                        name,
                        now.strftime("%d-%m-%Y"),
                        now.strftime("%H:%M:%S"),
                        "Present"
                    ])

                marked.add(name)

                print("Attendance marked for:", name)

        else:
            name = "Unknown"

        # Green rectangle
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        # Show name
        cv2.putText(
            frame,
            name,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 0, 0),
            2
        )

    # Show camera
    cv2.imshow("Face Recognition Attendance", frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()