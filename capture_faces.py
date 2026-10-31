import cv2
import os

person_name = "Yuva"

path = os.path.join("dataset", person_name)
os.makedirs(path, exist_ok=True)

camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not camera.isOpened():
    print("Error: Cannot open camera")
    exit()

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

count = 0

while True:
    ret, frame = camera.read()

    if not ret:
        print("Error: Cannot read frame")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(100, 100)
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

    cv2.imshow("Capture Faces", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("s"):
        if len(faces) > 0:
            count += 1
            filename = os.path.join(path, f"{count}.jpg")
            cv2.imwrite(filename, frame)
            print("Saved:", filename)
        else:
            print("No face detected")

    elif key == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()