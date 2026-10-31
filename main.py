import cv2
import os

camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not camera.isOpened():
    print("Error: Cannot open camera.")
    exit()

if not os.path.exists("images"):
    os.makedirs("images")

count = 1

while True:
    ret, frame = camera.read()

    if not ret:
        print("Error: Cannot read frame.")
        break

    cv2.imshow("Capture Face", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):
        filename = f"images/yuva_{count}.jpg"
        cv2.imwrite(filename, frame)
        print("Saved:", filename)
        count += 1

    elif key == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()