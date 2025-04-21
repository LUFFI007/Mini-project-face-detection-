import cv2
import os

# Create a directory to save snapshots
if not os.path.exists('snapshots'):
    os.makedirs('snapshots')

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)
snapshot_count = 0

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        face_crop = frame[y:y+h, x:x+w]

    cv2.imshow('Face Detection with Snapshot', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('s') and len(faces) > 0:
        filename = f"snapshots/face_{snapshot_count}.jpg"
        cv2.imwrite(filename, face_crop)
        print(f"Snapshot saved: {filename}")
        snapshot_count += 1
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
