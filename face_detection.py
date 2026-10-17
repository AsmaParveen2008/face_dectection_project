
import cv2

# Load the face detection model
model_path = (
    cv2.data.haarcascades
    + "haarcascade_frontalface_default.xml"
)

face_cascade = cv2.CascadeClassifier(model_path)

if face_cascade.empty():
    raise RuntimeError("Could not load face detection model")

# Open webcam
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise RuntimeError("Cannot open webcam")

print("Face Detection Started")
print("Press S to save image")
print("Press Q to quit")

try:
    while True:
        success, frame = camera.read()

        if not success:
            print("Cannot read webcam frame")
            break

        # Mirror the video
        frame = cv2.flip(frame, 1)

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        # Draw rectangles around faces
        for (x, y, w, h) in faces:
            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

        # Display face count
        count = len(faces)
        cv2.putText(
            frame,
            "Faces: " + str(count),
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        # Display video
        cv2.imshow("Face Detection", frame)

        # Get keyboard input
        key = cv2.waitKey(1) & 0xFF

        # Press S to save image
        if key == ord("s"):
            cv2.imwrite("captured_face.jpg", frame)
            print("Image saved successfully!")

        # Press Q to quit
        if key == ord("q"):
            break

finally:
    # Release webcam and close windows
    camera.release()
    cv2.destroyAllWindows()

print("Face Detection Stopped")
