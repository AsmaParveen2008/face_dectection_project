
import streamlit as st
import cv2
from PIL import Image
import numpy as np

st.title("Smart Face Detection System")

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades
    + "haarcascade_frontalface_default.xml"
)

def detect_faces(image):
    img = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=8,
        minSize=(60, 60)
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(
            img, (x, y), (x+w, y+h),
            (0, 255, 0), 3
        )

    return img, len(faces)

mode = st.radio(
    "Choose Detection Mode",
    ["Upload Image", "Live Camera"]
)

if mode == "Upload Image":
    file = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png"]
    )

    if file is not None:
        image = Image.open(file)
        result, count = detect_faces(image)

        st.image(result, caption="Detection Result")
        st.success(f"Faces detected: {count}")

        result_image = Image.fromarray(result)
        import io
        buffer = io.BytesIO()
        result_image.save(buffer, format="JPEG")

        st.download_button(
            "Download Detected Image",
            buffer.getvalue(),
            "detected_faces.jpg",
            "image/jpeg"
        )

else:
    camera_image = st.camera_input("Take a picture")

    if camera_image is not None:
        image = Image.open(camera_image)
        result, count = detect_faces(image)

        st.image(result, caption="Camera Detection Result")
        st.success(f"Faces detected: {count}")
