import streamlit as st
import cv2
from ultralytics import YOLO
import tempfile
import os
import time

# Load YOLO model
MODEL_PATH = r"C:\Users\sanik\Downloads\thermosolar_yolov82_earlystop\weights\best.pt"
model = YOLO(MODEL_PATH)

st.title("Solar Panel Anomaly Detector")

# Choose input type
option = st.radio("Choose input type:", ("Image", "Video"))

# IMAGE MODE
if option == "Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        # Save file to temp path
        suffix = os.path.splitext(uploaded_file.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp:
            temp.write(uploaded_file.read())
            temp_path = temp.name

        # Predict
        results = model(temp_path)
        res_img = results[0].plot()

        # Show result
        st.image(res_img, channels="BGR", caption="Detection Result")

        # Clean up
        os.remove(temp_path)

# VIDEO MODE
elif option == "Video":
    uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
    if uploaded_video:
        suffix = os.path.splitext(uploaded_video.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp:
            temp.write(uploaded_video.read())
            temp_path = temp.name

        cap = cv2.VideoCapture(temp_path)
        stframe = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            results = model(frame)
            frame = results[0].plot()
            stframe.image(frame, channels="BGR")
            time.sleep(0.03)  # ~30 FPS
        cap.release()

        os.remove(temp_path)
#


# # LIVE Camera MODE
# elif option == "Live Camera":
#     rtsp_url = st.text_input("Enter RTSP stream URL (Camera)")
#     if st.button("Start Stream") and rtsp_url:
#         cap = cv2.VideoCapture(rtsp_url)
#         stframe = st.empty()
#
#         while cap.isOpened():
#             ret, frame = cap.read()
#             if not ret:
#                 st.error("Stream ended or cannot connect.")
#                 break
#             results = model(frame)
#             frame = results[0].plot()
#             stframe.image(frame, channels="BGR")
#         cap.release()
