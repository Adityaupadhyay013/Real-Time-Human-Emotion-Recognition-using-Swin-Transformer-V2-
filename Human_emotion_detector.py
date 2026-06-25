import cv2 
import streamlit as st 
from streamlit_webrtc import webrtc_streamer , VideoProcessorBase 
import numpy as np
import pandas as pd 
from tensorflow.keras.models import load_model 
from tensorflow import keras 
from keras import Sequential
from keras.layers import GlobalAveragePooling2D , Dropout , Dense
import av
import mediapipe as mp
from keras_cv_attention_models import swin_transformer_v2
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)
from tensorflow.keras.models import load_model
model = load_model(
    "Emotion detection model(Fully finetuned swin).keras",
    compile=False
)
class_names = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Sad",
    "Surprise",
    "Neutral"
]
class VideoProcessor(VideoProcessorBase):

     def recv(self, frame):

        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray , 1.3 , 5)
        if len(faces) > 0:
            x1 , y1 , w1 , h = faces[0]
            face = img[y1:y1+h , x1:x1+w1]
            face_rgb = cv2.cvtColor(face , cv2.COLOR_BGR2RGB)
            x = cv2.resize(face_rgb, (224, 224))
            x = x.astype(np.float32) / 255.0
            x = np.expand_dims(x, axis=0)

            pred = model.predict(x, verbose=0)

            cls = np.argmax(pred)
            conf = np.max(pred)
            cv2.rectangle(img , (x1 , y1) , (x1+w1 , y1+h) , (0 , 255 , 0) , 2)
            label = f"{class_names[cls]}: {conf:.2f}"

            cv2.putText(
            img,
            label,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
            )

        return av.VideoFrame.from_ndarray(
            img,
            format="bgr24" 
            )
st.title("Live Human Emotion Detection")
webrtc_streamer(key = "classifier" , video_processor_factory = VideoProcessor , media_stream_constraints = {"video" : True , "audio": False} , async_processing=True)
