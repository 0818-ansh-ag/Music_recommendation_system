import requests
import streamlit as st
from PIL import Image
from streamlit_lottie import st_lottie
from streamlit_webrtc import webrtc_streamer
import av
import cv2 
import numpy as np 
import mediapipe as mp 
from keras.models import load_model
import webbrowser

with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

model  = load_model("model.h5")
label = np.load("labels.npy")
holistic = mp.solutions.holistic
hands = mp.solutions.hands
holis = holistic.Holistic()
drawing = mp.solutions.drawing_utils





def load_lottieurl(url):
    r= requests.get(url)
    if r.status_code != 200:
       return none
    return r.json()

lottie_coding = load_lottieurl("https://assets6.lottiefiles.com/packages/lf20_ikk4jhps.json")
lottie_maxican = load_lottieurl("https://assets2.lottiefiles.com/packages/lf20_ydhm6y.json")
lottie_camera = load_lottieurl("https://assets7.lottiefiles.com/packages/lf20_m5evszda.json")

col3,col1, col2,col4 = st.columns([0.5,1,3,1])

with col1:
     st_lottie(lottie_coding,height = 110,width = 350,key="coding")
     
with col2:
     st.markdown("<h1 style=' color: blue; text-align:left; '>Music Recommendation System</h1>", unsafe_allow_html=True)

     img = Image.open("image/music.jpg")
     st.image(img)

new_title = '<p style="font-family:sans-serif; color:pink; text-align: center;font-family:Courier; font-size: 42px;">  Hii, I am Moody</p>'
st.markdown(new_title, unsafe_allow_html=True)


if "run" not in st.session_state:
	st.session_state["run"] = "true"

try:
	emotion = np.load("emotion.npy")[0]
except:
	emotion=""

if not(emotion):
	st.session_state["run"] = "true"
else:
	st.session_state["run"] = "false"

class EmotionProcessor:
	def recv(self, frame):
		frm = frame.to_ndarray(format="bgr24")

		##############################
		frm = cv2.flip(frm, 1)

		res = holis.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))

		lst = []

		if res.face_landmarks:
			for i in res.face_landmarks.landmark:
				lst.append(i.x - res.face_landmarks.landmark[1].x)
				lst.append(i.y - res.face_landmarks.landmark[1].y)

			if res.left_hand_landmarks:
				for i in res.left_hand_landmarks.landmark:
					lst.append(i.x - res.left_hand_landmarks.landmark[8].x)
					lst.append(i.y - res.left_hand_landmarks.landmark[8].y)
			else:
				for i in range(42):
					lst.append(0.0)

			if res.right_hand_landmarks:
				for i in res.right_hand_landmarks.landmark:
					lst.append(i.x - res.right_hand_landmarks.landmark[8].x)
					lst.append(i.y - res.right_hand_landmarks.landmark[8].y)
			else:
				for i in range(42):
					lst.append(0.0)

			lst = np.array(lst).reshape(1,-1)

			pred = label[np.argmax(model.predict(lst))]

			print(pred)
			cv2.putText(frm, pred, (50,50),cv2.FONT_ITALIC, 1, (255,0,0),2)

			np.save("emotion.npy", np.array([pred]))

			
		drawing.draw_landmarks(frm, res.face_landmarks, holistic.FACEMESH_TESSELATION,
								landmark_drawing_spec=drawing.DrawingSpec(color=(0,0,255), thickness=-1, circle_radius=1),
								connection_drawing_spec=drawing.DrawingSpec(thickness=1))
		drawing.draw_landmarks(frm, res.left_hand_landmarks, hands.HAND_CONNECTIONS)
		drawing.draw_landmarks(frm, res.right_hand_landmarks, hands.HAND_CONNECTIONS)


		##############################

		return av.VideoFrame.from_ndarray(frm, format="bgr24")



col3,col1, col2 = st.columns([2,2,2])

with col1:
     
     lang = st.text_input("Enter the language",key=1)
     
     singer = st.text_input("Enter the name of singer",key=2)
     
with col2:
     st_lottie(lottie_maxican,height = 200,width = 200,key="ani")


if lang and singer and st.session_state["run"] == "true":
	webrtc_streamer(key="key", desired_playing_state=True,
				video_processor_factory=EmotionProcessor)



col1,col2, col3 , col4, col5 = st.columns([2,1.2,2,1,1])

with col1:
    pass
with col2:
    st_lottie(lottie_camera,height=75,width = 430,key="cam")
with col4:
    pass
with col5:
    pass
with col3 :
    btn = st.button("Recommend me songs")







if btn:
	if not(emotion):
		st.warning("Let us capture your emotion ")
		st.session_state["run"] = "true"
	else:
		webbrowser.open(f"https://www.youtube.com/results?search_query={emotion}+{lang}+songs+by+{singer}")
		np.save("emotion.npy", np.array([""]))		
		st.session_state["run"] = "false"
