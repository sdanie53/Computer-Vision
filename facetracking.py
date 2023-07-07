#!/usr/bin/env python3

import cv2
import numpy as np

def draw_rectangle(frame, rect):
    x, y, w, h = rect
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

face_cascade = cv2.CascadeClassifier('/home/steeve/miniconda3/envs/cvtest/share/opencv4/haarcascades/haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        draw_rectangle(frame, (x, y, w, h))

    cv2.imshow("Face Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

