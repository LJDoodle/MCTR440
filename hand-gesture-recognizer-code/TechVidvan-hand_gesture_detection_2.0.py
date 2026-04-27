# TechVidvan Hand Gesture Recognizer + Webots Control Client

import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model
import socket
import time
from collections import deque

# -----------------------------
# SOCKET SETUP (send to Webots)
# -----------------------------
sock = socket.socket()
sock.connect(("127.0.0.1", 5000))

# -----------------------------
# MEDIAPIPE SETUP
# -----------------------------
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

model = load_model('mp_hand_gesture')

with open('gesture.names', 'r') as f:
    classNames = f.read().split('\n')

print("Loaded gestures:", classNames)

# -----------------------------
# CAMERA
# -----------------------------
cap = cv2.VideoCapture(0)

# -----------------------------
# SMOOTHING
# -----------------------------
gesture_buffer = deque(maxlen=5)
current_command = "stop"

# -----------------------------
# MAP GESTURES → ROBOT COMMANDS
# -----------------------------
gesture_map = {
    "stop": "stop",
    "fist": "forward",
    "thumbs up": "left",
    "thumbs down": "right"
}

# -----------------------------
# LOOP
# -----------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    x, y, c = frame.shape

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = hands.process(rgb)

    className = ""
    confidence = 0

    if result.multi_hand_landmarks:
        for handslms in result.multi_hand_landmarks:

            landmarks = []

            for lm in handslms.landmark:
                landmarks.append([int(lm.x * x), int(lm.y * y)])

            mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)

            prediction = model.predict([landmarks], verbose=0)
            classID = np.argmax(prediction)

            className = classNames[classID]
            confidence = prediction[0][classID]

    # -----------------------------
    # STABILIZE GESTURE
    # -----------------------------
    if className in gesture_map and confidence > 0.8:
        gesture_buffer.append(className)

        if gesture_buffer.count(className) > 3:
            current_command = gesture_map[className]

    # -----------------------------
    # SEND TO WEBOTS
    # -----------------------------
    try:
        sock.send(current_command.encode())
    except:
        pass

    # -----------------------------
    # DISPLAY
    # -----------------------------
    cv2.putText(frame, f'{className} ({confidence*100:.1f}%)',
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2)

    cv2.putText(frame, f'Command: {current_command}',
                (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0),
                2)

    cv2.imshow("Gesture Control", frame)

    if cv2.waitKey(1) == ord('q'):
        break

    time.sleep(0.02)

# -----------------------------
# CLEANUP
# -----------------------------
cap.release()
cv2.destroyAllWindows()
sock.close()
