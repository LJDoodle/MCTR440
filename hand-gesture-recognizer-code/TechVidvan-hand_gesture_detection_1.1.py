import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model
import time

# Initialize mediapipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# Load the gesture recognizer model
model = load_model('mp_hand_gesture')

# Load class names from 'gesture.names'
f = open('gesture.names', 'r')
classNames = f.read().split('\n')
f.close()
print("Loaded gestures:", classNames)

# Initialize webcam
cap = cv2.VideoCapture(0)

# Function to wait for a gesture to be held for a certain amount of time
def wait_for_gesture(frame, target_gestures, hold_time=2.0, confidence_threshold=0.5):
    start_time = time.time()
    className = ""  # Initialize className to avoid the UnboundLocalError
    while time.time() - start_time < hold_time:
        _, frame = cap.read()
        frame = cv2.flip(frame, 1)
        framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(framergb)

        if result.multi_hand_landmarks:
            landmarks = []
            for handslms in result.multi_hand_landmarks:
                for lm in handslms.landmark:
                    lmx = int(lm.x * frame.shape[1])
                    lmy = int(lm.y * frame.shape[0])
                    landmarks.append([lmx, lmy])

            mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)
            prediction = model.predict([landmarks])
            classID = np.argmax(prediction)
            className = classNames[classID]  # This assigns the detected gesture name
            confidence = prediction[0][classID]  # Extract confidence score

            # Debugging: Print prediction class and confidence
            print(f"Predicted: {className} with confidence: {confidence*100:.2f}%")

            # Check if the predicted gesture matches the target gestures and confidence is above threshold
            if className in target_gestures and confidence >= confidence_threshold:
                cv2.putText(frame, f'Gesture Detected: {className} ({confidence*100:.2f}%)', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            else:
                cv2.putText(frame, f'Waiting for one of: {", ".join(target_gestures)} ({confidence*100:.2f}%)', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.imshow("Output", frame)
        if cv2.waitKey(1) == ord('q'):
            break

    return className in target_gestures

# Function to wait for a confirmation gesture (thumbs up/thumbs down)
def wait_for_confirmation(frame, target_gesture, hold_time=2.0, confidence_threshold=0.5):
    start_time = time.time()
    className = ""  # Initialize className
    while time.time() - start_time < hold_time:
        _, frame = cap.read()
        frame = cv2.flip(frame, 1)
        framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(framergb)

        if result.multi_hand_landmarks:
            landmarks = []
            for handslms in result.multi_hand_landmarks:
                for lm in handslms.landmark:
                    lmx = int(lm.x * frame.shape[1])
                    lmy = int(lm.y * frame.shape[0])
                    landmarks.append([lmx, lmy])

            mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)
            prediction = model.predict([landmarks])
            classID = np.argmax(prediction)
            className = classNames[classID]
            confidence = prediction[0][classID]

            # Debugging: Print prediction class and confidence
            print(f"Predicted: {className} with confidence: {confidence*100:.2f}%")

            # Only proceed if the target gesture is detected with sufficient confidence
            if className == target_gesture and confidence >= confidence_threshold:
                cv2.putText(frame, f'Confirmation Detected: {className} ({confidence*100:.2f}%)', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            else:
                cv2.putText(frame, f'Waiting for: {target_gesture} ({confidence*100:.2f}%)', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.imshow("Output", frame)
        if cv2.waitKey(1) == ord('q'):
            break

    return className == target_gesture

# Function to show "performing action" on the screen
def display_action_message(frame, action_message="Performing Action..."):
    cv2.putText(frame, action_message, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    # You can also change the background color to visually indicate the action
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 50), (0, 255, 0), -1)  # Green background at top
    cv2.putText(frame, action_message, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

# Main loop
while True:
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(framergb)

    if result.multi_hand_landmarks:
        landmarks = []
        for handslms in result.multi_hand_landmarks:
            for lm in handslms.landmark:
                lmx = int(lm.x * frame.shape[1])
                lmy = int(lm.y * frame.shape[0])
                landmarks.append([lmx, lmy])

        # Drawing landmarks on frames
        mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)

        # Get the gesture prediction
        prediction = model.predict([landmarks])
        classID = np.argmax(prediction)
        className = classNames[classID]
        confidence = prediction[0][classID]

        # Show the current gesture prediction and confidence
        cv2.putText(frame, f'{className} ({confidence*100:.2f}%)', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    # Display the frame
    cv2.imshow("Output", frame)

    # Wait for key press to start gesture detection
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):  # 's' key to start gesture detection
        print("Waiting for action gesture...")
        # Define the action gestures: 'stop' or 'rock'
        action_gestures = ['fist', 'rock']  # Check for either of these gestures
        action_detected = False
        for action in action_gestures:
            if wait_for_gesture(frame, action_gestures, hold_time=3.0, confidence_threshold=0.5):  # Increased hold time for better detection
                print(f"Action gesture {action} detected.")
                action_detected = True
                break
        
        if not action_detected:
            print("No valid action gesture detected. Exiting...")
            continue
        
        print("Now, confirm your action with thumbs up (positive) or thumbs down (negative).")

        # Wait for confirmation gesture (thumbs up or thumbs down)
        confirmation_gesture = 'thumbs up'  # Confirmation gesture for positive
        if wait_for_confirmation(frame, confirmation_gesture, hold_time=3.0, confidence_threshold=0.5):  # Increased hold time for confirmation
            print(f"Action confirmed with {confirmation_gesture}. Performing action...")
            display_action_message(frame, "Performing Action...")  # Show "Performing Action" message
        else:
            print("Waiting for cancellation gesture...")
            # Wait for thumbs down to cancel action
            cancel_gesture = 'thumbs down'  # Cancellation gesture
            if wait_for_confirmation(frame, cancel_gesture, hold_time=3.0, confidence_threshold=0.5):
                print(f"Action canceled with {cancel_gesture}.")
                cv2.putText(frame, "Action Canceled", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    if key == ord('q'):  # 'q' key to quit
        break

cap.release()
cv2.destroyAllWindows()
