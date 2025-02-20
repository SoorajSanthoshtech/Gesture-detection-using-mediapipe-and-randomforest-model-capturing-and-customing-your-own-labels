import numpy as np
import joblib
import cv2
import math
import pandas as pd
import mediapipe as mp

# Load the trained model and scaler
model_filename = "new_model3.pkl"
scaler_filename = "scaler3.pkl"

rf_model = joblib.load(model_filename)
scaler = joblib.load(scaler_filename)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Function to calculate the Euclidean distance between two landmarks
def calculate_distance(landmark1, landmark2):
    return math.sqrt((landmark2.x - landmark1.x) ** 2 + (landmark2.y - landmark1.y) ** 2 + (landmark2.z - landmark1.z) ** 2)

# Function to calculate the angle between three landmarks
def calculate_angle(landmark1, landmark2, landmark3):
    vec1 = [landmark2.x - landmark1.x, landmark2.y - landmark1.y, landmark2.z - landmark1.z]
    vec2 = [landmark3.x - landmark2.x, landmark3.y - landmark2.y, landmark3.z - landmark2.z]
    dot_product = sum(v1 * v2 for v1, v2 in zip(vec1, vec2))
    magnitude1 = math.sqrt(sum(v ** 2 for v in vec1))
    magnitude2 = math.sqrt(sum(v ** 2 for v in vec2))
    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0
    return math.degrees(math.acos(dot_product / (magnitude1 * magnitude2)))

# OpenCV setup
cap = cv2.VideoCapture(0)

with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process frame
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = cv2.flip(image, 1)
        results = hands.process(image)

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Detect hand landmarks
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Extract features (distances & angles)
                features = []
                for i in range(21):
                    for j in range(i + 1, 21):
                        features.append(calculate_distance(hand_landmarks.landmark[i], hand_landmarks.landmark[j]))
                for i in range(21):
                    for j in range(i + 1, 21):
                        for k in range(j + 1, 21):
                            features.append(calculate_angle(hand_landmarks.landmark[i], hand_landmarks.landmark[j], hand_landmarks.landmark[k]))

                # Convert to DataFrame and scale
                features_df = pd.DataFrame([features])
                features_df = pd.DataFrame(scaler.transform(features_df))

                # Predict gesture
                prediction = rf_model.predict(features_df)[0]
                cv2.putText(image, f'Prediction: {prediction}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Show output
        cv2.imshow('Hand Gesture Prediction', image)

        # Quit on 'q' key
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
