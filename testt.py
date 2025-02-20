import numpy as np
import joblib
import cv2
import math
import pandas as pd
import mediapipe as mp
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Load or initialize the model and scaler
MODEL_FILENAME = "appibiju.pkl"
SCALER_FILENAME = "scaler10.pkl"

try:
    rf_model = joblib.load(MODEL_FILENAME)
    scaler = joblib.load(SCALER_FILENAME)
    print("Model and scaler loaded successfully.")
except:
    rf_model = RandomForestClassifier(n_estimators=100)
    scaler = StandardScaler()
    print("New model and scaler initialized.")

# Initialize variables
gesture_data, gesture_labels = [], []
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def calculate_distance(landmark1, landmark2):
    return math.sqrt((landmark2.x - landmark1.x) ** 2 + (landmark2.y - landmark1.y) ** 2 + (landmark2.z - landmark1.z) ** 2)

def calculate_angle(landmark1, landmark2, landmark3):
    vec1 = [landmark2.x - landmark1.x, landmark2.y - landmark1.y, landmark2.z - landmark1.z]
    vec2 = [landmark3.x - landmark2.x, landmark3.y - landmark2.y, landmark3.z - landmark2.z]
    dot_product = sum(v1 * v2 for v1, v2 in zip(vec1, vec2))
    magnitude1 = math.sqrt(sum(v ** 2 for v in vec1))
    magnitude2 = math.sqrt(sum(v ** 2 for v in vec2))
    return 0.0 if magnitude1 == 0 or magnitude2 == 0 else math.degrees(math.acos(dot_product / (magnitude1 * magnitude2)))

cap = cv2.VideoCapture(0)
with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = cv2.flip(image, 1)
        results = hands.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                features = [
                    calculate_distance(hand_landmarks.landmark[i], hand_landmarks.landmark[j])
                    for i in range(21) for j in range(i + 1, 21)
                ] + [
                    calculate_angle(hand_landmarks.landmark[i], hand_landmarks.landmark[j], hand_landmarks.landmark[k])
                    for i in range(21) for j in range(i + 1, 21) for k in range(j + 1, 21)
                ]
                
                features_df = pd.DataFrame([features])
                if len(gesture_data) > 0:
                    features_df = pd.DataFrame(scaler.transform(features_df))
                    prediction = rf_model.predict(features_df)[0]
                    cv2.putText(image, f'Prediction: {prediction}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                key = cv2.waitKey(10) & 0xFF
                if key == ord('s'):
                    label = input("Enter gesture label: ")
                    gesture_data.append(features)
                    gesture_labels.append(label)
                    print(f"Gesture '{label}' saved!")

                    X = pd.DataFrame(gesture_data)
                    y = gesture_labels
                    X = scaler.fit_transform(X)

                    rf_model = RandomForestClassifier(n_estimators=100)
                    rf_model.fit(X, y)
                    joblib.dump(rf_model, MODEL_FILENAME)
                    joblib.dump(scaler, SCALER_FILENAME)
                    print("Model and scaler updated and saved.")

        cv2.imshow('Hand Gesture Recognition', image)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
