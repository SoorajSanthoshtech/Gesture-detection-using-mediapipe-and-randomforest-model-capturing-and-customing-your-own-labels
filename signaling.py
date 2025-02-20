from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from typing import List
import uvicorn
import numpy as np
import joblib
import json
import math

app = FastAPI()

# Load the trained model and scaler
rf_model = joblib.load("new_model3.pkl")
scaler = joblib.load("scaler3.pkl")

# Store active WebSocket connections
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()

# Function to calculate the angle between three landmarks
def calculate_angle(p1, p2, p3):
    vec1 = [p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]]
    vec2 = [p3[0] - p2[0], p3[1] - p2[1], p3[2] - p2[2]]
    dot_product = sum(v1 * v2 for v1, v2 in zip(vec1, vec2))
    magnitude1 = math.sqrt(sum(v ** 2 for v in vec1))
    magnitude2 = math.sqrt(sum(v ** 2 for v in vec2))
    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0
    return math.degrees(math.acos(dot_product / (magnitude1 * magnitude2)))

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if "hand_landmarks" in message:
                hand_landmarks = message["hand_landmarks"]
                features = []
                for i in range(len(hand_landmarks)):
                    for j in range(i + 1, len(hand_landmarks)):
                        x1, y1, z1 = hand_landmarks[i]
                        x2, y2, z2 = hand_landmarks[j]
                        features.append(np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2))
                for i in range(len(hand_landmarks)):
                    for j in range(i + 1, len(hand_landmarks)):
                        for k in range(j + 1, len(hand_landmarks)):
                            features.append(calculate_angle(hand_landmarks[i], hand_landmarks[j], hand_landmarks[k]))
                features_array = np.array(features).reshape(1, -1)
                scaled_features = scaler.transform(features_array)
                prediction = rf_model.predict(scaled_features)[0]
                await manager.broadcast(json.dumps({"gesture": prediction}))
            elif "offer" in message or "answer" in message or "candidate" in message:
                await manager.broadcast(data)
    except WebSocketDisconnect:
        manager.disconnect(websocket)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)
