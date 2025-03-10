from fastapi import FastAPI, WebSocket
import cv2
import mediapipe as mp
import numpy as np
import asyncio

app = FastAPI()
@app.get("/")
def read_root():
    return {"message": "API is running successfully 🚀"}
    
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        try:
            data = await websocket.receive_bytes()
            np_arr = np.frombuffer(data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = hands.process(frame_rgb)
            finger_count = 0

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    finger_tips = [8, 12, 16, 20]  # Index, Middle, Ring, Pinky
                    thumb_tip = 4
                    landmarks = hand_landmarks.landmark

                    for tip in finger_tips:
                        if landmarks[tip].y < landmarks[tip - 2].y:
                            finger_count += 1

                    if landmarks[thumb_tip].x > landmarks[thumb_tip - 2].x:
                        finger_count += 1

            await websocket.send_json({"finger_count": finger_count})

        except Exception as e:
            print(f"Error: {e}")
            break

    await websocket.close()
