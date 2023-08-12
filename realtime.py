import os
import cv2
import mediapipe as mp
import numpy as np
from keras.models import load_model

# Inisialisasi MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5)  # Sesuaikan dengan nilai yang Anda inginkan
mp_drawing = mp.solutions.drawing_utils

# Inisialisasi model yang telah dilatih sebelumnya
model_path = 'model/ABC_model.keras'  # Ubah sesuai dengan path model Anda
model = load_model(model_path)
data_dir = 'dataset'  # Ubah sesuai dengan path Anda
classes = sorted(os.listdir(data_dir))

# Inisialisasi Webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    # Flip citra secara horizontal
    frame = cv2.flip(frame, 1)

    # Ubah citra menjadi warna BGR ke RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Deteksi tangan
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            palm_coords = [(int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0]))
                           for landmark in hand_landmarks.landmark]

            min_x = min(palm_coords, key=lambda x: x[0])[0]
            max_x = max(palm_coords, key=lambda x: x[0])[0]
            min_y = min(palm_coords, key=lambda x: x[1])[1]
            max_y = max(palm_coords, key=lambda x: x[1])[1]

            # Tambahkan jarak pada koordinat untuk bounding box
            padding = 20
            min_x -= padding
            max_x += padding
            min_y -= padding
            max_y += padding

            cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), (0, 255, 0), 2)

            hand_roi = frame[min_y:max_y, min_x:max_x]
            if hand_roi.shape[0] > 0 and hand_roi.shape[1] > 0:
                hand_roi_resized = cv2.resize(hand_roi, (150, 150))
                hand_roi_resized = np.expand_dims(hand_roi_resized, axis=0)
                hand_roi_resized = hand_roi_resized / 255.0  # Normalisasi

                prediction = model.predict(hand_roi_resized)
                predicted_label = classes[np.argmax(prediction)]

                text = f"Predicted: {predicted_label}"
                cv2.putText(frame, text, (min_x, min_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow('Hand Detection', frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
