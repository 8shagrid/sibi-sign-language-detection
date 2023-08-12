import cv2
import numpy as np
import mediapipe as mp
import os

# Inisialisasi MediaPipe untuk mendeteksi tangan
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

# Inisialisasi webcam
cap = cv2.VideoCapture(0)

# Membuat direktori untuk menyimpan dataset
dataset_dir = "dataset_1"
os.makedirs(dataset_dir, exist_ok=True)

# Inisialisasi abjad yang sesuai dengan dataset
alphabet = 'ABC'

# Dictionary untuk menghitung jumlah gambar yang sudah diambil
image_counter = {char: 0 for char in alphabet}

# Jumlah frame yang akan diambil untuk setiap karakter
frames_per_char = 100

# Karakter abjad yang ingin diambil pertama kali
current_char = alphabet[0]

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
            # Mendapatkan koordinat titik tangan
            palm_coords = [(int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0]))
                           for landmark in hand_landmarks.landmark]

            # Menentukan bounding box tangan
            padding = 20
            min_x = min(palm_coords, key=lambda x: x[0])[0] - padding
            max_x = max(palm_coords, key=lambda x: x[0])[0] + padding
            min_y = min(palm_coords, key=lambda x: x[1])[1] - padding
            max_y = max(palm_coords, key=lambda x: x[1])[1] + padding

            # Gambar bounding box pada frame
            cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), (0, 255, 0), 2)

            # Ambil region of interest (ROI) tangan
            hand_roi = frame[min_y:max_y, min_x:max_x]

            if hand_roi.shape[0] > 0 and hand_roi.shape[1] > 0:
                # Resize gambar menjadi 150x150
                resized_hand_roi = cv2.resize(hand_roi, (150, 150))

                # Menampilkan citra ROI yang sudah diresize
                cv2.imshow("Hand ROI", resized_hand_roi)

                # Simpan gambar jika tombol spasi ditekan
                key = cv2.waitKey(1)
                if key == ord("q"):
                    break
                elif key == 32:  # Tombol spasi
                    char = current_char
                    char_dir = os.path.join(dataset_dir, char)
                    os.makedirs(char_dir, exist_ok=True)
                    image_name = f"{char}_{image_counter[char]}.jpg"
                    image_path = os.path.join(char_dir, image_name)
                    cv2.imwrite(image_path, resized_hand_roi)  # Simpan gambar yang sudah diresize
                    image_counter[char] += 1

                    if image_counter[current_char] >= frames_per_char:
                        # Setelah mengambil frames_per_char frame, pindah ke karakter berikutnya
                        alphabet_idx = alphabet.index(current_char)
                        if alphabet_idx < len(alphabet) - 1:
                            current_char = alphabet[alphabet_idx + 1]

            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Menampilkan teks dengan jumlah frame yang telah diambil untuk setiap karakter
    for char, count in image_counter.items():
        cv2.putText(frame, f"{char}: {count}/{frames_per_char}", (10, 30 + 30 * alphabet.index(char)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow('Hand Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
