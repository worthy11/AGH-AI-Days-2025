import cv2
import mediapipe as mp
import numpy as np
import os
import csv
from datetime import datetime

GESTURES = {
    0: "palm_up",
    1: "fist",
    2: "peace",
    3: "thumbs_up",
    4: "ok_sign"
}

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

os.makedirs('images/raw', exist_ok=True)
os.makedirs('images/landmarks', exist_ok=True)

# Create both training and testing CSV files
train_csv = 'data_train.csv'
test_csv = 'data_test.csv'

for csv_file in [train_csv, test_csv]:
    if not os.path.exists(csv_file):
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            headers = []
            for i in range(21):
                headers.extend([f'x_{i}', f'y_{i}', f'z_{i}'])
            writer.writerow(['timestamp', 'label'] + headers)

cap = cv2.VideoCapture(0)
show_landmarks = False
current_label_idx = 0
is_training_mode = True  # Toggle between training and testing mode

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally
    frame = cv2.flip(frame, 1)

    # Add label text to the frame
    label_text = f"Current Label: {GESTURES[current_label_idx]}"
    mode_text = f"Mode: {'Training' if is_training_mode else 'Testing'}"
    cv2.putText(frame, label_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, mode_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    frame_with_landmarks = frame.copy()

    if show_landmarks and results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                frame_with_landmarks,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )
    
    cv2.imshow('Hand Tracking', frame_with_landmarks if show_landmarks else frame)
    
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q'):
        break
    
    elif key == ord('k'):
        show_landmarks = not show_landmarks
    
    elif key == ord(' '):
        if results.multi_hand_landmarks:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            cv2.imwrite(f'images/raw/frame_{timestamp}.jpg', frame)
            cv2.imwrite(f'images/landmarks/frame_{timestamp}.jpg', frame_with_landmarks)
            
            for hand_landmarks in results.multi_hand_landmarks:
                landmarks_data = []
                for landmark in hand_landmarks.landmark:
                    landmarks_data.extend([landmark.x, landmark.y, landmark.z])
                
                # Choose the appropriate CSV file based on the current mode
                current_csv = train_csv if is_training_mode else test_csv
                with open(current_csv, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([timestamp, current_label_idx] + landmarks_data)
    
    elif key == ord('a'):
        current_label_idx = (current_label_idx - 1) % len(GESTURES)
    elif key == ord('d'):
        current_label_idx = (current_label_idx + 1) % len(GESTURES)
    elif key == ord('m'):  # Toggle between training and testing mode
        is_training_mode = not is_training_mode

cap.release()
cv2.destroyAllWindows()
