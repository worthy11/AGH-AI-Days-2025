import cv2
import mediapipe as mp
import numpy as np
import os
import pickle
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

models_dir = 'models'
models = []
model_names = ['svm', 'random_forest', 'knn', 'decision_tree']

for model_name in model_names:
    model_path = os.path.join(models_dir, f'{model_name}_model.pkl')
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            models.append(pickle.load(f))
    else:
        print(f"Warning: Model {model_name} not found at {model_path}")
        models.append(None)

cap = cv2.VideoCapture(0)
show_landmarks = False
current_model_idx = 0

def get_confidence_color(confidence):
    # Convert confidence to color (red for low confidence, green for high)
    # confidence should be between 0 and 1
    r = int(255 * (1 - confidence))
    g = int(255 * confidence)
    return (0, g, r)  # BGR format

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally
    frame = cv2.flip(frame, 1)

    # Add model name text to the frame
    model_name = model_names[current_model_idx] if models[current_model_idx] is not None else "No model"
    cv2.putText(frame, f"Model: {model_name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    frame_with_landmarks = frame.copy()

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            if show_landmarks:
                mp_draw.draw_landmarks(
                    frame_with_landmarks,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

            # Extract landmarks for prediction
            landmarks_data = []
            for landmark in hand_landmarks.landmark:
                landmarks_data.extend([landmark.x, landmark.y, landmark.z])
            
            # Make prediction if model is available
            if models[current_model_idx] is not None:
                try:
                    # Get prediction and confidence
                    if hasattr(models[current_model_idx], 'predict_proba'):
                        # For models that support probability estimates
                        probs = models[current_model_idx].predict_proba([landmarks_data])[0]
                        prediction = np.argmax(probs)
                        confidence = probs[prediction]
                    else:
                        # For models that don't support probability estimates
                        prediction = models[current_model_idx].predict([landmarks_data])[0]
                        confidence = 1.0  # Default confidence if not available

                    # Display prediction and confidence
                    prediction_text = f"Prediction: {GESTURES[prediction]}"
                    confidence_text = f"Confidence: {confidence:.2f}"
                    
                    # Get color based on confidence
                    color = get_confidence_color(confidence)
                    
                    # Display prediction and confidence
                    cv2.putText(frame_with_landmarks, prediction_text, (10, 70), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                    cv2.putText(frame_with_landmarks, confidence_text, (10, 110), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                except Exception as e:
                    print(f"Error making prediction: {e}")

    cv2.imshow('Hand Gesture Recognition', frame_with_landmarks if show_landmarks else frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('k'):
        show_landmarks = not show_landmarks
    elif key == ord('a'):
        current_model_idx = (current_model_idx - 1) % len(models)
    elif key == ord('d'):
        current_model_idx = (current_model_idx + 1) % len(models)

cap.release()
cv2.destroyAllWindows() 