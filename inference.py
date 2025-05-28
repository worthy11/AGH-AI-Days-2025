import os
import cv2
import pickle
import numpy as np
import mediapipe as mp
import tensorflow as tf

GESTURES = {
    0: "palm_up",
    1: "fist",
    2: "peace",
    3: "thumbs_up",
    4: "ok_sign"
}

def transform1(X):
    landmarks = X.reshape(-1, 21, 3).copy()
    # ???
    return landmarks.reshape(-1, 63)

def transform2(X):
    landmarks = X.reshape(-1, 21, 3)
    # ???
    return landmarks.reshape(-1, 21)

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
model_names = ['svc', 'rfc', 'knn', 'nn']

for model_name in model_names:
    if model_name == 'nn':
        model_path = os.path.join(models_dir, 'nn_model')
        if os.path.exists(model_path):
            try:
                model = tf.keras.models.load_model(model_path)
                models.append(model)
            except Exception as e:
                print(f"Error loading neural network model: {e}")
                models.append(None)
        else:
            print(f"Warning: Neural network model not found at {model_path}")
            models.append(None)
    else:
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
    r = int(255 * (1 - confidence))
    g = int(255 * confidence)
    return (0, g, r)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    model_name = model_names[current_model_idx] if models[current_model_idx] is not None else "No model"
    cv2.putText(frame, f"Model: {model_name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    frame_with_landmarks = frame.copy()

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                frame_with_landmarks,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            landmarks = []
            for landmark in hand_landmarks.landmark:
                landmarks.extend([landmark.x, landmark.y, landmark.z])
            
            landmarks = np.array(landmarks).reshape(-1, 63)
            if models[current_model_idx] is not None:
                try:
                    landmarks = transform1(landmarks)
                    landmarks = transform2(landmarks)
                    
                    if model_name == 'nn':
                        probs = models[current_model_idx].predict(landmarks, verbose=0)[0]
                        prediqction = np.argmax(probs)
                        confidence = probs[prediction]
                    elif hasattr(models[current_model_idx], 'predict_proba'):
                        probs = models[current_model_idx].predict_proba(landmarks)[0]
                        prediction = np.argmax(probs)
                        confidence = probs[prediction]
                    else:
                        prediction = models[current_model_idx].predict(landmarks)[0]
                        confidence = 1.0

                    prediction_text = f"Prediction: {GESTURES[prediction]}"
                    print(prediction_text)
                    confidence_text = f"Confidence: {confidence:.2f}"
                    print(confidence_text)
                    
                    color = get_confidence_color(confidence)
                    
                    cv2.putText(frame_with_landmarks, prediction_text, (10, 70), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                    cv2.putText(frame_with_landmarks, confidence_text, (10, 110), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                except Exception as e:
                    print(f"Error making prediction: {e}")

    cv2.imshow('Hand Gesture Recognition', frame_with_landmarks)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('a'):
        current_model_idx = (current_model_idx - 1) % len(models)
    elif key == ord('d'):
        current_model_idx = (current_model_idx + 1) % len(models)

cap.release()
cv2.destroyAllWindows() 