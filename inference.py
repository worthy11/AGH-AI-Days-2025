import cv2
import mediapipe as mp
import numpy as np
import os
import pickle
from datetime import datetime
import tensorflow as tf

GESTURES = {
    0: "palm_up",
    1: "fist",
    2: "peace",
    3: "thumbs_up",
    4: "ok_sign"
}

def normalize_hand_size(X):
    """
    Normalize hand size by scaling width and height to [0, 1] range.
    This makes the features invariant to the distance from the camera.
    """
    # Reshape to get landmarks as 3D points
    landmarks = X.reshape(-1, 21, 3)
    
    # Get min and max coordinates for each hand
    min_coords = np.min(landmarks, axis=1)
    max_coords = np.max(landmarks, axis=1)
    
    # Calculate width and height
    width = max_coords[:, 0] - min_coords[:, 0]
    height = max_coords[:, 1] - min_coords[:, 1]
    
    # Scale coordinates
    landmarks_normalized = landmarks.copy()
    landmarks_normalized[:, :, 0] = (landmarks[:, :, 0] - min_coords[:, 0:1]) / width[:, np.newaxis]
    landmarks_normalized[:, :, 1] = (landmarks[:, :, 1] - min_coords[:, 1:2]) / height[:, np.newaxis]
    
    # Return flattened array
    return landmarks_normalized.reshape(-1, 63)

def transform_to_relative_distances(X):
    """
    Transform landmarks to relative distances from the first landmark (wrist).
    Each landmark is represented by its Euclidean distance from the wrist.
    """
    # Reshape to get landmarks as 3D points
    landmarks = X.reshape(-1, 21, 3)
    
    # Get the wrist position (first landmark)
    wrist = landmarks[:, 0:1, :]
    
    # Calculate Euclidean distances from wrist to all other landmarks
    distances = np.sqrt(np.sum((landmarks - wrist) ** 2, axis=2))
    
    # Return flattened array of distances
    return distances.reshape(-1, 21)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

# Load models and scaler
models_dir = 'models'
models = []
model_names = ['svm', 'randomforest', 'knn', 'neural_network']

# Load scaler
scaler_path = os.path.join(models_dir, 'scaler.pkl')
if os.path.exists(scaler_path):
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
else:
    print(f"Warning: Scaler not found at {scaler_path}")
    scaler = None

# Load models
for model_name in model_names:
    if model_name == 'neural_network':
        # Load Keras model
        model_path = os.path.join(models_dir, 'neural_network_model')
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
        # Load pickle models
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
            mp_draw.draw_landmarks(
                frame_with_landmarks,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            landmarks_data = []
            for landmark in hand_landmarks.landmark:
                landmarks_data.extend([landmark.x, landmark.y, landmark.z])
            
            if models[current_model_idx] is not None and scaler is not None:
                try:
                    # First normalize hand size
                    landmarks_normalized = normalize_hand_size(np.array([landmarks_data]))
                    
                    # Then transform to relative distances
                    landmarks_transformed = transform_to_relative_distances(landmarks_normalized)
                    
                    # Scale the features
                    landmarks_scaled = scaler.transform(landmarks_transformed)
                    
                    # Get prediction and confidence
                    if model_name == 'neural_network':
                        # Neural network prediction
                        probs = models[current_model_idx].predict(landmarks_scaled, verbose=0)[0]
                        prediction = np.argmax(probs)
                        confidence = probs[prediction]
                    elif hasattr(models[current_model_idx], 'predict_proba'):
                        # For models that support probability estimates
                        probs = models[current_model_idx].predict_proba(landmarks_scaled)[0]
                        prediction = np.argmax(probs)
                        confidence = probs[prediction]
                    else:
                        # For models that don't support probability estimates
                        prediction = models[current_model_idx].predict(landmarks_scaled)[0]
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