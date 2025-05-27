import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import keras
import pickle
import os

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

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

# Load training and testing data
train_df = pd.read_csv("data_train.csv")
test_df = pd.read_csv("data_test.csv")

# Prepare training data
X_train = train_df.drop("label", axis=1).drop("timestamp", axis=1).values
y_train = train_df["label"].values

# Prepare testing data
X_test = test_df.drop("label", axis=1).drop("timestamp", axis=1).values
y_test = test_df["label"].values

print("Training data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)
print("\nTraining data sample:")
print(train_df.head(5))

# Transform landmarks
print("\nTransforming landmarks...")
# First normalize hand size
X_train = normalize_hand_size(X_train)
X_test = normalize_hand_size(X_test)
# Then transform to relative distances
X_train = transform_to_relative_distances(X_train)
X_test = transform_to_relative_distances(X_test)

print("Transformed training data shape:", X_train.shape)
print("Transformed testing data shape:", X_test.shape)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models_dict = {
    "KNN": KNeighborsClassifier(),
    "SVM": SVC(),
    "RandomForest": RandomForestClassifier()
}

# Dictionary to store model accuracies
model_accuracies = {}

print("\nModel Evaluation:")
print("-" * 50)
for name, model in models_dict.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    model_accuracies[name] = accuracy
    
    print(f"\n🔍 {name} Results:")
    print(f"Accuracy: {accuracy:.2f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save each model
    model_path = f'models/{name.lower()}_model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"💾 Saved {name} model to {model_path}")

# Neural Network model
print("\n🔮 Neural Network Training:")
print("-" * 50)
model_nn = keras.models.Sequential([
    keras.layers.Input(shape=(X_train.shape[1],)),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(len(np.unique(y_train)), activation='softmax')
])

model_nn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model_nn.fit(X_train_scaled, y_train, epochs=100, validation_split=0.2, verbose=1)

# Evaluate neural network
loss, acc = model_nn.evaluate(X_test_scaled, y_test)
model_accuracies["NeuralNetwork"] = acc
print(f"\n📊 Neural Network Final Results:")
print(f"Test Accuracy: {acc:.2f}")
print(f"Test Loss: {loss:.2f}")

# Save neural network model
model_nn.save('models/neural_network_model')
print("💾 Saved Neural Network model to models/neural_network_model")

# Save the scaler
scaler_path = 'models/scaler.pkl'
with open(scaler_path, 'wb') as f:
    pickle.dump(scaler, f)
print(f"💾 Saved scaler to {scaler_path}")

# Save the best performing model based on accuracy
best_model_name = max(model_accuracies, key=model_accuracies.get)
best_model = models_dict.get(best_model_name, model_nn)
best_model_path = f'models/best_model.pkl'

if best_model_name == "NeuralNetwork":
    best_model.save('models/best_model')
    print(f"\n🏆 Best model is {best_model_name} with accuracy {model_accuracies[best_model_name]:.2f}")
    print("💾 Saved best model to models/best_model")
else:
    with open(best_model_path, 'wb') as f:
        pickle.dump(best_model, f)
    print(f"\n🏆 Best model is {best_model_name} with accuracy {model_accuracies[best_model_name]:.2f}")
    print(f"💾 Saved best model to {best_model_path}")

# Save model accuracies for reference
accuracies_path = 'models/model_accuracies.pkl'
with open(accuracies_path, 'wb') as f:
    pickle.dump(model_accuracies, f)
print(f"💾 Saved model accuracies to {accuracies_path}")
