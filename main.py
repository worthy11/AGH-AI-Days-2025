import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import keras
import pickle

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


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models_dict = {
    "KNN": KNeighborsClassifier(),
    "SVM": SVC(),
    "RandomForest": RandomForestClassifier()
}

print("\nModel Evaluation:")
print("-" * 50)
for name, model in models_dict.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    
    print(f"\n🔍 {name} Results:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

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
history = model_nn.fit(X_train_scaled, y_train, epochs=20, validation_split=0.2, verbose=1)

# Evaluate neural network
loss, acc = model_nn.evaluate(X_test_scaled, y_test)
print(f"\n📊 Neural Network Final Results:")
print(f"Test Accuracy: {acc:.2f}")
print(f"Test Loss: {loss:.2f}")

# Save the best performing model (you can modify this based on your needs)
best_model = models_dict["RandomForest"]  # or choose based on performance
with open('best_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)
print("\n💾 Best model saved as 'best_model.pkl'")
