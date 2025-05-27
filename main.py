import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import keras
import pickle

url = "data.csv"
df = pd.read_csv(url)

X = df.drop("label", axis=1).drop("timestamp", axis=1).values
y = df["label"].values
print(df.head(5))

def plot_hand(points, title=""):
    points = np.array(points).reshape(-1, 3)
    plt.scatter(points[:, 0], -points[:, 1])
    for i, (x, y, _) in enumerate(points):
        plt.text(x, -y, str(i), fontsize=8)
    plt.title(title)
    plt.axis("equal")
    plt.grid(True)
    plt.show()

# plot_hand(X[0], title=f"Przykład gestu: {y[0]}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models_dict = {
    "KNN": KNeighborsClassifier(),
    "SVM": SVC(),
    "RandomForest": RandomForestClassifier()
}

for name, model in models_dict.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    
    print(f"🔍 {name} Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print(classification_report(y_test, y_pred))

model_nn = keras.models.Sequential([
    keras.layers.Input(shape=(X.shape[1],)),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(len(np.unique(y)), activation='softmax')
])

model_nn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model_nn.fit(X_train_scaled, y_train, epochs=20, validation_split=0.2, verbose=1)

loss, acc = model_nn.evaluate(X_test_scaled, y_test)
print(f"📊 Sieć neuronowa – dokładność: {acc:.2f}")
