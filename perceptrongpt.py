import numpy as np
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


# -------------------------------------------------------
# Perceptron Class (Clean + Minimal Helpful Print Statements)
# -------------------------------------------------------
class Perceptron:

    def __init__(self, num_inputs, learning_rate=0.01):
        self.weights = np.random.randn(num_inputs + 1)
        self.learning_rate = learning_rate
        print("[INIT] Initial Weights:", self.weights)

    def linear(self, x):
        return np.dot(x, self.weights[1:]) + self.weights[0]

    def activation(self, z):
        return 1 if z >= 0 else 0

    def predict(self, X):
        if X.ndim == 2:
            return np.array([self.activation(self.linear(x)) for x in X])
        return self.activation(self.linear(X))

    def train(self, x, target):
        prediction = self.predict(x)
        error = target - prediction

        # Light print — shows what is happening
        print(f"   [TRAIN] x={x}, target={target}, pred={prediction}, error={error}")

        # Update rule
        self.weights[1:] += self.learning_rate * error * x
        self.weights[0] += self.learning_rate * error

    def fit(self, X, y, epochs=10):
        print("\n[TRAINING STARTED]")
        for epoch in range(epochs):
            print(f"\n--- Epoch {epoch+1} ---")
            for xi, target in zip(X, y):
                self.train(xi, target)
            print(f"[EPOCH {epoch+1}] Updated Weights:", self.weights)
        print("\n[TRAINING FINISHED]\n")


# -------------------------------------------------------
# DATASET CREATION
# -------------------------------------------------------
X, y = make_blobs(
    n_samples=300,
    n_features=2,
    centers=2,
    cluster_std=2,
    random_state=42
)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Standardization
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("[INFO] Dataset prepared.")
print(" - Training samples:", X_train.shape[0])
print(" - Testing samples:", X_test.shape[0])

# -------------------------------------------------------
# TRAIN PERCEPTRON
# -------------------------------------------------------
model = Perceptron(num_inputs=2, learning_rate=0.01)
model.fit(X_train, y_train, epochs=5)

# -------------------------------------------------------
# TESTING
# -------------------------------------------------------
print("[TESTING] Predicting on test data...")
pred = model.predict(X_test)

accuracy = np.mean(pred == y_test)
print(f"\nFinal Accuracy: {accuracy*100:.2f}%")

# -------------------------------------------------------
# VISUALIZATION
# -------------------------------------------------------
plt.scatter(X_test[:, 0], X_test[:, 1], c=pred)
plt.title("Perceptron Predictions")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
