# Import the necessary library
import numpy as np


# Build the Perceptron Model
class Perceptron:

    def __init__(self, num_inputs, learning_rate=0.01):
        # Initialize the weight and learning rate
        self.weights = np.random.rand(num_inputs + 1)
        print("self.weights matrix shape = ", self.weights.shape)
        self.learning_rate = learning_rate
        print("Weight Matrix initialized = ", self.weights)

    # Define the first linear layer
    def linear(self, inputs):
        print("In Linear Func.")
        print("input shape: ", inputs.shape)
        print("Inputs = ", inputs)

        Z = inputs @ self.weights[1:].T + self.weights[0]

        print("Z calc = ", Z)
        print("Z shape = ", Z.shape)
        return Z

    # Define the Heaviside Step function
    def Heaviside_step_fn(self, z):
        if z >= 0:
            return 1
        else:
            return 0

    # Define the Prediction
    def predict(self, inputs):
        print("in predict func inputs = ", inputs)

        Z = self.linear(inputs)
        print(f"In Pred func, Predicted Z = {Z}, for inps = {inputs}")

        try:
            pred = []
            for z in Z:
                pred.append(self.Heaviside_step_fn(z))
        except:
            return self.Heaviside_step_fn(Z)

        return pred

    # Define the Loss function
    def loss(self, prediction, target):
        loss = prediction - target
        return loss

    # Define training
    def train(self, inputs, target):
        print("In train func, Inputs Passed = ", inputs)

        prediction = self.predict(inputs)

        print("In train func, Prediction = ", prediction, "for inputs = ", inputs)

        error = self.loss(prediction, target)

        print("In train func, Error = ", error)
        print("Initial Weights = ", self.weights[1:])
        print("Initial Bias = ", self.weights[0])

        self.weights[1:] += self.learning_rate * error * inputs
        self.weights[0] += self.learning_rate * error

        print("New Weights = ", self.weights[1:])
        print("New Bias = ", self.weights[0])

    # Fit the model
    def fit(self, X, y, num_epochs):
        print("X shape(inputs passed) = ", X.shape)

        for epoch in range(num_epochs):
            print("*" * 100)

            val = list(zip(X, y))
            print("Zipped Data = ", val)

            i = 1
            for inputs, target in zip(X, y):
                print("=" * 60)
                print(f"For epoch = {epoch}, {i}th input target pair:")
                print("Inputs: ", inputs)
                print("Target: ", target)

                self.train(inputs, target)
                i += 1

            print(f"Finished Training for epoch {epoch}")

        print("Finished Training")


# Import other required libraries
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Generate a linearly separable dataset with two classes
X, y = make_blobs(n_samples=10, n_features=2, centers=2, cluster_std=3, random_state=23)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=23, shuffle=True
)

# Scale the input features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Set random seed
np.random.seed(23)

# Metadata printout
print("X Shape = ", X.shape)
print("y Shape = ", y.shape)
print("X_train Shape = ", X_train.shape)
print("y_train Shape = ", y_train.shape)
print("X_test Shape = ", X_test.shape)
print("y_test Shape = ", y_test.shape)
print("X_train = ", X_train)
print("y_train = ", y_train)

# Initialize Perceptron
perceptron = Perceptron(num_inputs=X_train.shape[1])

# Manual weight initialization
perceptron.weights[0] = 0.5
perceptron.weights[1] = 1.2
perceptron.weights[2] = -0.7

# Train the Perceptron on the training data
perceptron.fit(X_train, y_train, num_epochs=2)

# Prediction
print("+" * 100)
print("Starting Testing")

pred = perceptron.predict(X_test)

# Test the accuracy of the trained Perceptron
accuracy = np.mean(pred != y_test)
print("Accuracy:", accuracy)

# Plot the dataset
plt.scatter(X_test[:, 0], X_test[:, 1], c=pred)
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
