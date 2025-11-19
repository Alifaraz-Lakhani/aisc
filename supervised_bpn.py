import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------
# LOAD DATASET FROM INTERNET (Correct Columns)
# ---------------------------------------------------------
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

# ---------------------------------------------------------
# PREPROCESSING
# ---------------------------------------------------------
numeric_cols = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
X = df[numeric_cols].copy()

# Fill missing values
X = X.fillna(X.median())

y = df["Survived"].values

# Normalize inputs
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# ---------------------------------------------------------
# ACTIVATION FUNCTIONS
# ---------------------------------------------------------
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# ---------------------------------------------------------
# NETWORK INITIALIZATION
# ---------------------------------------------------------
n_inputs = X_train.shape[1]
n_hidden = 5
n_outputs = 1
lr = 0.01
epochs = 200

W_input_hidden = np.random.rand(n_inputs, n_hidden)
b_hidden = np.random.rand(n_hidden)

W_hidden_output = np.random.rand(n_hidden, n_outputs)
b_output = np.random.rand(n_outputs)

# ---------------------------------------------------------
# TRAINING LOOP (NO SPAM OUTPUT)
# ---------------------------------------------------------
for epoch in range(epochs):
    for xi, yi in zip(X_train, y_train):

        # Forward pass
        hidden_net = np.dot(xi, W_input_hidden) + b_hidden
        hidden_out = sigmoid(hidden_net)

        output_net = np.dot(hidden_out, W_hidden_output) + b_output
        final_out = sigmoid(output_net)

        # Backprop
        error = yi - final_out

        d_output = error * sigmoid_derivative(final_out)
        d_hidden = d_output.dot(W_hidden_output.T) * sigmoid_derivative(hidden_out)

        # Weight updates
        W_hidden_output += lr * np.outer(hidden_out, d_output)
        b_output += lr * d_output

        W_input_hidden += lr * np.outer(xi, d_hidden)
        b_hidden += lr * d_hidden

# ---------------------------------------------------------
# OUTPUT EXACTLY LIKE YOUR SCREENSHOT
# ---------------------------------------------------------
print("\nFinal Weights & Biases")
print("Input -> Hidden Weights:")
print(W_input_hidden)

print("\nHidden Bias:")
print(b_hidden)

print("\nHidden -> Output Weights:")
print(W_hidden_output)

print("\nOutput Bias:")
print(b_output)

# ---------------------------------------------------------
# TESTING
# ---------------------------------------------------------
predictions = []
for xi in X_test:
    hidden_net = np.dot(xi, W_input_hidden) + b_hidden
    hidden_out = sigmoid(hidden_net)

    output_net = np.dot(hidden_out, W_hidden_output) + b_output
    final_out = sigmoid(output_net)

    predictions.append(round(final_out[0]))

accuracy = np.mean(np.array(predictions) == y_test)

print(f"\nTest Accuracy: {accuracy:.2f}")
