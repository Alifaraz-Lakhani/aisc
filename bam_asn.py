import numpy as np

# --------------------------------------------------
# Helper Functions
# --------------------------------------------------

def bipolar_sign(x):
    return np.where(x >= 0, 1, -1)

# --------------------------------------------------
# PART 1 — AUTOASSOCIATIVE NETWORK
# --------------------------------------------------

print("\n==============================")
print(" AUTOASSOCIATIVE NETWORK ")
print("==============================")

# Training vector
X = np.array([1, 1, 1, 1])

# Weight matrix using Hebbian rule: W = Xᵀ × X
W_auto = np.outer(X, X).astype(float)

# Remove self-connections (diagonal = 0)
np.fill_diagonal(W_auto, 0)

print("\nAutoassociative Weight Matrix:")
print(W_auto)

# Test vectors
test_vectors = {
    "Original": np.array([1, 1, 1, 1]),
    "One Missing": np.array([1, 1, 0, 1]),
    "One Mistake": np.array([1, -1, 1, 1]),
    "Two Missing": np.array([1, 0, 0, 1]),
    "Two Mistake": np.array([-1, -1, 1, 1])
}

print("\n--- Testing Autoassociative Memory ---")
for name, vec in test_vectors.items():
    print(f"\n{name}: input = {vec}")
    yin = W_auto @ vec
    yout = bipolar_sign(yin)
    print("Net Input:", yin)
    print("Reconstructed Output:", yout)

# --------------------------------------------------
# PART 2 — BAM NETWORK
# --------------------------------------------------

print("\n==============================")
print(" BAM (Bidirectional Memory) ")
print("==============================")

# 5×3 patterns for E and F

X_E = np.array([
    1, 1, 1,
    1, -1, -1,
    1, 1, 1,
    1, -1, -1,
    1, 1, 1
])

X_F = np.array([
    1, 1, 1,
    1, -1, -1,
    1, 1, 1,
    1, -1, -1,
    1, -1, -1
])

# Bipolar outputs
Y_E = np.array([-1, 1])
Y_F = np.array([1, 1])

# Hebbian weight matrix for BAM
W_bam = np.outer(X_E, Y_E) + np.outer(X_F, Y_F)

print("\nBAM Weight Matrix (15 × 2):")
print(W_bam)

# ----- Recall Y from X -----
def recall_Y_from_X(x_in, W):
    yin = x_in @ W
    yout = bipolar_sign(yin)
    return yin, yout

# ----- Recall X from Y -----
def recall_X_from_Y(y_in, W):
    xin = y_in @ W.T
    xout = bipolar_sign(xin)
    return xin, xout

# ---------------------
# TEST BAM
# ---------------------

for name, xvec, ytarget in [("E", X_E, Y_E), ("F", X_F, Y_F)]:
    
    print(f"\n--- Testing {name}: X → Y ---")
    yin, yout = recall_Y_from_X(xvec, W_bam)
    print("Net Input y_in:", yin)
    print("Activation y_out:", yout, "Target:", ytarget)

    print(f"--- Testing {name}: Y → X ---")
    xin, xout = recall_X_from_Y(ytarget, W_bam)
    print("Net Input x_in:", xin)
    print("Reconstructed Pattern (5×3):")
    print(xout.reshape(5, 3))
