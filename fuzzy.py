# Chapter 10, Question 6
import numpy as np

U = [0.64, 0.645, 0.65, 0.655, 0.66]

A_vals = [0, 0.75, 1, 0.5, 0]
B_vals = [0, 0.25, 0.75, 1, 0.5]

A = np.array(A_vals)
B = np.array(B_vals)

def union(X, Y):
    return np.maximum(X, Y)

def intersection(X, Y):
    return np.minimum(X, Y)

def complement(X):
    return 1 - X

operations = {
    "a": union(A, B),
    "b": intersection(A, B),
    "c": complement(A),
    "d": complement(B),
    "e": complement(union(A, B)),
    "f": complement(intersection(A, B))
}

labels = {
    "a": "A ∪ B",
    "b": "A ∩ B",
    "c": "A'",
    "d": "B'",
    "e": "(A ∪ B)'",
    "f": "(A ∩ B)'"
}

def pretty_set(output_vec):
    return "{ " + " , ".join(
        [f"{round(output_vec[i], 2)}/{U[i]}" for i in range(len(U))]
    ) + " }"

for key in sorted(operations.keys()):
    print(f"{labels[key]} = {pretty_set(operations[key])}\n")
-----------------------------------------------------------------------------------------
# Chapter 10, Question 10
import numpy as np

U = [0, 1, 2, 3, 4, 5]

A_vals = [0, 0.2, 0.7, 0.8, 0.9, 1]
B_vals = [0, 0.1, 0.3, 0.2, 0.4, 0.5]

A = np.array(A_vals)
B = np.array(B_vals)

def addition(X, Y):
    return X + Y

def product(X, Y):
    return X * Y

def difference(X, Y):
    return X - Y

operations = {
    "a": addition(A, B) - product(A, B),          # Algebraic sum
    "b": product(A, B),                           # Algebraic product
    "c": np.minimum(1, addition(A, B)),           # Bounded sum
    "d": np.maximum(0, difference(A, B)),         # Bounded difference
}

labels = {
    "a": "Algebraic Sum",
    "b": "Algebraic Product",
    "c": "Bounded Sum",
    "d": "Bounded Difference"
}

def pretty_set(output_vec):
    return "{ " + " , ".join(
        [f"{round(output_vec[i], 2)}/{U[i]}" for i in range(len(U))]
    ) + " }"

for key in sorted(operations.keys()):
    print(f"{labels[key]} = {pretty_set(operations[key])}\n")
------------------------------------------------------------------------------------------
# Chapter 11, Question 6
import numpy as np

X = np.array([0.72, 0.725, 0.75, 0.775, 0.78])
M = np.array([0.0, 0.8, 1.0, 0.8, 0.0])

Y = np.array([21, 22, 23, 24, 25, 26, 27])
N = np.array([0.0, 0.2, 0.7, 1.0, 0.7, 0.2, 0.0])

# Build fuzzy relation R (M × N) using min
R = np.zeros((len(X), len(Y)))
for i in range(len(X)):
    for j in range(len(Y)):
        R[i, j] = min(M[i], N[j])

print("Relation R (M × N):")
print(R)

# Modified fuzzy set M1
M1 = np.array([0.0, 0.8, 1.0, 0.6, 0.0])

# Max–Min composition
S_max_min = np.zeros(len(Y))

for z in range(len(Y)):
    max_min_value = 0
    for x in range(len(X)):
        min_value = min(M1[x], R[x, z])
        max_min_value = max(max_min_value, min_value)
    S_max_min[z] = max_min_value

print("\nMax-Min Composition S = M1 ∘ R:")
for z in range(len(Y)):
    print(f"Altitude {Y[z]} k-feet: {S_max_min[z]}")

# Max–Product composition
S_max_prod = np.zeros(len(Y))

for z in range(len(Y)):
    max_prod_value = 0
    for x in range(len(X)):
        prod_value = M1[x] * R[x, z]
        max_prod_value = max(max_prod_value, prod_value)
    S_max_prod[z] = max_prod_value

print("\nMax-Product Composition S' = M1 * R:")
for z in range(len(Y)):
    print(f"Altitude {Y[z]} k-feet: {S_max_prod[z]}")
