import numpy as np
from core.activations import sigmoid, sigmoid_derivative, relu, relu_derivative, tanh, tanh_derivative

# Test sigmoid
print("Testing Sigmoid:")
x = np.array([0, 1, -1, 2, -2])
print(f"Input: {x}")
print(f"Sigmoid:  {sigmoid(x)}")
print(f"Should be close to: [0.5, 0.73, 0.27, 0.88, 0.12]")
print()

# Test ReLU
print("Testing ReLU:")
x = np.array([0, 1, -1, 2, -2])
print(f"Input: {x}")
print(f"ReLU: {relu(x)}")
print(f"Should be:  [0, 1, 0, 2, 0]")
print()

# Test Tanh
print("Testing Tanh:")
x = np.array([0, 1, -1])
print(f"Input: {x}")
print(f"Tanh: {tanh(x)}")
print(f"Should be close to: [0, 0.76, -0.76]")
print()

# Test derivatives
print("Testing Sigmoid Derivative:")
x = np.array([0.5, 0.73, 0.27])  # These are sigmoid outputs
print(f"Input (sigmoid output): {x}")
print(f"Derivative: {sigmoid_derivative(x)}")
print(f"Should be close to: [0.25, 0.20, 0.20]")