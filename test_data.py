import matplotlib.pyplot as plt
from datasets.generators import generate_xor_data, generate_spiral_data, generate_circles_data

# Test XOR
X, y = generate_xor_data(400)
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.scatter(X[:, 0], X[:, 1], c=y. ravel(), cmap='viridis', edgecolors='k')
plt.title('XOR Dataset')

# Test Spiral
X, y = generate_spiral_data(100, 2)
plt.subplot(1, 3, 2)
plt.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap='viridis', edgecolors='k')
plt.title('Spiral Dataset')

# Test Circles
X, y = generate_circles_data(400)
plt.subplot(1, 3, 3)
plt.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap='viridis', edgecolors='k')
plt.title('Circles Dataset')

plt.tight_layout()
plt.savefig('test_datasets.png')
print("All datasets generated successfully!")
print("Saved visualization to test_datasets.png")
plt.show()