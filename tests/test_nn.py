from core.neural_network import NeuralNetwork
from datasets.generators import generate_xor_data
import matplotlib.pyplot as plt
import numpy as np

# Generate data
X, y = generate_xor_data(400)

# Create network
nn = NeuralNetwork()
nn.add_layer(2, 'relu')      # Input
nn.add_layer(4, 'relu')      # Hidden
nn.add_layer(1, 'sigmoid')   # Output

# Train
print("Training on XOR dataset...")
nn.train(X, y, epochs=1000, learning_rate=0.5)

# Test
predictions = nn.predict(X)
accuracy = np.mean(predictions == y)
print(f"\n🎯 Final Accuracy: {accuracy*100:.2f}%")

# Plot loss curve
plt.plot(nn.loss_history)
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.savefig('training_loss.png')
print("📊 Saved loss curve to training_loss.png")
plt.show()