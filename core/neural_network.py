import numpy as np
from .activations import sigmoid, relu, sigmoid_derivative, relu_derivative

class NeuralNetwork: 
    def __init__(self):
        self.layers = []
        self.weights = []
        self.biases = []
        self.activations = []
        self.layer_outputs = []
        self.layer_inputs = []
    
    def add_layer(self, num_neurons, activation='relu'):
        self.layers.append(num_neurons)
        self.activations.append(activation)
        
        if len(self.layers) > 1:
            prev_layer = self.layers[-2]
            curr_layer = self.layers[-1]
            
            weight = np.random.randn(prev_layer, curr_layer) * np.sqrt(2.0 / prev_layer)
            bias = np.zeros((1, curr_layer))
            
            self.weights.append(weight)
            self.biases.append(bias)
    
    def forward(self, X):
        self.layer_outputs = []
        self.layer_inputs = []
        
        activation = X
        self.layer_outputs.append(activation)
        
        for i in range(len(self.weights)):
            self.layer_inputs.append(activation)
            
            z = np.dot(activation, self.weights[i]) + self.biases[i]
            
            if self.activations[i+1] == 'sigmoid':
                activation = sigmoid(z)
            elif self.activations[i+1] == 'relu': 
                activation = relu(z)
            elif self.activations[i+1] == 'tanh':
                activation = np.tanh(z)
            
            self.layer_outputs.append(activation)
        
        return activation
    
    def compute_loss(self, y_true, y_pred):
        y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
        loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return loss
    
    def backward(self, X, y, learning_rate):
        m = X.shape[0]
        
        delta = self.layer_outputs[-1] - y
        
        for i in reversed(range(len(self.weights))):
            dW = np.dot(self. layer_inputs[i].T, delta) / m
            db = np.sum(delta, axis=0, keepdims=True) / m
            
            self. weights[i] -= learning_rate * dW
            self.biases[i] -= learning_rate * db
            
            if i > 0:
                delta = np.dot(delta, self. weights[i].T)
                
                if self.activations[i] == 'relu': 
                    delta *= relu_derivative(self.layer_outputs[i])
                elif self.activations[i] == 'sigmoid': 
                    delta *= sigmoid_derivative(self.layer_outputs[i])
    
    def train(self, X, y, epochs=1000, learning_rate=0.1, verbose=True): 
        self.loss_history = []
        
        for epoch in range(epochs):
            y_pred = self.forward(X)
            loss = self.compute_loss(y, y_pred)
            self.loss_history.append(loss)
            
            self.backward(X, y, learning_rate)
            
            if verbose and epoch % 100 == 0:
                accuracy = np.mean((y_pred > 0.5) == y)
                print(f"Epoch {epoch}/{epochs} - Loss: {loss:.4f} - Accuracy: {accuracy:.4f}")
        
        if verbose:
            print(f"Training complete! Final loss: {self.loss_history[-1]:.4f}")
    
    def predict(self, X):
        y_pred = self.forward(X)
        return (y_pred > 0.5).astype(int)