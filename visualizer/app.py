import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path
sys.path. append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.neural_network import NeuralNetwork
from datasets.generators import generate_xor_data, generate_spiral_data, generate_circles_data

# Page config
st.set_page_config(page_title="Neural Network Visualizer", layout="wide", page_icon="🧠")

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight:  bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .subtitle {
        color: #888;
        font-size: 1.1rem;
        margin-top:  -10px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">🧠 Neural Network Visualizer</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Build and train neural networks from scratch - watch them learn in real-time!</p>', unsafe_allow_html=True)

# Sidebar - Network Configuration
st.sidebar.header("⚙️ Network Configuration")

# Dataset selection
dataset = st.sidebar.selectbox(
    "Choose Dataset",
    ["XOR", "Spiral", "Circles"],
    help="Select which 2D dataset to train on"
)

# Architecture
st.sidebar.subheader("Network Architecture")
num_hidden_layers = st.sidebar.selectbox("Number of Hidden Layers", [1, 2, 3], index=0)

hidden_layers_config = []
for i in range(num_hidden_layers):
    neurons = st.sidebar.slider(f"Hidden Layer {i+1} Neurons", 2, 20, 4 if i == 0 else 4, key=f"layer_{i}")
    hidden_layers_config.append(neurons)

activation = st.sidebar.selectbox("Activation Function", ["relu", "sigmoid", "tanh"], 
                                  help="Activation function for hidden layers")

# Training parameters
st.sidebar.subheader("Training Parameters")
learning_rate = st.sidebar.slider("Learning Rate", 0.01, 2.0, 0.5, 0.01,
                                  help="Step size for gradient descent")
epochs = st.sidebar.slider("Epochs", 100, 10000, 1000, 100,
                          help="Number of training iterations")

# Visualization options
st.sidebar.subheader("Visualization Options")
show_decision_boundary = st.sidebar. checkbox("Show Decision Boundary", value=True,
                                             help="Visualize the classification regions")
show_train_animation = st.sidebar.checkbox("Animate Training", value=False,
                                           help="Show training progress in real-time (slower)")

# Generate data based on selection
if dataset == "XOR": 
    X, y = generate_xor_data(400)
elif dataset == "Spiral":
    X, y = generate_spiral_data(100, 2)
else:  # Circles
    X, y = generate_circles_data(400)

# Function to plot decision boundary
def plot_decision_boundary(model, X, y, title="Decision Boundary"):
    # Create mesh
    h = 0.02
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1]. min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # Predict for each point in mesh
    Z = model.forward(np.c_[xx. ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Decision boundary (filled contour)
    contour = ax.contourf(xx, yy, Z, levels=20, cmap='RdYlBu', alpha=0.6)
    
    # Data points
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y. ravel(), cmap='RdYlBu', 
                        edgecolors='black', s=50, linewidths=1.5, alpha=0.9)
    
    ax.set_xlabel('X1', fontsize=12)
    ax.set_ylabel('X2', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(contour, ax=ax)
    cbar.set_label('Prediction Probability', rotation=270, labelpad=20)
    
    return fig

# Display dataset
st.header(f"📊 {dataset} Dataset")
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Original Data")
    fig, ax = plt.subplots(figsize=(6, 6))
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y. ravel(), cmap='RdYlBu', 
                        edgecolors='k', s=50, alpha=0.8, linewidths=1.5)
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_title(f'{dataset} Dataset')
    plt.colorbar(scatter, ax=ax, label='Class')
    st.pyplot(fig)
    plt.close()

# Info about dataset
with col2:
    st.metric("Total Samples", len(X))
    st.metric("Features", X.shape[1])
    st.metric("Classes", len(np.unique(y)))
    
    # Architecture summary
    st.subheader("Network Architecture")
    architecture = [2] + hidden_layers_config + [1]
    st.code(" → ".join([f"Layer {i}:  {n} neurons" for i, n in enumerate(architecture)]))

# Train button
if st.sidebar.button("🚀 Train Network", type="primary"):
    
    # Create network
    nn = NeuralNetwork()
    nn.add_layer(2, activation)  # Input layer
    
    # Add hidden layers
    for neurons in hidden_layers_config:  
        nn.add_layer(neurons, activation)
    
    nn.add_layer(1, 'sigmoid')  # Output layer
    
    with st.spinner("🔄 Training neural network..."):
        
        # Training
        progress_bar = st.progress(0)
        status_text = st. empty()
        
        nn.loss_history = []
        for epoch in range(epochs):
            y_pred = nn.forward(X)
            loss = nn.compute_loss(y, y_pred)
            nn.loss_history.append(loss)
            nn.backward(X, y, learning_rate)
            
            if epoch % 100 == 0:
                progress_bar.progress((epoch + 1) / epochs)
                accuracy = np.mean((y_pred > 0.5) == y)
                status_text.text(f"Epoch {epoch}/{epochs} - Loss: {loss:.4f} - Accuracy: {accuracy:.4f}")
        
        progress_bar.progress(1.0)
        
        # Final predictions
        predictions = nn.predict(X)
        final_accuracy = np.mean(predictions == y)
        
        # Show results
        st.success(f"✅ Training Complete! Final Accuracy: {final_accuracy*100:.2f}%")
        
        # Results visualization
        st.header("🎯 Results")
        
        col_res1, col_res2 = st.columns(2)
        
        with col_res1:
            st.subheader("Training Data with Predictions")
            fig, ax = plt.subplots(figsize=(8, 6))
            
            # Plot all points
            scatter = ax.scatter(X[:, 0], X[:, 1], c=predictions. ravel(), 
                               cmap='RdYlBu', edgecolors='black', s=80, 
                               linewidths=2, alpha=0.8)
            
            # Mark misclassified points with X
            misclassified = predictions. ravel() != y.ravel()
            if misclassified.any():
                ax.scatter(X[misclassified, 0], X[misclassified, 1], 
                          marker='x', s=200, c='red', linewidths=3, 
                          label=f'Errors ({misclassified.sum()})')
                ax.legend()
            
            ax.set_xlabel('X1', fontsize=12)
            ax.set_ylabel('X2', fontsize=12)
            ax.set_title(f'Predictions (Accuracy: {final_accuracy*100:.1f}%)', 
                        fontsize=14, fontweight='bold')
            plt.colorbar(scatter, ax=ax, label='Predicted Class')
            st.pyplot(fig)
            plt.close()
        
        with col_res2:
            st.subheader("Decision Boundary")
            fig, ax = plt.subplots(figsize=(8, 6))
            
            # Create decision boundary
            h = 0.02
            x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
            y_min, y_max = X[:, 1].min() - 0.5, X[:, 1]. max() + 0.5
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                                 np.arange(y_min, y_max, h))
            
            # Predict for each point in mesh
            Z = nn.forward(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            
            # Plot decision boundary
            contour = ax.contourf(xx, yy, Z, levels=20, cmap='RdYlBu', alpha=0.6)
            
            # Plot original data points
            scatter = ax.scatter(X[:, 0], X[:, 1], c=y.ravel(), 
                               cmap='RdYlBu', edgecolors='black', 
                               s=80, linewidths=2, alpha=0.9)
            
            ax.set_xlabel('X1', fontsize=12)
            ax.set_ylabel('X2', fontsize=12)
            ax.set_title('What the Network Learned', fontsize=14, fontweight='bold')
            
            cbar = plt.colorbar(contour, ax=ax)
            cbar.set_label('Network Output (0 to 1)', rotation=270, labelpad=20)
            
            st.pyplot(fig)
            plt.close()
        
        # Loss curve
        st.header("📈 Training Progress")
        fig_loss, ax = plt.subplots(figsize=(12, 4))
        ax.plot(nn.loss_history, linewidth=2, color='#667eea')
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('Training Loss Over Time', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig_loss)
        plt.close()
        
        # Metrics
        st.header("📊 Performance Metrics")
        metric_cols = st.columns(4)
        
        correct = (predictions == y).sum()
        total = len(y)
        errors = total - correct
        
        with metric_cols[0]:
            st.metric("Final Accuracy", f"{final_accuracy*100:.2f}%", 
                     delta=f"{(final_accuracy - 0.5)*100:.1f}% vs random")
        with metric_cols[1]:
            st.metric("Correct Predictions", f"{correct}/{total}")
        with metric_cols[2]:
            st.metric("Errors", f"{errors}")
        with metric_cols[3]: 
            total_params = sum(w.size for w in nn.weights) + sum(b.size for b in nn.biases)
            st.metric("Network Parameters", total_params)
        
        # Show confusion info
        st.header("🔍 Classification Details")
        class_0_correct = ((predictions == 0) & (y == 0)).sum()
        class_0_total = (y == 0).sum()
        class_1_correct = ((predictions == 1) & (y == 1)).sum()
        class_1_total = (y == 1).sum()
        
        detail_cols = st.columns(2)
        with detail_cols[0]:
            st.info(f"**Class 0 (Blue):** {class_0_correct}/{class_0_total} correct ({class_0_correct/class_0_total*100:.1f}%)")
        with detail_cols[1]:
            st. info(f"**Class 1 (Red):** {class_1_correct}/{class_1_total} correct ({class_1_correct/class_1_total*100:.1f}%)")

# Info section
st.sidebar.markdown("---")
st.sidebar.info("""
**How to use:**
1. Choose a dataset (XOR, Spiral, or Circles)
2. Configure network architecture
3. Set training parameters
4. Toggle visualization options
5. Click 'Train Network'
6. Watch it learn! 

**Tips:**
- More neurons = more learning capacity
- Higher learning rate = faster but less stable
- More epochs = better learning (but slower)
- Decision boundary shows what the network "sees"
""")

st.sidebar.markdown("---")
st.sidebar.markdown("**Built from scratch with NumPy** 🔢")
st.sidebar.markdown("*No TensorFlow, no PyTorch - pure mathematics! *")