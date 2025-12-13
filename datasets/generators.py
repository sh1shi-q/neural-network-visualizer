import numpy as np

def generate_xor_data(n_samples=200):
    """Generate XOR dataset"""
    np.random.seed(42)
    n = n_samples // 4
    
    X1 = np.random.randn(n, 2) * 0.1 + [0, 0]
    X2 = np.random.randn(n, 2) * 0.1 + [0, 1]
    X3 = np.random.randn(n, 2) * 0.1 + [1, 0]
    X4 = np.random.randn(n, 2) * 0.1 + [1, 1]
    
    X = np.vstack([X1, X2, X3, X4])
    y = np.array([0]*n + [1]*n + [1]*n + [0]*n).reshape(-1, 1)
    
    return X, y

def generate_spiral_data(n_samples=100, n_classes=2):
    """Generate spiral dataset"""
    np.random.seed(42)
    
    X = np.zeros((n_samples * n_classes, 2))
    y = np.zeros(n_samples * n_classes)
    
    for class_num in range(n_classes):
        ix = range(n_samples * class_num, n_samples * (class_num + 1))
        r = np.linspace(0.0, 1, n_samples)
        t = np.linspace(class_num * 4, (class_num + 1) * 4, n_samples)
        t += np.random.randn(n_samples) * 0.2
        
        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        y[ix] = class_num
    
    return X, y.reshape(-1, 1)

def generate_circles_data(n_samples=200):
    """Generate concentric circles"""
    np.random.seed(42)
    n = n_samples // 2
    
    # Inner circle
    theta_inner = np.random.uniform(0, 2*np.pi, n)
    r_inner = 0.5 + np.random.randn(n) * 0.1
    x_inner = r_inner * np.cos(theta_inner)
    y_inner = r_inner * np.sin(theta_inner)
    X_inner = np.column_stack([x_inner, y_inner])
    y_inner = np.zeros(n)
    
    # Outer circle
    theta_outer = np.random.uniform(0, 2*np.pi, n)
    r_outer = 1.5 + np.random.randn(n) * 0.1
    x_outer = r_outer * np.cos(theta_outer)
    y_outer = r_outer * np.sin(theta_outer)
    X_outer = np.column_stack([x_outer, y_outer])
    y_outer = np.ones(n)
    
    # Combine
    X = np.vstack([X_inner, X_outer])
    y = np.concatenate([y_inner, y_outer]).reshape(-1, 1)
    
    return X, y