"""
my_lib_lab2.py - Machine Learning Lab 2 Library (TEACHER VERSION)
BFA3 Dauphine - M2 Level

This library contains:
- Gradient descent implementations (1D and 2D)
- Quadratic OLS objective function (complete implementation)
- Logistic Regression objective function (complete implementation)
"""

import numpy as np
import random


# =============================================================================
# GRADIENT DESCENT FUNCTIONS
# =============================================================================

def gradient_descent_1d(func, domain, derivative, alpha, max_iterations=1000, epsilon=1e-6):
    """
    Gradient descent for 1D parameter optimization.
    
    Args:
        func: The function to minimize (callable)
        domain: Tuple (a, b) representing the domain for the parameter
        derivative: The derivative function of func (callable)
        alpha: Step size (learning rate)
        max_iterations: Maximum number of iterations (default: 1000)
        epsilon: Convergence threshold (default: 1e-6)
    
    Returns:
        List of parameter values [p0, p1, ..., p_final] showing the path to minimum.
    """
    a, b = domain
    
    # Initialize at a random point in the domain
    x = random.uniform(a + 0.1 * (b - a), b - 0.1 * (b - a))
    
    path = [x]
    
    for i in range(max_iterations):
        grad = derivative(x)
        
        if abs(grad) < epsilon:
            break
        
        x_new = x - alpha * grad
        x_new = max(a, min(b, x_new))
        
        path.append(x_new)
        
        if abs(x_new - x) < 1e-10:
            break
            
        x = x_new
    
    return path


def gradient_descent_2d(func, domain, gradient, alpha, max_iterations=1000, epsilon=1e-6):
    """
    Gradient descent for 2D parameter optimization.
    
    Args:
        func: The function to minimize (callable), takes numpy array [p1, p2]
        domain: Tuple of tuples ((p1_min, p1_max), (p2_min, p2_max)) for parameter ranges
        gradient: The gradient function (callable), returns numpy array [dp1, dp2]
        alpha: Step size (learning rate)
        max_iterations: Maximum number of iterations (default: 1000)
        epsilon: Convergence threshold for gradient norm (default: 1e-6)
    
    Returns:
        List of parameter values [[p1_0, p2_0], [p1_1, p2_1], ..., [p1_final, p2_final]]
    """
    p1_range, p2_range = domain
    
    # Initialize at random point
    p1 = random.uniform(p1_range[0] + 0.1 * (p1_range[1] - p1_range[0]),
                        p1_range[1] - 0.1 * (p1_range[1] - p1_range[0]))
    p2 = random.uniform(p2_range[0] + 0.1 * (p2_range[1] - p2_range[0]),
                        p2_range[1] - 0.1 * (p2_range[1] - p2_range[0]))
    
    params = np.array([p1, p2])
    path = [params.tolist()]
    
    for i in range(max_iterations):
        grad = gradient(params)
        
        if np.linalg.norm(grad) < epsilon:
            break
        
        new_params = params - alpha * grad
        
        # Project onto domain
        new_params[0] = max(p1_range[0], min(p1_range[1], new_params[0]))
        new_params[1] = max(p2_range[0], min(p2_range[1], new_params[1]))
        
        path.append(new_params.tolist())
        
        if np.linalg.norm(new_params - params) < 1e-10:
            break
        
        params = new_params
    
    return path


# =============================================================================
# QUADRATIC OLS OBJECTIVE FUNCTION
# =============================================================================

def create_quadratic_ols_objective(X, y):
    """
    Create Quadratic OLS objective: L(a, b) with c computed optimally.
    Model: y = a*x^2 + b*x + c
    
    Args:
        X: Input features (1D numpy array of x values)
        y: Target values (1D numpy array)
    
    Returns:
        func: Function that computes loss given params [a, b]
        gradient: Function that computes gradient given params [a, b]
        get_params: Function that returns (a, b, c) given [a, b]
    """
    n = len(X)
    X2 = X**2
    y_mean = np.mean(y)
    x_mean = np.mean(X)
    x2_mean = np.mean(X2)
    
    def func(params):
        a, b = params
        # Compute optimal c
        c = y_mean - a * x2_mean - b * x_mean
        # Compute residuals
        residuals = y - (a * X2 + b * X + c)
        # Return MSE loss
        return np.sum(residuals**2) / n
    
    def gradient(params):
        a, b = params
        c = y_mean - a * x2_mean - b * x_mean
        residuals = y - (a * X2 + b * X + c)
        
        # Partial derivative with respect to a
        da = -2 * np.sum(residuals * (X2 - x2_mean)) / n
        # Partial derivative with respect to b
        db = -2 * np.sum(residuals * (X - x_mean)) / n
        
        return np.array([da, db])
    
    def get_params(params):
        a, b = params
        c = y_mean - a * x2_mean - b * x_mean
        return a, b, c
    
    return func, gradient, get_params


# =============================================================================
# LOGISTIC REGRESSION OBJECTIVE FUNCTION
# =============================================================================

def create_logistic_objective(X, y):
    """
    Create Logistic Regression objective (binary cross-entropy).
    
    Args:
        X: Input features (n x 2 numpy array)
        y: Binary labels (numpy array of 0s and 1s)
    
    Returns:
        func: Function that computes BCE loss given w1
        derivative: Function that computes gradient given w1
        get_params: Function that returns (w1, w2, b) given w1
    """
    n = len(y)
    x1 = X[:, 0]
    x2 = X[:, 1]
    
    x1_mean = np.mean(x1)
    x2_mean = np.mean(x2)
    
    var_x1 = np.var(x1)
    var_x2 = np.var(x2)
    var_ratio = var_x1 / var_x2 if var_x2 > 1e-10 else 1.0
    
    def sigmoid(z):
        # Clip to prevent overflow
        z = np.clip(z, -500, 500)
        # Sigmoid function
        return 1 / (1 + np.exp(-z))
    
    def func(w1):
        # Compute w2 and b from w1
        w2 = w1 * var_ratio
        b = -w1 * x1_mean - w2 * x2_mean
        
        # Compute z = w1*x1 + w2*x2 + b
        z = w1 * x1 + w2 * x2 + b
        
        # Compute predicted probabilities
        p = sigmoid(z)
        
        # Clip to avoid log(0)
        eps = 1e-15
        p = np.clip(p, eps, 1 - eps)
        
        # Binary cross-entropy loss
        loss = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
        return loss
    
    def derivative(w1):
        w2 = w1 * var_ratio
        b = -w1 * x1_mean - w2 * x2_mean
        
        z = w1 * x1 + w2 * x2 + b
        p = sigmoid(z)
        
        # Prediction error
        error = p - y
        
        # Derivative with respect to w1 (chain rule with w2 and b)
        dw1 = np.mean(error * (x1 - x1_mean + var_ratio * (x2 - x2_mean)))
        
        return dw1
    
    def get_params(w1):
        w2 = w1 * var_ratio
        b = -w1 * x1_mean - w2 * x2_mean
        return w1, w2, b
    
    return func, derivative, get_params
