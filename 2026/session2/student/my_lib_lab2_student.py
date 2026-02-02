"""
my_lib_lab2.py - Machine Learning Lab 2 Library (STUDENT VERSION)
BFA3 Dauphine - M2 Level

This library contains:
- Gradient descent implementations (1D and 2D) - PROVIDED
- Quadratic OLS objective function - TODO: Complete the implementation
- Logistic Regression objective function - TODO: Complete the implementation

Instructions:
1. Find all TODO comments
2. Replace the blanks (_______________) with the correct code
3. Test your implementation using the GUI
"""

import numpy as np
import random


# =============================================================================
# GRADIENT DESCENT FUNCTIONS (PROVIDED - DO NOT MODIFY)
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
# QUADRATIC OLS OBJECTIVE FUNCTION - TODO: COMPLETE THIS
# =============================================================================

def create_quadratic_ols_objective(X, y):
    """
    Create Quadratic OLS objective: L(a, b) with c computed optimally.
    Model: y = a*x^2 + b*x + c
    
    The objective function is the Mean Squared Error (MSE):
    L(a, b) = (1/n) * sum((y_i - (a*x_i^2 + b*x_i + c))^2)
    
    where c is computed optimally as: c = y_mean - a*x2_mean - b*x_mean
    
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
        
        # ============================================================
        # TODO 1: Compute optimal c
        # ============================================================
        # Formula: c = y_mean - a * x2_mean - b * x_mean
        c = _______________  # YOUR CODE HERE
        
        # ============================================================
        # TODO 2: Compute residuals
        # ============================================================
        # Residual for each point: r_i = y_i - (a * x_i^2 + b * x_i + c)
        # Use vectorized operations: y - (a * X2 + b * X + c)
        residuals = _______________  # YOUR CODE HERE
        
        # ============================================================
        # TODO 3: Compute and return MSE loss
        # ============================================================
        # MSE = (1/n) * sum(residuals^2)
        return _______________  # YOUR CODE HERE
    
    def gradient(params):
        a, b = params
        c = y_mean - a * x2_mean - b * x_mean
        residuals = y - (a * X2 + b * X + c)
        
        # ============================================================
        # TODO 4: Compute partial derivative with respect to a
        # ============================================================
        # Formula: da = -2/n * sum(residuals * (X2 - x2_mean))
        da = _______________  # YOUR CODE HERE
        
        # ============================================================
        # TODO 5: Compute partial derivative with respect to b
        # ============================================================
        # Formula: db = -2/n * sum(residuals * (X - x_mean))
        db = _______________  # YOUR CODE HERE
        
        return np.array([da, db])
    
    def get_params(params):
        a, b = params
        c = y_mean - a * x2_mean - b * x_mean
        return a, b, c
    
    return func, gradient, get_params


# =============================================================================
# LOGISTIC REGRESSION OBJECTIVE FUNCTION - TODO: COMPLETE THIS
# =============================================================================

def create_logistic_objective(X, y):
    """
    Create Logistic Regression objective (binary cross-entropy).
    
    The model predicts: P(y=1|x) = sigmoid(w1*x1 + w2*x2 + b)
    
    The objective function is the Binary Cross-Entropy (BCE):
    L = -(1/n) * sum(y_i * log(p_i) + (1 - y_i) * log(1 - p_i))
    
    where p_i = sigmoid(w1*x1_i + w2*x2_i + b)
    
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
        # Clip z to prevent overflow in exp
        z = np.clip(z, -500, 500)
        
        # ============================================================
        # TODO 1: Implement the sigmoid function
        # ============================================================
        # Formula: sigma(z) = 1 / (1 + exp(-z))
        return _______________  # YOUR CODE HERE
    
    def func(w1):
        # Compute w2 and b from w1 (these are derived parameters)
        w2 = w1 * var_ratio
        b = -w1 * x1_mean - w2 * x2_mean
        
        # ============================================================
        # TODO 2: Compute z = w1*x1 + w2*x2 + b
        # ============================================================
        z = _______________  # YOUR CODE HERE
        
        # ============================================================
        # TODO 3: Compute predicted probabilities using sigmoid
        # ============================================================
        p = _______________  # YOUR CODE HERE
        
        # Clip probabilities to avoid log(0)
        eps = 1e-15
        p = np.clip(p, eps, 1 - eps)
        
        # ============================================================
        # TODO 4: Compute and return binary cross-entropy loss
        # ============================================================
        # BCE = -(1/n) * sum(y * log(p) + (1-y) * log(1-p))
        # Use np.mean instead of sum/n
        loss = _______________  # YOUR CODE HERE
        return loss
    
    def derivative(w1):
        w2 = w1 * var_ratio
        b = -w1 * x1_mean - w2 * x2_mean
        
        z = w1 * x1 + w2 * x2 + b
        p = sigmoid(z)
        
        # ============================================================
        # TODO 5: Compute prediction error (p - y)
        # ============================================================
        error = _______________  # YOUR CODE HERE
        
        # Derivative with respect to w1 (chain rule accounting for w2 and b)
        dw1 = np.mean(error * (x1 - x1_mean + var_ratio * (x2 - x2_mean)))
        
        return dw1
    
    def get_params(w1):
        w2 = w1 * var_ratio
        b = -w1 * x1_mean - w2 * x2_mean
        return w1, w2, b
    
    return func, derivative, get_params


# =============================================================================
# TEST YOUR IMPLEMENTATION
# =============================================================================
if __name__ == "__main__":
    print("Testing your implementations...")
    print("=" * 60)
    
    # Test 1: Quadratic OLS
    print("\n[Test 1] Quadratic OLS Objective Function")
    print("-" * 40)
    
    # Create simple quadratic data: y = 0.5*x^2 - x + 2 + noise
    np.random.seed(42)
    X_test = np.linspace(-3, 3, 50)
    y_test = 0.5 * X_test**2 - X_test + 2 + np.random.randn(50) * 0.5
    
    try:
        func_q, grad_q, get_params_q = create_quadratic_ols_objective(X_test, y_test)
        
        # Test at true parameters
        loss_at_true = func_q(np.array([0.5, -1.0]))
        print(f"  Loss at true params [0.5, -1.0]: {loss_at_true:.4f}")
        
        # Test gradient
        grad_at_true = grad_q(np.array([0.5, -1.0]))
        print(f"  Gradient at true params: [{grad_at_true[0]:.4f}, {grad_at_true[1]:.4f}]")
        print(f"  (Gradient should be close to [0, 0] at optimal)")
        
        # Test get_params
        a, b, c = get_params_q(np.array([0.5, -1.0]))
        print(f"  Recovered params: a={a:.2f}, b={b:.2f}, c={c:.2f}")
        
        print("  ✓ Quadratic OLS test completed!")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        print("  Make sure you completed all TODOs in create_quadratic_ols_objective")
    
    # Test 2: Logistic Regression
    print("\n[Test 2] Logistic Regression Objective Function")
    print("-" * 40)
    
    # Create simple classification data
    np.random.seed(42)
    cluster0 = np.random.randn(25, 2) + np.array([-2, -2])
    cluster1 = np.random.randn(25, 2) + np.array([2, 2])
    X_test_log = np.vstack([cluster0, cluster1])
    y_test_log = np.array([0]*25 + [1]*25)
    
    try:
        func_l, deriv_l, get_params_l = create_logistic_objective(X_test_log, y_test_log)
        
        # Test at some parameters
        loss_at_w1 = func_l(1.0)
        print(f"  Loss at w1=1.0: {loss_at_w1:.4f}")
        
        # Test derivative
        deriv_at_w1 = deriv_l(1.0)
        print(f"  Derivative at w1=1.0: {deriv_at_w1:.4f}")
        
        # Test get_params
        w1, w2, b = get_params_l(1.0)
        print(f"  Params at w1=1.0: w1={w1:.2f}, w2={w2:.2f}, b={b:.2f}")
        
        print("  ✓ Logistic Regression test completed!")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        print("  Make sure you completed all TODOs in create_logistic_objective")
    
    print("\n" + "=" * 60)
    print("If both tests passed, run: python ml_lab2_gui.py")
