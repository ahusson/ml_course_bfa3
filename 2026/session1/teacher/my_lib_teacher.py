"""
my_lib.py - Machine Learning Lab Library
BFA3 Dauphine - M2 Level

This library contains core algorithm implementations for the ML Lab.
Students will implement the functions in this file.
"""

import random


def gradient_descent(func, domain, derivative, alpha, max_iterations=1000, epsilon=1e-6):
    """
    Find the minimum of a function using gradient descent.
    
    The gradient descent algorithm iteratively updates the current position
    by moving in the opposite direction of the gradient (steepest descent).
    
    Update rule: x_new = x - alpha * gradient(x)
    
    Args:
        func: The function to minimize (callable)
        domain: Tuple (a, b) representing the compact domain [a, b]
        derivative: The derivative function of func (callable)
        alpha: Step size (learning rate) - controls how big each step is
        max_iterations: Maximum number of iterations (default: 1000)
        epsilon: Convergence threshold - stop if |gradient| < epsilon (default: 1e-6)
    
    Returns:
        List of x values [x0, x1, ..., x_final] showing the path to minimum.
        The last element is the found minimum.
    
    Example:
        >>> def f(x): return (x - 2)**2  # Minimum at x=2
        >>> def df(x): return 2*(x - 2)   # Derivative
        >>> path = gradient_descent(f, (-5, 5), df, alpha=0.1)
        >>> print(f"Found minimum at x = {path[-1]:.4f}")
        Found minimum at x = 2.0000
    """
    a, b = domain
    
    # Initialize at a random point in the domain (not too close to boundaries)
    x = random.uniform(a + 0.1 * (b - a), b - 0.1 * (b - a))
    
    # Store the path of all visited points
    path = [x]
    
    for i in range(max_iterations):
        # Compute the gradient at current position
        grad = derivative(x)
        
        # Check convergence: if gradient is very small, we're at a minimum
        if abs(grad) < epsilon:
            break
        
        # Gradient descent update: move opposite to gradient direction
        x_new = x - alpha * grad
        
        # Project back onto domain if we stepped outside
        x_new = max(a, min(b, x_new))
        
        # Add new point to path
        path.append(x_new)
        
        # Check if we're stuck (not moving anymore)
        if abs(x_new - x) < 1e-10:
            break
            
        # Update current position
        x = x_new
    
    return path
