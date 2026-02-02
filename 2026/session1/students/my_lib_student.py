"""
my_lib.py - Machine Learning Lab Library (STUDENT VERSION)
BFA3 Dauphine - M2 Level

This library contains core algorithm implementations for the ML Lab.
Complete the TODOs below to implement the gradient descent algorithm.
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
        # ================================================================
        # TODO 1: Compute the gradient at current position
        # ================================================================
        # Hint: Use the 'derivative' function passed as argument
        # Mathematical notation: g = f'(x_k)
        grad = None  # YOUR CODE HERE - Replace None with the correct expression
        
        # ================================================================
        # TODO 2: Check for convergence
        # ================================================================
        # Hint: If the absolute value of the gradient is smaller than 
        #       epsilon, we have converged. Use the 'break' statement.
        # Mathematical condition: |g| < epsilon
        if False:  # YOUR CODE HERE - Replace False with the correct condition
            break
        
        # ================================================================
        # TODO 3: Apply the gradient descent update rule
        # ================================================================
        # Hint: Move in the opposite direction of the gradient
        # Mathematical formula: x_{k+1} = x_k - alpha * g
        x_new = None  # YOUR CODE HERE - Replace None with the correct expression
        
        # Project back onto domain if we stepped outside
        x_new = max(a, min(b, x_new))
        
        # Add new point to path
        path.append(x_new)
        
        # Check if we're stuck (not moving anymore)
        if abs(x_new - x) < 1e-10:
            break
            
        # Update current position for next iteration
        x = x_new
    
    return path


# ============================================================================
# TEST YOUR IMPLEMENTATION
# ============================================================================
if __name__ == "__main__":
    print("Testing gradient_descent implementation...")
    print("=" * 50)
    
    # Test 1: Simple quadratic function
    def f(x):
        return (x - 2)**2
    
    def df(x):
        return 2 * (x - 2)
    
    print("\nTest 1: f(x) = (x - 2)^2")
    print("Expected minimum at x = 2.0")
    
    try:
        path = gradient_descent(f, (-5, 5), df, alpha=0.1)
        print(f"  Number of iterations: {len(path)}")
        print(f"  Found minimum at x = {path[-1]:.6f}")
        print(f"  f(x*) = {f(path[-1]):.6f}")
        
        if abs(path[-1] - 2.0) < 0.01:
            print("  ✓ Test 1 PASSED!")
        else:
            print("  ✗ Test 1 FAILED - minimum not found correctly")
    except Exception as e:
        print(f"  ✗ Test 1 FAILED with error: {e}")
    
    # Test 2: Different quadratic
    def g(x):
        return 0.5 * (x + 1)**2 + 3
    
    def dg(x):
        return (x + 1)
    
    print("\nTest 2: g(x) = 0.5(x + 1)^2 + 3")
    print("Expected minimum at x = -1.0")
    
    try:
        path = gradient_descent(g, (-5, 5), dg, alpha=0.1)
        print(f"  Number of iterations: {len(path)}")
        print(f"  Found minimum at x = {path[-1]:.6f}")
        print(f"  g(x*) = {g(path[-1]):.6f}")
        
        if abs(path[-1] - (-1.0)) < 0.01:
            print("  ✓ Test 2 PASSED!")
        else:
            print("  ✗ Test 2 FAILED - minimum not found correctly")
    except Exception as e:
        print(f"  ✗ Test 2 FAILED with error: {e}")
    
    print("\n" + "=" * 50)
    print("If both tests passed, your implementation is correct!")
    print("You can now run: python ml_lab_gui.py")
