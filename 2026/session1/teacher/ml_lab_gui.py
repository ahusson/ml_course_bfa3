"""
Machine Learning Lab GUI - Course 1: Linear Regression & Optimization
BFA3 Dauphine - M2 Level

This application provides visual demonstrations of:
- Panel 1: Function optimization (convex vs egg-box) using gradient descent
- Panel 2: Linear regression with OLS objective function visualization
"""

import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import random

from my_lib_teacher import gradient_descent


# =============================================================================
# FUNCTION GENERATORS
# =============================================================================


def generate_convex_function(domain):
    """
    Generate a random convex C² function on the given domain.

    Returns:
        Tuple (func, derivative, name) where func is the function,
        derivative is its derivative, and name is a string description
    """
    a, b = domain

    # Random minimum location within domain
    x_min = random.uniform(a + 0.2 * (b - a), b - 0.2 * (b - a))

    # Random coefficients for quadratic + optional higher even powers
    c2 = random.uniform(0.5, 2.0)  # Quadratic coefficient (positive for convexity)
    c4 = random.uniform(0, 0.1)  # Optional quartic term
    offset = random.uniform(-5, 5)

    def func(x):
        return c2 * (x - x_min) ** 2 + c4 * (x - x_min) ** 4 + offset

    def derivative(x):
        return 2 * c2 * (x - x_min) + 4 * c4 * (x - x_min) ** 3

    name = (
        f"f(x) = {c2:.2f}(x - {x_min:.2f})² + {c4:.2f}(x - {x_min:.2f})⁴ + {offset:.2f}"
    )

    return func, derivative, name


def generate_eggbox_function(domain):
    """
    Generate a random egg-box (non-convex) C² function on the given domain.
    Has multiple local minima, making it hard to find global minimum.

    Returns:
        Tuple (func, derivative, name)
    """
    a, b = domain

    # Quadratic trend + sinusoidal oscillations
    x_center = (a + b) / 2
    c_quad = random.uniform(0.1, 0.3)  # Weak quadratic trend
    amplitude = random.uniform(1.0, 3.0)  # Oscillation amplitude
    frequency = random.uniform(3, 6)  # Number of oscillations
    phase = random.uniform(0, 2 * np.pi)
    offset = random.uniform(-2, 2)

    def func(x):
        return (
            c_quad * (x - x_center) ** 2
            + amplitude * np.sin(frequency * x + phase)
            + offset
        )

    def derivative(x):
        return 2 * c_quad * (x - x_center) + amplitude * frequency * np.cos(
            frequency * x + phase
        )

    name = f"f(x) = {c_quad:.2f}(x - {x_center:.2f})² + {amplitude:.2f}sin({frequency:.1f}x + {phase:.2f})"

    return func, derivative, name


def generate_ols_objective(X, y):
    """
    Generate the OLS objective function for linear regression.

    For y = ax + b, the OLS objective is:
    L(a, b) = sum((y_i - (a*x_i + b))^2)

    We parameterize by theta = [a, b] and create a 1D slice for visualization.

    Args:
        X: Input features (1D array)
        y: Target values (1D array)

    Returns:
        Tuple (func, derivative, name, get_params) where get_params converts
        the 1D optimization variable back to (a, b)
    """
    n = len(X)

    # Compute optimal b as function of a: b*(a) = mean(y) - a*mean(x)
    x_mean = np.mean(X)
    y_mean = np.mean(y)

    # OLS as function of slope 'a' only (with optimal intercept)
    def func(a):
        b = y_mean - a * x_mean
        residuals = y - (a * X + b)
        return np.sum(residuals**2) / n

    def derivative(a):
        b = y_mean - a * x_mean
        residuals = y - (a * X + b)
        # d/da of sum((y - ax - b)^2) where b = y_mean - a*x_mean
        # = sum(2*(y - ax - b)*(-x - (-x_mean)))
        # = sum(2*residuals*(x_mean - x))
        return 2 * np.sum(residuals * (x_mean - X)) / n

    def get_params(a):
        """Convert slope a to (a, b) tuple"""
        b = y_mean - a * x_mean
        return a, b

    name = "OLS: L(a) = (1/n)Σ(yᵢ - (axᵢ + b))²"

    return func, derivative, name, get_params


def generate_dataset():
    """
    Generate a 2D dataset with a visible linear trend.

    Returns:
        Tuple (X, y) of numpy arrays
    """
    n_points = 50

    # Random trend parameters
    true_slope = random.uniform(0.5, 2.0) * random.choice([-1, 1])
    true_intercept = random.uniform(-3, 3)
    noise_level = random.uniform(0.5, 1.5)

    # Generate data
    X = np.linspace(-5, 5, n_points) + np.random.randn(n_points) * 0.5
    X = np.sort(X)  # Sort for nicer plotting
    y = true_slope * X + true_intercept + np.random.randn(n_points) * noise_level

    return X, y


# =============================================================================
# PANEL 1: FUNCTION OPTIMIZATION
# =============================================================================


class Panel1(ttk.Frame):
    """
    Panel for visualizing function optimization.

    Shows convex or egg-box functions and demonstrates gradient descent.
    """

    def __init__(self, parent):
        super().__init__(parent)

        self.domain = (-5, 5)  # Compact domain
        self.func = None
        self.derivative = None
        self.func_name = ""
        self.path = []
        self.current_step = 0

        self._create_widgets()

    def _create_widgets(self):
        """Create all widgets for Panel 1"""

        # Main layout: controls on top, plot + stats below
        controls_frame = ttk.Frame(self)
        controls_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

        content_frame = ttk.Frame(self)
        content_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)

        # === CONTROLS ===
        # Function type selector
        ttk.Label(controls_frame, text="Function Type:").pack(side=tk.LEFT, padx=(0, 5))
        self.func_type_var = tk.StringVar(value="Convex")
        func_type_combo = ttk.Combobox(
            controls_frame,
            textvariable=self.func_type_var,
            values=["Convex", "Egg-box"],
            state="readonly",
            width=10,
        )
        func_type_combo.pack(side=tk.LEFT, padx=(0, 20))

        # Alpha input
        ttk.Label(controls_frame, text="α (step size):").pack(side=tk.LEFT, padx=(0, 5))
        self.alpha_var = tk.StringVar(value="0.1")
        self.alpha_entry = ttk.Entry(
            controls_frame, textvariable=self.alpha_var, width=8
        )
        self.alpha_entry.pack(side=tk.LEFT, padx=(0, 20))

        # Run buttons
        self.run_all_btn = ttk.Button(
            controls_frame, text="Run All", command=self._run_all
        )
        self.run_all_btn.pack(side=tk.LEFT, padx=(0, 5))

        self.run_step_btn = ttk.Button(
            controls_frame, text="Run Step", command=self._run_step
        )
        self.run_step_btn.pack(side=tk.LEFT, padx=(0, 5))

        self.reset_btn = ttk.Button(controls_frame, text="Reset", command=self._reset)
        self.reset_btn.pack(side=tk.LEFT, padx=(0, 5))

        # === PLOT ===
        plot_frame = ttk.Frame(content_frame)
        plot_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.fig = Figure(figsize=(7, 5), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # === STATS ===
        stats_frame = ttk.LabelFrame(content_frame, text="Statistics", padding=10)
        stats_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))

        self.stats_labels = {}
        stats_items = [
            ("Function:", "func_name"),
            ("Iterations:", "iterations"),
            ("Current x:", "current_x"),
            ("Current f(x):", "current_fx"),
            ("Final x:", "final_x"),
            ("Final f(x):", "final_fx"),
            ("α:", "alpha"),
        ]

        for i, (label_text, key) in enumerate(stats_items):
            ttk.Label(
                stats_frame, text=label_text, font=("TkDefaultFont", 9, "bold")
            ).grid(row=i, column=0, sticky=tk.W, pady=2)
            self.stats_labels[key] = ttk.Label(stats_frame, text="-", width=20)
            self.stats_labels[key].grid(
                row=i, column=1, sticky=tk.W, pady=2, padx=(5, 0)
            )

        # Initial plot
        self._reset()

    def _generate_function(self):
        """Generate a new random function based on selection"""
        func_type = self.func_type_var.get()

        if func_type == "Convex":
            self.func, self.derivative, self.func_name = generate_convex_function(
                self.domain
            )
        else:
            self.func, self.derivative, self.func_name = generate_eggbox_function(
                self.domain
            )

        self.path = []
        self.current_step = 0

    def _plot_function(self):
        """Plot the current function"""
        self.ax.clear()

        if self.func is None:
            self.canvas.draw()
            return

        # Plot function
        x = np.linspace(self.domain[0], self.domain[1], 500)
        y = [self.func(xi) for xi in x]

        self.ax.plot(x, y, "b-", linewidth=2, label="f(x)")
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("f(x)")
        self.ax.set_title(f"Function: {self.func_type_var.get()}")
        self.ax.grid(True, alpha=0.3)
        self.ax.legend()

        self.canvas.draw()

    def _plot_path(self, up_to_step=None):
        """Plot the optimization path up to given step"""
        self._plot_function()

        if not self.path:
            return

        if up_to_step is None:
            up_to_step = len(self.path)

        path_to_show = self.path[:up_to_step]

        if path_to_show:
            # Plot path points
            x_vals = path_to_show
            y_vals = [self.func(x) for x in path_to_show]

            # Plot all points
            self.ax.scatter(x_vals, y_vals, c="red", s=50, zorder=5, label="Steps")

            # Connect with lines
            if len(x_vals) > 1:
                self.ax.plot(x_vals, y_vals, "r--", alpha=0.5, linewidth=1)

            # Highlight current point
            self.ax.scatter(
                [x_vals[-1]],
                [y_vals[-1]],
                c="green",
                s=100,
                zorder=6,
                marker="*",
                label="Current",
            )

            # Highlight start point
            self.ax.scatter(
                [x_vals[0]],
                [y_vals[0]],
                c="orange",
                s=80,
                zorder=5,
                marker="s",
                label="Start",
            )

        self.ax.legend()
        self.canvas.draw()

    def _update_stats(self, step=None):
        """Update statistics display"""
        self.stats_labels["func_name"].config(text=self.func_type_var.get())
        self.stats_labels["alpha"].config(text=self.alpha_var.get())

        if step is None:
            step = len(self.path)

        self.stats_labels["iterations"].config(text=str(step))

        if self.path and step > 0:
            current_x = self.path[step - 1]
            self.stats_labels["current_x"].config(text=f"{current_x:.6f}")
            self.stats_labels["current_fx"].config(text=f"{self.func(current_x):.6f}")

            final_x = self.path[-1]
            self.stats_labels["final_x"].config(text=f"{final_x:.6f}")
            self.stats_labels["final_fx"].config(text=f"{self.func(final_x):.6f}")
        else:
            for key in ["current_x", "current_fx", "final_x", "final_fx"]:
                self.stats_labels[key].config(text="-")

    def _reset(self):
        """Reset the panel"""
        self._generate_function()
        self._plot_function()
        self._update_stats(0)

    def _run_all(self):
        """Run the complete optimization"""
        self._generate_function()

        try:
            alpha = float(self.alpha_var.get())
        except ValueError:
            self.stats_labels["alpha"].config(text="Error: Invalid α")
            return

        self.path = gradient_descent(self.func, self.domain, self.derivative, alpha)

        self.current_step = len(self.path)
        self._plot_path()
        self._update_stats()

    def _run_step(self):
        """Run one step of gradient descent"""
        try:
            alpha = float(self.alpha_var.get())
        except ValueError:
            self.stats_labels["alpha"].config(text="Error: Invalid α")
            return

        # If no path yet, generate function and initialize
        if not self.path:
            self._generate_function()
            # Initialize with random starting point
            a, b = self.domain
            x0 = random.uniform(a + 0.1 * (b - a), b - 0.1 * (b - a))
            self.path = [x0]
            self.current_step = 1
        else:
            # Take one gradient step
            x = self.path[-1]
            grad = self.derivative(x)

            # Check convergence
            if abs(grad) < 1e-6:
                self._plot_path()
                self._update_stats()
                return

            x_new = x - alpha * grad

            # Project onto domain
            a, b = self.domain
            x_new = max(a, min(b, x_new))

            self.path.append(x_new)
            self.current_step = len(self.path)

        self._plot_path()
        self._update_stats()


# =============================================================================
# PANEL 2: LINEAR REGRESSION
# =============================================================================


class Panel2(ttk.Frame):
    """
    Panel for visualizing linear regression.

    Shows OLS objective function and dataset with regression lines.
    """

    def __init__(self, parent):
        super().__init__(parent)

        self.domain = (-3, 3)  # Domain for slope parameter
        self.X = None
        self.y = None
        self.func = None
        self.derivative = None
        self.get_params = None
        self.func_name = ""
        self.path = []
        self.current_step = 0

        self._create_widgets()

    def _create_widgets(self):
        """Create all widgets for Panel 2"""

        # Main layout
        controls_frame = ttk.Frame(self)
        controls_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

        content_frame = ttk.Frame(self)
        content_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)

        # === CONTROLS ===
        # Generate dataset button
        self.generate_btn = ttk.Button(
            controls_frame, text="Generate Dataset", command=self._generate_dataset
        )
        self.generate_btn.pack(side=tk.LEFT, padx=(0, 20))

        # Alpha input
        ttk.Label(controls_frame, text="α (step size):").pack(side=tk.LEFT, padx=(0, 5))
        self.alpha_var = tk.StringVar(value="0.1")
        self.alpha_entry = ttk.Entry(
            controls_frame, textvariable=self.alpha_var, width=8
        )
        self.alpha_entry.pack(side=tk.LEFT, padx=(0, 20))

        # Run buttons
        self.run_all_btn = ttk.Button(
            controls_frame, text="Run All", command=self._run_all
        )
        self.run_all_btn.pack(side=tk.LEFT, padx=(0, 5))

        self.run_step_btn = ttk.Button(
            controls_frame, text="Run Step", command=self._run_step
        )
        self.run_step_btn.pack(side=tk.LEFT, padx=(0, 5))

        self.reset_btn = ttk.Button(controls_frame, text="Reset", command=self._reset)
        self.reset_btn.pack(side=tk.LEFT, padx=(0, 5))

        # === PLOTS ===
        plots_frame = ttk.Frame(content_frame)
        plots_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.fig = Figure(figsize=(7, 8), dpi=100)
        self.ax1 = self.fig.add_subplot(211)  # OLS objective
        self.ax2 = self.fig.add_subplot(212)  # Dataset + regression lines
        self.fig.tight_layout(pad=3.0)

        self.canvas = FigureCanvasTkAgg(self.fig, master=plots_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # === STATS ===
        stats_frame = ttk.LabelFrame(content_frame, text="Statistics", padding=10)
        stats_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))

        self.stats_labels = {}
        stats_items = [
            ("Iterations:", "iterations"),
            ("Current slope (a):", "current_a"),
            ("Current intercept (b):", "current_b"),
            ("Current Loss:", "current_loss"),
            ("Final slope (a):", "final_a"),
            ("Final intercept (b):", "final_b"),
            ("Final Loss:", "final_loss"),
            ("α:", "alpha"),
        ]

        for i, (label_text, key) in enumerate(stats_items):
            ttk.Label(
                stats_frame, text=label_text, font=("TkDefaultFont", 9, "bold")
            ).grid(row=i, column=0, sticky=tk.W, pady=2)
            self.stats_labels[key] = ttk.Label(stats_frame, text="-", width=20)
            self.stats_labels[key].grid(
                row=i, column=1, sticky=tk.W, pady=2, padx=(5, 0)
            )

        # Initial state
        self._reset()

    def _generate_dataset(self):
        """Generate new dataset and OLS objective"""
        self.X, self.y = generate_dataset()
        self.func, self.derivative, self.func_name, self.get_params = (
            generate_ols_objective(self.X, self.y)
        )
        self.path = []
        self.current_step = 0

        self._plot_all()
        self._update_stats(0)

    def _plot_all(self, up_to_step=None):
        """Plot OLS objective and dataset"""
        self.ax1.clear()
        self.ax2.clear()

        if self.X is None or self.func is None:
            self.canvas.draw()
            return

        # === Plot 1: OLS Objective ===
        a_vals = np.linspace(self.domain[0], self.domain[1], 500)
        loss_vals = [self.func(a) for a in a_vals]

        self.ax1.plot(a_vals, loss_vals, "b-", linewidth=2, label="OLS Loss")
        self.ax1.set_xlabel("Slope (a)")
        self.ax1.set_ylabel("Loss L(a)")
        self.ax1.set_title("OLS Objective Function")
        self.ax1.grid(True, alpha=0.3)

        # Plot path on objective function
        if self.path:
            if up_to_step is None:
                up_to_step = len(self.path)

            path_to_show = self.path[:up_to_step]

            if path_to_show:
                a_path = path_to_show
                loss_path = [self.func(a) for a in a_path]

                self.ax1.scatter(
                    a_path, loss_path, c="red", s=50, zorder=5, label="Steps"
                )
                if len(a_path) > 1:
                    self.ax1.plot(a_path, loss_path, "r--", alpha=0.5, linewidth=1)

                self.ax1.scatter(
                    [a_path[-1]],
                    [loss_path[-1]],
                    c="green",
                    s=100,
                    zorder=6,
                    marker="*",
                    label="Current",
                )
                self.ax1.scatter(
                    [a_path[0]],
                    [loss_path[0]],
                    c="orange",
                    s=80,
                    zorder=5,
                    marker="s",
                    label="Start",
                )

        self.ax1.legend()

        # === Plot 2: Dataset and Regression Lines ===
        self.ax2.scatter(self.X, self.y, c="blue", alpha=0.6, label="Data")
        self.ax2.set_xlabel("X")
        self.ax2.set_ylabel("y")
        self.ax2.set_title("Dataset and Regression Lines")
        self.ax2.grid(True, alpha=0.3)

        # Plot regression lines for path
        if self.path and self.get_params:
            if up_to_step is None:
                up_to_step = len(self.path)

            path_to_show = self.path[:up_to_step]
            x_line = np.array([self.X.min(), self.X.max()])

            # Plot intermediate lines with fading alpha
            n_lines = len(path_to_show)
            for i, a in enumerate(path_to_show[:-1] if n_lines > 1 else []):
                slope, intercept = self.get_params(a)
                y_line = slope * x_line + intercept
                alpha = 0.2 + 0.3 * (i / max(n_lines - 1, 1))
                self.ax2.plot(x_line, y_line, "r-", alpha=alpha, linewidth=1)

            # Plot current/final line prominently
            if path_to_show:
                slope, intercept = self.get_params(path_to_show[-1])
                y_line = slope * x_line + intercept
                self.ax2.plot(
                    x_line,
                    y_line,
                    "g-",
                    linewidth=2,
                    label=f"Current: y = {slope:.2f}x + {intercept:.2f}",
                )

        self.ax2.legend()

        self.fig.tight_layout(pad=3.0)
        self.canvas.draw()

    def _update_stats(self, step=None):
        """Update statistics display"""
        self.stats_labels["alpha"].config(text=self.alpha_var.get())

        if step is None:
            step = len(self.path)

        self.stats_labels["iterations"].config(text=str(step))

        if self.path and step > 0 and self.get_params:
            current_a = self.path[step - 1]
            current_slope, current_intercept = self.get_params(current_a)
            self.stats_labels["current_a"].config(text=f"{current_slope:.6f}")
            self.stats_labels["current_b"].config(text=f"{current_intercept:.6f}")
            self.stats_labels["current_loss"].config(text=f"{self.func(current_a):.6f}")

            final_a = self.path[-1]
            final_slope, final_intercept = self.get_params(final_a)
            self.stats_labels["final_a"].config(text=f"{final_slope:.6f}")
            self.stats_labels["final_b"].config(text=f"{final_intercept:.6f}")
            self.stats_labels["final_loss"].config(text=f"{self.func(final_a):.6f}")
        else:
            for key in [
                "current_a",
                "current_b",
                "current_loss",
                "final_a",
                "final_b",
                "final_loss",
            ]:
                self.stats_labels[key].config(text="-")

    def _reset(self):
        """Reset the panel"""
        self.X = None
        self.y = None
        self.func = None
        self.derivative = None
        self.get_params = None
        self.path = []
        self.current_step = 0

        self.ax1.clear()
        self.ax2.clear()
        self.ax1.set_title("OLS Objective Function (Generate dataset first)")
        self.ax2.set_title("Dataset (Generate dataset first)")
        self.canvas.draw()

        self._update_stats(0)

    def _run_all(self):
        """Run the complete optimization"""
        if self.func is None:
            self._generate_dataset()

        try:
            alpha = float(self.alpha_var.get())
        except ValueError:
            self.stats_labels["alpha"].config(text="Error: Invalid α")
            return

        self.path = gradient_descent(self.func, self.domain, self.derivative, alpha)

        self.current_step = len(self.path)
        self._plot_all()
        self._update_stats()

    def _run_step(self):
        """Run one step of gradient descent"""
        if self.func is None:
            self._generate_dataset()
            return

        try:
            alpha = float(self.alpha_var.get())
        except ValueError:
            self.stats_labels["alpha"].config(text="Error: Invalid α")
            return

        # If no path yet, initialize
        if not self.path:
            a, b = self.domain
            a0 = random.uniform(a + 0.1 * (b - a), b - 0.1 * (b - a))
            self.path = [a0]
            self.current_step = 1
        else:
            # Take one gradient step
            a = self.path[-1]
            grad = self.derivative(a)

            # Check convergence
            if abs(grad) < 1e-6:
                self._plot_all()
                self._update_stats()
                return

            a_new = a - alpha * grad

            # Project onto domain
            dom_min, dom_max = self.domain
            a_new = max(dom_min, min(dom_max, a_new))

            self.path.append(a_new)
            self.current_step = len(self.path)

        self._plot_all()
        self._update_stats()


# =============================================================================
# MAIN APPLICATION
# =============================================================================


class MLLabApp(tk.Tk):
    """Main application window with tabbed panels"""

    def __init__(self):
        super().__init__()

        self.title("ML Lab - Linear Regression & Optimization")
        self.geometry("1000x700")

        # Create notebook (tabbed interface)
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Create panels
        self.panel1 = Panel1(self.notebook)
        self.panel2 = Panel2(self.notebook)

        # Add panels to notebook
        self.notebook.add(self.panel1, text="Panel 1: Function Optimization")
        self.notebook.add(self.panel2, text="Panel 2: Linear Regression")


def main():
    """Main entry point"""
    app = MLLabApp()
    app.mainloop()


if __name__ == "__main__":
    main()
