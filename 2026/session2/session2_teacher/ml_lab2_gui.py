"""
Machine Learning Lab 2 GUI - Regression & Classification
BFA3 Dauphine - M2 Level

This application provides visual demonstrations of:
- Different dataset types (white noise, linear, quadratic, two clusters)
- Different fitting models (Linear OLS, Quadratic OLS, Logistic Regression)
- Gradient descent optimization with visualization
"""

import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
import random

from my_lib_lab2_teacher import (
    gradient_descent_1d,
    gradient_descent_2d,
    create_quadratic_ols_objective,
    create_logistic_objective,
)


# =============================================================================
# DATASET GENERATORS
# =============================================================================


def generate_white_noise(n_points=100):
    """
    Generate white noise dataset (no underlying pattern).
    """
    X = np.random.uniform(-5, 5, n_points)
    y = np.random.uniform(-5, 5, n_points)
    return X, y


def generate_linear_dataset(n_points=100):
    """
    Generate a 2D dataset with a visible linear trend.
    """
    true_slope = random.uniform(0.5, 2.0) * random.choice([-1, 1])
    true_intercept = random.uniform(-2, 2)
    noise_level = random.uniform(0.5, 1.5)

    X = np.linspace(-5, 5, n_points) + np.random.randn(n_points) * 0.3
    X = np.sort(X)
    y = true_slope * X + true_intercept + np.random.randn(n_points) * noise_level

    return X, y


def generate_quadratic_dataset(n_points=100):
    """
    Generate a 2D dataset with a quadratic trend (polynomial degree 2).
    """
    true_a = random.uniform(0.1, 0.5) * random.choice([-1, 1])
    true_b = random.uniform(-1, 1)
    true_c = random.uniform(-2, 2)
    noise_level = random.uniform(0.5, 1.5)

    X = np.linspace(-5, 5, n_points) + np.random.randn(n_points) * 0.2
    X = np.sort(X)
    y = true_a * X**2 + true_b * X + true_c + np.random.randn(n_points) * noise_level

    return X, y


def generate_two_clusters(n_points=100):
    """
    Generate two separated clusters sampled from two Gaussians with different means.
    Returns (X, y) where X is (n_points, 2) array of (x1, x2) and y is binary labels.
    """
    n_per_cluster = n_points // 2

    # Cluster 0
    mean1 = [random.uniform(-3, -1), random.uniform(-3, -1)]
    cov1 = [[random.uniform(0.3, 0.8), 0], [0, random.uniform(0.3, 0.8)]]
    cluster1 = np.random.multivariate_normal(mean1, cov1, n_per_cluster)

    # Cluster 1
    mean2 = [random.uniform(1, 3), random.uniform(1, 3)]
    cov2 = [[random.uniform(0.3, 0.8), 0], [0, random.uniform(0.3, 0.8)]]
    cluster2 = np.random.multivariate_normal(mean2, cov2, n_per_cluster)

    X = np.vstack([cluster1, cluster2])
    y = np.array([0] * n_per_cluster + [1] * n_per_cluster)

    # Shuffle
    indices = np.random.permutation(len(y))
    X = X[indices]
    y = y[indices]

    return X, y


# =============================================================================
# LINEAR OLS OBJECTIVE (kept in GUI as reference - not a student task)
# =============================================================================


def create_linear_ols_objective(X, y):
    """
    Create Linear OLS objective: L(a) with b computed optimally.
    """
    n = len(X)
    x_mean = np.mean(X)
    y_mean = np.mean(y)

    def func(a):
        b = y_mean - a * x_mean
        residuals = y - (a * X + b)
        return np.sum(residuals**2) / n

    def derivative(a):
        b = y_mean - a * x_mean
        residuals = y - (a * X + b)
        return 2 * np.sum(residuals * (x_mean - X)) / n

    def get_params(a):
        b = y_mean - a * x_mean
        return a, b

    return func, derivative, get_params


# =============================================================================
# MAIN APPLICATION
# =============================================================================


class Lab2App(tk.Tk):
    """Main application window for Lab 2"""

    def __init__(self):
        super().__init__()

        self.title("ML Lab 2 - Regression & Classification")
        self.geometry("1100x750")

        # Data and model state
        self.X = None
        self.y = None
        self.dataset_type = None
        self.func = None
        self.derivative = None
        self.gradient = None
        self.get_params = None
        self.path = []
        self.current_step = 0
        self.is_2d_optimization = False

        # Domains
        self.domain_1d = (-5, 5)
        self.domain_2d = ((-1, 1), (-3, 3))

        self._create_widgets()

    def _create_widgets(self):
        """Create all widgets"""

        # === CONTROLS FRAME ===
        controls_frame = ttk.Frame(self)
        controls_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

        # Data generation controls
        data_frame = ttk.LabelFrame(controls_frame, text="Data Generation", padding=5)
        data_frame.pack(side=tk.LEFT, padx=(0, 20))

        ttk.Label(data_frame, text="Dataset Type:").pack(side=tk.LEFT, padx=(0, 5))
        self.dataset_type_var = tk.StringVar(value="Linear")
        dataset_combo = ttk.Combobox(
            data_frame,
            textvariable=self.dataset_type_var,
            values=["White Noise", "Linear", "Quadratic", "Two Clusters"],
            state="readonly",
            width=12,
        )
        dataset_combo.pack(side=tk.LEFT, padx=(0, 10))

        generate_btn = ttk.Button(
            data_frame, text="Generate Dataset", command=self._generate_dataset
        )
        generate_btn.pack(side=tk.LEFT)

        # Model fitting controls
        model_frame = ttk.LabelFrame(controls_frame, text="Model Fitting", padding=5)
        model_frame.pack(side=tk.LEFT, padx=(0, 20))

        ttk.Label(model_frame, text="Model:").pack(side=tk.LEFT, padx=(0, 5))
        self.model_type_var = tk.StringVar(value="Linear OLS")
        model_combo = ttk.Combobox(
            model_frame,
            textvariable=self.model_type_var,
            values=["Linear OLS", "Quadratic OLS", "Logistic Regression"],
            state="readonly",
            width=18,
        )
        model_combo.pack(side=tk.LEFT, padx=(0, 10))

        ttk.Label(model_frame, text="α:").pack(side=tk.LEFT, padx=(0, 5))
        self.alpha_var = tk.StringVar(value="0.1")
        alpha_entry = ttk.Entry(model_frame, textvariable=self.alpha_var, width=8)
        alpha_entry.pack(side=tk.LEFT, padx=(0, 10))

        run_step_btn = ttk.Button(model_frame, text="Run Step", command=self._run_step)
        run_step_btn.pack(side=tk.LEFT, padx=(0, 5))

        run_all_btn = ttk.Button(model_frame, text="Run All", command=self._run_all)
        run_all_btn.pack(side=tk.LEFT, padx=(0, 5))

        reset_btn = ttk.Button(model_frame, text="Reset", command=self._reset)
        reset_btn.pack(side=tk.LEFT)

        # === CONTENT FRAME ===
        content_frame = ttk.Frame(self)
        content_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)

        # === PLOTS ===
        plots_frame = ttk.Frame(content_frame)
        plots_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.fig = Figure(figsize=(8, 8), dpi=100)
        self.ax1 = self.fig.add_subplot(211)
        self.ax2 = self.fig.add_subplot(212)
        self.fig.tight_layout(pad=3.0)

        self.canvas = FigureCanvasTkAgg(self.fig, master=plots_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Initial plot titles
        self.ax1.set_title("Objective Function (Generate dataset and fit model)")
        self.ax2.set_title("Dataset (Click 'Generate Dataset')")

        # === STATS ===
        stats_frame = ttk.LabelFrame(content_frame, text="Statistics", padding=10)
        stats_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))

        self.stats_labels = {}
        stats_items = [
            ("Dataset:", "dataset"),
            ("Model:", "model"),
            ("Iterations:", "iterations"),
            ("Current param 1:", "current_p1"),
            ("Current param 2:", "current_p2"),
            ("Current param 3:", "current_p3"),
            ("Current Loss:", "current_loss"),
            ("Final param 1:", "final_p1"),
            ("Final param 2:", "final_p2"),
            ("Final param 3:", "final_p3"),
            ("Final Loss:", "final_loss"),
            ("α:", "alpha"),
        ]

        for i, (label_text, key) in enumerate(stats_items):
            ttk.Label(
                stats_frame, text=label_text, font=("TkDefaultFont", 9, "bold")
            ).grid(row=i, column=0, sticky=tk.W, pady=2)
            self.stats_labels[key] = ttk.Label(stats_frame, text="-", width=18)
            self.stats_labels[key].grid(
                row=i, column=1, sticky=tk.W, pady=2, padx=(5, 0)
            )

    def _generate_dataset(self):
        """Generate dataset based on selection"""
        dataset_type = self.dataset_type_var.get()
        self.dataset_type = dataset_type

        if dataset_type == "White Noise":
            self.X, self.y = generate_white_noise()
        elif dataset_type == "Linear":
            self.X, self.y = generate_linear_dataset()
        elif dataset_type == "Quadratic":
            self.X, self.y = generate_quadratic_dataset()
        elif dataset_type == "Two Clusters":
            self.X, self.y = generate_two_clusters()

        # Reset fitting state
        self.path = []
        self.current_step = 0
        self.func = None
        self.derivative = None
        self.gradient = None
        self.get_params = None

        self._plot_dataset()
        self.ax1.clear()
        self.ax1.set_title("Objective Function (Select model and click Run)")
        self._update_stats()
        self.canvas.draw()

    def _plot_dataset(self):
        """Plot the current dataset"""
        if self.X is None:
            return

        self.ax2.clear()

        if self.dataset_type == "Two Clusters":
            colors = ["blue" if label == 0 else "red" for label in self.y]
            self.ax2.scatter(
                self.X[:, 0], self.X[:, 1], c=colors, alpha=0.6, label="Data"
            )
            self.ax2.set_xlabel("x₁")
            self.ax2.set_ylabel("x₂")
        else:
            self.ax2.scatter(self.X, self.y, c="blue", alpha=0.6, label="Data")
            self.ax2.set_xlabel("X")
            self.ax2.set_ylabel("y")

        self.ax2.set_title(f"Dataset: {self.dataset_type}")
        self.ax2.grid(True, alpha=0.3)
        self.ax2.legend()

    def _get_regression_data(self):
        """Get X, y data appropriate for regression models"""
        if self.dataset_type == "Two Clusters":
            return self.X[:, 0], self.X[:, 1]
        else:
            return self.X, self.y

    def _get_classification_data(self):
        """Get X, y data appropriate for classification"""
        if self.dataset_type == "Two Clusters":
            return self.X, self.y
        else:
            X_2d = np.column_stack([self.X, self.y])
            y_binary = (self.y > np.median(self.y)).astype(int)
            return X_2d, y_binary

    def _setup_objective_function(self):
        """Setup objective function based on model selection"""
        model_type = self.model_type_var.get()

        if model_type == "Linear OLS":
            X_reg, y_reg = self._get_regression_data()
            self.func, self.derivative, self.get_params = create_linear_ols_objective(
                X_reg, y_reg
            )
            self.is_2d_optimization = False

        elif model_type == "Quadratic OLS":
            X_reg, y_reg = self._get_regression_data()
            self.func, self.gradient, self.get_params = create_quadratic_ols_objective(
                X_reg, y_reg
            )
            self.is_2d_optimization = True

        elif model_type == "Logistic Regression":
            X_cls, y_cls = self._get_classification_data()
            self.func, self.derivative, self.get_params = create_logistic_objective(
                X_cls, y_cls
            )
            self.is_2d_optimization = False

    def _setup_3d_plot(self):
        """Setup figure with 3D plot for top subplot"""
        self.fig.clear()
        self.ax1 = self.fig.add_subplot(211, projection="3d")
        self.ax2 = self.fig.add_subplot(212)
        self.fig.tight_layout(pad=3.0)

    def _setup_2d_plot(self):
        """Setup figure with 2D plots"""
        self.fig.clear()
        self.ax1 = self.fig.add_subplot(211)
        self.ax2 = self.fig.add_subplot(212)
        self.fig.tight_layout(pad=3.0)

    def _plot_objective_1d(self, up_to_step=None):
        """Plot 1D objective function (Linear OLS, Logistic)"""
        self.ax1.clear()

        if self.func is None:
            return

        model_type = self.model_type_var.get()

        param_vals = np.linspace(self.domain_1d[0], self.domain_1d[1], 200)
        loss_vals = [self.func(p) for p in param_vals]

        self.ax1.plot(param_vals, loss_vals, "b-", linewidth=2, label="Loss")

        if model_type == "Linear OLS":
            self.ax1.set_xlabel("Slope (a)")
            self.ax1.set_title("Linear OLS Objective Function")
        else:
            self.ax1.set_xlabel("w₁")
            self.ax1.set_title("Logistic Regression Objective (Cross-Entropy)")

        self.ax1.set_ylabel("Loss")
        self.ax1.grid(True, alpha=0.3)

        if self.path:
            if up_to_step is None:
                up_to_step = len(self.path)

            path_to_show = self.path[:up_to_step]

            if path_to_show:
                p_path = [p if np.isscalar(p) else p[0] for p in path_to_show]
                loss_path = [self.func(p) for p in p_path]

                self.ax1.scatter(
                    p_path, loss_path, c="red", s=50, zorder=5, label="Steps"
                )
                if len(p_path) > 1:
                    self.ax1.plot(p_path, loss_path, "r--", alpha=0.5, linewidth=1)

                self.ax1.scatter(
                    [p_path[-1]],
                    [loss_path[-1]],
                    c="green",
                    s=100,
                    zorder=6,
                    marker="*",
                    label="Current",
                )
                self.ax1.scatter(
                    [p_path[0]],
                    [loss_path[0]],
                    c="orange",
                    s=80,
                    zorder=5,
                    marker="s",
                    label="Start",
                )

        self.ax1.legend()

    def _plot_objective_3d(self, up_to_step=None):
        """Plot 3D objective function (Quadratic OLS)"""
        self.ax1.clear()

        if self.func is None:
            return

        a_range = np.linspace(self.domain_2d[0][0], self.domain_2d[0][1], 30)
        b_range = np.linspace(self.domain_2d[1][0], self.domain_2d[1][1], 30)
        A, B = np.meshgrid(a_range, b_range)

        Z = np.zeros_like(A)
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                Z[i, j] = self.func(np.array([A[i, j], B[i, j]]))

        self.ax1.plot_surface(A, B, Z, cmap="viridis", alpha=0.7, edgecolor="none")
        self.ax1.set_xlabel("a (x² coef)")
        self.ax1.set_ylabel("b (x coef)")
        self.ax1.set_zlabel("Loss")
        self.ax1.set_title("Quadratic OLS Objective Function")

        if self.path:
            if up_to_step is None:
                up_to_step = len(self.path)

            path_to_show = self.path[:up_to_step]

            if path_to_show:
                a_path = [p[0] for p in path_to_show]
                b_path = [p[1] for p in path_to_show]
                loss_path = [self.func(np.array(p)) for p in path_to_show]

                self.ax1.scatter(a_path, b_path, loss_path, c="red", s=50, zorder=5)
                if len(a_path) > 1:
                    self.ax1.plot(
                        a_path, b_path, loss_path, "r-", linewidth=2, alpha=0.7
                    )

                self.ax1.scatter(
                    [a_path[-1]],
                    [b_path[-1]],
                    [loss_path[-1]],
                    c="green",
                    s=150,
                    marker="*",
                    zorder=6,
                )
                self.ax1.scatter(
                    [a_path[0]],
                    [b_path[0]],
                    [loss_path[0]],
                    c="orange",
                    s=100,
                    marker="s",
                    zorder=5,
                )

    def _plot_fitted_curves(self, up_to_step=None):
        """Plot fitted curves on dataset"""
        self._plot_dataset()

        if not self.path or self.get_params is None:
            return

        if up_to_step is None:
            up_to_step = len(self.path)

        path_to_show = self.path[:up_to_step]
        model_type = self.model_type_var.get()

        if model_type == "Linear OLS":
            self._plot_linear_fits(path_to_show)
        elif model_type == "Quadratic OLS":
            self._plot_quadratic_fits(path_to_show)
        elif model_type == "Logistic Regression":
            self._plot_logistic_fits(path_to_show)

        self.ax2.legend()

    def _plot_linear_fits(self, path_to_show):
        """Plot linear regression lines"""
        if self.dataset_type == "Two Clusters":
            x_line = np.array([self.X[:, 0].min(), self.X[:, 0].max()])
        else:
            x_line = np.array([self.X.min(), self.X.max()])

        n_lines = len(path_to_show)
        for i, p in enumerate(path_to_show[:-1] if n_lines > 1 else []):
            p_val = p if np.isscalar(p) else p[0]
            a, b = self.get_params(p_val)
            y_line = a * x_line + b
            alpha = 0.2 + 0.3 * (i / max(n_lines - 1, 1))
            self.ax2.plot(x_line, y_line, "r-", alpha=alpha, linewidth=1)

        if path_to_show:
            p_val = (
                path_to_show[-1]
                if np.isscalar(path_to_show[-1])
                else path_to_show[-1][0]
            )
            a, b = self.get_params(p_val)
            y_line = a * x_line + b
            self.ax2.plot(
                x_line,
                y_line,
                "g-",
                linewidth=2,
                label=f"Current: y = {a:.2f}x + {b:.2f}",
            )

    def _plot_quadratic_fits(self, path_to_show):
        """Plot quadratic regression curves"""
        if self.dataset_type == "Two Clusters":
            x_range = np.linspace(
                self.X[:, 0].min() - 0.5, self.X[:, 0].max() + 0.5, 100
            )
        else:
            x_range = np.linspace(self.X.min() - 0.5, self.X.max() + 0.5, 100)

        n_curves = len(path_to_show)
        for i, p in enumerate(path_to_show[:-1] if n_curves > 1 else []):
            a, b, c = self.get_params(np.array(p))
            y_curve = a * x_range**2 + b * x_range + c
            alpha = 0.2 + 0.3 * (i / max(n_curves - 1, 1))
            self.ax2.plot(x_range, y_curve, "r-", alpha=alpha, linewidth=1)

        if path_to_show:
            a, b, c = self.get_params(np.array(path_to_show[-1]))
            y_curve = a * x_range**2 + b * x_range + c
            self.ax2.plot(
                x_range,
                y_curve,
                "g-",
                linewidth=2,
                label=f"Current: y = {a:.2f}x² + {b:.2f}x + {c:.2f}",
            )

    def _plot_logistic_fits(self, path_to_show):
        """Plot logistic regression decision boundaries"""
        if self.dataset_type == "Two Clusters":
            X_cls = self.X
        else:
            X_cls = np.column_stack([self.X, self.y])

        x1_range = np.linspace(X_cls[:, 0].min() - 1, X_cls[:, 0].max() + 1, 100)

        n_lines = len(path_to_show)
        for i, p in enumerate(path_to_show[:-1] if n_lines > 1 else []):
            p_val = p if np.isscalar(p) else p[0]
            w1, w2, b = self.get_params(p_val)
            if abs(w2) > 1e-10:
                x2_boundary = -(w1 * x1_range + b) / w2
                alpha = 0.2 + 0.3 * (i / max(n_lines - 1, 1))
                self.ax2.plot(x1_range, x2_boundary, "r-", alpha=alpha, linewidth=1)

        if path_to_show:
            p_val = (
                path_to_show[-1]
                if np.isscalar(path_to_show[-1])
                else path_to_show[-1][0]
            )
            w1, w2, b = self.get_params(p_val)
            if abs(w2) > 1e-10:
                x2_boundary = -(w1 * x1_range + b) / w2
                self.ax2.plot(
                    x1_range,
                    x2_boundary,
                    "g-",
                    linewidth=2,
                    label=f"Boundary: {w1:.2f}x₁ + {w2:.2f}x₂ + {b:.2f} = 0",
                )

        self.ax2.set_xlim(X_cls[:, 0].min() - 1, X_cls[:, 0].max() + 1)
        self.ax2.set_ylim(X_cls[:, 1].min() - 1, X_cls[:, 1].max() + 1)

    def _update_stats(self, step=None):
        """Update statistics display"""
        self.stats_labels["dataset"].config(text=self.dataset_type_var.get())
        self.stats_labels["model"].config(text=self.model_type_var.get())
        self.stats_labels["alpha"].config(text=self.alpha_var.get())

        if step is None:
            step = len(self.path)

        self.stats_labels["iterations"].config(text=str(step))

        model_type = self.model_type_var.get()

        if self.path and step > 0 and self.get_params:
            current_p = self.path[step - 1]
            final_p = self.path[-1]

            if model_type == "Linear OLS":
                p_val = current_p if np.isscalar(current_p) else current_p[0]
                a, b = self.get_params(p_val)
                self.stats_labels["current_p1"].config(text=f"a = {a:.6f}")
                self.stats_labels["current_p2"].config(text=f"b = {b:.6f}")
                self.stats_labels["current_p3"].config(text="-")
                self.stats_labels["current_loss"].config(text=f"{self.func(p_val):.6f}")

                p_val = final_p if np.isscalar(final_p) else final_p[0]
                a, b = self.get_params(p_val)
                self.stats_labels["final_p1"].config(text=f"a = {a:.6f}")
                self.stats_labels["final_p2"].config(text=f"b = {b:.6f}")
                self.stats_labels["final_p3"].config(text="-")
                self.stats_labels["final_loss"].config(text=f"{self.func(p_val):.6f}")

            elif model_type == "Quadratic OLS":
                a, b, c = self.get_params(np.array(current_p))
                self.stats_labels["current_p1"].config(text=f"a = {a:.6f}")
                self.stats_labels["current_p2"].config(text=f"b = {b:.6f}")
                self.stats_labels["current_p3"].config(text=f"c = {c:.6f}")
                self.stats_labels["current_loss"].config(
                    text=f"{self.func(np.array(current_p)):.6f}"
                )

                a, b, c = self.get_params(np.array(final_p))
                self.stats_labels["final_p1"].config(text=f"a = {a:.6f}")
                self.stats_labels["final_p2"].config(text=f"b = {b:.6f}")
                self.stats_labels["final_p3"].config(text=f"c = {c:.6f}")
                self.stats_labels["final_loss"].config(
                    text=f"{self.func(np.array(final_p)):.6f}"
                )

            elif model_type == "Logistic Regression":
                p_val = current_p if np.isscalar(current_p) else current_p[0]
                w1, w2, b = self.get_params(p_val)
                self.stats_labels["current_p1"].config(text=f"w₁ = {w1:.6f}")
                self.stats_labels["current_p2"].config(text=f"w₂ = {w2:.6f}")
                self.stats_labels["current_p3"].config(text=f"b = {b:.6f}")
                self.stats_labels["current_loss"].config(text=f"{self.func(p_val):.6f}")

                p_val = final_p if np.isscalar(final_p) else final_p[0]
                w1, w2, b = self.get_params(p_val)
                self.stats_labels["final_p1"].config(text=f"w₁ = {w1:.6f}")
                self.stats_labels["final_p2"].config(text=f"w₂ = {w2:.6f}")
                self.stats_labels["final_p3"].config(text=f"b = {b:.6f}")
                self.stats_labels["final_loss"].config(text=f"{self.func(p_val):.6f}")
        else:
            for key in [
                "current_p1",
                "current_p2",
                "current_p3",
                "current_loss",
                "final_p1",
                "final_p2",
                "final_p3",
                "final_loss",
            ]:
                self.stats_labels[key].config(text="-")

    def _reset(self):
        """Reset the fitting state"""
        self.path = []
        self.current_step = 0
        self.func = None
        self.derivative = None
        self.gradient = None
        self.get_params = None

        self._setup_2d_plot()

        if self.X is not None:
            self._plot_dataset()

        self.ax1.set_title("Objective Function (Select model and click Run)")
        self._update_stats()
        self.canvas.draw()

    def _run_all(self):
        """Run complete optimization"""
        if self.X is None:
            return

        try:
            alpha = float(self.alpha_var.get())
        except ValueError:
            return

        self._setup_objective_function()

        model_type = self.model_type_var.get()

        if model_type == "Quadratic OLS":
            self._setup_3d_plot()
        else:
            self._setup_2d_plot()

        if self.is_2d_optimization:
            self.path = gradient_descent_2d(
                self.func, self.domain_2d, self.gradient, alpha
            )
        else:
            self.path = gradient_descent_1d(
                self.func, self.domain_1d, self.derivative, alpha
            )

        self.current_step = len(self.path)

        if model_type == "Quadratic OLS":
            self._plot_objective_3d()
        else:
            self._plot_objective_1d()

        self._plot_fitted_curves()
        self._update_stats()
        self.canvas.draw()

    def _run_step(self):
        """Run one step of gradient descent"""
        if self.X is None:
            return

        try:
            alpha = float(self.alpha_var.get())
        except ValueError:
            return

        model_type = self.model_type_var.get()

        if not self.path:
            self._setup_objective_function()

            if model_type == "Quadratic OLS":
                self._setup_3d_plot()
            else:
                self._setup_2d_plot()

            if self.is_2d_optimization:
                a_range, b_range = self.domain_2d
                p1 = random.uniform(
                    a_range[0] + 0.1 * (a_range[1] - a_range[0]),
                    a_range[1] - 0.1 * (a_range[1] - a_range[0]),
                )
                p2 = random.uniform(
                    b_range[0] + 0.1 * (b_range[1] - b_range[0]),
                    b_range[1] - 0.1 * (b_range[1] - b_range[0]),
                )
                self.path = [[p1, p2]]
            else:
                a, b = self.domain_1d
                x0 = random.uniform(a + 0.1 * (b - a), b - 0.1 * (b - a))
                self.path = [x0]

            self.current_step = 1
        else:
            current = self.path[-1]

            if self.is_2d_optimization:
                params = np.array(current)
                grad = self.gradient(params)

                if np.linalg.norm(grad) < 1e-6:
                    pass
                else:
                    new_params = params - alpha * grad
                    a_range, b_range = self.domain_2d
                    new_params[0] = max(a_range[0], min(a_range[1], new_params[0]))
                    new_params[1] = max(b_range[0], min(b_range[1], new_params[1]))
                    self.path.append(new_params.tolist())
            else:
                x = current if np.isscalar(current) else current[0]
                grad = self.derivative(x)

                if abs(grad) < 1e-6:
                    pass
                else:
                    x_new = x - alpha * grad
                    a, b = self.domain_1d
                    x_new = max(a, min(b, x_new))
                    self.path.append(x_new)

            self.current_step = len(self.path)

        if model_type == "Quadratic OLS":
            self._plot_objective_3d()
        else:
            self._plot_objective_1d()

        self._plot_fitted_curves()
        self._update_stats()
        self.canvas.draw()


def main():
    """Main entry point"""
    app = Lab2App()
    app.mainloop()


if __name__ == "__main__":
    main()
