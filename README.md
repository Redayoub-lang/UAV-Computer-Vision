# DroneVision-Flow: Visual Ego-Motion Estimation & Feature Tracking for UAVs

A pure Monocular Computer Vision pipeline for Unmanned Aerial Vehicle (UAV) feature tracking and relative 5-DoF ego-motion estimation without relying on external sensors (IMU/GPS).

---

## 1. Mathematical Formulation

### A. Optical Flow Constraint Equation
Assuming constant brightness for a pixel at location $(x, y)$ at time $t$:

$$I(x, y, t) = I(x + dx, y + dy, t + dt)$$

Applying a first-order Taylor series expansion yields the fundamental optical flow constraint:

$$I_x u + I_y v + I_t = 0$$

Where:
* $I_x = \frac{\partial I}{\partial x}$, $I_y = \frac{\partial I}{\partial y}$, $I_t = \frac{\partial I}{\partial t}$ are the spatial and temporal intensity gradients.
* $\mathbf{v} = [u, v]^T = \left[\frac{dx}{dt}, \frac{dy}{dt}\right]^T$ is the local optical flow vector.

---

### B. Feature Selection (Shi-Tomasi Corner Detector)
Features are identified by analyzing the local structure tensor matrix $M$:

$$M = \sum_{W} w(x,y) \begin{bmatrix} I_x^2 & I_x I_y \\ I_x I_y & I_y^2 \end{bmatrix}$$

Corners are detected where the minimum eigenvalue satisfies:

$$R = \min(\lambda_1, \lambda_2) > \lambda_{\text{threshold}}$$

---

### C. Epipolar Geometry & Essential Matrix Estimation
To estimate relative rotation $R$ and translation $t$ between consecutive frames, the epipolar constraint is evaluated:

$$\mathbf{x}_2^T \mathbf{E} \mathbf{x}_1 = 0$$

Where:
* $\mathbf{x}_1, \mathbf{x}_2$ are normalized image coordinates derived from camera intrinsic matrix $K$ ($\mathbf{x} = K^{-1} \mathbf{p}$).
* $\mathbf{E} = [\mathbf{t}]_{\times} \mathbf{R}$ is the Essential Matrix.

---

## 2. Project Architecture

* **`drone_vision_tracker.py`**: Core Python implementation for visual odometry, feature tracking, dynamic outlier filtering, and trajectory rendering.
* **`README.md`**: Technical documentation and mathematical formulation.

---

## 3. Setup & Execution

### Prerequisites
```bash
pip install opencv-python numpy