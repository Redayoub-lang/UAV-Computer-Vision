# DroneVision-Suite: Autonomous UAV Vision & Guidance Pipeline

A complete Monocular Computer Vision and Visual Servoing suite for Unmanned Aerial Vehicles (UAVs). The system runs in real-time to provide 6-DoF pose estimation, feature tracking, dynamic obstacle/target detection, precision landing, and velocity control commands without relying on external GPS/IMU sensors.

---

## System Architecture
+-------------------------------+
                   |      Monocular Camera Stream  |
                   +---------------+---------------+
                                   |
                 +-----------------+-----------------+
                 |                                   |
                 v                                   v
   +---------------------------+       +---------------------------+
   | Visual Odometry (6-DoF)   |       |   AI Target & Marker      |
   | - Grid-Based Feature LK   |       | - YOLOv8 Target Tracking  |
   | - Essential Matrix RANSAC |       | - ArUco Precision Landing |
   +-------------+-------------+       +-------------+-------------+
                 |                                   |
                 +-----------------+-----------------+
                                   |
                                   v
                   +-------------------------------+
                   |  Visual Servoing Controller   |
                   | - Compute Vx, Vy, Vz, Yaw     |
                   | - Render UAV Tactical HUD     |
                   +-------------------------------+

---

## Mathematical & Control Formulation

### 1. 6-DoF Pose & Euler Angle Extraction
Relative camera rotation $R$ and translation $t$ are derived via Essential Matrix decomposition ($E = [\mathbf{t}]_{\times} \mathbf{R}$). The Roll ($\phi$), Pitch ($\theta$), and Yaw ($\psi$) angles are extracted using:

$$\theta = \arcsin(-R_{2,0})$$

$$\phi = \arctan2(R_{2,1}, R_{2,2}), \quad \psi = \arctan2(R_{1,0}, R_{0,0})$$

---

### 2. Visual Servoing Guidance Law
The P-controller calculates velocity setpoints $(V_x, V_y, V_z, \dot{\psi})$ based on pixel offset $(\Delta x, \Delta y)$ from the frame center $(c_x, c_y)$:

$$V_x = K_p \cdot (c_y - y_{\text{target}}), \quad V_y = K_p \cdot (x_{\text{target}} - c_x)$$

$$\dot{\psi} = K_{\text{yaw}} \cdot \frac{x_{\text{target}} - c_x}{w / 2}$$

---

## Core Features

* **Monocular Visual Odometry (MVO):** 6-DoF motion estimation using Pyramidal Lucas-Kanade Optical Flow with uniform grid feature selection.
* **Dynamic Outlier Rejection:** Median flow deviation filtering to eliminate moving objects from pose calculations.
* **YOLOv8 Target Tracking:** Real-time multi-class object detection for aerial tracking.
* **ArUco Precision Landing:** Visual marker detection for autonomous touchdown protocols.
* **Tactical Flight HUD:** Real-time display of velocity commands, camera orientation, and trajectory map.

---

## Quick Start

### Installation
```bash
pip install opencv-python numpy ultralytics
Execution
Bash
python drone_vision_tracker.py