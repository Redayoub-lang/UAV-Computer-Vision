#!/usr/bin/env python3
"""
DroneVision-VO: Advanced Monocular Visual Odometry & 3D Mapping System for UAVs
-------------------------------------------------------------------------------
Architecture:
  - Feature Tracking (Shi-Tomasi / Pyramidal Lucas-Kanade + Forward-Backward Validation)
  - Monocular Visual Odometry Pipeline (5-Point Essential Matrix + Relative Pose Integration)
  - 3D Map Point Triangulation & Local Scale Management
  - Keyframe-based Pose Estimation with Parallax-driven Initialization
  - Real-time Trajectory Mapping (Birds-Eye-View Canvas)
  - Telemetry Logger & Real-time Visual Dashboard

Author: Senior Computer Vision & Autonomous Navigation Engineer
"""

import cv2
import numpy as np
import time
from dataclasses import dataclass
from typing import Tuple, List, Optional, Dict, Any
from enum import Enum


class TrackingState(Enum):
    SYSTEM_NOT_INITIALIZED = 0
    INITIALIZING = 1
    TRACKING_OK = 2
    TRACKING_LOST = 3


@dataclass
class CameraConfig:
    """Camera intrinsic properties and optical parameters."""
    width: int = 640
    height: int = 480
    fx: float = 600.0
    fy: float = 600.0
    cx: float = 320.0
    cy: float = 240.0
    k1: float = 0.0
    k2: float = 0.0
    p1: float = 0.0
    p2: float = 0.0

    @property
    def K(self) -> np.ndarray:
        return np.array([
            [self.fx, 0.0, self.cx],
            [0.0, self.fy, self.cy],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)


@dataclass
class TrackerConfig:
    """Configurable parameters for feature tracking and pose estimation."""
    max_features: int = 400
    quality_level: float = 0.01
    min_distance: int = 12
    lk_win_size: Tuple[int, int] = (21, 21)
    lk_max_level: int = 3
    min_parallax_deg: float = 1.5
    min_inliers_count: int = 15
    ransac_prob: float = 0.999
    ransac_threshold_px: float = 1.0


class MapPoint:
    """3D point representation in world coordinates."""
    _id_counter = 0

    def __init__(self, position_3d: np.ndarray):
        MapPoint._id_counter += 1
        self.id = MapPoint._id_counter
        self.pt = np.array(position_3d, dtype=np.float64).reshape(3, 1)


class Frame:
    """Representation of a captured camera frame with extracted features."""
    _id_counter = 0

    def __init__(self, image: np.ndarray, timestamp: float, camera: CameraConfig):
        Frame._id_counter += 1
        self.id = Frame._id_counter
        self.timestamp = timestamp
        self.camera = camera
        self.gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        self.keypoints_2d: np.ndarray = np.empty((0, 2), dtype=np.float32)
        
        # World to Camera Transformation [R | t]
        self.R_cw: np.ndarray = np.eye(3, dtype=np.float64)
        self.t_cw: np.ndarray = np.zeros((3, 1), dtype=np.float64)

    @property
    def pose_matrix(self) -> np.ndarray:
        """4x4 Transformation Matrix T_cw (World to Camera)."""
        T = np.eye(4, dtype=np.float64)
        T[0:3, 0:3] = self.R_cw
        T[0:3, 3:4] = self.t_cw
        return T


class FeatureTracker:
    """Feature detection and Optical Flow propagation engine."""

    def __init__(self, config: TrackerConfig):
        self.config = config
        self.lk_params = dict(
            winSize=self.config.lk_win_size,
            maxLevel=self.config.lk_max_level,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )

    def detect_features(self, gray_img: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Detect strong corners using Shi-Tomasi algorithm."""
        pts = cv2.goodFeaturesToTrack(
            gray_img,
            maxCorners=self.config.max_features,
            qualityLevel=self.config.quality_level,
            minDistance=self.config.min_distance,
            mask=mask,
            blockSize=7
        )
        return pts.reshape(-1, 2) if pts is not None else np.empty((0, 2), dtype=np.float32)

    def track_optical_flow(
        self, prev_gray: np.ndarray, curr_gray: np.ndarray, prev_pts: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Track features using Pyramidal LK with Forward-Backward consistency check."""
        if len(prev_pts) == 0:
            return np.empty((0, 2)), np.empty((0, 2)), np.array([], dtype=bool)

        # Forward tracking
        curr_pts_fwd, status_fwd, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray, curr_gray, prev_pts.astype(np.float32), None, **self.lk_params
        )

        # Backward tracking
        prev_pts_bwd, status_bwd, _ = cv2.calcOpticalFlowPyrLK(
            curr_gray, prev_gray, curr_pts_fwd, None, **self.lk_params
        )

        # Distance error threshold
        fb_dist = np.linalg.norm(prev_pts - prev_pts_bwd, axis=1)
        valid = (status_fwd.ravel() == 1) & (status_bwd.ravel() == 1) & (fb_dist < 1.0)

        return prev_pts[valid], curr_pts_fwd[valid], valid


class MonocularVisualOdometry:
    """Core Monocular Visual Odometry System."""

    def __init__(self, camera_cfg: CameraConfig, tracker_cfg: TrackerConfig):
        self.cam = camera_cfg
        self.cfg = tracker_cfg
        self.tracker = FeatureTracker(tracker_cfg)

        self.state = TrackingState.SYSTEM_NOT_INITIALIZED
        self.ref_frame: Optional[Frame] = None
        self.curr_frame: Optional[Frame] = None

        self.trajectory: List[np.ndarray] = []
        self.map_points_3d: List[MapPoint] = []

        # World Pose State (Camera to World Transformation)
        self.R_wc = np.eye(3, dtype=np.float64)
        self.t_wc = np.zeros((3, 1), dtype=np.float64)

    def initialize(self, frame: Frame, prev_pts: np.ndarray, curr_pts: np.ndarray) -> bool:
        """Initialize pipeline with 5-Point Essential Matrix and Parallax check."""
        if len(prev_pts) < self.cfg.min_inliers_count:
            return False

        E, mask = cv2.findEssentialMat(
            prev_pts, curr_pts,
            focal=self.cam.fx,
            pp=(self.cam.cx, self.cam.cy),
            method=cv2.RANSAC,
            prob=self.cfg.ransac_prob,
            threshold=self.cfg.ransac_threshold_px
        )

        if E is None or E.shape != (3, 3):
            return False

        inlier_mask = mask.ravel() == 1
        pts1_in = prev_pts[inlier_mask]
        pts2_in = curr_pts[inlier_mask]

        if len(pts1_in) < self.cfg.min_inliers_count:
            return False

        _, R_rel, t_rel, _ = cv2.recoverPose(
            E, pts1_in, pts2_in,
            focal=self.cam.fx,
            pp=(self.cam.cx, self.cam.cy)
        )

        parallax = self._compute_parallax(pts1_in, pts2_in)
        if parallax < self.cfg.min_parallax_deg:
            return False

        self.ref_frame.R_cw = np.eye(3)
        self.ref_frame.t_cw = np.zeros((3, 1))

        frame.R_cw = R_rel
        frame.t_cw = t_rel

        self.R_wc = R_rel.T
        self.t_wc = -R_rel.T @ t_rel

        # Triangulate initial 3D Map
        points_3d = self._triangulate_points(
            pts1_in, pts2_in,
            self.ref_frame.pose_matrix[:3, :],
            frame.pose_matrix[:3, :]
        )

        for pt3d in points_3d:
            self.map_points_3d.append(MapPoint(pt3d))

        self.trajectory.append(self.t_wc.copy())
        self.state = TrackingState.TRACKING_OK
        return True

    def process_frame(self, image: np.ndarray, timestamp: float) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Process video frame and extract current visual state."""
        frame = Frame(image, timestamp, self.cam)
        telemetry = {
            "state": self.state.name,
            "tracked_features": 0,
            "position": self.t_wc.ravel().tolist(),
            "orientation_euler": self._rotation_matrix_to_euler(self.R_wc).tolist(),
            "inliers": 0,
            "map_points_count": len(self.map_points_3d)
        }

        if self.state == TrackingState.SYSTEM_NOT_INITIALIZED:
            self.ref_frame = frame
            self.ref_frame.keypoints_2d = self.tracker.detect_features(frame.gray)
            self.state = TrackingState.INITIALIZING
            self.trajectory.append(np.zeros((3, 1)))
            return image, telemetry

        if self.state in [TrackingState.INITIALIZING, TrackingState.TRACKING_OK]:
            prev_pts = self.ref_frame.keypoints_2d
            matched_prev, matched_curr, _ = self.tracker.track_optical_flow(
                self.ref_frame.gray, frame.gray, prev_pts
            )

            telemetry["tracked_features"] = len(matched_curr)

            if self.state == TrackingState.INITIALIZING:
                success = self.initialize(frame, matched_prev, matched_curr)
                if not success:
                    self.ref_frame = frame
                    self.ref_frame.keypoints_2d = self.tracker.detect_features(frame.gray)
                    return image, telemetry

            elif self.state == TrackingState.TRACKING_OK:
                success = self._track_frame(frame, matched_prev, matched_curr, telemetry)
                if not success:
                    self.state = TrackingState.TRACKING_LOST

            if len(matched_curr) < self.cfg.max_features // 2:
                new_pts = self.tracker.detect_features(frame.gray)
                if len(new_pts) > 0:
                    frame.keypoints_2d = self._combine_features(matched_curr, new_pts)
            else:
                frame.keypoints_2d = matched_curr

            self.curr_frame = frame
            self.ref_frame = frame

        return image, telemetry

    def _track_frame(self, frame: Frame, pts1: np.ndarray, pts2: np.ndarray, telemetry: dict) -> bool:
        """Estimate frame pose and integrate translation vectors."""
        if len(pts2) < self.cfg.min_inliers_count:
            return False

        E, mask = cv2.findEssentialMat(
            pts1, pts2,
            focal=self.cam.fx,
            pp=(self.cam.cx, self.cam.cy),
            method=cv2.RANSAC,
            prob=self.cfg.ransac_prob,
            threshold=self.cfg.ransac_threshold_px
        )

        if E is None or E.shape != (3, 3):
            return False

        inliers = mask.ravel() == 1
        telemetry["inliers"] = int(np.sum(inliers))

        if np.sum(inliers) < self.cfg.min_inliers_count:
            return False

        _, R_rel, t_rel, _ = cv2.recoverPose(
            E, pts1[inliers], pts2[inliers],
            focal=self.cam.fx,
            pp=(self.cam.cx, self.cam.cy)
        )

        # Pose Integration
        self.t_wc = self.t_wc + self.R_wc @ t_rel
        self.R_wc = self.R_wc @ R_rel

        frame.R_cw = self.R_wc.T
        frame.t_cw = -self.R_wc.T @ self.t_wc

        self.trajectory.append(self.t_wc.copy())
        return True

    def _triangulate_points(
        self, pts1: np.ndarray, pts2: np.ndarray, P1: np.ndarray, P2: np.ndarray
    ) -> np.ndarray:
        """Triangulate 3D points from projection matrices and keypoint pairs."""
        pts1_norm = cv2.undistortPoints(pts1.reshape(-1, 1, 2), self.cam.K, None).reshape(-1, 2)
        pts2_norm = cv2.undistortPoints(pts2.reshape(-1, 1, 2), self.cam.K, None).reshape(-1, 2)

        pts4D = cv2.triangulatePoints(P1, P2, pts1_norm.T, pts2_norm.T)
        pts3D = pts4D[:3, :] / pts4D[3, :]
        return pts3D.T

    def _compute_parallax(self, pts1: np.ndarray, pts2: np.ndarray) -> float:
        """Compute average parallax angle in degrees between matched pairs."""
        K_inv = np.linalg.inv(self.cam.K)
        angles = []
        for p1, p2 in zip(pts1, pts2):
            ray1 = K_inv @ np.array([p1[0], p1[1], 1.0])
            ray2 = K_inv @ np.array([p2[0], p2[1], 1.0])
            ray1 /= np.linalg.norm(ray1)
            ray2 /= np.linalg.norm(ray2)
            
            cos_angle = np.clip(np.dot(ray1, ray2), -1.0, 1.0)
            angles.append(np.degrees(np.arccos(cos_angle)))

        return float(np.mean(angles)) if len(angles) > 0 else 0.0

    @staticmethod
    def _rotation_matrix_to_euler(R: np.ndarray) -> np.ndarray:
        """Extract Roll, Pitch, Yaw angles from Rotation Matrix (degrees)."""
        sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
        if sy >= 1e-6:
            x = np.arctan2(R[2, 1], R[2, 2])
            y = np.arctan2(-R[2, 0], sy)
            z = np.arctan2(R[1, 0], R[0, 0])
        else:
            x = np.arctan2(-R[1, 2], R[1, 1])
            y = np.arctan2(-R[2, 0], sy)
            z = 0.0
        return np.degrees(np.array([x, y, z]))

    @staticmethod
    def _combine_features(existing_pts: np.ndarray, new_pts: np.ndarray, min_dist: float = 10.0) -> np.ndarray:
        """Filter and merge newly detected features avoiding spatial clustering."""
        if len(existing_pts) == 0:
            return new_pts

        combined = list(existing_pts)
        for npt in new_pts:
            dists = np.linalg.norm(existing_pts - npt, axis=1)
            if np.all(dists > min_dist):
                combined.append(npt)

        return np.array(combined, dtype=np.float32)


class TrajectoryVisualizer:
    """Dashboard & Birds-Eye-View (BEV) Visualizer for Odometry Trajectory."""

    def __init__(self, width: int = 500, height: int = 500, scale: float = 40.0):
        self.w = width
        self.h = height
        self.scale = scale
        self.canvas = np.zeros((height, width, 3), dtype=np.uint8)
        self.origin_x = width // 2
        self.origin_y = height // 2

    def draw_trajectory(self, trajectory: List[np.ndarray], current_frame: np.ndarray, telemetry: dict) -> np.ndarray:
        """Render side-by-side Camera View and 2D Trajectory Map."""
        self.canvas.fill(15)

        # Draw Grid Lines
        for x in range(0, self.w, 50):
            cv2.line(self.canvas, (x, 0), (x, self.h), (30, 30, 30), 1)
        for y in range(0, self.h, 50):
            cv2.line(self.canvas, (0, y), (self.w, y), (30, 30, 30), 1)

        cv2.drawMarker(self.canvas, (self.origin_x, self.origin_y), (0, 255, 255), cv2.MARKER_CROSS, 15, 1)

        # Plot 2D Trajectory (X-Z plane)
        if len(trajectory) > 1:
            for i in range(1, len(trajectory)):
                p1 = trajectory[i - 1].ravel()
                p2 = trajectory[i].ravel()

                x1 = int(self.origin_x + p1[0] * self.scale)
                y1 = int(self.origin_y - p1[2] * self.scale)
                x2 = int(self.origin_x + p2[0] * self.scale)
                y2 = int(self.origin_y - p2[2] * self.scale)

                cv2.line(self.canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)

            curr_pos = trajectory[-1].ravel()
            cx = int(self.origin_x + curr_pos[0] * self.scale)
            cy = int(self.origin_y - curr_pos[2] * self.scale)
            cv2.circle(self.canvas, (cx, cy), 5, (0, 0, 255), -1)

        frame_resized = cv2.resize(current_frame, (self.w, self.h))

        # Overlay Telemetry
        cv2.putText(frame_resized, f"VO State: {telemetry['state']}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame_resized, f"Tracked Features: {telemetry['tracked_features']}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        pos = telemetry["position"]
        cv2.putText(frame_resized, f"Pos [X,Y,Z]: [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]", (10, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        euler = telemetry["orientation_euler"]
        cv2.putText(frame_resized, f"Euler [R,P,Y]: [{euler[0]:.1f}, {euler[1]:.1f}, {euler[2]:.1f}] deg", (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 255), 1)

        return np.hstack((frame_resized, self.canvas))


def main():
    print("[INFO] Launching Monocular Visual Odometry Engine...")

    cam_cfg = CameraConfig(width=640, height=480, fx=600.0, fy=600.0, cx=320.0, cy=240.0)
    tracker_cfg = TrackerConfig(max_features=500, min_parallax_deg=1.5)

    vo_system = MonocularVisualOdometry(cam_cfg, tracker_cfg)
    visualizer = TrajectoryVisualizer(width=500, height=500, scale=40.0)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_cfg.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_cfg.height)

    start_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        timestamp = time.time() - start_time
        processed_img, telemetry = vo_system.process_frame(frame, timestamp)

        if vo_system.curr_frame is not None and len(vo_system.curr_frame.keypoints_2d) > 0:
            for pt in vo_system.curr_frame.keypoints_2d:
                x, y = pt.astype(int)
                cv2.circle(processed_img, (x, y), 2, (0, 255, 0), -1)

        dashboard = visualizer.draw_trajectory(vo_system.trajectory, processed_img, telemetry)

        cv2.imshow("UAV Visual Odometry Engine", dashboard)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()