import cv2
import numpy as np
import math


class DroneVisionTracker:
    def __init__(self, focal_length=800.0, principal_point=(320.0, 240.0), grid_size=(4, 4), max_pts_per_cell=15):
        # 1. Camera Intrinsic Matrix K
        self.fx, self.fy = focal_length, focal_length
        self.cx, self.cy = principal_point
        self.K = np.array([[self.fx, 0, self.cx],
                           [0, self.fy, self.cy],
                           [0, 0, 1.0]], dtype=np.float64)

        # 2. Tracking Parameters
        self.grid_size = grid_size
        self.max_pts_per_cell = max_pts_per_cell
        self.lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )
        self.feature_params = dict(
            maxCorners=max_pts_per_cell,
            qualityLevel=0.01,
            minDistance=8,
            blockSize=7
        )

        # 3. State Variables
        self.prev_frame = None
        self.prev_pts = None
        self.cur_R = np.eye(3)
        self.cur_t = np.zeros((3, 1))
        self.trajectory_3d = [np.zeros((3, 1))]
        self.euler_angles = np.zeros(3)  # Roll, Pitch, Yaw in degrees

        # Motion thresholding
        self.min_parallax = 1.5
        self.scale_factor = 0.05

        # 4. ArUco Precision Landing Detector Setup
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)

    def detect_features_grid(self, gray_frame):
        """Distribute feature points uniformly across grid cells."""
        h, w = gray_frame.shape
        gh, gw = h // self.grid_size[0], w // self.grid_size[1]
        grid_pts = []

        for r in range(self.grid_size[0]):
            for c in range(self.grid_size[1]):
                cell = gray_frame[r * gh:(r + 1) * gh, c * gw:(c + 1) * gw]
                corners = cv2.goodFeaturesToTrack(cell, mask=None, **self.feature_params)
                if corners is not None:
                    for pt in corners:
                        pt[0][0] += c * gw
                        pt[0][1] += r * gh
                        grid_pts.append(pt)

        return np.float32(grid_pts) if len(grid_pts) > 0 else None

    def filter_dynamic_outliers(self, p0, p1):
        """Filter out moving objects in frame."""
        if p0 is None or p1 is None or len(p0) < 8:
            return p0, p1

        flow = p1 - p0
        dx, dy = flow[:, 0, 0], flow[:, 0, 1]

        med_dx, med_dy = np.median(dx), np.median(dy)
        dev = np.sqrt((dx - med_dx) ** 2 + (dy - med_dy) ** 2)

        inliers = dev < (1.5 * np.std(dev) + 2.0)
        return p0[inliers], p1[inliers]

    def rotation_matrix_to_euler(self, R):
        """Extract Roll, Pitch, Yaw angles from Rotation Matrix R."""
        sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
        singular = sy < 1e-6

        if not singular:
            x = math.atan2(R[2, 1], R[2, 2])
            y = math.atan2(-R[2, 0], sy)
            z = math.atan2(R[1, 0], R[0, 0])
        else:
            x = math.atan2(-R[1, 2], R[1, 1])
            y = math.atan2(-R[2, 0], sy)
            z = 0

        return np.degrees([x, y, z])

    def detect_landing_target(self, frame):
        """Detect ArUco landing marker for precision landing."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = self.aruco_detector.detectMarkers(gray)
        landing_info = None

        if ids is not None:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            # Center coordinates of the primary landing marker
            c = corners[0][0]
            center_x = int(np.mean(c[:, 0]))
            center_y = int(np.mean(c[:, 1]))
            cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
            cv2.putText(frame, f"LANDING PAD ID: {ids[0][0]}", (center_x - 50, center_y - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            landing_info = (center_x, center_y, ids[0][0])

        return landing_info

    def process_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        annotated = frame.copy()
        status_msg = "TRACKING_OK"

        # Detect landing pad
        landing_info = self.detect_landing_target(annotated)
        if landing_info is not None:
            status_msg = f"LANDING_PAD_DETECTED [ID:{landing_info[2]}]"

        if self.prev_frame is None:
            self.prev_frame = gray
            self.prev_pts = self.detect_features_grid(gray)
            return annotated, status_msg

        if self.prev_pts is None or len(self.prev_pts) < 40:
            self.prev_pts = self.detect_features_grid(gray)

        if self.prev_pts is None or len(self.prev_pts) < 8:
            self.prev_frame = gray
            return annotated, "SEARCHING_FEATURES"

        p1, st, _ = cv2.calcOpticalFlowPyrLK(self.prev_frame, gray, self.prev_pts, None, **self.lk_params)
        good_old = self.prev_pts[st == 1]
        good_new = p1[st == 1]

        good_old, good_new = self.filter_dynamic_outliers(
            good_old.reshape(-1, 1, 2), good_new.reshape(-1, 1, 2)
        )

        if good_new is not None and len(good_new) >= 8:
            disp = np.linalg.norm(good_new - good_old, axis=2)
            avg_parallax = np.mean(disp)

            if avg_parallax >= self.min_parallax:
                E, mask = cv2.findEssentialMat(
                    good_new, good_old, self.K, method=cv2.RANSAC, prob=0.999, threshold=1.0
                )
                if E is not None and E.shape == (3, 3):
                    _, R, t, _ = cv2.recoverPose(E, good_new, good_old, self.K)

                    scaled_t = t * (avg_parallax * self.scale_factor)
                    self.cur_t = self.cur_t + self.cur_R.dot(scaled_t)
                    self.cur_R = R.dot(self.cur_R)

                    # Update 3D pose trajectory
                    self.trajectory_3d.append(self.cur_t.copy())
                    self.euler_angles = self.rotation_matrix_to_euler(self.cur_R)
            else:
                if "LANDING" not in status_msg:
                    status_msg = "STATIONARY"

            for pt in good_new:
                x, y = map(int, pt.ravel())
                cv2.circle(annotated, (x, y), 3, (0, 255, 0), -1)

            self.prev_pts = good_new.reshape(-1, 1, 2)
        else:
            status_msg = "OUTLIER_REJECTED"

        self.prev_frame = gray
        return annotated, status_msg

    def draw_trajectory_map(self, width=600, height=600):
        traj_img = np.zeros((height, width, 3), dtype=np.uint8)

        for i in range(0, width, 50):
            cv2.line(traj_img, (i, 0), (i, height), (30, 30, 30), 1)
            cv2.line(traj_img, (0, i), (width, i), (30, 30, 30), 1)

        ox, oy = width // 2, height // 2

        if len(self.trajectory_3d) > 1:
            points = []
            for pos in self.trajectory_3d:
                x = int(pos[0][0] * 5) + ox
                z = int(-pos[2][0] * 5) + oy
                points.append((x, z))

            for i in range(1, len(points)):
                cv2.line(traj_img, points[i - 1], points[i], (0, 255, 0), 2)

            cv2.circle(traj_img, points[-1], 5, (0, 0, 255), -1)

        return traj_img


def main():
    cap = cv2.VideoCapture(0)
    tracker = DroneVisionTracker()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 480))
        processed_frame, status = tracker.process_frame(frame)
        traj_map = tracker.draw_trajectory_map(width=640, height=480)

        # Display Telemetry Overlay
        color = (0, 255, 0) if "OK" in status else ((255, 255, 0) if "STATIONARY" in status or "LANDING" in status else (0, 0, 255))
        cv2.putText(processed_frame, f"VO State: {status}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        pos_str = f"Pos (X,Y,Z): [{tracker.cur_t[0][0]:.2f}, {tracker.cur_t[1][0]:.2f}, {tracker.cur_t[2][0]:.2f}]"
        cv2.putText(processed_frame, pos_str, (20, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        euler_str = f"Euler [R,P,Y]: [{tracker.euler_angles[0]:.1f}, {tracker.euler_angles[1]:.1f}, {tracker.euler_angles[2]:.1f}] deg"
        cv2.putText(processed_frame, euler_str, (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

        dashboard = np.hstack((processed_frame, traj_map))
        cv2.imshow("UAV Autonomous Vision Engine", dashboard)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()