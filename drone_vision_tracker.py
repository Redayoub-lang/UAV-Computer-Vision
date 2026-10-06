import cv2
import numpy as np


class DroneVisionTracker:
    def __init__(self, focal_length=800.0, principal_point=(320.0, 240.0), grid_size=(4, 4), max_pts_per_cell=15):
        self.fx, self.fy = focal_length, focal_length
        self.cx, self.cy = principal_point
        self.K = np.array([[self.fx, 0, self.cx],
                           [0, self.fy, self.cy],
                           [0, 0, 1.0]], dtype=np.float64)

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

        self.prev_frame = None
        self.prev_pts = None
        self.cur_R = np.eye(3)
        self.cur_t = np.zeros((3, 1))
        self.trajectory = [np.zeros((2, 1))]

        # Motion threshold and visual scale setup
        self.min_parallax = 1.5
        self.scale_factor = 0.05

    def detect_features_grid(self, gray_frame):
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
        if p0 is None or p1 is None or len(p0) < 8:
            return p0, p1

        flow = p1 - p0
        dx, dy = flow[:, 0, 0], flow[:, 0, 1]

        med_dx, med_dy = np.median(dx), np.median(dy)
        dev = np.sqrt((dx - med_dx) ** 2 + (dy - med_dy) ** 2)

        inliers = dev < (1.5 * np.std(dev) + 2.0)
        return p0[inliers], p1[inliers]

    def process_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        annotated = frame.copy()
        status_msg = "TRACKING_OK"

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
            # Parallax calculation
            disp = np.linalg.norm(good_new - good_old, axis=2)
            avg_parallax = np.mean(disp)

            # Update pose only when camera actually moves
            if avg_parallax >= self.min_parallax:
                E, mask = cv2.findEssentialMat(
                    good_new, good_old, self.K, method=cv2.RANSAC, prob=0.999, threshold=1.0
                )
                if E is not None and E.shape == (3, 3):
                    _, R, t, _ = cv2.recoverPose(E, good_new, good_old, self.K)

                    scaled_t = t * (avg_parallax * self.scale_factor)
                    self.cur_t = self.cur_t + self.cur_R.dot(scaled_t)
                    self.cur_R = R.dot(self.cur_R)

                    self.trajectory.append(np.array([[self.cur_t[0][0]], [self.cur_t[2][0]]]))
            else:
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

        if len(self.trajectory) > 1:
            points = []
            for pos in self.trajectory:
                x = int(pos[0][0] * 5) + ox
                z = int(-pos[1][0] * 5) + oy
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

        color = (0, 255, 0) if "OK" in status else ((255, 255, 0) if "STATIONARY" in status else (0, 0, 255))
        cv2.putText(processed_frame, f"VO State: {status}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(processed_frame, f"Tracked Features: {len(tracker.prev_pts) if tracker.prev_pts is not None else 0}",
                    (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        dashboard = np.hstack((processed_frame, traj_map))
        cv2.imshow("UAV Visual Odometry Engine", dashboard)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()