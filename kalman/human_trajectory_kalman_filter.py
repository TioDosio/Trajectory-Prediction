import numpy as np
import cv2
from scipy.spatial.transform import Rotation
from dataclasses import dataclass
from typing import List, Tuple, Optional
import copy

import rospy
from human_awareness_msgs.msg import Person, PersonsList, BodyPart
from geometry_msgs.msg import Quaternion, PoseArray

@dataclass
class KeyPoint:
    """Structure matching the user's ROS message format"""
    part_id: str
    x: float
    y: float
    confidence: float

@dataclass 
class HeadPose:
    """Head pose with position and quaternion orientation"""
    position: np.ndarray  # [x, y, z]
    orientation: np.ndarray  # [x, y, z, w] quaternion

@dataclass
class BodyPose:
    """Body pose on plane z=0"""
    position: np.ndarray  # [x, y, z=0]
    orientation: np.ndarray  # [x, y, z, w] quaternion

class HumanTrajectoryKalmanFilter:
    """
    Kalman Filter implementation tailored for the user's specific setup:
    - 25 keypoints on plane z=0 at 1Hz
    - Head pose with full 3D position and orientation
    - Body pose with planar position
    """

    def __init__(self):
        # State vector design: Key joints + head pose (16D)
        # [neck_x, neck_y, neck_vx, neck_vy,
        #  midhip_x, midhip_y, midhip_vx, midhip_vy,
        #  rankle_x, rankle_y, rankle_vx, rankle_vy,
        #  lankle_x, lankle_y, lankle_vx, lankle_vy]

        self.state_dim = 16  # 4 keypoints × 4 states each
        self.obs_dim = 8     # 4 keypoints × 2 measurements each

        # Time step (1Hz = 1 second)
        self.dt = 1.0

        # Initialize state vector
        self.x = np.zeros((self.state_dim, 1))

        # Initialize covariance matrix
        self.P = np.eye(self.state_dim) * 1000.0  # High initial uncertainty

        # State transition matrix (constant velocity model)
        self.F = self._build_transition_matrix()

        # Measurement matrix (observe positions only)
        self.H = self._build_measurement_matrix()

        # Process noise covariance (tuned for 1Hz human motion)
        self.Q = self._build_process_noise()

        # Measurement noise covariance (based on keypoint confidence)
        self.R = self._build_measurement_noise()

        # Head pose tracking (separate filter for orientation)
        self.head_position = np.array([0.0, 0.0, 0.0])
        self.head_velocity = np.array([0.0, 0.0, 0.0])
        self.head_orientation = np.array([0.0, 0.0, 0.0, 1.0])  # quaternion

        # Key joint indices in the full keypoint list
        self.key_joint_indices = {
            'Neck': 1,
            'MidHip': 8, 
            'RAnkle': 11,
            'LAnkle': 14
        }

        # Initialize flag
        self.initialized = False

    def _build_transition_matrix(self):
        """Build state transition matrix for constant velocity model"""
        F = np.eye(self.state_dim)

        # For each keypoint (4 keypoints, each with x,y,vx,vy)
        for i in range(4):
            base_idx = i * 4
            # x = x + vx * dt
            F[base_idx, base_idx + 2] = self.dt
            # y = y + vy * dt  
            F[base_idx + 1, base_idx + 3] = self.dt
            # vx = vx (constant velocity)
            # vy = vy (constant velocity)

        return F

    def _build_measurement_matrix(self):
        """Build measurement matrix (observe positions only)"""
        H = np.zeros((self.obs_dim, self.state_dim))

        # For each keypoint, observe x and y positions
        for i in range(4):
            H[i*2, i*4] = 1.0       # observe x position
            H[i*2+1, i*4+1] = 1.0   # observe y position

        return H

    def _build_process_noise(self):
        """Build process noise covariance matrix tuned for 1Hz human motion"""
        Q = np.zeros((self.state_dim, self.state_dim))

        # Process noise parameters (tuned for human motion at 1Hz)
        sigma_pos = 0.1    # Position uncertainty (meters)
        sigma_vel = 0.5    # Velocity uncertainty (m/s)

        for i in range(4):
            base_idx = i * 4
            # Position noise
            Q[base_idx, base_idx] = sigma_pos**2
            Q[base_idx+1, base_idx+1] = sigma_pos**2
            # Velocity noise  
            Q[base_idx+2, base_idx+2] = sigma_vel**2
            Q[base_idx+3, base_idx+3] = sigma_vel**2

        return Q

    def _build_measurement_noise(self):
        """Build measurement noise covariance matrix"""
        # Base measurement noise (depends on keypoint detection accuracy)
        sigma_measurement = 0.05  # 5cm measurement uncertainty
        return np.eye(self.obs_dim) * sigma_measurement**2

    def _extract_key_joints(self, keypoints: List[KeyPoint]) -> Tuple[np.ndarray, np.ndarray]:
        """Extract key joint positions and confidences"""
        positions = []
        confidences = []

        key_joints = ['Neck', 'MidHip', 'RAnkle', 'LAnkle']

        for joint_name in key_joints:
            # Find keypoint by name
            keypoint = next((kp for kp in keypoints if kp.part_id == joint_name), None)
            if keypoint:
                positions.extend([keypoint.x, keypoint.y])
                confidences.extend([keypoint.confidence, keypoint.confidence])
            else:
                positions.extend([0.0, 0.0])
                confidences.extend([0.0, 0.0])

        return np.array(positions).reshape(-1, 1), np.array(confidences)

    def _adaptive_measurement_noise(self, confidences: np.ndarray):
        """Adapt measurement noise based on keypoint confidence"""
        R = self.R.copy()

        for i, conf in enumerate(confidences):
            if conf > 0.1:  # Valid detection
                # Lower noise for higher confidence
                R[i, i] = self.R[i, i] / (conf + 0.1)
            else:  # Low confidence or missing detection
                R[i, i] = self.R[i, i] * 10.0  # Higher noise

        return R

    def initialize(self, keypoints: List[KeyPoint], head_pose: HeadPose):
        """Initialize filter with first measurement"""
        positions, confidences = self._extract_key_joints(keypoints)

        # Initialize state with positions (velocities start at 0)
        for i in range(4):
            self.x[i*4] = positions[i*2]      # x position
            self.x[i*4+1] = positions[i*2+1]  # y position
            # velocities remain 0

        # Initialize head pose
        self.head_position = head_pose.position.copy()
        self.head_orientation = head_pose.orientation.copy()

        self.initialized = True

    def predict(self) -> np.ndarray:
        """Prediction step"""
        if not self.initialized:
            return self.x.copy()

        # Predict state
        self.x = self.F @ self.x

        # Predict covariance
        self.P = self.F @ self.P @ self.F.T + self.Q

        # Predict head position (simple constant velocity)
        self.head_position += self.head_velocity * self.dt

        return self.x.copy()

    def update(self, keypoints: List[KeyPoint], head_pose: HeadPose):
        """Update step with measurements"""
        if not self.initialized:
            self.initialize(keypoints, head_pose)
            return

        # Extract measurements
        z, confidences = self._extract_key_joints(keypoints)

        # Adaptive measurement noise
        R = self._adaptive_measurement_noise(confidences)

        # Innovation
        y = z - self.H @ self.x

        # Innovation covariance
        S = self.H @ self.P @ self.H.T + R

        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # Update state
        self.x = self.x + K @ y

        # Update covariance
        I = np.eye(self.state_dim)
        self.P = (I - K @ self.H) @ self.P

        # Update head pose
        self._update_head_pose(head_pose)

    def _update_head_pose(self, head_pose: HeadPose):
        """Update head pose tracking"""
        # Simple update for head position (can be enhanced with EKF for orientation)
        dt = self.dt

        # Update velocity estimate
        self.head_velocity = (head_pose.position - self.head_position) / dt

        # Update position
        self.head_position = head_pose.position.copy()

        # Update orientation (simple direct update, can be improved)
        self.head_orientation = head_pose.orientation.copy()

    def get_trajectory_prediction(self, steps_ahead: int = 5) -> np.ndarray:
        """Predict trajectory multiple steps into the future"""
        predictions = []
        x_pred = self.x.copy()

        for step in range(steps_ahead):
            x_pred = self.F @ x_pred
            predictions.append(x_pred.copy())

        return np.array(predictions)

    def get_center_of_mass(self) -> Tuple[float, float]:
        """Calculate center of mass from key joint positions"""
        # Extract positions from state
        positions = []
        for i in range(4):
            x_pos = self.x[i*4, 0]
            y_pos = self.x[i*4+1, 0]
            positions.append([x_pos, y_pos])

        positions = np.array(positions)

        # Simple average (can be weighted by body segment masses)
        com_x = np.mean(positions[:, 0])
        com_y = np.mean(positions[:, 1])

        return com_x, com_y

    def get_movement_direction(self) -> float:
        """Estimate movement direction from velocities"""
        # Extract velocities
        velocities = []
        for i in range(4):
            vx = self.x[i*4+2, 0]
            vy = self.x[i*4+3, 0]
            velocities.append([vx, vy])

        velocities = np.array(velocities)

        # Average velocity
        avg_velocity = np.mean(velocities, axis=0)

        # Direction angle
        direction = np.arctan2(avg_velocity[1], avg_velocity[0])

        return direction

# Example usage class for coordinate transformations
class CoordinateTransformer:
    """Handle coordinate transformations as provided by the user"""

    def __init__(self, P_matrix, K_matrix, RT_matrix):
        self.P = P_matrix  # Projection matrix
        self.K = K_matrix  # Camera intrinsic matrix  
        self.RT = RT_matrix  # Rotation-translation matrix

    def estimate_3d_points_with_homography(self, points_2d):
        """User's homography function"""
        points_2d = np.vstack((points_2d, np.ones([1, points_2d.shape[1]])))
        homography = np.hstack([self.P[:, 0:2], self.P[:, 3].reshape(3, 1)])
        points_3d = np.linalg.solve(homography, points_2d)
        points_3d = points_3d/points_3d[2]
        points_3d[2] = 0
        return points_3d

    def estimate_3d_points_from_feet(self, points_2d, feetpoints_2d):
        """User's feet-based height estimation function"""
        # Implementation as provided by user
        p2d_floor = points_2d.copy()
        p2d_floor[1, :] = feetpoints_2d[1, :]

        p3d = self.estimate_3d_points_with_homography(p2d_floor)

        for pxy, xy, i in zip(p3d.T[:, 0:2], points_2d.T, range(0, len(points_2d.T))):
            txy = self.RT[:, 3].reshape(3, 1) + np.array([[self.RT[0, 0]*pxy[0]+self.RT[0, 1]*pxy[1]],
                                          [self.RT[1, 0]*pxy[0]+self.RT[1, 1]*pxy[1]],
                                          [self.RT[2, 0]*pxy[0]+self.RT[2, 1]*pxy[1]]])

            r3_t = np.hstack([self.RT[:, 2].reshape(3, 1), txy])
            A = np.matmul(self.K, r3_t)

            beta = (A[1, 1]-xy[1]*A[2, 1])/(xy[1]*A[2,0]-A[1,0])
            p3d[2, i] = beta

        return p3d

# Usage example
def main():
    """Example of how to use the filter with user's data format"""

    # Initialize filter
    kf = HumanTrajectoryKalmanFilter()

    # Example keypoints (user's format)
    keypoints = [
        KeyPoint("Neck", 318, 285, 0.894),
        KeyPoint("MidHip", 318, 330, 0.938), 
        KeyPoint("RAnkle", 316, 401, 0.910),
        KeyPoint("LAnkle", 325, 398, 0.828)
        # ... other keypoints
    ]

    # Example head pose (user's format)
    head_pose = HeadPose(
        position=np.array([3.568, 0.298, 1.392]),
        orientation=np.array([-0.140, -0.017, 0.989, -0.042])
    )

    # Process measurement
    prediction = kf.predict()
    kf.update(keypoints, head_pose)

    # Get trajectory prediction
    future_trajectory = kf.get_trajectory_prediction(steps_ahead=5)

    # Get center of mass
    com_x, com_y = kf.get_center_of_mass()

    print(f"Center of Mass: ({com_x:.2f}, {com_y:.2f})")
    print(f"Movement Direction: {kf.get_movement_direction():.2f} radians")

if __name__ == "__main__":
    main()
