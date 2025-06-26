#!/usr/bin/env python2
# -*- coding: utf-8 -*-

"""
Human Trajectory Tracker with Kalman Filter
Subscribes to image_detections, raw_heads, and raw_bodies topics
Processes human pose data with Kalman filtering for trajectory prediction

Compatible with ROS 1 Melodic and Python 2.7
"""

import rospy
import numpy as np
import math
import tf
from human_awareness_msgs.msg import Person, PersonsList, BodyPart
from geometry_msgs.msg import Quaternion, PoseArray, Pose, Point, Vector3, PoseStamped
from visualization_msgs.msg import Marker, MarkerArray
from nav_msgs.msg import Path
import copy

class KalmanFilter3D:
    """
    A Kalman filter implementation for 3D pose tracking (position and orientation)
    State vector: [x, y, z, vx, vy, vz, roll, pitch, yaw, roll_rate, pitch_rate, yaw_rate]
    """
    def __init__(self, dt, std_acc=1.0, std_meas_pos=0.1, std_meas_orient=0.1):
        self.dt = dt
        
        # State vector dimension: x, y, z, vx, vy, vz, roll, pitch, yaw, roll_rate, pitch_rate, yaw_rate
        state_dim = 12
        
        # Measurement vector dimension: x, y, z, roll, pitch, yaw
        meas_dim = 6
        
        # Initial state vector
        self.x = np.zeros((state_dim, 1))
        
        # Define the State Transition Matrix A
        self.A = np.eye(state_dim)
        # Position update with velocity
        self.A[0, 3] = self.dt  # x += vx * dt
        self.A[1, 4] = self.dt  # y += vy * dt
        self.A[2, 5] = self.dt  # z += vz * dt
        # Orientation update with angular velocity
        self.A[6, 9] = self.dt  # roll += roll_rate * dt
        self.A[7, 10] = self.dt  # pitch += pitch_rate * dt
        self.A[8, 11] = self.dt  # yaw += yaw_rate * dt
        
        # Define the Measurement Matrix H
        self.H = np.zeros((meas_dim, state_dim))
        self.H[0, 0] = 1.0  # x
        self.H[1, 1] = 1.0  # y
        self.H[2, 2] = 1.0  # z
        self.H[3, 6] = 1.0  # roll
        self.H[4, 7] = 1.0  # pitch
        self.H[5, 8] = 1.0  # yaw
        
        # Process noise covariance matrix Q
        self.Q = np.eye(state_dim) * std_acc**2
        
        # Measurement noise covariance matrix R
        self.R = np.eye(meas_dim)
        self.R[0:3, 0:3] *= std_meas_pos**2  # position measurement noise
        self.R[3:6, 3:6] *= std_meas_orient**2  # orientation measurement noise
        
        # Initial covariance matrix
        self.P = np.eye(state_dim)
    
    def predict(self):
        """Predict the state forward by dt"""
        # x = A * x
        self.x = np.dot(self.A, self.x)
        
        # P = A * P * A' + Q
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q
        
        return self.x
    
    def update(self, z):
        """Update the filter with a new measurement z"""
        # y = z - H * x (measurement residual)
        y = z - np.dot(self.H, self.x)
        
        # S = H * P * H' + R (residual covariance)
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        
        # K = P * H' * inv(S) (Kalman gain)
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        
        # x = x + K * y (update state estimate)
        self.x = self.x + np.dot(K, y)
        
        # P = (I - K * H) * P (update covariance)
        I = np.eye(self.A.shape[0])
        self.P = np.dot((I - np.dot(K, self.H)), self.P)
        
        return self.x
    
    def get_position(self):
        """Return the current position estimate"""
        return self.x[0:3].flatten()
    
    def get_velocity(self):
        """Return the current velocity estimate"""
        return self.x[3:6].flatten()
    
    def get_orientation(self):
        """Return the current orientation estimate as roll, pitch, yaw"""
        return self.x[6:9].flatten()
    
    def get_angular_velocity(self):
        """Return the current angular velocity estimate"""
        return self.x[9:12].flatten()

class PersonTracker:
    """
    Tracks a single person using a Kalman filter
    """
    def __init__(self, person_id, initial_pos, initial_orient=None, dt=0.1):
        self.person_id = person_id
        self.kf = KalmanFilter3D(dt=dt)
        
        # Initialize position
        self.kf.x[0:3] = initial_pos.reshape((3, 1))
        
        # Initialize orientation if provided
        if initial_orient is not None:
            self.kf.x[6:9] = initial_orient.reshape((3, 1))
        
        # Store history of measurements and predictions
        self.positions = [initial_pos]
        self.orientations = [initial_orient] if initial_orient is not None else [np.zeros(3)]
        self.predicted_positions = [initial_pos]
        
        # Track update time
        self.last_update_time = rospy.Time.now()
        self.active = True
        self.missed_updates = 0
        self.max_missed_updates = 10  # Maximum number of consecutive missed updates before considering the person lost
    
    def update(self, position, orientation=None):
        """Update the tracker with a new measurement"""
        # Create measurement vector
        z = np.zeros((6, 1))
        z[0:3] = position.reshape((3, 1))
        
        if orientation is not None:
            z[3:6] = orientation.reshape((3, 1))
        
        # Update the Kalman filter
        self.kf.update(z)
        
        # Store the measurement
        self.positions.append(position)
        self.orientations.append(orientation if orientation is not None else np.zeros(3))
        
        # Update tracking info
        self.last_update_time = rospy.Time.now()
        self.missed_updates = 0
    
    def predict(self):
        """Predict the next state"""
        self.kf.predict()
        predicted_position = self.kf.get_position()
        self.predicted_positions.append(predicted_position)
        
        if rospy.Time.now() - self.last_update_time > rospy.Duration(1.0):
            self.missed_updates += 1
            if self.missed_updates > self.max_missed_updates:
                self.active = False
        
        return {
            'position': predicted_position,
            'velocity': self.kf.get_velocity(),
            'orientation': self.kf.get_orientation(),
            'angular_velocity': self.kf.get_angular_velocity()
        }
    
    def get_predicted_trajectory(self, n_steps=10):
        """Predict the trajectory for n_steps into the future"""
        # Save current state
        saved_x = copy.deepcopy(self.kf.x)
        saved_P = copy.deepcopy(self.kf.P)
        
        trajectory = [self.kf.get_position()]
        
        # Predict forward n_steps
        for _ in range(n_steps):
            self.kf.predict()
            trajectory.append(self.kf.get_position())
        
        # Restore saved state
        self.kf.x = saved_x
        self.kf.P = saved_P
        
        return trajectory

class PersonTrackerManager:
    """
    Manages multiple person trackers
    """
    def __init__(self):
        self.trackers = {}  # Dict of person_id -> PersonTracker
        self.next_id = 0
        self.association_threshold = 1.0  # Maximum distance for data association
    
    def update(self, detections):
        """
        Update trackers with new detections.
        detections: list of {'position': [x,y,z], 'orientation': [roll,pitch,yaw]}
        """
        # First, predict all existing trackers
        for tracker in self.trackers.values():
            tracker.predict()
        
        # Get all active trackers' predicted positions
        active_trackers = {pid: tracker for pid, tracker in self.trackers.items() if tracker.active}
        if not active_trackers:
            # No active trackers, create new ones for all detections
            for detection in detections:
                self._create_new_tracker(detection['position'], detection['orientation'])
            return
        
        # Create a cost matrix for data association
        cost_matrix = np.zeros((len(detections), len(active_trackers)))
        
        for i, detection in enumerate(detections):
            for j, (pid, tracker) in enumerate(active_trackers.items()):
                # Calculate Euclidean distance between detection and tracker prediction
                pred_pos = tracker.kf.get_position()
                dist = np.linalg.norm(detection['position'] - pred_pos)
                cost_matrix[i, j] = dist
        
        # Simple greedy association - associate each detection to closest tracker
        assigned_detections = set()
        assigned_trackers = set()
        
        # Sort all detection-tracker pairs by distance
        indices = np.dstack(np.unravel_index(np.argsort(cost_matrix.ravel()), cost_matrix.shape))[0]
        
        for i, j in indices:
            # If either detection or tracker is already assigned, skip
            if i in assigned_detections or j in assigned_trackers:
                continue
            
            # If distance is too large, skip
            if cost_matrix[i, j] > self.association_threshold:
                continue
            
            # Associate detection i with tracker j
            det = detections[i]
            pid = list(active_trackers.keys())[j]
            self.trackers[pid].update(det['position'], det['orientation'])
            
            assigned_detections.add(i)
            assigned_trackers.add(j)
        
        # Create new trackers for unassigned detections
        for i, detection in enumerate(detections):
            if i not in assigned_detections:
                self._create_new_tracker(detection['position'], detection['orientation'])
        
        # Remove inactive trackers
        self.trackers = {pid: tracker for pid, tracker in self.trackers.items() if tracker.active}
    
    def _create_new_tracker(self, position, orientation=None):
        """Create a new tracker with the given position and orientation"""
        tracker = PersonTracker(self.next_id, position, orientation)
        self.trackers[self.next_id] = tracker
        self.next_id += 1
        return tracker
    
    def get_predictions(self):
        """Get predictions from all active trackers"""
        predictions = {}
        for pid, tracker in self.trackers.items():
            if tracker.active:
                predictions[pid] = tracker.predict()
        return predictions
    
    def get_trajectories(self, n_steps=10):
        """Get predicted trajectories for all active trackers"""
        trajectories = {}
        for pid, tracker in self.trackers.items():
            if tracker.active:
                trajectories[pid] = tracker.get_predicted_trajectory(n_steps)
        return trajectories

class HumanTrajectoryTracker:
    """
    ROS node to track human trajectories using Kalman filtering
    """
    def __init__(self):
        rospy.init_node('human_trajectory_tracker')
        
        # Parameters
        self.tracker_update_rate = rospy.get_param('~tracker_update_rate', 10.0)  # Hz
        self.prediction_horizon = rospy.get_param('~prediction_horizon', 3.0)  # seconds
        self.publish_markers = rospy.get_param('~publish_markers', True)
        self.publish_trajectory = rospy.get_param('~publish_trajectory', True)
        
        # Calculate prediction steps based on update rate and horizon
        self.prediction_steps = int(self.prediction_horizon * self.tracker_update_rate)
        
        # Initialize tracker manager
        self.tracker_manager = PersonTrackerManager()
        
        # Publishers
        self.trajectory_pub = rospy.Publisher('predicted_trajectories', MarkerArray, queue_size=10)
        self.pose_pub = rospy.Publisher('tracked_humans', PoseArray, queue_size=10)
        self.paths_pub = rospy.Publisher('human_paths', Path, queue_size=10)
        
        # Subscribers
        rospy.Subscriber('image_detections', PersonsList, self.personsList_callback)
        rospy.Subscriber('raw_heads', PoseArray, self.heads_callback)
        rospy.Subscriber('raw_bodies', PoseArray, self.bodies_callback)
        
        # Data storage
        self.last_personsList = None
        self.last_heads = None
        self.last_bodies = None
        
        # Timer for tracker updates
        self.update_timer = rospy.Timer(rospy.Duration(1.0/self.tracker_update_rate), self.update_trackers)
        
        rospy.loginfo("Human trajectory tracker initialized")
    
    def personsList_callback(self, msg):
        """Callback for image_detections topic"""
        self.last_personsList = msg
    
    def heads_callback(self, msg):
        """Callback for raw_heads topic"""
        self.last_heads = msg
    
    def bodies_callback(self, msg):
        """Callback for raw_bodies topic"""
        self.last_bodies = msg
    
    def update_trackers(self, event):
        """Update trackers with latest detections"""
        if self.last_personsList is None:
            return
        
        # Process detections from personsList
        detections = []
        
        for person in self.last_personsList.persons:
            # Extract position
            pos = np.array([
                person.body_pose.position.x,
                person.body_pose.position.y,
                person.body_pose.position.z
            ])
            
            # Extract orientation (convert quaternion to euler)
            quat = [
                person.body_pose.orientation.x,
                person.body_pose.orientation.y,
                person.body_pose.orientation.z,
                person.body_pose.orientation.w
            ]
            
            euler = tf.transformations.euler_from_quaternion(quat)
            orient = np.array(euler)
            
            detections.append({
                'position': pos,
                'orientation': orient
            })
        
        # Update trackers with new detections
        self.tracker_manager.update(detections)
        
        # Get predictions and publish
        self._publish_predictions()
    
    def _publish_predictions(self):
        """Publish prediction results"""
        if not self.tracker_manager.trackers:
            return
        
        # Get predictions
        predictions = self.tracker_manager.get_predictions()
        
        # Publish tracked human poses
        if self.pose_pub.get_num_connections() > 0:
            pose_array = PoseArray()
            pose_array.header.stamp = rospy.Time.now()
            pose_array.header.frame_id = "base_footprint"
            
            for pid, pred in predictions.items():
                pose = Pose()
                
                # Position
                pose.position.x = pred['position'][0]
                pose.position.y = pred['position'][1]
                pose.position.z = pred['position'][2]
                
                # Orientation (convert euler to quaternion)
                quat = tf.transformations.quaternion_from_euler(
                    pred['orientation'][0],
                    pred['orientation'][1],
                    pred['orientation'][2]
                )
                pose.orientation.x = quat[0]
                pose.orientation.y = quat[1]
                pose.orientation.z = quat[2]
                pose.orientation.w = quat[3]
                
                pose_array.poses.append(pose)
            
            self.pose_pub.publish(pose_array)
        
        # Publish predicted trajectories as markers
        if self.publish_markers and self.trajectory_pub.get_num_connections() > 0:
            trajectories = self.tracker_manager.get_trajectories(self.prediction_steps)
            
            marker_array = MarkerArray()
            
            for pid, trajectory in trajectories.items():
                # Line marker for trajectory
                line_marker = Marker()
                line_marker.header.frame_id = "base_footprint"
                line_marker.header.stamp = rospy.Time.now()
                line_marker.ns = "human_trajectories"
                line_marker.id = pid
                line_marker.type = Marker.LINE_STRIP
                line_marker.action = Marker.ADD
                line_marker.scale.x = 0.05  # line width
                
                # Set color based on person ID (for differentiation)
                line_marker.color.r = (pid * 100) % 255 / 255.0
                line_marker.color.g = (pid * 50) % 255 / 255.0
                line_marker.color.b = (pid * 150) % 255 / 255.0
                line_marker.color.a = 1.0
                
                # Add trajectory points
                for point in trajectory:
                    p = Point()
                    p.x = point[0]
                    p.y = point[1]
                    p.z = point[2]
                    line_marker.points.append(p)
                
                marker_array.markers.append(line_marker)
                
                # Arrow marker for velocity
                arrow_marker = Marker()
                arrow_marker.header.frame_id = "base_footprint"
                arrow_marker.header.stamp = rospy.Time.now()
                arrow_marker.ns = "human_velocities"
                arrow_marker.id = pid
                arrow_marker.type = Marker.ARROW
                arrow_marker.action = Marker.ADD
                
                # Current position
                start_point = Point()
                start_point.x = trajectory[0][0]
                start_point.y = trajectory[0][1]
                start_point.z = trajectory[0][2]
                
                # Velocity vector
                velocity = predictions[pid]['velocity']
                vel_mag = np.linalg.norm(velocity)
                
                # Only show velocity if magnitude is significant
                if vel_mag > 0.1:
                    end_point = Point()
                    # Scale velocity for visualization
                    scale = 1.0
                    end_point.x = start_point.x + velocity[0] * scale
                    end_point.y = start_point.y + velocity[1] * scale
                    end_point.z = start_point.z + velocity[2] * scale
                    
                    arrow_marker.points.append(start_point)
                    arrow_marker.points.append(end_point)
                    
                    # Set arrow size
                    arrow_marker.scale.x = 0.05  # shaft diameter
                    arrow_marker.scale.y = 0.1   # head diameter
                    arrow_marker.scale.z = 0.2   # head length
                    
                    # Same color as trajectory
                    arrow_marker.color.r = line_marker.color.r
                    arrow_marker.color.g = line_marker.color.g
                    arrow_marker.color.b = line_marker.color.b
                    arrow_marker.color.a = 1.0
                    
                    marker_array.markers.append(arrow_marker)
            
            self.trajectory_pub.publish(marker_array)
        
        # Publish paths for visualization
        if self.publish_trajectory and self.paths_pub.get_num_connections() > 0:
            path = Path()
            path.header.stamp = rospy.Time.now()
            path.header.frame_id = "base_footprint"
            
            for pid, tracker in self.tracker_manager.trackers.items():
                if not tracker.active:
                    continue
                
                # Add historical positions to path
                for i, pos in enumerate(tracker.positions[-20:]):  # Last 20 positions
                    pose_stamped = PoseStamped()
                    pose_stamped.header.stamp = rospy.Time.now() - rospy.Duration(i * 0.1)
                    pose_stamped.header.frame_id = "base_footprint"
                    
                    pose_stamped.pose.position.x = pos[0]
                    pose_stamped.pose.position.y = pos[1]
                    pose_stamped.pose.position.z = pos[2]
                    
                    path.poses.append(pose_stamped)
            
            self.paths_pub.publish(path)

if __name__ == '__main__':
    try:
        tracker = HumanTrajectoryTracker()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass