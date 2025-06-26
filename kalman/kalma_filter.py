import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped, Vector3, PoseArray
from tf.transformations import euler_from_quaternion, quaternion_multiply
import message_filters
from visualization_msgs.msg import Marker, MarkerArray

class WalkingDirectionKF:
    def __init__(self):
        # State vector: [x, y, vx, vy, heading, angular_vel]
        self.state = np.zeros(6)
        
        # State covariance matrix
        self.P = np.eye(6)
        self.P[0:2, 0:2] *= 0.05  # position uncertainty
        self.P[2:4, 2:4] *= 0.3  # velocity uncertainty
        self.P[4:6, 4:6] *= 0.2  # orientation uncertainty
        
        # Process noise covariance
        self.Q = np.eye(6)
        self.Q[0:2, 0:2] *= 0.005  # position process noise
        self.Q[2:4, 2:4] *= 0.05   # velocity process noise
        self.Q[4:6, 4:6] *= 0.03  # orientation process noise
        
        # Measurement noise covariance
        self.R = np.eye(6)
        self.R[0:2, 0:2] *= 0.2   # position measurement noise
        self.R[2:4, 2:4] *= 0.3   # velocity measurement noise
        self.R[4:6, 4:6] *= 0.1   # orientation measurement noise
        
        # State transition matrix
        self.dt = 0.1  # time step
        self.F = np.eye(6)
        self.F[0, 2] = self.dt  # x += vx * dt
        self.F[1, 3] = self.dt  # y += vy * dt
        self.F[4, 5] = self.dt  # heading += angular_vel * dt
        
        # Measurement matrix
        self.H = np.eye(6)
        
    def predict(self):
        # Predict state
        self.state = np.dot(self.F, self.state)
        
        # Predict covariance
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        
        return self.state
        
    def update(self, measurement):
        # Kalman gain
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        
        # Update state
        y = measurement - np.dot(self.H, self.state)
        self.state = self.state + np.dot(K, y)
        
        # Update covariance
        I = np.eye(len(self.state))
        self.P = np.dot(I - np.dot(K, self.H), self.P)
        
        return self.state

class HumanTracker:
    def __init__(self, person_id):
        self.id = person_id
        self.kf = WalkingDirectionKF()
        self.last_update_time = rospy.Time.now()
        self.active = True
        self.missed_frames = 0
        self.position_history = []
        self.MAX_MISSED_FRAMES = 10
        
        # Add motion state classification
        self.motion_state = "unknown"  # Can be "standing", "walking", "running"
        self.velocity_history = []
        self.velocity_window = 10  # frames
    
    def update(self, body_pose, head_pose):
        current_time = rospy.Time.now()
        dt = (current_time - self.last_update_time).to_sec()
        
        # Update kalman filter time step
        if dt > 0:
            self.kf.dt = dt
            self.kf.F[0, 2] = dt
            self.kf.F[1, 3] = dt
            self.kf.F[4, 5] = dt
        
        # Extract position from body pose
        pos_x = body_pose.position.x
        pos_y = body_pose.position.y
        
        # Extract orientation from head pose
        orientation_q = head_pose.orientation
        roll, pitch, yaw = euler_from_quaternion([
            orientation_q.x, orientation_q.y, 
            orientation_q.z, orientation_q.w
        ])
        
        # Calculate velocity if we have a state
        if np.any(self.kf.state[0:2] != 0):
            vx = (pos_x - self.kf.state[0]) / dt
            vy = (pos_y - self.kf.state[1]) / dt
        else:
            vx = 0
            vy = 0
        
        # Calculate angular velocity
        if self.kf.state[4] != 0:
            angular_vel = (yaw - self.kf.state[4]) / dt
        else:
            angular_vel = 0
        
        # Create measurement vector [x, y, vx, vy, heading, angular_vel]
        measurement = np.array([pos_x, pos_y, vx, vy, yaw, angular_vel])
        
        # Update Kalman filter
        self.kf.update(measurement)
        
        # Store velocity for classification
        if len(self.velocity_history) >= self.velocity_window:
            self.velocity_history.pop(0)
        self.velocity_history.append((vx, vy))

        print("Current velocity history:", self.velocity_history)
        # Classify motion state
        if len(self.velocity_history) >= 5:
            avg_speed = np.mean([np.sqrt(v[0]**2 + v[1]**2) for v in self.velocity_history])
            if avg_speed < 0.2:
                self.motion_state = "standing"
                # Use lower process noise for position and velocity
                self.kf.Q[0:2, 0:2] *= 0.5  # Less position uncertainty
                self.kf.Q[2:4, 2:4] *= 0.2  # Much less velocity uncertainty
            elif avg_speed < 1.2:
                self.motion_state = "walking"
                # Use normal process noise
                self.kf.Q[0:2, 0:2] = np.eye(2) * 0.01
                self.kf.Q[2:4, 2:4] = np.eye(2) * 0.1
            else:
                self.motion_state = "running"
                # Use higher process noise for velocity
                self.kf.Q[2:4, 2:4] *= 1.5  # More velocity uncertainty
        
        # Update time
        self.last_update_time = current_time
        self.missed_frames = 0
        
        # Store position in history
        self.position_history.append((pos_x, pos_y))
        if len(self.position_history) > 30:  # Keep only last 30 positions
            self.position_history.pop(0)
    
    def predict(self):
        # Predict next state
        state = self.kf.predict()
        return state
    
    def mark_missing(self):
        self.missed_frames += 1
        if self.missed_frames > self.MAX_MISSED_FRAMES:
            self.active = False
        return self.active

class HumanTrackingNode:
    def __init__(self):
        rospy.init_node('human_tracking_node')
        
        # Initialize human trackers dictionary
        self.trackers = {}
        self.next_id = 0
        
        # Parameters
        self.max_association_distance = rospy.get_param('~max_association_distance', 1.0)  # meters
        self.prediction_steps = rospy.get_param('~prediction_steps', 5)
        self.prediction_dt = rospy.get_param('~prediction_dt', 1)  # seconds
        
        # Publishers
        self.trajectory_pub = rospy.Publisher('/predicted_trajectories', MarkerArray, queue_size=10)
        self.direction_pub = rospy.Publisher('/predicted_walking_direction', Vector3, queue_size=10)
        
        # Subscribers using message filters for synchronization
        self.head_sub = message_filters.Subscriber('/raw_heads', PoseArray)
        self.body_sub = message_filters.Subscriber('/raw_bodies', PoseArray)
        
        # Synchronize messages
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.head_sub, self.body_sub], 10, 0.2)
        self.ts.registerCallback(self.poses_callback)
        
        rospy.loginfo("Human tracking node initialized")
    
    def poses_callback(self, head_poses_msg, body_poses_msg):
        current_trackers = set()
        
        # Match detections to existing trackers
        for i in range(len(body_poses_msg.poses)):
            body_pose = body_poses_msg.poses[i]
            head_pose = head_poses_msg.poses[i]
            print("Processing body pose:", body_pose, "and head pose:", head_pose)
            # Find closest tracker
            min_dist = float('inf')
            best_tracker_id = None
            
            for tracker_id, tracker in self.trackers.items():
                if not tracker.active:
                    continue
                
                dist = np.sqrt((body_pose.position.x - tracker.kf.state[0])**2 + 
                              (body_pose.position.y - tracker.kf.state[1])**2)
                
                if dist < min_dist and dist < self.max_association_distance:
                    min_dist = dist
                    best_tracker_id = tracker_id
            
            # Update existing tracker or create new one
            if best_tracker_id is not None:
                self.trackers[best_tracker_id].update(body_pose, head_pose)
                current_trackers.add(best_tracker_id)
            else:
                new_id = self.next_id
                self.next_id += 1
                self.trackers[new_id] = HumanTracker(new_id)
                self.trackers[new_id].update(body_pose, head_pose)
                current_trackers.add(new_id)
        
        # Mark trackers without matches as missing
        for tracker_id, tracker in list(self.trackers.items()):
            if tracker_id not in current_trackers:
                active = tracker.mark_missing()
                if not active:
                    del self.trackers[tracker_id]
        
        # Predict trajectories and publish visualization
        self.predict_and_visualize()
    
    def predict_and_visualize(self):
        marker_array = MarkerArray()
        
        for tracker_id, tracker in self.trackers.items():
            if not tracker.active:
                continue
            
            # Create path marker for past positions
            past_marker = Marker()
            past_marker.header.frame_id = "base_footprint"
            past_marker.header.stamp = rospy.Time.now()
            past_marker.ns = "past_trajectories"
            past_marker.id = tracker_id
            past_marker.type = Marker.LINE_STRIP
            past_marker.action = Marker.ADD
            past_marker.scale.x = 0.05  # line width
            past_marker.color.r = 0.0
            past_marker.color.g = 0.7
            past_marker.color.b = 0.0
            past_marker.color.a = 0.8
            past_marker.lifetime = rospy.Duration(1.0)
            
            for pos in tracker.position_history:
                p = Vector3()
                p.x = pos[0]
                p.y = pos[1]
                p.z = 0.1  # Slightly above ground
                past_marker.points.append(p)
            
            marker_array.markers.append(past_marker)
            
            # Create future trajectory marker
            future_marker = Marker()
            future_marker.header.frame_id = "base_footprint"
            future_marker.header.stamp = rospy.Time.now()
            future_marker.ns = "future_trajectories"
            future_marker.id = tracker_id
            future_marker.type = Marker.LINE_STRIP
            future_marker.action = Marker.ADD
            future_marker.scale.x = 0.05  # line width
            future_marker.color.r = 0.8
            future_marker.color.g = 0.1
            future_marker.color.b = 0.1
            future_marker.color.a = 0.8
            future_marker.lifetime = rospy.Duration(1.0)
            
            # Start with current position
            current_state = tracker.kf.state.copy()
            p = Vector3()
            p.x = current_state[0]
            p.y = current_state[1]
            p.z = 0.1  # Slightly above ground
            future_marker.points.append(p)
            
            # Predict future positions
            for i in range(self.prediction_steps):
                # Save current dt
                original_dt = tracker.kf.dt
                
                # Set prediction time step
                tracker.kf.dt = self.prediction_dt
                tracker.kf.F[0, 2] = self.prediction_dt
                tracker.kf.F[1, 3] = self.prediction_dt
                tracker.kf.F[4, 5] = self.prediction_dt
                
                # Predict
                predicted_state = tracker.kf.predict()
                
                # Add to marker
                p = Vector3()
                p.x = predicted_state[0]
                p.y = predicted_state[1]
                p.z = 0.1  # Slightly above ground
                future_marker.points.append(p)
                
                # Restore original state for real-time tracking
                tracker.kf.state = current_state.copy()
                tracker.kf.dt = original_dt
                tracker.kf.F[0, 2] = original_dt
                tracker.kf.F[1, 3] = original_dt
                tracker.kf.F[4, 5] = original_dt
            
            marker_array.markers.append(future_marker)
            
            # Publish walking direction for the closest person to the robot
            # (this is a simple heuristic - you might want a more sophisticated approach)
            if len(self.trackers) > 0:
                closest_dist = float('inf')
                closest_state = None
                
                for tr in self.trackers.values():
                    if not tr.active:
                        continue
                    dist = np.sqrt(tr.kf.state[0]**2 + tr.kf.state[1]**2)
                    if dist < closest_dist:
                        closest_dist = dist
                        closest_state = tr.kf.state
                
                if closest_state is not None:
                    direction_msg = Vector3()
                    direction_msg.x = closest_state[2]  # vx
                    direction_msg.y = closest_state[3]  # vy
                    direction_msg.z = 0.0
                    self.direction_pub.publish(direction_msg)
        
        # Publish markers
        if len(marker_array.markers) > 0:
            self.trajectory_pub.publish(marker_array)

if __name__ == '__main__':
    try:
        node = HumanTrackingNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass