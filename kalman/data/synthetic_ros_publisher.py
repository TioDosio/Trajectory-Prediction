import rospy
from geometry_msgs.msg import PoseArray, Pose, Point, Quaternion
import numpy as np
from tf.transformations import quaternion_from_euler

def generate_and_publish_trajectories():
    rospy.init_node("synthetic_trajectory_publisher", anonymous=True)

    head_pub = rospy.Publisher("/raw_heads", PoseArray, queue_size=10)
    body_pub = rospy.Publisher("/raw_bodies", PoseArray, queue_size=10)

    rate = rospy.Rate(1)  # 1 Hz, 1 second interval

    # Trajectory parameters
    duration_seconds = 120
    dt = 1.0
    noise_std_pos = 0.1  # Standard deviation of noise for position (m)
    noise_std_heading = 0.05  # Standard deviation of noise for heading (radians)

    # Ground truth initialization
    pos_x_gt, pos_y_gt = 0.0, 0.0
    vel_x_gt, vel_y_gt = 0.0, 0.0
    heading_gt = 0.0  # radians
    angular_vel_gt = 0.0

    # Define a few trajectory segments (with right turn in the middle)
    trajectory_segments = [
        (0.0, 0.0, 0.0, 8),   # Standing still
        (0.3, 0.0, 0.0, 8),   # Walking straight forward
        (0.3, 0.0, 0.0, 8),   # Continue walking straight
        (0.3, 0.0, -np.deg2rad(10), 8),  # Turn RIGHT in the middle (negative angular velocity)
        (0.3, 0.0, -np.deg2rad(10), 8),  # Continue turning right
        (0.3, 0.0, 0.0, 8),   # Walking straight after turn
        (0.3, 0.0, 0.0, 8),   # Continue walking straight
        (0.6, 0.0, 0.0, 8),   # Accelerating forward
        (0.0, 0.0, 0.0, 8),   # Decelerating to stop
    ]

    current_segment_idx = 0
    segment_step_count = 0

    rospy.loginfo("Starting synthetic trajectory publisher...")

    for i in range(int(duration_seconds / dt)):
        if rospy.is_shutdown():
            break

        # Get current segment targets
        target_vx, target_vy, target_angular_vel, segment_duration = trajectory_segments[current_segment_idx]

        # Simple acceleration/deceleration to target velocities
        alpha = 0.1  # Smoothing factor
        vel_x_gt = (1 - alpha) * vel_x_gt + alpha * target_vx
        vel_y_gt = (1 - alpha) * vel_y_gt + alpha * target_vy
        angular_vel_gt = (1 - alpha) * angular_vel_gt + alpha * target_angular_vel

        # Update ground truth state
        # Convert velocity from body frame to world frame based on current heading
        world_vel_x = vel_x_gt * np.cos(heading_gt) - vel_y_gt * np.sin(heading_gt)
        world_vel_y = vel_x_gt * np.sin(heading_gt) + vel_y_gt * np.cos(heading_gt)
        
        pos_x_gt += world_vel_x * dt
        pos_y_gt += world_vel_y * dt
        heading_gt += angular_vel_gt * dt

        # Add noise to measurements
        measured_pos_x = pos_x_gt + np.random.normal(0, noise_std_pos)
        measured_pos_y = pos_y_gt + np.random.normal(0, noise_std_pos)
        measured_heading = heading_gt + np.random.normal(0, noise_std_heading)

        # Create PoseArray messages
        head_poses_msg = PoseArray()
        head_poses_msg.header.stamp = rospy.Time.now()
        head_poses_msg.header.frame_id = "map"

        body_poses_msg = PoseArray()
        body_poses_msg.header.stamp = rospy.Time.now()
        body_poses_msg.header.frame_id = "map"

        # Create a single Pose for the person
        body_pose = Pose()
        body_pose.position = Point(measured_pos_x, measured_pos_y, 0.0) # Assuming z=0 for body position
        # For body orientation, let's assume it's aligned with the heading for simplicity
        q_body = quaternion_from_euler(0, 0, measured_heading) # Roll, Pitch, Yaw
        body_pose.orientation = Quaternion(q_body[0], q_body[1], q_body[2], q_body[3])
        body_poses_msg.poses.append(body_pose)

        head_pose = Pose()
        head_pose.position = Point(measured_pos_x, measured_pos_y, 1.7) # Assuming head is above body, z=1.7m
        q_head = quaternion_from_euler(0, 0, measured_heading) # Head looks where body walks
        head_pose.orientation = Quaternion(q_head[0], q_head[1], q_head[2], q_head[3])
        head_poses_msg.poses.append(head_pose)

        # Publish messages
        head_pub.publish(head_poses_msg)
        body_pub.publish(body_poses_msg)

        rospy.loginfo("Published data for time: {:.2f}".format(rospy.Time.now().to_sec()))

        segment_step_count += 1
        if segment_step_count >= segment_duration:
            current_segment_idx = (current_segment_idx + 1) % len(trajectory_segments)
            segment_step_count = 0

        rate.sleep()

    rospy.loginfo("Synthetic trajectory publishing finished.")

if __name__ == "__main__":
    try:
        generate_and_publish_trajectories()
    except rospy.ROSInterruptException:
        pass


