import numpy as np
import pandas as pd

def generate_synthetic_walking_data(duration_seconds=60, dt=1.0, noise_std_pos=0.1, noise_std_vel=0.05, noise_std_heading=0.05):
    """
    Generates synthetic walking data for a person, including body position and head orientation.
    The head orientation is assumed to be in the direction of walking.

    Args:
        duration_seconds (int): Total duration of the simulation in seconds.
        dt (float): Time step for data acquisition in seconds.
        noise_std_pos (float): Standard deviation of noise for position (m).
        noise_std_vel (float): Standard deviation of noise for velocity (m/s).
        noise_std_heading (float): Standard deviation of noise for heading (radians).

    Returns:
        pd.DataFrame: DataFrame containing 'time', 'pos_x', 'pos_y', 'vel_x', 'vel_y', 'heading', 'angular_vel' (ground truth)
                      and 'measured_pos_x', 'measured_pos_y', 'measured_heading' (noisy measurements).
    """
    times = np.arange(0, duration_seconds, dt)
    num_steps = len(times)

    # Ground truth initialization
    pos_x_gt, pos_y_gt = 0.0, 0.0
    vel_x_gt, vel_y_gt = 0.0, 0.0
    heading_gt = 0.0  # radians
    angular_vel_gt = 0.0

    data = []

    # Define a few trajectory segments
    # Each segment has (target_vx, target_vy, target_angular_vel, duration_steps)
    trajectory_segments = [
        # Standing still
        (0.0, 0.0, 0.0, 10),
        # Walking straight
        (0.5, 0.0, 0.0, 10),
        # Turning left while walking
        (0.5, 0.0, np.deg2rad(5), 10),
        # Walking straight again
        (0.5, 0.0, 0.0, 10),
        # Turning right while walking
        (0.5, 0.0, -np.deg2rad(5), 10),
        # Accelerating forward
        (1.0, 0.0, 0.0, 10),
        # Decelerating to stop
        (0.0, 0.0, 0.0, 10),
    ]

    current_segment_idx = 0
    segment_step_count = 0

    for i in range(num_steps):
        # Get current segment targets
        target_vx, target_vy, target_angular_vel, segment_duration = trajectory_segments[current_segment_idx]

        # Simple acceleration/deceleration to target velocities
        alpha = 0.1 # Smoothing factor
        vel_x_gt = (1 - alpha) * vel_x_gt + alpha * target_vx
        vel_y_gt = (1 - alpha) * vel_y_gt + alpha * target_vy
        angular_vel_gt = (1 - alpha) * angular_vel_gt + alpha * target_angular_vel

        # Update ground truth state
        pos_x_gt += vel_x_gt * dt
        pos_y_gt += vel_y_gt * dt
        heading_gt += angular_vel_gt * dt

        # Add noise to measurements
        measured_pos_x = pos_x_gt + np.random.normal(0, noise_std_pos)
        measured_pos_y = pos_y_gt + np.random.normal(0, noise_std_pos)
        measured_heading = heading_gt + np.random.normal(0, noise_std_heading)

        data.append({
            'time': times[i],
            'pos_x_gt': pos_x_gt,
            'pos_y_gt': pos_y_gt,
            'vel_x_gt': vel_x_gt,
            'vel_y_gt': vel_y_gt,
            'heading_gt': heading_gt,
            'angular_vel_gt': angular_vel_gt,
            'measured_pos_x': measured_pos_x,
            'measured_pos_y': measured_pos_y,
            'measured_heading': measured_heading
        })

        segment_step_count += 1
        if segment_step_count >= segment_duration:
            current_segment_idx = (current_segment_idx + 1) % len(trajectory_segments)
            segment_step_count = 0

    return pd.DataFrame(data)

if __name__ == '__main__':
    synthetic_data = generate_synthetic_walking_data(duration_seconds=120, dt=1.0)
    synthetic_data.to_csv('synthetic_walking_data.csv', index=False)
    print("Synthetic data generated and saved to 'synthetic_walking_data.csv'")
    print(synthetic_data.head())


