# Trajectory-Prediction

A comprehensive trajectory prediction system for pedestrians using Kalman filtering and skeleton tracking. This repository implements advanced techniques for predicting human movement patterns based on real-time sensor data from the Vizzy robot platform.

## Overview

This project focuses on predicting the future trajectory of pedestrians by analyzing their past movements using data from ROS (Robot Operating System) messages. The system combines computer vision-based skeleton tracking with probabilistic filtering techniques to provide accurate and robust trajectory predictions.

### Key Features

- **Multi-modal Tracking**: Combines body position and head orientation data for comprehensive human pose estimation
- **Kalman Filtering**: Implements Extended Kalman Filter (EKF) for robust state estimation and prediction
- **Real-time Processing**: Designed for real-time trajectory prediction in robotics applications
- **ROS Integration**: Seamless integration with ROS ecosystem for robot navigation and human-robot interaction
- **Visualization Tools**: Interactive visualization of skeleton data and predicted trajectories
- **Synthetic Data Generation**: Tools for generating test data and evaluating prediction performance

## Repository Structure

### 📁 `kalman/`

Contains the core Kalman filtering implementations for trajectory prediction:

- **`human_trajectory_kalman_filter.py`** - Main Kalman filter implementation for human trajectory prediction with body pose and head orientation tracking
- **`kalma_filter.py`** - Basic Kalman filter implementation and utilities
- **`kalman_w_keypoints_pos_Horient.py`** - Enhanced Kalman filter incorporating keypoint positions and head orientation data

#### 📁 `kalman/data/`

Data generation and handling utilities:

- **`generate_synthetic_data.py`** - Generates synthetic walking trajectory data with realistic noise patterns
- **`synthetic_ros_publisher.py`** - ROS publisher for synthetic trajectory data testing
- **`synthetic_walking_data.csv`** - Pre-generated synthetic dataset for testing and validation

### 📁 `skeleton/`

Visualization and skeleton tracking components:

- **`skeleton_visualizer.py`** - ROS-based visualization tool for displaying body keypoints and skeleton data
- **`skeleton_visualizer-v2.py`** - Enhanced version of the skeleton visualizer with additional features

## Technical Approach

### Trajectory Prediction Pipeline

1. **Data Acquisition**: Receives human pose data from Vizzy robot's perception system via ROS messages
2. **State Estimation**: Uses Extended Kalman Filter to estimate current position, velocity, and orientation
3. **Motion Modeling**: Implements constant velocity and constant angular velocity motion models
4. **Prediction**: Projects future trajectory based on current state estimate and motion dynamics
5. **Visualization**: Real-time display of tracked skeleton and predicted trajectory

### State Representation

The system tracks the following state variables:

- **Body Position**: 2D position (x, y) on the ground plane
- **Body Velocity**: 2D velocity vector (vx, vy)
- **Head Orientation**: 3D orientation quaternion for head pose
- **Angular Velocity**: Rotational velocity of head orientation

### Kalman Filter Implementation

- **Process Model**: Constant velocity motion model with acceleration noise
- **Measurement Model**: Direct observation of body keypoints and head pose
- **Noise Handling**: Adaptive noise parameters based on keypoint confidence scores
- **State Prediction**: Multi-step ahead trajectory prediction capabilities

## Dependencies

- **Python 3.x**
- **NumPy** - Numerical computations
- **OpenCV** - Computer vision operations
- **SciPy** - Scientific computing (rotation utilities)
- **ROS (Robot Operating System)** - Communication framework
- **Pandas** - Data manipulation and analysis
- **human_awareness_msgs** - Custom ROS message types for human detection

## Usage

### Running the Trajectory Predictor

```bash
# Launch the main trajectory prediction node
rosrun trajectory_prediction human_trajectory_kalman_filter.py
```

### Visualizing Skeleton Data

```bash
# Start the skeleton visualizer
rosrun trajectory_prediction skeleton_visualizer.py
```

### Generating Synthetic Data

```python
from kalman.data.generate_synthetic_data import generate_synthetic_walking_data

# Generate 60 seconds of walking data
data = generate_synthetic_walking_data(duration_seconds=60, dt=0.1)
```

## Applications

- **Robot Navigation**: Predictive path planning for mobile robots in human environments
- **Human-Robot Interaction**: Anticipatory behavior for collaborative robots
- **Surveillance Systems**: Automated tracking and behavior analysis
- **Autonomous Vehicles**: Pedestrian trajectory prediction for safety systems
- **Smart Environments**: Context-aware systems in public spaces

## Future Enhancements

- Integration of machine learning models for improved prediction accuracy
- Multi-person tracking and trajectory prediction
- Integration with SLAM systems for enhanced spatial understanding
- Real-time parameter adaptation based on environment conditions

## Contributing

This project is part of ongoing research in human trajectory prediction and robotics. Contributions and improvements are welcome.

## License

This project is developed for research purposes. Please refer to the license file for usage terms.
