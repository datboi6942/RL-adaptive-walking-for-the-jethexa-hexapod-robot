# JetHexa RL Training with Python 2/3 Bridge

This package implements reinforcement learning for the JetHexa hexapod robot in Gazebo simulation, using a bridge architecture to support both ROS 1 (Python 2) and modern RL libraries (Python 3).

## Architecture

The system uses a dual-language architecture to solve the Python 2/3 compatibility issues:

1. **Python 2 ROS Node (Bridge)**: Handles Gazebo simulation, robot control, and terrain generation using ROS 1
2. **Python 3 RL Training**: Implements PPO training with Stable-Baselines3 in Python 3

The two components communicate through ROS topics, allowing each part to run in its optimal environment.

## File Structure

- `launch/train.launch`: Main launch file for starting the Gazebo simulation and bridge
- `scripts/gym_bridge_ros.py`: Python 2 ROS node that interfaces with Gazebo (runs in ROS 1)
- `scripts/train_ppo.py`: Python 3 script that handles RL training using Stable-Baselines3
- `scripts/cpg_controller.py`: Central Pattern Generator for hexapod locomotion
- `scripts/terrain_generator.py`: Generates random terrain for curriculum learning

## Setup Instructions

### Prerequisites

1. ROS 1 (Melodic/Noetic)
2. Gazebo
3. Python 3.6+ with pip
4. Python 2.7 (for ROS 1)

### Python 3 Dependencies

Install the required Python 3 packages:

```bash
pip3 install numpy gym stable-baselines3 matplotlib
```

### Python 2 Dependencies

Make sure ROS dependencies are installed:

```bash
sudo apt-get install python-rospy python-std-msgs python-tf python-geometry-msgs python-gazebo-msgs
```

## Running the Training

### 1. Start the ROS Bridge and Gazebo Simulation

```bash
roslaunch jethexa_rl train.launch
```

This will start:
- Gazebo simulation with the JetHexa robot
- The Python 2 bridge node that interfaces with Gazebo

### 2. Start the RL Training

In a new terminal:

```bash
cd /path/to/catkin_ws
source devel/setup.bash
python3 src/jethexa_rl/scripts/train_ppo.py --train --curriculum
```

Options:
- `--train`: Start a new training session
- `--curriculum`: Enable curriculum learning (gradually increasing difficulty)
- `--timesteps 2000000`: Set the total number of timesteps (default: 1000000)

### 3. Evaluating a Trained Model

```bash
python3 src/jethexa_rl/scripts/train_ppo.py --evaluate --model /path/to/model.zip
```

## Bridge Communication

The bridge uses the following ROS topics for communication:

- From Python 3 (RL) to Python 2 (Gazebo):
  - `/jethexa_rl/action`: Joint positions to apply
  - `/jethexa_rl/reset`: Signal to reset the environment
  - `/jethexa_rl/set_difficulty`: Set the terrain difficulty level

- From Python 2 (Gazebo) to Python 3 (RL):
  - `/jethexa_rl/observation`: Robot state observations
  - `/jethexa_rl/reward`: Reward signal
  - `/jethexa_rl/done`: Episode termination signal
  - `/jethexa_rl/info`: Additional episode information
  - `/jethexa_rl/reset_complete`: Signal that the reset is complete

## Troubleshooting

### ImportError for tf.transformations

If you encounter this error, use the terrain_generator.py which has a custom quaternion_from_euler implementation.

### Timing Issues

If observation/response delays occur, you might need to adjust the timeout values in the Python 3 environment.

### ROS Package Path Issues

Make sure to properly source your ROS workspace before running either script. 