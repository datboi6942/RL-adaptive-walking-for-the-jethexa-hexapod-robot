# JetHexa Adaptive Locomotion Training

This repository contains a complete setup for training a JetHexa hexapod robot to walk using reinforcement learning in a Gazebo simulation environment, with the ability to deploy trained models to the physical robot running on a Jetson Nano.

## Overview

The system consists of two main ROS packages:

1. **jethexa_description**: Contains the URDF robot description, meshes, and basic configuration needed to simulate JetHexa in Gazebo.

2. **jethexa_gym_env**: Provides the Gym-compatible reinforcement learning environment that wraps around the Gazebo simulation, with reward functions optimized for locomotive behavior.

## Requirements

- Ubuntu 18.04
- ROS Melodic
- Gazebo 9+
- Python 3.6+
- Docker (recommended for containerized setup)

## Quick Start with Docker

1. **Build the Docker image**:

```bash
docker build -t jethexa_rl_training .
```

2. **Run the Docker container** with hardware acceleration and display passthrough:

```bash
docker run --rm -it \
  --gpus all \
  --privileged \
  --net=host \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -e DISPLAY=$DISPLAY \
  jethexa_rl_training
```

3. **Inside the container, build the catkin workspace**:

```bash
cd /catkin_ws
catkin build
source devel/setup.bash
```

## Training the Walking Policy

1. **Launch the training environment**:

```bash
roslaunch jethexa_gym_env train_env.launch
```

2. **In a new terminal in the Docker container, start the training**:

```bash
cd /catkin_ws
source devel/setup.bash
roscd jethexa_gym_env/scripts
python3 train_ppo.py --timesteps 1000000
```

Training will run for 1 million timesteps by default, saving checkpoints at regular intervals.

## Evaluating a Trained Policy

After training, you can evaluate the policy:

```bash
python3 eval_policy.py --model_path /catkin_ws/models/jethexa_ppo_final_model --episodes 5
```

## Exporting for Deployment

To deploy on the Jetson Nano, export your model to ONNX:

```bash
python3 eval_policy.py --model_path /catkin_ws/models/jethexa_ppo_final_model --export_onnx
```

This will create an ONNX file that can be used with TensorRT on the Jetson Nano.

## Reward Function Configuration

Edit `jethexa_gym_env/config/reward_params.yaml` to tune the rewards:

- **forward_velocity**: Rewards for moving forward at target speed
- **stability**: Rewards for maintaining body height and orientation
- **energy_usage**: Penalties for excessive joint movements
- **foot_contact**: Rewards for appropriate foot contact patterns
- **smoothness**: Rewards for smooth joint transitions

## Python 3 with ROS Melodic

This setup uses Python 3 for the reinforcement learning components, while ROS Melodic natively uses Python 2. The Docker environment has been configured to ensure compatibility between the two by making Python 2 ROS modules accessible in Python 3.

## Troubleshooting

- **GPU Issues**: If you have NVIDIA driver problems, consider using the `--gpus all` flag with Docker
- **Display Problems**: Ensure X11 forwarding is correctly set up for Gazebo visualization
- **Training Stability**: Adjust learning rate and batch size if training becomes unstable
- **Python/ROS compatibility**: If you encounter issues with Python 3 importing ROS modules, check that the PYTHONPATH environment variable is correctly set

## Acknowledgments

- Built with Stable-Baselines3 for RL implementations
- Uses ROS Melodic and Gazebo for physics simulation 