# JetHexa Adaptive Locomotion with CPG-RL

This repository implements a hybrid Central Pattern Generator (CPG) and Reinforcement Learning approach for hexapod locomotion, specifically designed for the JetHexa robot. The system learns to generate adaptive gaits that respond to the robot's environment and terrain conditions.

## System Architecture

The implementation uses a dual-component architecture:

1. **CPG Controller**: Generates rhythmic patterns for leg coordination
   - Dynamic pattern generation based on environment feedback
   - Learns to adapt phases and amplitudes in real-time
   - No pre-configured gaits - fully emergent behavior

2. **RL Policy**: Optimizes locomotion parameters
   - Learns to modulate CPG parameters based on terrain and robot state
   - Develops gaits naturally through interaction with environment
   - Optimizes for stability and efficient movement

## Key Features

- **Emergent Behavior**: Gaits emerge naturally from environment interaction rather than being pre-programmed
- **Adaptive Movement**: Movement patterns automatically adjust to terrain and robot state
- **Energy Efficiency**: Penalties for excessive joint movements and power usage
- **Stability Focus**: Exponential stability rewards based on roll and pitch
- **Anti-Rotation**: Strong penalties prevent undesired turning behavior
- **Self-Collision Avoidance**: Proximity-based penalties between leg segments

## Training System

### Environment

- ROS Melodic + Gazebo simulation
- Python 2/3 bridge architecture for compatibility
- Real-time reward computation and state tracking
- Automated checkpoint saving every 20,000 timesteps

### RL Configuration

- Algorithm: PPO (Stable-Baselines3)
- Learning Rate: 3e-5 (fine-tuned for stability)
- Observation Space: Joint positions/velocities, robot pose, IMU data
- Action Space: CPG parameters (frequency, gait type, phases, amplitudes)

### Reward Structure

The reward function combines multiple components:

```python
reward = (
    FORWARD_WEIGHT * forward_movement +      # Forward progress
    STABILITY_WEIGHT * stability_reward +    # Level body maintenance
    HEIGHT_WEIGHT * height_reward +          # Consistent height
    ENERGY_WEIGHT * energy_penalty +         # Power efficiency
    ROTATION_PENALTY_WEIGHT * rotation_penalty + # Anti-turning
    LATERAL_PENALTY_WEIGHT * lateral_penalty +   # Straight-line motion
    proximity_penalties +                    # Collision avoidance
    additional_terms...                      # Other behavioral incentives
)
```

## Usage

### Training

1. **Launch the Environment**:
```bash
roslaunch jethexa_rl train.launch
```

2. **Start Training**:
```bash
python3 scripts/train_ppo.py --timesteps 5000000 --curriculum
```

3. **Monitor Progress**:
- Use TensorBoard to track rewards and learning metrics
- Check terminal output for detailed reward breakdowns
- Monitor gait exploration through action logging

### Evaluation

```bash
src/jethexa_rl/scripts/run_training.sh --timesteps 5000000 --curriculum
```

## Monitoring and Debugging

### TensorBoard Metrics

- Episode rewards
- Policy loss
- Value function loss
- Detailed reward components
- Action distributions

### Logged Components

- Forward/backward movement
- Stability metrics
- Energy usage
- Rotation and lateral deviation
- Collision proximity
- Gait type exploration

## Current Status

The system is implemented with:
- Functional CPG-RL integration
- Stable training pipeline
- Comprehensive reward structure
- Automated checkpointing
- Real-time monitoring

Training typically shows:
1. Initial reward decline as policy unlearns undesired behaviors
2. Gradual improvement as it learns stable gaits
3. Convergence to efficient forward locomotion

## Future Improvements

- Fine-tune reward weights for more natural movement
- Implement adaptive curriculum learning
- Add terrain complexity progression
- Enhance gait transition smoothness
- Optimize energy efficiency further

## Requirements

- Ubuntu 18.04
- ROS Melodic
- Python 2.7 (ROS) and Python 3.6+ (RL)
- NVIDIA GPU recommended for training
- Stable-Baselines3 and dependencies 