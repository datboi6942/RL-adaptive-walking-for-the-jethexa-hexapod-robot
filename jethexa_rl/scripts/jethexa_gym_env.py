#!/usr/bin/env python3
import sys
import os

import rospy
import gym
import numpy as np
import time
import os
from std_srvs.srv import Empty
from sensor_msgs.msg import JointState, Imu
from std_msgs.msg import Float64
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import Twist, Pose, Point, Quaternion
# Replace tf.transformations with a direct implementation
import math

# Define our own euler_from_quaternion function since tf isn't Python3 compatible
def euler_from_quaternion(quaternion):
    """
    Convert a quaternion to Euler angles (roll, pitch, yaw).
    
    Args:
        quaternion: [x, y, z, w]
    
    Returns:
        roll, pitch, yaw
    """
    x, y, z, w = quaternion
    
    # Roll (x-axis rotation)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    
    # Pitch (y-axis rotation)
    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = math.copysign(math.pi / 2, sinp)  # Use 90 degrees if out of range
    else:
        pitch = math.asin(sinp)
    
    # Yaw (z-axis rotation)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    
    return (roll, pitch, yaw)

# Import the CPG controller and terrain generator
from cpg_controller import CPGController
from terrain_generator import TerrainGenerator

class JetHexaEnv(gym.Env):
    """
    Gym environment for JetHexa robot in Gazebo.
    
    This environment uses a Central Pattern Generator (CPG) for controlling
    the robot's gait, and supports curriculum learning with progressively 
    more difficult terrains.
    """
    def __init__(self):
        super(JetHexaEnv, self).__init__()
        rospy.init_node('jethexa_gym_env', anonymous=True)
        
        # Initialize the CPG controller
        self.cpg_controller = CPGController(n_legs=6, n_joints_per_leg=3)
        
        # Initialize the terrain generator for curriculum learning
        self.terrain_generator = TerrainGenerator()
        self.current_difficulty = 0
        
        # Joint publishers for controlling the robot
        self.joint_publishers = [
            rospy.Publisher(f'/jethexa/joint{i}_position_controller/command', Float64, queue_size=1)
            for i in range(18)
        ]
        
        # Subscribe to joint states for observation
        self.joint_states = np.zeros(18)
        self.joint_velocities = np.zeros(18)
        rospy.Subscriber('/joint_states', JointState, self.joint_state_cb)
        
        # Subscribe to model states for robot position and orientation
        self.robot_pose = np.zeros(6)  # x, y, z, roll, pitch, yaw
        self.prev_position = np.zeros(3)
        self.start_position = np.zeros(3)
        rospy.Subscriber('/gazebo/model_states', ModelStates, self.model_state_cb)
        
        # Subscribe to IMU data for additional observation
        self.imu_data = np.zeros(9)  # linear acceleration (3), angular velocity (3), orientation (3)
        rospy.Subscriber('/jethexa/imu', Imu, self.imu_cb)
        
        # Set up simulation reset service
        self.reset_sim = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
        self.pause_physics = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.unpause_physics = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        
        # Domain randomization parameters
        self.use_domain_randomization = True
        
        # Define CPG parameter space (for RL action space)
        # Parameters: [global_freq, gait_type, leg_phases(6), joint_amplitudes(18)]
        self.n_cpg_params = 2 + 6 + 18
        
        # Define observation and action spaces
        # Observation: joint positions, joint velocities, robot pose, IMU data
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(18 + 18 + 6 + 9,), dtype=np.float32
        )
        
        # Action: CPG parameters
        self.action_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(self.n_cpg_params,), dtype=np.float32
        )
        
        # Initialize episode parameters
        self.episode_steps = 0
        self.max_episode_steps = 1000
        self.episode_count = 0
        self.total_reward = 0
        
        # Performance tracking
        self.distance_traveled = 0
        self.falls_count = 0
        self.energy_used = 0
        self.prev_position = np.zeros(3) # Keep track of previous position
        self.prev_orientation = np.zeros(3) # Keep track of previous orientation (roll, pitch, yaw)
        
        # Set up logging directory
        self.log_dir = os.path.join(os.path.dirname(__file__), "../logs")
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Initialize time tracking
        self.last_step_time = None
        self.sim_time = 0
        
        # Wait for connections to establish
        rospy.sleep(1.0)
        rospy.loginfo("JetHexaEnv initialized")

    def joint_state_cb(self, msg):
        """Update joint states from sensor feedback."""
        for i, name in enumerate(msg.name):
            if i < 18:  # Only use the first 18 joints
                self.joint_states[i] = msg.position[i]
                self.joint_velocities[i] = msg.velocity[i]
                
                # Track energy usage (approximated by sum of squared velocities)
                self.energy_used += abs(msg.velocity[i]) * 0.01

    def model_state_cb(self, msg):
        """Update robot pose from Gazebo model states."""
        if 'jethexa' in msg.name:
            idx = msg.name.index('jethexa')
            pos = msg.pose[idx].position
            ori = msg.pose[idx].orientation
            
            # Store position
            self.robot_pose[0] = pos.x
            self.robot_pose[1] = pos.y
            self.robot_pose[2] = pos.z
            
            # Convert quaternion to Euler angles
            euler = euler_from_quaternion([ori.x, ori.y, ori.z, ori.w])
            self.robot_pose[3] = euler[0]  # roll
            self.robot_pose[4] = euler[1]  # pitch
            self.robot_pose[5] = euler[2]  # yaw
            
            # Detect falls (robot is tilted too much or too low to ground)
            if abs(euler[0]) > 0.7 or abs(euler[1]) > 0.7 or pos.z < 0.05:
                self.falls_count += 1

    def imu_cb(self, msg):
        """Update IMU data for additional observation."""
        # Linear acceleration
        self.imu_data[0] = msg.linear_acceleration.x
        self.imu_data[1] = msg.linear_acceleration.y
        self.imu_data[2] = msg.linear_acceleration.z
        
        # Angular velocity
        self.imu_data[3] = msg.angular_velocity.x
        self.imu_data[4] = msg.angular_velocity.y
        self.imu_data[5] = msg.angular_velocity.z
        
        # Orientation (euler angles)
        q = [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w]
        euler = euler_from_quaternion(q)
        self.imu_data[6] = euler[0]  # roll
        self.imu_data[7] = euler[1]  # pitch
        self.imu_data[8] = euler[2]  # yaw

    def randomize_physics(self):
        """Apply domain randomization to physics parameters."""
        if not self.use_domain_randomization:
            return
            
        # This would call a service to modify physics parameters in Gazebo
        # For this implementation, we'll log the intent
        mass_scale = 1.0 + np.random.uniform(-0.1, 0.1)  # ±10% mass variation
        friction_scale = 1.0 + np.random.uniform(-0.2, 0.2)  # ±20% friction variation
        rospy.loginfo(f"Domain randomization: mass_scale={mass_scale:.2f}, friction_scale={friction_scale:.2f}")
        
        # In a full implementation, this would call Gazebo services to modify parameters
        # e.g., change link masses, friction coefficients, etc.

    def step(self, action):
        """
        Take a step in the environment using the CPG controller.
        
        Args:
            action: Numpy array of CPG parameters
                   [global_freq, gait_type, leg_phases, joint_amplitudes]
        
        Returns:
            observation, reward, done, info
        """
        # Update time tracking
        current_time = time.time()
        if self.last_step_time is None:
            dt = 0.1
        else:
            dt = current_time - self.last_step_time
        self.sim_time += dt
        self.last_step_time = current_time
        
        # Set CPG parameters from action and generate joint commands
        self.cpg_controller.set_gait_params(action)
        joint_positions = self.cpg_controller.update(self.sim_time)
        
        # Apply joint commands to robot
        for i in range(18):
            self.joint_publishers[i].publish(Float64(joint_positions[i]))
        
        # Small delay to let simulation step
        rospy.sleep(0.1)
        
        # Increment step counter
        self.episode_steps += 1
        
        # Construct observation
        obs = np.concatenate([
            self.joint_states,
            self.joint_velocities,
            self.robot_pose,
            self.imu_data
        ])
        
        # Calculate reward
        reward = self.compute_reward()
        self.total_reward += reward
        
        # Check if episode is done
        done = self.episode_steps >= self.max_episode_steps
        
        # Check for falls
        if abs(self.robot_pose[3]) > 0.7 or abs(self.robot_pose[4]) > 0.7:
            # Robot has fallen over
            reward -= 10  # Penalty for falling
            done = True
        
        # Update distance traveled
        current_pos = np.array(self.robot_pose[:3])
        step_distance = current_pos[0] - self.prev_position[0]  # x-axis distance
        self.distance_traveled += max(0, step_distance)  # Only count forward movement
        
        # Update previous position and orientation *after* computing reward components that use them
        self.prev_position = current_pos.copy()
        self.prev_orientation = np.array(self.robot_pose[3:6]) # Store current roll, pitch, yaw
        
        # Additional info
        info = {
            'robot_pose': self.robot_pose,
            'distance_traveled': self.distance_traveled,
            'falls': self.falls_count,
            'energy_used': self.energy_used,
            'difficulty_level': self.current_difficulty
        }
        
        return obs, reward, done, info

    def compute_reward(self):
        """
        Calculate reward based on movement, stability, and energy efficiency.
        Reward function is shaped to encourage:
        1. Forward movement (x-axis)
        2. Stability (penalize excessive roll and pitch)
        3. Energy efficiency (penalize high joint velocities)
        4. Height maintenance (staying off the ground)
        """
        # Current position and orientation
        current_pos = np.array(self.robot_pose[:3])
        current_roll, current_pitch, current_yaw = self.robot_pose[3:6] # Get current RPY
        
        # Forward movement reward (primary objective)
        forward_movement = current_pos[0] - self.prev_position[0]  # x-axis movement
        # Calculate forward reward (increased from 15.0 to 30.0 for stronger speed incentive)
        forward_reward = 30.0 * forward_movement
        
        # Stability reward (penalize roll and pitch deviations to stay upright)
        roll, pitch = self.robot_pose[3], self.robot_pose[4]
        stability_cost = (abs(roll) + abs(pitch)) * 2.0
        stability_reward = 1.0 - min(1.0, stability_cost)  # Normalize to [0,1]
        
        # Height maintenance (reward for keeping body elevated)
        target_height = 0.15  # Target height for body
        height_diff = abs(current_pos[2] - target_height)
        height_reward = 1.0 - min(1.0, height_diff / 0.15)  # Normalize
        
        # Energy efficiency (penalize large joint velocities)
        energy_cost = np.sum(np.square(self.joint_velocities)) * 0.001
        energy_penalty = -min(1.0, energy_cost)  # Cap the penalty
        
        # Lateral movement penalty (discourage sideways drift)
        lateral_movement = abs(current_pos[1] - self.prev_position[1])
        # Increased penalty magnitude (applied in final sum)
        lateral_penalty = -lateral_movement * 5.0
        
        # Rotation penalty (discourage turning)
        prev_roll, prev_pitch, prev_yaw = self.prev_orientation # Get previous RPY
        delta_yaw = current_yaw - prev_yaw
        # Handle angle wrapping around +/- pi
        if delta_yaw > np.pi:
            delta_yaw -= 2 * np.pi
        elif delta_yaw < -np.pi:
            delta_yaw += 2 * np.pi
        # Increased penalty magnitude (applied in final sum)
        rotation_penalty = -abs(delta_yaw) * 5.0
        
        # Combined reward - Adjusted weights
        reward = (
            forward_reward +           # Forward progress (primary goal)
            0.5 * stability_reward +   # Stay upright (Weight unchanged)
            0.3 * height_reward +      # Maintain height (Weight unchanged)
            0.4 * energy_penalty +     # Energy efficiency (Weight increased)
            2.0 * lateral_penalty +    # Minimize sideways drift (Weight increased from 0.8)
            3.0 * rotation_penalty     # Minimize turning/rotation (Weight significantly increased from 1.5)
        )
        
        # Scale reward based on current difficulty level
        # Higher difficulty levels get higher rewards to encourage learning
        difficulty_bonus = 1.0 + (self.current_difficulty * 0.2)
        
        return reward * difficulty_bonus

    def reset(self):
        """
        Reset the environment for a new episode.
        
        This includes:
        1. Resetting the simulation
        2. Generating new terrain based on current difficulty
        3. Randomizing physics parameters (if enabled)
        4. Resetting internal state variables
        
        Returns:
            Initial observation
        """
        try:
            # Pause physics during reset for stability
            self.pause_physics()
            
            # Reset the simulation
            self.reset_sim()
            
            # Generate new terrain based on current difficulty
            self.terrain_generator.set_difficulty(self.current_difficulty)
            self.terrain_generator.reset_terrain()
            
            # Apply domain randomization
            self.randomize_physics()
            
            # Unpause physics
            self.unpause_physics()
        except rospy.ServiceException as e:
            rospy.logerr(f"Reset service call failed: {e}")
        
        # Reset time tracking
        self.last_step_time = None
        self.sim_time = 0
        
        # Reset episode counters and stats
        self.episode_steps = 0
        self.episode_count += 1
        self.total_reward = 0
        self.distance_traveled = 0
        self.energy_used = 0
        
        # Wait for simulation to stabilize
        rospy.sleep(1.0)
        
        # Reset CPG controller state
        self.cpg_controller = CPGController()
        
        # Store initial position for reference
        self.start_position = np.array(self.robot_pose[:3])
        self.prev_position = self.start_position.copy()
        # Initialize previous orientation based on initial pose
        # Assuming robot_pose[3:6] contains roll, pitch, yaw
        self.prev_orientation = np.array(self.robot_pose[3:6]) 
        
        # Log episode start
        rospy.loginfo(f"Starting episode {self.episode_count} at difficulty {self.current_difficulty}")
        
        # Return initial observation
        obs = np.concatenate([
            self.joint_states,
            self.joint_velocities,
            self.robot_pose,
            self.imu_data
        ])
        
        return obs

    def render(self, mode='human'):
        """
        Render the environment.
        
        For Gazebo simulation, visualization is handled separately.
        This method could be extended to record videos or capture screenshots.
        """
        pass

    def close(self):
        """Cleanup resources."""
        # Reset terrain to flat before closing
        try:
            self.terrain_generator.set_difficulty(0)
            self.terrain_generator.reset_terrain()
        except:
            pass
            
        rospy.signal_shutdown("Environment closed")

    def set_difficulty(self, level):
        """
        Set the terrain difficulty level for curriculum learning.
        
        Args:
            level: Integer 0-4 indicating difficulty level
        """
        self.current_difficulty = max(0, min(4, level))
        rospy.loginfo(f"Setting environment difficulty to level {self.current_difficulty}")
        
        # Update terrain if this is called mid-episode
        if self.episode_steps > 0:
            self.terrain_generator.set_difficulty(self.current_difficulty)
            self.terrain_generator.reset_terrain()
            
    def get_reward_components(self):
        """
        Calculate and return individual reward components for visualization/debugging.
        
        Returns:
            Dictionary of reward components
        """
        current_pos = np.array(self.robot_pose[:3])
        current_roll, current_pitch, current_yaw = self.robot_pose[3:6] # Get current RPY
        
        # Forward movement
        forward_movement = current_pos[0] - self.prev_position[0]
        
        # Stability
        roll, pitch = self.robot_pose[3], self.robot_pose[4]
        stability = 1.0 - (abs(roll) + abs(pitch)) / 2.0
        
        # Height 
        height_diff = abs(current_pos[2] - 0.15)
        height = 1.0 - min(1.0, height_diff / 0.15)
        
        # Energy
        energy = -np.sum(np.square(self.joint_velocities)) * 0.001
        
        # Lateral drift
        lateral_drift = -abs(current_pos[1] - self.prev_position[1])
        
        # Yaw change (for rotation penalty visualization)
        delta_yaw = current_yaw - self.prev_orientation[2]
        if delta_yaw > np.pi: delta_yaw -= 2 * np.pi
        elif delta_yaw < -np.pi: delta_yaw += 2 * np.pi
        rotation = -abs(delta_yaw) # Base penalty value
        
        return {
            'forward': forward_movement * 10.0,
            'stability': stability * 0.5,
            'height': height * 0.3,
            'energy': energy * 0.2,
            'lateral': lateral_drift * 0.3,
            'rotation': rotation * 0.5, # Visualize the weighted rotation component
            'total': self.compute_reward()
        }


if __name__ == '__main__':
    # Test environment if run directly
    env = JetHexaEnv()
    obs = env.reset()
    print(f"Observation shape: {obs.shape}")
    print(f"Action space: {env.action_space}")
    
    # Define a fixed, symmetrical action (e.g., basic forward gait)
    # Parameters: [global_freq, gait_type, leg_phases(6), joint_amplitudes(18)]
    # Note: Adjust these values based on expected ranges for your CPG
    fixed_action = np.zeros(env.action_space.shape[0])
    fixed_action[0] = 0.5  # Global frequency (example value)
    fixed_action[1] = 0.0  # Gait type (e.g., 0 for tripod)
    # Symmetrical phases (example: alternating tripod)
    fixed_action[2:8] = np.array([0.0, 0.5, 0.0, 0.5, 0.0, 0.5]) 
    # Symmetrical amplitudes (example: moderate forward movement)
    # Example: Set all shoulder forward/backward amplitudes the same
    # Example: Set all hip up/down amplitudes the same
    # Example: Set all knee bend amplitudes the same
    # Indices depend on your CPG implementation order (shoulder, hip, knee?)
    # Assuming S-H-K order for joints 0-2, 3-5, ..., 15-17
    amp_shoulder = 0.2 # Example amplitude
    amp_hip = 0.3      # Example amplitude
    amp_knee = 0.3     # Example amplitude
    for i in range(6): # For each leg
        base_idx = 2 + 6 + i * 3
        fixed_action[base_idx + 0] = amp_shoulder # Shoulder
        fixed_action[base_idx + 1] = amp_hip      # Hip
        fixed_action[base_idx + 2] = amp_knee     # Knee

    print(f"Using fixed symmetrical action: {np.round(fixed_action, 2)}")

    # Step with the fixed action
    for i in range(100):
        # Use the FIXED action instead of random sampling
        obs, reward, done, info = env.step(fixed_action) 
        reward_components = env.get_reward_components()
        
        print(f"Step {i}, Reward: {reward:.3f}, Done: {done}")
        # Include pose for checking rotation
        pose = info.get('robot_pose', np.zeros(6))
        print(f"  Pose (x,y,yaw): {pose[0]:.3f}, {pose[1]:.3f}, {pose[5]:.3f}") 
        print(f"  Forward: {reward_components['forward']:.3f}, Stability: {reward_components['stability']:.3f}, Rotation: {reward_components['rotation']:.3f}") 
        print(f"  Distance: {info['distance_traveled']:.3f}, Falls: {info['falls']}")
        
        if done:
            print("Episode finished")
            # Optional: break or reset with fixed action again
            # obs = env.reset()
            break 
    
    env.close() 