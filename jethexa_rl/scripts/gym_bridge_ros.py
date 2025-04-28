#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from __future__ import unicode_literals
from io import open
import six

import sys
import os
import rospy
import numpy as np
import math
import time
import os
from std_srvs.srv import Empty
from sensor_msgs.msg import JointState, Imu
from std_msgs.msg import Float64, Float32MultiArray, Bool, Int32
from gazebo_msgs.msg import ModelStates, ModelState, LinkStates, ODEPhysics
from gazebo_msgs.srv import SetModelState, GetPhysicsProperties, SetPhysicsProperties
from gazebo_msgs.srv import SetModelConfiguration
from geometry_msgs.msg import Pose, Point, Quaternion, Twist
from tf.transformations import euler_from_quaternion, quaternion_from_euler
import logging
import traceback
from datetime import datetime

# Import the local module for terrain generation
from terrain_generator import TerrainGenerator
# <<< ADDED: Import the CPGControl service definition
from jethexa_rl.srv import CPGControl, CPGControlRequest

# Set up logging
log_dir = os.path.join(os.path.dirname(__file__), "../logs")
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

log_file = os.path.join(log_dir, "gym_bridge_{}.log".format(datetime.now().strftime('%Y%m%d_%H%M%S')))
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)

def log_exception(logger, e, context=""):
    """Helper function to log exceptions with full traceback."""
    logger.error("{} Exception: {}".format(context, str(e)))
    logger.error("Traceback:\n{}".format(traceback.format_exc()))

class JetHexaGymBridge:
    """
    ROS-based bridge between Gazebo and the Python 3 RL framework.
    
    This node handles the interface with the simulated robot in Gazebo using Python 2,
    while communicating with the RL training algorithm in Python 3 through ROS topics.
    """
    def __init__(self):
        try:
            rospy.init_node('jethexa_gym_bridge', anonymous=True)
            logging.info("Initializing JetHexa Gym Bridge (Python 2)")
            
            # Training parameters
            self.max_episode_steps = 1000
            self.total_timesteps = 0
            self.episode_steps = 0
            self.episode_count = 0
            self.total_reward = 0
            self.distance_traveled = 0
            self.falls_count = 0
            self.energy_used = 0
            self.last_step_time = None
            self.sim_time = 0
            
            # Initialize the terrain generator for curriculum learning
            self.terrain_generator = TerrainGenerator()
            self.current_difficulty = 0
            
            # Define the correct order of joint names
            joint_names = [
                'coxa_joint_LF', 'femur_joint_LF', 'tibia_joint_LF',
                'coxa_joint_LM', 'femur_joint_LM', 'tibia_joint_LM',
                'coxa_joint_LR', 'femur_joint_LR', 'tibia_joint_LR',
                'coxa_joint_RF', 'femur_joint_RF', 'tibia_joint_RF',
                'coxa_joint_RM', 'femur_joint_RM', 'tibia_joint_RM',
                'coxa_joint_RR', 'femur_joint_RR', 'tibia_joint_RR',
            ] # Total 18 joints
            self.joint_names = joint_names # <<< ADDED: Store joint names for later use

            # Joint publishers for controlling the robot using correct names
            self.joint_publishers = [
                rospy.Publisher('/jethexa/{}_position_controller/command'.format(name), Float64, queue_size=1)
                for name in joint_names
            ]
            # Create a mapping from joint name to index for state arrays
            self.joint_name_to_index = {name: i for i, name in enumerate(joint_names)}
            
            # --- New: Default Stance --- 
            # Define the "sprawled" ready pose (replace with your specific values in radians)
            self.default_joint_positions = [
                0.5,  0.6, -1.2,   # LF: coxa, femur, tibia
                0.5,  0.6, -1.2,   # LM
                0.5,  0.6, -1.2,   # LR
               -0.5,  0.6, -1.2,   # RF (mirror sign if needed)
               -0.5,  0.6, -1.2,   # RM
               -0.5,  0.6, -1.2    # RR
            ]
            # --- End Default Stance ---

            # State tracking
            self.joint_states = np.zeros(18)
            self.joint_velocities = np.zeros(18)
            self.robot_pose = np.zeros(6)  # x, y, z, roll, pitch, yaw
            self.prev_position = np.zeros(3)
            self.start_position = np.zeros(3)
            self.prev_orientation = np.zeros(3) # Store previous RPY
            self.imu_data = np.zeros(9)  # linear acceleration (3), angular velocity (3), orientation (3)
            self.current_action = None # <<< ADDED: Store the latest action received
            self.last_state_update_time = None # <<< ADDED: Track time for dt calculation
            self.prev_action = None
            self.link_poses = {} # <<< Initialize link poses dictionary
            
            # --- NEW: Initialize reward tracking for 2-second window ---
            self.reward_history = {
                'forward': [],
                'backward': [],
                'stability': [],
                'angular_velocity': [],
                'height': [],
                'energy': [],
                'lateral': [],
                'rotation': [],
                'orientation': [],
                'action_rate': [],
                'collision': [],
                'survival': [],
                'fall': [],
                'dynamic_collision': [],
                'bounce': [],
                'dynamic_energy': [],
                'total': []
            }
            self.last_reward_log_time = rospy.get_time()
            self.reward_log_interval = 2.0  # Log every 2 seconds
            # --- END NEW ---
            
            # Connect to simulation services
            logging.info("Bridge: Connecting to Gazebo services...")
            try:
                rospy.wait_for_service('/gazebo/reset_simulation', timeout=5.0)
                rospy.wait_for_service('/gazebo/pause_physics', timeout=5.0)
                rospy.wait_for_service('/gazebo/unpause_physics')
                rospy.wait_for_service('/gazebo/set_model_state')
                self.reset_sim = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
                self.pause_physics = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
                self.unpause_physics = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
                self.set_model_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
                # Add service proxies for getting and setting physics properties
                rospy.wait_for_service('/gazebo/get_physics_properties', timeout=5.0)
                self.get_physics_properties = rospy.ServiceProxy('/gazebo/get_physics_properties', GetPhysicsProperties)
                rospy.wait_for_service('/gazebo/set_physics_properties', timeout=5.0)
                self.set_physics_properties = rospy.ServiceProxy('/gazebo/set_physics_properties', SetPhysicsProperties)
                # <<< ADDED: Service proxy for reset_world >>>
                rospy.wait_for_service('/gazebo/reset_world', timeout=5.0)
                self.reset_world = rospy.ServiceProxy('/gazebo/reset_world', Empty)
            except Exception as e:
                log_exception(logging, e, "Failed to connect to Gazebo services")
                raise
            
            # Episode tracking
            self.episode_steps = 0
            self.episode_count = 0
            self.total_reward = 0
            self.distance_traveled = 0
            self.falls_count = 0
            self.energy_used = 0
            self.last_step_time = None
            self.sim_time = 0
            
            # Domain randomization
            self.use_domain_randomization = rospy.get_param('~domain_randomization', True)
            # --- New: dynamic ramp settings ---
            self.ramp_duration = rospy.get_param('~ramp_duration', 500000) # Use integer literal
            self.global_step   = 0
            # Base weights for the three new penalties
            self.BASE_W_COL    = 2.0      # self-collision
            self.BASE_W_BNC    = 0.5      # vertical bounce
            self.BASE_W_EN     = 0.01     # energy penalty
            # --- Initialize state vars needed for new penalties ---
            self.self_col_count = 0     # Updated by collision checks (if implemented)
            self.base_twist = None      # Updated by link_state_cb
            self.last_joint_effort = np.zeros(18) # <<< Initialized as numpy array
            self.dt = 0.0               # Updated by action_cb
            self.sim_step_dt = 0.02     # <<< ADDED: Assumed simulation step time (e.g., for 50Hz)
            
            # Subscribers for robot state
            rospy.Subscriber('/joint_states', JointState, self.joint_state_cb)
            rospy.Subscriber('/gazebo/link_states', LinkStates, self.link_state_cb)
            rospy.Subscriber('/jethexa/imu', Imu, self.imu_cb)
            
            # Bridge publishers (to Python 3 training script)
            self.obs_pub = rospy.Publisher('/jethexa_rl/observation', Float32MultiArray, queue_size=1)
            self.reward_pub = rospy.Publisher('/jethexa_rl/reward', Float64, queue_size=1)
            self.done_pub = rospy.Publisher('/jethexa_rl/done', Bool, queue_size=1)
            self.info_pub = rospy.Publisher('/jethexa_rl/info', Float32MultiArray, queue_size=1)
            self.reset_complete_pub = rospy.Publisher('/jethexa_rl/reset_complete', Bool, queue_size=1)
            
            # Bridge subscribers (from Python 3 training script)
            rospy.Subscriber('/jethexa_rl/action', Float32MultiArray, self.action_cb)
            rospy.Subscriber('/jethexa_rl/reset', Bool, self.reset_cb)
            rospy.Subscriber('/jethexa_rl/set_difficulty', Int32, self.set_difficulty_cb)
            
            # Set up logging directory
            self.log_dir = os.path.join(os.path.dirname(__file__), "../logs")
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)
            # Setup robot position logging file
            self.robot_position_log_path = os.path.join(self.log_dir, "robot_position_log.csv")
            try:
                with open(self.robot_position_log_path, 'w') as f:
                    header = "timestamp,x,y,z\n"
                    f.write(header)
                logging.info("Initialized robot position log at: {}".format(self.robot_position_log_path))
            except Exception as e:
                log_exception(logging, e, "Failed to initialize robot position log file")
            
            # Wait for connections to establish
            rospy.sleep(1.0)
            logging.info("JetHexa Gym Bridge initialized")

            # ROS Service Clients
            rospy.wait_for_service('/gazebo/set_physics_properties', timeout=5.0)
            self.set_physics_client = rospy.ServiceProxy('/gazebo/set_physics_properties', SetPhysicsProperties)
            rospy.wait_for_service('/gazebo/get_physics_properties', timeout=5.0)
            self.get_physics_client = rospy.ServiceProxy('/gazebo/get_physics_properties', GetPhysicsProperties)
            # Added CPG Service Client (ensure it's available)
            try:
                rospy.wait_for_service('/cpg_control', timeout=3.0) 
                self.cpg_service_client = rospy.ServiceProxy('/cpg_control', CPGControl)
                logging.info("Successfully connected to /cpg_control service.")
            except (rospy.ROSException, rospy.ServiceException) as e:
                logging.warning("Could not connect to /cpg_control service: {}. CPG features disabled.".format(e))
                self.cpg_service_client = None # Set to None if not available

            # Robot configuration
            self.robot_name = rospy.get_param('~robot_name', 'jethexa')
            self.initial_height = rospy.get_param('~initial_height', 0.18) # Default height
            
            # Add metrics tracking
            self.metrics = {
                'episode_rewards': [],
                'episode_lengths': [],
                'forward_velocities': [],
                'stability_scores': [],
                'energy_usage': [],
                'fall_counts': [],
                'total_distance': [],
                'avg_roll': [],
                'avg_pitch': [],
                'avg_yaw': [],
                'success_rate': []  # Percentage of episodes without falls
            }
            
            # Metrics logging parameters
            self.metrics_log_interval = 100  # Log every 100 episodes
            self.metrics_window_size = 100   # Rolling window for averages
            self.metrics_file = os.path.join(self.log_dir, "training_metrics.csv")
            
            # Initialize metrics file with headers
            with open(self.metrics_file, 'w') as f:
                headers = [
                    'episode', 'total_timesteps', 'mean_reward', 'mean_episode_length',
                    'mean_forward_velocity', 'mean_stability', 'mean_energy',
                    'fall_rate', 'total_distance', 'mean_roll', 'mean_pitch',
                    'mean_yaw', 'success_rate'
                ]
                f.write(','.join(headers) + '\n')
                
        except Exception as e:
            log_exception(logging, e, "Failed to initialize JetHexaGymBridge")
            raise

    def joint_state_cb(self, msg):
        """Update joint states from sensor feedback."""
        # Iterate through received joints and update state array based on name matching
        for i, name in enumerate(msg.name):
            if name in self.joint_name_to_index:
                idx = self.joint_name_to_index[name]
                # Ensure index is within bounds (should be if name matches)
                if idx < len(self.joint_states):
                    self.joint_states[idx] = msg.position[i]
                    self.joint_velocities[idx] = msg.velocity[i]
                    
                    # --- Attempt to update effort ---
                    if hasattr(msg, 'effort') and len(msg.effort) == len(msg.name):
                         self.last_joint_effort[idx] = msg.effort[i]
                    elif i == 0: # Log warning only once per message if effort is wrong/missing
                         # Check if effort field is missing entirely or just empty/wrong size
                         if not hasattr(msg, 'effort'):
                              rospy.logwarn_throttle(10, "joint_state_cb: msg does not contain 'effort' field. Cannot calculate energy penalty.")
                         elif len(msg.effort) != len(msg.name):
                              rospy.logwarn_throttle(10, "joint_state_cb: msg 'effort' field length ({}) does not match name/position length ({}). Cannot use for energy penalty.".format(len(msg.effort), len(msg.name)))
                         # If field exists but is empty and length is 0 (and names exist), this implies effort is not published
                         elif len(msg.effort) == 0 and len(msg.name) > 0:
                             rospy.logwarn_throttle(10, "joint_state_cb: msg 'effort' field is empty. Cannot calculate energy penalty.")

                    # --- ADDED: Debug velocity data ---
                    # Log only occasionally to avoid spamming
                    if i == 0 and np.random.rand() < 0.01: # Log first joint's data 1% of the time
                         rospy.logdebug("joint_state_cb: Received velocities (sample): {}".format(np.round(msg.velocity, 3)))
                         rospy.logdebug("joint_state_cb: Stored self.joint_velocities (sample): {}".format(np.round(self.joint_velocities, 3)))
                    # --- End Debug ---

                    # Track energy usage (using the correctly mapped velocity)
                    # self.energy_used += abs(msg.velocity[i]) * 0.01 # Moved energy calculation to compute_reward if needed

    def link_state_cb(self, msg):
        """Update robot pose, base twist, and handle reward calculation/publishing."""
        
        current_update_time = rospy.get_time()
        dt = 0.0
        if self.last_state_update_time is not None:
            dt = current_update_time - self.last_state_update_time
        self.last_state_update_time = current_update_time

        # --- Update robot pose and link poses (as before) ---
        target_links = [
            'jethexa::tibia_LF', 'jethexa::tibia_LM', 'jethexa::tibia_LR',
            'jethexa::tibia_RF', 'jethexa::tibia_RM', 'jethexa::tibia_RR',
            'jethexa::femur_LF', 'jethexa::femur_LM', 'jethexa::femur_LR',
            'jethexa::femur_RF', 'jethexa::femur_RM', 'jethexa::femur_RR',
            'jethexa::coxa_LF', 'jethexa::coxa_LM', 'jethexa::coxa_LR',
            'jethexa::coxa_RF', 'jethexa::coxa_RM', 'jethexa::coxa_RR',
        ]
        temp_link_poses = {}
        found_base = False
        base_pose_updated = False # Flag if base pose was found in this message
        current_robot_pose = self.robot_pose.copy() # Keep a copy of the pose before update
        
        for i, name in enumerate(msg.name):
            if name == 'jethexa::base_link':
                pos = msg.pose[i].position
                ori = msg.pose[i].orientation
                twist = msg.twist[i]
                self.robot_pose[0] = pos.x
                self.robot_pose[1] = pos.y
                self.robot_pose[2] = pos.z
                euler = euler_from_quaternion([ori.x, ori.y, ori.z, ori.w])
                self.robot_pose[3] = euler[0]
                self.robot_pose[4] = euler[1]
                self.robot_pose[5] = euler[2]
                self.base_twist = twist
                # <<< ADDED: Log base twist linear z when updated >>>
                rospy.logdebug("LinkStateCB: Updated self.base_twist.linear.z = {:.4f}".format(self.base_twist.linear.z))
                # <<< END ADDED >>>
                found_base = True
                base_pose_updated = True
            elif name in target_links:
                pos = msg.pose[i].position
                short_name = name.split('::')[-1]
                temp_link_poses[short_name] = np.array([pos.x, pos.y, pos.z])
                
        self.link_poses = temp_link_poses
        if not found_base:
             rospy.logwarn_throttle(5, "link_state_cb: Did not find 'jethexa::base_link' in message.")
             return # Don't proceed if base pose wasn't updated

        # --- Execute Step Logic ONLY if an action is pending and base pose was updated ---        
        if self.current_action is not None and base_pose_updated:
            rospy.logdebug("LinkStateCB: Processing step for pending action.")
            action_to_process = self.current_action.copy() # Process this action
            self.current_action = None # Clear pending action BEFORE potential errors/returns
            
            try:
                # Increment step counter
                self.episode_steps += 1
                
                # Construct observation (using the LATEST state updated above)
                obs = np.concatenate([
                    self.joint_states,
                    self.joint_velocities,
                    self.robot_pose, # Use the newly updated pose
                    self.imu_data
                ])
                
                # Calculate reward (pass the processed action and calculated dt)
                # compute_reward uses self.robot_pose (new) and self.prev_position (old)
                reward = self.compute_reward(action=action_to_process, dt=dt) 
                self.total_reward += reward
                
                # Check if episode is done (timeout or fall)
                timeout_done = self.episode_steps >= self.max_episode_steps
                is_fallen = abs(self.robot_pose[3]) > 1.5 or abs(self.robot_pose[4]) > 1.5 # Increased from 0.7 to 1.5 to be more lenient
                # The FALL_PENALTY is added within compute_reward, just check for termination here
                done = timeout_done or is_fallen
                
                # Log why the episode is done
                if done:
                    if timeout_done:
                        rospy.logwarn("Episode done due to timeout: {} steps".format(self.episode_steps))
                    if is_fallen:
                        rospy.logwarn("Episode done due to fall: roll={:.2f}, pitch={:.2f}".format(
                            abs(self.robot_pose[3]), abs(self.robot_pose[4])))
                
                # Update distance traveled (using NEW and OLD positions)
                current_pos_for_reward = np.array(self.robot_pose[:3]) # Use the NEW pose
                step_distance = current_pos_for_reward[0] - self.prev_position[0]  # Compare NEW X to OLD X
                self.distance_traveled += max(0, step_distance) # Only count forward movement
                
                # --- Log Robot Position (as before) ---
                try:
                    with open(self.robot_position_log_path, 'a') as f:
                        log_line = "{:.4f},{:.4f},{:.4f},{:.4f}\n".format(
                            current_update_time, 
                            current_pos_for_reward[0], # x
                            current_pos_for_reward[1], # y
                            current_pos_for_reward[2]  # z
                        )
                        f.write(log_line)
                except Exception as e:
                    rospy.logerr("ERROR writing to robot position log file: {}".format(e))
                # --- End Log Robot Position ---
                
                # Publish observation
                obs_msg = Float32MultiArray()
                obs_msg.data = obs.astype(np.float32).tolist()
                rospy.logdebug("LinkStateCB: Publishing observation...")
                self.obs_pub.publish(obs_msg)
                
                # Publish reward
                self.reward_pub.publish(Float64(reward))
                
                # Publish done
                self.done_pub.publish(Bool(done))
                
                # Publish info
                info_msg = Float32MultiArray()
                info_data = [
                    self.distance_traveled,
                    float(self.falls_count),
                    self.energy_used,
                    float(self.current_difficulty)
                ]
                info_msg.data = info_data
                self.info_pub.publish(info_msg)
                
                # Update previous state *using the state just processed*
                self.prev_position = current_pos_for_reward.copy() 
                self.prev_orientation = np.array([self.robot_pose[3], self.robot_pose[4], self.robot_pose[5]])
                self.prev_action = action_to_process.copy() 

                # If episode is done, log completion
                if done:
                    rospy.loginfo("Episode {} finished with total reward: {:.2f}".format(
                        self.episode_count, self.total_reward))
                    
                    # Update metrics for episode completion
                    self.metrics['episode_rewards'].append(self.total_reward)
                    self.metrics['episode_lengths'].append(self.episode_steps)
                    
                    # Increment falls count if fallen
                    if is_fallen:
                        self.falls_count += 1
                        
                    # Log metrics if we've reached the interval
                    if len(self.metrics['episode_rewards']) % self.metrics_log_interval == 0:
                        window = min(self.metrics_window_size, len(self.metrics['episode_rewards']))
                        if window > 0: # Avoid division by zero if window is 0
                            recent_rewards = self.metrics['episode_rewards'][-window:]
                            recent_lengths = self.metrics['episode_lengths'][-window:]
                            recent_velocities = self.metrics['forward_velocities'][-window:]
                            recent_stability = self.metrics['stability_scores'][-window:]
                            recent_energy = self.metrics['energy_usage'][-window:]
                            recent_roll = self.metrics['avg_roll'][-window:]
                            recent_pitch = self.metrics['avg_pitch'][-window:]
                            recent_yaw = self.metrics['avg_yaw'][-window:]

                            # Calculate statistics
                            mean_reward = np.mean(recent_rewards)
                            mean_length = np.mean(recent_lengths)
                            mean_velocity = np.mean(recent_velocities)
                            mean_stability = np.mean(recent_stability)
                            mean_energy = np.mean(recent_energy)
                            # Use self.falls_count for overall rate, not just window
                            fall_rate = float(self.falls_count) / max(1, self.episode_count)  # Overall fall rate
                            success_rate = 1.0 - fall_rate
                            mean_roll = np.mean(recent_roll)
                            mean_pitch = np.mean(recent_pitch)
                            mean_yaw = np.mean(recent_yaw)

                            # Write metrics to file
                            with open(self.metrics_file, 'a') as f:
                                metrics_line = [
                                    str(self.episode_count),
                                    str(self.total_timesteps),
                                    "{:.3f}".format(mean_reward),
                                    "{:.1f}".format(mean_length),
                                    "{:.3f}".format(mean_velocity),
                                    "{:.3f}".format(mean_stability),
                                    "{:.3f}".format(mean_energy),
                                    "{:.3f}".format(fall_rate),
                                    "{:.2f}".format(self.distance_traveled), # Cumulative distance
                                    "{:.3f}".format(mean_roll),
                                    "{:.3f}".format(mean_pitch),
                                    "{:.3f}".format(mean_yaw),
                                    "{:.3f}".format(success_rate)
                                ]
                                f.write(','.join(metrics_line) + '\n')

                            # Log metrics to ROS
                            rospy.loginfo("\n=== Training Metrics (Last {} episodes) ===".format(window))
                            rospy.loginfo("Episode: {}, Total Timesteps: {}".format(self.episode_count, self.total_timesteps))
                            rospy.loginfo("Mean Reward: {:.3f}".format(mean_reward))
                            rospy.loginfo("Mean Episode Length: {:.1f}".format(mean_length))
                            rospy.loginfo("Mean Forward Velocity: {:.3f}".format(mean_velocity))
                            rospy.loginfo("Mean Stability Score: {:.3f}".format(mean_stability))
                            rospy.loginfo("Mean Energy Usage: {:.3f}".format(mean_energy))
                            rospy.loginfo("Overall Fall Rate: {:.3f}".format(fall_rate))
                            rospy.loginfo("Success Rate: {:.3f}".format(success_rate))
                            rospy.loginfo("Mean Roll: {:.3f}".format(mean_roll))
                            rospy.loginfo("Mean Pitch: {:.3f}".format(mean_pitch))
                            rospy.loginfo("Mean Yaw: {:.3f}".format(mean_yaw))
                            rospy.loginfo("Total Distance: {:.2f}".format(self.distance_traveled))
                            rospy.loginfo("=====================================\n")
                        
            except Exception as e:
                 rospy.logerr("Error during step processing in link_state_cb: {}".format(e))
                 # Consider publishing done=True or handling error state
                 self.current_action = None # Ensure action is cleared on error
                 # Maybe publish default/error observation/reward? 
                 self.done_pub.publish(Bool(True)) # Signal done on error

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

    def action_cb(self, msg):
        """ Store action received from Python 3 RL script and apply joint commands."""
        try:
            action = np.array(msg.data)
            self.current_action = action.copy() # Store the latest action
            rospy.logdebug("ActionCB: Received Action via Topic: {}".format(np.round(action[:6], 3))) # Log first 6 values
            
            # Apply joint commands immediately
            for i in range(min(len(action), 18)):
                try:
                    command = Float64(action[i]) # Use the received action directly
                    if i < 3:
                         rospy.logdebug("    Pub Joint {}: {:.3f}".format(i, command.data))
                    elif i == 3:
                         rospy.logdebug("    ... (publishing remaining joints) ...")
                    self.joint_publishers[i].publish(command)
                except Exception as e:
                     rospy.logerr("ActionCB: ERROR publishing joint command for index {}: {}".format(i, e))
                     return # Exit if we can't publish commands
            
            # Wait a small amount of time for commands to be processed
            rospy.sleep(0.02) # Wait for 20ms (50Hz control rate)
            
            # Get current state for immediate observation
            observation = self._get_observation()
            
            # Publish observation immediately after action is applied
            obs_msg = Float32MultiArray()
            obs_msg.data = observation.astype(np.float32).tolist()
            self.obs_pub.publish(obs_msg)
            
            # Calculate and publish reward
            reward = self.compute_reward(action=self.current_action)
            self.reward_pub.publish(Float64(reward))
            
            # Check if episode is done
            done = self.episode_steps >= self.max_episode_steps or \
                   abs(self.robot_pose[3]) > 1.5 or abs(self.robot_pose[4]) > 1.5
            self.done_pub.publish(Bool(done))
            
            # Publish info
            info_msg = Float32MultiArray()
            info_data = [
                self.distance_traveled,
                float(self.falls_count),
                self.energy_used,
                float(self.current_difficulty)
            ]
            info_msg.data = info_data
            self.info_pub.publish(info_msg)
            
        except Exception as e:
            rospy.logerr("ActionCB: Error processing action: {}".format(e))
            import traceback
            rospy.logerr(traceback.format_exc())

    def reset_cb(self, msg):
        """Handle reset request from Python 3 RL training script."""
        rospy.loginfo("Bridge: Reset callback triggered with value: {}".format(msg.data))
        if msg.data:
            try:
                # Call the main reset logic. Reset complete signal is now sent from within self.reset()
                rospy.loginfo("Bridge: Calling reset() function...")
                self.reset() 
                rospy.loginfo("Bridge: reset() function returned successfully.")
            except Exception as e:
                 rospy.logerr("Bridge: Error occurred during reset_cb processing: {}".format(e))
                 # Optionally publish a failure signal or handle differently
        else:
             rospy.loginfo("Bridge: Reset callback received False, not resetting.")

    def set_difficulty_cb(self, msg):
        """Handle difficulty level change request."""
        level = msg.data
        self.set_difficulty(level)

    def randomize_physics(self):
        """Randomize physics properties like friction, damping, etc."""
        if not self.use_domain_randomization:
            return

        rospy.logdebug("Randomizing physics properties...")
        try:
            # Get current physics properties
            props = self.get_physics_client()

            # Generate randomized values ONLY for ERP and CFM
            new_erp = np.random.uniform(0.1, 0.3)
            new_cfm = np.random.uniform(0.0, 0.01)
            
            # Log the changes being applied
            rospy.logdebug("Applying randomized ERP: {:.4f} (was {:.4f}), CFM: {:.4f} (was {:.4f})"\
                         .format(new_erp, props.ode_config.erp, new_cfm, props.ode_config.cfm))

            # Set new physics properties, keeping others unchanged from 'props'
            self.set_physics_properties(
                time_step=props.time_step,
                max_update_rate=props.max_update_rate,
                gravity=props.gravity,
                ode_config=ODEPhysics(
                    auto_disable_bodies=props.ode_config.auto_disable_bodies,
                    sor_pgs_precon_iters=props.ode_config.sor_pgs_precon_iters,
                    sor_pgs_iters=props.ode_config.sor_pgs_iters,
                    sor_pgs_w=props.ode_config.sor_pgs_w,
                    sor_pgs_rms_error_tol=props.ode_config.sor_pgs_rms_error_tol,
                    erp=new_erp, # Apply randomized ERP
                    cfm=new_cfm, # Apply randomized CFM
                    # --- Keep these unchanged from props --- #
                    contact_surface_layer=props.ode_config.contact_surface_layer,
                    contact_max_correcting_vel=props.ode_config.contact_max_correcting_vel,
                    max_contacts=props.ode_config.max_contacts
                )
            )
            # rospy.loginfo("Physics properties randomized (ERP, CFM only).") # Log if needed
        except (rospy.ServiceException, rospy.ROSException) as e:
            rospy.logwarn("Failed to get/set physics properties: {}".format(e))
        except Exception as e:
             rospy.logerr("Unexpected error during physics randomization: {}".format(e))

    def reset(self):
        """Reset the environment to initial state."""
        # Perform reset attempts
        success = False
        for attempt in range(1, 4):
            try:
                rospy.loginfo("Reset attempt {}/3...".format(attempt))
                if self._reset_robot(): # <<< If reset sequence is successful...
                    success = True
                    # Reset episode-specific state
                    self.episode_steps = 0
                    self.total_reward = 0
                    self.distance_traveled = 0
                    self.energy_used = 0
                    self.last_step_time = None
                    # Reset metrics accumulators if needed (e.g., per-episode stability)
                    # self.some_per_episode_metric = 0 
                    
                    self.episode_count += 1 # Increment AFTER successful reset
                    logging.info("Starting episode {}".format(self.episode_count))

                    # <<< MOVED: Publish reset complete AFTER potential observation send >>>
                    # --- ADDED: Fetch and publish initial observation ---
                    try:
                        initial_observation = self._get_observation()
                        obs_msg = Float32MultiArray()
                        obs_msg.data = initial_observation.astype(np.float32).tolist()
                        rospy.logdebug("Reset: Publishing initial observation...")
                        self.obs_pub.publish(obs_msg)
                        rospy.logdebug("Reset: Initial observation published.")
                    except Exception as obs_e:
                        log_exception(logging, obs_e, "Failed to get/publish initial observation after reset")
                        success = False # Mark reset as failed if we can't get obs
                    # --- END ADDED ---

                    self.publish_reset_complete(success)
                    break # Exit retry loop on success
                else:
                    rospy.logwarn("Reset attempt {} failed. Retrying...".format(attempt))
                    rospy.sleep(2.0) # Wait before retrying
            except Exception as e:
                log_exception(logging, e, "Error during reset attempt {}".format(attempt))
                rospy.sleep(2.0)

        if not success:
            rospy.logerr("Failed to reset robot after multiple attempts.")
            # Publish failure signal if all attempts failed
            self.publish_reset_complete(False)

    def publish_reset_complete(self, success):
        """Publish the reset completion status."""
        try:
            rospy.logdebug("Bridge: Publishing reset_complete signal: {}".format(success))
            self.reset_complete_pub.publish(Bool(success))
            rospy.logdebug("Bridge: Reset complete signal published successfully")
        except Exception as e:
            log_exception(logging, e, "Failed to publish reset_complete signal")

    def set_difficulty(self, level):
        """Set the terrain difficulty level for curriculum learning."""
        self.current_difficulty = max(0, min(4, level))
        rospy.loginfo("Setting environment difficulty to level {}".format(self.current_difficulty))
        
        # Update terrain if this is called mid-episode
        if self.episode_steps > 0:
            self.terrain_generator.set_difficulty(self.current_difficulty)
            self.terrain_generator.reset_terrain()

    def _track_reward_components(self, components):
        """Track individual reward components and output them periodically."""
        if not hasattr(self, '_last_reward_log_time'):
            self._last_reward_log_time = rospy.get_time()
            self._reward_components = {
                'forward_reward': 0.0,
                'backward_penalty': 0.0,
                'stability_reward': 0.0,
                'angular_vel_penalty': 0.0,
                'height_reward': 0.0,
                'energy_penalty': 0.0,
                'lateral_penalty': 0.0,
                'rotation_penalty': 0.0,
                'orientation_penalty': 0.0,
                'action_rate_penalty': 0.0,
                'proximity_penalty': 0.0,
                'survival_reward': 0.0,
                'fall_penalty': 0.0,
                'dyn_col_term': 0.0,
                'dyn_bnc_term': 0.0,
                'dyn_en_term': 0.0
            }
            return

        # Update components
        self._reward_components.update(components)

        # Check if it's time to log (every 2 seconds)
        current_time = rospy.get_time()
        if current_time - self._last_reward_log_time >= 2.0:  # Changed from 1.0 to 2.0 seconds
            # Calculate total reward
            total_reward = sum(self._reward_components.values())
            
            # Log reward components
            rospy.loginfo("\n=== Reward Components (Last 2 Seconds) ===")  # Updated message
            rospy.loginfo("Total Reward: {:.3f}".format(total_reward))
            for component, value in self._reward_components.items():
                if abs(value) > 1e-6:  # Only show non-zero components
                    rospy.loginfo("{:20s}: {:8.3f}".format(component, value))
            rospy.loginfo("=====================================\n")
            
            # Reset timer
            self._last_reward_log_time = current_time

    def compute_reward(self, action=None, dt=0.0):
        """
        Calculate reward based on achieving stable forward locomotion.
        Includes dynamically weighted penalties for collisions, bounce, and energy.
        Accepts dt for energy calculation.
        """
        # Update total timesteps
        self.total_timesteps += 1
            
        self.global_step += 1
        self.self_col_count = 0 # <<< ADDED: Reset collision count each step

        # --- Define "Smoothness Focus" Reward Weights ---
        FORWARD_WEIGHT = 100.0  # Keep high to encourage forward movement
        FORWARD_VEL_BONUS_WEIGHT = 0.0  # Remove bonus to simplify
        STABILITY_WEIGHT = 2.0  # Increased from 0.8 to align with research (1-10 range)
        ANGULAR_VEL_WEIGHT = 0.5  # Increased from 0.3 for better stability
        HEIGHT_WEIGHT = 0.5  # Remove to simplify reward structure
        ENERGY_WEIGHT = 0.2  # Increased from 0.1 to encourage efficiency
        LATERAL_PENALTY_WEIGHT = 0.0  # Remove to simplify
        ROTATION_PENALTY_WEIGHT = 0.0  # Remove to simplify
        ORIENTATION_PENALTY_WEIGHT = 0.0  # Remove to simplify
        ACTION_RATE_WEIGHT = 0.0  # Remove to simplify
        COLLISION_PENALTY_WEIGHT = 0.0  # Keep disabled
        DYNAMIC_COLLISION_WEIGHT = 0.0  # Keep disabled
        FALL_PENALTY = -200.0  # Keep within research range (-100 to -500)
        SURVIVAL_REWARD = 0.05  # Keep small survival bonus
        BASE_W_BNC = 0.0  # Keep disabled
        BASE_W_EN = 0.0   # Keep disabled

        # Track metrics for this step
        if not hasattr(self, 'metrics'):
            self.metrics = {
                'episode_rewards': [],
                'episode_lengths': [],
                'forward_velocities': [],
                'stability_scores': [],
                'energy_usage': [],
                'fall_counts': [],
                'total_distance': [],
                'avg_roll': [],
                'avg_pitch': [],
                'avg_yaw': [],
                'success_rate': []
            }
            self.metrics_log_interval = 100
            self.metrics_window_size = 100
            self.metrics_file = os.path.join(os.path.dirname(__file__), "../logs/training_metrics.csv")
            
            # Initialize metrics file with headers
            with open(self.metrics_file, 'w') as f:
                headers = [
                    'episode', 'total_timesteps', 'mean_reward', 'mean_episode_length',
                    'mean_forward_velocity', 'mean_stability', 'mean_energy',
                    'fall_rate', 'total_distance', 'mean_roll', 'mean_pitch',
                    'mean_yaw', 'success_rate'
                ]
                f.write(','.join(headers) + '\n')

        # --- Parameters (Constants) ---
        # Kept for calculations even if weight is 0
        FORWARD_VEL_CLIP = 0.05
        STABILITY_DECAY = 2.0
        HEIGHT_DECAY = 5.0
        ENERGY_PENALTY_CAP = 0.5
        LATERAL_DEVIATION_THRESHOLD = 0.1
        ROTATION_THRESHOLD_RAD = float(np.pi) / 32.0
        ORIENTATION_THRESHOLD_RAD = float(np.pi) / 12.0
        FALL_THRESHOLD_ROLL_PITCH = 1.5  # Increased from 1.2 to be more lenient
        COLLISION_THRESHOLD_DISTANCE = 0.065
        # Define pairs of links to check for collision (use short names)
        COLLISION_CHECK_PAIRS = [
            ('tibia_LF', 'tibia_LM'), ('tibia_LM', 'tibia_LR'),
            ('tibia_RF', 'tibia_RM'), ('tibia_RM', 'tibia_RR'),
            ('femur_LF', 'femur_LM'), ('femur_LM', 'femur_LR'),
            ('femur_RF', 'femur_RM'), ('femur_RM', 'femur_RR'),
            ('coxa_LF', 'coxa_LM'), ('coxa_LM', 'coxa_LR'),
            ('coxa_RF', 'coxa_RM'), ('coxa_RM', 'coxa_RR'),
        ]
        # --- End Parameters ---

        # --- State ---
        # Ensure pose/velocities are recent enough
        if not hasattr(self, 'robot_pose') or not hasattr(self, 'prev_position') or self.robot_pose is None or self.prev_position is None:
             rospy.logwarn_throttle(5, "Reward computed before state initialized. Returning 0.")
             return 0.0 # Cannot compute reward without state

        current_pos = np.array(self.robot_pose[:3])
        roll, pitch, yaw = self.robot_pose[3], self.robot_pose[4], self.robot_pose[5]
        
        # Log roll and pitch values to check if they're causing falls - use throttled logging
        if abs(roll) > 0.5 or abs(pitch) > 0.5:
            rospy.logwarn_throttle(5, "High roll/pitch detected: roll={:.2f}, pitch={:.2f}".format(roll, pitch))

        if not hasattr(self, 'prev_orientation'): # Initialize if first step after reset
            self.prev_orientation = (roll, pitch, yaw)
        prev_roll, prev_pitch, prev_yaw = self.prev_orientation

        if self.joint_velocities is None: # Ensure joint velocities received
            rospy.logwarn_throttle(5, "Reward computed before joint velocities received. Returning 0.")
            return 0.0

        # Get angular velocities from IMU
        roll_rate = self.imu_data[3]
        pitch_rate = self.imu_data[4]
        yaw_rate = self.imu_data[5]

        # --- Calculate Reward Components ---

        # 1. Forward movement (Primary Objective)
        forward_movement = current_pos[0] - self.prev_position[0]
        forward_reward_base = FORWARD_WEIGHT * forward_movement
        # Bonus scales up to FORWARD_VEL_BONUS_WEIGHT when forward_movement reaches FORWARD_VEL_CLIP
        forward_bonus = 0.0
        if forward_movement > 1e-6: # Add bonus only if moving forward significantly
             forward_bonus = FORWARD_VEL_BONUS_WEIGHT * min(1.0, float(forward_movement) / FORWARD_VEL_CLIP)
        forward_reward = forward_reward_base + forward_bonus

        # --- NEW: Add Penalty for Backward Movement ---
        backward_penalty = 0.0
        if forward_movement < -1e-6: # Penalize if moving backward significantly
            # Scale penalty quadratically with backward speed, maybe use FORWARD_WEIGHT?
            backward_penalty = -FORWARD_WEIGHT * abs(forward_movement) # Linear penalty for now
        # --- End Backward Penalty ---

        # 2. Stability (Reward low roll/pitch deviations)
        stability_cost = abs(roll) + abs(pitch)
        stability_reward = np.exp(-STABILITY_DECAY * stability_cost) # Exponential decay: 1 at 0 cost

        # 3. Height maintenance (Reward staying near target)
        target_height = 0.15 # Consider making this a parameter if needed
        height_diff = abs(current_pos[2] - target_height)
        height_reward = np.exp(-HEIGHT_DECAY * height_diff) # Exponential decay: 1 at 0 diff

        # 4. Energy efficiency (Penalize high joint velocities)
        energy_cost = np.sum(np.square(self.joint_velocities)) * 0.001 
        energy_penalty = -min(ENERGY_PENALTY_CAP, energy_cost) # Capped penalty [-CAP, 0]

        # 5. Lateral movement penalty (Penalize sideways drift)
        lateral_deviation = abs(current_pos[1])
        # Quadratic penalty, normalized and capped (Uses new threshold)
        lateral_penalty = -min(1.0, float(lateral_deviation) / LATERAL_DEVIATION_THRESHOLD**2) # [-1, 0]

        # 6. Rotation penalty (Penalize excessive turning rate)
        delta_yaw = yaw - prev_yaw
        # Handle angle wrapping
        if delta_yaw > np.pi: delta_yaw -= 2 * np.pi
        elif delta_yaw < -np.pi: delta_yaw += 2 * np.pi
        # Quadratic penalty, normalized and capped (Uses new threshold)
        rotation_penalty = -min(1.0, float(abs(delta_yaw)) / ROTATION_THRESHOLD_RAD**2) # [-1, 0]

        # 7. Orientation penalty (Penalize facing away from forward)
        # Assumes yaw=0 is forward, yaw is in [-pi, pi]
        orientation_deviation = abs(yaw)
        # Quadratic penalty, normalized and capped (Uses new threshold)
        orientation_penalty = -min(1.0, float(orientation_deviation) / ORIENTATION_THRESHOLD_RAD**2) # [-1, 0]

        # 8. Angular Velocity Penalty (Penalize wobbling rate)
        angular_vel_cost = roll_rate**2 + pitch_rate**2 + yaw_rate**2 # Calculation remains
        angular_vel_penalty = -angular_vel_cost # Scaled later by weight (which is now 0)

        # --- Calculate Collision Penalty --- 
        # Calculation remains, but weight is 0
        proximity_penalty = 0.0
        links_available = hasattr(self, 'link_poses') and self.link_poses
        if not links_available:
             rospy.logwarn_throttle(5, "Reward computed before link poses initialized. Skipping collision check.")
        else:
            checked_pairs = 0
            min_dist_found = float('inf')

            for link1_name, link2_name in COLLISION_CHECK_PAIRS:
                # Check if both link poses were received
                if link1_name in self.link_poses and link2_name in self.link_poses:
                    pos1 = self.link_poses[link1_name]
                    pos2 = self.link_poses[link2_name]

                    distance = np.linalg.norm(pos1 - pos2)
                    min_dist_found = min(min_dist_found, distance)
                    checked_pairs += 1
                    
                    if distance < COLLISION_THRESHOLD_DISTANCE:
                        # Calculate base penalty but COLLISION_PENALTY_WEIGHT is 0
                        penalty = COLLISION_PENALTY_WEIGHT * (COLLISION_THRESHOLD_DISTANCE - distance)
                        proximity_penalty -= penalty 
                        self.self_col_count += 1

            # Ensure penalty isn't applied if no pairs could be checked
            if checked_pairs == 0:
                 proximity_penalty = 0.0

        # 9. Action Rate Penalty (Penalize large action changes)
        # Calculation remains, but weight is 0
        action_rate_penalty = 0.0
        if self.prev_action is not None and action is not None:
            action_diff = np.sum(np.square(action - self.prev_action))
            action_rate_penalty = -action_diff 

        # 10. Termination Penalty (Check for fall)
        # This check determines if the fall penalty should be applied in this step's reward
        is_fallen = abs(roll) > FALL_THRESHOLD_ROLL_PITCH or abs(pitch) > FALL_THRESHOLD_ROLL_PITCH
        fall_penalty_term = FALL_PENALTY if is_fallen else 0.0
        
        # Log if the agent has fallen
        if is_fallen:
            rospy.logwarn("Agent has fallen! Roll: {:.2f}, Pitch: {:.2f}, Threshold: {:.2f}".format(
                abs(roll), abs(pitch), FALL_THRESHOLD_ROLL_PITCH))

        # --- Combine Rewards (using mostly zero weights) ---

        # Calculate dynamic penalty terms separately (weights are 0)
        dyn_col_term = -DYNAMIC_COLLISION_WEIGHT * self.self_col_count
        imu_accel_z = self.imu_data[2] if hasattr(self, 'imu_data') and self.imu_data is not None else 0.0
        dyn_bnc_term = -BASE_W_BNC * abs(imu_accel_z)
        dyn_en_term = -BASE_W_EN * (np.sum(np.abs(np.array(self.last_joint_effort) * self.joint_velocities)) * dt if hasattr(self, 'last_joint_effort') and self.last_joint_effort is not None and self.joint_velocities is not None and dt > 0 else 0.0)

        # Most terms will be zero due to zero weights
        reward = (
            forward_reward + # Uses FORWARD_WEIGHT
            backward_penalty + # Uses FORWARD_WEIGHT internally
            STABILITY_WEIGHT * stability_reward + # Non-zero
            ANGULAR_VEL_WEIGHT * angular_vel_penalty + # Non-zero
            HEIGHT_WEIGHT * height_reward + # Zero
            ENERGY_WEIGHT * energy_penalty + # Zero
            LATERAL_PENALTY_WEIGHT * lateral_penalty + # Zero
            ROTATION_PENALTY_WEIGHT * rotation_penalty + # Zero
            ORIENTATION_PENALTY_WEIGHT * orientation_penalty + # Zero
            ACTION_RATE_WEIGHT * action_rate_penalty + # Non-zero
            proximity_penalty + # Zero
            SURVIVAL_REWARD + # Non-zero
            fall_penalty_term + # Non-zero
            dyn_col_term + # Zero
            dyn_bnc_term + # Zero
            dyn_en_term  # Zero
        )

        # --- NEW: Track reward components for 2-second window ---
        # Store current reward components
        reward_components = {
            'forward': forward_reward,
            'backward': backward_penalty,
            'stability': STABILITY_WEIGHT * stability_reward,
            'angular_velocity': ANGULAR_VEL_WEIGHT * angular_vel_penalty,
            'height': HEIGHT_WEIGHT * height_reward,
            'energy': ENERGY_WEIGHT * energy_penalty,
            'lateral': LATERAL_PENALTY_WEIGHT * lateral_penalty,
            'rotation': ROTATION_PENALTY_WEIGHT * rotation_penalty,
            'orientation': ORIENTATION_PENALTY_WEIGHT * orientation_penalty,
            'action_rate': ACTION_RATE_WEIGHT * action_rate_penalty,
            'collision': proximity_penalty,
            'survival': SURVIVAL_REWARD,
            'fall': fall_penalty_term,
            'dynamic_collision': dyn_col_term,
            'bounce': dyn_bnc_term,
            'dynamic_energy': dyn_en_term,
            'total': reward
        }
        
        # Add current reward components to history
        for component, value in reward_components.items():
            self.reward_history[component].append(value)
        
        # Check if it's time to log (every 2 seconds)
        current_time = rospy.get_time()
        if current_time - self.last_reward_log_time >= self.reward_log_interval:
            # Calculate average reward components over the 2-second window
            avg_reward_components = {}
            for component, values in self.reward_history.items():
                if values:  # Check if there are any values
                    avg_reward_components[component] = sum(values) / len(values)
                else:
                    avg_reward_components[component] = 0.0
            
            # --- MODIFIED: Calculate percentages based on sum of absolute values ---
            total_avg_reward = avg_reward_components.get('total', 0.0)
            # Calculate sum of absolute values of all components (excluding total)
            sum_abs_components = sum(abs(v) for k, v in avg_reward_components.items() if k != 'total')
            
            reward_percentages = {}
            if sum_abs_components > 1e-6:  # Avoid division by zero
                for k, v in avg_reward_components.items():
                    if k != 'total':
                        # Percentage represents the component's contribution to the total magnitude
                        # The sign indicates whether it was positive or negative contribution
                        reward_percentages[k] = (v / sum_abs_components) * 100
            else:
                # If sum of abs is zero, all components are zero
                reward_percentages = {k: 0.0 for k in avg_reward_components.keys() if k != 'total'}
            # --- END MODIFICATION ---
            
            # Log average reward components
            rospy.loginfo("\n=== Avg Reward Components (Last {:.1f}s) | Total: {:.2f} ===".format(
                self.reward_log_interval, total_avg_reward))
            # Sort components by absolute percentage contribution for clarity
            sorted_components = sorted(reward_percentages.items(), key=lambda item: abs(item[1]), reverse=True)
            for component, percentage in sorted_components:
                # Show components contributing more than 0.1% magnitude
                if abs(percentage) > 0.1:
                    rospy.loginfo("{:>20}: {:6.1f}% ({:+.3f})".format(component, percentage, avg_reward_components[component]))
            rospy.loginfo("="*30 + "\n")
            
            # Reset reward history and update last log time
            for component in self.reward_history:
                self.reward_history[component] = []
            self.last_reward_log_time = current_time
        # --- END NEW ---

        # Update metrics at the end of the method
        self.metrics['forward_velocities'].append(forward_movement)
        self.metrics['stability_scores'].append(-stability_cost)
        self.metrics['energy_usage'].append(energy_cost)
        self.metrics['avg_roll'].append(abs(roll))
        self.metrics['avg_pitch'].append(abs(pitch))
        self.metrics['avg_yaw'].append(abs(yaw))

        # Log metrics if we've reached the interval
        if len(self.metrics['episode_rewards']) % self.metrics_log_interval == 0:
            window = min(self.metrics_window_size, len(self.metrics['episode_rewards']))
            if window > 0: # Avoid division by zero if window is 0
                 recent_rewards = self.metrics['episode_rewards'][-window:]
                 recent_lengths = self.metrics['episode_lengths'][-window:]
                 recent_velocities = self.metrics['forward_velocities'][-window:]
                 recent_stability = self.metrics['stability_scores'][-window:]
                 recent_energy = self.metrics['energy_usage'][-window:]
                 recent_roll = self.metrics['avg_roll'][-window:]
                 recent_pitch = self.metrics['avg_pitch'][-window:]
                 recent_yaw = self.metrics['avg_yaw'][-window:]

                 # Calculate statistics
                 mean_reward = np.mean(recent_rewards)
                 mean_length = np.mean(recent_lengths)
                 mean_velocity = np.mean(recent_velocities)
                 mean_stability = np.mean(recent_stability)
                 mean_energy = np.mean(recent_energy)
                 # Use self.falls_count for overall rate, not just window
                 fall_rate = float(self.falls_count) / max(1, self.episode_count)  # Overall fall rate
                 success_rate = 1.0 - fall_rate
                 mean_roll = np.mean(recent_roll)
                 mean_pitch = np.mean(recent_pitch)
                 mean_yaw = np.mean(recent_yaw)

                 # Write metrics to file
                 with open(self.metrics_file, 'a') as f:
                     metrics_line = [
                         str(self.episode_count),
                         str(self.total_timesteps),
                         "{:.3f}".format(mean_reward),
                         "{:.1f}".format(mean_length),
                         "{:.3f}".format(mean_velocity),
                         "{:.3f}".format(mean_stability),
                         "{:.3f}".format(mean_energy),
                         "{:.3f}".format(fall_rate),
                         "{:.2f}".format(self.distance_traveled), # Cumulative distance
                         "{:.3f}".format(mean_roll),
                         "{:.3f}".format(mean_pitch),
                         "{:.3f}".format(mean_yaw),
                         "{:.3f}".format(success_rate)
                     ]
                     f.write(','.join(metrics_line) + '\n')

                 # Log metrics to ROS
                 rospy.loginfo("\n=== Training Metrics (Last {} episodes) ===".format(window))
                 rospy.loginfo("Episode: {}, Total Timesteps: {}".format(self.episode_count, self.total_timesteps))
                 rospy.loginfo("Mean Reward: {:.3f}".format(mean_reward))
                 rospy.loginfo("Mean Episode Length: {:.1f}".format(mean_length))
                 rospy.loginfo("Mean Forward Velocity: {:.3f}".format(mean_velocity))
                 rospy.loginfo("Mean Stability Score: {:.3f}".format(mean_stability))
                 rospy.loginfo("Mean Energy Usage: {:.3f}".format(mean_energy))
                 rospy.loginfo("Overall Fall Rate: {:.3f}".format(fall_rate))
                 rospy.loginfo("Success Rate: {:.3f}".format(success_rate))
                 rospy.loginfo("Mean Roll: {:.3f}".format(mean_roll))
                 rospy.loginfo("Mean Pitch: {:.3f}".format(mean_pitch))
                 rospy.loginfo("Mean Yaw: {:.3f}".format(mean_yaw))
                 rospy.loginfo("Total Distance: {:.2f}".format(self.distance_traveled))
                 rospy.loginfo("=====================================\n")

        return reward

    def run(self):
        """Main loop for the bridge node. Uses rospy.spin() to process callbacks."""
        rospy.loginfo("Bridge: Entering rospy.spin() to process callbacks.")
        try:
            rospy.spin()
        except KeyboardInterrupt:
            rospy.loginfo("Bridge: KeyboardInterrupt received, shutting down spin loop.")
        except Exception as e:
            rospy.logerr("Bridge: Error during rospy.spin(): %s", e)
        finally:
            rospy.loginfo("Bridge: Exiting rospy.spin() loop.")

    def step(self, action):
        """Execute one time step within the environment."""
        try:
            # Convert action to joint positions
            joint_positions = self.action_to_joint_positions(action)
            
            # Send joint positions to robot
            self.publish_joint_positions(joint_positions)
            
            # Wait for next control cycle
            rospy.sleep(self.dt)
            
            # Get current state
            observation = self.get_observation()
            
            # Calculate reward
            reward, info = self.calculate_reward(observation)
            
            # Check if episode is done
            done = self.is_done(observation)
            
            # Increment episode counter if done
            if done:
                self.episode_count += 1
                
            # Increment total timesteps
            self.total_timesteps += 1
            
            return observation, reward, done, info
            
        except Exception as e:
            log_exception(logging, e, "Error in step")
            return self.get_observation(), 0, True, {}

    def _reset_robot(self):
        """Reset the robot to its initial pose in Gazebo."""
        try:
            # <<< ADDED: Short delay before resetting simulation >>>
            rospy.sleep(0.2) # Increased from 0.1
            
            # Reset the simulation world (models and state, not time)
            rospy.loginfo("Bridge: Resetting Gazebo world...")
            try:
                # <<< CHANGED: Use reset_world instead of reset_sim >>>
                self.reset_world()
                rospy.sleep(0.5) # <<< ADDED: Wait after reset_world call
            except Exception as e:
                # "ROS time moved backwards" should NOT happen with reset_world
                # Handle other potential service call errors
                rospy.logerr("Error during reset_world service call: {}".format(e))
                raise # Re-raise the error for the outer handler
            
            # Wait for simulation to stabilize after world reset
            rospy.loginfo("Bridge: Waiting for simulation to stabilize after reset...")
            rospy.sleep(1.5)  # Increased wait time from 1.0
            
            # Set robot to initial pose with lower height for better stability
            model_state = ModelState()
            model_state.model_name = self.robot_name
            model_state.pose.position.x = 0.0
            model_state.pose.position.y = 0.0
            model_state.pose.position.z = 0.15  # Lower initial height
            model_state.pose.orientation.w = 1.0  # Identity quaternion (no rotation)
            model_state.pose.orientation.x = 0.0
            model_state.pose.orientation.y = 0.0
            model_state.pose.orientation.z = 0.0
            
            # Set the model state in Gazebo
            rospy.loginfo("Bridge: Setting model state...")
            try:
                self.set_model_state(model_state)
                rospy.sleep(0.3) # <<< ADDED: Wait after set_model_state
            except Exception as e:
                rospy.logerr("Failed to set model state: {}".format(e))
                raise
            
            # Set initial joint positions with more stable stance (physics running)
            rospy.loginfo("Bridge: Setting initial joint positions (physics running)...")
            stable_stance = [
                0.3,  0.4, -0.8,   # LF: coxa, femur, tibia
                0.3,  0.4, -0.8,   # LM
                0.3,  0.4, -0.8,   # LR
               -0.3,  0.4, -0.8,   # RF
               -0.3,  0.4, -0.8,   # RM
               -0.3,  0.4, -0.8    # RR
            ]
            
            rospy.loginfo("Bridge: --- Entering joint setting loop ---")
            for i, joint_name in enumerate(self.joint_names):
                try:
                    rospy.logdebug("Bridge: Setting joint {} ({}) to {:.2f}".format(i, joint_name, stable_stance[i]))
                    position = stable_stance[i]
                    self.joint_publishers[i].publish(Float64(position))
                    rospy.sleep(0.05)  # Small delay between joint commands (keep this short)
                except Exception as e:
                    rospy.logerr("Error setting joint {}: {}".format(joint_name, e))
            rospy.loginfo("Bridge: --- Exited joint setting loop ---")
            
            # Wait for joint commands to be processed/robot to settle
            rospy.loginfo("Bridge: Waiting for joints to settle (physics running)...")
            rospy.sleep(1.0)  # Increased wait time from 0.5 since physics is running
            
            # Wait for simulation to stabilize after joint setting
            rospy.loginfo("Bridge: Final stabilization wait (physics running)...")
            rospy.sleep(1.5)  # Keep this stabilization wait
            
            # Randomize physics properties if enabled
            if self.use_domain_randomization:
                self.randomize_physics()
            
            rospy.loginfo("Bridge: Robot reset sequence in _reset_robot complete")
            
            return True
            
        except Exception as e:
            rospy.logerr("Error in _reset_robot sequence: {}".format(e))
            # Try to unpause physics if we failed while it was paused
            try:
                self.unpause_physics()
                rospy.logwarn("Attempted to unpause physics after error in _reset_robot.")
            except:
                pass
            return False

    def _get_observation(self):
        """Get the current observation from robot state."""
        try:
            # Ensure we have received all necessary state information
            if self.joint_states is None or self.joint_velocities is None or self.robot_pose is None or self.imu_data is None:
                rospy.logwarn_throttle(5, "Missing state information in _get_observation. Returning zero observation.")
                return np.zeros(51)  # Return zero observation (18 + 18 + 6 + 9 = 51)
            
            # Ensure arrays are the correct size
            if (len(self.joint_states) != 18 or len(self.joint_velocities) != 18 or 
                len(self.robot_pose) != 6 or len(self.imu_data) != 9):
                rospy.logwarn_throttle(5, "Incorrect state array sizes in _get_observation. Returning zero observation.")
                return np.zeros(51)
            
            # Construct observation from current state
            observation = np.concatenate([
                self.joint_states.astype(np.float32),      # 18 joint positions
                self.joint_velocities.astype(np.float32),  # 18 joint velocities
                self.robot_pose.astype(np.float32),        # 6 robot pose (x, y, z, roll, pitch, yaw)
                self.imu_data.astype(np.float32)          # 9 IMU data (linear accel, angular vel, orientation)
            ])
            
            # Log if observation contains NaN or infinite values
            if not np.all(np.isfinite(observation)):
                rospy.logwarn_throttle(5, "Observation contains NaN or infinite values. Replacing with zeros.")
                observation = np.nan_to_num(observation, nan=0.0, posinf=0.0, neginf=0.0)
            
            return observation
            
        except Exception as e:
            rospy.logerr("Error in _get_observation: {}".format(e))
            return np.zeros(51)  # Return zero observation on error (18 + 18 + 6 + 9 = 51)

if __name__ == '__main__':
    try:
        bridge = JetHexaGymBridge()
        bridge.run()
    except rospy.ROSInterruptException:
        pass 