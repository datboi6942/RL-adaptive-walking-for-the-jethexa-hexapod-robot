#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python 2 script for ROS/Gazebo interfacing

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

# Import the local module for terrain generation
from terrain_generator import TerrainGenerator
# <<< ADDED: Import the CPGControl service definition
from jethexa_rl.srv import CPGControl, CPGControlRequest

class JetHexaGymBridge:
    """
    ROS-based bridge between Gazebo and the Python 3 RL framework.
    
    This node handles the interface with the simulated robot in Gazebo using Python 2,
    while communicating with the RL training algorithm in Python 3 through ROS topics.
    """
    def __init__(self):
        rospy.init_node('jethexa_gym_bridge', anonymous=True)
        rospy.loginfo("Initializing JetHexa Gym Bridge (Python 2)")
        
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
        
        # Connect to simulation services
        rospy.loginfo("Bridge: Connecting to Gazebo services...") # Added log
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
        except Exception as e:
            rospy.logerr("Bridge: Error connecting to Gazebo services: {}".format(e))
            raise # Reraise exception to prevent node from starting with missing services
        
        # Episode tracking
        self.episode_steps = 0
        self.max_episode_steps = 1000
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
        # rospy.Subscriber('/gazebo/model_states', ModelStates, self.model_state_cb) # <<< COMMENTED OUT
        rospy.Subscriber('/gazebo/link_states', LinkStates, self.link_state_cb) # <<< ADDED: Subscribe to link states
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
            rospy.loginfo("Initialized robot position log at: {}".format(self.robot_position_log_path))
        except Exception as e:
            rospy.logerr("ERROR initializing robot position log file: {}".format(e))
        
        # Wait for connections to establish
        rospy.sleep(1.0)
        rospy.loginfo("JetHexa Gym Bridge initialized")

        # ROS Service Clients
        rospy.wait_for_service('/gazebo/set_physics_properties', timeout=5.0)
        self.set_physics_client = rospy.ServiceProxy('/gazebo/set_physics_properties', SetPhysicsProperties)
        rospy.wait_for_service('/gazebo/get_physics_properties', timeout=5.0)
        self.get_physics_client = rospy.ServiceProxy('/gazebo/get_physics_properties', GetPhysicsProperties)
        # Added CPG Service Client (ensure it's available)
        try:
             rospy.wait_for_service('/cpg_control', timeout=3.0) 
             self.cpg_service_client = rospy.ServiceProxy('/cpg_control', CPGControl)
             rospy.loginfo("Successfully connected to /cpg_control service.")
        except (rospy.ROSException, rospy.ServiceException) as e:
             rospy.logwarn("Could not connect to /cpg_control service: {}. CPG features disabled.".format(e))
             self.cpg_service_client = None # Set to None if not available

        self.robot_name = rospy.get_param('~robot_name', 'jethexa')
        self.initial_height = rospy.get_param('~initial_height', 0.18) # Default height
        self.cpg_warmup_steps = rospy.get_param('~cpg_warmup_steps', 10) # Default CPG warmup steps
    
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
                is_fallen = abs(self.robot_pose[3]) > 0.7 or abs(self.robot_pose[4]) > 0.7 # Check fall using NEW pose
                # The FALL_PENALTY is added within compute_reward, just check for termination here
                done = timeout_done or is_fallen
                
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
        action = np.array(msg.data)
        self.current_action = action.copy() # Store the latest action
        # <<< ADDED: Log action received via ROS topic >>>
        rospy.logdebug("ActionCB: Received Action via Topic: {}".format(np.round(action[:6], 3))) # Log first 6 values
        # <<< END LOG >>>
        
        # Apply joint commands immediately
        for i in range(min(len(action), 18)):
            try:
                command = Float64(action[i]) # Use the received action directly
                # <<< ADDED: Log first few joint commands published >>>
                if i < 3:
                     rospy.logdebug("    Pub Joint {}: {:.3f}".format(i, command.data))
                elif i == 3:
                     rospy.logdebug("    ... (publishing remaining joints) ...")
                # <<< END LOG >>>
                self.joint_publishers[i].publish(command)
            except Exception as e:
                 rospy.logerr("ActionCB: ERROR publishing joint command for index {}: {}".format(i, e))
        
        # --- REMOVED --- 
        # - Time tracking (now in link_state_cb)
        # - Step counter increment (now in link_state_cb)
        # - Observation construction (now in link_state_cb)
        # - Reward calculation (now in link_state_cb)
        # - Done check (now in link_state_cb)
        # - Distance update (now in link_state_cb)
        # - Publishing obs/reward/done/info (now in link_state_cb)
        # - Updating prev_position/orientation/action (now in link_state_cb)
        # ------------- 

    def reset_cb(self, msg):
        """Handle reset request from Python 3 RL training script."""
        rospy.loginfo("Bridge: Reset callback triggered.")
        if msg.data:
            try:
                # Call the main reset logic. Reset complete signal is now sent from within self.reset()
                self.reset() 
                rospy.loginfo("Bridge: reset() function returned.")
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
            self.set_physics_client(
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
        reset_start_time = rospy.get_time()
        rospy.loginfo("Bridge: --- Starting reset() method at t={:.4f} ---".format(reset_start_time))
        
        paused_physics = False 
        set_state_success = False
        
        try:
            rospy.loginfo("Bridge: Reset - STEP 1: Pause Physics")
            t0 = rospy.get_time()
            self.pause_physics()
            paused_physics = True # Mark as paused
            t1 = rospy.get_time()
            rospy.loginfo("Bridge: Reset - STEP 1: Pause Physics COMPLETED")

            rospy.loginfo("Bridge: Reset - STEP 2: Set Zero Heading")
            # --- Define zero_state OUTSIDE the service call try block --- 
            from gazebo_msgs.msg import ModelState
            from gazebo_msgs.srv import SetModelState
            
            zero_state = ModelState()
            zero_state.model_name = self.robot_name
            zero_state.pose.position.x = 0.0
            zero_state.pose.position.y = 0.0
            zero_state.pose.position.z = 0.2 # Use Z=0.2
            quat = quaternion_from_euler(0.0, 0.0, 0.0)
            zero_state.pose.orientation.x = quat[0]
            zero_state.pose.orientation.y = quat[1]
            zero_state.pose.orientation.z = quat[2]
            zero_state.pose.orientation.w = quat[3]
            zero_state.twist.linear.x = 0.0
            zero_state.twist.linear.y = 0.0
            zero_state.twist.linear.z = 0.0
            zero_state.twist.angular.x = 0.0
            zero_state.twist.angular.y = 0.0
            zero_state.twist.angular.z = 0.0
            # -----------------------------------------------------------

            set_state_success = False # Track success
            rospy.loginfo("Bridge: Reset - Attempting to call /gazebo/set_model_state...")
            try:
                rospy.wait_for_service('/gazebo/set_model_state', timeout=2.0)
                set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
                resp = set_state(zero_state) # Call the service with pre-defined state
                if resp.success:
                    rospy.loginfo("Bridge: Reset - set_model_state call successful.")
                    set_state_success = True
                else:
                    rospy.logwarn("Bridge: Reset - set_model_state call failed: {}".format(resp.status_message))
            except (rospy.ServiceException, rospy.ROSException, rospy.ROSInterruptException) as e:
                rospy.logwarn("Bridge: Reset - Exception during set_model_state call: {}".format(e))
            
            rospy.loginfo("Bridge: Reset - STEP 2: Set Zero Heading COMPLETED (Call Success Flag: {})".format(set_state_success))

            # --- Reset internal state variables (Uses zero_state defined above) ---
            rospy.loginfo("Bridge: Reset - STEP 3: Reset Internal State Vars")
            self.last_step_time = None
            self.sim_time = 0
            self.episode_steps = 0
            self.episode_count += 1
            self.total_reward = 0
            self.distance_traveled = 0
            self.energy_used = 0
            self.start_position = np.array([zero_state.pose.position.x, zero_state.pose.position.y, zero_state.pose.position.z]) # Now zero_state is in scope
            self.prev_position = self.start_position.copy()
            self.prev_orientation = np.zeros(3) 
            self.prev_action = None 
            self.self_col_count = 0 
            self.link_poses = {} 
            self.base_twist = None 
            rospy.loginfo("Bridge: Reset - STEP 3: Reset Internal State Vars COMPLETED")

            # --- CPG Controller Reset and Warm-up --- 
            rospy.loginfo("Bridge: Reset - STEP 4: CPG Reset/Warmup")
            if self.cpg_service_client is not None:
                try:
                    # 1. Reset the CPG state via service
                    rospy.loginfo("Bridge: Reset - Calling CPG service: reset")
                    reset_req = CPGControlRequest(command='reset')
                    reset_resp = self.cpg_service_client(reset_req)
                    if not reset_resp.success:
                        rospy.logwarn("Bridge: Reset - CPG service RESET call failed: {}".format(reset_resp.message))
                        # Consider raising an error or handling failure more robustly
                    else:
                        rospy.loginfo("Bridge: Reset - CPG service RESET successful.")
                    
                    # 2. Warm-up CPG by calling UPDATE via service (Exact Steps)
                    rospy.loginfo("Bridge: Reset - Warming up CPG for exactly {} steps via service...".format(self.cpg_warmup_steps))
                    update_req = CPGControlRequest(command='update', dt=self.sim_step_dt) # Assuming self.sim_step_dt is correct
                    for i in range(self.cpg_warmup_steps):
                        rospy.logdebug("Bridge: Reset - Calling CPG service: UPDATE (step {}/{})".format(i+1, self.cpg_warmup_steps))
                        update_resp = self.cpg_service_client(update_req)
                        if not update_resp.success:
                            rospy.logwarn("Bridge: Reset - CPG service UPDATE call failed (step {}): {}".format(i+1, update_resp.message))
                            break # Stop warmup on failure
                        # Optionally publish joints during warmup? Current code doesn't seem to.
                    rospy.loginfo("Bridge: Reset - CPG service warm-up finished (or was interrupted).")
        
                except rospy.ServiceException as e:
                    rospy.logerr("Bridge: Reset - ServiceException during CPG reset/warmup: {}".format(e))
                # Removed specific ROSException handler as ServiceException covers timeout
                except Exception as e:
                    rospy.logerr("Bridge: Reset - Unexpected error during CPG service reset/warmup: {}".format(e))
            else:
                 rospy.logwarn("Bridge: Reset - /cpg_control service client not available. Skipping CPG reset/warmup.")
            rospy.loginfo("Bridge: Reset - STEP 4: CPG Reset/Warmup COMPLETED")

            # --- Reset Terrain --- (If applicable)
            # rospy.loginfo("Bridge: Reset - STEP 5: Reset Terrain")
            # ... terrain reset logic ...
            # rospy.loginfo("Bridge: Reset - STEP 5: Reset Terrain COMPLETED")

            # --- Domain Randomization ---
            rospy.loginfo("Bridge: Reset - STEP 6: Randomize Physics")
            t0_physics = rospy.get_time()
            try:
                 self.randomize_physics() 
                 t1_physics = rospy.get_time()
                 rospy.loginfo("Bridge: Reset - STEP 6: Randomize Physics COMPLETED")
            except Exception as e:
                 rospy.logerr("Bridge: Reset - Error during randomize_physics: {}".format(e))

            # --- Unpause Physics ---
            rospy.loginfo("Bridge: Reset - STEP 7: Unpause Physics")
            t0_unpause = rospy.get_time()
            try:
                self.unpause_physics()
                rospy.loginfo("Bridge: Reset - STEP 7: Unpause Physics COMPLETED")
            except rospy.ServiceException as e:
                rospy.logerr("Bridge: Reset - Error calling unpause_physics: {}".format(e))
            except Exception as e:
                 rospy.logerr("Bridge: Reset - Unexpected error during unpause_physics: {}".format(e))
            t1_unpause = rospy.get_time()
            rospy.loginfo("Bridge: Reset - STEP 7: Unpause Physics call took {:.4f}s".format(t1_unpause-t0_unpause))

            # --- Final Logs and Observation Publishing ---
            rospy.loginfo("Bridge: Reset - STEP 8: Publish Initial Observation")
            t0 = rospy.get_time()
            obs = np.concatenate([
                self.joint_states,
                self.joint_velocities,
                self.robot_pose,
                self.imu_data
            ])
            obs_msg = Float32MultiArray()
            obs_msg.data = obs.astype(np.float32).tolist()
            self.obs_pub.publish(obs_msg)
            t1 = rospy.get_time()
            rospy.loginfo("Bridge: Reset - STEP 8: Publish Initial Observation COMPLETED")
            
            # Signal that reset is complete
            rospy.loginfo("Bridge: Reset - STEP 9: Publish Reset Complete Signal")
            self.reset_complete_pub.publish(Bool(True))
            t_rc = rospy.get_time()
            rospy.loginfo("Bridge: Reset - STEP 9: Publish Reset Complete Signal COMPLETED (at t={:.4f})".format(t_rc))

            # Ensure relevant state is cleared for the new architecture
            self.current_action = None
            self.last_state_update_time = None

        except Exception as e:
            rospy.logerr("Bridge: Reset - UNEXPECTED EXCEPTION in reset main block: {}".format(e))
            # Ensure reset_complete doesn't hang the caller on error
            rospy.logwarn("Bridge: Reset - Publishing Reset Complete=False due to error.")
            self.reset_complete_pub.publish(Bool(False)) # Publish False on error
        
        reset_end_time = rospy.get_time()
        rospy.loginfo("Bridge: --- Finished reset() method at t={:.4f} (Total Duration: {:.4f}s) ---".format(
            reset_end_time, reset_end_time - reset_start_time))

    def set_difficulty(self, level):
        """Set the terrain difficulty level for curriculum learning."""
        self.current_difficulty = max(0, min(4, level))
        rospy.loginfo("Setting environment difficulty to level {}".format(self.current_difficulty))
        
        # Update terrain if this is called mid-episode
        if self.episode_steps > 0:
            self.terrain_generator.set_difficulty(self.current_difficulty)
            self.terrain_generator.reset_terrain()

    def compute_reward(self, action=None, dt=0.0):
        """
        Calculate reward based on achieving stable forward locomotion.
        Includes dynamically weighted penalties for collisions, bounce, and energy.
        Accepts dt for energy calculation.
        """
        # --- New: count env steps for dynamic ramp ---
        self.global_step += 1
        self.self_col_count = 0 # <<< ADDED: Reset collision count each step

        # --- Parameters (v14 Tuning: Reduce Static Rewards for Initial Exploration) ---
        FORWARD_WEIGHT = 600.0 # Keep from v7
        FORWARD_VEL_BONUS_WEIGHT = 25.0 # Keep from v9
        FORWARD_VEL_CLIP = 0.05 # Keep from v7
        STABILITY_WEIGHT = 0.5 # <<< DECREASED AGAIN: Was 1.5, Significantly reduce static stability reward
        STABILITY_DECAY = 2.0 # Keep from v7
        ANGULAR_VEL_WEIGHT = 0.25 # Keep from v13
        HEIGHT_WEIGHT = 0.6 # <<< DECREASED: Was 2.5, Reduce reward for just being still at a certain height
        HEIGHT_DECAY = 5.0 # Keep from v7
        ENERGY_WEIGHT = 0.3 # Keep from v8
        ENERGY_PENALTY_CAP = 0.5 # Keep from v7
        LATERAL_PENALTY_WEIGHT = 5.0 # INCREASED from 2.5: Penalize sideways drift more
        LATERAL_DEVIATION_THRESHOLD = 0.15 # <<< Reduced from 0.5 >>>
        ROTATION_PENALTY_WEIGHT = 6.0 # INCREASED from 3.0: Penalize turning rate more
        ROTATION_THRESHOLD_RAD = np.pi / 24 # <<< Reduced from np.pi / 4 (approx 7.5 degrees) >>>
        ORIENTATION_PENALTY_WEIGHT = 12.0 # INCREASED from 8.0: Penalize facing away more
        ORIENTATION_THRESHOLD_RAD = np.pi / 6 # <<< Reduced from np.pi / 2 (approx 30 degrees) >>>
        ACTION_RATE_WEIGHT = 0.2 # Keep from v10
        SURVIVAL_REWARD = 0.05 # Keep from v7
        FALL_PENALTY = -200.0 # Keep from v7
        FALL_THRESHOLD_ROLL_PITCH = 1.2 # Keep from v7
        # --- ADDED: Collision Penalty Parameters ---
        COLLISION_PENALTY_WEIGHT = 10.0 # Adjust as needed
        COLLISION_THRESHOLD_DISTANCE = 0.02 # <<< CHANGED: 2cm threshold
        # Define pairs of links to check for collision (use short names)
        COLLISION_CHECK_PAIRS = [
            ('tibia_LF', 'tibia_LM'), ('tibia_LM', 'tibia_LR'),
            ('tibia_RF', 'tibia_RM'), ('tibia_RM', 'tibia_RR'),
            ('femur_LF', 'femur_LM'), ('femur_LM', 'femur_LR'),
            ('femur_RF', 'femur_RM'), ('femur_RM', 'femur_RR'),
            ('coxa_LF', 'coxa_LM'), ('coxa_LM', 'coxa_LR'),
            ('coxa_RF', 'coxa_RM'), ('coxa_RM', 'coxa_RR'),
        ]
        # --- End Collision Params ---

        # --- New: fade-in penalties over first ramp_duration steps ---
        progress = min(1.0, float(self.global_step) / self.ramp_duration)
        w_col     = self.BASE_W_COL * progress
        w_bnc     = self.BASE_W_BNC * progress
        w_en      = self.BASE_W_EN  * progress

        # --- State ---
        # Ensure pose/velocities are recent enough
        if not hasattr(self, 'robot_pose') or not hasattr(self, 'prev_position') or self.robot_pose is None or self.prev_position is None:
             rospy.logwarn_throttle(5, "Reward computed before state initialized. Returning 0.")
             return 0.0 # Cannot compute reward without state

        current_pos = np.array(self.robot_pose[:3])
        roll, pitch, yaw = self.robot_pose[3], self.robot_pose[4], self.robot_pose[5]

        if not hasattr(self, 'prev_orientation'): # Initialize if first step after reset
            self.prev_orientation = (roll, pitch, yaw)
        prev_roll, prev_pitch, prev_yaw = self.prev_orientation

        if self.joint_velocities is None: # Ensure joint velocities received
            rospy.logwarn_throttle(5, "Reward computed before joint velocities received. Returning 0.")
            return 0.0

        # Get angular velocities from IMU
        roll_rate = self.imu_data[3]
        pitch_rate = self.imu_data[4]
        yaw_rate = self.imu_data[5] # <<< ADDED: Get yaw rate

        # --- Calculate Reward Components ---

        # 1. Forward movement (Primary Objective)
        forward_movement = current_pos[0] - self.prev_position[0]
        forward_reward_base = FORWARD_WEIGHT * forward_movement
        # Bonus scales up to FORWARD_VEL_BONUS_WEIGHT when forward_movement reaches FORWARD_VEL_CLIP
        forward_bonus = 0.0
        if forward_movement > 1e-6: # Add bonus only if moving forward significantly
             forward_bonus = FORWARD_VEL_BONUS_WEIGHT * min(1.0, forward_movement / FORWARD_VEL_CLIP)
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
        lateral_penalty = -min(1.0, (lateral_deviation / LATERAL_DEVIATION_THRESHOLD)**2) # [-1, 0]

        # 6. Rotation penalty (Penalize excessive turning rate)
        delta_yaw = yaw - prev_yaw
        # Handle angle wrapping
        if delta_yaw > np.pi: delta_yaw -= 2 * np.pi
        elif delta_yaw < -np.pi: delta_yaw += 2 * np.pi
        # Quadratic penalty, normalized and capped (Uses new threshold)
        rotation_penalty = -min(1.0, (abs(delta_yaw) / ROTATION_THRESHOLD_RAD)**2) # [-1, 0]

        # 7. Orientation penalty (Penalize facing away from forward)
        # Assumes yaw=0 is forward, yaw is in [-pi, pi]
        orientation_deviation = abs(yaw)
        # Quadratic penalty, normalized and capped (Uses new threshold)
        orientation_penalty = -min(1.0, (orientation_deviation / ORIENTATION_THRESHOLD_RAD)**2) # [-1, 0]

        # 8. Angular Velocity Penalty (Penalize wobbling rate)
        angular_vel_cost = roll_rate**2 + pitch_rate**2 + yaw_rate**2 # <<< MODIFIED: Include yaw_rate^2
        angular_vel_penalty = -angular_vel_cost # Scaled later by weight

        # --- Calculate Collision Penalty ---
        # Note: This calculates proximity penalty, actual self_col_count needs separate update logic
        proximity_penalty = 0.0 # Renamed from collision_penalty to avoid confusion with r_selfcol
        links_available = hasattr(self, 'link_poses') and self.link_poses
        if not links_available:
             rospy.logwarn_throttle(5, "Reward computed before link poses initialized. Skipping collision check.")
        else:
            checked_pairs = 0
            min_dist_found = float('inf') # For logging/debugging

            for link1_name, link2_name in COLLISION_CHECK_PAIRS:
                # Check if both link poses were received in the last model_state message
                if link1_name in self.link_poses and link2_name in self.link_poses:
                    pos1 = self.link_poses[link1_name]
                    pos2 = self.link_poses[link2_name]

                    distance = np.linalg.norm(pos1 - pos2)
                    min_dist_found = min(min_dist_found, distance) # Track minimum distance for logging
                    checked_pairs += 1

                    if distance < COLLISION_THRESHOLD_DISTANCE:
                        # Calculate penalty (increases as distance decreases)
                        penalty = COLLISION_PENALTY_WEIGHT * (COLLISION_THRESHOLD_DISTANCE - distance)
                        proximity_penalty -= penalty # Subtract from total reward (add penalty)
                        # Increment actual collision counter if threshold is breached
                        # IMPORTANT: This simple increment might count multiple steps for one continuous collision.
                        # A more robust implementation might check for *new* collisions.
                        self.self_col_count += 1

                # else: # Optional: Log if a link needed for check is missing
                #    rospy.logwarn_throttle(5, "Skipping collision check pair ({}, {}): Link pose missing.".format(
                #       link1_name, link2_name))

            # Ensure penalty isn't applied if no pairs could be checked
            if checked_pairs == 0:
                 proximity_penalty = 0.0
                 # rospy.logwarn_throttle(5, "Collision check skipped: No valid link pairs found in self.link_poses.")
        # --- End Collision Penalty Calculation ---

        # 9. Action Rate Penalty (Penalize large action changes)
        action_rate_penalty = 0.0
        if self.prev_action is not None and action is not None:
            action_diff = np.sum(np.square(action - self.prev_action))
            # Scale penalty, potentially cap it if needed
            action_rate_penalty = -action_diff 

        # 10. Termination Penalty (Check for fall)
        # This check determines if the fall penalty should be applied in this step's reward
        is_fallen = abs(roll) > FALL_THRESHOLD_ROLL_PITCH or abs(pitch) > FALL_THRESHOLD_ROLL_PITCH
        # TODO: Consider adding base contact check if sensor exists
        # has_base_contact = self.check_base_contact()
        # is_fallen = is_fallen or has_base_contact

        fall_penalty_term = FALL_PENALTY if is_fallen else 0.0

        # --- Combine Rewards ---
        reward = (
            forward_reward +
            backward_penalty +
            STABILITY_WEIGHT * stability_reward +
            ANGULAR_VEL_WEIGHT * angular_vel_penalty +
            HEIGHT_WEIGHT * height_reward +
            ENERGY_WEIGHT * energy_penalty + # Note: This is the old energy penalty based on vel^2
            LATERAL_PENALTY_WEIGHT * lateral_penalty +
            ROTATION_PENALTY_WEIGHT * rotation_penalty +
            ORIENTATION_PENALTY_WEIGHT * orientation_penalty +
            ACTION_RATE_WEIGHT * action_rate_penalty +
            proximity_penalty + # Renamed from collision_penalty
            SURVIVAL_REWARD +
            fall_penalty_term +
            # --- ADDED new dynamic penalties ---
            -w_col * self.self_col_count +
            -w_bnc * (self.base_twist.linear.z if hasattr(self, 'base_twist') and self.base_twist is not None else 0.0) +
            -w_en * (np.sum(np.abs(np.array(self.last_joint_effort) * self.joint_velocities)) * dt if hasattr(self, 'last_joint_effort') and self.last_joint_effort is not None and self.joint_velocities is not None and dt > 0 else 0.0)
        )

        # --- Enhanced Logging for Reward Analysis (Python 2 compatible) ---
        log_prob = 0.02 # Log 2% of steps for better visibility
        if np.random.rand() < log_prob:
             # Calculate percentages of total reward (handle potential division by zero)
             reward_components = [
                 forward_reward, backward_penalty,
                 STABILITY_WEIGHT * stability_reward, ANGULAR_VEL_WEIGHT * angular_vel_penalty,
                 HEIGHT_WEIGHT * height_reward, ENERGY_WEIGHT * energy_penalty, # Old energy penalty
                 LATERAL_PENALTY_WEIGHT * lateral_penalty, ROTATION_PENALTY_WEIGHT * rotation_penalty,
                 ORIENTATION_PENALTY_WEIGHT * orientation_penalty, ACTION_RATE_WEIGHT * action_rate_penalty,
                 proximity_penalty, # Renamed
                 SURVIVAL_REWARD, fall_penalty_term,
                 -w_col * self.self_col_count,
                 -w_bnc * (self.base_twist.linear.z if hasattr(self, 'base_twist') and self.base_twist is not None else 0.0),
                 -w_en * (np.sum(np.abs(np.array(self.last_joint_effort) * self.joint_velocities)) * dt if hasattr(self, 'last_joint_effort') and self.last_joint_effort is not None and self.joint_velocities is not None and dt > 0 else 0.0)
             ]
             total_abs = sum(abs(x) for x in reward_components if x is not None) # Handle potential None
             if total_abs < 1e-6: # Avoid division by zero
                 total_abs = 1e-6
             
             # Use .format() syntax for Python 2
             log_comps = {
                 "Fwd":    "{:.2f} ({:.1f}%)".format(forward_reward, 100 * forward_reward / total_abs),
                 "Back":   "{:.2f} ({:.1f}%)".format(backward_penalty, 100 * backward_penalty / total_abs),
                 "Stab":   "{:.2f} ({:.1f}%)".format(STABILITY_WEIGHT * stability_reward, 100 * STABILITY_WEIGHT * stability_reward / total_abs),
                 "AngVel": "{:.2f} ({:.1f}%)".format(ANGULAR_VEL_WEIGHT * angular_vel_penalty, 100 * ANGULAR_VEL_WEIGHT * angular_vel_penalty / total_abs),
                 "Hght":   "{:.2f} ({:.1f}%)".format(HEIGHT_WEIGHT * height_reward, 100 * HEIGHT_WEIGHT * height_reward / total_abs),
                 "Enrg":   "{:.2f} ({:.1f}%)".format(ENERGY_WEIGHT * energy_penalty, 100 * ENERGY_WEIGHT * energy_penalty / total_abs),
                 "Lat":    "{:.2f} ({:.1f}%)".format(LATERAL_PENALTY_WEIGHT * lateral_penalty, 100 * LATERAL_PENALTY_WEIGHT * lateral_penalty / total_abs),
                 "Rot":    "{:.2f} ({:.1f}%)".format(ROTATION_PENALTY_WEIGHT * rotation_penalty, 100 * ROTATION_PENALTY_WEIGHT * rotation_penalty / total_abs),
                 "Orient": "{:.2f} ({:.1f}%)".format(ORIENTATION_PENALTY_WEIGHT * orientation_penalty, 100 * ORIENTATION_PENALTY_WEIGHT * orientation_penalty / total_abs),
                 "ActRate":"{:.2f} ({:.1f}%)".format(ACTION_RATE_WEIGHT * action_rate_penalty, 100 * ACTION_RATE_WEIGHT * action_rate_penalty / total_abs),
                 "ProxPen": "{:.2f}".format(proximity_penalty),
                 "Surv":   "{:.2f}".format(SURVIVAL_REWARD),
                 "Fall":   "{:.2f}".format(fall_penalty_term),
                 "SlfCol": "{:.2f}".format(-w_col * self.self_col_count),
                 "Bounce": "{:.2f}".format(-w_bnc * (self.base_twist.linear.z if hasattr(self, 'base_twist') and self.base_twist is not None else 0.0)),
                 "NrgUse": "{:.2f}".format(-w_en * (np.sum(np.abs(np.array(self.last_joint_effort) * self.joint_velocities)) * dt if hasattr(self, 'last_joint_effort') and self.last_joint_effort is not None and self.joint_velocities is not None and dt > 0 else 0.0)),
                 "Total":  "{:.2f}".format(reward)
             }
             rospy.logwarn("\nReward Component Breakdown:")
             for comp, value in sorted(log_comps.items()): # Sort for consistent order
                 rospy.logwarn("{:>8}: {}".format(comp, value))
             
             # Log additional state information using .format()
             rospy.logwarn("\nState Details:")
             rospy.logwarn("Position (x,y,z): ({:.2f}, {:.2f}, {:.2f})".format(current_pos[0], current_pos[1], current_pos[2]))
             rospy.logwarn("Orientation (r,p,y): ({:.2f}, {:.2f}, {:.2f})".format(roll, pitch, yaw))
             rospy.logwarn("Forward Movement: {:.3f}".format(forward_movement))
             # --- ADDED: Log Collision Check Info & Raw Energy Cost --- 
             if links_available and checked_pairs > 0:
                 rospy.logwarn("Min Leg Dist: {:.3f}m".format(min_dist_found))
             else:
                 rospy.logwarn("Min Leg Dist: N/A (Check skipped or failed)")
             if self.joint_velocities is not None:
                raw_energy_term = np.sum(np.square(self.joint_velocities))
                rospy.logwarn("Raw Energy Term (Sum Vel^2): {:.4f}".format(raw_energy_term))
             # --- End Added --- 
             rospy.logwarn("Height Diff: {:.3f}".format(height_diff))
             rospy.logwarn("Stability Cost: {:.3f}".format(stability_cost))
             rospy.logwarn("Angular Velocity (r,p,y): ({:.2f}, {:.2f}, {:.2f})".format(roll_rate, pitch_rate, yaw_rate))
             rospy.logwarn("----------------------------------------")

        # Update previous state for next step's calculation *before* returning
        self.prev_position = current_pos.copy()
        self.prev_orientation = (roll, pitch, yaw) # <<< MOVED HERE

        # Note: The actual episode termination based on 'is_fallen' or other conditions
        # must be handled by the environment's step function logic that calls this compute_reward.
        # This function now includes the *penalty* for falling in the reward calculation.

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


if __name__ == '__main__':
    try:
        bridge = JetHexaGymBridge()
        bridge.run()
    except rospy.ROSInterruptException:
        pass 