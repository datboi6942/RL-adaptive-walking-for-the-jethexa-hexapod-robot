#!/usr/bin/env python3
"""
JetHexa RL training script using Stable-Baselines3 with a ROS bridge to Gazebo.

This script runs in Python 3 and communicates with the Python 2 ROS node
via ROS topics, allowing us to use modern RL libraries while leveraging
ROS 1 Gazebo for simulation.
"""

import os
import time
import numpy as np
import gym
from gym import spaces
import rospy
import threading
import signal
import subprocess
import sys
from std_msgs.msg import Float64, Float32MultiArray, Bool, Int32
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
import matplotlib.pyplot as plt
from datetime import datetime
import yaml # For loading curriculum config
import argparse # Make sure argparse is imported

# Enable detailed debugging - Default to False
DEBUG = False

def debug_print(msg):
    if DEBUG:
        print(f"DEBUG: {msg}", flush=True)

# --- Explicit TensorBoard Import Check ---
try:
    import tensorboard
    # Use debug_print ONLY if DEBUG is True
    if DEBUG:
        debug_print("Explicit TensorBoard import successful!")
except ImportError as e:
    # Use debug_print ONLY if DEBUG is True
    if DEBUG:
        debug_print(f"Explicit TensorBoard import FAILED: {e}")
        debug_print(f"Python sys.path: {sys.path}") # Log path
# --- End Check ---

# Check Python version at startup
debug_print(f"Python version: {sys.version}")
debug_print(f"Python executable: {sys.executable}")
debug_print(f"Current directory: {os.getcwd()}")

# Try to import the CPG controller
try:
    debug_print("Attempting to import CPG controller...")
    
    # Check if the script is in the sys.path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    debug_print(f"Script directory: {script_dir}")
    if script_dir not in sys.path:
        debug_print(f"Adding script directory to sys.path: {script_dir}")
        sys.path.append(script_dir)
    
    # Check which files are in the directory
    debug_print(f"Files in script directory: {os.listdir(script_dir)}")
    
    # Try importing the controller
    from cpg_controller import CPGController
    debug_print("CPG controller imported successfully")
except ImportError as e:
    debug_print(f"ImportError importing CPG controller: {e}")
    debug_print("Trying alternative import method...")
    try:
        # Try to locate the file manually and import it
        cpg_file = os.path.join(script_dir, "cpg_controller.py")
        if os.path.exists(cpg_file):
            debug_print(f"CPG controller file exists at: {cpg_file}")
            import importlib.util
            spec = importlib.util.spec_from_file_location("cpg_controller", cpg_file)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            CPGController = module.CPGController
            debug_print("CPG controller imported using alternative method")
        else:
            debug_print(f"ERROR: CPG controller file not found at {cpg_file}")
    except Exception as e:
        debug_print(f"Alternative import failed: {e}")
        raise

try:
    debug_print("Checking for numpy...")
    debug_print(f"numpy version: {np.__version__}")
    
    debug_print("Checking for ROS Python...")
    debug_print(f"rospy exists: {rospy is not None}")
    
    debug_print("Checking for CPG controller...")
    debug_print(f"CPGController exists: {CPGController is not None}")
    
    debug_print("Checking torch...")
    debug_print(f"torch version: {torch.__version__}")
    
    debug_print("Checking for gym...")
    debug_print(f"gym version: {gym.__version__}")
    
    debug_print("Checking for stable_baselines3...")
    debug_print(f"PPO exists: {PPO is not None}")
except Exception as e:
    debug_print(f"Import check error: {e}")


class JetHexaROSEnv(gym.Env):
    """
    Gym environment that interfaces with the ROS bridge to communicate with Gazebo.
    
    This environment runs in Python 3 and uses ROS topics to send actions and
    receive observations from the Python 2 ROS node running Gazebo.
    """
    
    def __init__(self):
        super(JetHexaROSEnv, self).__init__()
        debug_print("Initializing JetHexaROSEnv")
        
        try:
            # Initialize ROS node (anonymous to allow multiple instances)
            debug_print("About to initialize ROS node...")
            # rospy.init_node('jethexa_rl_training', anonymous=True, disable_signals=True) # Commented out: Initialization should be handled by the launch script
            debug_print("ROS node initialization skipped (handled by launch script)")
            
            # CPG Controller for generating joint positions from actions
            debug_print("Creating CPG controller...")
            self.cpg_controller = CPGController(n_legs=6, n_joints_per_leg=3)
            debug_print("CPG controller created")
            
            # Define action and observation spaces
            self.n_cpg_params = 2 + 6 + 18  # global_freq, gait_type, leg_phases, joint_amplitudes
            self.action_space = spaces.Box(low=0.0, high=1.0, shape=(self.n_cpg_params,), dtype=np.float32)
            
            # Observation: joint positions, joint velocities, robot pose, IMU data
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(18 + 18 + 6 + 9,), dtype=np.float32
            )
            
            # Initialize observation, reward, done variables
            self.observation = np.zeros(self.observation_space.shape[0], dtype=np.float32)
            self.reward = 0.0
            self.done_from_bridge = False # Done signal received from bridge (fall condition)
            self.info = {}
            
            # Synchronization
            self.obs_received = False
            self.reset_complete = False
            self.lock = threading.Lock()
            
            # Episode Timeout Logic
            self.max_steps_per_episode = 1000
            self.current_step_count = 0
            self.sim_step_dt = 0.05 # Target simulation time step (e.g., 20 Hz)
            self.last_log_time = time.time() # Timer for joint logging
            self.log_interval = 1.0 # Log every 1 second
            self.smoothed_joint_positions = None # Initialize smoothed joint positions state
            self.action_smoothing_alpha = 0.3 # Smoothing factor (lower = more smoothing)

            # Setup joint angle logging file
            log_dir = os.path.join(os.path.dirname(__file__), "../logs")
            os.makedirs(log_dir, exist_ok=True)
            self.joint_log_path = os.path.join(log_dir, "joint_target_log.csv")
            # Setup CPG parameter logging file
            self.cpg_params_log_path = os.path.join(log_dir, "cpg_params_log.csv")
            try:
                with open(self.joint_log_path, 'w') as f:
                    header = "timestamp," + ",".join([f"joint_{i}" for i in range(18)]) + "\n"
                    f.write(header)
                debug_print(f"Initialized joint target log at: {self.joint_log_path}")
                # Initialize CPG params log file
                with open(self.cpg_params_log_path, 'w') as f:
                    # Assuming 6 CPG parameters based on previous logs/common practice
                    # If this number is wrong, the header/logging needs adjustment
                    header = "timestamp," + ",".join([f"param_{i}" for i in range(27)]) + "\n"
                    f.write(header)
                debug_print(f"Initialized CPG params log at: {self.cpg_params_log_path}")
            except Exception as e:
                 debug_print(f"ERROR initializing log files: {e}")
            
            # Setup ROS publishers (to Python 2 bridge)
            debug_print("Setting up ROS publishers...")
            self.action_pub = rospy.Publisher('/jethexa_rl/action', Float32MultiArray, queue_size=1)
            self.reset_pub = rospy.Publisher('/jethexa_rl/reset', Bool, queue_size=1)
            self.difficulty_pub = rospy.Publisher('/jethexa_rl/set_difficulty', Int32, queue_size=1)
            debug_print("ROS publishers created")
            
            # Setup ROS subscribers (from Python 2 bridge)
            debug_print("Setting up ROS subscribers...")
            rospy.Subscriber('/jethexa_rl/observation', Float32MultiArray, self.observation_callback)
            rospy.Subscriber('/jethexa_rl/reward', Float64, self.reward_callback)
            rospy.Subscriber('/jethexa_rl/done', Bool, self.done_callback)
            rospy.Subscriber('/jethexa_rl/info', Float32MultiArray, self.info_callback)
            rospy.Subscriber('/jethexa_rl/reset_complete', Bool, self._reset_complete_callback)
            debug_print("ROS subscribers set up")
            
            # Wait for connections to be established
            debug_print("Waiting for connections to establish...")
            time.sleep(1)
            debug_print("JetHexa ROS Gym environment initialized successfully")
            
            # Check if ROS is running properly
            debug_print(f"ROS Master URI: {os.environ.get('ROS_MASTER_URI', 'Not set')}")
            try:
                if not rospy.is_shutdown():
                    debug_print("ROS node is active")
                else:
                    debug_print("WARNING: ROS node is shutdown")
            except Exception as e:
                debug_print(f"Error checking ROS status: {e}")
                
        except Exception as e:
            debug_print(f"ERROR during initialization: {e}")
            raise
    
    def observation_callback(self, msg):
        """
        Callback for receiving observations from the ROS bridge.
        """
        try:
            # Track callback timing
            current_time = time.time()
            if hasattr(self, 'last_callback_time'):
                time_diff = current_time - self.last_callback_time
                if time_diff > 0.1:  # Log if callbacks are more than 100ms apart
                    debug_print(f"Long delay between callbacks: {time_diff:.3f}s")
            self.last_callback_time = current_time

            with self.lock:
                debug_print("Observation Callback Triggered!")
                self.observation = np.array(msg.data)
                self.obs_received = True
                
                # Periodically check observation validity
                if self.current_step_count % 100 == 0:
                    if not np.all(np.isfinite(self.observation)):
                        debug_print(f"WARNING: Non-finite values in observation at step {self.current_step_count}")
                        debug_print(f"Observation: {self.observation}")
                    
        except Exception as e:
            debug_print(f"ERROR in observation_callback: {e}")
            import traceback
            debug_print(traceback.format_exc())
            
    def reward_callback(self, msg):
        """
        Callback for receiving rewards from the ROS bridge.
        """
        try:
            with self.lock:
                self.reward = msg.data
                # Log extreme rewards
                if abs(self.reward) > 10.0:
                    debug_print(f"Large reward received at step {self.current_step_count}: {self.reward}")
        except Exception as e:
            debug_print(f"ERROR in reward_callback: {e}")
            import traceback
            debug_print(traceback.format_exc())
            
    def done_callback(self, msg):
        """
        Callback for receiving done signals from the ROS bridge.
        """
        try:
            with self.lock:
                self.done_from_bridge = msg.data
                if self.done_from_bridge:
                    debug_print(f"Episode terminated by bridge at step {self.current_step_count}")
        except Exception as e:
            debug_print(f"ERROR in done_callback: {e}")
            import traceback
            debug_print(traceback.format_exc())
            
    def info_callback(self, msg):
        """
        Callback for receiving info dictionary from the ROS bridge.
        """
        try:
            with self.lock:
                # Check if data is a tuple/list (as expected from Float32MultiArray)
                if isinstance(msg.data, (list, tuple)):
                    # Store raw data for now. The Monitor wrapper might extract needed info.
                    # Or, we might need to update the bridge node to send a dict-like string.
                    self.info = {'raw_bridge_info': msg.data} # Store it under a specific key
                    # Log a warning only once to avoid spamming
                    if not hasattr(self, '_info_warning_logged'):
                         debug_print(f"Warning: Received info as {type(msg.data)}. Storing raw data. Consider updating bridge node to send JSON string if dict is needed.")
                         self._info_warning_logged = True
                elif isinstance(msg.data, str):
                    # If it IS a string somehow, try to eval (original intent)
                    try:
                        self.info = eval(msg.data)
                    except Exception as eval_e:
                        debug_print(f"ERROR evaluating info string: {eval_e}")
                        self.info = {'eval_error': str(eval_e)}
                else:
                    # Handle unexpected types
                    debug_print(f"Warning: Received info with unexpected data type: {type(msg.data)}")
                    self.info = {'unknown_info_type': str(msg.data)}
                
                # Log any error messages in info (if it was successfully parsed as dict)
                if isinstance(self.info, dict) and 'error' in self.info:
                    debug_print(f"Error message in info at step {self.current_step_count}: {self.info['error']}")
        except Exception as e:
            debug_print(f"ERROR in info_callback: {e}")
            import traceback
            debug_print(traceback.format_exc())
    
    def _reset_complete_callback(self, msg):
        """Callback for receiving reset completion signal from the ROS bridge."""
        with self.lock:
            self.reset_complete = bool(msg.data)
            # debug_print(f"Received reset complete signal: {self.reset_complete}") # COMMENTED OUT - Too verbose
    
    def step(self, action):
        debug_print(f"Step {self.current_step_count}: Entered step method.")
        # <<< ADDED: Log raw action from policy >>>
        debug_print(f"Step {self.current_step_count}: Raw Action Received: {np.round(action, 3)}") 
        # <<< END LOG >>>
        self.current_step_count += 1 # Increment step counter
        
        # Add periodic step debugging
        if self.current_step_count % 500 == 0:  # <<< INCREASED: Log every 500 steps (was 100)
            debug_print(f"Environment step {self.current_step_count}")
            # debug_print(f"ROS node status: {'shutdown' if rospy.is_shutdown() else 'active'}") # Still potentially verbose
        
        with self.lock:
            self.obs_received = False
            self.done_from_bridge = False # Reset bridge done flag for this step
        
        # --- Log the raw CPG parameters received ---
        # debug_print(f"STEP {self.current_step_count} - Raw CPG Action Params: {np.round(action, 3)}") # COMMENTED OUT - Too verbose
        # Log raw CPG action parameters to file
        current_time_cpg = time.time() # Use a separate timestamp for this log if needed, or reuse
        try:
            with open(self.cpg_params_log_path, 'a') as f:
                # Assuming action is a numpy array with the CPG params
                if isinstance(action, np.ndarray):
                     log_line = f"{current_time_cpg:.4f}," + ",".join([f"{param:.4f}" for param in action]) + "\n"
                     f.write(log_line)
                else:
                    debug_print(f"Warning: step() received non-numpy action of type: {type(action)}")
        except Exception as e:
            debug_print(f"ERROR writing to CPG params log file in step(): {e}")
        # -------------------------------------------

        # Convert high-level parameters to joint angles using CPG
        try:
            # Use raw action here, smoothing will happen on output
            self.cpg_controller.set_gait_params(action) 
            joint_positions = self.cpg_controller.update(self.sim_step_dt)
        except Exception as e:
            debug_print(f"ERROR in CPG controller during step(): {e}")
            import traceback
            debug_print(traceback.format_exc())
            # Return a failed state
            return self.observation.copy(), -1.0, True, self.info
        
        # --- Apply Smoothing to Joint Positions ---
        if self.smoothed_joint_positions is None:
            self.smoothed_joint_positions = joint_positions.copy()
        else:
            self.smoothed_joint_positions = self.action_smoothing_alpha * joint_positions + \
                                         (1 - self.action_smoothing_alpha) * self.smoothed_joint_positions
        
        # Send the joint positions through ROS
        try:
            action_msg = Float32MultiArray()
            action_msg.data = self.smoothed_joint_positions.tolist()
            debug_print(f"Step {self.current_step_count}: Publishing action...")
            self.action_pub.publish(action_msg)
        except Exception as e:
            debug_print(f"ERROR publishing action in step(): {e}")
            import traceback
            debug_print(traceback.format_exc())
            # Return a failed state
            return self.observation.copy(), -1.0, True, self.info
        
        # Wait for observation to be received
        timeout = time.time() + 12.0 
        start_wait = time.time()
        wait_logged = False # Flag to log wait message only once
        while not self.obs_received and time.time() < timeout:
            if not wait_logged:
                 debug_print(f"Step {self.current_step_count}: Waiting for observation...")
                 wait_logged = True
            time.sleep(0.01)
            
        if not self.obs_received:
            debug_print(f"WARNING: Step {self.current_step_count} timed out waiting for observation after {time.time() - start_wait:.2f}s")
            # If observation isn't received, assume episode ended poorly
            # Return current state but mark as done
            with self.lock:
                 return self.observation.copy(), -1.0, True, self.info # Timeout on comms
        
        # Check for episode timeout
        timeout_done = self.current_step_count >= self.max_steps_per_episode
        if timeout_done:
            debug_print(f"Episode timeout reached at step {self.current_step_count}")
            
        # Determine final done state (fall from bridge OR timeout)
        final_done = self.done_from_bridge or timeout_done
        
        # Return the latest observation, reward, done, info
        with self.lock:
            return self.observation.copy(), self.reward, final_done, self.info
    
    def reset(self):
        """
        Reset the environment by sending a reset signal to the ROS bridge.
        
        Returns:
            Initial observation
        """
        debug_print("reset() called")
        # Reset step counter for the new episode
        self.current_step_count = 0 
        
        with self.lock:
            self.reset_complete = False
        
        # Send reset signal to bridge
        # debug_print("Publishing reset signal") # <<< COMMENTED OUT - redundant with reset() call log
        self.reset_pub.publish(Bool(True))
        
        # Wait for reset to complete
        timeout = time.time() + 20.0  # Increased timeout to 20 seconds (was 15)
        debug_print("Waiting for reset to complete...")
        while not self.reset_complete and time.sleep(0.01):
            pass
        
        if not self.reset_complete:
            debug_print("WARNING: Reset timed out")
        else:
            debug_print("Reset completed successfully")
        
        # Return the initial observation
        with self.lock:
            return self.observation.copy()
    
    def set_difficulty(self, level):
        """Set the terrain difficulty level (0-4)."""
        debug_print(f"Setting difficulty to level {level}")
        self.difficulty_pub.publish(Int32(level))


def plot_training_results(log_dir):
    """Plot training rewards and episode lengths from monitor files."""
    import pandas as pd
    
    # Load monitor files
    monitor_files = [os.path.join(log_dir, file) 
                     for file in os.listdir(log_dir) 
                     if file.startswith('monitor')]
    
    if not monitor_files:
        print("No monitor files found in", log_dir)
        return
    
    # Load the data
    data_frames = []
    for file in monitor_files:
        try:
            df = pd.read_csv(file, skiprows=1)
            data_frames.append(df)
        except:
            print(f"Error reading {file}")
    
    if not data_frames:
        print("No valid data found")
        return
    
    # Concatenate all dataframes
    data = pd.concat(data_frames)
    
    # Plot rewards
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(data['r'])
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    
    plt.subplot(1, 2, 2)
    plt.plot(data['l'])
    plt.title('Episode Lengths')
    plt.xlabel('Episode')
    plt.ylabel('Length (steps)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, 'training_progress.png'))
    plt.close()


# --- Custom Callback to Save VecNormalize ---
class SaveVecNormalizeCheckpointCallback(CheckpointCallback):
    """
    A custom callback that inherits from CheckpointCallback and saves
    the VecNormalize statistics when saving the model checkpoint.
    """
    def _on_step(self) -> bool:
        # First, call the parent's _on_step method to handle model saving
        continue_training = super()._on_step()

        # Check if the parent actually saved the model in this step
        # We use self.n_calls because self.last_checkpoint_step might not be updated
        # reliably depending on SB3 version specifics when resumed.
        # Checking if n_calls is a multiple of save_freq is more robust.
        if continue_training and self.n_calls % self.save_freq == 0:
            # Check if self.training_env is VecNormalize
            if isinstance(self.training_env, VecNormalize):
                # Construct the VecNormalize save path based on the model checkpoint path
                # Replace the .zip extension with _vecnormalize.pkl
                
                # --- UPDATED PATH LOGIC ---
                # OLD: Relied on attribute potentially not set by parent class
                # model_path = self.last_checkpoint_path # Path saved by parent
                
                # NEW: Construct the path manually using known attributes
                model_path = os.path.join(self.save_path, f"{self.name_prefix}_{self.num_timesteps}_steps.zip")
                debug_print(f"Constructed model path for VecNormalize saving: {model_path}")
                # --- END UPDATED PATH LOGIC ---
                
                if model_path: # Ensure a path was actually saved/constructed
                    # --- Refined path generation ---
                    # vec_normalize_path = model_path.replace(\".zip\", \"_vecnormalize.pkl\") # Old way
                    base_path, _ = os.path.splitext(model_path) # Separate base path from .zip
                    vec_normalize_path = base_path + "_vecnormalize.pkl" # Add new extension
                    # --- End refined path generation ---
                    
                    debug_print(f"Attempting to save VecNormalize stats to: {vec_normalize_path}")
                    try:
                        debug_print(f"---> Calling self.training_env.save('{vec_normalize_path}')...")
                        self.training_env.save(vec_normalize_path)
                        debug_print(f"---> self.training_env.save() completed successfully.")
                        debug_print(f"VecNormalize stats saved successfully.") # Original success message
                    except Exception as e:
                         # This might not catch all crash types, but will catch Python exceptions
                         debug_print(f"!!! ERROR saving VecNormalize stats: {e}")
                         import traceback
                         debug_print(traceback.format_exc())
                else:
                    debug_print("Warning: Could not construct model path for VecNormalize saving.") # Updated warning
            else:
                debug_print("Warning: training_env is not VecNormalize, skipping VecNormalize save.")

        return continue_training
# --- End Custom Callback ---


def train_ppo(args, total_timesteps=1000000, curriculum=True):
    """
    Train a PPO agent for the JetHexa robot.
    
    Args:
        total_timesteps: Total number of timesteps to train for
        curriculum: Whether to use curriculum learning
    """
    debug_print("Starting train_ppo function")
    
    # Note: Removed rostopic check - communication handled by bridge.

    # Create log and base model directories
    log_dir = "logs"
    base_model_dir = "models" # Renamed to avoid confusion
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(base_model_dir, exist_ok=True)
    debug_print(f"Base log directory set to: {os.path.abspath(log_dir)}") # Added absolute path log
    debug_print(f"Base model directory set to: {os.path.abspath(base_model_dir)}")

    # Generate timestamped directory for this specific run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_specific_model_dir = os.path.join(base_model_dir, f"jethexa_ppo_{timestamp}")
    os.makedirs(run_specific_model_dir, exist_ok=True)
    debug_print(f"Run-specific model directory created: {run_specific_model_dir}")
    
    # Create environment
    debug_print("Creating vectorized environment wrapper - STEP 1: Defining make_env")
    def make_env():
        debug_print("make_env() called - STEP 2: Creating JetHexaROSEnv instance")
        env_instance = JetHexaROSEnv()
        debug_print("make_env() finished - STEP 3: JetHexaROSEnv instance created")
        # Explicitly wrap with Monitor, logging to the main log_dir
        # debug_print(f"make_env() - STEP 3a: Wrapping with Monitor, logging to {log_dir}") # <<< COMMENTED OUT - setup detail
        env_instance = Monitor(env_instance, log_dir)
        # debug_print("make_env() - STEP 3b: Monitor wrapper applied") # <<< COMMENTED OUT - setup detail
        return env_instance
    
    debug_print("STEP 4: Setting up DummyVecEnv")
    env = DummyVecEnv([make_env])
    debug_print("STEP 5: DummyVecEnv created")
    # debug_print(f"STEP 5a: Type of env after DummyVecEnv: {type(env)}") # Log env type <<< COMMENTED OUT - less important

    # Check if loading VecNormalize
    debug_print("STEP 6: Checking for VecNormalize loading...") # Added log
    if args.load_vecnormalize and os.path.exists(args.load_vecnormalize):
        debug_print(f"STEP 6a: Loading VecNormalize stats from: {args.load_vecnormalize}")
        try:
            env = VecNormalize.load(args.load_vecnormalize, env)
            debug_print("STEP 6b: VecNormalize loaded successfully.") # Moved log here
            debug_print(f"STEP 6c: Type of env after VecNormalize.load: {type(env)}") # Log env type
            env.training = True
            env.norm_reward = True # Ensure reward normalization is enabled when loading
            debug_print("STEP 6d: VecNormalize flags set (training=True, norm_reward=True).")
        except Exception as e:
            debug_print(f"ERROR loading VecNormalize: {e}")
            import traceback
            debug_print(traceback.format_exc())
            return None # Exit if loading fails
    else:
        debug_print("STEP 6: Initializing new VecNormalize stats.")
        env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)
        debug_print("STEP 6b: New VecNormalize initialized.")
        debug_print(f"STEP 6c: Type of env after new VecNormalize: {type(env)}") # Log env type

    # Check if loading Model
    model = None # Initialize model to None
    debug_print("STEP 7: Checking for PPO model loading...")

    # --- Add Debugging for Loading --- 
    # debug_print(f"  Args received: load_model='{args.load_model}', load_vecnormalize='{args.load_vecnormalize}'") # <<< COMMENTED OUT - Covered by step 7a
    if args.load_model and os.path.exists(args.load_model):
        debug_print(f"STEP 7a: Loading model from: {args.load_model}")
        try:
            # <<< Added Debug for TensorBoard Path >>>
            debug_print(f"--> Passing tensorboard_log='{log_dir}' to PPO.load()")
            model = PPO.load(
                args.load_model,
                env=env,
                tensorboard_log=log_dir, # Ensure this is passed
                verbose=1 
                # print_system_info=True # Optional: adds system info to logs
            )
            debug_print(f"STEP 7b: Model loaded. Current timesteps: {model.num_timesteps}")
            debug_print("STEP 7c: Environment implicitly associated.")
            
            # <<< Set Learning Rate for Fine-tuning >>>
            if args.learning_rate is not None:
                initial_lr_finetune = args.learning_rate
                debug_print(f"--> Fine-tuning: Setting initial learning rate to {initial_lr_finetune} with LINEAR schedule.")
                # Define the linear schedule function
                def linear_schedule_finetune(progress_remaining: float) -> float:
                    """
                    Linear learning rate schedule.
                    :param progress_remaining: progress remaining (1 to 0)
                    :return: learning rate
                    """
                    return progress_remaining * initial_lr_finetune
                # Assign the schedule function to the model's learning rate
                model.learning_rate = linear_schedule_finetune
            else:
                 # If no LR is specified, use whatever was saved with the model (could be a constant or a schedule)
                 debug_print(f"--> Using learning rate/schedule loaded from model: {model.learning_rate}")
            # <<< End Set Learning Rate >>>

            # ==== Force load VecNormalize ==== 
            if args.load_vecnormalize:
                vec_path = args.load_vecnormalize 
                debug_print(f"Attempting to force load VecNormalize stats from: {vec_path}")
                assert os.path.isfile(vec_path), "VecNormalize file missing: " + vec_path
                # Load into a temporary VecNormalize wrapper to get stats
                try:
                     vecnorm_loaded = VecNormalize.load(vec_path, DummyVecEnv([lambda: env.envs[0]])) # Need dummy env to load
                     # Apply the loaded stats to the actual environment wrapper
                     if isinstance(env, VecNormalize):
                          env.obs_rms = vecnorm_loaded.obs_rms
                          env.ret_rms = vecnorm_loaded.ret_rms
                          env.training = False # Set to inference mode
                          env.norm_obs = True # Ensure observation normalization is active
                          env.norm_reward = False # Usually false for inference
                          debug_print("Successfully force-loaded and applied VecNormalize statistics.")
                     else:
                          debug_print("Warning: Environment is not VecNormalize wrapped. Cannot apply loaded stats.")
                     del vecnorm_loaded # Clean up temporary wrapper
                except Exception as e:
                     debug_print(f"Error force-loading VecNormalize: {e}. Proceeding without applying stats.")
            else:
                 # Ensure VecNormalize wrapper (if present) is in inference mode even if not loading
                 if isinstance(env, VecNormalize):
                      env.training = False
                      env.norm_reward = False
                      debug_print("Running in inference mode (VecNormalize training=False).")
        except Exception as e:
            debug_print(f"ERROR loading PPO Model: {e}")
            import traceback
            debug_print(traceback.format_exc())
            return None # Exit if loading fails
    else:
        debug_print("STEP 7: Creating NEW PPO model")
        policy_kwargs = dict(
            activation_fn=torch.nn.ReLU,
            net_arch=[dict(pi=[256, 256], vf=[256, 256])]
        )
        
        # Use provided LR if given, otherwise use default schedule
        initial_lr = args.learning_rate if args.learning_rate is not None else 5e-5
        debug_print(f"--> Using initial learning rate for new model: {initial_lr}")
        def lr_schedule(progress_remaining):
            return progress_remaining * initial_lr
            
        # <<< Added Debug for TensorBoard Path >>>
        debug_print(f"--> Passing tensorboard_log='{log_dir}' to PPO() constructor")
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=lr_schedule if args.learning_rate is None else initial_lr, # Pass float if specified
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.002,
            policy_kwargs=policy_kwargs,
            tensorboard_log=log_dir, # Ensure this is passed
            verbose=1
        )
        debug_print("STEP 7b: NEW PPO model created")
        debug_print(f"STEP 7c: New model timesteps: {model.num_timesteps}")
    
    # Test the environment with a reset (AFTER model loading/creation)
    debug_print("STEP 8: Testing environment with initial reset operation AFTER model/vecnorm loading/creation") # Clarified log
    try:
        debug_print("STEP 8a: --- Initiating env.reset() --- ")
        obs = env.reset()
        debug_print(f"STEP 8b: --- env.reset() completed. Observation shape: {obs.shape if obs is not None else 'None'} ---")
        if obs is None:
             debug_print("ERROR: Initial env.reset() returned None! Check ROS bridge.")
             return None # Exit if reset fails critically
    except Exception as e:
        debug_print(f"ERROR during initial environment reset: {e}")
        import traceback
        debug_print(traceback.format_exc())
        debug_print("This likely indicates a problem with the ROS bridge communication during reset. Exiting.")
        return None # Exit if reset fails
    debug_print("STEP 8c: Initial environment reset successful.")

    # Create checkpoint callback pointing to the run-specific directory
    # --- USE Custom SaveVecNormalizeCheckpointCallback ---
    checkpoint_callback = SaveVecNormalizeCheckpointCallback( # NEW custom callback
        save_freq=20000,
        save_path=run_specific_model_dir,
        name_prefix="ppo_jethexa",
        verbose=1 # Add verbose logging for the callback itself
    )
    # --------------------------------------------------------------------
    
    # Implement curriculum learning if enabled
    if curriculum:
        # Initialize at the easiest difficulty
        # debug_print("Setting up curriculum learning, starting at difficulty 0") # Moved/changed below
        # env.env_method('set_difficulty', 0) # Moved/changed below
        
        # Define difficulty thresholds and corresponding levels
        difficulty_thresholds = {
            0: 250000,   # MORE TIME AT LEVEL 0: Timesteps before moving to difficulty 1
            1: 400000,   # Shift subsequent thresholds accordingly
            2: 600000,   # Shift subsequent thresholds accordingly
            3: 800000    # Shift subsequent thresholds accordingly
        }
        # NEW: Determine starting difficulty based on loaded timesteps
        starting_difficulty = 0
        # Check if model was loaded (timesteps_so_far > 0)
        if model and model.num_timesteps > 0:
            for level, threshold in sorted(difficulty_thresholds.items()):
                if model.num_timesteps >= threshold:
                    starting_difficulty = level + 1 # Start at the level *after* the threshold passed
                else:
                    break # Stop checking once a threshold hasn't been met
            starting_difficulty = min(starting_difficulty, 4) # Cap at max difficulty
            debug_print(f"Loaded model at {model.num_timesteps} steps. Setting initial difficulty to {starting_difficulty}")
            env.env_method('set_difficulty', starting_difficulty)
        else:
            # Starting from scratch, set difficulty to 0
            debug_print("Starting new model. Setting initial difficulty to 0")
            env.env_method('set_difficulty', 0)
        
        # Track current difficulty level
        current_difficulty = starting_difficulty
    
    # Training loop
    debug_print("STEP 9: Entering main training loop section")
    # save_interval = 20000  # Define the save frequency
    # last_save_timestep = model.num_timesteps # Initialize based on loaded/new model
    # debug_print(f"STEP 9.5: Initializing training loop. last_save_timestep={last_save_timestep}, save_interval={save_interval}")
    try:
        # Ensure model is actually defined before accessing num_timesteps
        if model is None:
             debug_print("ERROR: Model is None before training loop!")
             return None
             
        debug_print("STEP 9a: Checking model.num_timesteps before loop...")
        timesteps_so_far = model.num_timesteps
        debug_print(f"STEP 10: Starting single model.learn() call. Current steps: {timesteps_so_far}, Target: {total_timesteps}")
        
        # --- Pre-learn() Environment Health Check ---
        debug_print("Performing pre-learn() health checks...")
        try:
            # Check ROS node status
            if rospy.is_shutdown():
                debug_print("ERROR: ROS node is shutdown before learn()")
                return None
            
            # Verify environment is responsive
            debug_print("Testing environment reset before learn()...")
            test_obs = env.reset()
            if test_obs is None:
                debug_print("ERROR: Environment reset failed before learn()")
                return None
            debug_print("Environment reset successful")
            
            # Check model and environment compatibility
            # debug_print(f"Model observation space: {model.observation_space}") # <<< COMMENTED OUT - verbose
            # debug_print(f"Environment observation space: {env.observation_space}") # <<< COMMENTED OUT - verbose
            
            # Log memory usage
            import psutil
            process = psutil.Process()
            debug_print(f"Memory usage before learn(): {process.memory_info().rss / 1024 / 1024:.2f} MB")
        except Exception as e:
            debug_print(f"ERROR in pre-learn() health checks: {e}")
            import traceback
            debug_print(traceback.format_exc())
            return None
            
        # --- Call learn() ONCE for the total duration --- 
        debug_print("@@@ STARTING model.learn() @@@")
        try:
            model.learn(
                total_timesteps=total_timesteps, # Target the overall total timesteps
                reset_num_timesteps=False, # Continue counting from loaded model
                log_interval=1, # SB3 internal logging frequency (prints table every N rollouts)
                callback=checkpoint_callback # <<< Use the custom callback instance
            )
            debug_print("@@@ model.learn() COMPLETED NORMALLY @@@")
        except KeyboardInterrupt:
            debug_print("@@@ model.learn() interrupted by Ctrl+C @@@")
            raise  # Re-raise to trigger the finally block
        except Exception as e:
            debug_print(f"@@@ ERROR in model.learn(): {str(e)} @@@")
            import traceback
            debug_print(traceback.format_exc())
            # Check ROS and environment state after error
            debug_print(f"ROS node status after error: {'shutdown' if rospy.is_shutdown() else 'active'}")
            try:
                test_obs = env.reset()
                debug_print("Environment still responsive after error")
            except:
                debug_print("Environment unresponsive after error")
            raise  # Re-raise to trigger the finally block
        
        debug_print("--- Single model.learn() call completed --- ")
        # --- END Single learn() call --- 

        # --- COMMENTED OUT: Original while loop --- 
        # debug_print("STEP 10a: --- Entering main training loop --- ")
        # while timesteps_so_far < total_timesteps:
            # --- Curriculum Difficulty Update ---
            # ... (commented out) ... 
            # --- End Curriculum Update ---

            # Determine steps for this iteration: model's n_steps or remaining steps
            # ... (commented out) ...

            # print("@@@ REACHED CODE AFTER LEARN @@@", flush=True) # Simplest possible check
            
            # Update total timesteps AFTER learn() completes
            # timesteps_so_far = model.num_timesteps
            # debug_print(f"Current total timesteps: {timesteps_so_far}/{total_timesteps}") # Keep as debug for now

            # --- Precise Saving Logic ---
            # ... (commented out) ...
            # --- End Saving Logic ---
        # --- END COMMENTED OUT: Original while loop ---

        # Save the final model to the run-specific directory
        debug_print("Training loop section complete, saving final model")
        # Ensure final save uses the absolute latest timestep count
        final_timesteps = model.num_timesteps
        model.save(os.path.join(run_specific_model_dir, f"ppo_jethexa_{final_timesteps}_final")) # Add final timestep to name
        env.save(os.path.join(run_specific_model_dir, f"vec_normalize_{final_timesteps}_final.pkl")) # Add final timestep to name
        
        # Plot training results
        debug_print("Plotting training results")
        plot_training_results(log_dir)
        
        debug_print("Training complete!")

    finally:
        # Ensure environment is closed AND SAVE MODEL ON EXIT
        try:
            if 'model' in locals() and model and 'env' in locals() and env and 'run_specific_model_dir' in locals():
                 # Save final state on exit (normal or exception)
                 final_timesteps = model.num_timesteps
                 model_path = os.path.join(run_specific_model_dir, f"ppo_jethexa_{final_timesteps}_steps_EXIT") # Add EXIT tag
                 vec_path = os.path.join(run_specific_model_dir, f"vec_normalize_{final_timesteps}_steps_EXIT.pkl")
                 debug_print(f"Saving final state on exit to {model_path}.zip")
                 try:
                      model.save(model_path)
                      env.save(vec_path)
                      debug_print("Final state saved successfully.")
                 except Exception as e:
                      debug_print(f"ERROR saving final state: {e}")

            if 'env' in locals() and env: 
                 debug_print("Closing environment after training loop/exception...")
                 env.close()
        except NameError: # env might not be defined if init failed
             pass
        except Exception as e:
             debug_print(f"Error during final save or close in finally block: {e}")

    # Return model only on successful completion (i.e., if no exception occurred in the try block)
    return model


def run_and_visualize_policy(model_path, vec_normalize_path=None, episodes=3):
    """
    Run and visualize a trained policy.
    
    Args:
        model_path: Path to the saved model
        vec_normalize_path: Path to the saved VecNormalize object
        episodes: Number of episodes to run
    """
    # Create environment
    env = JetHexaROSEnv()
    
    # Load VecNormalize stats if available
    if vec_normalize_path and os.path.exists(vec_normalize_path):
        env = VecNormalize.load(vec_normalize_path, DummyVecEnv([lambda: env]))
        # Don't update the normalization statistics during evaluation
        env.training = False
        env.norm_reward = False
    else:
        env = DummyVecEnv([lambda: env])
    
    # Load the trained model
    model = PPO.load(model_path, env=env)
    
    # Run the policy
    for episode in range(episodes):
        obs = env.reset()
        done = False
        total_reward = 0
        step = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += reward
            step += 1
            
            print(f"Episode {episode+1}, Step {step}, Reward: {reward:.2f}, Done: {done}")
            if step % 10 == 0:
                print(f"  Distance: {info[0]['distance_traveled']:.2f}, Falls: {info[0]['falls']}")
            
            # Small delay for visualization
            time.sleep(0.01)
        
        print(f"Episode {episode+1} finished with total reward: {total_reward:.2f}")
    
    # Close the environment
    env.close()


if __name__ == "__main__":
    import argparse
    
    # No debug print here, DEBUG is False by default
    # debug_print("Script started") 
    
    parser = argparse.ArgumentParser(description='Train a PPO agent on the JetHexa environment')
    parser.add_argument('--timesteps', type=int, default=1000000, help='Total number of timesteps to train for')
    parser.add_argument('--curriculum', action='store_true', help='Enable curriculum learning')
    parser.add_argument('--load-model', type=str, default=None, help='Path to saved model .zip file to load and continue training')
    parser.add_argument('--load-vecnormalize', type=str, default=None, help='Path to saved VecNormalize .pkl file to load')
    parser.add_argument('--learning-rate', type=float, default=None, help='Set a specific learning rate for new training or fine-tuning') # <<< Added LR arg >>>
    parser.add_argument('--debug', action='store_true', help='Enable detailed debug printing')
    args = parser.parse_args()

    # Update global DEBUG variable if --debug flag is present
    if args.debug:
        DEBUG = True
        # Now debug_print works
        debug_print("--- DEBUGGING ENABLED via command line ---")

        # --- Explicit TensorBoard Import Check (Moved Here) ---
        debug_print("Running explicit TensorBoard import check...")
        try:
            import tensorboard
            debug_print("Explicit TensorBoard import successful!")
        except ImportError as e:
            debug_print(f"Explicit TensorBoard import FAILED: {e}")
            debug_print(f"Python sys.path: {sys.path}") # Log path
        # --- End Check ---

    # Now conditional debug prints will work
    debug_print("Script started (DEBUGGING ENABLED)")

    model = None # Initialize model to None
    try:
        # RE-ADD: Initialize ROS node for this Python 3 process
        # Use a unique name and anonymous=True
        debug_print("Initializing ROS node for Python 3 script...")
        rospy.init_node('jethexa_rl_training_py3', anonymous=True)
        debug_print("ROS node for Python 3 script initialized.")
        
        # Create model directory if it doesn't exist (Still useful)
        model_dir = "models"
        os.makedirs(model_dir, exist_ok=True)
        
        # Train the model
        model = train_ppo(args, total_timesteps=args.timesteps, curriculum=args.curriculum)

    except KeyboardInterrupt:
        debug_print("\nCtrl+C detected in main block. Exiting cleanly...")
    except Exception as e:
        debug_print(f"An error occurred in the main execution block: {e}")
        import traceback
        debug_print(traceback.format_exc())
    # REMOVED: finally block that tried to shut down ROS (Keep Removed)

    # Added simple finish message here instead
    debug_print("Python script execution finished.") 