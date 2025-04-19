#!/usr/bin/env python3
"""
Debug script to help identify issues with the JetHexa RL environment.
"""

import sys
import os
import time
import traceback

def main():
    print("Python executable: {}".format(sys.executable))
    print("Python version: {}".format(sys.version))
    print("Python path:")
    for p in sys.path:
        print("  - {}".format(p))
    
    print("\nAttempting to import required packages...")
    
    try:
        import rospy
        print("✓ Successfully imported rospy")
    except ImportError as e:
        print("✗ Failed to import rospy: {}".format(e))
        print("  Fix: Install with 'pip3 install rospkg' or 'apt-get install python3-rospy'")
    
    try:
        import numpy as np
        print("✓ Successfully imported numpy")
    except ImportError as e:
        print("✗ Failed to import numpy: {}".format(e))
        print("  Fix: Install with 'pip3 install numpy'")
    
    try:
        import gym
        print("✓ Successfully imported gym")
    except ImportError as e:
        print("✗ Failed to import gym: {}".format(e))
        print("  Fix: Install with 'pip3 install gym'")
    
    try:
        from stable_baselines3 import PPO
        print("✓ Successfully imported stable_baselines3")
    except ImportError as e:
        print("✗ Failed to import stable_baselines3: {}".format(e))
        print("  Fix: Install with 'pip3 install stable_baselines3'")
    
    try:
        import matplotlib.pyplot as plt
        print("✓ Successfully imported matplotlib")
    except ImportError as e:
        print("✗ Failed to import matplotlib: {}".format(e))
        print("  Fix: Install with 'pip3 install matplotlib'")
    
    print("\nChecking ROS environment...")
    if 'ROS_MASTER_URI' in os.environ:
        print("ROS_MASTER_URI: {}".format(os.environ['ROS_MASTER_URI']))
    else:
        print("ROS_MASTER_URI not set! This could be a problem.")
        print("  Fix: Make sure to source the ROS setup.bash file")
    
    print("\nAttempting to initialize ROS node...")
    try:
        import rospy
        rospy.init_node('debug_node', anonymous=True, disable_signals=True)
        print("✓ Successfully initialized ROS node")
        
        # Check if we can get a list of topics
        topics = rospy.get_published_topics()
        print("Found {} ROS topics:".format(len(topics)))
        for topic, topic_type in topics[:5]:  # Show first 5 topics only
            print("  - {} ({})".format(topic, topic_type))
        if len(topics) > 5:
            print("  - ... and {} more".format(len(topics) - 5))
    except Exception as e:
        print("✗ Failed to initialize ROS node: {}".format(e))
        traceback.print_exc()
    
    print("\nChecking specific topics needed for the gym environment...")
    # List of topics we need
    required_topics = [
        '/joint_states',
        '/gazebo/model_states',
        '/jethexa/imu'
    ]
    
    try:
        all_topics = [t[0] for t in rospy.get_published_topics()]
        for topic in required_topics:
            if topic in all_topics:
                print("✓ Found required topic: {}".format(topic))
            else:
                print("✗ Missing required topic: {}".format(topic))
    except Exception as e:
        print("Error checking topics: {}".format(e))
    
    print("\nNow trying to import our gym environment...")
    try:
        from jethexa_gym_env import JetHexaEnv
        print("✓ Successfully imported JetHexaEnv")
        
        print("\nTrying to create the environment...")
        try:
            env = JetHexaEnv()
            print("✓ Successfully created JetHexaEnv")
            
            print("\nTrying to reset the environment...")
            try:
                obs = env.reset()
                print("✓ Successfully reset the environment")
                print("Observation shape: {}".format(obs.shape))
                
                print("\nTrying to take a step in the environment...")
                try:
                    action = env.action_space.sample()
                    obs, reward, done, info = env.step(action)
                    print("✓ Successfully took a step in the environment")
                    print("Reward: {}".format(reward))
                    print("Info: {}".format(info))
                except Exception as e:
                    print("✗ Failed to take a step in the environment: {}".format(e))
                    traceback.print_exc()
            except Exception as e:
                print("✗ Failed to reset the environment: {}".format(e))
                traceback.print_exc()
        except Exception as e:
            print("✗ Failed to create the environment: {}".format(e))
            traceback.print_exc()
    except Exception as e:
        print("✗ Failed to import JetHexaEnv: {}".format(e))
        traceback.print_exc()
    
    print("\nDebug complete.")

if __name__ == "__main__":
    main() 