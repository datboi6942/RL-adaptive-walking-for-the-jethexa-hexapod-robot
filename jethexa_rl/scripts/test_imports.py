#!/usr/bin/env python3
import rospy
import sys
import os
import traceback

print("Python version:", sys.version)
print("Python executable:", sys.executable)
print("PYTHONPATH:", os.environ.get('PYTHONPATH', 'Not set'))

try:
    import numpy as np
    print("Numpy version:", np.__version__)
except ImportError as e:
    print("Error importing numpy:", e)

try:
    from tf import transformations
    print("Successfully imported tf.transformations")
except ImportError as e:
    print("Error importing tf.transformations:", e)
    traceback.print_exc()

try:
    import gym
    print("Gym version:", gym.__version__)
except ImportError as e:
    print("Error importing gym:", e)

try:
    rospy.init_node('test_imports', anonymous=True)
    print("Successfully initialized ROS node")
except Exception as e:
    print("Error initializing ROS node:", e)

print("Import test complete") 