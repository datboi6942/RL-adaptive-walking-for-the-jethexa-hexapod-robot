#!/usr/bin/env python3
import sys
import os
# Make sure Python 3 packages are found first
python3_path = [p for p in sys.path if 'python3' in p]
python2_path = [p for p in sys.path if 'python2' in p]
sys.path = python3_path + [p for p in sys.path if p not in python2_path and p not in python3_path]

import rospy
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.msg import ModelState
from geometry_msgs.msg import Pose, Twist, Point, Quaternion

def reset_jethexa_position():
    """
    Reset the JetHexa robot to its initial position in the Gazebo simulation.
    This script can be used instead of a full simulation reset for faster training cycles.
    """
    # Initialize ROS node
    rospy.init_node('reset_jethexa', anonymous=True)
    
    # Wait for the service to be available
    rospy.loginfo("Waiting for /gazebo/set_model_state service...")
    rospy.wait_for_service('/gazebo/set_model_state')
    
    try:
        # Create a proxy for the service
        set_model_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        
        # Create a model state message
        state = ModelState()
        state.model_name = 'jethexa'
        
        # Set position (slightly above ground to avoid ground collision)
        state.pose = Pose(
            position=Point(x=0.0, y=0.0, z=0.2),
            orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
        )
        
        # Set velocity (zero initial velocity)
        state.twist = Twist()
        
        # Set reference frame
        state.reference_frame = 'world'
        
        # Call the service
        resp = set_model_state(state)
        
        if resp.success:
            rospy.loginfo("JetHexa model reset successful")
        else:
            rospy.logerr(f"Failed to reset JetHexa model: {resp.status_message}")
            
    except rospy.ServiceException as e:
        rospy.logerr(f"Service call failed: {e}")

if __name__ == "__main__":
    try:
        reset_jethexa_position()
    except rospy.ROSInterruptException:
        pass 