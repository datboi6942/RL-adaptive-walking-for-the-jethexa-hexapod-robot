#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
from cpg_controller import CPGController # Import your Python 3 CPG implementation
from jethexa_rl.srv import CPGControl, CPGControlResponse # Import the service definition
import numpy as np

class CPGServiceNode:
    def __init__(self):
        rospy.init_node('cpg_service_node')
        rospy.loginfo("Starting CPG Service Node...")
        
        # --- Instantiate your CPG Controller --- 
        # Adjust parameters if your CPGController needs them
        try:
            self.cpg_controller = CPGController()
            rospy.loginfo("CPG Controller instantiated.")
            # Call reset() initially to ensure it's in the default state
            self.cpg_controller.reset()
            rospy.loginfo("Initial CPG Controller reset complete.")
        except Exception as e:
            rospy.logfatal("Failed to instantiate CPGController: {} - Shutting down.".format(e))
            # If CPG fails to load, this node is useless
            rospy.signal_shutdown("CPG Controller instantiation failed") 
            return

        # Create the ROS Service Server
        self.service = rospy.Service('/cpg_control', CPGControl, self.handle_cpg_control)
        rospy.loginfo("CPG Control service ready at /cpg_control")

    def handle_cpg_control(self, req):
        """Callback function for the CPGControl service."""
        response = CPGControlResponse()
        response.success = False # Default to failure
        
        try:
            if req.command == 'reset':
                rospy.logdebug("CPG Service: Received RESET command.")
                self.cpg_controller.reset()
                response.success = True
                response.message = "CPG state reset."
                response.joint_positions = [] # No positions for reset command
                rospy.logdebug("CPG Service: Reset successful.")
            
            elif req.command == 'update':
                rospy.logdebug("CPG Service: Received UPDATE command with dt={:.4f}".format(req.dt))
                if req.dt <= 0:
                    response.message = "Invalid dt ({:.4f}) for update command.".format(req.dt)
                    rospy.logwarn(response.message)
                else:
                    joint_positions = self.cpg_controller.update(req.dt)
                    # Ensure output is a list of float64 for the service response
                    response.joint_positions = np.array(joint_positions).astype(np.float64).tolist()
                    if len(response.joint_positions) == 18: # Check expected size
                        response.success = True
                        response.message = "CPG updated."
                        rospy.logdebug("CPG Service: Update successful. Returning {} joint positions.".format(len(response.joint_positions)))
                    else:
                         response.message = "CPG update returned incorrect number of joints ({} instead of 18).".format(len(response.joint_positions))
                         rospy.logerr(response.message)
                         response.joint_positions = [] # Return empty list on error
            
            else:
                response.message = "Unknown command: {}".format(req.command)
                rospy.logwarn(response.message)
                
        except AttributeError as e:
             error_msg = "AttributeError in CPG handling: {} Does CPGController have reset/update?".format(e)
             rospy.logerr(error_msg)
             response.message = error_msg
             response.joint_positions = []
        except Exception as e:
            error_msg = "Unexpected error processing CPG command '{}': {}".format(req.command, e)
            rospy.logerr(error_msg)
            response.message = error_msg
            response.joint_positions = []
            
        return response

    def run(self):
        rospy.spin() # Keep the node alive to handle service calls

if __name__ == '__main__':
    try:
        node = CPGServiceNode()
        # Only run if initialization succeeded (important check due to shutdown in __init__)
        if not rospy.is_shutdown():
            node.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("CPG Service node shutdown requested.")
    except Exception as e:
        rospy.logfatal("Unhandled exception in CPG Service Node: {}".format(e)) 