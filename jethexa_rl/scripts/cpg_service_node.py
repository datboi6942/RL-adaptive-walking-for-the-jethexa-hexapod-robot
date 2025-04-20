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
                    return response
                else:
                    # --- MODIFIED: Handle optional params --- 
                    params_provided = req.params and len(req.params) > 0
                    if params_provided:
                        try:
                            # Convert params from list of float32 to numpy array
                            cpg_params = np.array(req.params, dtype=np.float32)
                            rospy.logdebug("CPG Service: Setting gait params: {}".format(np.round(cpg_params, 3)))
                            self.cpg_controller.set_gait_params(cpg_params)
                        except Exception as e:
                            response.message = "Error calling set_gait_params: {}".format(e)
                            rospy.logerr(response.message)
                            return response # Return immediately on param setting error
                    else:
                        rospy.logdebug("CPG Service: No params provided with update command (likely warm-up). Using existing CPG params.")
                    # --- END MODIFICATION ---
                    
                    # --- Now update the CPG (either with newly set or existing params) ---\
                    try:
                        joint_positions = self.cpg_controller.update(req.dt)
                        rospy.logdebug("CPG Service: Calculated joint positions (first 6): {}".format(np.round(joint_positions[:6], 3)))
                    except Exception as e:
                        response.message = "Error calling cpg_controller.update(): {}".format(e)
                        rospy.logerr(response.message)
                        response.joint_positions = []
                        return response # Return on update error

                    # Ensure output is a list of float64 for the service response
                    response.joint_positions = np.array(joint_positions).astype(np.float64).tolist()
                    if len(response.joint_positions) == 18: # Check expected size
                        response.success = True
                        # Modify message based on whether params were set
                        if params_provided:
                            response.message = "CPG parameters set and updated."
                        else:
                            response.message = "CPG updated (no new parameters provided)."
                        rospy.logdebug("CPG Service: Update successful. Returning {} joint positions.".format(len(response.joint_positions)))
                    else:
                         response.message = "CPG update returned incorrect number of joints ({} instead of 18).".format(len(response.joint_positions))
                         rospy.logerr(response.message)
                         response.joint_positions = [] # Return empty list on error
                         response.success = False # Ensure success is False
            
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