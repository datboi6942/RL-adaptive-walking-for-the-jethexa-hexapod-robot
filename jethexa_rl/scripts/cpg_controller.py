#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
# import matplotlib.pyplot as plt # Removed dependency for Python 2 compatibility
import rospy

class CPGController:
    """
    Central Pattern Generator for hexapod locomotion.
    
    This controller generates coordinated rhythmic patterns for hexapod walking
    using coupled oscillators, producing smooth, biologically-inspired gaits.
    
    The controller accepts high-level parameters and converts them to joint angles,
    dramatically reducing the dimensionality of the control problem.
    """
    def __init__(self, n_legs=6, n_joints_per_leg=3):
        self.n_legs = n_legs
        self.n_joints_per_leg = n_joints_per_leg
        
        # Default CPG parameters
        self.frequencies = np.ones(n_legs) * 1.0  # Hz
        self.amplitudes = np.ones((n_legs, n_joints_per_leg)) * 0.5  # radians
        
        # Default phase offsets for tripod gait
        # Legs 0, 2, 4 are in phase, legs 1, 3, 5 are 180 degrees out of phase
        self.phase_offsets = np.array([0, np.pi, 0, np.pi, 0, np.pi])
        
        # Joint angle bias (standing pose)
        self.joint_bias = np.zeros((n_legs, n_joints_per_leg))
        # Default standing pose for JetHexa
        # Shoulder joints (wider stance) - REMOVED ASYMMETRY
        # coxa_bias_magnitude = 0.1 
        # self.joint_bias[0:3, 0] = coxa_bias_magnitude  # Left legs (positive bias)
        # self.joint_bias[3:6, 0] = -coxa_bias_magnitude # Right legs (negative bias)
        self.joint_bias[:, 0] = 0.0 # Set shoulder bias to zero for all legs
        # Hip joints (adjusting upward lift)
        self.joint_bias[:, 1] = 0.6  # Reduced Hip Forward Bias
        # Knee joints (adjusting bend)
        self.joint_bias[:, 2] = -0.7 # Reduced bend magnitude (straighter knee)
        
        # Internal oscillator state
        # Initialize phase based on the default offsets for immediate stability
        self.phase = self.phase_offsets.copy()
        
        # Coupling weights for coordination (helps maintain stable gait)
        self.coupling_weights = np.zeros((n_legs, n_legs))
        self._setup_coupling()
    
    def _setup_coupling(self):
        """Set up coupling weights between oscillators for stable gait"""
        # For tripod gait: positive coupling within tripod, negative across tripods
        for i in range(self.n_legs):
            for j in range(self.n_legs):
                if (i % 2) == (j % 2):  # Same tripod
                    self.coupling_weights[i, j] = 0.1
                else:  # Opposite tripod
                    self.coupling_weights[i, j] = -0.1
    
    def set_gait_params(self, params):
        """
        Set CPG parameters from a flat parameter vector.
        
        Args:
            params: Numpy array of parameters with shape (n_params,) where
                   n_params = 2 + n_legs + n_legs*n_joints_per_leg
                   [global_freq, gait_type, leg_phases, joint_amplitudes]
                   **Assumes input parameters are normalized between 0 and 1.**
        """
        expected_params = 2 + self.n_legs + self.n_legs * self.n_joints_per_leg
        assert len(params) == expected_params, \
            "Expected {} parameters, got {}".format(expected_params, len(params))
        
        # --- MODIFIED: Reduced Logging ---
        # Only log at debug level and only the first few parameters
        rospy.logdebug("CPG Params (first 5): {}".format(np.round(params[:5], 3)))
        
        # --- ADD SCALING FROM [0, 1] to physical ranges ---
        MIN_FREQ = 0.5  # Hz
        MAX_FREQ = 2.5  # Hz
        MAX_PHASE_ADJUST = np.pi / 8.0 # Max adjustment in radians (+/-)
        MIN_AMP = 0.1   # Radians
        MAX_AMP = 0.8   # Radians (Adjust based on joint limits/desired motion)
        # --- END SCALING RANGES ---

        # Global frequency (Hz) - controls overall speed
        # Scale param[0] from [0, 1] to [MIN_FREQ, MAX_FREQ]
        scaled_global_freq = MIN_FREQ + params[0] * (MAX_FREQ - MIN_FREQ)
        self.frequencies = np.ones(self.n_legs) * scaled_global_freq # Clip is redundant now
        
        # Gait type parameter (0: tripod, 1: wave, between: mixed)
        gait_type = np.clip(params[1], 0, 1)
        
        # Select base phase offsets based on gait type
        if gait_type < 0.33:  # Tripod gait
            self.phase_offsets = np.array([0, np.pi, 0, np.pi, 0, np.pi])
        elif gait_type < 0.66:  # Tetrapod gait (Example - adjust if needed)
             self.phase_offsets = np.array([0, np.pi/2, np.pi, 3*np.pi/2, 0, np.pi/2]) # Example Tetrapod
        else:  # Wave gait
            self.phase_offsets = np.array([0, np.pi/3, 2*np.pi/3, np.pi, 4*np.pi/3, 5*np.pi/3])
        
        # Fine-tune phase offsets with individual leg adjustments
        phase_adjust_params = params[2:2+self.n_legs]
        # Scale phase_adjust_params from [0, 1] to [-MAX_PHASE_ADJUST, +MAX_PHASE_ADJUST]
        # Shift [0, 1] to [-0.5, 0.5], then scale by 2*MAX_PHASE_ADJUST
        scaled_phase_adjustments = (phase_adjust_params - 0.5) * (2 * MAX_PHASE_ADJUST)
        self.phase_offsets += scaled_phase_adjustments # Apply scaled adjustments
        
        # Set joint amplitudes (controls step size and leg movement)
        amp_params = params[2+self.n_legs:]
        # Scale amp_params from [0, 1] to [MIN_AMP, MAX_AMP]
        scaled_amplitudes = MIN_AMP + amp_params * (MAX_AMP - MIN_AMP)
        self.amplitudes = scaled_amplitudes.reshape(self.n_legs, self.n_joints_per_leg)
        
        # --- MODIFIED: Enforce Symmetry with Minimal Logging ---
        rospy.logdebug("Enforcing CPG parameter symmetry at set_gait_params...")
        
        # Average Phase Adjustments (modifies self.phase_offsets which includes adjustments)
        # Note: Base offsets are already symmetric based on gait type logic above
        avg_phase_adj_03 = (self.phase_offsets[0] + self.phase_offsets[3]) / 2.0
        self.phase_offsets[0] = avg_phase_adj_03
        self.phase_offsets[3] = avg_phase_adj_03
        
        avg_phase_adj_14 = (self.phase_offsets[1] + self.phase_offsets[4]) / 2.0
        self.phase_offsets[1] = avg_phase_adj_14
        self.phase_offsets[4] = avg_phase_adj_14
        
        avg_phase_adj_25 = (self.phase_offsets[2] + self.phase_offsets[5]) / 2.0
        self.phase_offsets[2] = avg_phase_adj_25
        self.phase_offsets[5] = avg_phase_adj_25
        
        # Average Amplitudes (modifies self.amplitudes)
        for joint_idx in range(self.n_joints_per_leg):
            # LF / RF (Leg 0 / Leg 3)
            avg_amp_03 = (self.amplitudes[0, joint_idx] + self.amplitudes[3, joint_idx]) / 2.0
            self.amplitudes[0, joint_idx] = avg_amp_03
            self.amplitudes[3, joint_idx] = avg_amp_03
            # LM / RM (Leg 1 / Leg 4)
            avg_amp_14 = (self.amplitudes[1, joint_idx] + self.amplitudes[4, joint_idx]) / 2.0
            self.amplitudes[1, joint_idx] = avg_amp_14
            self.amplitudes[4, joint_idx] = avg_amp_14
            # LR / RR (Leg 2 / Leg 5)
            avg_amp_25 = (self.amplitudes[2, joint_idx] + self.amplitudes[5, joint_idx]) / 2.0
            self.amplitudes[2, joint_idx] = avg_amp_25
            self.amplitudes[5, joint_idx] = avg_amp_25
    
    def reset(self):
        """Reset the internal state of the CPG oscillators."""
        # Reset phase based on current phase offsets
        self.phase = self.phase_offsets.copy()
        # Reset frequencies and amplitudes? Might not be desired unless params change
        # For now, just resetting phase seems appropriate for the service call
        print("CPGController: Internal phase reset.") # Added print for confirmation

    def update(self, dt):
        """
        Update the CPG state and return joint angles based on a fixed time step.
        
        Args:
            dt: Time step in seconds
            
        Returns:
            joint_angles: Array of shape (n_legs*n_joints_per_leg,) with joint angles
        """
        # Update oscillator phases
        for i in range(self.n_legs):
            # Update phase based on frequency and the provided dt
            self.phase[i] += 2 * np.pi * self.frequencies[i] * dt
            
            # Apply coupling from other oscillators (improves stability)
            for j in range(self.n_legs):
                if i != j:
                    phase_diff = self.phase[j] - self.phase[i] - \
                                (self.phase_offsets[j] - self.phase_offsets[i])
                    # Ensure phase_diff calculation is correct for coupling update
                    phase_diff = (phase_diff + np.pi) % (2 * np.pi) - np.pi # Wrap to [-pi, pi]
                    self.phase[i] += dt * self.coupling_weights[i, j] * np.sin(phase_diff)

        # Keep phase within [0, 2π]
        self.phase = self.phase % (2 * np.pi)
        
        # Generate joint angles using the oscillator outputs
        joint_angles = np.zeros((self.n_legs, self.n_joints_per_leg))
        
        for i in range(self.n_legs):
            phase_rad = (self.phase[i] + self.phase_offsets[i]) % (2 * np.pi)

            # Shoulder: side-to-side movement
            joint_angles[i, 0] = self.joint_bias[i, 0] + \
                                 self.amplitudes[i, 0] * np.sin(phase_rad)

            # --- Smoothed Hip Control --- 
            # Use cosine: Max lift near pi/2 (mid-swing), min lift near 3pi/2 (mid-stance)
            if phase_rad < np.pi: # Swing phase
                 lift_amplitude_scale = 1.0
            else: # Stance phase
                 lift_amplitude_scale = 0.1 # Reduce vertical movement during stance
            hip_angle = self.joint_bias[i, 1] + self.amplitudes[i, 1] * lift_amplitude_scale * np.cos(phase_rad)
            joint_angles[i, 1] = hip_angle
            # --- End Smoothed Hip Control --- 

            # --- Smoothed Knee Control --- 
            # Use sine: Positive sine for flexion (swing), negative for extension (stance)
            if phase_rad < np.pi: # Swing phase (more flexion)
                 flex_amplitude_scale = 1.0
            else: # Stance phase (less flexion / extension)
                 flex_amplitude_scale = 0.5 
            knee_angle = self.joint_bias[i, 2] + self.amplitudes[i, 2] * flex_amplitude_scale * np.sin(phase_rad)
            joint_angles[i, 2] = knee_angle
            # --- End Smoothed Knee Control ---
        
        # Flatten to match the JetHexa controller format
        return joint_angles.flatten()
    
    # def visualize_gait(self, duration=5.0, dt=0.05):
    #     """
    #     Visualize the current gait pattern over time.
        
    #     Args:
    #         duration: Duration to simulate in seconds
    #         dt: Time step in seconds
    #     """
    #     # Commented out due to matplotlib dependency causing issues in Python 2 ROS node
    #     # Requires matplotlib installation in the Python 2 environment if uncommented.
    #     time_steps = np.arange(0, duration, dt)
    #     joint_angles_over_time = []
        
    #     # Reset phase for visualization
    #     old_phase = self.phase.copy()
    #     self.phase = np.zeros(self.n_legs)
        
    #     for t in time_steps:
    #         joint_angles = self.update(dt)
    #         joint_angles_over_time.append(joint_angles)
        
    #     # Restore original phase
    #     self.phase = old_phase
        
    #     joint_angles_over_time = np.array(joint_angles_over_time)
        
    #     # plt.figure(figsize=(12, 8))
        
    #     # # Plot first joint (shoulder) for each leg
    #     # plt.subplot(3, 1, 1)
    #     # for leg in range(self.n_legs):
    #     #     idx = leg * self.n_joints_per_leg
    #     #     plt.plot(time_steps, joint_angles_over_time[:, idx], 
    #     #              label='Leg {} Shoulder'.format(leg+1))
    #     # plt.title('Shoulder Joint Angles')
    #     # plt.ylabel('Angle (rad)')
    #     # plt.legend()
        
    #     # # Plot second joint (hip) for each leg
    #     # plt.subplot(3, 1, 2)
    #     # for leg in range(self.n_legs):
    #     #     idx = leg * self.n_joints_per_leg + 1
    #     #     plt.plot(time_steps, joint_angles_over_time[:, idx], 
    #     #              label='Leg {} Hip'.format(leg+1))
    #     # plt.title('Hip Joint Angles')
    #     # plt.ylabel('Angle (rad)')
    #     # plt.legend()
        
    #     # # Plot third joint (knee) for each leg
    #     # plt.subplot(3, 1, 3)
    #     # for leg in range(self.n_legs):
    #     #     idx = leg * self.n_joints_per_leg + 2
    #     #     plt.plot(time_steps, joint_angles_over_time[:, idx], 
    #     #              label='Leg {} Knee'.format(leg+1))
    #     # plt.title('Knee Joint Angles')
    #     # plt.xlabel('Time (s)')
    #     # plt.ylabel('Angle (rad)')
    #     # plt.legend()
        
    #     # plt.tight_layout()
    #     # plt.show()

if __name__ == "__main__":
    # Test the CPG controller
    controller = CPGController()
    
    # Create random parameters
    n_params = 2 + 6 + 6*3  # global_freq, gait_type, leg_phases, joint_amplitudes
    params = np.random.uniform(0, 1, n_params)
    
    # Set gait parameters (includes initial symmetry enforcement on params)
    controller.set_gait_params(params)
    
    # Visualize the resulting gait
    # controller.visualize_gait() # Commented out due to dependency
    
    # Run update once
    joint_angles = controller.update(0.05)
    print("Joint angles:", joint_angles) 