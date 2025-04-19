#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
# import matplotlib.pyplot as plt # Removed dependency for Python 2 compatibility

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
        # Shoulder joints (wider stance)
        coxa_bias_magnitude = 0.1 
        self.joint_bias[0:3, 0] = coxa_bias_magnitude  # Left legs (positive bias)
        self.joint_bias[3:6, 0] = -coxa_bias_magnitude # Right legs (negative bias)
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
        """
        expected_params = 2 + self.n_legs + self.n_legs * self.n_joints_per_leg
        assert len(params) == expected_params, \
            "Expected {} parameters, got {}".format(expected_params, len(params))
        
        # Global frequency (Hz) - controls overall speed
        global_freq = params[0]
        self.frequencies = np.ones(self.n_legs) * np.clip(global_freq, 0.2, 2.0)
        
        # Gait type parameter (0: tripod, 1: wave, between: mixed)
        gait_type = np.clip(params[1], 0, 1)
        
        if gait_type < 0.33:  # Tripod gait
            self.phase_offsets = np.array([0, np.pi, 0, np.pi, 0, np.pi])
        elif gait_type < 0.66:  # Tetrapod gait
            self.phase_offsets = np.array([0, 2*np.pi/3, 4*np.pi/3, 0, 2*np.pi/3, 4*np.pi/3])
        else:  # Wave gait
            self.phase_offsets = np.array([0, np.pi/3, 2*np.pi/3, np.pi, 4*np.pi/3, 5*np.pi/3])
        
        # Fine-tune phase offsets with individual leg adjustments
        phase_adjustments = params[2:2+self.n_legs]
        for i in range(self.n_legs):
            self.phase_offsets[i] += phase_adjustments[i] * 0.2  # Small adjustments
        
        # Set joint amplitudes (controls step size and leg movement)
        amp_params = params[2+self.n_legs:]
        self.amplitudes = amp_params.reshape(self.n_legs, self.n_joints_per_leg)
        
        # Clip to reasonable ranges (increased upper bound)
        self.amplitudes = np.clip(self.amplitudes, 0.1, 1.4)
    
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
            phase_rad = self.phase[i] + self.phase_offsets[i]
            phase_rad = phase_rad % (2 * np.pi) # Ensure phase is [0, 2*pi]

            # Shoulder: side-to-side movement
            coxa_angle_offset = self.amplitudes[i, 0] * np.sin(phase_rad)
            if i >= 3: # Right side legs (RF, RM, RR)
                joint_angles[i, 0] = self.joint_bias[i, 0] - coxa_angle_offset
            else: # Left side legs (LF, LM, LR)
                joint_angles[i, 0] = self.joint_bias[i, 0] + coxa_angle_offset

            # --- Smoothed Hip Control --- 
            # Use cosine: Max lift near pi/2 (mid-swing), min lift near 3pi/2 (mid-stance)
            if phase_rad < np.pi: # Swing phase
                 lift_amplitude_scale = 1.0
            else: # Stance phase
                 lift_amplitude_scale = 0.1 # Reduce vertical movement during stance
            hip_angle = self.joint_bias[i, 1] - self.amplitudes[i, 1] * lift_amplitude_scale * np.cos(phase_rad)
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
    
    # Set gait parameters
    controller.set_gait_params(params)
    
    # Visualize the resulting gait
    # controller.visualize_gait() # Commented out due to dependency
    
    # Flatten to match the JetHexa controller format
    joint_angles = controller.update(0.05)
    print("Joint angles:", joint_angles) 