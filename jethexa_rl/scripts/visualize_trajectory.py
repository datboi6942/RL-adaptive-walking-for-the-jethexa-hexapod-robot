#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
from datetime import datetime

class TrajectoryVisualizer:
    def __init__(self):
        # Setup paths
        self.log_dir = "/catkin_ws/logs"
        self.position_log_path = os.path.join(self.log_dir, "robot_position_log.csv")
        self.metrics_log_path = os.path.join(self.log_dir, "training_metrics.csv")
        
        # Create figure with two subplots
        self.fig = plt.figure(figsize=(15, 10))
        self.ax_3d = self.fig.add_subplot(121, projection='3d')
        self.ax_metrics = self.fig.add_subplot(122)
        
        # Initialize empty plots
        self.line_3d, = self.ax_3d.plot([], [], [], 'b-', label='Trajectory')
        self.scatter_3d = self.ax_3d.scatter([], [], [], c='r', marker='o', label='Current Position')
        
        # Initialize metrics plots
        self.line_dx, = self.ax_metrics.plot([], [], 'b-', label='Δx')
        self.line_dy, = self.ax_metrics.plot([], [], 'g-', label='Δy')
        self.line_dyaw, = self.ax_metrics.plot([], [], 'r-', label='Δyaw')
        
        # Setup plot formatting
        self.setup_plots()
        
        # Initialize data storage
        self.last_position_update = 0
        self.last_metrics_update = 0
        self.update_interval = 0.5  # seconds
        
    def setup_plots(self):
        # 3D Trajectory Plot
        self.ax_3d.set_xlabel('X (m)')
        self.ax_3d.set_ylabel('Y (m)')
        self.ax_3d.set_zlabel('Z (m)')
        self.ax_3d.set_title('Robot 3D Trajectory')
        self.ax_3d.legend()
        
        # Metrics Plot
        self.ax_metrics.set_xlabel('Episode')
        self.ax_metrics.set_ylabel('Displacement')
        self.ax_metrics.set_title('Per-Episode Displacement')
        self.ax_metrics.legend()
        self.ax_metrics.grid(True)
        
        # Adjust layout
        plt.tight_layout()
        
    def update_3d_plot(self):
        try:
            # Read position data
            df = pd.read_csv(self.position_log_path)
            if len(df) > 0:
                # Update trajectory line
                self.line_3d.set_data(df['x'].values, df['y'].values)
                self.line_3d.set_3d_properties(df['z'].values)
                
                # Update current position scatter
                self.scatter_3d._offsets3d = ([df['x'].iloc[-1]], 
                                            [df['y'].iloc[-1]], 
                                            [df['z'].iloc[-1]])
                
                # Update axis limits with some padding
                x_range = df['x'].max() - df['x'].min()
                y_range = df['y'].max() - df['y'].min()
                z_range = df['z'].max() - df['z'].min()
                
                self.ax_3d.set_xlim(df['x'].min() - 0.1*x_range, df['x'].max() + 0.1*x_range)
                self.ax_3d.set_ylim(df['y'].min() - 0.1*y_range, df['y'].max() + 0.1*y_range)
                self.ax_3d.set_zlim(df['z'].min() - 0.1*z_range, df['z'].max() + 0.1*z_range)
                
        except Exception as e:
            print(f"Error updating 3D plot: {e}")
            
    def update_metrics_plot(self):
        try:
            # Read metrics data
            df = pd.read_csv(self.metrics_log_path)
            if len(df) > 0:
                episodes = df['episode'].values
                
                # Update displacement lines
                self.line_dx.set_data(episodes, df['delta_x'].values)
                self.line_dy.set_data(episodes, df['delta_y'].values)
                self.line_dyaw.set_data(episodes, df['delta_yaw'].values)
                
                # Update axis limits
                self.ax_metrics.set_xlim(0, max(episodes) + 1)
                y_min = min(df['delta_x'].min(), df['delta_y'].min(), df['delta_yaw'].min())
                y_max = max(df['delta_x'].max(), df['delta_y'].max(), df['delta_yaw'].max())
                self.ax_metrics.set_ylim(y_min - 0.1*abs(y_min), y_max + 0.1*abs(y_max))
                
        except Exception as e:
            print(f"Error updating metrics plot: {e}")
            
    def run(self):
        print("Starting trajectory visualization...")
        print(f"Position log: {self.position_log_path}")
        print(f"Metrics log: {self.metrics_log_path}")
        print("Press Ctrl+C to exit")
        
        try:
            while True:
                current_time = time.time()
                
                # Update plots if enough time has passed
                if current_time - self.last_position_update >= self.update_interval:
                    self.update_3d_plot()
                    self.last_position_update = current_time
                    
                if current_time - self.last_metrics_update >= self.update_interval:
                    self.update_metrics_plot()
                    self.last_metrics_update = current_time
                
                # Redraw the figure
                self.fig.canvas.draw()
                self.fig.canvas.flush_events()
                
                # Small sleep to prevent high CPU usage
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print("\nVisualization stopped by user")
        except Exception as e:
            print(f"Error in visualization: {e}")
        finally:
            plt.close()

if __name__ == "__main__":
    visualizer = TrajectoryVisualizer()
    visualizer.run() 