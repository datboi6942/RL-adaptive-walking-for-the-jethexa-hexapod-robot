#!/usr/bin/env python
import sys
import os

import rospy
import random
import numpy as np
from gazebo_msgs.srv import SpawnModel, DeleteModel
from geometry_msgs.msg import Pose, Point, Quaternion
from tf.transformations import quaternion_from_euler

class TerrainGenerator:
    """
    Generates randomized terrain for curriculum learning in Gazebo.
    
    This class creates procedurally generated terrain of varying difficulty:
    - Level 0: Flat ground (baseline)
    - Level 1: Gentle slopes and small bumps
    - Level 2: Moderate terrain with steps and ramps
    - Level 3: Challenging terrain with gaps and steep inclines
    - Level 4: "Terrain hell" with complex obstacles
    
    Each reset randomly generates new terrain of the specified difficulty.
    """
    def __init__(self):
        """Initialize the terrain generator."""
        rospy.loginfo("Initializing TerrainGenerator...")
        
        # Connect to Gazebo services
        rospy.wait_for_service('/gazebo/spawn_sdf_model')
        rospy.wait_for_service('/gazebo/delete_model')
        self.spawn_model = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
        self.delete_model = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)
        
        # Package path for model files
        self.package_path = os.path.join(os.path.dirname(__file__), '..')
        
        # Terrain difficulty level (0-4)
        self.difficulty = 0
        
        # Keep track of spawned objects
        self.spawned_objects = []
        
        # Parameters for terrain generation
        self.terrain_size = 10.0  # meters
        self.terrain_center_x = 5.0  # meters in front of robot
        self.max_obstacles = 20
        
        rospy.loginfo("TerrainGenerator initialized")
    
    def set_difficulty(self, level):
        """
        Set the terrain difficulty level (0-4).
        
        Args:
            level: Integer 0-4 indicating difficulty
        """
        self.difficulty = max(0, min(4, level))
        rospy.loginfo("Setting terrain difficulty to level {}".format(self.difficulty))
    
    def reset_terrain(self):
        """Clear existing terrain and generate new terrain based on difficulty."""
        self._clear_terrain()
        
        if self.difficulty == 0:
            # Level 0: Flat ground (do nothing)
            pass
        elif self.difficulty == 1:
            # Level 1: Gentle terrain
            self._generate_gentle_terrain()
        elif self.difficulty == 2:
            # Level 2: Moderate terrain
            self._generate_moderate_terrain()
        elif self.difficulty == 3:
            # Level 3: Challenging terrain
            self._generate_challenging_terrain()
        else:
            # Level 4: "Terrain hell"
            self._generate_hell_terrain()
        
        rospy.loginfo("Terrain reset complete (level {})".format(self.difficulty))
    
    def _clear_terrain(self):
        """Remove all spawned terrain objects."""
        for model_name in self.spawned_objects:
            try:
                self.delete_model(model_name)
                rospy.loginfo("Deleted model: {}".format(model_name))
            except rospy.ServiceException as e:
                rospy.logwarn("Failed to delete model {}: {}".format(model_name, e))
        
        self.spawned_objects = []
    
    def _spawn_box(self, name, size, position, orientation=(0, 0, 0), color=(0.8, 0.8, 0.8, 1.0)):
        """
        Spawn a box in the Gazebo world.
        
        Args:
            name: Unique model name
            size: Tuple of (length, width, height)
            position: Tuple of (x, y, z)
            orientation: Tuple of (roll, pitch, yaw) in radians
            color: Tuple of (r, g, b, a)
        
        Returns:
            Success flag
        """
        l, w, h = size
        x, y, z = position
        roll, pitch, yaw = orientation
        
        # Generate SDF XML for a box
        box_sdf = """
        <?xml version="1.0" ?>
        <sdf version="1.6">
          <model name="{name}">
            <static>true</static>
            <link name="link">
              <collision name="collision">
                <geometry>
                  <box>
                    <size>{l} {w} {h}</size>
                  </box>
                </geometry>
              </collision>
              <visual name="visual">
                <geometry>
                  <box>
                    <size>{l} {w} {h}</size>
                  </box>
                </geometry>
                <material>
                  <ambient>{r} {g} {b} {a}</ambient>
                  <diffuse>{r} {g} {b} {a}</diffuse>
                </material>
              </visual>
            </link>
          </model>
        </sdf>
        """.format(
            name=name,
            l=l, w=w, h=h,
            r=color[0], g=color[1], b=color[2], a=color[3]
        )
        
        # Convert orientation to quaternion
        q = quaternion_from_euler(roll, pitch, yaw)
        pose = Pose(
            position=Point(x=x, y=y, z=z),
            orientation=Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])
        )
        
        try:
            self.spawn_model(name, box_sdf, "", pose, "world")
            self.spawned_objects.append(name)
            return True
        except rospy.ServiceException as e:
            rospy.logwarn("Failed to spawn box {}: {}".format(name, e))
            return False
    
    def _spawn_cylinder(self, name, radius, height, position, color=(0.8, 0.8, 0.8, 1.0)):
        """
        Spawn a cylinder in the Gazebo world.
        
        Args:
            name: Unique model name
            radius: Cylinder radius
            height: Cylinder height
            position: Tuple of (x, y, z)
            color: Tuple of (r, g, b, a)
        
        Returns:
            Success flag
        """
        x, y, z = position
        
        # Generate SDF XML for a cylinder
        cylinder_sdf = """
        <?xml version="1.0" ?>
        <sdf version="1.6">
          <model name="{name}">
            <static>true</static>
            <link name="link">
              <collision name="collision">
                <geometry>
                  <cylinder>
                    <radius>{radius}</radius>
                    <length>{height}</length>
                  </cylinder>
                </geometry>
              </collision>
              <visual name="visual">
                <geometry>
                  <cylinder>
                    <radius>{radius}</radius>
                    <length>{height}</length>
                  </cylinder>
                </geometry>
                <material>
                  <ambient>{r} {g} {b} {a}</ambient>
                  <diffuse>{r} {g} {b} {a}</diffuse>
                </material>
              </visual>
            </link>
          </model>
        </sdf>
        """.format(
            name=name,
            radius=radius,
            height=height,
            r=color[0], g=color[1], b=color[2], a=color[3]
        )
        
        pose = Pose(
            position=Point(x=x, y=y, z=z),
            orientation=Quaternion(x=0, y=0, z=0, w=1)
        )
        
        try:
            self.spawn_model(name, cylinder_sdf, "", pose, "world")
            self.spawned_objects.append(name)
            return True
        except rospy.ServiceException as e:
            rospy.logwarn("Failed to spawn cylinder {}: {}".format(name, e))
            return False
    
    def _generate_gentle_terrain(self):
        """Generate level 1 terrain with gentle slopes and small bumps."""
        # Add a few small ramps
        for i in range(3):
            x = self.terrain_center_x + random.uniform(-3, 3)
            y = random.uniform(-3, 3)
            yaw = random.uniform(0, 2*np.pi)
            size = (random.uniform(1, 2), random.uniform(1, 2), random.uniform(0.05, 0.1))
            
            self._spawn_box(
                "ramp_{}".format(i), 
                size, 
                (x, y, size[2]/2), 
                (0, 0, yaw), 
                (0.7, 0.7, 0.5, 1.0)
            )
        
        # Add a few bumps (small cylinders)
        for i in range(5):
            x = self.terrain_center_x + random.uniform(-4, 4)
            y = random.uniform(-4, 4)
            radius = random.uniform(0.05, 0.2)
            height = random.uniform(0.03, 0.08)
            
            self._spawn_cylinder(
                "bump_{}".format(i), 
                radius, 
                height, 
                (x, y, height/2), 
                (0.6, 0.6, 0.6, 1.0)
            )
    
    def _generate_moderate_terrain(self):
        """Generate level 2 terrain with steps and moderate obstacles."""
        # Create a stepped area
        step_height = 0.07
        for i in range(4):
            x = self.terrain_center_x + i * 0.7  # Steps progressively further
            w = 6.0 - i * 0.5  # Steps get narrower
            self._spawn_box(
                "step_{}".format(i),
                (0.7, w, step_height * (i+1)),
                (x, 0, step_height * (i+1) / 2),
                (0, 0, 0),
                (0.7, 0.7, 0.5, 1.0)
            )
        
        # Add some ramps at various angles
        for i in range(3):
            x = self.terrain_center_x + random.uniform(-3, 3)
            y = random.uniform(-3, 3)
            roll = random.uniform(-0.2, 0.2)  # Small incline
            pitch = random.uniform(-0.3, 0.3)  # Moderate slope
            yaw = random.uniform(0, 2*np.pi)
            
            self._spawn_box(
                "ramp_m_{}".format(i),
                (random.uniform(1.0, 2.0), random.uniform(1.0, 2.0), 0.05),
                (x, y, 0.025),
                (roll, pitch, yaw),
                (0.6, 0.7, 0.8, 1.0)
            )
        
        # Add a few larger obstacles
        for i in range(5):
            x = self.terrain_center_x + random.uniform(-4, 4)
            y = random.uniform(-4, 4)
            size = (
                random.uniform(0.2, 0.5),
                random.uniform(0.2, 0.5),
                random.uniform(0.1, 0.2)
            )
            
            self._spawn_box(
                "obstacle_m_{}".format(i),
                size,
                (x, y, size[2]/2),
                (0, 0, random.uniform(0, 2*np.pi)),
                (0.5, 0.5, 0.5, 1.0)
            )
    
    def _generate_challenging_terrain(self):
        """Generate level 3 terrain with difficult obstacles and gaps."""
        # Create uneven terrain with gaps
        for i in range(8):
            x = self.terrain_center_x + i * random.uniform(0.8, 1.2)
            y = random.uniform(-1, 1)
            width = random.uniform(2, 4)
            
            # Occasionally skip a platform to create gaps
            if random.random() > 0.3:  # 30% chance of gap
                self._spawn_box(
                    "platform_{}".format(i),
                    (0.8, width, 0.1),
                    (x, y, 0.05),
                    (random.uniform(-0.1, 0.1), random.uniform(-0.1, 0.1), random.uniform(-0.1, 0.1)),
                    (0.4, 0.4, 0.6, 1.0)
                )
        
        # Add steep ramps and obstacles
        for i in range(4):
            x = self.terrain_center_x + random.uniform(-3, 3)
            y = random.uniform(-3, 3)
            pitch = random.uniform(0.3, 0.5)  # Steep incline
            
            self._spawn_box(
                "steep_ramp_{}".format(i),
                (1.5, 1.0, 0.05),
                (x, y, 0.4),
                (0, pitch, random.uniform(0, 2*np.pi)),
                (0.7, 0.5, 0.3, 1.0)
            )
        
        # Add some challenging obstacles
        for i in range(7):
            x = self.terrain_center_x + random.uniform(-4, 4)
            y = random.uniform(-4, 4)
            height = random.uniform(0.2, 0.35)
            
            if random.random() > 0.5:
                # Tall obstacle
                self._spawn_box(
                    "tall_obstacle_{}".format(i),
                    (0.3, 0.3, height),
                    (x, y, height/2),
                    (0, 0, 0),
                    (0.3, 0.3, 0.3, 1.0)
                )
            else:
                # Cylinder obstacle
                self._spawn_cylinder(
                    "cylinder_obstacle_{}".format(i),
                    0.2,
                    height,
                    (x, y, height/2),
                    (0.3, 0.3, 0.3, 1.0)
                )
    
    def _generate_hell_terrain(self):
        """Generate level 4 "terrain hell" with extreme challenges."""
        # Create a chaotic terrain with steep drops, walls, and obstacles
        
        # Main pathway with varying heights
        for i in range(10):
            x = self.terrain_center_x + i * 0.8
            height = random.uniform(0.1, 0.5)
            width = random.uniform(1.5, 3.0)
            y_offset = random.uniform(-0.5, 0.5)
            
            # Some platforms are tilted for extra challenge
            roll = random.uniform(-0.3, 0.3)
            pitch = random.uniform(-0.3, 0.3)
            
            if random.random() > 0.2:  # 20% chance of gap
                self._spawn_box(
                    "hell_platform_{}".format(i),
                    (0.8, width, 0.1),
                    (x, y_offset, height),
                    (roll, pitch, random.uniform(-0.1, 0.1)),
                    (0.3, 0.3, 0.3, 1.0)
                )
        
        # Add walls and barriers
        for i in range(5):
            x = self.terrain_center_x + random.uniform(1, 8)
            y = random.uniform(-1.5, 1.5)
            height = random.uniform(0.3, 0.6)
            
            self._spawn_box(
                "wall_{}".format(i),
                (0.1, random.uniform(0.5, 1.5), height),
                (x, y, height/2),
                (0, 0, random.uniform(0, 2*np.pi)),
                (0.2, 0.2, 0.2, 1.0)
            )
        
        # Add nasty obstacles in random positions
        for i in range(12):
            x = self.terrain_center_x + random.uniform(0, 8)
            y = random.uniform(-3, 3)
            
            if random.random() > 0.7:
                # Spike-like obstacle
                self._spawn_cylinder(
                    "spike_{}".format(i),
                    random.uniform(0.05, 0.15),
                    random.uniform(0.3, 0.5),
                    (x, y, random.uniform(0.15, 0.25)),
                    (0.2, 0.2, 0.2, 1.0)
                )
            else:
                # Random block
                size = (
                    random.uniform(0.2, 0.4),
                    random.uniform(0.2, 0.4),
                    random.uniform(0.2, 0.4)
                )
                
                self._spawn_box(
                    "block_{}".format(i),
                    size,
                    (x, y, size[2]/2 + random.uniform(0, 0.1)),
                    (random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5), random.uniform(0, 2*np.pi)),
                    (0.3, 0.3, 0.3, 1.0)
                )


if __name__ == "__main__":
    try:
        rospy.init_node('terrain_generator_test')
        generator = TerrainGenerator()
        
        # Test generating each difficulty level
        for level in range(5):
            generator.set_difficulty(level)
            generator.reset_terrain()
            rospy.loginfo("Generated terrain at level {}. Press Enter to continue...".format(level))
            input()
        
        rospy.loginfo("Terrain generator test complete")
    except rospy.ROSInterruptException:
        pass 