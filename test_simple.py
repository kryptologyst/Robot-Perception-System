"""
Simple test script for the Robot Perception System

This script provides a basic test of the system functionality.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from perception.robot_perception_system import RobotPerceptionSystem
from perception.visualization import PerceptionVisualizer

def test_basic_functionality():
    """Test basic system functionality."""
    print("Testing Robot Perception System...")
    
    # Initialize system
    perception = RobotPerceptionSystem(
        grid_size=(15, 15),
        robot_position=(7.5, 7.5),
        sensor_range=4.0,
        resolution=0.2,
        seed=42
    )
    
    # Generate environment
    perception.generate_synthetic_environment(num_obstacles=6)
    print(f"Generated environment with {len(perception.detected_obstacles)} obstacles")
    
    # Perform sensor fusion
    detected_obstacles = perception.fuse_sensor_data()
    print(f"Detected {len(detected_obstacles)} obstacles through sensor fusion")
    
    # Update occupancy grid
    perception.update_occupancy_grid(detected_obstacles)
    
    # Get safe directions
    safe_dirs = perception.get_safe_directions()
    print(f"Found {len(safe_dirs)} safe movement directions")
    
    # Get perception summary
    summary = perception.get_perception_summary()
    print(f"Perception summary: {summary}")
    
    # Test visualization
    visualizer = PerceptionVisualizer(perception)
    print("Creating visualization...")
    
    # Save visualization
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    visualizer.plot_perception_state(
        detected_obstacles,
        save_path=f"{output_dir}/test_visualization.png",
        show_safe_directions=True,
        show_sensor_range=True
    )
    
    print("Test completed successfully!")
    print(f"Visualization saved to {output_dir}/test_visualization.png")

if __name__ == "__main__":
    test_basic_functionality()
