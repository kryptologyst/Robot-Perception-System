"""
Robot Perception System - Quick Start Guide

This script provides a quick demonstration of the modernized robot perception system.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from perception.robot_perception_system import RobotPerceptionSystem
from perception.visualization import PerceptionVisualizer
from planning.path_planner import PathPlanner, PlanningAlgorithm

def quick_demo():
    """Run a quick demonstration of the system."""
    print("="*60)
    print("ROBOT PERCEPTION SYSTEM - QUICK DEMO")
    print("="*60)
    
    # Initialize the perception system
    print("\n1. Initializing Robot Perception System...")
    perception = RobotPerceptionSystem(
        grid_size=(20, 20),
        robot_position=(10.0, 10.0),
        sensor_range=5.0,
        resolution=0.1,
        seed=42
    )
    
    # Generate synthetic environment
    print("2. Generating synthetic environment...")
    perception.generate_synthetic_environment(num_obstacles=8)
    print(f"   Generated {len(perception.detected_obstacles)} obstacles")
    
    # Perform sensor fusion
    print("3. Performing sensor fusion...")
    detected_obstacles = perception.fuse_sensor_data()
    print(f"   Detected {len(detected_obstacles)} obstacles through sensor fusion")
    
    # Update occupancy grid
    print("4. Updating occupancy grid...")
    perception.update_occupancy_grid(detected_obstacles)
    
    # Get safe directions
    print("5. Computing safe movement directions...")
    safe_directions = perception.get_safe_directions()
    print(f"   Found {len(safe_directions)} safe movement directions")
    
    # Test path planning
    print("6. Testing path planning...")
    path_planner = PathPlanner(perception)
    
    # Plan to a goal
    goal = (15.0, 15.0)
    path = path_planner.plan_path(goal, PlanningAlgorithm.A_STAR)
    
    if path:
        print(f"   Successfully planned path with {len(path.points)} waypoints")
        print(f"   Path cost: {path.total_cost:.2f}")
    else:
        print("   No feasible path found")
    
    # Create visualization
    print("7. Creating visualization...")
    visualizer = PerceptionVisualizer(perception)
    
    # Create output directory
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    # Save visualization
    visualizer.plot_perception_state(
        detected_obstacles,
        save_path=output_dir / "quick_demo_visualization.png",
        show_safe_directions=True,
        show_sensor_range=True
    )
    
    # Get system summary
    print("8. System Summary:")
    summary = perception.get_perception_summary()
    for key, value in summary.items():
        print(f"   {key}: {value}")
    
    print("\n" + "="*60)
    print("DEMO COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"Visualization saved to: {output_dir / 'quick_demo_visualization.png'}")
    print("\nTo run the full demo with all features:")
    print("python demo.py --demo all")
    print("\nFor more information, see README.md")

if __name__ == "__main__":
    quick_demo()
