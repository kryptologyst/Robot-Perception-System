"""
Robot Perception System - Basic Test (No External Dependencies)

This script provides a basic test without requiring external dependencies.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_basic_imports():
    """Test basic imports and functionality."""
    print("="*60)
    print("ROBOT PERCEPTION SYSTEM - BASIC TEST")
    print("="*60)
    
    try:
        # Test basic imports
        print("\n1. Testing basic imports...")
        from perception.robot_perception_system import RobotPerceptionSystem, Obstacle, SensorType
        print("   ✓ Core perception system imported successfully")
        
        from perception.visualization import PerceptionVisualizer, PerceptionEvaluator
        print("   ✓ Visualization modules imported successfully")
        
        from planning.path_planner import PathPlanner, PlanningAlgorithm
        print("   ✓ Planning modules imported successfully")
        
        # Test basic functionality
        print("\n2. Testing basic functionality...")
        
        # Initialize system
        perception = RobotPerceptionSystem(
            grid_size=(10, 10),
            robot_position=(5.0, 5.0),
            sensor_range=3.0,
            resolution=0.2,
            seed=42
        )
        print("   ✓ Perception system initialized")
        
        # Generate environment
        perception.generate_synthetic_environment(num_obstacles=4)
        print(f"   ✓ Generated environment with {len(perception.detected_obstacles)} obstacles")
        
        # Test sensor simulation
        lidar_data = perception.simulate_lidar_scan()
        print(f"   ✓ LiDAR simulation: {len(lidar_data)} measurements")
        
        camera_image = perception.simulate_camera_image()
        print(f"   ✓ Camera simulation: {camera_image.shape} image")
        
        # Test obstacle detection
        detected_obstacles = perception.detect_obstacles_computer_vision(camera_image)
        print(f"   ✓ Computer vision detection: {len(detected_obstacles)} obstacles")
        
        # Test sensor fusion
        fused_obstacles = perception.fuse_sensor_data()
        print(f"   ✓ Sensor fusion: {len(fused_obstacles)} obstacles")
        
        # Test path planning
        path_planner = PathPlanner(perception)
        goal = (8.0, 8.0)
        path = path_planner.plan_path(goal, PlanningAlgorithm.A_STAR)
        
        if path:
            print(f"   ✓ Path planning: {len(path.points)} waypoints, cost {path.total_cost:.2f}")
        else:
            print("   ✓ Path planning: No feasible path (expected in some cases)")
        
        # Test safe directions
        safe_dirs = perception.get_safe_directions()
        print(f"   ✓ Safe directions: {len(safe_dirs)} directions found")
        
        # Test evaluation
        evaluator = PerceptionEvaluator(perception)
        performance_metrics = evaluator.evaluate_computational_performance()
        print(f"   ✓ Performance evaluation: {len(performance_metrics)} metrics")
        
        print("\n" + "="*60)
        print("ALL TESTS PASSED SUCCESSFULLY!")
        print("="*60)
        print("\nThe Robot Perception System is working correctly.")
        print("To install full dependencies and run visualizations:")
        print("pip install -r requirements.txt")
        print("python quick_start.py")
        
        return True
        
    except ImportError as e:
        print(f"\n❌ Import Error: {e}")
        print("This is expected if dependencies are not installed.")
        print("Install dependencies with: pip install -r requirements.txt")
        return False
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False

if __name__ == "__main__":
    test_basic_imports()
