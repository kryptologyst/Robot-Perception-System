"""
Robot Perception System Package

A comprehensive robot perception system for obstacle detection, object recognition,
and environment mapping using modern computer vision and sensor fusion techniques.
"""

__version__ = "1.0.0"
__author__ = "AI Assistant"
__email__ = "ai@example.com"

from .robot_perception_system import RobotPerceptionSystem, Obstacle, SensorType, SensorReading
from .visualization import PerceptionVisualizer, PerceptionEvaluator

__all__ = [
    "RobotPerceptionSystem",
    "Obstacle", 
    "SensorType",
    "SensorReading",
    "PerceptionVisualizer",
    "PerceptionEvaluator"
]
