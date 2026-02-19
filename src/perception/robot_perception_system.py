"""
Robot Perception System - Modern Implementation

A comprehensive robot perception system for obstacle detection, object recognition,
and environment mapping using modern computer vision and sensor fusion techniques.

Author: AI Assistant
Date: 2024
License: MIT
"""

import numpy as np
import cv2
import open3d as o3d
from typing import List, Tuple, Dict, Optional, Union
from dataclasses import dataclass
from enum import Enum
import logging
import random
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SensorType(Enum):
    """Types of sensors available in the perception system."""
    LIDAR = "lidar"
    CAMERA = "camera"
    DEPTH_CAMERA = "depth_camera"
    ULTRASONIC = "ultrasonic"
    IMU = "imu"


@dataclass
class Obstacle:
    """Represents an obstacle detected in the environment."""
    position: Tuple[float, float]
    confidence: float
    size: Tuple[float, float]
    obstacle_type: str = "unknown"


@dataclass
class SensorReading:
    """Represents a sensor reading with metadata."""
    sensor_type: SensorType
    data: np.ndarray
    timestamp: float
    position: Tuple[float, float, float]
    orientation: Tuple[float, float, float, float]


class RobotPerceptionSystem:
    """
    Modern robot perception system with advanced computer vision capabilities.
    
    Features:
    - Multi-sensor fusion
    - Real-time obstacle detection
    - Environment mapping
    - Object recognition
    - Path planning integration
    """
    
    def __init__(
        self,
        grid_size: Tuple[int, int] = (20, 20),
        robot_position: Tuple[float, float] = (10.0, 10.0),
        sensor_range: float = 5.0,
        resolution: float = 0.1,
        seed: Optional[int] = None
    ):
        """
        Initialize the robot perception system.
        
        Args:
            grid_size: Size of the environment grid (width, height)
            robot_position: Initial position of the robot (x, y)
            sensor_range: Maximum range of sensors in meters
            resolution: Grid resolution in meters per cell
            seed: Random seed for reproducible results
        """
        self.grid_size = grid_size
        self.robot_position = np.array(robot_position, dtype=np.float32)
        self.sensor_range = sensor_range
        self.resolution = resolution
        
        # Initialize deterministic random state
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        # Environment representation
        self.occupancy_grid = np.zeros(grid_size, dtype=np.float32)  # 0=free, 1=occupied, 0.5=unknown
        self.confidence_map = np.zeros(grid_size, dtype=np.float32)  # Confidence in occupancy
        self.obstacle_map = np.zeros(grid_size, dtype=np.float32)  # Obstacle probability
        
        # Sensor data storage
        self.sensor_history: List[SensorReading] = []
        self.detected_obstacles: List[Obstacle] = []
        
        # Camera parameters (simulated)
        self.camera_params = {
            'fx': 525.0, 'fy': 525.0,  # Focal length
            'cx': 320.0, 'cy': 240.0,  # Principal point
            'width': 640, 'height': 480
        }
        
        # Initialize sensors
        self.active_sensors = [SensorType.LIDAR, SensorType.CAMERA, SensorType.DEPTH_CAMERA]
        
        logger.info(f"Robot Perception System initialized with grid size {grid_size}")
    
    def generate_synthetic_environment(
        self,
        num_obstacles: int = 8,
        obstacle_types: List[str] = None
    ) -> None:
        """
        Generate a synthetic environment with obstacles for testing.
        
        Args:
            num_obstacles: Number of obstacles to generate
            obstacle_types: Types of obstacles to generate
        """
        if obstacle_types is None:
            obstacle_types = ['wall', 'box', 'cylinder', 'person']
        
        logger.info(f"Generating synthetic environment with {num_obstacles} obstacles")
        
        for i in range(num_obstacles):
            # Generate random obstacle position
            x = np.random.randint(1, self.grid_size[0] - 1)
            y = np.random.randint(1, self.grid_size[1] - 1)
            
            # Avoid placing obstacles too close to robot
            robot_grid_pos = self.world_to_grid(self.robot_position)
            if np.linalg.norm(np.array([x, y]) - robot_grid_pos) < 3:
                continue
            
            # Set obstacle in occupancy grid
            self.occupancy_grid[x, y] = 1.0
            self.confidence_map[x, y] = 1.0
            
            # Add some variation in obstacle size
            size = np.random.uniform(0.5, 2.0)
            obstacle_type = np.random.choice(obstacle_types)
            
            obstacle = Obstacle(
                position=(x * self.resolution, y * self.resolution),
                confidence=1.0,
                size=(size, size),
                obstacle_type=obstacle_type
            )
            self.detected_obstacles.append(obstacle)
    
    def world_to_grid(self, world_pos: np.ndarray) -> Tuple[int, int]:
        """Convert world coordinates to grid coordinates."""
        grid_x = int(world_pos[0] / self.resolution)
        grid_y = int(world_pos[1] / self.resolution)
        return np.clip(grid_x, 0, self.grid_size[0] - 1), np.clip(grid_y, 0, self.grid_size[1] - 1)
    
    def grid_to_world(self, grid_pos: Tuple[int, int]) -> Tuple[float, float]:
        """Convert grid coordinates to world coordinates."""
        return grid_pos[0] * self.resolution, grid_pos[1] * self.resolution
    
    def simulate_lidar_scan(self) -> np.ndarray:
        """
        Simulate a LiDAR scan around the robot.
        
        Returns:
            Array of distance measurements in meters
        """
        angles = np.linspace(0, 2 * np.pi, 360)  # 360 degree scan
        distances = np.full(360, self.sensor_range)
        
        robot_grid_x, robot_grid_y = self.world_to_grid(self.robot_position)
        
        for i, angle in enumerate(angles):
            # Ray casting to find obstacles
            for r in np.arange(0.1, self.sensor_range, 0.1):
                x = robot_grid_x + int(r * np.cos(angle) / self.resolution)
                y = robot_grid_y + int(r * np.sin(angle) / self.resolution)
                
                if (0 <= x < self.grid_size[0] and 
                    0 <= y < self.grid_size[1] and 
                    self.occupancy_grid[x, y] > 0.5):
                    distances[i] = r
                    break
        
        return distances
    
    def simulate_camera_image(self) -> np.ndarray:
        """
        Simulate a camera image with detected obstacles.
        
        Returns:
            RGB image array
        """
        img = np.zeros((self.camera_params['height'], self.camera_params['width'], 3), dtype=np.uint8)
        
        # Add some background texture
        img[:, :] = [50, 50, 50]  # Dark gray background
        
        # Project obstacles onto image plane
        robot_grid_x, robot_grid_y = self.world_to_grid(self.robot_position)
        
        for x in range(self.grid_size[0]):
            for y in range(self.grid_size[1]):
                if self.occupancy_grid[x, y] > 0.5:
                    # Convert grid to world coordinates
                    world_x, world_y = self.grid_to_world((x, y))
                    
                    # Project to image coordinates (simplified projection)
                    rel_x = world_x - self.robot_position[0]
                    rel_y = world_y - self.robot_position[1]
                    
                    # Simple perspective projection
                    if rel_x > 0:  # Only objects in front
                        img_x = int(self.camera_params['cx'] + rel_y * 100)
                        img_y = int(self.camera_params['cy'] - rel_x * 100)
                        
                        if (0 <= img_x < self.camera_params['width'] and 
                            0 <= img_y < self.camera_params['height']):
                            img[img_y, img_x] = [255, 0, 0]  # Red for obstacles
        
        return img
    
    def detect_obstacles_computer_vision(self, image: np.ndarray) -> List[Obstacle]:
        """
        Detect obstacles using computer vision techniques.
        
        Args:
            image: Input RGB image
            
        Returns:
            List of detected obstacles
        """
        obstacles = []
        
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Define range for red color (obstacles)
        lower_red = np.array([0, 50, 50])
        upper_red = np.array([10, 255, 255])
        
        # Create mask for red objects
        mask = cv2.inRange(hsv, lower_red, upper_red)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            if cv2.contourArea(contour) > 50:  # Filter small detections
                # Get bounding box
                x, y, w, h = cv2.boundingRect(contour)
                
                # Convert image coordinates back to world coordinates
                world_x = self.robot_position[0] + (y - self.camera_params['cy']) / 100
                world_y = self.robot_position[1] + (x - self.camera_params['cx']) / 100
                
                obstacle = Obstacle(
                    position=(world_x, world_y),
                    confidence=min(cv2.contourArea(contour) / 1000, 1.0),
                    size=(w * self.resolution, h * self.resolution),
                    obstacle_type="detected_object"
                )
                obstacles.append(obstacle)
        
        return obstacles
    
    def fuse_sensor_data(self) -> List[Obstacle]:
        """
        Fuse data from multiple sensors to improve obstacle detection.
        
        Returns:
            Fused list of obstacles with improved confidence
        """
        logger.info("Fusing sensor data from multiple sources")
        
        # Get sensor readings
        lidar_distances = self.simulate_lidar_scan()
        camera_image = self.simulate_camera_image()
        cv_obstacles = self.detect_obstacles_computer_vision(camera_image)
        
        # Create sensor readings
        lidar_reading = SensorReading(
            sensor_type=SensorType.LIDAR,
            data=lidar_distances,
            timestamp=0.0,  # Simplified timing
            position=(self.robot_position[0], self.robot_position[1], 0.0),
            orientation=(0.0, 0.0, 0.0, 1.0)
        )
        
        camera_reading = SensorReading(
            sensor_type=SensorType.CAMERA,
            data=camera_image,
            timestamp=0.0,
            position=(self.robot_position[0], self.robot_position[1], 0.0),
            orientation=(0.0, 0.0, 0.0, 1.0)
        )
        
        self.sensor_history.extend([lidar_reading, camera_reading])
        
        # Fuse LiDAR and camera data
        fused_obstacles = []
        
        # Process LiDAR data
        angles = np.linspace(0, 2 * np.pi, len(lidar_distances))
        for i, (angle, distance) in enumerate(zip(angles, lidar_distances)):
            if distance < self.sensor_range * 0.9:  # Valid obstacle detection
                world_x = self.robot_position[0] + distance * np.cos(angle)
                world_y = self.robot_position[1] + distance * np.sin(angle)
                
                obstacle = Obstacle(
                    position=(world_x, world_y),
                    confidence=0.8,  # High confidence from LiDAR
                    size=(0.5, 0.5),  # Default size
                    obstacle_type="lidar_detected"
                )
                fused_obstacles.append(obstacle)
        
        # Add computer vision detections with lower confidence
        for cv_obs in cv_obstacles:
            cv_obs.confidence *= 0.6  # Lower confidence for CV
            fused_obstacles.append(cv_obs)
        
        # Remove duplicates and merge nearby obstacles
        merged_obstacles = self._merge_nearby_obstacles(fused_obstacles)
        
        logger.info(f"Fused sensor data: {len(merged_obstacles)} obstacles detected")
        return merged_obstacles
    
    def _merge_nearby_obstacles(self, obstacles: List[Obstacle], threshold: float = 0.5) -> List[Obstacle]:
        """
        Merge obstacles that are close to each other.
        
        Args:
            obstacles: List of obstacles to merge
            threshold: Distance threshold for merging
            
        Returns:
            List of merged obstacles
        """
        if not obstacles:
            return []
        
        merged = []
        used = set()
        
        for i, obs1 in enumerate(obstacles):
            if i in used:
                continue
            
            # Find nearby obstacles
            nearby_indices = [i]
            for j, obs2 in enumerate(obstacles):
                if j != i and j not in used:
                    distance = np.linalg.norm(
                        np.array(obs1.position) - np.array(obs2.position)
                    )
                    if distance < threshold:
                        nearby_indices.append(j)
            
            # Merge nearby obstacles
            if len(nearby_indices) > 1:
                positions = [obstacles[idx].position for idx in nearby_indices]
                confidences = [obstacles[idx].confidence for idx in nearby_indices]
                
                # Weighted average position
                avg_pos = np.average(positions, axis=0, weights=confidences)
                max_confidence = max(confidences)
                
                merged_obs = Obstacle(
                    position=tuple(avg_pos),
                    confidence=max_confidence,
                    size=obstacles[i].size,
                    obstacle_type="merged"
                )
                merged.append(merged_obs)
                
                used.update(nearby_indices)
            else:
                merged.append(obs1)
                used.add(i)
        
        return merged
    
    def update_occupancy_grid(self, obstacles: List[Obstacle]) -> None:
        """
        Update the occupancy grid based on detected obstacles.
        
        Args:
            obstacles: List of detected obstacles
        """
        for obstacle in obstacles:
            grid_x, grid_y = self.world_to_grid(np.array(obstacle.position))
            
            if (0 <= grid_x < self.grid_size[0] and 
                0 <= grid_y < self.grid_size[1]):
                
                # Update occupancy with confidence weighting
                current_occupancy = self.occupancy_grid[grid_x, grid_y]
                new_occupancy = obstacle.confidence
                
                # Bayesian update
                self.occupancy_grid[grid_x, grid_y] = min(
                    current_occupancy + new_occupancy * 0.1, 1.0
                )
                self.confidence_map[grid_x, grid_y] = max(
                    self.confidence_map[grid_x, grid_y], obstacle.confidence
                )
    
    def get_safe_directions(self) -> List[Tuple[float, float]]:
        """
        Determine safe directions for robot movement based on perception.
        
        Returns:
            List of safe direction vectors
        """
        safe_directions = []
        
        # Sample directions around the robot
        num_directions = 16
        angles = np.linspace(0, 2 * np.pi, num_directions, endpoint=False)
        
        for angle in angles:
            is_safe = True
            
            # Check path in this direction
            for distance in np.arange(0.5, self.sensor_range, 0.2):
                check_x = self.robot_position[0] + distance * np.cos(angle)
                check_y = self.robot_position[1] + distance * np.sin(angle)
                
                grid_x, grid_y = self.world_to_grid(np.array([check_x, check_y]))
                
                if (0 <= grid_x < self.grid_size[0] and 
                    0 <= grid_y < self.grid_size[1] and 
                    self.occupancy_grid[grid_x, grid_y] > 0.3):
                    is_safe = False
                    break
            
            if is_safe:
                safe_directions.append((np.cos(angle), np.sin(angle)))
        
        return safe_directions
    
    def get_perception_summary(self) -> Dict:
        """
        Get a summary of the current perception state.
        
        Returns:
            Dictionary containing perception metrics
        """
        return {
            'robot_position': tuple(self.robot_position),
            'num_obstacles_detected': len(self.detected_obstacles),
            'sensor_range': self.sensor_range,
            'grid_resolution': self.resolution,
            'occupancy_ratio': np.mean(self.occupancy_grid),
            'confidence_mean': np.mean(self.confidence_map),
            'safe_directions_count': len(self.get_safe_directions()),
            'sensor_history_length': len(self.sensor_history)
        }
