"""
Visualization and Evaluation Module for Robot Perception System

Provides comprehensive visualization tools and evaluation metrics for
the robot perception system.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import cv2
from typing import List, Dict, Tuple, Optional
import logging
from pathlib import Path
import json
import time

from .robot_perception_system import RobotPerceptionSystem, Obstacle, SensorType

logger = logging.getLogger(__name__)


class PerceptionVisualizer:
    """
    Advanced visualization tools for the robot perception system.
    """
    
    def __init__(self, perception_system: RobotPerceptionSystem):
        """
        Initialize the visualizer.
        
        Args:
            perception_system: The robot perception system to visualize
        """
        self.perception_system = perception_system
        self.fig = None
        self.axes = None
        
    def plot_perception_state(
        self,
        obstacles: List[Obstacle],
        save_path: Optional[str] = None,
        show_safe_directions: bool = True,
        show_sensor_range: bool = True
    ) -> None:
        """
        Create a comprehensive visualization of the perception state.
        
        Args:
            obstacles: List of detected obstacles
            save_path: Path to save the plot
            show_safe_directions: Whether to show safe movement directions
            show_sensor_range: Whether to show sensor range circle
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Robot Perception System - Comprehensive View', fontsize=16)
        
        # Main occupancy grid
        ax1 = axes[0, 0]
        self._plot_occupancy_grid(ax1, obstacles, show_safe_directions, show_sensor_range)
        
        # Confidence map
        ax2 = axes[0, 1]
        self._plot_confidence_map(ax2)
        
        # Sensor data visualization
        ax3 = axes[1, 0]
        self._plot_sensor_data(ax3)
        
        # Camera view simulation
        ax4 = axes[1, 1]
        self._plot_camera_view(ax4)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Visualization saved to {save_path}")
        
        plt.show()
    
    def _plot_occupancy_grid(
        self,
        ax,
        obstacles: List[Obstacle],
        show_safe_directions: bool,
        show_sensor_range: bool
    ) -> None:
        """Plot the occupancy grid with robot and obstacles."""
        # Plot occupancy grid
        im = ax.imshow(
            self.perception_system.occupancy_grid.T,
            cmap='RdYlBu_r',
            origin='lower',
            extent=(0, self.perception_system.grid_size[0] * self.perception_system.resolution,
                   0, self.perception_system.grid_size[1] * self.perception_system.resolution),
            vmin=0, vmax=1
        )
        
        # Add colorbar
        plt.colorbar(im, ax=ax, label='Occupancy Probability')
        
        # Plot robot position
        ax.scatter(
            self.perception_system.robot_position[0],
            self.perception_system.robot_position[1],
            c='blue', s=200, marker='o', label='Robot', edgecolors='black', linewidth=2
        )
        
        # Plot detected obstacles
        if obstacles:
            obs_x = [obs.position[0] for obs in obstacles]
            obs_y = [obs.position[1] for obs in obstacles]
            confidences = [obs.confidence for obs in obstacles]
            
            scatter = ax.scatter(obs_x, obs_y, c=confidences, s=100, cmap='Reds',
                               marker='s', label='Detected Obstacles', edgecolors='black')
            plt.colorbar(scatter, ax=ax, label='Detection Confidence')
        
        # Show safe directions
        if show_safe_directions:
            safe_dirs = self.perception_system.get_safe_directions()
            for dx, dy in safe_dirs:
                ax.arrow(
                    self.perception_system.robot_position[0],
                    self.perception_system.robot_position[1],
                    dx * 2, dy * 2,
                    head_width=0.2, head_length=0.2, fc='green', ec='green', alpha=0.7
                )
        
        # Show sensor range
        if show_sensor_range:
            circle = plt.Circle(
                self.perception_system.robot_position,
                self.perception_system.sensor_range,
                fill=False, color='blue', linestyle='--', alpha=0.5, label='Sensor Range'
            )
            ax.add_patch(circle)
        
        ax.set_title('Occupancy Grid & Robot State')
        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Y Position (m)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
    
    def _plot_confidence_map(self, ax) -> None:
        """Plot the confidence map."""
        im = ax.imshow(
            self.perception_system.confidence_map.T,
            cmap='viridis',
            origin='lower',
            extent=(0, self.perception_system.grid_size[0] * self.perception_system.resolution,
                   0, self.perception_system.grid_size[1] * self.perception_system.resolution),
            vmin=0, vmax=1
        )
        
        plt.colorbar(im, ax=ax, label='Confidence')
        ax.set_title('Detection Confidence Map')
        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Y Position (m)')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
    
    def _plot_sensor_data(self, ax) -> None:
        """Plot LiDAR sensor data."""
        if not self.perception_system.sensor_history:
            ax.text(0.5, 0.5, 'No sensor data available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Sensor Data')
            return
        
        # Get latest LiDAR data
        lidar_data = None
        for reading in reversed(self.perception_system.sensor_history):
            if reading.sensor_type == SensorType.LIDAR:
                lidar_data = reading.data
                break
        
        if lidar_data is not None:
            angles = np.linspace(0, 2 * np.pi, len(lidar_data))
            x = lidar_data * np.cos(angles)
            y = lidar_data * np.sin(angles)
            
            ax.scatter(x, y, c=lidar_data, cmap='plasma', s=20)
            ax.set_title('LiDAR Scan Data')
            ax.set_xlabel('X Distance (m)')
            ax.set_ylabel('Y Distance (m)')
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal')
        else:
            ax.text(0.5, 0.5, 'No LiDAR data available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Sensor Data')
    
    def _plot_camera_view(self, ax) -> None:
        """Plot simulated camera view."""
        camera_image = self.perception_system.simulate_camera_image()
        
        # Convert BGR to RGB for matplotlib
        camera_image_rgb = cv2.cvtColor(camera_image, cv2.COLOR_BGR2RGB)
        
        ax.imshow(camera_image_rgb)
        ax.set_title('Simulated Camera View')
        ax.set_xlabel('Pixel X')
        ax.set_ylabel('Pixel Y')
        ax.axis('off')
    
    def create_animation(
        self,
        robot_trajectory: List[Tuple[float, float]],
        obstacles_history: List[List[Obstacle]],
        save_path: Optional[str] = None,
        fps: int = 10
    ) -> FuncAnimation:
        """
        Create an animated visualization of robot movement and perception.
        
        Args:
            robot_trajectory: List of robot positions over time
            obstacles_history: List of obstacle detections over time
            save_path: Path to save the animation
            fps: Frames per second for the animation
            
        Returns:
            Matplotlib animation object
        """
        fig, ax = plt.subplots(figsize=(10, 10))
        
        def animate(frame):
            ax.clear()
            
            if frame < len(robot_trajectory):
                # Update robot position
                self.perception_system.robot_position = np.array(robot_trajectory[frame])
                
                # Get obstacles for this frame
                obstacles = obstacles_history[frame] if frame < len(obstacles_history) else []
                
                # Plot current state
                self._plot_occupancy_grid(ax, obstacles, True, True)
                
                # Plot trajectory
                if len(robot_trajectory) > 1:
                    traj_x = [pos[0] for pos in robot_trajectory[:frame+1]]
                    traj_y = [pos[1] for pos in robot_trajectory[:frame+1]]
                    ax.plot(traj_x, traj_y, 'b-', alpha=0.7, linewidth=2, label='Robot Trajectory')
                
                ax.set_title(f'Robot Perception Animation - Frame {frame}')
        
        anim = FuncAnimation(fig, animate, frames=len(robot_trajectory), interval=1000//fps, repeat=True)
        
        if save_path:
            anim.save(save_path, writer='pillow', fps=fps)
            logger.info(f"Animation saved to {save_path}")
        
        return anim


class PerceptionEvaluator:
    """
    Comprehensive evaluation metrics for the robot perception system.
    """
    
    def __init__(self, perception_system: RobotPerceptionSystem):
        """
        Initialize the evaluator.
        
        Args:
            perception_system: The robot perception system to evaluate
        """
        self.perception_system = perception_system
        self.metrics_history = []
        
    def evaluate_detection_performance(
        self,
        ground_truth_obstacles: List[Obstacle],
        detected_obstacles: List[Obstacle],
        distance_threshold: float = 0.5
    ) -> Dict[str, float]:
        """
        Evaluate obstacle detection performance using standard metrics.
        
        Args:
            ground_truth_obstacles: True obstacles in the environment
            detected_obstacles: Obstacles detected by the system
            distance_threshold: Distance threshold for considering a detection correct
            
        Returns:
            Dictionary of evaluation metrics
        """
        if not ground_truth_obstacles:
            return {
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
                'false_positive_rate': 0.0,
                'miss_rate': 0.0
            }
        
        # Calculate true positives, false positives, and false negatives
        tp = 0  # True positives
        fp = 0  # False positives
        fn = 0  # False negatives
        
        detected_positions = [obs.position for obs in detected_obstacles]
        gt_positions = [obs.position for obs in ground_truth_obstacles]
        
        # Find matches between detected and ground truth
        matched_detections = set()
        matched_gt = set()
        
        for i, det_pos in enumerate(detected_positions):
            for j, gt_pos in enumerate(gt_positions):
                distance = np.linalg.norm(np.array(det_pos) - np.array(gt_pos))
                if distance <= distance_threshold and j not in matched_gt:
                    tp += 1
                    matched_detections.add(i)
                    matched_gt.add(j)
                    break
        
        fp = len(detected_obstacles) - tp
        fn = len(ground_truth_obstacles) - tp
        
        # Calculate metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        false_positive_rate = fp / len(ground_truth_obstacles) if ground_truth_obstacles else 0.0
        miss_rate = fn / len(ground_truth_obstacles) if ground_truth_obstacles else 0.0
        
        metrics = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'false_positive_rate': false_positive_rate,
            'miss_rate': miss_rate,
            'true_positives': tp,
            'false_positives': fp,
            'false_negatives': fn
        }
        
        self.metrics_history.append(metrics)
        return metrics
    
    def evaluate_mapping_accuracy(
        self,
        ground_truth_map: np.ndarray,
        distance_threshold: float = 0.5
    ) -> Dict[str, float]:
        """
        Evaluate the accuracy of the occupancy grid mapping.
        
        Args:
            ground_truth_map: True occupancy grid
            distance_threshold: Distance threshold for accuracy calculation
            
        Returns:
            Dictionary of mapping accuracy metrics
        """
        if ground_truth_map.shape != self.perception_system.occupancy_grid.shape:
            logger.warning("Ground truth map shape doesn't match occupancy grid")
            return {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0}
        
        # Binary classification: occupied vs free
        gt_binary = (ground_truth_map > 0.5).astype(int)
        pred_binary = (self.perception_system.occupancy_grid > 0.5).astype(int)
        
        # Calculate accuracy metrics
        correct = np.sum(gt_binary == pred_binary)
        total = gt_binary.size
        
        accuracy = correct / total
        
        # Precision and recall for occupied cells
        tp = np.sum((gt_binary == 1) & (pred_binary == 1))
        fp = np.sum((gt_binary == 0) & (pred_binary == 1))
        fn = np.sum((gt_binary == 1) & (pred_binary == 0))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'total_cells': total,
            'correct_cells': correct
        }
    
    def evaluate_computational_performance(self) -> Dict[str, float]:
        """
        Evaluate computational performance metrics.
        
        Returns:
            Dictionary of performance metrics
        """
        # Measure processing time for different operations
        times = {}
        
        # LiDAR simulation time
        start_time = time.time()
        _ = self.perception_system.simulate_lidar_scan()
        times['lidar_simulation_ms'] = (time.time() - start_time) * 1000
        
        # Camera simulation time
        start_time = time.time()
        _ = self.perception_system.simulate_camera_image()
        times['camera_simulation_ms'] = (time.time() - start_time) * 1000
        
        # Obstacle detection time
        camera_image = self.perception_system.simulate_camera_image()
        start_time = time.time()
        _ = self.perception_system.detect_obstacles_computer_vision(camera_image)
        times['obstacle_detection_ms'] = (time.time() - start_time) * 1000
        
        # Sensor fusion time
        start_time = time.time()
        _ = self.perception_system.fuse_sensor_data()
        times['sensor_fusion_ms'] = (time.time() - start_time) * 1000
        
        # Memory usage (approximate)
        times['memory_usage_mb'] = (
            self.perception_system.occupancy_grid.nbytes +
            self.perception_system.confidence_map.nbytes +
            self.perception_system.obstacle_map.nbytes
        ) / (1024 * 1024)
        
        return times
    
    def generate_evaluation_report(
        self,
        ground_truth_obstacles: List[Obstacle],
        ground_truth_map: np.ndarray,
        save_path: Optional[str] = None
    ) -> Dict:
        """
        Generate a comprehensive evaluation report.
        
        Args:
            ground_truth_obstacles: True obstacles for detection evaluation
            ground_truth_map: True occupancy grid for mapping evaluation
            save_path: Path to save the report
            
        Returns:
            Complete evaluation report
        """
        logger.info("Generating comprehensive evaluation report")
        
        # Get current detected obstacles
        detected_obstacles = self.perception_system.fuse_sensor_data()
        
        # Evaluate different aspects
        detection_metrics = self.evaluate_detection_performance(ground_truth_obstacles, detected_obstacles)
        mapping_metrics = self.evaluate_mapping_accuracy(ground_truth_map)
        performance_metrics = self.evaluate_computational_performance()
        perception_summary = self.perception_system.get_perception_summary()
        
        # Compile report
        report = {
            'timestamp': time.time(),
            'detection_metrics': detection_metrics,
            'mapping_metrics': mapping_metrics,
            'performance_metrics': performance_metrics,
            'perception_summary': perception_summary,
            'system_config': {
                'grid_size': self.perception_system.grid_size,
                'sensor_range': self.perception_system.sensor_range,
                'resolution': self.perception_system.resolution,
                'active_sensors': [sensor.value for sensor in self.perception_system.active_sensors]
            }
        }
        
        # Save report if requested
        if save_path:
            with open(save_path, 'w') as f:
                json.dump(report, f, indent=2)
            logger.info(f"Evaluation report saved to {save_path}")
        
        return report
    
    def create_leaderboard(self, reports: List[Dict]) -> Dict:
        """
        Create a leaderboard comparing multiple evaluation reports.
        
        Args:
            reports: List of evaluation reports
            
        Returns:
            Leaderboard with rankings and statistics
        """
        if not reports:
            return {}
        
        # Extract key metrics
        metrics_data = {
            'f1_score': [r['detection_metrics']['f1_score'] for r in reports],
            'precision': [r['detection_metrics']['precision'] for r in reports],
            'recall': [r['detection_metrics']['recall'] for r in reports],
            'mapping_accuracy': [r['mapping_metrics']['accuracy'] for r in reports],
            'lidar_time': [r['performance_metrics']['lidar_simulation_ms'] for r in reports],
            'fusion_time': [r['performance_metrics']['sensor_fusion_ms'] for r in reports]
        }
        
        # Calculate statistics
        leaderboard = {}
        for metric, values in metrics_data.items():
            leaderboard[metric] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'median': np.median(values)
            }
        
        # Find best configurations
        best_f1_idx = np.argmax(metrics_data['f1_score'])
        best_accuracy_idx = np.argmax(metrics_data['mapping_accuracy'])
        fastest_idx = np.argmin(metrics_data['fusion_time'])
        
        leaderboard['best_configurations'] = {
            'best_detection': {
                'index': int(best_f1_idx),
                'f1_score': metrics_data['f1_score'][best_f1_idx],
                'config': reports[best_f1_idx]['system_config']
            },
            'best_mapping': {
                'index': int(best_accuracy_idx),
                'accuracy': metrics_data['mapping_accuracy'][best_accuracy_idx],
                'config': reports[best_accuracy_idx]['system_config']
            },
            'fastest': {
                'index': int(fastest_idx),
                'fusion_time_ms': metrics_data['fusion_time'][fastest_idx],
                'config': reports[fastest_idx]['system_config']
            }
        }
        
        return leaderboard
