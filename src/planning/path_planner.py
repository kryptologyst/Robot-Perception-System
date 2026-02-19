"""
Path Planning and Control Integration Module

Integrates the perception system with path planning and control algorithms
for autonomous robot navigation.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
import logging
from dataclasses import dataclass
from enum import Enum
import heapq
from scipy.spatial.distance import cdist
from scipy.optimize import minimize

from .robot_perception_system import RobotPerceptionSystem, Obstacle

logger = logging.getLogger(__name__)


class PlanningAlgorithm(Enum):
    """Available path planning algorithms."""
    A_STAR = "a_star"
    RRT = "rrt"
    DIJKSTRA = "dijkstra"
    POTENTIAL_FIELD = "potential_field"


@dataclass
class PathPoint:
    """Represents a point in a planned path."""
    position: Tuple[float, float]
    velocity: Tuple[float, float] = (0.0, 0.0)
    timestamp: float = 0.0
    cost: float = 0.0


@dataclass
class Path:
    """Represents a complete path from start to goal."""
    points: List[PathPoint]
    total_cost: float
    algorithm: PlanningAlgorithm
    is_feasible: bool = True


class PathPlanner:
    """
    Advanced path planning system integrated with perception.
    """
    
    def __init__(self, perception_system: RobotPerceptionSystem):
        """
        Initialize the path planner.
        
        Args:
            perception_system: The robot perception system
        """
        self.perception_system = perception_system
        self.current_path: Optional[Path] = None
        self.goal_position: Optional[Tuple[float, float]] = None
        
        # Planning parameters
        self.max_iterations = 1000
        self.step_size = 0.5
        self.goal_tolerance = 0.3
        self.collision_margin = 0.2
        
    def plan_path(
        self,
        goal: Tuple[float, float],
        algorithm: PlanningAlgorithm = PlanningAlgorithm.A_STAR,
        max_planning_time: float = 5.0
    ) -> Optional[Path]:
        """
        Plan a path from current robot position to goal.
        
        Args:
            goal: Target position (x, y)
            algorithm: Planning algorithm to use
            max_planning_time: Maximum time allowed for planning
            
        Returns:
            Planned path or None if no path found
        """
        self.goal_position = goal
        
        logger.info(f"Planning path to {goal} using {algorithm.value}")
        
        if algorithm == PlanningAlgorithm.A_STAR:
            path = self._plan_a_star(goal)
        elif algorithm == PlanningAlgorithm.RRT:
            path = self._plan_rrt(goal)
        elif algorithm == PlanningAlgorithm.DIJKSTRA:
            path = self._plan_dijkstra(goal)
        elif algorithm == PlanningAlgorithm.POTENTIAL_FIELD:
            path = self._plan_potential_field(goal)
        else:
            raise ValueError(f"Unknown planning algorithm: {algorithm}")
        
        if path and path.is_feasible:
            self.current_path = path
            logger.info(f"Path planned successfully with {len(path.points)} points, cost: {path.total_cost:.2f}")
        else:
            logger.warning("No feasible path found")
        
        return path
    
    def _plan_a_star(self, goal: Tuple[float, float]) -> Optional[Path]:
        """Plan path using A* algorithm."""
        start = tuple(self.perception_system.robot_position)
        
        # Convert to grid coordinates
        start_grid = self.perception_system.world_to_grid(np.array(start))
        goal_grid = self.perception_system.world_to_grid(np.array(goal))
        
        # Priority queue: (f_cost, g_cost, position)
        open_set = [(0, 0, start_grid)]
        came_from = {}
        g_cost = {start_grid: 0}
        
        while open_set:
            current_f, current_g, current = heapq.heappop(open_set)
            
            # Check if goal reached
            if np.linalg.norm(np.array(current) - np.array(goal_grid)) < 2:
                # Reconstruct path
                path_points = self._reconstruct_path(came_from, current, start_grid)
                if path_points:
                    return Path(
                        points=path_points,
                        total_cost=current_g,
                        algorithm=PlanningAlgorithm.A_STAR
                    )
            
            # Explore neighbors
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    
                    neighbor = (current[0] + dx, current[1] + dy)
                    
                    # Check bounds
                    if (neighbor[0] < 0 or neighbor[0] >= self.perception_system.grid_size[0] or
                        neighbor[1] < 0 or neighbor[1] >= self.perception_system.grid_size[1]):
                        continue
                    
                    # Check collision
                    if self.perception_system.occupancy_grid[neighbor[0], neighbor[1]] > 0.3:
                        continue
                    
                    # Calculate costs
                    tentative_g = current_g + np.sqrt(dx*dx + dy*dy)
                    
                    if neighbor not in g_cost or tentative_g < g_cost[neighbor]:
                        g_cost[neighbor] = tentative_g
                        h_cost = np.linalg.norm(np.array(neighbor) - np.array(goal_grid))
                        f_cost = tentative_g + h_cost
                        
                        heapq.heappush(open_set, (f_cost, tentative_g, neighbor))
                        came_from[neighbor] = current
        
        return None
    
    def _plan_rrt(self, goal: Tuple[float, float]) -> Optional[Path]:
        """Plan path using Rapidly-exploring Random Tree (RRT)."""
        start = tuple(self.perception_system.robot_position)
        
        # Tree structure: {node: parent}
        tree = {start: None}
        nodes = [start]
        
        for _ in range(self.max_iterations):
            # Sample random point
            if np.random.random() < 0.1:  # 10% chance to sample goal
                sample = goal
            else:
                sample = self._sample_random_point()
            
            # Find nearest node
            nearest_node = min(nodes, key=lambda n: np.linalg.norm(np.array(n) - np.array(sample)))
            
            # Extend towards sample
            direction = np.array(sample) - np.array(nearest_node)
            distance = np.linalg.norm(direction)
            
            if distance > self.step_size:
                direction = direction / distance * self.step_size
            
            new_node = tuple(np.array(nearest_node) + direction)
            
            # Check collision
            if self._is_collision_free(nearest_node, new_node):
                tree[new_node] = nearest_node
                nodes.append(new_node)
                
                # Check if goal reached
                if np.linalg.norm(np.array(new_node) - np.array(goal)) < self.goal_tolerance:
                    # Reconstruct path
                    path_points = self._reconstruct_rrt_path(tree, new_node, start)
                    if path_points:
                        return Path(
                            points=path_points,
                            total_cost=self._calculate_path_cost(path_points),
                            algorithm=PlanningAlgorithm.RRT
                        )
        
        return None
    
    def _plan_dijkstra(self, goal: Tuple[float, float]) -> Optional[Path]:
        """Plan path using Dijkstra's algorithm."""
        start = tuple(self.perception_system.robot_position)
        
        # Convert to grid coordinates
        start_grid = self.perception_system.world_to_grid(np.array(start))
        goal_grid = self.perception_system.world_to_grid(np.array(goal))
        
        # Priority queue: (distance, position)
        distances = {start_grid: 0}
        pq = [(0, start_grid)]
        visited = set()
        came_from = {}
        
        while pq:
            current_dist, current = heapq.heappop(pq)
            
            if current in visited:
                continue
            
            visited.add(current)
            
            # Check if goal reached
            if np.linalg.norm(np.array(current) - np.array(goal_grid)) < 2:
                path_points = self._reconstruct_path(came_from, current, start_grid)
                if path_points:
                    return Path(
                        points=path_points,
                        total_cost=current_dist,
                        algorithm=PlanningAlgorithm.DIJKSTRA
                    )
            
            # Explore neighbors
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    
                    neighbor = (current[0] + dx, current[1] + dy)
                    
                    if neighbor in visited:
                        continue
                    
                    # Check bounds and collision
                    if (neighbor[0] < 0 or neighbor[0] >= self.perception_system.grid_size[0] or
                        neighbor[1] < 0 or neighbor[1] >= self.perception_system.grid_size[1] or
                        self.perception_system.occupancy_grid[neighbor[0], neighbor[1]] > 0.3):
                        continue
                    
                    edge_cost = np.sqrt(dx*dx + dy*dy)
                    new_dist = current_dist + edge_cost
                    
                    if neighbor not in distances or new_dist < distances[neighbor]:
                        distances[neighbor] = new_dist
                        heapq.heappush(pq, (new_dist, neighbor))
                        came_from[neighbor] = current
        
        return None
    
    def _plan_potential_field(self, goal: Tuple[float, float]) -> Optional[Path]:
        """Plan path using potential field method."""
        start = tuple(self.perception_system.robot_position)
        
        def potential_function(pos):
            """Calculate potential at given position."""
            x, y = pos
            
            # Attractive potential (goal)
            goal_dist = np.linalg.norm(np.array(pos) - np.array(goal))
            attractive = 0.5 * goal_dist**2
            
            # Repulsive potential (obstacles)
            repulsive = 0
            for obstacle in self.perception_system.detected_obstacles:
                obs_dist = np.linalg.norm(np.array(pos) - np.array(obstacle.position))
                if obs_dist < 2.0:  # Influence radius
                    repulsive += 0.5 * (1/obs_dist - 1/2.0)**2
            
            return attractive + repulsive
        
        # Gradient descent
        current_pos = np.array(start)
        path_points = [PathPoint(position=tuple(current_pos))]
        
        for _ in range(self.max_iterations):
            # Calculate gradient numerically
            eps = 0.01
            grad_x = (potential_function((current_pos[0] + eps, current_pos[1])) - 
                     potential_function((current_pos[0] - eps, current_pos[1]))) / (2 * eps)
            grad_y = (potential_function((current_pos[0], current_pos[1] + eps)) - 
                     potential_function((current_pos[0], current_pos[1] - eps))) / (2 * eps)
            
            gradient = np.array([grad_x, grad_y])
            
            # Update position
            current_pos -= 0.1 * gradient
            
            # Check if goal reached
            if np.linalg.norm(current_pos - np.array(goal)) < self.goal_tolerance:
                path_points.append(PathPoint(position=tuple(goal)))
                break
            
            # Check collision
            if self._is_position_collision(current_pos):
                logger.warning("Potential field planning hit obstacle")
                return None
            
            path_points.append(PathPoint(position=tuple(current_pos)))
        
        if path_points:
            return Path(
                points=path_points,
                total_cost=self._calculate_path_cost(path_points),
                algorithm=PlanningAlgorithm.POTENTIAL_FIELD
            )
        
        return None
    
    def _sample_random_point(self) -> Tuple[float, float]:
        """Sample a random point in the environment."""
        x = np.random.uniform(0, self.perception_system.grid_size[0] * self.perception_system.resolution)
        y = np.random.uniform(0, self.perception_system.grid_size[1] * self.perception_system.resolution)
        return (x, y)
    
    def _is_collision_free(self, start: Tuple[float, float], end: Tuple[float, float]) -> bool:
        """Check if the path between two points is collision-free."""
        # Sample points along the line
        num_samples = int(np.linalg.norm(np.array(end) - np.array(start)) / 0.1) + 1
        
        for i in range(num_samples + 1):
            t = i / num_samples
            pos = np.array(start) + t * (np.array(end) - np.array(start))
            
            if self._is_position_collision(pos):
                return False
        
        return True
    
    def _is_position_collision(self, position: np.ndarray) -> bool:
        """Check if a position is in collision."""
        grid_x, grid_y = self.perception_system.world_to_grid(position)
        
        if (grid_x < 0 or grid_x >= self.perception_system.grid_size[0] or
            grid_y < 0 or grid_y >= self.perception_system.grid_size[1]):
            return True
        
        return self.perception_system.occupancy_grid[grid_x, grid_y] > 0.3
    
    def _reconstruct_path(self, came_from: Dict, current: Tuple[int, int], start: Tuple[int, int]) -> List[PathPoint]:
        """Reconstruct path from A* or Dijkstra search."""
        path = []
        
        while current in came_from:
            world_pos = self.perception_system.grid_to_world(current)
            path.append(PathPoint(position=world_pos))
            current = came_from[current]
        
        # Add start position
        world_pos = self.perception_system.grid_to_world(start)
        path.append(PathPoint(position=world_pos))
        
        # Reverse to get start-to-goal path
        path.reverse()
        
        return path
    
    def _reconstruct_rrt_path(self, tree: Dict, goal: Tuple[float, float], start: Tuple[float, float]) -> List[PathPoint]:
        """Reconstruct path from RRT tree."""
        path = []
        current = goal
        
        while current is not None:
            path.append(PathPoint(position=current))
            current = tree[current]
        
        # Reverse to get start-to-goal path
        path.reverse()
        
        return path
    
    def _calculate_path_cost(self, path_points: List[PathPoint]) -> float:
        """Calculate total cost of a path."""
        if len(path_points) < 2:
            return 0.0
        
        total_cost = 0.0
        for i in range(1, len(path_points)):
            dist = np.linalg.norm(
                np.array(path_points[i].position) - np.array(path_points[i-1].position)
            )
            total_cost += dist
        
        return total_cost
    
    def get_next_waypoint(self) -> Optional[Tuple[float, float]]:
        """
        Get the next waypoint from the current path.
        
        Returns:
            Next waypoint position or None if no path
        """
        if not self.current_path or not self.current_path.points:
            return None
        
        # Find closest point on path to current robot position
        robot_pos = self.perception_system.robot_position
        min_dist = float('inf')
        closest_idx = 0
        
        for i, point in enumerate(self.current_path.points):
            dist = np.linalg.norm(np.array(point.position) - robot_pos)
            if dist < min_dist:
                min_dist = dist
                closest_idx = i
        
        # Return next waypoint
        if closest_idx + 1 < len(self.current_path.points):
            return self.current_path.points[closest_idx + 1].position
        
        return None
    
    def is_path_valid(self) -> bool:
        """
        Check if the current path is still valid given updated perception.
        
        Returns:
            True if path is valid, False otherwise
        """
        if not self.current_path:
            return False
        
        # Check each segment of the path for collisions
        for i in range(len(self.current_path.points) - 1):
            start = self.current_path.points[i].position
            end = self.current_path.points[i + 1].position
            
            if not self._is_collision_free(start, end):
                logger.info("Path segment collision detected, path invalidated")
                return False
        
        return True


class RobotController:
    """
    Simple robot controller for following planned paths.
    """
    
    def __init__(self, perception_system: RobotPerceptionSystem, path_planner: PathPlanner):
        """
        Initialize the robot controller.
        
        Args:
            perception_system: The robot perception system
            path_planner: The path planner
        """
        self.perception_system = perception_system
        self.path_planner = path_planner
        
        # Control parameters
        self.max_velocity = 1.0  # m/s
        self.max_angular_velocity = 1.0  # rad/s
        self.position_tolerance = 0.2  # m
        self.orientation_tolerance = 0.1  # rad
        
        # Safety limits
        self.emergency_stop_distance = 0.5  # m
        self.min_safe_distance = 0.3  # m
        
    def compute_control_command(self) -> Tuple[float, float]:
        """
        Compute control command (linear and angular velocity).
        
        Returns:
            Tuple of (linear_velocity, angular_velocity)
        """
        # Check for emergency stop conditions
        if self._check_emergency_stop():
            logger.warning("Emergency stop activated")
            return (0.0, 0.0)
        
        # Get next waypoint
        next_waypoint = self.path_planner.get_next_waypoint()
        if next_waypoint is None:
            return (0.0, 0.0)
        
        # Calculate control command
        robot_pos = self.perception_system.robot_position
        target_pos = np.array(next_waypoint)
        
        # Calculate desired velocity vector
        direction = target_pos - robot_pos
        distance = np.linalg.norm(direction)
        
        if distance < self.position_tolerance:
            return (0.0, 0.0)
        
        # Normalize direction and scale by max velocity
        direction = direction / distance
        linear_velocity = min(self.max_velocity, distance * 2.0)  # Proportional control
        
        # Calculate angular velocity (simplified)
        angle_to_target = np.arctan2(direction[1], direction[0])
        angular_velocity = np.clip(angle_to_target * 2.0, -self.max_angular_velocity, self.max_angular_velocity)
        
        return (linear_velocity, angular_velocity)
    
    def _check_emergency_stop(self) -> bool:
        """
        Check if emergency stop conditions are met.
        
        Returns:
            True if emergency stop should be activated
        """
        robot_pos = self.perception_system.robot_position
        
        # Check for obstacles too close
        for obstacle in self.perception_system.detected_obstacles:
            distance = np.linalg.norm(np.array(obstacle.position) - robot_pos)
            if distance < self.emergency_stop_distance:
                return True
        
        # Check occupancy grid for nearby obstacles
        robot_grid_x, robot_grid_y = self.perception_system.world_to_grid(robot_pos)
        
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                check_x = robot_grid_x + dx
                check_y = robot_grid_y + dy
                
                if (0 <= check_x < self.perception_system.grid_size[0] and
                    0 <= check_y < self.perception_system.grid_size[1]):
                    
                    if self.perception_system.occupancy_grid[check_x, check_y] > 0.5:
                        distance = np.sqrt(dx*dx + dy*dy) * self.perception_system.resolution
                        if distance < self.min_safe_distance:
                            return True
        
        return False
    
    def execute_path(self, goal: Tuple[float, float], max_steps: int = 100) -> Dict:
        """
        Execute a complete path to the goal.
        
        Args:
            goal: Target position
            max_steps: Maximum number of control steps
            
        Returns:
            Execution results dictionary
        """
        logger.info(f"Executing path to goal {goal}")
        
        # Plan initial path
        path = self.path_planner.plan_path(goal)
        if not path:
            return {
                'success': False,
                'reason': 'No feasible path found',
                'final_position': tuple(self.perception_system.robot_position),
                'steps_taken': 0
            }
        
        initial_position = tuple(self.perception_system.robot_position)
        steps_taken = 0
        
        for step in range(max_steps):
            # Check if goal reached
            current_pos = self.perception_system.robot_position
            distance_to_goal = np.linalg.norm(current_pos - np.array(goal))
            
            if distance_to_goal < self.position_tolerance:
                logger.info(f"Goal reached in {steps_taken} steps")
                return {
                    'success': True,
                    'reason': 'Goal reached',
                    'final_position': tuple(current_pos),
                    'steps_taken': steps_taken,
                    'path_cost': path.total_cost
                }
            
            # Check if path is still valid
            if not self.path_planner.is_path_valid():
                logger.info("Path invalidated, replanning")
                path = self.path_planner.plan_path(goal)
                if not path:
                    return {
                        'success': False,
                        'reason': 'Path replanning failed',
                        'final_position': tuple(current_pos),
                        'steps_taken': steps_taken
                    }
            
            # Compute and apply control command
            linear_vel, angular_vel = self.compute_control_command()
            
            # Simulate robot movement (simplified)
            dt = 0.1  # Time step
            self.perception_system.robot_position += np.array([
                linear_vel * np.cos(angular_vel * dt),
                linear_vel * np.sin(angular_vel * dt)
            ]) * dt
            
            steps_taken += 1
        
        logger.warning(f"Path execution timeout after {max_steps} steps")
        return {
            'success': False,
            'reason': 'Execution timeout',
            'final_position': tuple(self.perception_system.robot_position),
            'steps_taken': steps_taken
        }
