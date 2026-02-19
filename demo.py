"""
Robot Perception System - Main Demo Script

Demonstrates the complete robot perception system with visualization,
evaluation, and path planning capabilities.
"""

import numpy as np
import matplotlib.pyplot as plt
import logging
import argparse
import json
from pathlib import Path
import time
from typing import List, Tuple, Dict

# Import our modules
from src.perception.robot_perception_system import RobotPerceptionSystem, Obstacle
from src.perception.visualization import PerceptionVisualizer, PerceptionEvaluator
from src.planning.path_planner import PathPlanner, RobotController, PlanningAlgorithm

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RobotPerceptionDemo:
    """
    Main demo class for the robot perception system.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the demo.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        
        # Initialize perception system
        self.perception_system = RobotPerceptionSystem(
            grid_size=config['grid_size'],
            robot_position=config['robot_position'],
            sensor_range=config['sensor_range'],
            resolution=config['resolution'],
            seed=config.get('seed', 42)
        )
        
        # Initialize visualization and evaluation
        self.visualizer = PerceptionVisualizer(self.perception_system)
        self.evaluator = PerceptionEvaluator(self.perception_system)
        
        # Initialize planning and control
        self.path_planner = PathPlanner(self.perception_system)
        self.controller = RobotController(self.perception_system, self.path_planner)
        
        # Create output directories
        self.output_dir = Path(config.get('output_dir', 'output'))
        self.output_dir.mkdir(exist_ok=True)
        (self.output_dir / 'visualizations').mkdir(exist_ok=True)
        (self.output_dir / 'evaluations').mkdir(exist_ok=True)
        (self.output_dir / 'animations').mkdir(exist_ok=True)
        
        logger.info("Robot Perception Demo initialized")
    
    def run_static_perception_demo(self) -> None:
        """Run static perception demonstration."""
        logger.info("Running static perception demo")
        
        # Generate synthetic environment
        self.perception_system.generate_synthetic_environment(
            num_obstacles=self.config.get('num_obstacles', 8)
        )
        
        # Perform sensor fusion
        detected_obstacles = self.perception_system.fuse_sensor_data()
        
        # Update occupancy grid
        self.perception_system.update_occupancy_grid(detected_obstacles)
        
        # Create visualization
        self.visualizer.plot_perception_state(
            detected_obstacles,
            save_path=self.output_dir / 'visualizations' / 'static_perception.png',
            show_safe_directions=True,
            show_sensor_range=True
        )
        
        # Generate evaluation report
        ground_truth_obstacles = self.perception_system.detected_obstacles.copy()
        ground_truth_map = self.perception_system.occupancy_grid.copy()
        
        report = self.evaluator.generate_evaluation_report(
            ground_truth_obstacles,
            ground_truth_map,
            save_path=self.output_dir / 'evaluations' / 'static_evaluation.json'
        )
        
        logger.info(f"Static demo completed. F1 Score: {report['detection_metrics']['f1_score']:.3f}")
    
    def run_path_planning_demo(self) -> None:
        """Run path planning demonstration."""
        logger.info("Running path planning demo")
        
        # Generate environment
        self.perception_system.generate_synthetic_environment()
        
        # Define goals
        goals = [
            (15.0, 15.0),
            (5.0, 18.0),
            (18.0, 5.0),
            (2.0, 2.0)
        ]
        
        # Test different planning algorithms
        algorithms = [
            PlanningAlgorithm.A_STAR,
            PlanningAlgorithm.RRT,
            PlanningAlgorithm.DIJKSTRA,
            PlanningAlgorithm.POTENTIAL_FIELD
        ]
        
        results = {}
        
        for algorithm in algorithms:
            logger.info(f"Testing {algorithm.value} algorithm")
            algorithm_results = []
            
            for i, goal in enumerate(goals):
                logger.info(f"Planning to goal {i+1}: {goal}")
                
                # Plan path
                start_time = time.time()
                path = self.path_planner.plan_path(goal, algorithm)
                planning_time = time.time() - start_time
                
                if path:
                    algorithm_results.append({
                        'goal': goal,
                        'path_length': len(path.points),
                        'path_cost': path.total_cost,
                        'planning_time': planning_time,
                        'success': True
                    })
                    
                    # Create visualization for this path
                    self._visualize_path(path, goal, algorithm.value, i)
                else:
                    algorithm_results.append({
                        'goal': goal,
                        'success': False,
                        'planning_time': planning_time
                    })
            
            results[algorithm.value] = algorithm_results
        
        # Save results
        with open(self.output_dir / 'evaluations' / 'path_planning_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        # Print summary
        self._print_planning_summary(results)
    
    def run_navigation_demo(self) -> None:
        """Run complete navigation demonstration."""
        logger.info("Running navigation demo")
        
        # Generate environment
        self.perception_system.generate_synthetic_environment()
        
        # Define navigation goals
        goals = [(15.0, 15.0), (5.0, 18.0), (18.0, 5.0)]
        
        navigation_results = []
        robot_trajectory = [tuple(self.perception_system.robot_position)]
        obstacles_history = []
        
        for i, goal in enumerate(goals):
            logger.info(f"Navigating to goal {i+1}: {goal}")
            
            # Execute path
            result = self.controller.execute_path(goal, max_steps=200)
            navigation_results.append(result)
            
            # Record trajectory
            robot_trajectory.append(tuple(self.perception_system.robot_position))
            
            # Record obstacles at this step
            obstacles_history.append(self.perception_system.detected_obstacles.copy())
            
            if result['success']:
                logger.info(f"Successfully reached goal {i+1} in {result['steps_taken']} steps")
            else:
                logger.warning(f"Failed to reach goal {i+1}: {result['reason']}")
        
        # Create animation
        self.visualizer.create_animation(
            robot_trajectory,
            obstacles_history,
            save_path=self.output_dir / 'animations' / 'navigation_demo.gif',
            fps=5
        )
        
        # Save results
        with open(self.output_dir / 'evaluations' / 'navigation_results.json', 'w') as f:
            json.dump(navigation_results, f, indent=2)
        
        logger.info("Navigation demo completed")
    
    def run_comprehensive_evaluation(self) -> None:
        """Run comprehensive evaluation with multiple configurations."""
        logger.info("Running comprehensive evaluation")
        
        # Test different configurations
        configurations = [
            {'sensor_range': 3.0, 'resolution': 0.2, 'num_obstacles': 5},
            {'sensor_range': 5.0, 'resolution': 0.1, 'num_obstacles': 8},
            {'sensor_range': 7.0, 'resolution': 0.15, 'num_obstacles': 12},
            {'sensor_range': 4.0, 'resolution': 0.25, 'num_obstacles': 6}
        ]
        
        evaluation_reports = []
        
        for i, config in enumerate(configurations):
            logger.info(f"Testing configuration {i+1}: {config}")
            
            # Create new perception system with this configuration
            test_system = RobotPerceptionSystem(
                grid_size=(20, 20),
                robot_position=(10.0, 10.0),
                sensor_range=config['sensor_range'],
                resolution=config['resolution'],
                seed=42
            )
            
            # Generate environment
            test_system.generate_synthetic_environment(
                num_obstacles=config['num_obstacles']
            )
            
            # Perform perception
            detected_obstacles = test_system.fuse_sensor_data()
            test_system.update_occupancy_grid(detected_obstacles)
            
            # Evaluate
            evaluator = PerceptionEvaluator(test_system)
            ground_truth_obstacles = test_system.detected_obstacles.copy()
            ground_truth_map = test_system.occupancy_grid.copy()
            
            report = evaluator.generate_evaluation_report(
                ground_truth_obstacles,
                ground_truth_map
            )
            
            # Add configuration info
            report['configuration'] = config
            evaluation_reports.append(report)
        
        # Create leaderboard
        leaderboard = self.evaluator.create_leaderboard(evaluation_reports)
        
        # Save results
        with open(self.output_dir / 'evaluations' / 'comprehensive_evaluation.json', 'w') as f:
            json.dump(evaluation_reports, f, indent=2)
        
        with open(self.output_dir / 'evaluations' / 'leaderboard.json', 'w') as f:
            json.dump(leaderboard, f, indent=2)
        
        # Print leaderboard
        self._print_leaderboard(leaderboard)
    
    def _visualize_path(self, path, goal: Tuple[float, float], algorithm: str, goal_idx: int) -> None:
        """Create visualization for a specific path."""
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Plot occupancy grid
        im = ax.imshow(
            self.perception_system.occupancy_grid.T,
            cmap='RdYlBu_r',
            origin='lower',
            extent=(0, self.perception_system.grid_size[0] * self.perception_system.resolution,
                   0, self.perception_system.grid_size[1] * self.perception_system.resolution),
            vmin=0, vmax=1
        )
        
        # Plot robot position
        ax.scatter(
            self.perception_system.robot_position[0],
            self.perception_system.robot_position[1],
            c='blue', s=200, marker='o', label='Robot', edgecolors='black', linewidth=2
        )
        
        # Plot goal
        ax.scatter(goal[0], goal[1], c='green', s=200, marker='*', label='Goal', edgecolors='black', linewidth=2)
        
        # Plot path
        if path and path.points:
            path_x = [point.position[0] for point in path.points]
            path_y = [point.position[1] for point in path.points]
            ax.plot(path_x, path_y, 'r-', linewidth=3, label=f'Path ({algorithm})')
        
        ax.set_title(f'Path Planning - {algorithm} - Goal {goal_idx+1}')
        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Y Position (m)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        plt.tight_layout()
        plt.savefig(
            self.output_dir / 'visualizations' / f'path_{algorithm}_goal_{goal_idx+1}.png',
            dpi=300, bbox_inches='tight'
        )
        plt.close()
    
    def _print_planning_summary(self, results: Dict) -> None:
        """Print summary of path planning results."""
        print("\n" + "="*60)
        print("PATH PLANNING EVALUATION SUMMARY")
        print("="*60)
        
        for algorithm, algorithm_results in results.items():
            print(f"\n{algorithm.upper()}:")
            successful_plans = sum(1 for r in algorithm_results if r['success'])
            total_plans = len(algorithm_results)
            success_rate = successful_plans / total_plans * 100
            
            if successful_plans > 0:
                avg_cost = np.mean([r['path_cost'] for r in algorithm_results if r['success']])
                avg_time = np.mean([r['planning_time'] for r in algorithm_results if r['success']])
                print(f"  Success Rate: {success_rate:.1f}% ({successful_plans}/{total_plans})")
                print(f"  Average Path Cost: {avg_cost:.2f}")
                print(f"  Average Planning Time: {avg_time:.3f}s")
            else:
                print(f"  Success Rate: 0% (0/{total_plans})")
    
    def _print_leaderboard(self, leaderboard: Dict) -> None:
        """Print evaluation leaderboard."""
        print("\n" + "="*60)
        print("COMPREHENSIVE EVALUATION LEADERBOARD")
        print("="*60)
        
        print(f"\nDetection Performance:")
        print(f"  Best F1 Score: {leaderboard['f1_score']['max']:.3f} ± {leaderboard['f1_score']['std']:.3f}")
        print(f"  Average Precision: {leaderboard['precision']['mean']:.3f} ± {leaderboard['precision']['std']:.3f}")
        print(f"  Average Recall: {leaderboard['recall']['mean']:.3f} ± {leaderboard['recall']['std']:.3f}")
        
        print(f"\nMapping Performance:")
        print(f"  Best Accuracy: {leaderboard['mapping_accuracy']['max']:.3f} ± {leaderboard['mapping_accuracy']['std']:.3f}")
        
        print(f"\nComputational Performance:")
        print(f"  Fastest Fusion Time: {leaderboard['fusion_time']['min']:.1f}ms")
        print(f"  Average Fusion Time: {leaderboard['fusion_time']['mean']:.1f}ms")
        
        if 'best_configurations' in leaderboard:
            print(f"\nBest Configurations:")
            best_det = leaderboard['best_configurations']['best_detection']
            print(f"  Best Detection: F1={best_det['f1_score']:.3f}, Config={best_det['config']}")
            
            best_map = leaderboard['best_configurations']['best_mapping']
            print(f"  Best Mapping: Accuracy={best_map['accuracy']:.3f}, Config={best_map['config']}")


def load_config(config_path: str) -> Dict:
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Robot Perception System Demo')
    parser.add_argument('--config', type=str, default='config/demo_config.json',
                       help='Configuration file path')
    parser.add_argument('--demo', type=str, choices=['static', 'planning', 'navigation', 'evaluation', 'all'],
                       default='all', help='Demo type to run')
    parser.add_argument('--output', type=str, default='output',
                       help='Output directory')
    
    args = parser.parse_args()
    
    # Load configuration
    try:
        config = load_config(args.config)
    except FileNotFoundError:
        logger.warning(f"Config file {args.config} not found, using defaults")
        config = {
            'grid_size': (20, 20),
            'robot_position': (10.0, 10.0),
            'sensor_range': 5.0,
            'resolution': 0.1,
            'num_obstacles': 8,
            'seed': 42,
            'output_dir': args.output
        }
    
    # Create demo
    demo = RobotPerceptionDemo(config)
    
    # Run selected demo
    if args.demo == 'static' or args.demo == 'all':
        demo.run_static_perception_demo()
    
    if args.demo == 'planning' or args.demo == 'all':
        demo.run_path_planning_demo()
    
    if args.demo == 'navigation' or args.demo == 'all':
        demo.run_navigation_demo()
    
    if args.demo == 'evaluation' or args.demo == 'all':
        demo.run_comprehensive_evaluation()
    
    logger.info("Demo completed successfully!")


if __name__ == '__main__':
    main()
