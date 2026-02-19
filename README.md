# Robot Perception System

A comprehensive robot perception system for obstacle detection, object recognition, and environment mapping using modern computer vision and sensor fusion techniques.

## Features

- **Multi-sensor Fusion**: Integrates LiDAR, camera, and depth sensor data
- **Advanced Computer Vision**: Real-time obstacle detection using OpenCV
- **Environment Mapping**: Occupancy grid mapping with confidence tracking
- **Path Planning**: Multiple algorithms (A*, RRT, Dijkstra, Potential Field)
- **Robot Control**: Safe navigation with emergency stop capabilities
- **Comprehensive Evaluation**: Detection metrics, mapping accuracy, and performance benchmarks
- **Interactive Visualization**: Real-time perception state and trajectory visualization

## Quick Start

### Prerequisites

- Python 3.10+
- Required packages (see requirements.txt)

### Installation

```bash
# Clone the repository
git clone https://github.com/kryptologyst/Robot-Perception-System.git
cd Robot-Perception-System

# Install dependencies
pip install -r requirements.txt

# Run the demo
python demo.py --demo all
```

### Basic Usage

```python
from src.perception.robot_perception_system import RobotPerceptionSystem
from src.perception.visualization import PerceptionVisualizer

# Initialize perception system
perception = RobotPerceptionSystem(
    grid_size=(20, 20),
    robot_position=(10.0, 10.0),
    sensor_range=5.0
)

# Generate environment and detect obstacles
perception.generate_synthetic_environment(num_obstacles=8)
obstacles = perception.fuse_sensor_data()

# Visualize results
visualizer = PerceptionVisualizer(perception)
visualizer.plot_perception_state(obstacles)
```

## System Architecture

### Perception Module
- **RobotPerceptionSystem**: Core perception system with multi-sensor fusion
- **Sensor Simulation**: LiDAR, camera, and depth sensor simulation
- **Computer Vision**: OpenCV-based obstacle detection
- **Environment Mapping**: Occupancy grid with confidence tracking

### Planning Module
- **PathPlanner**: Multiple path planning algorithms
- **RobotController**: Safe robot control with emergency stops
- **Collision Avoidance**: Real-time collision detection and avoidance

### Visualization Module
- **PerceptionVisualizer**: Comprehensive visualization tools
- **PerceptionEvaluator**: Performance evaluation and metrics
- **Animation Support**: Real-time trajectory and perception animation

## Configuration

The system is configured via JSON files in the `config/` directory:

- `demo_config.json`: Main demo configuration
- `planning_config.json`: Path planning parameters
- `control_config.json`: Robot control parameters

## Evaluation Metrics

### Detection Performance
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1 Score**: Harmonic mean of precision and recall
- **False Positive Rate**: False positives / Total ground truth obstacles

### Mapping Accuracy
- **Occupancy Accuracy**: Correct occupancy classifications / Total cells
- **Precision**: Correct occupied cells / Total predicted occupied cells
- **Recall**: Correct occupied cells / Total ground truth occupied cells

### Computational Performance
- **Processing Time**: LiDAR simulation, camera processing, obstacle detection
- **Memory Usage**: Occupancy grid and sensor data storage
- **Real-time Capability**: Frame rate and latency measurements

## Demo Modes

### Static Perception Demo
```bash
python demo.py --demo static
```
Demonstrates obstacle detection and environment mapping without robot movement.

### Path Planning Demo
```bash
python demo.py --demo planning
```
Tests different path planning algorithms (A*, RRT, Dijkstra, Potential Field) with multiple goals.

### Navigation Demo
```bash
python demo.py --demo navigation
```
Complete navigation demonstration with robot movement and trajectory recording.

### Comprehensive Evaluation
```bash
python demo.py --demo evaluation
```
Runs multiple configurations and generates performance leaderboard.

## Safety Features

- **Emergency Stop**: Automatic stop when obstacles are too close
- **Collision Avoidance**: Real-time collision detection and avoidance
- **Safe Distance Monitoring**: Continuous monitoring of safe distances
- **Velocity Limits**: Maximum velocity and acceleration constraints

## Coordinate Frames

The system uses standard robotics coordinate conventions:
- **X-axis**: Forward direction
- **Y-axis**: Left direction
- **Origin**: Robot starting position
- **Units**: Meters for distances, radians for angles

## Sensor Specifications

### Simulated LiDAR
- **Range**: Configurable (default: 5.0m)
- **Angular Resolution**: 1 degree
- **Update Rate**: 10 Hz

### Simulated Camera
- **Resolution**: 640x480 pixels
- **Field of View**: 60 degrees
- **Focal Length**: 525 pixels
- **Update Rate**: 30 Hz

### Simulated Depth Camera
- **Range**: 0.1m - 5.0m
- **Resolution**: 640x480 pixels
- **Update Rate**: 30 Hz

## File Structure

```
robot-perception-system/
├── src/
│   ├── perception/
│   │   ├── robot_perception_system.py
│   │   └── visualization.py
│   └── planning/
│       └── path_planner.py
├── config/
│   └── demo_config.json
├── demo.py
├── requirements.txt
├── README.md
└── DISCLAIMER.md
```

## Output Files

The system generates various output files in the `output/` directory:

- `visualizations/`: Static plots and visualizations
- `evaluations/`: Performance metrics and evaluation reports
- `animations/`: Animated demonstrations and trajectories

## Known Limitations

1. **Simulation Only**: This system is designed for research and education, not real-world deployment
2. **Simplified Dynamics**: Robot dynamics are simplified for demonstration purposes
3. **Limited Sensor Models**: Sensor models are basic approximations
4. **No Real-time Hardware**: No integration with real robot hardware
5. **Single Robot**: No multi-robot coordination capabilities

## Contributing

This project is designed for educational and research purposes. Contributions should focus on:
- Improved sensor models
- Additional planning algorithms
- Enhanced visualization capabilities
- Performance optimizations
- Extended evaluation metrics

## License

MIT License - See LICENSE file for details.

## Disclaimer

**IMPORTANT**: This system is designed for research and educational purposes only. It should NOT be used for real-world robot deployment without expert review and proper safety measures. The system lacks critical safety features required for real-world operation and may not handle edge cases or failure modes appropriately.
# Robot-Perception-System
