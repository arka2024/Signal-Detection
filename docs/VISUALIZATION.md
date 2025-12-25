# Traffic Signal Visualization and Demo Videos

This folder contains demo videos showing the adaptive traffic signal control system in action.
Videos demonstrate the system's ability to analyze traffic patterns and make intelligent signal timing decisions.

## Available Videos

### 1. Basic Signal Operation Demo
- **Filename**: `basic_signal_operation.mp4`
- **Description**: Demonstrates the basic operation of the adaptive signal controller, showing transitions between phases and real-time metrics.
- **Duration**: 2 minutes

### 2. Rush Hour Traffic Response Demo
- **Filename**: `rush_hour_adaptation.mp4`
- **Description**: Shows how the system adapts to increased traffic volume during simulated rush hour conditions.
- **Duration**: 3 minutes

### 3. Comparison of Control Modes
- **Filename**: `control_mode_comparison.mp4`
- **Description**: Side-by-side comparison of different control modes:
  - Pure RL-based control
  - Rule-based control from traffic analysis
  - Hybrid mode combining both approaches
- **Duration**: 5 minutes

### 4. Special Event Traffic Management
- **Filename**: `special_event_management.mp4`
- **Description**: Demonstrates system response to sudden traffic increase simulating a special event (concert, sporting event, etc.)
- **Duration**: 4 minutes

### 5. Emergency Vehicle Priority Demo
- **Filename**: `emergency_vehicle_priority.mp4`
- **Description**: Shows how the system detects and prioritizes emergency vehicles through intersections.
- **Duration**: 2 minutes

## How to Generate Your Own Demo Videos

You can generate your own visualization videos using the visualization system:

```
python visualization.py --video path/to/traffic/footage.mp4 --use-analysis --model models/adaptive_controller.pth
```

Options:
- `--video`: Path to traffic video footage or camera index
- `--model`: Path to trained adaptive controller model
- `--use-analysis`: Use traffic analysis results for optimization
- `--duration`: Duration to run in seconds
- `--width`: Window width (default: 1280)
- `--height`: Window height (default: 720)

During visualization, press 'R' to start/stop recording.

## Connecting to Real Signal Systems

For integration with actual traffic signal hardware:

1. The system outputs standardized signal commands via a signal API interface
2. Signal commands include:
   - Phase index (0-3)
   - Phase duration in seconds
   - Priority level (normal/emergency)
   
3. Hardware integration requires implementing the signal interface in `src/hardware/signal_interface.py`

## Real-world Deployment Considerations

When deploying to real traffic systems:

1. Ensure proper fail-safe mechanisms are in place
2. Implement manual override capabilities for traffic managers
3. Validate system performance in a closed test environment before deployment
4. Comply with local traffic signal regulations and standards
5. Configure system to log all decisions for audit and performance analysis
