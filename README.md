# Intelligent Traffic Signal Management System

## Overview
An AI-based traffic management system designed to optimize signal timings and reduce congestion in urban areas. The system uses reinforcement learning to analyze real-time traffic data and make dynamic signal timing decisions to improve traffic flow.

## Project Goals
- Reduce average commute time by 10% in simulated urban environments
- Create an interactive dashboard for traffic authorities to monitor and control signals
- Implement adaptive signal timing based on real-time traffic conditions
- Handle special conditions like peak hours, festivals, and emergencies

## Components
1. **Traffic Simulation Environment**: SUMO (Simulation of Urban MObility) based traffic simulator
2. **Reinforcement Learning Model**: Deep Q-Network for traffic signal optimization
3. **Computer Vision Module**: For processing camera feeds (primarily ANPR cameras)
4. **Dashboard**: Real-time monitoring and manual control interface
5. **Data Collection & Processing**: Integration with traffic cameras and future IoT sensors

## Technical Implementation
- **Computer Vision**: OpenCV for traffic detection and density estimation
- **Machine Learning**: PyTorch/TensorFlow for reinforcement learning models
- **Simulation**: SUMO for urban traffic simulation
- **Dashboard**: Flask/Streamlit web interface

## Setup Instructions

### Prerequisites
- Python 3.8 or later
- SUMO (Simulation of Urban MObility) - See [SUMO_SETUP.md](./SUMO_SETUP.md) for installation instructions
- PyTorch or TensorFlow (as specified in requirements.txt)

### Installation

1. Clone the repository:
```
git clone https://github.com/username/traffic-signal-management.git
cd traffic-signal-management
```

2. Install Python dependencies:
```
pip install -r requirements.txt
```

3. Set up SUMO (optional - required for full simulation):
```
# Follow instructions in SUMO_SETUP.md
# or run:
./install_sumo.bat  # Windows
```

4. Create sample data:
```
python setup.py --create-sample-data
```

### Running the System

#### Development Mode (No SUMO required):
```
python simple_test.py
```

#### Full Training (SUMO required):
```
python main.py --mode train --episodes 1000
```

#### Testing a Trained Model:
```
python main.py --mode test --model models/dqn_agent_final.pth
```

#### Dashboard:
```
streamlit run dashboard/dashboard.py
```

## Future Prospects
- Integration with IoT sensors for more granular traffic data
- Handling India-specific scenarios (roadside parking, festival congestion)
- Emergency vehicle priority system
- Predictive analytics for proactive traffic management
