<<<<<<< HEAD
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
=======
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
>>>>>>> aa24d7e0891a97a482b890c56651bf2a883f56d5



<<<<<<< HEAD
# Setting Up SUMO for Traffic Signal Management System

This guide will help you install and configure SUMO (Simulation of Urban MObility) for use with the Traffic Signal Management System.

## What is SUMO?

SUMO (Simulation of Urban MObility) is an open-source, microscopic and continuous traffic simulation package designed to handle large road networks. It's a key component of our Traffic Signal Management System, providing realistic traffic simulation for our reinforcement learning models.

## Installation Options

### 1. Windows Installation

1. **Download the installer**:
   - Visit the [SUMO Downloads page](https://sumo.dlr.de/docs/Downloads.html)
   - Download the latest Windows installer (e.g., `sumo-win64-1.15.0.msi`)

2. **Run the installer**:
   - Execute the downloaded MSI file
   - Follow the installation wizard
   - Recommended installation path: `C:\Program Files (x86)\Eclipse\Sumo` or `C:\Program Files\Eclipse\Sumo`

3. **Set environment variables**:
   - You can use our helper script: `install_sumo.bat`
   - Or manually set the following:
     - Set `SUMO_HOME` to your SUMO installation directory (e.g., `C:\Program Files (x86)\Eclipse\Sumo`)
     - Add `%SUMO_HOME%\bin` to your PATH

### 2. Linux Installation

For Ubuntu/Debian:
```bash
sudo add-apt-repository ppa:sumo/stable
sudo apt-get update
sudo apt-get install sumo sumo-tools sumo-doc
```

Set environment variables:
```bash
echo 'export SUMO_HOME=/usr/share/sumo' >> ~/.bashrc
source ~/.bashrc
```

### 3. macOS Installation

Using Homebrew:
```bash
brew tap dlr-ts/sumo
brew install sumo
```

Set environment variables:
```bash
echo 'export SUMO_HOME=/usr/local/opt/sumo/share/sumo' >> ~/.bash_profile
source ~/.bash_profile
```

## Verifying Installation

Run our test script to verify that SUMO is properly installed:
```
python test_sumo.py
```

You should see confirmation that:
- SUMO_HOME is correctly set
- SUMO binaries are found
- The traci Python module can be imported

## Running without SUMO (Development Mode)

For development purposes, the system includes a mock mode that simulates SUMO without requiring actual installation. This is useful for testing the overall architecture but will not provide realistic traffic simulation.

To run in mock mode, simply run the system without setting up SUMO:
```
python simple_test.py
```

Note that for full functionality and training, a proper SUMO installation is required.

## Troubleshooting

### Common Issues

1. **"No module named 'traci'"**:
   - Ensure SUMO_HOME is correctly set
   - Make sure that `$SUMO_HOME/tools` is in your Python path
   - Add this to your scripts: `import os, sys; sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))`

2. **"'SUMO_HOME' is not recognized"**:
   - You need to set the SUMO_HOME environment variable
   - Use the `install_sumo.bat` script or set it manually

3. **Issues with network/route files**:
   - Make sure the data files exist
   - Run `python setup.py --create-sample-data` to create sample files

## Additional Resources

- [SUMO Documentation](https://sumo.dlr.de/docs/index.html)
- [TraCI Documentation](https://sumo.dlr.de/docs/TraCI.html) (Traffic Control Interface)
- [SUMO Tutorials](https://sumo.dlr.de/docs/Tutorials/index.html)
=======
# Setting Up SUMO for Traffic Signal Management System

This guide will help you install and configure SUMO (Simulation of Urban MObility) for use with the Traffic Signal Management System.

## What is SUMO?

SUMO (Simulation of Urban MObility) is an open-source, microscopic and continuous traffic simulation package designed to handle large road networks. It's a key component of our Traffic Signal Management System, providing realistic traffic simulation for our reinforcement learning models.

## Installation Options

### 1. Windows Installation

1. **Download the installer**:
   - Visit the [SUMO Downloads page](https://sumo.dlr.de/docs/Downloads.html)
   - Download the latest Windows installer (e.g., `sumo-win64-1.15.0.msi`)

2. **Run the installer**:
   - Execute the downloaded MSI file
   - Follow the installation wizard
   - Recommended installation path: `C:\Program Files (x86)\Eclipse\Sumo` or `C:\Program Files\Eclipse\Sumo`

3. **Set environment variables**:
   - You can use our helper script: `install_sumo.bat`
   - Or manually set the following:
     - Set `SUMO_HOME` to your SUMO installation directory (e.g., `C:\Program Files (x86)\Eclipse\Sumo`)
     - Add `%SUMO_HOME%\bin` to your PATH

### 2. Linux Installation

For Ubuntu/Debian:
```bash
sudo add-apt-repository ppa:sumo/stable
sudo apt-get update
sudo apt-get install sumo sumo-tools sumo-doc
```

Set environment variables:
```bash
echo 'export SUMO_HOME=/usr/share/sumo' >> ~/.bashrc
source ~/.bashrc
```

### 3. macOS Installation

Using Homebrew:
```bash
brew tap dlr-ts/sumo
brew install sumo
```

Set environment variables:
```bash
echo 'export SUMO_HOME=/usr/local/opt/sumo/share/sumo' >> ~/.bash_profile
source ~/.bash_profile
```

## Verifying Installation

Run our test script to verify that SUMO is properly installed:
```
python test_sumo.py
```

You should see confirmation that:
- SUMO_HOME is correctly set
- SUMO binaries are found
- The traci Python module can be imported

## Running without SUMO (Development Mode)

For development purposes, the system includes a mock mode that simulates SUMO without requiring actual installation. This is useful for testing the overall architecture but will not provide realistic traffic simulation.

To run in mock mode, simply run the system without setting up SUMO:
```
python simple_test.py
```

Note that for full functionality and training, a proper SUMO installation is required.

## Troubleshooting

### Common Issues

1. **"No module named 'traci'"**:
   - Ensure SUMO_HOME is correctly set
   - Make sure that `$SUMO_HOME/tools` is in your Python path
   - Add this to your scripts: `import os, sys; sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))`

2. **"'SUMO_HOME' is not recognized"**:
   - You need to set the SUMO_HOME environment variable
   - Use the `install_sumo.bat` script or set it manually

3. **Issues with network/route files**:
   - Make sure the data files exist
   - Run `python setup.py --create-sample-data` to create sample files

## Additional Resources

- [SUMO Documentation](https://sumo.dlr.de/docs/index.html)
- [TraCI Documentation](https://sumo.dlr.de/docs/TraCI.html) (Traffic Control Interface)
- [SUMO Tutorials](https://sumo.dlr.de/docs/Tutorials/index.html)
>>>>>>> aa24d7e0891a97a482b890c56651bf2a883f56d5

