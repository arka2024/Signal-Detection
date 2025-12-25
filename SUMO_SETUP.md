
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
