"""
Setup script for Traffic Signal Management System
"""

import os
import sys
import subprocess
import argparse
import shutil
import platform
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Setup Traffic Signal Management System')
    parser.add_argument('--skip-dependencies', action='store_true', 
                       help='Skip installing Python dependencies')
    parser.add_argument('--skip-sumo', action='store_true',
                       help='Skip SUMO installation')
    parser.add_argument('--create-sample-data', action='store_true',
                       help='Create sample data files')
    return parser.parse_args()

def install_python_dependencies():
    """Install Python dependencies from requirements.txt"""
    logger.info("Installing Python dependencies...")
    
    try:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'], check=True)
        logger.info("Dependencies installed successfully.")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error installing dependencies: {e}")
        sys.exit(1)

def check_sumo_installation():
    """Check if SUMO is installed and set up environment variable"""
    logger.info("Checking SUMO installation...")
    
    # Check if SUMO_HOME is set
    sumo_home = os.environ.get('SUMO_HOME')
    if sumo_home:
        if os.path.exists(sumo_home):
            logger.info(f"SUMO found at {sumo_home}")
            return True
        else:
            logger.warning(f"SUMO_HOME set to {sumo_home}, but directory doesn't exist")
    
    # Try to find SUMO in common locations
    common_paths = []
    
    if platform.system() == 'Windows':
        program_files = os.environ.get('PROGRAMFILES', 'C:\\Program Files')
        common_paths = [
            os.path.join(program_files, 'Eclipse', 'Sumo'),
            os.path.join(program_files, 'SUMO')
        ]
    elif platform.system() == 'Linux':
        common_paths = [
            '/usr/local/share/sumo',
            '/usr/share/sumo'
        ]
    elif platform.system() == 'Darwin':  # macOS
        common_paths = [
            '/Applications/SUMO.app/Contents/MacOS',
            '/usr/local/share/sumo'
        ]
    
    # Check common paths
    for path in common_paths:
        if os.path.exists(path) and os.path.exists(os.path.join(path, 'bin')):
            logger.info(f"Found SUMO at {path}")
            # Set SUMO_HOME environment variable for this session
            os.environ['SUMO_HOME'] = path
            
            # Suggest to set it permanently
            if platform.system() == 'Windows':
                logger.info("To set SUMO_HOME permanently, run:")
                logger.info(f"setx SUMO_HOME \"{path}\"")
            else:
                logger.info("To set SUMO_HOME permanently, add to your shell profile:")
                logger.info(f"export SUMO_HOME=\"{path}\"")
            
            return True
    
    logger.warning("SUMO not found. You'll need to install it manually.")
    return False

def install_sumo():
    """Provide instructions for installing SUMO"""
    logger.info("SUMO (Simulation of Urban MObility) needs to be installed.")
    
    if platform.system() == 'Windows':
        logger.info("Download SUMO for Windows from: https://sumo.dlr.de/docs/Downloads.php")
        logger.info("After installation, set the SUMO_HOME environment variable.")
    
    elif platform.system() == 'Linux':
        logger.info("On Ubuntu/Debian, you can install SUMO with:")
        logger.info("sudo add-apt-repository ppa:sumo/stable")
        logger.info("sudo apt-get update")
        logger.info("sudo apt-get install sumo sumo-tools sumo-doc")
    
    elif platform.system() == 'Darwin':
        logger.info("On macOS, you can install SUMO with Homebrew:")
        logger.info("brew tap dlr-ts/sumo")
        logger.info("brew install sumo")
    
    logger.info("For more details, visit: https://sumo.dlr.de/docs/Installing/index.html")
    
    # Ask if user wants to continue without SUMO
    response = input("Continue setup without SUMO? (y/n): ")
    if response.lower() not in ['y', 'yes']:
        sys.exit(1)

def create_sample_data():
    """Create sample data files for simulation"""
    logger.info("Creating sample data files...")
    
    # Create data directories
    os.makedirs(os.path.join('data', 'networks'), exist_ok=True)
    os.makedirs(os.path.join('data', 'routes'), exist_ok=True)
    
    # Create a simple intersection network file
    network_file = os.path.join('data', 'networks', 'default.net.xml')
    with open(network_file, 'w') as f:
        f.write("""<?xml version="1.0" encoding="UTF-8"?>
<net version="1.9.0" junctionCornerDetail="5" limitTurnSpeed="5.50" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/net_file.xsd">
    <location netOffset="100.00,100.00" convBoundary="0.00,0.00,200.00,200.00" origBoundary="-100.00,-100.00,100.00,100.00" projParameter="!"/>
    
    <!-- Junction definitions -->
    <junction id="center" type="traffic_light" x="100.00" y="100.00" incLanes="north_to_center_0 east_to_center_0 south_to_center_0 west_to_center_0" intLanes=":center_0_0 :center_1_0 :center_2_0 :center_3_0 :center_4_0 :center_5_0 :center_6_0 :center_7_0 :center_8_0 :center_9_0 :center_10_0 :center_11_0" shape="96.80,109.60 103.20,109.60 103.60,106.80 104.20,104.60 105.00,102.90 106.00,101.60 107.20,100.70 108.60,100.20 109.60,96.80 109.60,103.20 112.40,103.60 114.60,104.20 116.30,105.00 117.60,106.00 118.50,107.20 119.00,108.60 103.20,109.60 102.90,112.40 102.20,114.60 101.00,116.30 99.40,117.60 97.50,118.50 95.40,119.00 90.40,103.20 89.60,102.90 87.40,102.20 85.70,101.00 84.40,99.40 83.50,97.50 83.00,95.40 96.80,90.40 97.10,89.60 97.80,87.40 99.00,85.70 100.60,84.40 102.50,83.50 104.60,83.00" radius="5.00"/>
    <junction id="east" type="dead_end" x="200.00" y="100.00" incLanes="center_to_east_0" intLanes="" shape="200.00,96.80 200.00,103.20"/>
    <junction id="north" type="dead_end" x="100.00" y="200.00" incLanes="center_to_north_0" intLanes="" shape="103.20,200.00 96.80,200.00"/>
    <junction id="south" type="dead_end" x="100.00" y="0.00" incLanes="center_to_south_0" intLanes="" shape="96.80,0.00 103.20,0.00"/>
    <junction id="west" type="dead_end" x="0.00" y="100.00" incLanes="center_to_west_0" intLanes="" shape="0.00,103.20 0.00,96.80"/>
    
    <!-- Edge definitions -->
    <edge id="center_to_east" from="center" to="east" priority="1">
        <lane id="center_to_east_0" index="0" speed="13.89" length="90.40" shape="108.60,100.20 200.00,100.00"/>
    </edge>
    <edge id="center_to_north" from="center" to="north" priority="1">
        <lane id="center_to_north_0" index="0" speed="13.89" length="90.40" shape="103.20,109.60 100.00,200.00"/>
    </edge>
    <edge id="center_to_south" from="center" to="south" priority="1">
        <lane id="center_to_south_0" index="0" speed="13.89" length="90.40" shape="96.80,90.40 100.00,0.00"/>
    </edge>
    <edge id="center_to_west" from="center" to="west" priority="1">
        <lane id="center_to_west_0" index="0" speed="13.89" length="90.40" shape="90.40,103.20 0.00,100.00"/>
    </edge>
    <edge id="east_to_center" from="east" to="center" priority="1">
        <lane id="east_to_center_0" index="0" speed="13.89" length="90.40" shape="200.00,96.80 109.60,96.80"/>
    </edge>
    <edge id="north_to_center" from="north" to="center" priority="1">
        <lane id="north_to_center_0" index="0" speed="13.89" length="90.40" shape="96.80,200.00 96.80,109.60"/>
    </edge>
    <edge id="south_to_center" from="south" to="center" priority="1">
        <lane id="south_to_center_0" index="0" speed="13.89" length="90.40" shape="103.20,0.00 103.20,90.40"/>
    </edge>
    <edge id="west_to_center" from="west" to="center" priority="1">
        <lane id="west_to_center_0" index="0" speed="13.89" length="90.40" shape="0.00,103.20 90.40,103.20"/>
    </edge>

    <!-- Traffic light definitions -->
    <tlLogic id="center" type="static" programID="0" offset="0">
        <phase duration="31" state="GGgrrrGGgrrr"/>
        <phase duration="4"  state="yygrrryygrrr"/>
        <phase duration="31" state="rrrGGgrrrGGg"/>
        <phase duration="4"  state="rrryygrrryyg"/>
    </tlLogic>

    <!-- Connection definitions -->
    <connection from="east_to_center" to="center_to_north" fromLane="0" toLane="0" via=":center_4_0" tl="center" linkIndex="4" dir="r" state="o"/>
    <connection from="east_to_center" to="center_to_west" fromLane="0" toLane="0" via=":center_5_0" tl="center" linkIndex="5" dir="s" state="o"/>
    <connection from="east_to_center" to="center_to_south" fromLane="0" toLane="0" via=":center_6_0" tl="center" linkIndex="6" dir="l" state="o"/>
    <connection from="north_to_center" to="center_to_west" fromLane="0" toLane="0" via=":center_0_0" tl="center" linkIndex="0" dir="r" state="o"/>
    <connection from="north_to_center" to="center_to_south" fromLane="0" toLane="0" via=":center_1_0" tl="center" linkIndex="1" dir="s" state="o"/>
    <connection from="north_to_center" to="center_to_east" fromLane="0" toLane="0" via=":center_2_0" tl="center" linkIndex="2" dir="l" state="o"/>
    <connection from="south_to_center" to="center_to_east" fromLane="0" toLane="0" via=":center_8_0" tl="center" linkIndex="8" dir="r" state="o"/>
    <connection from="south_to_center" to="center_to_north" fromLane="0" toLane="0" via=":center_9_0" tl="center" linkIndex="9" dir="s" state="o"/>
    <connection from="south_to_center" to="center_to_west" fromLane="0" toLane="0" via=":center_10_0" tl="center" linkIndex="10" dir="l" state="o"/>
    <connection from="west_to_center" to="center_to_south" fromLane="0" toLane="0" via=":center_3_0" tl="center" linkIndex="3" dir="r" state="o"/>
    <connection from="west_to_center" to="center_to_east" fromLane="0" toLane="0" via=":center_7_0" tl="center" linkIndex="7" dir="s" state="o"/>
    <connection from="west_to_center" to="center_to_north" fromLane="0" toLane="0" via=":center_11_0" tl="center" linkIndex="11" dir="l" state="o"/>
</net>
""")
    logger.info(f"Created network file: {network_file}")
    
    # Create a simple routes file
    routes_file = os.path.join('data', 'routes', 'default.rou.xml')
    with open(routes_file, 'w') as f:
        f.write("""<?xml version="1.0" encoding="UTF-8"?>
<routes>
    <!-- Vehicle types -->
    <vType id="car" accel="3.0" decel="6.0" sigma="0.5" length="5.0" minGap="2.5" maxSpeed="16.67" guiShape="passenger"/>
    <vType id="bus" accel="2.0" decel="4.0" sigma="0.5" length="12.0" minGap="3.0" maxSpeed="13.89" guiShape="bus"/>
    <vType id="truck" accel="1.0" decel="4.0" sigma="0.5" length="10.0" minGap="3.0" maxSpeed="11.11" guiShape="truck"/>
    <vType id="motorcycle" accel="4.0" decel="7.0" sigma="0.5" length="2.0" minGap="1.5" maxSpeed="19.44" guiShape="motorcycle"/>
    <vType id="bicycle" accel="2.0" decel="4.0" sigma="0.5" length="1.5" minGap="1.0" maxSpeed="5.56" guiShape="bicycle"/>
    
    <!-- Routes -->
    <route id="north_to_south" edges="north_to_center center_to_south"/>
    <route id="north_to_east" edges="north_to_center center_to_east"/>
    <route id="north_to_west" edges="north_to_center center_to_west"/>
    
    <route id="south_to_north" edges="south_to_center center_to_north"/>
    <route id="south_to_east" edges="south_to_center center_to_east"/>
    <route id="south_to_west" edges="south_to_center center_to_west"/>
    
    <route id="east_to_west" edges="east_to_center center_to_west"/>
    <route id="east_to_north" edges="east_to_center center_to_north"/>
    <route id="east_to_south" edges="east_to_center center_to_south"/>
    
    <route id="west_to_east" edges="west_to_center center_to_east"/>
    <route id="west_to_north" edges="west_to_center center_to_north"/>
    <route id="west_to_south" edges="west_to_center center_to_south"/>
    
    <!-- Traffic flow - regular pattern -->
    <flow id="flow_n_s" type="car" route="north_to_south" begin="0" end="3600" vehsPerHour="300"/>
    <flow id="flow_n_e" type="car" route="north_to_east" begin="0" end="3600" vehsPerHour="100"/>
    <flow id="flow_n_w" type="car" route="north_to_west" begin="0" end="3600" vehsPerHour="100"/>
    
    <flow id="flow_s_n" type="car" route="south_to_north" begin="0" end="3600" vehsPerHour="300"/>
    <flow id="flow_s_e" type="car" route="south_to_east" begin="0" end="3600" vehsPerHour="100"/>
    <flow id="flow_s_w" type="car" route="south_to_west" begin="0" end="3600" vehsPerHour="100"/>
    
    <flow id="flow_e_w" type="car" route="east_to_west" begin="0" end="3600" vehsPerHour="400"/>
    <flow id="flow_e_n" type="car" route="east_to_north" begin="0" end="3600" vehsPerHour="150"/>
    <flow id="flow_e_s" type="car" route="east_to_south" begin="0" end="3600" vehsPerHour="150"/>
    
    <flow id="flow_w_e" type="car" route="west_to_east" begin="0" end="3600" vehsPerHour="400"/>
    <flow id="flow_w_n" type="car" route="west_to_north" begin="0" end="3600" vehsPerHour="150"/>
    <flow id="flow_w_s" type="car" route="west_to_south" begin="0" end="3600" vehsPerHour="150"/>
    
    <!-- Add some buses -->
    <flow id="bus_n_s" type="bus" route="north_to_south" begin="0" end="3600" period="300"/>
    <flow id="bus_s_n" type="bus" route="south_to_north" begin="0" end="3600" period="300"/>
    <flow id="bus_e_w" type="bus" route="east_to_west" begin="0" end="3600" period="300"/>
    <flow id="bus_w_e" type="bus" route="west_to_east" begin="0" end="3600" period="300"/>
    
    <!-- Add some trucks -->
    <flow id="truck_n_s" type="truck" route="north_to_south" begin="0" end="3600" period="600"/>
    <flow id="truck_e_w" type="truck" route="east_to_west" begin="0" end="3600" period="400"/>
</routes>
""")
    logger.info(f"Created routes file: {routes_file}")

def main():
    """Main function"""
    logger.info("Setting up Traffic Signal Management System...")
    
    # Parse command line arguments
    args = parse_args()
    
    # Install Python dependencies
    if not args.skip_dependencies:
        install_python_dependencies()
    
    # Check SUMO installation
    if not args.skip_sumo:
        if not check_sumo_installation():
            install_sumo()
    
    # Create sample data
    if args.create_sample_data:
        create_sample_data()
    
    logger.info("Setup complete!")
    logger.info("You can run the system with:")
    logger.info("  python main.py --mode train   # Train the RL agent")
    logger.info("  python main.py --mode test    # Test the trained agent")
    logger.info("  python main.py --mode demo    # Run the vision demo (requires video file)")
    logger.info("  streamlit run dashboard/dashboard.py  # Launch the dashboard")

if __name__ == "__main__":
    main()
