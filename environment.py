"""
Traffic Simulation Environment using SUMO and OpenAI Gym
"""

import os
import sys
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import yaml
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Check if SUMO_HOME is in the environment variables
sumo_installed = False
if 'SUMO_HOME' in os.environ:
    try:
        tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
        sys.path.append(tools)
        import traci
        sumo_installed = True
    except ImportError:
        logger.warning("SUMO tools path exists but traci module not found.")
else:
    logger.warning("SUMO_HOME environment variable not set. Running in mock mode.")
    
# Create a mock traci module if not available
if not sumo_installed:
    logger.warning("Creating mock SUMO environment. This is for development only.")
    
    class MockTraCI:
        """Mock TraCI module for development without SUMO"""
        @staticmethod
        def isLoaded():
            return False
            
        @staticmethod
        def start(*args, **kwargs):
            logger.info("Mock TraCI: Starting simulation")
            return
            
        @staticmethod
        def simulationStep():
            return
            
        @staticmethod
        def close():
            return
            
        class trafficlight:
            @staticmethod
            def getIDList():
                return ["traffic_light_1"]
                
            @staticmethod
            def setPhase(tl_id, phase_index):
                return
                
            @staticmethod
            def getControlledLanes(tl_id):
                return [f"lane_{i}" for i in range(8)]
                
        class lane:
            @staticmethod
            def getLastStepVehicleNumber(lane_id):
                return np.random.randint(0, 10)
                
            @staticmethod
            def getLastStepMeanSpeed(lane_id):
                return np.random.uniform(0, 15)
                
            @staticmethod
            def getLastStepHaltingNumber(lane_id):
                return np.random.randint(0, 5)
                
            @staticmethod
            def getWaitingTime(lane_id):
                return np.random.uniform(0, 30)
    
    # Replace traci with mock implementation
    import types
    traci = MockTraCI()

class TrafficSignalEnv(gym.Env):
    """
    Traffic Signal Control Environment using SUMO simulation
    """
    
    def __init__(self, config_file="config.yaml"):
        """Initialize the traffic signal environment"""
        super(TrafficSignalEnv, self).__init__()
        
        # Load configuration
        with open(config_file, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # SUMO configuration
        self.network_file = self.config['simulation']['network_file']
        self.routes_file = self.config['simulation']['routes_file']
        self.gui = self.config['simulation']['gui']
        self.max_steps = self.config['simulation']['max_steps']
        
        # Initialize SUMO command
        self.sumo_cmd = self._get_sumo_command()
        
        # Initialize state and action spaces
        self._init_spaces()
        
        # Current simulation step
        self.current_step = 0
        
        # List of traffic lights in the network
        self.traffic_lights = []
        
        # Simulation metrics
        self.metrics = {
            'average_waiting_time': 0,
            'queue_length': 0,
            'throughput': 0,
            'congestion_index': 0
        }
    
    def _get_sumo_command(self):
        """Generate the SUMO command to start the simulation"""
        # Check if SUMO is installed
        if 'SUMO_HOME' in os.environ:
            if self.gui:
                sumo_binary = os.path.join(os.environ['SUMO_HOME'], 'bin', 'sumo-gui')
            else:
                sumo_binary = os.path.join(os.environ['SUMO_HOME'], 'bin', 'sumo')
                
            return [
                sumo_binary,
                '-n', self.network_file,
                '-r', self.routes_file,
                '--no-warnings',
                '--random',
                '--start',
                '--quit-on-end'
            ]
        else:
            # Return a mock command for development when SUMO is not installed
            logger.warning("SUMO not installed, returning mock command")
            return ["mock_sumo_command"]
    
    def _init_spaces(self):
        """Initialize state and action spaces"""
        # State space: traffic density, queue length, waiting time per lane
        # Simplified for initial implementation - will expand with actual lanes
        num_lanes = 8  # Assuming 4 approaches with 2 lanes each
        
        # State includes: vehicles count, average speed, queue length for each lane
        self.observation_space = spaces.Box(
            low=0, 
            high=np.inf, 
            shape=(num_lanes, 3),
            dtype=np.float32
        )
        
        # Action space: phase selection for each traffic light
        # Simplified for initial implementation with fixed phases
        num_phases = 4  # Common phases for a 4-way intersection
        self.action_space = spaces.Discrete(num_phases)
    
    def reset(self, **kwargs):
        """Reset the environment to initial state"""
        # Close existing SUMO connection if any
        self._close_traci()
        
        try:
            # Start a new simulation
            traci.start(self.sumo_cmd)
            
            # Get all traffic lights
            self.traffic_lights = traci.trafficlight.getIDList()
        except Exception as e:
            logger.warning(f"Error starting SUMO simulation: {e}")
            logger.warning("Running in mock mode")
            self.traffic_lights = ["traffic_light_1"]  # Mock traffic light
        
        # Reset step counter
        self.current_step = 0
        
        # Get initial state
        state = self._get_state()
        
        return state, {}  # Return state and info dict (gym standard)
    
    def step(self, action):
        """Execute action and advance simulation by one step"""
        # Apply the selected phase to the traffic light
        self._apply_action(action)
        
        # Advance simulation by one step
        traci.simulationStep()
        self.current_step += 1
        
        # Get new state
        new_state = self._get_state()
        
        # Calculate reward
        reward = self._get_reward()
        
        # Check if simulation is done
        done = self.current_step >= self.max_steps
        
        # Update metrics
        self._update_metrics()
        
        return new_state, reward, done, False, {'metrics': self.metrics}
    
    def _apply_action(self, action):
        """Apply action to traffic light"""
        # For simplicity, we'll control only the first traffic light
        if self.traffic_lights:
            traffic_light_id = self.traffic_lights[0]
            # Map action to phase index
            traci.trafficlight.setPhase(traffic_light_id, action)
    
    def _get_state(self):
        """Get current state of the environment"""
        # For simplicity, initial implementation will just use random values
        # In a real implementation, this would use SUMO API to get actual traffic data
        state = np.zeros((8, 3), dtype=np.float32)
        
        if self.traffic_lights:
            traffic_light_id = self.traffic_lights[0]
            lanes = traci.trafficlight.getControlledLanes(traffic_light_id)
            
            for i, lane in enumerate(lanes[:8]):  # Limit to 8 lanes for our state space
                # Get vehicle count
                vehicles = traci.lane.getLastStepVehicleNumber(lane)
                # Get average speed
                mean_speed = traci.lane.getLastStepMeanSpeed(lane)
                # Get halting vehicles (queue)
                queue = traci.lane.getLastStepHaltingNumber(lane)
                
                state[i, 0] = vehicles
                state[i, 1] = mean_speed
                state[i, 2] = queue
        
        return state
    
    def _get_reward(self):
        """Calculate reward based on traffic conditions"""
        # Simple reward function based on reducing waiting time and queue length
        reward = 0
        
        if self.traffic_lights:
            traffic_light_id = self.traffic_lights[0]
            lanes = traci.trafficlight.getControlledLanes(traffic_light_id)
            
            total_waiting_time = 0
            total_queue = 0
            
            for lane in lanes:
                # Sum waiting time of all vehicles in the lane
                waiting_time = traci.lane.getWaitingTime(lane)
                # Count halting vehicles
                queue = traci.lane.getLastStepHaltingNumber(lane)
                
                total_waiting_time += waiting_time
                total_queue += queue
            
            # Negative reward for waiting time and queue length
            reward = -(total_waiting_time + total_queue)
        
        return reward
    
    def _update_metrics(self):
        """Update simulation metrics"""
        if not self.traffic_lights:
            return
            
        traffic_light_id = self.traffic_lights[0]
        lanes = traci.trafficlight.getControlledLanes(traffic_light_id)
        
        # Calculate metrics
        total_waiting_time = 0
        total_queue = 0
        throughput = 0
        
        for lane in lanes:
            total_waiting_time += traci.lane.getWaitingTime(lane)
            total_queue += traci.lane.getLastStepHaltingNumber(lane)
            throughput += traci.lane.getLastStepVehicleNumber(lane) - traci.lane.getLastStepHaltingNumber(lane)
        
        num_lanes = len(lanes)
        if num_lanes > 0:
            self.metrics['average_waiting_time'] = total_waiting_time / num_lanes
            self.metrics['queue_length'] = total_queue
            self.metrics['throughput'] = throughput
            
            # Simple congestion index: ratio of stopped vehicles to total vehicles
            total_vehicles = sum([traci.lane.getLastStepVehicleNumber(lane) for lane in lanes])
            if total_vehicles > 0:
                self.metrics['congestion_index'] = total_queue / total_vehicles
    
    def _close_traci(self):
        """Close TraCI connection if it exists"""
        if traci.isLoaded():
            traci.close()
    
    def close(self):
        """Close the environment"""
        self._close_traci()

    def render(self, mode='human'):
        """Render the environment"""
        # For TrafficSignalEnv, rendering is handled by SUMO-GUI if enabled
        pass

if __name__ == "__main__":
    # Simple test of the environment
    env = TrafficSignalEnv()
    state, _ = env.reset()
    
    for _ in range(10):
        action = env.action_space.sample()
        state, reward, done, _, info = env.step(action)
        print(f"Reward: {reward}, Metrics: {info['metrics']}")
        
        if done:
            break
    
    env.close()
