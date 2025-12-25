"""
Hardware interface module for controlling actual traffic signals

This module defines the interface for connecting the adaptive traffic signal
controller to physical traffic signal hardware. Implement this interface
with the specific hardware communication protocol required for your deployment.
"""

import os
import sys
import time
import logging
import json
import requests
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SignalInterface:
    """Interface for communicating with physical traffic signal hardware"""
    
    def __init__(self, config_file=None):
        """Initialize the signal interface"""
        self.config = {}
        if config_file and os.path.exists(config_file):
            with open(config_file, 'r') as f:
                self.config = json.load(f)
        
        # Connection parameters
        self.api_endpoint = self.config.get('api_endpoint', 'http://localhost:8080/api/signal')
        self.api_key = self.config.get('api_key', '')
        self.intersection_id = self.config.get('intersection_id', 'intersection_001')
        
        # Connection status
        self.connected = False
        
        # Signal phases mapping
        self.phases = {
            0: "PHASE_NS_GREEN",
            1: "PHASE_NS_YELLOW",
            2: "PHASE_EW_GREEN",
            3: "PHASE_EW_YELLOW"
        }
        
        # Current signal state
        self.current_phase = None
        self.phase_start_time = None
        self.phase_duration = None
        
        # Initialize connection
        self._initialize_connection()
    
    def _initialize_connection(self):
        """Initialize connection to the signal hardware"""
        try:
            # Check if we're in demo/simulation mode
            if self.config.get('simulation_mode', True):
                logger.info("Running in simulation mode - no physical signals will be controlled")
                self.connected = True
                return
            
            # For real hardware, establish connection
            response = requests.get(
                f"{self.api_endpoint}/status",
                headers={"Authorization": f"Bearer {self.api_key}"},
                params={"intersection_id": self.intersection_id}
            )
            
            if response.status_code == 200:
                self.connected = True
                logger.info(f"Connected to traffic signal system at {self.api_endpoint}")
                
                # Get current phase
                status = response.json()
                self.current_phase = status.get('current_phase')
                self.phase_start_time = datetime.fromisoformat(status.get('phase_start_time'))
                self.phase_duration = status.get('phase_duration')
                
                logger.info(f"Current signal state: {self.phases.get(self.current_phase, 'UNKNOWN')}, "
                          f"Started at: {self.phase_start_time}, Duration: {self.phase_duration}s")
            else:
                logger.error(f"Failed to connect to signal system: {response.status_code} {response.text}")
                self.connected = False
        
        except Exception as e:
            logger.error(f"Error connecting to signal system: {e}")
            self.connected = False
    
    def set_phase(self, phase_index, duration, priority=0):
        """
        Set the traffic signal phase
        
        Args:
            phase_index (int): Index of the phase to set (0-3)
            duration (float): Duration of the phase in seconds
            priority (int): Priority level (0=normal, 1=priority, 2=emergency)
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.connected:
            logger.warning("Not connected to signal system")
            return False
        
        # If in simulation mode, just log the change
        if self.config.get('simulation_mode', True):
            logger.info(f"[SIMULATION] Setting signal phase to {self.phases.get(phase_index, 'UNKNOWN')}"
                      f" for {duration:.1f}s with priority {priority}")
            self.current_phase = phase_index
            self.phase_start_time = datetime.now()
            self.phase_duration = duration
            return True
        
        # For real hardware, send the command
        try:
            response = requests.post(
                f"{self.api_endpoint}/set_phase",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "intersection_id": self.intersection_id,
                    "phase": phase_index,
                    "duration": duration,
                    "priority": priority
                }
            )
            
            if response.status_code == 200:
                logger.info(f"Set signal phase to {self.phases.get(phase_index, 'UNKNOWN')}"
                          f" for {duration:.1f}s with priority {priority}")
                self.current_phase = phase_index
                self.phase_start_time = datetime.now()
                self.phase_duration = duration
                return True
            else:
                logger.error(f"Failed to set signal phase: {response.status_code} {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Error setting signal phase: {e}")
            return False
    
    def get_current_phase(self):
        """
        Get the current traffic signal phase
        
        Returns:
            tuple: (phase_index, elapsed_time, duration) or None if error
        """
        if not self.connected:
            logger.warning("Not connected to signal system")
            return None
        
        # If in simulation mode, return simulated state
        if self.config.get('simulation_mode', True):
            if self.current_phase is not None and self.phase_start_time is not None:
                elapsed = (datetime.now() - self.phase_start_time).total_seconds()
                return (self.current_phase, elapsed, self.phase_duration)
            return (0, 0, 30)  # Default values
        
        # For real hardware, query the state
        try:
            response = requests.get(
                f"{self.api_endpoint}/status",
                headers={"Authorization": f"Bearer {self.api_key}"},
                params={"intersection_id": self.intersection_id}
            )
            
            if response.status_code == 200:
                status = response.json()
                phase = status.get('current_phase')
                start_time = datetime.fromisoformat(status.get('phase_start_time'))
                duration = status.get('phase_duration')
                
                elapsed = (datetime.now() - start_time).total_seconds()
                
                # Update cached state
                self.current_phase = phase
                self.phase_start_time = start_time
                self.phase_duration = duration
                
                return (phase, elapsed, duration)
            else:
                logger.error(f"Failed to get signal status: {response.status_code} {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Error getting signal status: {e}")
            return None
    
    def emergency_override(self, direction="north_south", duration=30):
        """
        Trigger emergency override mode for emergency vehicles
        
        Args:
            direction (str): Direction of emergency vehicle approach
            duration (float): Duration of override in seconds
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.connected:
            logger.warning("Not connected to signal system")
            return False
        
        # Map direction to phase
        if direction in ["north", "south", "north_south"]:
            phase = 0  # North-South green
        else:
            phase = 2  # East-West green
        
        # Set phase with emergency priority
        return self.set_phase(phase, duration, priority=2)
    
    def manual_override(self, phase_index, duration):
        """
        Trigger manual override mode
        
        Args:
            phase_index (int): Index of the phase to set
            duration (float): Duration of override in seconds
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.connected:
            logger.warning("Not connected to signal system")
            return False
        
        # Set phase with priority level 1 (overrides normal but not emergency)
        return self.set_phase(phase_index, duration, priority=1)
    
    def release_override(self):
        """
        Release any active overrides and return to normal operation
        
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.connected:
            logger.warning("Not connected to signal system")
            return False
        
        # If in simulation mode, just log the change
        if self.config.get('simulation_mode', True):
            logger.info("[SIMULATION] Released signal override, returning to normal operation")
            return True
        
        # For real hardware, send the command
        try:
            response = requests.post(
                f"{self.api_endpoint}/release_override",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={"intersection_id": self.intersection_id}
            )
            
            if response.status_code == 200:
                logger.info("Released signal override, returning to normal operation")
                return True
            else:
                logger.error(f"Failed to release override: {response.status_code} {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Error releasing override: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from the signal system"""
        if not self.connected:
            return
        
        # If in simulation mode, just log
        if self.config.get('simulation_mode', True):
            logger.info("[SIMULATION] Disconnected from signal system")
            self.connected = False
            return
        
        # For real hardware, properly disconnect
        try:
            # Some signal systems might need a formal disconnect
            # Here we just log it
            logger.info("Disconnected from signal system")
            self.connected = False
            
        except Exception as e:
            logger.error(f"Error disconnecting from signal system: {e}")

# Example usage
if __name__ == "__main__":
    # Create config file
    config = {
        "simulation_mode": True,  # Set to False for real hardware
        "api_endpoint": "http://localhost:8080/api/signal",
        "api_key": "test_key",
        "intersection_id": "intersection_001"
    }
    
    # Save config for testing
    with open("signal_config.json", "w") as f:
        json.dump(config, f, indent=4)
    
    # Create signal interface
    interface = SignalInterface("signal_config.json")
    
    # Test sequence
    if interface.connected:
        print("Connected to signal system")
        
        # Set North-South green
        interface.set_phase(0, 30)
        print("Set North-South green for 30s")
        
        # Wait a bit
        time.sleep(2)
        
        # Get current phase
        status = interface.get_current_phase()
        if status:
            phase, elapsed, duration = status
            print(f"Current phase: {phase}, Elapsed: {elapsed:.1f}s, Duration: {duration}s")
        
        # Simulate emergency
        interface.emergency_override("east_west", 20)
        print("Emergency override: East-West direction for 20s")
        
        # Wait a bit
        time.sleep(2)
        
        # Release override
        interface.release_override()
        print("Released override")
        
        # Disconnect
        interface.disconnect()
    else:
        print("Failed to connect to signal system")
