"""
Simple test script for the Traffic Signal Management System
"""

import os
import sys
import numpy as np
import logging
import yaml
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try to import project modules
try:
    from src.environment import TrafficSignalEnv
    from src.agent import DQNAgent
    from src.vision import VehicleDetector
    modules_imported = True
except ImportError as e:
    logger.error(f"Error importing modules: {e}")
    modules_imported = False

def test_environment():
    """Test the traffic signal environment"""
    logger.info("Testing traffic signal environment...")
    
    # Create environment
    env = TrafficSignalEnv()
    
    # Reset environment
    logger.info("Resetting environment...")
    state, _ = env.reset()
    
    logger.info(f"State shape: {state.shape}")
    logger.info(f"Action space: {env.action_space}")
    
    # Take a few random actions
    logger.info("Taking random actions...")
    for i in range(5):
        action = env.action_space.sample()
        logger.info(f"Action {i+1}: {action}")
        
        state, reward, done, _, info = env.step(action)
        logger.info(f"Reward: {reward}, Done: {done}")
        logger.info(f"Metrics: {info['metrics']}")
        
        if done:
            logger.info("Episode ended")
            break
    
    # Close environment
    env.close()
    logger.info("Environment test completed")

def test_agent():
    """Test the DQN agent"""
    logger.info("Testing DQN agent...")
    
    # Create agent
    agent = DQNAgent()
    
    # Initialize with sample dimensions
    state_size = 24  # 8 lanes x 3 features
    action_size = 4  # 4 traffic light phases
    
    logger.info(f"Initializing agent with state_size={state_size}, action_size={action_size}")
    agent.initialize(state_size, action_size)
    
    # Test action selection
    test_state = np.random.rand(1, state_size)
    action = agent.act(test_state)
    logger.info(f"Selected action for random state: {action}")
    
    logger.info("Agent test completed")

def test_vision():
    """Test the vision module"""
    logger.info("Testing vision module...")
    
    # Create a test frame
    test_frame = np.ones((480, 640, 3), dtype=np.uint8) * 255
    
    # Create detector
    detector = VehicleDetector()
    
    # Process frame
    result = detector.process_frame(test_frame)
    
    logger.info(f"Detected {len(result['tracked_vehicles'])} vehicles")
    logger.info(f"Traffic density: {result['traffic_density']['density_level']}")
    
    logger.info("Vision test completed")

def main():
    """Main function"""
    logger.info("=" * 50)
    logger.info("Traffic Signal Management System - Simple Test")
    logger.info("=" * 50)
    
    if not modules_imported:
        logger.error("Required modules could not be imported.")
        logger.error("Please install dependencies and try again.")
        sys.exit(1)
    
    try:
        # Test environment
        test_environment()
        print()
        
        # Test agent
        test_agent()
        print()
        
        # Test vision
        test_vision()
    except Exception as e:
        logger.error(f"Error running tests: {e}")
    
    logger.info("=" * 50)
    logger.info("Test completed")

if __name__ == "__main__":
    main()
