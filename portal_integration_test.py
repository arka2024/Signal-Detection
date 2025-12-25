"""
Portal Integration Test Script

This script demonstrates how the adaptive controller integrates with 
the portal interface, receiving vehicle data and making dynamic signal 
timing decisions based on traffic congestion.
"""

import os
import sys
import time
import json
import logging
import argparse
import numpy as np
from datetime import datetime
import torch

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import project modules
from src.adaptive_controller import AdaptiveSignalController
from src.vision import VehicleDetector

def simulate_portal_data(time_step, pattern="random"):
    """
    Simulate vehicle detection data coming from the portal
    
    Args:
        time_step: Current time step (used for patterns)
        pattern: Type of traffic pattern to simulate
            - "random": Random traffic levels
            - "rush_hour": Simulates morning/evening rush hour
            - "incident": Simulates traffic incident with sudden congestion
    
    Returns:
        Dictionary containing simulated portal data
    """
    # Base vehicle count
    if pattern == "random":
        vehicle_count = np.random.randint(5, 25)
    elif pattern == "rush_hour":
        # Simulate rush hour pattern with peak at time_step=30
        vehicle_count = 10 + int(15 * np.sin(np.pi * time_step / 60))
    elif pattern == "incident":
        # Simulate traffic incident around time_step=30
        if 25 <= time_step <= 35:
            vehicle_count = np.random.randint(20, 30)
        else:
            vehicle_count = np.random.randint(5, 15)
    else:
        vehicle_count = 10
        
    # Calculate congestion based on vehicle count
    congestion = min(1.0, vehicle_count / 30.0)
    
    # Create vehicle type distribution
    cars = int(vehicle_count * 0.8)
    buses = int(vehicle_count * 0.1)
    trucks = int(vehicle_count * 0.1)
    
    # Ensure we account for all vehicles after rounding
    remaining = vehicle_count - (cars + buses + trucks)
    cars += remaining
    
    # Create directional bias
    if pattern == "rush_hour":
        # Morning rush (into city) vs evening rush (out of city)
        current_hour = datetime.now().hour
        if 7 <= current_hour <= 10:  # Morning
            ns_ratio = 0.7
        elif 16 <= current_hour <= 19:  # Evening
            ns_ratio = 0.3
        else:
            ns_ratio = 0.5
    else:
        ns_ratio = np.random.uniform(0.3, 0.7)
    
    # Create portal data structure
    portal_data = {
        'vehicle_count': vehicle_count,
        'vehicle_types': {
            'cars': cars,
            'buses': buses,
            'trucks': trucks
        },
        'lane_occupancy': {
            'north': ns_ratio,
            'south': ns_ratio,
            'east': 1 - ns_ratio,
            'west': 1 - ns_ratio
        },
        'approach_congestion': {
            'north': congestion * ns_ratio,
            'south': congestion * ns_ratio,
            'east': congestion * (1 - ns_ratio),
            'west': congestion * (1 - ns_ratio)
        },
        'timestamp': datetime.now().isoformat()
    }
    
    return portal_data

def main():
    parser = argparse.ArgumentParser(description='Test portal integration with adaptive controller')
    parser.add_argument('--pattern', type=str, default='random', choices=['random', 'rush_hour', 'incident'],
                        help='Traffic pattern to simulate')
    parser.add_argument('--duration', type=int, default=60,
                        help='Duration of the simulation in seconds')
    parser.add_argument('--model', type=str, default='models/adaptive_controller.pth',
                        help='Path to the trained model')
    
    args = parser.parse_args()
    
    # Initialize controller
    controller = AdaptiveSignalController()
    
    # Load model if available
    if os.path.exists(args.model):
        try:
            controller.agent.qnetwork_local.load_state_dict(torch.load(args.model))
            logger.info(f"Loaded trained model from {args.model}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
    
    # Run the test
    logger.info(f"Starting portal integration test with {args.pattern} pattern")
    logger.info(f"Duration: {args.duration} seconds")
    
    start_time = time.time()
    time_step = 0
    
    try:
        while time.time() - start_time < args.duration:
            # Simulate portal data
            portal_data = simulate_portal_data(time_step, args.pattern)
            
            # Update controller with portal data
            controller.update_from_portal(portal_data)
            
            # Get phase decision
            phase, duration = controller.select_phase()
            
            # Display the decision
            logger.info(f"Time step {time_step}: Vehicles={portal_data['vehicle_count']}, " + 
                       f"Congestion={portal_data['approach_congestion']['north'] + portal_data['approach_congestion']['east']:.2f}, " +
                       f"Phase={phase}, Duration={duration}s")
            
            # Wait between updates (simulate real-time data)
            time.sleep(1)
            time_step += 1
            
    except KeyboardInterrupt:
        logger.info("Test stopped by user")
    
    logger.info("Test completed")

if __name__ == "__main__":
    main()
