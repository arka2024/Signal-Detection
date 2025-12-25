"""
Adaptive Traffic Signal Controller

This module implements an intelligent traffic signal controller that uses
reinforcement learning and real-time traffic metrics to optimize signal timing.
The controller integrates analysis results and vision data to make dynamic signal
timing decisions that respond to changing traffic conditions.
"""

import os
import sys
import numpy as np
import json
import time
import logging
import yaml
from datetime import datetime, timedelta
import torch
from collections import defaultdict, deque

# Import project modules
from src.agent import DQNAgent
from src.vision import VehicleDetector

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AdaptiveSignalController:
    """
    Intelligent traffic signal controller that uses AI to optimize traffic flow.
    Combines reinforcement learning with traffic analysis metrics to make decisions.
    """
    
    def __init__(self, config_file="config.yaml"):
        """Initialize the adaptive signal controller"""
        # Load configuration
        with open(config_file, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Controller settings
        controller_config = self.config.get('controller', {})
        self.update_frequency = controller_config.get('update_frequency', 5)  # seconds
        self.analysis_weight = controller_config.get('analysis_weight', 0.3)
        self.rl_weight = controller_config.get('rl_weight', 0.7)
        self.prediction_horizon = controller_config.get('prediction_horizon', 15)  # minutes
        
        # Phase definitions - mapping from phase index to description
        self.phases = {
            0: "North-South Green",
            1: "North-South Yellow",
            2: "East-West Green",
            3: "East-West Yellow"
        }
        
        # Phase durations (in seconds)
        self.min_green_time = controller_config.get('min_green_time', 10)
        self.max_green_time = controller_config.get('max_green_time', 60)
        self.yellow_time = controller_config.get('yellow_time', 3)
        
        # Initialize reinforcement learning agent
        self.agent = DQNAgent(config_file)
        
        # State space: traffic density, queue length, waiting time per lane
        # 8 lanes (2 per approach) x 3 features per lane
        self.state_size = 24
        
        # Action space: 4 phases for a typical 4-way intersection
        self.action_size = 4
        
        # Initialize the RL agent
        self.agent.initialize(self.state_size, self.action_size)
        
        # Initialize vehicle detector for vision-based detection
        self.detector = VehicleDetector(config_file)
        
        # Store current traffic state and metrics
        self.current_state = np.zeros(self.state_size)
        self.current_phase = 0
        self.phase_duration = 0
        self.metrics_history = {
            'waiting_time': deque(maxlen=100),
            'queue_length': deque(maxlen=100),
            'throughput': deque(maxlen=100),
            'congestion': deque(maxlen=100)
        }
        
        # Traffic patterns - initialize from analysis results if available
        self.traffic_patterns = self._load_traffic_patterns()
        
        # Rules based on traffic analysis
        self.rules = self._load_analysis_rules()
        
        # Current controller mode
        self.mode = "hybrid"  # Options: "rl", "rules", "hybrid"
        
        # Performance metrics
        self.performance = {
            'avg_waiting_time': 0,
            'avg_queue_length': 0,
            'avg_throughput': 0,
            'avg_congestion': 0,
            'decisions_made': 0
        }
    
    def _load_traffic_patterns(self):
        """Load traffic patterns from analysis data if available"""
        patterns = {}
        
        try:
            analysis_file = os.path.join("analysis_results", "analysis_results.json")
            if os.path.exists(analysis_file):
                with open(analysis_file, 'r') as f:
                    analysis_data = json.load(f)
                
                # Extract peak hours for each intersection
                for intersection, metrics in analysis_data.get('metrics', {}).items():
                    patterns[intersection] = {
                        'peak_hours': metrics.get('peak_hours', []),
                        'direction_ratio': metrics.get('direction_ratio', 1.0),
                        'ns_traffic': metrics.get('ns_traffic', 0),
                        'ew_traffic': metrics.get('ew_traffic', 0)
                    }
                
                logger.info(f"Loaded traffic patterns for {len(patterns)} intersections")
            else:
                logger.warning("No analysis results found. Using default traffic patterns.")
        except Exception as e:
            logger.error(f"Error loading traffic patterns: {e}")
        
        # Default pattern if none loaded
        if not patterns:
            patterns["default"] = {
                'peak_hours': [8, 17, 12],
                'direction_ratio': 1.0,
                'ns_traffic': 10,
                'ew_traffic': 10
            }
        
        return patterns
    
    def _load_analysis_rules(self):
        """Load rules derived from traffic analysis"""
        rules = []
        
        try:
            analysis_file = os.path.join("analysis_results", "analysis_results.json")
            if os.path.exists(analysis_file):
                with open(analysis_file, 'r') as f:
                    analysis_data = json.load(f)
                
                # Extract recommendations for each intersection
                for intersection, recommendations in analysis_data.get('recommendations', {}).items():
                    for rec in recommendations:
                        # Parse recommendations into rules
                        if "favor" in rec.lower() and "direction" in rec.lower():
                            if "north-south" in rec.lower():
                                rules.append({
                                    "condition": "direction_imbalance",
                                    "direction": "north-south",
                                    "intersection": intersection,
                                    "action": "extend_green_time",
                                    "factor": 1.5
                                })
                            elif "east-west" in rec.lower():
                                rules.append({
                                    "condition": "direction_imbalance",
                                    "direction": "east-west",
                                    "intersection": intersection,
                                    "action": "extend_green_time",
                                    "factor": 1.5
                                })
                        
                        if "peak" in rec.lower() and "hour" in rec.lower():
                            hours = [int(h) for h in rec.split("at ")[1].split(":")[0].split(", ") if h.isdigit()]
                            rules.append({
                                "condition": "peak_hour",
                                "hours": hours,
                                "intersection": intersection,
                                "action": "adaptive_timing",
                                "factor": 0.7  # reduce cycle length by 30%
                            })
                
                logger.info(f"Loaded {len(rules)} traffic rules from analysis")
            else:
                logger.warning("No analysis results found. Using default rules.")
        except Exception as e:
            logger.error(f"Error loading traffic rules: {e}")
        
        # Default rules if none loaded
        if not rules:
            rules = [
                {
                    "condition": "peak_hour",
                    "hours": [8, 17],
                    "action": "adaptive_timing",
                    "factor": 0.7
                },
                {
                    "condition": "high_congestion",
                    "threshold": 0.7,
                    "action": "extend_green_time",
                    "factor": 1.3
                }
            ]
        
        return rules
    
    def update_state_from_vision(self, frame):
        """
        Update traffic state using computer vision.
        
        Args:
            frame: Camera frame to process
        """
        # Process the frame using vehicle detector
        detection_result = self.detector.process_frame(frame)
        
        # Extract traffic data
        tracked_vehicles = detection_result['tracked_vehicles']
        traffic_density = detection_result['traffic_density']
        
        # Update state vector
        # For simplicity, let's assume 4 approaches with 2 lanes each
        directions = ["north", "south", "east", "west"]
        lanes_per_direction = 2
        
        # Reset state vector
        new_state = np.zeros(self.state_size)
        
        # Calculate vehicles per direction
        vehicles_by_direction = defaultdict(int)
        for vehicle in tracked_vehicles.values():
            # In a real system, this would classify vehicles by direction
            # Here we'll just randomly assign to directions
            direction = np.random.choice(directions)
            vehicles_by_direction[direction] += 1
        
        # Update state vector with density information
        # Structure: [n_vehicles, avg_speed, queue] per lane
        idx = 0
        for direction in directions:
            for lane in range(lanes_per_direction):
                # Vehicle count (normalized by max expected)
                vehicles = vehicles_by_direction[direction] / lanes_per_direction
                new_state[idx] = min(1.0, vehicles / 20.0)  # Normalize
                
                # Average speed (normalized)
                # Assume speed is inversely related to vehicle count
                speed = max(5.0, 55.0 - (vehicles * 2))
                new_state[idx + 1] = speed / 55.0  # Normalize
                
                # Queue length (normalized)
                queue = vehicles * 0.4  # Simple estimation
                new_state[idx + 2] = min(1.0, queue / 10.0)  # Normalize
                
                idx += 3
        
        # Update current state
        self.current_state = new_state
    
    def update_state_from_sensors(self, sensor_data):
        """
        Update traffic state using sensor data.
        
        Args:
            sensor_data: Dictionary with sensor readings
                - vehicle_counts: Vehicle counts per lane
                - speeds: Average speeds per lane
                - queues: Queue lengths per lane
        """
        # Extract data from sensors
        vehicle_counts = sensor_data.get('vehicle_counts', [])
        speeds = sensor_data.get('speeds', [])
        queues = sensor_data.get('queues', [])
        
        # Ensure data arrays have the right length
        vehicle_counts = vehicle_counts[:8]
        speeds = speeds[:8]
        queues = queues[:8]
        
        # Pad arrays if needed
        vehicle_counts = vehicle_counts + [0] * (8 - len(vehicle_counts))
        speeds = speeds + [0] * (8 - len(speeds))
        queues = queues + [0] * (8 - len(queues))
        
        # Update state vector
        new_state = np.zeros(self.state_size)
        for i in range(8):
            new_state[i*3] = min(1.0, vehicle_counts[i] / 20.0)
            new_state[i*3 + 1] = speeds[i] / 55.0
            new_state[i*3 + 2] = min(1.0, queues[i] / 10.0)
        
        # Update current state
        self.current_state = new_state
    
    def select_phase(self, intersection_id="traffic_light_1"):
        """
        Select the next traffic signal phase based on current state.
        
        Args:
            intersection_id: ID of the intersection to control
            
        Returns:
            phase_index: Index of the selected phase
            duration: Suggested duration for the phase in seconds
        """
        # Get current time
        current_time = datetime.now()
        current_hour = current_time.hour
        
        # Get current congestion level for adaptive timing
        congestion_level = self._calculate_congestion_level()
        
        # Calculate dynamic min and max green times based on congestion
        # For high congestion: shorter max times to cycle through phases more quickly
        # For low congestion: longer green times for the dominant flow direction
        if congestion_level > 0.7:  # High congestion
            # In high congestion, we want to cycle through phases more quickly
            # This helps clear queues in all directions more frequently
            dynamic_min_green = max(5, self.min_green_time * 0.7)
            dynamic_max_green = max(15, self.max_green_time * 0.5)
            logger.info(f"High congestion mode ({congestion_level:.2f}): faster phase cycling")
        elif congestion_level > 0.4:  # Medium congestion
            # Balanced approach for medium congestion
            dynamic_min_green = self.min_green_time
            dynamic_max_green = max(20, self.max_green_time * 0.7)
            logger.info(f"Medium congestion mode ({congestion_level:.2f}): balanced phases")
        else:  # Low congestion
            # For low congestion, allow longer green times
            dynamic_min_green = max(8, self.min_green_time * 1.2)
            dynamic_max_green = self.max_green_time
            logger.info(f"Low congestion mode ({congestion_level:.2f}): extended green phases")
        
        # Default duration starts at dynamic minimum
        duration = dynamic_min_green
        
        if self.mode == "rl":
            # Pure RL approach
            phase_index = self.agent.act(self.current_state)
            
            # Calculate duration based on Q-values and congestion
            q_values = self.agent.qnetwork_local(torch.FloatTensor(self.current_state)).detach().numpy()
            confidence = (q_values[phase_index] - np.min(q_values)) / (np.max(q_values) - np.min(q_values) + 1e-6)
            
            # Higher confidence gets more time, but congestion reduces maximum time available
            duration = dynamic_min_green + confidence * (dynamic_max_green - dynamic_min_green)
        
        elif self.mode == "rules":
            # Pure rule-based approach
            phase_index, duration = self._apply_rules(intersection_id, current_hour)
        
        else:  # hybrid mode
            # Get RL action
            rl_phase = self.agent.act(self.current_state)
            
            # Get rule-based action
            rule_phase, rule_duration = self._apply_rules(intersection_id, current_hour)
            
            # Combine decisions
            if self.current_phase == 0 or self.current_phase == 2:  # If currently in a green phase
                # Check if we need to change or extend
                if self.phase_duration >= dynamic_max_green:
                    # Force phase change due to maximum time reached
                    phase_index = (self.current_phase + 1) % 4
                    duration = self.yellow_time if phase_index % 2 == 1 else dynamic_min_green
                else:
                    # Decide whether to extend or change based on weighted decision
                    # In high congestion, give more weight to rule-based decisions
                    adaptive_weight = self.rl_weight * (1 - congestion_level * 0.5)
                    if np.random.random() < adaptive_weight:
                        phase_index = rl_phase
                    else:
                        phase_index = rule_phase
                    
                    if phase_index == self.current_phase:
                        # Extending current green - calculate duration
                        # In higher congestion, extend for shorter periods
                        extension_factor = 0.5 * (1 - congestion_level * 0.5)
                        duration = min(
                            dynamic_max_green - self.phase_duration,
                            rule_duration * extension_factor
                        )
                    else:
                        # Changing phase - first go to yellow
                        phase_index = self.current_phase + 1
                        duration = self.yellow_time
            
            else:  # Currently in a yellow phase
                # Must go to the next green phase
                phase_index = (self.current_phase + 1) % 4
                
                # Determine appropriate green time based on traffic conditions for the next phase
                if phase_index == 0:  # North-South Green
                    lanes = np.reshape(self.current_state, (8, 3))
                    ns_density = np.mean(lanes[0:4, 0])
                    duration = dynamic_min_green + ns_density * (1 - congestion_level) * 5
                elif phase_index == 2:  # East-West Green
                    lanes = np.reshape(self.current_state, (8, 3))
                    ew_density = np.mean(lanes[4:8, 0])
                    duration = dynamic_min_green + ew_density * (1 - congestion_level) * 5
        
        # Ensure duration is within bounds - using dynamic bounds based on congestion
        duration = max(dynamic_min_green, min(dynamic_max_green, duration))
        
        # Yellow phases should always have fixed duration regardless of congestion
        if phase_index % 2 == 1:  # Yellow phase
            duration = self.yellow_time
        
        # Round duration to nearest second
        duration = round(duration)
        
        # Log the decision with congestion level for debugging
        logger.info(f"Phase selected: {phase_index} with duration {duration}s (congestion: {congestion_level:.2f})")
        
        # Update controller state
        self.current_phase = phase_index
        self.phase_duration = duration
        
        # Track decision for performance monitoring
        self.performance['decisions_made'] += 1
        
        return phase_index, duration
    
    def _apply_rules(self, intersection_id, current_hour):
        """Apply rule-based logic to select phase and duration"""
        # Get traffic patterns for the intersection
        patterns = self.traffic_patterns.get(intersection_id, self.traffic_patterns.get("default", {}))
        
        # Initialize with default values
        phase_index = 0  # North-South Green
        
        # Dynamic minimum green time based on congestion level
        congestion_level = self._calculate_congestion_level()
        
        # Adjust minimum green time based on congestion level
        dynamic_min_green = max(5, self.min_green_time - (congestion_level * 5))
        dynamic_max_green = max(15, self.max_green_time - (congestion_level * 15))
        
        # Start with dynamic minimum
        duration = dynamic_min_green
        
        # Check direction imbalance
        direction_ratio = patterns.get('direction_ratio', 1.0)
        
        # Check if current hour is a peak hour
        is_peak_hour = current_hour in patterns.get('peak_hours', [])
        
        # Extract state information
        lanes = np.reshape(self.current_state, (8, 3))
        
        # Calculate average metrics per direction
        ns_vehicles = np.mean(lanes[0:4, 0])
        ew_vehicles = np.mean(lanes[4:8, 0])
        
        ns_queue = np.mean(lanes[0:4, 2])
        ew_queue = np.mean(lanes[4:8, 2])
        
        # Decision logic
        if ns_queue > ew_queue * 1.5:
            # North-South has significantly more queue
            phase_index = 0  # North-South Green
            # Longer duration for higher queue ratio, but inversely affected by congestion
            queue_ratio = min(5.0, ns_queue / (ew_queue + 0.1))
            duration = dynamic_min_green + (queue_ratio * (15 - congestion_level * 3))
        elif ew_queue > ns_queue * 1.5:
            # East-West has significantly more queue
            phase_index = 2  # East-West Green
            # Longer duration for higher queue ratio, but inversely affected by congestion
            queue_ratio = min(5.0, ew_queue / (ns_queue + 0.1))
            duration = dynamic_min_green + (queue_ratio * (15 - congestion_level * 3))
        else:
            # No significant imbalance, choose based on vehicle count
            if ns_vehicles >= ew_vehicles:
                phase_index = 0
            else:
                phase_index = 2
            
            # Base duration on overall traffic volume, inversely affected by congestion
            avg_volume = (ns_vehicles + ew_vehicles) / 2
            duration = dynamic_min_green + min(20, avg_volume * (8 - congestion_level * 2))
        
        # Apply rules from analysis
        for rule in self.rules:
            if rule["condition"] == "direction_imbalance":
                if rule["direction"] == "north-south" and phase_index == 0:
                    duration *= rule["factor"]
                elif rule["direction"] == "east-west" and phase_index == 2:
                    duration *= rule["factor"]
            
            elif rule["condition"] == "peak_hour" and is_peak_hour:
                # During peak hours, we want shorter cycles but still effective
                if congestion_level > 0.7:
                    # Very congested during peak hour - even shorter cycles
                    duration *= min(rule["factor"], 0.7)
                else:
                    duration *= rule["factor"]  # Normal peak hour adjustment
        
        # Ensure duration is within bounds - using dynamic bounds based on congestion
        duration = max(dynamic_min_green, min(dynamic_max_green, duration))
        
        # Round to nearest second
        duration = round(duration)
        
        return phase_index, duration
        
    def _calculate_congestion_level(self):
        """
        Calculate overall congestion level from current state and vehicle detection data
        Returns value between 0 (no congestion) and 1 (maximum congestion)
        
        This method integrates with the portal's vehicle detection data when available.
        """
        # Check if we have portal vehicle count data available
        if hasattr(self, 'portal_data') and 'vehicle_count' in self.portal_data:
            # Extract vehicle counts from portal
            vehicle_count = self.portal_data['vehicle_count']
            # Normalize vehicle count to congestion level (0-1)
            # Adjust these thresholds based on your specific intersection capacity
            max_capacity = 30  # Maximum expected number of vehicles for highest congestion
            congestion = min(1.0, vehicle_count / max_capacity)
            
            # Store this in performance metrics for historical tracking
            if hasattr(self, 'performance'):
                # Update with exponential moving average for smoothing
                if 'congestion' in self.performance:
                    alpha = 0.3  # Smoothing factor
                    self.performance['congestion'] = (1-alpha) * self.performance['congestion'] + alpha * congestion
                else:
                    self.performance['congestion'] = congestion
                    
            logger.info(f"Calculated congestion level from portal data: {congestion:.2f} (vehicle count: {vehicle_count})")
            return congestion
            
        # Fallback to existing methods if portal data not available
        if hasattr(self, 'performance') and 'congestion' in self.performance:
            return min(1.0, self.performance.get('congestion', 0.5))
            
        # Fallback: calculate from current state
        lanes = np.reshape(self.current_state, (8, 3))
        
        # Average vehicle density and queue length across all lanes
        avg_density = np.mean(lanes[:, 0])
        avg_queue = np.mean(lanes[:, 2])
        
        # Normalize to 0-1 range (assuming reasonable max values)
        norm_density = min(1.0, avg_density / 15.0)
        norm_queue = min(1.0, avg_queue / 10.0)
        
        # Combined congestion metric (weighted average)
        congestion = 0.6 * norm_density + 0.4 * norm_queue
        
        return congestion
    
    def update_metrics(self, metrics):
        """
        Update performance metrics.
        
        Args:
            metrics: Dictionary with current metrics
                - waiting_time: Average waiting time
                - queue_length: Total queue length
                - throughput: Vehicle throughput
                - congestion: Congestion index
        """
        # Add metrics to history
        self.metrics_history['waiting_time'].append(metrics.get('waiting_time', 0))
        self.metrics_history['queue_length'].append(metrics.get('queue_length', 0))
        self.metrics_history['throughput'].append(metrics.get('throughput', 0))
        self.metrics_history['congestion'].append(metrics.get('congestion', 0))
        
        # Update rolling averages
        self.performance['avg_waiting_time'] = np.mean(self.metrics_history['waiting_time'])
        self.performance['avg_queue_length'] = np.mean(self.metrics_history['queue_length'])
        self.performance['avg_throughput'] = np.mean(self.metrics_history['throughput'])
        
        # Store congestion level for use in decision making
        self.performance['congestion'] = metrics.get('congestion', 0.5)
        
    def update_from_portal(self, portal_data):
        """
        Update controller with data from traffic monitoring portal
        
        Args:
            portal_data: Dictionary containing portal metrics such as:
                - vehicle_count: Total number of vehicles detected
                - vehicle_types: Dictionary of counts by vehicle type (cars, buses, trucks)
                - lane_occupancy: Dictionary of occupancy ratios by lane
                - approach_congestion: Dictionary of congestion levels by approach
        """
        # Store portal data for use in congestion calculation
        self.portal_data = portal_data
        
        # Convert portal data to our metrics format
        metrics = {}
        
        # Calculate waiting time based on vehicle count and congestion
        if 'vehicle_count' in portal_data:
            # Calculate congestion from vehicle count
            congestion = min(1.0, portal_data['vehicle_count'] / 30.0)  # Normalize by max expected vehicles
            metrics['congestion'] = congestion
            
            # Estimate waiting time based on congestion
            metrics['waiting_time'] = 5.0 + (congestion * 55.0)  # Scale from 5s to 60s based on congestion
            
            # Estimate queue length based on vehicle count
            metrics['queue_length'] = portal_data['vehicle_count'] * 0.7  # Assume 70% of vehicles are in queue
            
            # Estimate throughput
            metrics['throughput'] = portal_data['vehicle_count'] * (1.0 - congestion * 0.5)
        
        # Log portal update
        logger.info(f"Updated from portal: Vehicles: {portal_data.get('vehicle_count', 'N/A')}, "
                   f"Congestion: {metrics.get('congestion', 'N/A'):.2f}")
        
        # Update metrics
        self.update_metrics(metrics)
        
        # Save data to database if connection exists
        self.save_to_database(portal_data)
        
    def save_to_database(self, data):
        """Save traffic data to database for historical analysis"""
        # Implementation depends on your database setup
        # This is a placeholder for database storage functionality
        try:
            # Example database logging code (commented out)
            # db_conn = get_database_connection()
            # timestamp = datetime.now().isoformat()
            # data_with_timestamp = {**data, "timestamp": timestamp}
            # db_conn.insert("traffic_data", data_with_timestamp)
            # db_conn.close()
            pass
        except Exception as e:
            logger.error(f"Failed to save to database: {e}")
            pass
        self.performance['avg_congestion'] = np.mean(self.metrics_history['congestion'])
    
    def learn_from_experience(self, state, action, reward, next_state, done=False):
        """
        Update the RL agent based on experience.
        
        Args:
            state: Previous state
            action: Action taken
            reward: Reward received
            next_state: New state
            done: Whether this is a terminal state
        """
        # Update the RL agent
        self.agent.step(state, action, reward, next_state, done)
    
    def calculate_reward(self, metrics):
        """
        Calculate reward based on traffic metrics.
        
        Args:
            metrics: Dictionary with current metrics
                - waiting_time: Average waiting time
                - queue_length: Total queue length
                - throughput: Vehicle throughput
                - congestion: Congestion index
                
        Returns:
            reward: Calculated reward value
        """
        # Get current metrics
        waiting_time = metrics.get('waiting_time', 0)
        queue_length = metrics.get('queue_length', 0)
        throughput = metrics.get('throughput', 0)
        congestion = metrics.get('congestion', 0)
        
        # Get previous metrics
        prev_waiting_time = self.performance.get('avg_waiting_time', 0)
        prev_queue_length = self.performance.get('avg_queue_length', 0)
        prev_congestion = self.performance.get('avg_congestion', 0)
        
        # Calculate changes
        delta_waiting = prev_waiting_time - waiting_time
        delta_queue = prev_queue_length - queue_length
        delta_congestion = prev_congestion - congestion
        
        # Calculate reward components
        waiting_reward = delta_waiting * 0.1  # Positive reward for reduced waiting time
        queue_reward = delta_queue * 0.2  # Positive reward for reduced queue length
        throughput_reward = throughput * 0.05  # Positive reward for throughput
        congestion_reward = delta_congestion * 15  # Positive reward for reduced congestion
        
        # Calculate total reward
        reward = waiting_reward + queue_reward + throughput_reward + congestion_reward
        
        return reward
    
    def save_model(self, path="models/adaptive_controller.pth"):
        """Save the trained model"""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save the agent's model
        self.agent.save(path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path="models/adaptive_controller.pth"):
        """Load a trained model"""
        if os.path.exists(path):
            self.agent.load(path)
            logger.info(f"Model loaded from {path}")
        else:
            logger.warning(f"Model file not found at {path}")
    
    def get_performance_stats(self):
        """Get performance statistics of the controller"""
        return self.performance

if __name__ == "__main__":
    # Simple test of the controller
    controller = AdaptiveSignalController()
    
    # Simulate traffic state
    state = np.random.rand(24)
    controller.current_state = state
    
    # Select phase
    phase, duration = controller.select_phase()
    
    print(f"Selected phase: {controller.phases[phase]}, Duration: {duration}s")
    
    # Simulate metrics update
    metrics = {
        'waiting_time': np.random.uniform(0, 30),
        'queue_length': np.random.randint(0, 20),
        'throughput': np.random.randint(5, 15),
        'congestion': np.random.uniform(0, 1)
    }
    
    controller.update_metrics(metrics)
    
    # Calculate reward
    reward = controller.calculate_reward(metrics)
    
    print(f"Reward: {reward}")
    print(f"Performance stats: {controller.get_performance_stats()}")
