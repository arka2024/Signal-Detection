"""
Traffic Signal Visualization using Matplotlib (No Pygame)

This script creates a visual demonstration of the adaptive traffic signal controller in action,
using only matplotlib for visualization to avoid pygame dependency issues.
"""

import os
import sys
import time
import json
import argparse
import numpy as np
import cv2
from datetime import datetime, timedelta
import matplotlib
matplotlib.use("TkAgg")  # Use TkAgg backend
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import logging

# Import project modules
try:
    from src.adaptive_controller import AdaptiveSignalController
    from src.vision import VehicleDetector
except ImportError as e:
    print(f"Error importing project modules: {e}")
    print("Make sure you're running this script from the project root directory.")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TrafficSignalVisualization:
    """Traffic signal visualization system that integrates with the adaptive controller"""
    
    def __init__(self, video_source=None, use_analysis=True, model_path=None, config_file="config.yaml"):
        """Initialize the visualization system"""
        # Initialize controller and detector
        self.controller = AdaptiveSignalController(config_file=config_file)
        if model_path and os.path.exists(model_path):
            self.controller.load_model(model_path)
            logger.info(f"Loaded model from {model_path}")
        
        # Set controller mode based on analysis flag
        if use_analysis:
            self.controller.mode = "hybrid"
            logger.info("Using hybrid mode (RL + analysis rules)")
        else:
            self.controller.mode = "rl"
            logger.info("Using pure RL mode")
        
        # Initialize vehicle detector
        self.detector = VehicleDetector(config_file=config_file)
        
        # Initialize video source
        self.cap = None
        if video_source:
            if video_source.isdigit():
                self.cap = cv2.VideoCapture(int(video_source))
                logger.info(f"Using camera device: {video_source}")
            elif os.path.exists(video_source):
                self.cap = cv2.VideoCapture(video_source)
                logger.info(f"Using video file: {video_source}")
        
        # If no video source or failed to open, we'll use simulated traffic
        if not self.cap or not self.cap.isOpened():
            logger.info("No valid video source. Using simulated traffic patterns.")
            self.cap = None
        
        # Traffic light state
        self.current_phase = 0
        self.phase_start_time = time.time()
        self.phase_duration = 30  # Default duration in seconds
        
        # Performance metrics history
        self.metrics_history = {
            'waiting_time': [],
            'queue_length': [],
            'throughput': [],
            'congestion': [],
            'reward': []
        }
        self.history_length = 60  # Keep 60 data points for plotting
        
        # Decision metrics
        self.decisions = {
            'rl': 0,
            'rules': 0,
            'hybrid': 0
        }
        
        # Current frame and detection results
        self.current_frame = None
        self.detection_results = None
        
        # Current metrics
        self.current_metrics = {
            'waiting_time': 0,
            'queue_length': 0,
            'throughput': 0,
            'congestion': 0,
            'reward': 0
        }
        
        # Traffic signal phases
        self.phases = {
            0: "North-South Green",
            1: "North-South Yellow",
            2: "East-West Green",
            3: "East-West Yellow"
        }
        
        # Recording settings
        self.recording = False
        self.record_frames = []
        self.max_record_frames = 1800  # 1 minute at 30fps
        
        # Create figure and axes for plotting
        self.fig = plt.figure(figsize=(15, 8))
        self.setup_plot()
        
        logger.info("Visualization system initialized")
    
    def setup_plot(self):
        """Set up the matplotlib plot layout"""
        # Main traffic view
        self.ax_traffic = self.fig.add_subplot(2, 2, 1)
        self.ax_traffic.set_title("Traffic Camera View")
        self.ax_traffic.axis('off')
        
        # Traffic light status
        self.ax_signal = self.fig.add_subplot(2, 2, 2)
        self.ax_signal.set_title("Traffic Signal Status")
        self.ax_signal.axis('off')
        
        # Metrics plot
        self.ax_metrics = self.fig.add_subplot(2, 2, 3)
        self.ax_metrics.set_title("Traffic Metrics")
        self.ax_metrics.set_xlabel("Time Steps")
        self.ax_metrics.set_ylabel("Normalized Value")
        self.ax_metrics.grid(True)
        
        # Decision pie chart
        self.ax_pie = self.fig.add_subplot(2, 2, 4)
        self.ax_pie.set_title("Decision Sources")
        
        # Adjust layout
        self.fig.tight_layout()
        plt.subplots_adjust(hspace=0.3, wspace=0.3)
    
    def get_frame(self):
        """Get current frame from video source or generate a simulated frame"""
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                # Loop video if at end
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.cap.read()
                if not ret:
                    return self.generate_simulated_frame()
            return frame
        else:
            return self.generate_simulated_frame()
    
    def generate_simulated_frame(self):
        """Generate a simulated traffic frame"""
        # Create a base frame
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 255
        
        # Draw a road intersection
        cv2.rectangle(frame, (0, 220), (640, 260), (100, 100, 100), -1)  # Horizontal road
        cv2.rectangle(frame, (300, 0), (340, 480), (100, 100, 100), -1)  # Vertical road
        
        # Draw road markings
        for x in range(0, 640, 30):
            if x < 280 or x > 360:
                cv2.line(frame, (x, 240), (x+15, 240), (255, 255, 255), 2)
        
        for y in range(0, 480, 30):
            if y < 200 or y > 280:
                cv2.line(frame, (320, y), (320, y+15), (255, 255, 255), 2)
        
        # Draw traffic light
        if self.current_phase == 0:  # North-South Green
            cv2.circle(frame, (350, 190), 10, (0, 255, 0), -1)
            cv2.circle(frame, (280, 290), 10, (0, 255, 0), -1)
            cv2.circle(frame, (350, 290), 10, (255, 0, 0), -1)
            cv2.circle(frame, (280, 190), 10, (255, 0, 0), -1)
        elif self.current_phase == 1:  # North-South Yellow
            cv2.circle(frame, (350, 190), 10, (255, 255, 0), -1)
            cv2.circle(frame, (280, 290), 10, (255, 255, 0), -1)
            cv2.circle(frame, (350, 290), 10, (255, 0, 0), -1)
            cv2.circle(frame, (280, 190), 10, (255, 0, 0), -1)
        elif self.current_phase == 2:  # East-West Green
            cv2.circle(frame, (350, 190), 10, (255, 0, 0), -1)
            cv2.circle(frame, (280, 290), 10, (255, 0, 0), -1)
            cv2.circle(frame, (350, 290), 10, (0, 255, 0), -1)
            cv2.circle(frame, (280, 190), 10, (0, 255, 0), -1)
        elif self.current_phase == 3:  # East-West Yellow
            cv2.circle(frame, (350, 190), 10, (255, 0, 0), -1)
            cv2.circle(frame, (280, 290), 10, (255, 0, 0), -1)
            cv2.circle(frame, (350, 290), 10, (255, 255, 0), -1)
            cv2.circle(frame, (280, 190), 10, (255, 255, 0), -1)
        
        # Generate random vehicles based on current metrics
        # The number of vehicles depends on congestion level
        num_vehicles = max(3, min(20, int(self.current_metrics['queue_length'] * 2)))
        
        # Direction probabilities based on current phase
        # More vehicles flow in the green direction
        if self.current_phase == 0 or self.current_phase == 1:  # NS green or yellow
            ns_prob = 0.7
        else:  # EW green or yellow
            ns_prob = 0.3
            
        for _ in range(num_vehicles):
            # Decide direction (NS or EW)
            if np.random.random() < ns_prob:
                # North-South vehicle
                x = np.random.randint(305, 335)
                y = np.random.randint(0, 480)
                w, h = 25, 40
                color = tuple(np.random.randint(50, 250, 3).tolist())
                cv2.rectangle(frame, (x, y), (x+w, h+y), color, -1)
                # Add car details
                cv2.rectangle(frame, (x+5, y+5), (x+w-5, y+10), (0, 0, 0), -1)  # Windows
                cv2.rectangle(frame, (x+5, y+h-15), (x+w-5, y+h-5), (0, 0, 0), -1)
            else:
                # East-West vehicle
                x = np.random.randint(0, 640)
                y = np.random.randint(225, 255)
                w, h = 40, 25
                color = tuple(np.random.randint(50, 250, 3).tolist())
                cv2.rectangle(frame, (x, y), (x+w, h+y), color, -1)
                # Add car details
                cv2.rectangle(frame, (x+5, y+5), (x+10, y+h-5), (0, 0, 0), -1)  # Windows
                cv2.rectangle(frame, (x+w-15, y+5), (x+w-5, y+h-5), (0, 0, 0), -1)
        
        # Add time and current phase text
        time_str = datetime.now().strftime("%H:%M:%S")
        cv2.putText(frame, f"Time: {time_str}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                    
        cv2.putText(frame, f"Phase: {self.phases[self.current_phase]}", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        # Add traffic metrics
        cv2.putText(frame, f"Queue: {int(self.current_metrics['queue_length'])} vehicles", (10, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                    
        cv2.putText(frame, f"Congestion: {self.current_metrics['congestion']:.2f}", (10, 120), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        # Add time remaining in phase
        elapsed = time.time() - self.phase_start_time
        remaining = max(0, self.phase_duration - elapsed)
        cv2.putText(frame, f"Remaining: {int(remaining)}s", (10, 150), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        return frame
    
    def process_frame(self, frame):
        """Process the current frame and update metrics"""
        # Process with vehicle detector
        detection_results = self.detector.process_frame(frame)
        
        # Update controller state
        self.controller.update_state_from_vision(frame)
        
        # Calculate metrics based on detection
        traffic_density = detection_results['traffic_density']
        
        # Update current metrics with real or simulated values
        metrics = {
            'waiting_time': np.random.uniform(5, 30) * (traffic_density['vehicle_count'] / 10),
            'queue_length': traffic_density['vehicle_count'] * 0.4,
            'throughput': traffic_density['vehicle_count'] * 0.6,
            'congestion': min(1.0, traffic_density['vehicle_count'] / 20.0)
        }
        
        # Update controller metrics
        self.controller.update_metrics(metrics)
        
        # Calculate reward
        reward = self.controller.calculate_reward(metrics)
        metrics['reward'] = reward
        
        # Update current metrics
        self.current_metrics = metrics
        
        # Add metrics to history
        for key in self.metrics_history:
            self.metrics_history[key].append(metrics[key])
            # Keep history at fixed length
            if len(self.metrics_history[key]) > self.history_length:
                self.metrics_history[key].pop(0)
        
        # Check if we need to change the traffic signal phase
        elapsed = time.time() - self.phase_start_time
        if elapsed >= self.phase_duration:
            self.select_next_phase()
        
        return detection_results
    
    def _estimate_traffic_density(self):
        """Estimate traffic density and other metrics for phase decision-making"""
        # Either use detection results or generate simulated data
        if hasattr(self, 'detection_results') and self.detection_results:
            traffic_density = self.detection_results.get('traffic_density', {})
            if not traffic_density:
                # Generate simulated data if detection hasn't provided any
                vehicle_count = np.random.randint(5, 25)
                density_level = "Medium" if vehicle_count < 15 else "High"
                
                # Direction bias based on phase
                if self.current_phase == 0 or self.current_phase == 1:  # NS phase
                    ns_ratio = 0.7  # More N-S traffic during N-S green
                else:
                    ns_ratio = 0.3  # Less N-S traffic during E-W green
                
                traffic_density = {
                    'vehicle_count': vehicle_count,
                    'density_level': density_level,
                    'ns_vehicle_ratio': ns_ratio,
                    'waiting_time': vehicle_count * 1.5,
                    'queue_length': vehicle_count * 0.8
                }
        else:
            # Generate completely simulated data with time-of-day patterns
            current_hour = datetime.now().hour
            
            # Morning and evening rush hours have higher traffic
            is_rush_hour = (7 <= current_hour <= 9) or (16 <= current_hour <= 18)
            
            if is_rush_hour:
                base_count = np.random.randint(15, 25)
            else:
                base_count = np.random.randint(5, 15)
            
            # Add randomness to simulate traffic patterns
            vehicle_count = int(base_count * np.random.uniform(0.8, 1.2))
            density_level = "Low" if vehicle_count < 10 else "Medium" if vehicle_count < 20 else "High"
            
            # Direction bias changes throughout the day
            if 7 <= current_hour <= 10:  # Morning commute (into city)
                ns_ratio = np.random.uniform(0.6, 0.8)
            elif 15 <= current_hour <= 19:  # Evening commute (out of city)
                ns_ratio = np.random.uniform(0.6, 0.8)
            else:
                ns_ratio = np.random.uniform(0.4, 0.6)
                
            traffic_density = {
                'vehicle_count': vehicle_count,
                'density_level': density_level,
                'ns_vehicle_ratio': ns_ratio,
                'waiting_time': vehicle_count * 1.5,
                'queue_length': vehicle_count * 0.8
            }
            
        return traffic_density
    
    def select_next_phase(self):
        """Select the next traffic signal phase"""
        # Get current traffic metrics to inform phase selection
        traffic_density = self._estimate_traffic_density()
        congestion_level = min(1.0, traffic_density['vehicle_count'] / 20.0)
        
        # Create portal-like data structure that would come from the real portal
        portal_data = {
            'vehicle_count': traffic_density['vehicle_count'],
            'vehicle_types': {
                'cars': int(traffic_density['vehicle_count'] * 0.8),
                'buses': int(traffic_density['vehicle_count'] * 0.1),
                'trucks': int(traffic_density['vehicle_count'] * 0.1)
            },
            'lane_occupancy': {
                'north': traffic_density.get('ns_vehicle_ratio', 0.5),
                'south': traffic_density.get('ns_vehicle_ratio', 0.5),
                'east': 1 - traffic_density.get('ns_vehicle_ratio', 0.5),
                'west': 1 - traffic_density.get('ns_vehicle_ratio', 0.5)
            },
            'approach_congestion': {
                'north': congestion_level * traffic_density.get('ns_vehicle_ratio', 0.5),
                'south': congestion_level * traffic_density.get('ns_vehicle_ratio', 0.5),
                'east': congestion_level * (1 - traffic_density.get('ns_vehicle_ratio', 0.5)),
                'west': congestion_level * (1 - traffic_density.get('ns_vehicle_ratio', 0.5))
            },
            'timestamp': datetime.now().isoformat()
        }
        
        # Update controller with portal data
        if hasattr(self.controller, 'update_from_portal'):
            self.controller.update_from_portal(portal_data)
        else:
            # Fallback to regular metrics update
            metrics = {
                'waiting_time': traffic_density['waiting_time'],
                'queue_length': traffic_density['queue_length'],
                'throughput': traffic_density['vehicle_count'] * 0.6,
                'congestion': congestion_level
            }
            self.controller.update_metrics(metrics)
        
        # Get next phase from controller with appropriate duration
        phase_index, duration = self.controller.select_phase()
        
        # Update phase
        self.current_phase = phase_index
        self.phase_start_time = time.time()
        self.phase_duration = duration
        
        # Increment decision counter
        self.decisions[self.controller.mode] += 1
        
        # Log the phase change with context about traffic conditions
        logger.info(f"New phase: {self.phases[phase_index]}, Duration: {duration}s")
        logger.info(f"Traffic conditions: Vehicles: {traffic_density['vehicle_count']}, Congestion: {congestion_level:.2f}")
        
        return phase_index, duration
    
    def draw_traffic_signal(self):
        """Draw a traffic signal visualization with congestion indicators"""
        # Create a blank image for the traffic signal
        signal_img = np.ones((400, 300, 3), dtype=np.uint8) * 255
        
        # Get current congestion level
        congestion_level = self.controller._calculate_congestion_level() if hasattr(self.controller, '_calculate_congestion_level') else 0.5
        
        # Draw the traffic light housing
        cv2.rectangle(signal_img, (100, 50), (200, 250), (80, 80, 80), -1)
        
        # Draw the lights based on current phase
        # Red light
        red_color = (255, 0, 0) if self.current_phase in [1, 2, 3] else (100, 0, 0)
        cv2.circle(signal_img, (150, 100), 30, red_color, -1)
        
        # Yellow light
        yellow_color = (255, 255, 0) if self.current_phase in [1, 3] else (100, 100, 0)
        cv2.circle(signal_img, (150, 150), 30, yellow_color, -1)
        
        # Green light
        green_color = (0, 255, 0) if self.current_phase in [0, 2] else (0, 100, 0)
        cv2.circle(signal_img, (150, 200), 30, green_color, -1)
        
        # Add phase text
        cv2.putText(signal_img, f"Phase: {self.phases[self.current_phase]}", 
                    (50, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        # Add time remaining
        elapsed = time.time() - self.phase_start_time
        remaining = max(0, self.phase_duration - elapsed)
        cv2.putText(signal_img, f"Remaining: {int(remaining)}s", 
                    (50, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        # Draw congestion gauge
        gauge_width = 200
        gauge_height = 20
        gauge_x = 50
        gauge_y = 320
        
        # Draw gauge background
        cv2.rectangle(signal_img, (gauge_x, gauge_y), 
                     (gauge_x + gauge_width, gauge_y + gauge_height), 
                     (200, 200, 200), -1)
        
        # Draw congestion level
        congestion_width = int(gauge_width * congestion_level)
        
        # Color based on congestion level
        if congestion_level < 0.4:
            color = (0, 255, 0)  # Green for low congestion
        elif congestion_level < 0.7:
            color = (0, 255, 255)  # Yellow for medium congestion
        else:
            color = (0, 0, 255)  # Red for high congestion
            
        cv2.rectangle(signal_img, (gauge_x, gauge_y), 
                     (gauge_x + congestion_width, gauge_y + gauge_height), 
                     color, -1)
        
        # Add congestion label
        cv2.putText(signal_img, f"Congestion: {congestion_level:.2f}", 
                   (gauge_x, gauge_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.6, (0, 0, 0), 1)
        
        # Add vehicle count if available
        if hasattr(self.controller, 'portal_data') and 'vehicle_count' in self.controller.portal_data:
            vehicle_count = self.controller.portal_data['vehicle_count']
            cv2.putText(signal_img, f"Vehicles: {vehicle_count}", 
                      (gauge_x, gauge_y + gauge_height + 20), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        
        return signal_img
    
    def update_plot(self, frame_num):
        """Update the matplotlib plot with new data"""
        # Get new frame and process it
        frame = self.get_frame()
        self.detection_results = self.process_frame(frame)
        
        # Update traffic view
        self.ax_traffic.clear()
        self.ax_traffic.set_title("Traffic Camera View")
        self.ax_traffic.imshow(cv2.cvtColor(self.detection_results['annotated_frame'], cv2.COLOR_BGR2RGB))
        self.ax_traffic.axis('off')
        
        # Calculate remaining time for current phase
        elapsed = time.time() - self.phase_start_time
        remaining = max(0, self.phase_duration - elapsed)
        
        # Update traffic light visualization
        self.ax_signal.clear()
        phase_name = self.phases[self.current_phase]
        congestion_level = self.controller._calculate_congestion_level() if hasattr(self.controller, '_calculate_congestion_level') else 0.5
        self.ax_signal.set_title(f"{phase_name}\nRemaining: {int(remaining)}s | Congestion: {congestion_level:.2f}")
        self.ax_signal.imshow(cv2.cvtColor(self.draw_traffic_signal(), cv2.COLOR_BGR2RGB))
        self.ax_signal.axis('off')
        
        # Update metrics plot
        self.ax_metrics.clear()
        self.ax_metrics.set_title("Traffic Metrics")
        self.ax_metrics.set_xlabel("Time Steps")
        self.ax_metrics.set_ylabel("Normalized Value")
        
        x = list(range(len(self.metrics_history['waiting_time'])))
        if x:
            # Normalize metrics for better visualization
            waiting_time = [w/30 for w in self.metrics_history['waiting_time']]
            queue_length = [q/15 for q in self.metrics_history['queue_length']]
            congestion = self.metrics_history['congestion']
            
            self.ax_metrics.plot(x, waiting_time, 'r-', label='Wait Time')
            self.ax_metrics.plot(x, queue_length, 'b-', label='Queue')
            self.ax_metrics.plot(x, congestion, 'g-', label='Congestion')
            
            self.ax_metrics.set_xlim(0, self.history_length)
            self.ax_metrics.set_ylim(0, 1.1)
            self.ax_metrics.legend(loc='upper right')
            self.ax_metrics.grid(True, alpha=0.3)
        
        # Update decision pie chart
        self.ax_pie.clear()
        self.ax_pie.set_title("Decision Sources")
        
        total_decisions = sum(self.decisions.values())
        if total_decisions > 0:
            labels = list(self.decisions.keys())
            sizes = list(self.decisions.values())
            colors = ['#4285F4', '#EA4335', '#34A853']  # Blue, Red, Green
            self.ax_pie.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                           startangle=90, shadow=True)
            self.ax_pie.axis('equal')
        
        # Add overall metrics as text
        info_text = (
            f"Control Mode: {self.controller.mode.upper()}\n"
            f"Current Time: {datetime.now().strftime('%H:%M:%S')}\n"
            f"Queue Length: {self.current_metrics['queue_length']:.1f} vehicles\n"
            f"Waiting Time: {self.current_metrics['waiting_time']:.1f} seconds\n"
            f"Congestion: {self.current_metrics['congestion']:.2f}\n"
            f"Last Reward: {self.current_metrics['reward']:.2f}"
        )
        self.fig.text(0.01, 0.01, info_text, fontsize=10)
        
        # Record frame if recording
        if self.recording and len(self.record_frames) < self.max_record_frames:
            # Save the current figure as an image
            fig_img = self.get_figure_image()
            self.record_frames.append(fig_img)
    
    def get_figure_image(self):
        """Convert the current figure to an image"""
        self.fig.canvas.draw()
        img = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
        return img
    
    def save_recording(self):
        """Save the recorded frames as a video"""
        if not self.record_frames:
            return
            
        # Define video writer
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"recordings/visualization_{timestamp}.avi"
        os.makedirs("recordings", exist_ok=True)
        
        # Get dimensions from first frame
        height, width = self.record_frames[0].shape[:2]
        
        out = cv2.VideoWriter(output_file, fourcc, 30.0, (width, height))
        
        # Write frames
        for frame in self.record_frames:
            # Convert from RGB to BGR
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
        
        out.release()
        logger.info(f"Recording saved to {output_file}")
    
    def toggle_recording(self):
        """Toggle recording on/off"""
        self.recording = not self.recording
        if self.recording:
            logger.info("Recording started")
            self.record_frames = []
        else:
            if self.record_frames:
                self.save_recording()
            logger.info("Recording stopped")
    
    def run(self, duration=None):
        """Run the visualization loop"""
        # Set up the animation
        anim = FuncAnimation(self.fig, self.update_plot, interval=100, cache_frame_data=False)
        
        # Register key event handler
        def on_key_press(event):
            if event.key == 'q':
                plt.close(self.fig)
            elif event.key == ' ':
                self.select_next_phase()
            elif event.key == 'r':
                self.toggle_recording()
            elif event.key == 'm':
                # Toggle control mode
                if self.controller.mode == "rl":
                    self.controller.mode = "rules"
                elif self.controller.mode == "rules":
                    self.controller.mode = "hybrid"
                else:
                    self.controller.mode = "rl"
                logger.info(f"Switched to {self.controller.mode} mode")
        
        self.fig.canvas.mpl_connect('key_press_event', on_key_press)
        
        # Show plot instructions
        print("\nVisualization Controls:")
        print("  Press 'q' to quit")
        print("  Press 'space' to force next phase")
        print("  Press 'm' to change control mode")
        print("  Press 'r' to toggle recording")
        
        # Create a timer for auto-close if duration is specified
        if duration:
            def close_after_duration():
                plt.close(self.fig)
            
            timer = self.fig.canvas.new_timer(interval=duration*1000)
            timer.add_callback(close_after_duration)
            timer.start()
        
        plt.show()
        
        # Clean up
        if self.cap:
            self.cap.release()
        
        # Save recording if active
        if self.recording and self.record_frames:
            self.save_recording()
        
        logger.info("Visualization closed")

def main():
    """Main function"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Traffic Signal Visualization System")
    parser.add_argument("--video", type=str, default=None, 
                        help="Path to video file or camera index")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to trained model")
    parser.add_argument("--use-analysis", action="store_true",
                        help="Use traffic analysis results for signal optimization")
    parser.add_argument("--duration", type=int, default=None,
                        help="Duration to run in seconds")
    
    args = parser.parse_args()
    
    # Create visualization
    vis = TrafficSignalVisualization(
        video_source=args.video,
        use_analysis=args.use_analysis,
        model_path=args.model
    )
    
    # Run visualization
    vis.run(args.duration)

if __name__ == "__main__":
    main()
