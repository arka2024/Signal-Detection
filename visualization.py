"""
Traffic Signal Visualization and Demo System

This script creates a visual demonstration of the adaptive traffic signal controller in action,
showing real-time metrics, signal decisions, and simulated traffic conditions. The visualization
includes:

1. Traffic signal visualization (current phase and timing)
2. Real-time traffic metrics display (waiting time, queue length, congestion)
3. Decision metrics from the controller (RL decisions vs rule-based decisions)
4. Simulated traffic flow with visual representation

This can be used to demonstrate the system without requiring actual hardware integration.
"""

import os
import sys
import time
import json
import argparse
import numpy as np
import pygame
import cv2
from datetime import datetime, timedelta
import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.backends.backend_agg as agg
import logging

# Import project modules
from src.adaptive_controller import AdaptiveSignalController
from src.vision import VehicleDetector

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
GRAY = (200, 200, 200)
DARK_GRAY = (100, 100, 100)

class TrafficSignalVisualization:
    """Traffic signal visualization system that integrates with the adaptive controller"""
    
    def __init__(self, width=1280, height=720, video_source=None, use_analysis=True,
                 model_path=None, config_file="config.yaml"):
        """Initialize the visualization system"""
        self.width = width
        self.height = height
        self.video_source = video_source
        
        # Initialize pygame
        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Intelligent Traffic Signal Management System")
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont('Arial', 18)
        self.font_medium = pygame.font.SysFont('Arial', 24)
        self.font_large = pygame.font.SysFont('Arial', 36)
        
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
        
        # Simulation time acceleration (for demonstration)
        self.time_acceleration = 4.0  # 4x speed
        
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
        
        # Create surfaces for plotting
        self.plot_surface = pygame.Surface((400, 200))
        
        logger.info("Visualization system initialized")
    
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
    
    def select_next_phase(self):
        """Select the next traffic signal phase"""
        # Get next phase from controller
        phase_index, duration = self.controller.select_phase()
        
        # Update phase
        self.current_phase = phase_index
        self.phase_start_time = time.time()
        self.phase_duration = duration
        
        # Increment decision counter
        self.decisions[self.controller.mode] += 1
        
        logger.info(f"New phase: {self.phases[phase_index]}, Duration: {duration}s")
        
        return phase_index, duration
    
    def create_metrics_plot(self):
        """Create a matplotlib plot of metrics history"""
        fig = plt.figure(figsize=(4, 2), dpi=100)
        ax = fig.add_subplot(111)
        
        # Plot metrics
        x = list(range(len(self.metrics_history['waiting_time'])))
        if x:
            # Normalize metrics for better visualization
            waiting_time = [w/30 for w in self.metrics_history['waiting_time']]
            queue_length = [q/15 for q in self.metrics_history['queue_length']]
            congestion = self.metrics_history['congestion']
            
            ax.plot(x, waiting_time, 'r-', label='Wait Time')
            ax.plot(x, queue_length, 'b-', label='Queue')
            ax.plot(x, congestion, 'g-', label='Congestion')
            
            ax.set_xlim(0, self.history_length)
            ax.set_ylim(0, 1.1)
            ax.set_title('Traffic Metrics (Normalized)')
            ax.legend(loc='upper right', fontsize='small')
            ax.grid(True, alpha=0.3)
        
        # Convert plot to pygame surface
        canvas = agg.FigureCanvasAgg(fig)
        canvas.draw()
        renderer = canvas.get_renderer()
        raw_data = renderer.tostring_rgb()
        size = canvas.get_width_height()
        
        # Close figure to avoid memory leak
        plt.close(fig)
        
        # Create pygame surface
        surf = pygame.image.fromstring(raw_data, size, "RGB")
        
        return surf
    
    def render_decision_pie(self, surface, rect):
        """Render a pie chart of decision sources"""
        total = sum(self.decisions.values())
        if total == 0:
            return
        
        # Center and radius
        center_x = rect[0] + rect[2]//2
        center_y = rect[1] + rect[3]//2
        radius = min(rect[2], rect[3]) // 2 - 10
        
        # Colors for each decision type
        colors = {
            'rl': (0, 128, 255),     # Blue
            'rules': (255, 128, 0),  # Orange
            'hybrid': (0, 192, 0)     # Green
        }
        
        # Draw pie slices
        start_angle = 0
        for decision, count in self.decisions.items():
            if count > 0:
                angle = count / total * 360
                end_angle = start_angle + angle
                
                # Draw slice
                pygame.draw.arc(surface, colors[decision], 
                               (center_x-radius, center_y-radius, radius*2, radius*2),
                               np.radians(start_angle), np.radians(end_angle), radius)
                
                # Fill slice
                for r in range(radius):
                    pygame.draw.arc(surface, colors[decision], 
                                   (center_x-r, center_y-r, r*2, r*2),
                                   np.radians(start_angle), np.radians(end_angle), 1)
                
                # Draw label
                mid_angle = np.radians((start_angle + end_angle) / 2)
                text_x = center_x + int(radius * 0.7 * np.cos(mid_angle))
                text_y = center_y - int(radius * 0.7 * np.sin(mid_angle))
                text = self.font_small.render(f"{decision}: {count}", True, WHITE)
                text_rect = text.get_rect(center=(text_x, text_y))
                surface.blit(text, text_rect)
                
                start_angle = end_angle
        
        # Draw border
        pygame.draw.circle(surface, WHITE, (center_x, center_y), radius, 1)
        
        # Title
        title = self.font_medium.render("Decision Sources", True, WHITE)
        title_rect = title.get_rect(center=(center_x, rect[1] + 20))
        surface.blit(title, title_rect)
    
    def render(self):
        """Render the visualization"""
        # Fill screen with black
        self.screen.fill(BLACK)
        
        # Get and process current frame
        frame = self.get_frame()
        self.current_frame = frame
        self.detection_results = self.process_frame(frame)
        
        # Convert frame to pygame surface
        frame_rgb = cv2.cvtColor(self.detection_results['annotated_frame'], cv2.COLOR_BGR2RGB)
        frame_surface = pygame.surfarray.make_surface(frame_rgb.swapaxes(0, 1))
        
        # Display frame in the left portion of the screen
        frame_rect = pygame.Rect(20, 20, 640, 480)
        self.screen.blit(frame_surface, frame_rect)
        
        # Display traffic signal info
        signal_rect = pygame.Rect(680, 20, 580, 100)
        pygame.draw.rect(self.screen, DARK_GRAY, signal_rect)
        
        # Phase info
        phase_text = self.font_large.render(f"Current Phase: {self.phases[self.current_phase]}", 
                                           True, WHITE)
        self.screen.blit(phase_text, (700, 30))
        
        # Time remaining
        elapsed = time.time() - self.phase_start_time
        remaining = max(0, self.phase_duration - elapsed)
        
        # Choose color based on time remaining (green -> yellow -> red)
        if remaining > self.phase_duration * 0.6:
            time_color = GREEN
        elif remaining > self.phase_duration * 0.3:
            time_color = YELLOW
        else:
            time_color = RED
            
        time_text = self.font_large.render(f"Time Remaining: {int(remaining)}s", True, time_color)
        self.screen.blit(time_text, (700, 70))
        
        # Display current metrics
        metrics_rect = pygame.Rect(680, 140, 280, 200)
        pygame.draw.rect(self.screen, DARK_GRAY, metrics_rect)
        
        # Title
        title = self.font_medium.render("Current Traffic Metrics", True, WHITE)
        self.screen.blit(title, (700, 150))
        
        # Metrics
        y_pos = 190
        metrics_to_display = [
            ("Wait Time", f"{self.current_metrics['waiting_time']:.1f}s"),
            ("Queue Length", f"{self.current_metrics['queue_length']:.1f} veh"),
            ("Throughput", f"{self.current_metrics['throughput']:.1f} veh/min"),
            ("Congestion", f"{self.current_metrics['congestion']:.2f}"),
            ("Reward", f"{self.current_metrics['reward']:.2f}")
        ]
        
        for label, value in metrics_to_display:
            label_text = self.font_small.render(f"{label}:", True, WHITE)
            value_text = self.font_small.render(value, True, WHITE)
            self.screen.blit(label_text, (700, y_pos))
            self.screen.blit(value_text, (860, y_pos))
            y_pos += 30
        
        # Display metrics plot
        plot_rect = pygame.Rect(680, 360, 580, 300)
        pygame.draw.rect(self.screen, DARK_GRAY, plot_rect)
        
        # Create and display plot
        plot_surface = self.create_metrics_plot()
        self.screen.blit(plot_surface, (700, 380))
        
        # Display decision metrics
        decision_rect = pygame.Rect(980, 140, 280, 200)
        pygame.draw.rect(self.screen, DARK_GRAY, decision_rect)
        
        # Render decision pie chart
        self.render_decision_pie(self.screen, decision_rect)
        
        # Display current date and time
        time_rect = pygame.Rect(20, 520, 640, 40)
        pygame.draw.rect(self.screen, DARK_GRAY, time_rect)
        
        time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        time_text = self.font_medium.render(f"Current Time: {time_str}", True, WHITE)
        self.screen.blit(time_text, (30, 530))
        
        # Display controller mode
        mode_rect = pygame.Rect(20, 580, 640, 40)
        pygame.draw.rect(self.screen, DARK_GRAY, mode_rect)
        
        mode_text = self.font_medium.render(f"Control Mode: {self.controller.mode.upper()}", True, WHITE)
        self.screen.blit(mode_text, (30, 590))
        
        # Record frame if recording
        if self.recording and len(self.record_frames) < self.max_record_frames:
            # Create a surface from the current screen
            frame = pygame.surfarray.array3d(self.screen)
            frame = frame.swapaxes(0, 1)  # Swap axes to match OpenCV format
            self.record_frames.append(frame)
        
        # Update display
        pygame.display.flip()
    
    def handle_events(self):
        """Handle pygame events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                elif event.key == pygame.K_SPACE:
                    # Force next phase
                    self.select_next_phase()
                elif event.key == pygame.K_m:
                    # Toggle control mode
                    if self.controller.mode == "rl":
                        self.controller.mode = "rules"
                    elif self.controller.mode == "rules":
                        self.controller.mode = "hybrid"
                    else:
                        self.controller.mode = "rl"
                    logger.info(f"Switched to {self.controller.mode} mode")
                elif event.key == pygame.K_r:
                    # Toggle recording
                    self.recording = not self.recording
                    if self.recording:
                        logger.info("Recording started")
                        self.record_frames = []
                    else:
                        if self.record_frames:
                            self.save_recording()
                        logger.info("Recording stopped")
                elif event.key == pygame.K_UP:
                    # Increase time acceleration
                    self.time_acceleration = min(10.0, self.time_acceleration + 0.5)
                    logger.info(f"Time acceleration: {self.time_acceleration}x")
                elif event.key == pygame.K_DOWN:
                    # Decrease time acceleration
                    self.time_acceleration = max(1.0, self.time_acceleration - 0.5)
                    logger.info(f"Time acceleration: {self.time_acceleration}x")
        
        return True
    
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
    
    def run(self, duration=None):
        """Run the visualization loop"""
        running = True
        start_time = time.time()
        
        while running:
            # Handle events
            running = self.handle_events()
            
            # Render visualization
            self.render()
            
            # Check if duration exceeded
            if duration and (time.time() - start_time) > duration:
                running = False
            
            # Control frame rate
            self.clock.tick(30 * self.time_acceleration)
        
        # Clean up
        if self.cap:
            self.cap.release()
        pygame.quit()
        
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
    parser.add_argument("--width", type=int, default=1280,
                        help="Window width")
    parser.add_argument("--height", type=int, default=720,
                        help="Window height")
    
    args = parser.parse_args()
    
    # Create visualization
    vis = TrafficSignalVisualization(
        width=args.width,
        height=args.height,
        video_source=args.video,
        use_analysis=args.use_analysis,
        model_path=args.model
    )
    
    # Run visualization
    vis.run(args.duration)

if __name__ == "__main__":
    main()
