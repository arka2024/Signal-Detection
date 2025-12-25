"""
Computer Vision Module for Traffic Detection and Analysis
"""

import os
import cv2
import numpy as np
import yaml
import logging
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VehicleDetector:
    """
    Class for detecting and tracking vehicles in traffic camera feeds
    using computer vision techniques.
    """
    
    def __init__(self, config_file="config.yaml"):
        """Initialize the vehicle detector"""
        # Load configuration
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        vision_config = config['vision']
        
        # Detection parameters
        self.detection_threshold = vision_config['detection_threshold']
        self.vehicle_classes = vision_config['vehicle_classes']
        self.use_anpr = vision_config['use_anpr']
        
        # Initialize detector
        self._init_detector()
        
        # Vehicle tracking data
        self.tracked_vehicles = {}
        self.vehicle_counts = defaultdict(int)
    
    def _init_detector(self):
        """Initialize the object detection model"""
        # For actual implementation, we would load pre-trained models
        # Here, we'll simulate using OpenCV's DNN module with YOLO or similar
        
        # In production, replace with actual model paths
        try:
            # Load YOLOv4 configuration and weights
            model_path = os.path.join("models", "detection")
            config_path = os.path.join(model_path, "yolov4.cfg")
            weights_path = os.path.join(model_path, "yolov4.weights")
            
            # Check if files exist, if not, log a warning but continue
            if not (os.path.exists(config_path) and os.path.exists(weights_path)):
                logger.warning(f"Detection model files not found. Using mock detection instead.")
                self.net = None
            else:
                # Load the network
                self.net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
                
                # Set preferred backend and target
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)  # Use GPU if available
        except Exception as e:
            logger.error(f"Error initializing detector: {e}")
            self.net = None
    
    def detect_vehicles(self, frame):
        """
        Detect vehicles in a video frame
        
        Args:
            frame: OpenCV image frame
            
        Returns:
            detections: List of dictionaries with detection information
                - class_id: Class of the vehicle
                - confidence: Detection confidence
                - box: Bounding box coordinates [x, y, w, h]
        """
        # If no model is loaded, return mock detections
        if self.net is None:
            return self._mock_detection(frame)
        
        # Get image dimensions
        height, width = frame.shape[:2]
        
        # Create a blob from the image
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
        
        # Set input to the network
        self.net.setInput(blob)
        
        # Get the output layer names
        layer_names = self.net.getLayerNames()
        output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
        
        # Forward pass through the network
        outputs = self.net.forward(output_layers)
        
        # Process the outputs
        detections = []
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if confidence > self.detection_threshold and class_id < len(self.vehicle_classes):
                    # Scale bounding box coordinates to image size
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    
                    # Get the top-left corner coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    
                    detections.append({
                        'class_id': self.vehicle_classes[class_id],
                        'confidence': float(confidence),
                        'box': [x, y, w, h]
                    })
        
        return detections
    
    def _mock_detection(self, frame):
        """
        Create mock detections for testing when no model is loaded
        
        Args:
            frame: OpenCV image frame
            
        Returns:
            detections: List of simulated detections
        """
        height, width = frame.shape[:2]
        
        # Create random detections for testing
        num_vehicles = np.random.randint(1, 10)
        detections = []
        
        for _ in range(num_vehicles):
            # Random vehicle class
            class_id = np.random.choice(self.vehicle_classes)
            
            # Random confidence
            confidence = np.random.uniform(0.5, 0.99)
            
            # Random box dimensions
            x = np.random.randint(0, width - 100)
            y = np.random.randint(0, height - 100)
            w = np.random.randint(50, 150)
            h = np.random.randint(50, 150)
            
            detections.append({
                'class_id': class_id,
                'confidence': float(confidence),
                'box': [x, y, w, h]
            })
        
        return detections
    
    def track_vehicles(self, frame, detections):
        """
        Track vehicles across frames using simple tracking
        
        Args:
            frame: Current frame
            detections: List of detections
            
        Returns:
            tracked: Dictionary mapping tracking IDs to vehicle information
        """
        # In a real implementation, this would use a proper tracking algorithm
        # like SORT, DeepSORT, or similar
        
        # For simplicity, we'll use a very basic approach that just assigns
        # new IDs to vehicles in each frame
        
        tracked = {}
        for i, detection in enumerate(detections):
            tracking_id = f"vehicle_{i}"
            
            tracked[tracking_id] = {
                'class_id': detection['class_id'],
                'confidence': detection['confidence'],
                'box': detection['box'],
                'speed': np.random.randint(0, 60)  # Mock speed in km/h
            }
            
            # Update vehicle count
            self.vehicle_counts[detection['class_id']] += 1
        
        return tracked
    
    def recognize_license_plate(self, frame, box):
        """
        Apply ANPR (Automatic Number Plate Recognition) to the vehicle
        
        Args:
            frame: Current frame
            box: Bounding box of the vehicle [x, y, w, h]
            
        Returns:
            plate_number: Recognized license plate number or None
        """
        if not self.use_anpr:
            return None
            
        # In a real implementation, this would use OCR and ANPR techniques
        # For now, just return a mock plate number
        
        # Generate a random license plate
        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        numbers = "0123456789"
        
        plate_format = np.random.choice(["XX 00 XX 0000", "XX 00 XXXX", "XX 00 XX"])
        plate_number = ""
        
        for char in plate_format:
            if char == 'X':
                plate_number += np.random.choice(list(letters))
            elif char == '0':
                plate_number += np.random.choice(list(numbers))
            else:
                plate_number += char
        
        return plate_number
    
    def analyze_traffic_density(self, frame, tracked_vehicles):
        """
        Analyze traffic density from vehicle tracking data
        
        Args:
            frame: Current frame
            tracked_vehicles: Dictionary of tracked vehicles
            
        Returns:
            density_info: Dictionary with traffic density metrics
        """
        height, width = frame.shape[:2]
        frame_area = height * width
        
        # Calculate area occupied by vehicles
        vehicle_area = 0
        for vehicle in tracked_vehicles.values():
            x, y, w, h = vehicle['box']
            vehicle_area += w * h
        
        # Calculate density metrics
        occupancy_ratio = min(1.0, vehicle_area / frame_area)
        vehicle_count = len(tracked_vehicles)
        
        # Classify density
        if occupancy_ratio < 0.1:
            density_level = "Light"
        elif occupancy_ratio < 0.25:
            density_level = "Moderate"
        elif occupancy_ratio < 0.5:
            density_level = "Heavy"
        else:
            density_level = "Severe"
        
        return {
            'vehicle_count': vehicle_count,
            'occupancy_ratio': occupancy_ratio,
            'density_level': density_level,
            'vehicle_types': dict(self.vehicle_counts)
        }
    
    def process_frame(self, frame):
        """
        Process a single video frame for traffic analysis
        
        Args:
            frame: OpenCV image frame
            
        Returns:
            result: Dictionary with processing results
                - tracked_vehicles: Tracked vehicles information
                - traffic_density: Traffic density analysis
                - annotated_frame: Frame with visualizations
        """
        # Make a copy of the frame for visualization
        viz_frame = frame.copy()
        
        # Detect vehicles
        detections = self.detect_vehicles(frame)
        
        # Track vehicles
        tracked_vehicles = self.track_vehicles(frame, detections)
        
        # Analyze traffic density
        traffic_density = self.analyze_traffic_density(frame, tracked_vehicles)
        
        # Annotate frame with detections
        for tracking_id, vehicle in tracked_vehicles.items():
            x, y, w, h = vehicle['box']
            class_id = vehicle['class_id']
            confidence = vehicle['confidence']
            
            # Draw bounding box
            cv2.rectangle(viz_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Add label
            label = f"{class_id}: {confidence:.2f}"
            cv2.putText(viz_frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Add tracking ID
            cv2.putText(viz_frame, tracking_id, (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            # If ANPR is enabled, recognize and display plate
            if self.use_anpr:
                plate_number = self.recognize_license_plate(frame, vehicle['box'])
                if plate_number:
                    cv2.putText(viz_frame, plate_number, (x, y + h + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        # Add density info
        density_text = f"Traffic: {traffic_density['density_level']} ({traffic_density['vehicle_count']} vehicles)"
        cv2.putText(viz_frame, density_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return {
            'tracked_vehicles': tracked_vehicles,
            'traffic_density': traffic_density,
            'annotated_frame': viz_frame
        }

    def process_video(self, video_path, output_path=None, max_frames=None):
        """
        Process a video file for traffic analysis
        
        Args:
            video_path: Path to the input video file
            output_path: Path to save the output video (optional)
            max_frames: Maximum number of frames to process (optional)
            
        Returns:
            results: List of processing results for each frame
        """
        # Open the video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Error: Could not open video {video_path}")
            return []
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Initialize video writer if output path is provided
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Process frames
        results = []
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Process the frame
            result = self.process_frame(frame)
            results.append(result)
            
            # Write to output video if needed
            if writer:
                writer.write(result['annotated_frame'])
            
            # Increment frame count
            frame_count += 1
            
            # Check if we've reached the maximum number of frames
            if max_frames and frame_count >= max_frames:
                break
        
        # Release resources
        cap.release()
        if writer:
            writer.release()
        
        return results

if __name__ == "__main__":
    # Test the vehicle detector
    detector = VehicleDetector()
    
    # Create a test frame
    test_frame = np.ones((480, 640, 3), dtype=np.uint8) * 255
    
    # Process the test frame
    result = detector.process_frame(test_frame)
    
    # Print results
    print(f"Detected {len(result['tracked_vehicles'])} vehicles")
    print(f"Traffic density: {result['traffic_density']['density_level']}")
    
    # Display the annotated frame
    cv2.imshow("Traffic Detection", result['annotated_frame'])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
