"""
Main script to run the Traffic Signal Management System

This script integrates all components of the traffic signal management system:
- Traffic simulation (SUMO or mock environment)
- Computer vision for vehicle detection
- Adaptive signal controller
- Performance monitoring and logging

Run modes:
- train: Train the controller using reinforcement learning
- test: Test a trained controller
- deploy: Run in deployment mode with real or simulated camera feeds
"""

import os
import sys
import argparse
import yaml
import logging
import numpy as np
import cv2
import json
import time
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    # Import project modules
    from src.environment import TrafficSignalEnv
    from src.agent import DQNAgent
    from src.vision import VehicleDetector
    from src.adaptive_controller import AdaptiveSignalController
    all_modules_imported = True
    logger.info("Successfully imported all modules")
except ImportError as e:
    logger.error(f"Error importing modules: {e}")
    all_modules_imported = False

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Traffic Signal Management System')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'deploy'],
                        help='Mode of operation: train, test, or deploy')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to the configuration file')
    parser.add_argument('--episodes', type=int, default=None,
                        help='Number of episodes to train/test (overrides config)')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to saved model for testing or continuing training')
    parser.add_argument('--video', type=str, default=None,
                        help='Path to video file for vision analysis')
    parser.add_argument('--gui', action='store_true',
                        help='Enable GUI for SUMO simulation')
    parser.add_argument('--duration', type=int, default=60,
                      help='Duration to run deployment in minutes')
    parser.add_argument('--source', type=str, default=None,
                      help='Video source (file path or camera device number)')
    parser.add_argument('--use-analysis', action='store_true',
                      help='Use traffic analysis results to optimize signal timing')
    return parser.parse_args()

def load_config(config_path):
    """Load configuration from YAML file"""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        sys.exit(1)

def setup_directories():
    """Set up directories for models and results"""
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Create timestamp for this run
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join('results', f'run_{timestamp}')
    os.makedirs(run_dir, exist_ok=True)
    
    return run_dir

def train_agent(config, args, run_dir):
    """Train the reinforcement learning agent"""
    logger.info("Starting agent training...")
    
    # Override config with command line args
    if args.episodes:
        config['reinforcement_learning']['num_episodes'] = args.episodes
    
    if args.gui:
        config['simulation']['gui'] = True
    
    # Create environment
    env = TrafficSignalEnv(config_file=args.config)
    
    # Create agent
    agent = DQNAgent(config_file=args.config)
    
    # Get state and action dimensions
    state, _ = env.reset()
    state_size = np.prod(state.shape)  # Flatten for the neural network
    action_size = env.action_space.n
    
    # Initialize agent
    agent.initialize(state_size, action_size)
    
    # Load model if continuing training
    if args.model:
        agent.load(args.model)
        logger.info(f"Loaded model from {args.model}")
    
    # Training loop
    num_episodes = config['reinforcement_learning']['num_episodes']
    max_steps = config['reinforcement_learning']['max_steps_per_episode']
    
    # Track metrics
    scores = []
    avg_scores = []
    waiting_times = []
    congestion_indices = []
    
    for episode in range(1, num_episodes + 1):
        state, _ = env.reset()
        state = state.reshape(1, -1)  # Flatten state
        score = 0
        
        episode_waiting_times = []
        episode_congestion = []
        
        for step in range(max_steps):
            # Select action
            action = agent.act(state)
            
            # Take action
            next_state, reward, done, _, info = env.step(action)
            next_state = next_state.reshape(1, -1)  # Flatten next state
            
            # Store experience and learn
            agent.step(state, action, reward, next_state, done)
            
            # Update state and score
            state = next_state
            score += reward
            
            # Track metrics
            episode_waiting_times.append(info['metrics']['average_waiting_time'])
            episode_congestion.append(info['metrics']['congestion_index'])
            
            if done:
                break
        
        # Record episode results
        scores.append(score)
        avg_score = np.mean(scores[-100:])  # Moving average of last 100 episodes
        avg_scores.append(avg_score)
        
        avg_waiting_time = np.mean(episode_waiting_times)
        waiting_times.append(avg_waiting_time)
        
        avg_congestion = np.mean(episode_congestion)
        congestion_indices.append(avg_congestion)
        
        logger.info(f"Episode {episode}/{num_episodes} | Score: {score:.2f} | Avg Score: {avg_score:.2f} | "
                   f"Waiting Time: {avg_waiting_time:.2f} | Congestion: {avg_congestion:.2f}")
        
        # Save model periodically
        if episode % 100 == 0:
            model_path = os.path.join('models', f'dqn_agent_ep{episode}.pth')
            agent.save(model_path)
            logger.info(f"Model saved to {model_path}")
            
            # Save metrics plot
            plot_metrics(scores, avg_scores, waiting_times, congestion_indices, run_dir, episode)
    
    # Save final model
    final_model_path = os.path.join('models', 'dqn_agent_final.pth')
    agent.save(final_model_path)
    logger.info(f"Final model saved to {final_model_path}")
    
    # Save final metrics plot
    plot_metrics(scores, avg_scores, waiting_times, congestion_indices, run_dir, num_episodes)
    
    # Clean up
    env.close()
    logger.info("Training complete.")
    
    return final_model_path

def test_agent(config, args, run_dir):
    """Test the trained reinforcement learning agent"""
    logger.info("Starting agent testing...")
    
    if args.gui:
        config['simulation']['gui'] = True
    
    # Create environment
    env = TrafficSignalEnv(config_file=args.config)
    
    # Create agent
    agent = DQNAgent(config_file=args.config)
    
    # Get state and action dimensions
    state, _ = env.reset()
    state_size = np.prod(state.shape)
    action_size = env.action_space.n
    
    # Initialize agent
    agent.initialize(state_size, action_size)
    
    # Load model
    model_path = args.model if args.model else os.path.join('models', 'dqn_agent_final.pth')
    agent.load(model_path)
    logger.info(f"Loaded model from {model_path}")
    
    # Testing parameters
    num_episodes = args.episodes if args.episodes else 10
    max_steps = config['reinforcement_learning']['max_steps_per_episode']
    
    # Track metrics
    all_metrics = {
        'scores': [],
        'waiting_times': [],
        'queue_lengths': [],
        'throughputs': [],
        'congestion_indices': []
    }
    
    for episode in range(1, num_episodes + 1):
        state, _ = env.reset()
        state = state.reshape(1, -1)
        score = 0
        
        episode_metrics = {
            'waiting_times': [],
            'queue_lengths': [],
            'throughputs': [],
            'congestion_indices': []
        }
        
        for step in range(max_steps):
            # Select action (no exploration)
            action = agent.act(state, eps=0.0)
            
            # Take action
            next_state, reward, done, _, info = env.step(action)
            next_state = next_state.reshape(1, -1)
            
            # Update state and score
            state = next_state
            score += reward
            
            # Track metrics
            episode_metrics['waiting_times'].append(info['metrics']['average_waiting_time'])
            episode_metrics['queue_lengths'].append(info['metrics']['queue_length'])
            episode_metrics['throughputs'].append(info['metrics']['throughput'])
            episode_metrics['congestion_indices'].append(info['metrics']['congestion_index'])
            
            if done:
                break
        
        # Record episode results
        all_metrics['scores'].append(score)
        all_metrics['waiting_times'].append(np.mean(episode_metrics['waiting_times']))
        all_metrics['queue_lengths'].append(np.mean(episode_metrics['queue_lengths']))
        all_metrics['throughputs'].append(np.mean(episode_metrics['throughputs']))
        all_metrics['congestion_indices'].append(np.mean(episode_metrics['congestion_indices']))
        
        logger.info(f"Episode {episode}/{num_episodes} | Score: {score:.2f} | "
                   f"Waiting Time: {all_metrics['waiting_times'][-1]:.2f} | "
                   f"Queue: {all_metrics['queue_lengths'][-1]:.1f} | "
                   f"Throughput: {all_metrics['throughputs'][-1]:.1f}")
    
    # Calculate overall metrics
    avg_score = np.mean(all_metrics['scores'])
    avg_waiting_time = np.mean(all_metrics['waiting_times'])
    avg_queue_length = np.mean(all_metrics['queue_lengths'])
    avg_throughput = np.mean(all_metrics['throughputs'])
    avg_congestion = np.mean(all_metrics['congestion_indices'])
    
    logger.info("Testing Results:")
    logger.info(f"Average Score: {avg_score:.2f}")
    logger.info(f"Average Waiting Time: {avg_waiting_time:.2f} seconds")
    logger.info(f"Average Queue Length: {avg_queue_length:.1f} vehicles")
    logger.info(f"Average Throughput: {avg_throughput:.1f} vehicles/min")
    logger.info(f"Average Congestion Index: {avg_congestion:.2f}")
    
    # Save test results
    save_test_results(all_metrics, run_dir)
    
    # Clean up
    env.close()
    logger.info("Testing complete.")

def run_vision_demo(config, args, run_dir):
    """Run the computer vision module on a video file"""
    logger.info("Running computer vision demo...")
    
    if not args.video:
        logger.error("No video file specified for vision demo")
        sys.exit(1)
    
    # Create vision detector
    detector = VehicleDetector(config_file=args.config)
    
    # Process video
    output_path = os.path.join(run_dir, 'vision_output.avi')
    results = detector.process_video(args.video, output_path, max_frames=500)
    
    logger.info(f"Processed {len(results)} frames")
    logger.info(f"Output saved to {output_path}")
    
    # Calculate average metrics
    if results:
        avg_density = np.mean([r['traffic_density']['vehicle_count'] for r in results])
        logger.info(f"Average vehicle count: {avg_density:.1f}")
        
        if 'density_level' in results[0]['traffic_density']:
            density_levels = [r['traffic_density']['density_level'] for r in results]
            unique_levels, counts = np.unique(density_levels, return_counts=True)
            for level, count in zip(unique_levels, counts):
                logger.info(f"Traffic level '{level}': {count} frames ({count/len(results)*100:.1f}%)")
    
    logger.info("Vision demo complete.")

def plot_metrics(scores, avg_scores, waiting_times, congestion_indices, run_dir, episode):
    """Plot and save training metrics"""
    plt.figure(figsize=(15, 10))
    
    # Plot scores
    plt.subplot(3, 1, 1)
    plt.plot(scores, label='Score')
    plt.plot(avg_scores, label='Avg Score (100 ep)')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.title('Training Scores')
    plt.legend()
    
    # Plot waiting times
    plt.subplot(3, 1, 2)
    plt.plot(waiting_times)
    plt.xlabel('Episode')
    plt.ylabel('Avg Waiting Time (s)')
    plt.title('Average Waiting Time per Episode')
    
    # Plot congestion
    plt.subplot(3, 1, 3)
    plt.plot(congestion_indices)
    plt.xlabel('Episode')
    plt.ylabel('Congestion Index')
    plt.title('Average Congestion Index per Episode')
    
    plt.tight_layout()
    
    # Save plot
    plt.savefig(os.path.join(run_dir, f'metrics_ep{episode}.png'))
    plt.close()

def save_test_results(metrics, run_dir):
    """Save test results to file and create plots"""
    # Save numerical results
    import json
    with open(os.path.join(run_dir, 'test_results.json'), 'w') as f:
        json.dump({
            'average_score': float(np.mean(metrics['scores'])),
            'average_waiting_time': float(np.mean(metrics['waiting_times'])),
            'average_queue_length': float(np.mean(metrics['queue_lengths'])),
            'average_throughput': float(np.mean(metrics['throughputs'])),
            'average_congestion': float(np.mean(metrics['congestion_indices'])),
            'all_metrics': {
                'scores': [float(x) for x in metrics['scores']],
                'waiting_times': [float(x) for x in metrics['waiting_times']],
                'queue_lengths': [float(x) for x in metrics['queue_lengths']],
                'throughputs': [float(x) for x in metrics['throughputs']],
                'congestion_indices': [float(x) for x in metrics['congestion_indices']]
            }
        }, f, indent=4)
    
    # Create plots
    plt.figure(figsize=(15, 10))
    
    # Plot waiting times
    plt.subplot(2, 2, 1)
    plt.plot(metrics['waiting_times'])
    plt.xlabel('Episode')
    plt.ylabel('Waiting Time (s)')
    plt.title('Average Waiting Time')
    
    # Plot queue lengths
    plt.subplot(2, 2, 2)
    plt.plot(metrics['queue_lengths'])
    plt.xlabel('Episode')
    plt.ylabel('Queue Length')
    plt.title('Average Queue Length')
    
    # Plot throughput
    plt.subplot(2, 2, 3)
    plt.plot(metrics['throughputs'])
    plt.xlabel('Episode')
    plt.ylabel('Throughput (veh/min)')
    plt.title('Average Throughput')
    
    # Plot congestion
    plt.subplot(2, 2, 4)
    plt.plot(metrics['congestion_indices'])
    plt.xlabel('Episode')
    plt.ylabel('Congestion Index')
    plt.title('Average Congestion Index')
    
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, 'test_results.png'))
    plt.close()

def deploy_adaptive_controller(config, args, run_dir):
    """Run the adaptive controller in deployment mode"""
    logger.info("Starting adaptive controller deployment...")
    
    # Create controller
    controller = AdaptiveSignalController(config_file=args.config)
    
    # Load model if specified
    if args.model:
        controller.load_model(args.model)
        logger.info(f"Loaded model from {args.model}")
    else:
        logger.info("No model specified, running with default parameters")
    
    # Create vehicle detector
    detector = VehicleDetector(config_file=args.config)
    
    # Initialize video source
    if args.video:
        cap = cv2.VideoCapture(args.video)
        video_source = f"video file '{args.video}'"
    elif args.source and args.source.isdigit():
        cap = cv2.VideoCapture(int(args.source))
        video_source = f"camera {args.source}"
    else:
        cap = None
        video_source = "mock camera"
    
    logger.info(f"Using {video_source} for traffic detection")
    
    # Set up performance logging
    performance_log = []
    deployment_start = datetime.now()
    deployment_end = deployment_start + timedelta(minutes=args.duration)
    
    # If using analysis results
    if args.use_analysis:
        logger.info("Using traffic analysis results for signal optimization")
        controller.mode = "hybrid"  # Use hybrid mode that combines RL and analysis rules
    else:
        logger.info("Using pure reinforcement learning for signal control")
        controller.mode = "rl"
    
    try:
        step = 0
        while datetime.now() < deployment_end:
            step_start = time.time()
            step += 1
            
            # Get frame from video or generate mock frame
            if cap and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    # Loop video if at end
                    logger.info("Restarting video")
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = cap.read()
                    if not ret:
                        logger.error("Failed to read from video source")
                        break
            else:
                # Generate a mock frame (white background)
                frame = np.ones((480, 640, 3), dtype=np.uint8) * 255
            
            # Process frame
            detection_result = detector.process_frame(frame)
            
            # Update controller state
            controller.update_state_from_vision(frame)
            
            # Select phase and duration
            phase_index, duration = controller.select_phase()
            phase_name = controller.phases[phase_index]
            
            # Update metrics based on detection results
            metrics = {
                'waiting_time': np.random.uniform(5, 30),  # Mock values
                'queue_length': detection_result['traffic_density']['vehicle_count'] * 0.4,
                'throughput': detection_result['traffic_density']['vehicle_count'] * 0.6,
                'congestion': min(1.0, detection_result['traffic_density']['vehicle_count'] / 20.0)
            }
            controller.update_metrics(metrics)
            
            # Calculate reward (for learning)
            reward = controller.calculate_reward(metrics)
            
            # Log performance
            if step % 10 == 0:
                logger.info(f"Step {step} | Phase: {phase_name} | Duration: {duration}s | "
                          f"Vehicles: {detection_result['traffic_density']['vehicle_count']} | "
                          f"Density: {detection_result['traffic_density']['density_level']} | "
                          f"Reward: {reward:.2f}")
            
            # Record metrics
            performance_log.append({
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'step': step,
                'phase': phase_index,
                'phase_name': phase_name,
                'duration': duration,
                'vehicle_count': detection_result['traffic_density']['vehicle_count'],
                'density_level': detection_result['traffic_density']['density_level'],
                'waiting_time': metrics['waiting_time'],
                'queue_length': metrics['queue_length'],
                'throughput': metrics['throughput'],
                'congestion': metrics['congestion'],
                'reward': reward
            })
            
            # Display annotated frame
            cv2.putText(detection_result['annotated_frame'], 
                       f"Signal: {phase_name}", 
                       (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (0, 0, 255), 2)
            
            cv2.putText(detection_result['annotated_frame'], 
                       f"Duration: {duration}s", 
                       (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (0, 0, 255), 2)
            
            cv2.imshow("Adaptive Traffic Signal Control", detection_result['annotated_frame'])
            key = cv2.waitKey(100) & 0xFF
            if key == ord('q'):
                logger.info("User terminated deployment")
                break
            
            # Simulate passage of time (accelerated for demonstration)
            elapsed = time.time() - step_start
            sleep_time = min(1.0, max(0.1, duration / 10.0) - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    except KeyboardInterrupt:
        logger.info("Deployment interrupted by user")
    
    finally:
        # Clean up
        if cap and cap.isOpened():
            cap.release()
        cv2.destroyAllWindows()
        
        # Save performance log
        log_path = os.path.join(run_dir, 'deployment_log.json')
        with open(log_path, 'w') as f:
            json.dump(performance_log, f, indent=2)
        
        # Calculate summary statistics
        if performance_log:
            avg_waiting_time = np.mean([entry['waiting_time'] for entry in performance_log])
            avg_queue_length = np.mean([entry['queue_length'] for entry in performance_log])
            avg_throughput = np.mean([entry['throughput'] for entry in performance_log])
            avg_reward = np.mean([entry['reward'] for entry in performance_log])
            
            logger.info("\nDeployment Summary:")
            logger.info(f"Total steps: {len(performance_log)}")
            logger.info(f"Average waiting time: {avg_waiting_time:.2f} seconds")
            logger.info(f"Average queue length: {avg_queue_length:.2f} vehicles")
            logger.info(f"Average throughput: {avg_throughput:.2f} vehicles")
            logger.info(f"Average reward: {avg_reward:.2f}")
            
            # Save model if improved
            if args.model:
                controller.save_model(os.path.join(run_dir, 'adaptive_controller_updated.pth'))
                logger.info(f"Updated model saved to {os.path.join(run_dir, 'adaptive_controller_updated.pth')}")
        
        logger.info(f"Deployment log saved to {log_path}")
        logger.info("Deployment complete")

def main():
    """Main function"""
    # Check if all required modules are imported
    if not all_modules_imported:
        logger.error("Cannot proceed: Some required modules could not be imported.")
        logger.error("Please run setup.py first to install dependencies.")
        logger.error("If using SUMO features, make sure SUMO is installed and SUMO_HOME environment variable is set.")
        sys.exit(1)
        
    # Parse command line arguments
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set up directories
    run_dir = setup_directories()
    logger.info(f"Results will be saved to {run_dir}")
    
    # Add new 'deploy' mode that uses the adaptive controller
    if args.mode == 'deploy':
        deploy_adaptive_controller(config, args, run_dir)
    
    # Run the appropriate mode
    if args.mode == 'train':
        train_agent(config, args, run_dir)
    elif args.mode == 'test':
        test_agent(config, args, run_dir)
    elif args.mode == 'demo':
        run_vision_demo(config, args, run_dir)
    else:
        logger.error(f"Unknown mode: {args.mode}")
        sys.exit(1)

if __name__ == "__main__":
    main()
