#!/usr/bin/env python
"""
Sample data generation script for traffic data.
This script creates realistic traffic data for development and testing
without requiring SUMO installation.
"""

import os
import sys
import argparse
import numpy as np
import json
import random
from datetime import datetime, timedelta

def generate_traffic_patterns(output_file, days=7, intersections=3):
    """
    Generate realistic traffic pattern data for multiple intersections.
    
    Args:
        output_file (str): Path to save the generated data
        days (int): Number of days to generate data for
        intersections (int): Number of intersections to simulate
    """
    print(f"Generating traffic data for {days} days across {intersections} intersections...")
    
    # Intersection names
    intersection_names = [
        "Main St & Broadway",
        "5th Ave & Park Rd",
        "University Blvd & Market St",
        "Highland Ave & Sunset Blvd",
        "Oak Lane & River Rd"
    ][:intersections]
    
    # Directions for each intersection
    directions = ["north_south", "east_west"]
    
    # Traffic patterns (hour of day, multiplier)
    # These create realistic rush hours and daily patterns
    hourly_patterns = {
        0: 0.2,  # Midnight
        1: 0.15,
        2: 0.1,
        3: 0.1,
        4: 0.15,
        5: 0.3,
        6: 0.7,
        7: 1.5,  # Morning rush begins
        8: 2.0,  # Peak morning rush
        9: 1.7,
        10: 1.2,
        11: 1.0,
        12: 1.3,  # Lunch hour
        13: 1.1,
        14: 1.0,
        15: 1.2,
        16: 1.5,  # Afternoon rush begins
        17: 2.2,  # Peak evening rush
        18: 2.0,
        19: 1.5,
        20: 1.0,
        21: 0.8,
        22: 0.5,
        23: 0.3
    }
    
    # Day of week patterns (0=Monday, 6=Sunday)
    day_patterns = {
        0: 1.0,  # Monday
        1: 1.0,
        2: 1.0,
        3: 1.05,
        4: 1.1,  # Friday slightly busier
        5: 0.7,  # Saturday less commuting
        6: 0.5   # Sunday least traffic
    }
    
    # Base traffic parameters for each intersection
    intersection_params = []
    for i in range(intersections):
        # Each intersection has different base traffic levels
        intersection_params.append({
            "name": intersection_names[i],
            "north_south": {
                "base_flow": 8 + random.randint(-3, 3),
                "variance": 2.5,
            },
            "east_west": {
                "base_flow": 7 + random.randint(-2, 4),
                "variance": 2.0,
            }
        })
    
    # Generate data
    start_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    data = []
    
    # Sample interval in minutes
    interval = 5
    samples_per_day = 24 * (60 // interval)
    
    for day in range(days):
        current_date = start_date + timedelta(days=day)
        day_of_week = current_date.weekday()
        day_factor = day_patterns[day_of_week]
        
        for interval_idx in range(samples_per_day):
            minutes = interval_idx * interval
            hour = minutes // 60
            minute = minutes % 60
            
            timestamp = current_date.replace(hour=hour, minute=minute)
            hour_factor = hourly_patterns[hour]
            
            # Special events (randomly occurring)
            special_event = False
            special_event_factor = 1.0
            if random.random() < 0.02:  # 2% chance of special event
                special_event = True
                special_event_factor = random.uniform(1.2, 1.8)
            
            # Weather effects (randomly occurring)
            weather_factor = 1.0
            weather_condition = "clear"
            if random.random() < 0.1:  # 10% chance of bad weather
                weather_factor = random.uniform(0.7, 0.9)
                weather_condition = random.choice(["rain", "snow", "fog"])
            
            # Generate data for each intersection
            for i, intersection in enumerate(intersection_params):
                for direction in directions:
                    # Calculate traffic flow
                    base_flow = intersection[direction]["base_flow"]
                    variance = intersection[direction]["variance"]
                    
                    # Apply all factors
                    mean_flow = base_flow * hour_factor * day_factor * special_event_factor * weather_factor
                    
                    # Add some randomness
                    flow = max(1, np.random.normal(mean_flow, variance))
                    
                    # Calculate derived metrics
                    avg_speed = max(5, min(60, 55 - (flow * 0.7) + random.uniform(-3, 3)))
                    queue_length = max(0, min(20, int((flow * 0.3) + random.randint(-1, 2))))
                    wait_time = max(0, min(180, int(queue_length * 7 + random.randint(-5, 10))))
                    
                    # Record data point
                    data_point = {
                        "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                        "intersection": intersection["name"],
                        "direction": direction,
                        "vehicle_count": int(flow),
                        "avg_speed": round(avg_speed, 1),
                        "queue_length": queue_length,
                        "wait_time": wait_time,
                        "weather": weather_condition
                    }
                    
                    if special_event:
                        data_point["special_event"] = True
                    
                    data.append(data_point)
    
    # Save to file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Generated {len(data)} traffic data points saved to {output_file}")


def generate_vehicle_traces(output_file, num_vehicles=100, grid_size=5):
    """
    Generate vehicle movement traces through a traffic grid.
    
    Args:
        output_file (str): Path to save the generated data
        num_vehicles (int): Number of vehicles to simulate
        grid_size (int): Size of the traffic grid (grid_size x grid_size)
    """
    print(f"Generating movement traces for {num_vehicles} vehicles...")
    
    # Define the grid - each entry is an intersection ID
    grid = [[f"{x}_{y}" for y in range(grid_size)] for x in range(grid_size)]
    
    # Generate vehicle traces
    traces = []
    
    for vehicle_id in range(1, num_vehicles + 1):
        # Random start and end points
        start_x, start_y = random.randint(0, grid_size-1), random.randint(0, grid_size-1)
        end_x, end_y = random.randint(0, grid_size-1), random.randint(0, grid_size-1)
        
        # Make sure start and end are different
        while (start_x, start_y) == (end_x, end_y):
            end_x, end_y = random.randint(0, grid_size-1), random.randint(0, grid_size-1)
        
        # Generate a path (simple implementation - just moves toward destination)
        path = []
        current_x, current_y = start_x, start_y
        
        # Starting time (random within a day)
        start_time = datetime.now().replace(
            hour=random.randint(0, 23),
            minute=random.randint(0, 59),
            second=0,
            microsecond=0
        )
        current_time = start_time
        
        # Create the path
        while (current_x, current_y) != (end_x, end_y):
            path.append({
                "intersection": grid[current_x][current_y],
                "timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S"),
            })
            
            # Move toward destination (simple pathfinding)
            if current_x < end_x:
                current_x += 1
            elif current_x > end_x:
                current_x -= 1
            elif current_y < end_y:
                current_y += 1
            elif current_y > end_y:
                current_y -= 1
                
            # Update time (random interval between 1-3 minutes)
            current_time += timedelta(minutes=random.randint(1, 3))
        
        # Add final destination
        path.append({
            "intersection": grid[end_x][end_y],
            "timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S"),
        })
        
        # Add to traces
        traces.append({
            "vehicle_id": f"V{vehicle_id:04d}",
            "start_time": start_time.strftime("%Y-%m-%d %H:%M:%S"),
            "end_time": current_time.strftime("%Y-%m-%d %H:%M:%S"),
            "path": path
        })
    
    # Save to file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(traces, f, indent=2)
    
    print(f"Generated traces for {num_vehicles} vehicles saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate sample traffic data for development")
    parser.add_argument("--traffic-patterns", action="store_true", help="Generate traffic pattern data")
    parser.add_argument("--vehicle-traces", action="store_true", help="Generate vehicle movement traces")
    parser.add_argument("--days", type=int, default=7, help="Number of days to generate data for")
    parser.add_argument("--intersections", type=int, default=3, help="Number of intersections to simulate")
    parser.add_argument("--vehicles", type=int, default=100, help="Number of vehicles to simulate for traces")
    
    args = parser.parse_args()
    
    # If no specific action is requested, generate all data
    if not (args.traffic_patterns or args.vehicle_traces):
        args.traffic_patterns = True
        args.vehicle_traces = True
    
    if args.traffic_patterns:
        output_file = "data/traffic_patterns/traffic_data.json"
        generate_traffic_patterns(output_file, args.days, args.intersections)
    
    if args.vehicle_traces:
        output_file = "data/traffic_patterns/vehicle_traces.json"
        generate_vehicle_traces(output_file, args.vehicles)
        
    print("Sample data generation complete!")
