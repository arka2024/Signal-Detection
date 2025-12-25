#!/usr/bin/env python
"""
Traffic Signal Performance Analysis Tool
This script analyzes traffic data and signal performance to generate insights.
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

def load_traffic_data(file_path):
    """Load and parse traffic data from JSON file."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        return pd.DataFrame(data)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading traffic data: {e}")
        sys.exit(1)

def calculate_metrics(df):
    """Calculate key performance metrics from traffic data."""
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Add hour and day columns for analysis
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    
    # Group by intersection and calculate metrics
    metrics = {}
    
    for intersection in df['intersection'].unique():
        int_df = df[df['intersection'] == intersection]
        
        # Average metrics
        avg_speed = int_df['avg_speed'].mean()
        avg_queue = int_df['queue_length'].mean()
        avg_wait = int_df['wait_time'].mean()
        
        # Peak metrics
        peak_hours = int_df.groupby('hour')['vehicle_count'].mean().sort_values(ascending=False).head(3).index.tolist()
        peak_queue = int_df.groupby('hour')['queue_length'].mean().sort_values(ascending=False).head(1).index.tolist()[0]
        peak_queue_length = int_df[int_df['hour'] == peak_queue]['queue_length'].mean()
        
        # Directional imbalance (ratio of north-south to east-west traffic)
        ns_traffic = int_df[int_df['direction'] == 'north_south']['vehicle_count'].mean()
        ew_traffic = int_df[int_df['direction'] == 'east_west']['vehicle_count'].mean()
        direction_ratio = ns_traffic / ew_traffic if ew_traffic > 0 else float('inf')
        
        # Store metrics
        metrics[intersection] = {
            "avg_speed": round(avg_speed, 2),
            "avg_queue_length": round(avg_queue, 2),
            "avg_wait_time": round(avg_wait, 2),
            "peak_hours": peak_hours,
            "max_avg_queue": round(peak_queue_length, 2),
            "max_queue_hour": peak_queue,
            "direction_ratio": round(direction_ratio, 2),
            "ns_traffic": round(ns_traffic, 2),
            "ew_traffic": round(ew_traffic, 2)
        }
        
    return metrics

def analyze_signal_efficiency(df, metrics):
    """Analyze how efficient the signals are based on wait times and queue lengths."""
    results = {}
    
    for intersection in metrics.keys():
        int_df = df[df['intersection'] == intersection]
        
        # Calculate signal efficiency score (lower wait times and queue lengths = better)
        # Normalize by traffic volume
        total_traffic = int_df['vehicle_count'].mean()
        avg_wait = metrics[intersection]['avg_wait_time']
        avg_queue = metrics[intersection]['avg_queue_length']
        
        # Signal efficiency score (0-100, higher is better)
        # Formula: 100 - (normalized wait time + normalized queue length)
        wait_factor = min(50, (avg_wait / total_traffic) * 25)
        queue_factor = min(50, (avg_queue / total_traffic) * 50)
        
        efficiency_score = 100 - wait_factor - queue_factor
        efficiency_score = max(0, min(100, efficiency_score))
        
        # Analyze daily patterns
        daily_patterns = int_df.groupby('day_of_week')['vehicle_count'].mean().to_dict()
        weekday_avg = np.mean([daily_patterns.get(i, 0) for i in range(5)])  # Mon-Fri
        weekend_avg = np.mean([daily_patterns.get(i, 0) for i in range(5, 7)])  # Sat-Sun
        weekend_ratio = weekend_avg / weekday_avg if weekday_avg > 0 else 0
        
        results[intersection] = {
            "efficiency_score": round(efficiency_score, 1),
            "weekday_avg_traffic": round(weekday_avg, 2),
            "weekend_avg_traffic": round(weekend_avg, 2),
            "weekend_weekday_ratio": round(weekend_ratio, 2)
        }
    
    return results

def plot_traffic_patterns(df, output_dir):
    """Generate plots of traffic patterns and save to files."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot 1: Hourly traffic patterns by intersection
    plt.figure(figsize=(12, 8))
    
    for intersection in df['intersection'].unique():
        int_df = df[df['intersection'] == intersection]
        hourly_avg = int_df.groupby('hour')['vehicle_count'].mean()
        plt.plot(hourly_avg.index, hourly_avg.values, marker='o', label=intersection)
    
    plt.xlabel('Hour of Day')
    plt.ylabel('Average Vehicle Count')
    plt.title('Hourly Traffic Patterns by Intersection')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'hourly_traffic_patterns.png'))
    
    # Plot 2: Wait time vs. Queue Length by Direction
    plt.figure(figsize=(12, 8))
    
    for direction in df['direction'].unique():
        dir_df = df[df['direction'] == direction]
        plt.scatter(
            dir_df['queue_length'], 
            dir_df['wait_time'],
            alpha=0.5,
            label=direction
        )
    
    plt.xlabel('Queue Length')
    plt.ylabel('Wait Time (seconds)')
    plt.title('Relationship Between Queue Length and Wait Time')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'queue_vs_wait.png'))
    
    # Plot 3: Traffic volume by day of week
    plt.figure(figsize=(12, 8))
    
    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    daily_avg = df.groupby('day_of_week')['vehicle_count'].mean()
    
    plt.bar([day_names[i] for i in daily_avg.index], daily_avg.values)
    plt.xlabel('Day of Week')
    plt.ylabel('Average Vehicle Count')
    plt.title('Traffic Volume by Day of Week')
    plt.grid(True, axis='y')
    plt.savefig(os.path.join(output_dir, 'daily_traffic_volume.png'))
    
    print(f"Generated plots saved to {output_dir}")

def generate_recommendations(metrics, efficiency_results):
    """Generate traffic signal optimization recommendations."""
    recommendations = {}
    
    for intersection, data in metrics.items():
        recommendations[intersection] = []
        
        # Check direction imbalance
        if data['direction_ratio'] > 1.5 or data['direction_ratio'] < 0.67:
            major_dir = "north-south" if data['direction_ratio'] > 1 else "east-west"
            recommendations[intersection].append(
                f"Significant traffic imbalance detected: {major_dir} has {abs(round((data['direction_ratio'] - 1) * 100))}% "
                f"more traffic. Consider adjusting signal timing to favor {major_dir} direction."
            )
        
        # Check efficiency score
        eff_score = efficiency_results[intersection]['efficiency_score']
        if eff_score < 60:
            recommendations[intersection].append(
                f"Low efficiency score ({eff_score}/100). Consider optimizing signal timing "
                f"to reduce average wait time of {data['avg_wait_time']} seconds."
            )
        
        # Check peak hours
        recommendations[intersection].append(
            f"Peak traffic hours detected at {', '.join(map(str, data['peak_hours']))}:00. "
            f"Consider implementing rush hour signal timing plans during these periods."
        )
        
        # Check weekend vs weekday patterns
        weekend_ratio = efficiency_results[intersection]['weekend_weekday_ratio']
        if weekend_ratio < 0.6:
            recommendations[intersection].append(
                f"Weekend traffic is {round((1-weekend_ratio)*100)}% lower than weekdays. "
                f"Consider different signal timing plans for weekends."
            )
        
    return recommendations

def main():
    parser = argparse.ArgumentParser(description="Analyze traffic signal performance")
    parser.add_argument("--data", default="data/traffic_patterns/traffic_data.json", 
                        help="Path to traffic data JSON file")
    parser.add_argument("--output", default="analysis_results",
                        help="Directory to save analysis results and plots")
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading traffic data from {args.data}...")
    df = load_traffic_data(args.data)
    print(f"Loaded {len(df)} traffic data points across {df['intersection'].nunique()} intersections.")
    
    # Calculate metrics
    print("Calculating traffic metrics...")
    metrics = calculate_metrics(df)
    
    # Analyze signal efficiency
    print("Analyzing signal efficiency...")
    efficiency_results = analyze_signal_efficiency(df, metrics)
    
    # Generate plots
    print("Generating traffic pattern plots...")
    plot_traffic_patterns(df, args.output)
    
    # Generate recommendations
    print("Generating optimization recommendations...")
    recommendations = generate_recommendations(metrics, efficiency_results)
    
    # Save results to JSON
    results = {
        "analysis_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "metrics": metrics,
        "efficiency_results": efficiency_results,
        "recommendations": recommendations
    }
    
    os.makedirs(args.output, exist_ok=True)
    with open(os.path.join(args.output, "analysis_results.json"), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\n=== Traffic Signal Performance Analysis Summary ===")
    for intersection, data in efficiency_results.items():
        print(f"\nIntersection: {intersection}")
        print(f"Efficiency Score: {data['efficiency_score']}/100")
        print("Recommendations:")
        for i, rec in enumerate(recommendations[intersection], 1):
            print(f"  {i}. {rec}")
    
    print(f"\nDetailed results saved to {os.path.join(args.output, 'analysis_results.json')}")

if __name__ == "__main__":
    main()
