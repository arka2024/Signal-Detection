"""
Dashboard for Traffic Signal Management System
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import yaml
import os
import sys
import time
from datetime import datetime, timedelta

# Add parent directory to path to import project modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class TrafficDashboard:
    """
    Interactive dashboard for traffic signal monitoring and control
    using Streamlit
    """
    
    def __init__(self, config_file="config.yaml"):
        """Initialize the dashboard"""
        # Load configuration
        with open(config_file, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.dashboard_config = self.config['dashboard']
        self.metrics = self.dashboard_config['metrics']
        self.show_map = self.dashboard_config['map_view']
        self.allow_control = self.dashboard_config['signal_control']
        
        # Initialize session state for storing dashboard state
        if 'initialized' not in st.session_state:
            st.session_state.initialized = False
            st.session_state.simulation_running = False
            st.session_state.current_step = 0
            st.session_state.selected_intersection = None
            st.session_state.selected_scenario = "normal"
            st.session_state.metrics_history = {metric: [] for metric in self.metrics}
            st.session_state.timestamps = []
    
    def initialize_dashboard(self):
        """Set up the initial dashboard structure"""
        # Set page config
        st.set_page_config(
            page_title="Traffic Signal Management",
            page_icon="🚦",
            layout="wide"
        )
        
        # Set title and description
        st.title("🚦 Intelligent Traffic Signal Management System")
        st.markdown("""
        This dashboard provides real-time monitoring and control of traffic signals.
        Use the controls below to view traffic conditions and manage signal timings.
        """)
        
        # Create sidebar
        with st.sidebar:
            st.header("Control Panel")
            self.create_control_panel()
        
        # Mark as initialized
        st.session_state.initialized = True
    
    def create_control_panel(self):
        """Create the control panel in the sidebar"""
        # Intersection selection
        st.subheader("Intersection")
        intersections = ["Main St & Broadway", "Oak Rd & River Blvd", "Central Ave & Park St"]
        selected = st.selectbox("Select Intersection", intersections)
        if selected != st.session_state.selected_intersection:
            st.session_state.selected_intersection = selected
        
        # Scenario selection
        st.subheader("Traffic Scenario")
        scenarios = list(s["name"] for s in self.config["simulation"]["scenarios"])
        scenario = st.selectbox("Select Traffic Scenario", scenarios)
        if scenario != st.session_state.selected_scenario:
            st.session_state.selected_scenario = scenario
        
        # Simulation control
        st.subheader("Simulation Control")
        col1, col2 = st.columns(2)
        
        if not st.session_state.simulation_running:
            if col1.button("▶️ Start Simulation"):
                st.session_state.simulation_running = True
                st.session_state.current_step = 0
                st.session_state.metrics_history = {metric: [] for metric in self.metrics}
                st.session_state.timestamps = []
        else:
            if col1.button("⏸️ Pause Simulation"):
                st.session_state.simulation_running = False
        
        if col2.button("🔄 Reset"):
            st.session_state.simulation_running = False
            st.session_state.current_step = 0
            st.session_state.metrics_history = {metric: [] for metric in self.metrics}
            st.session_state.timestamps = []
        
        # Manual control (if enabled)
        if self.allow_control:
            st.subheader("Signal Control")
            st.write("Override automatic signal timing:")
            
            col1, col2 = st.columns(2)
            
            if col1.button("🟢 Set Green (N-S)"):
                st.toast("North-South direction set to GREEN")
            
            if col2.button("🟢 Set Green (E-W)"):
                st.toast("East-West direction set to GREEN")
            
            st.write("Emergency mode:")
            emergency_mode = st.checkbox("Enable emergency vehicle priority")
            if emergency_mode:
                direction = st.selectbox("Emergency vehicle approaching from:", 
                                        ["North", "South", "East", "West"])
                if st.button("🚨 Trigger Emergency Protocol"):
                    st.toast(f"Emergency vehicle priority set for {direction} approach")
    
    def mock_data_update(self):
        """Generate mock data for dashboard visualization"""
        if st.session_state.simulation_running:
            # Update step
            st.session_state.current_step += 1
            
            # Add timestamp
            current_time = datetime.now()
            st.session_state.timestamps.append(current_time)
            
            # Update metrics
            # Waiting time (seconds)
            waiting_time = 15 + 10 * np.sin(st.session_state.current_step / 10) + np.random.normal(0, 3)
            st.session_state.metrics_history["average_waiting_time"].append(max(0, waiting_time))
            
            # Queue length (vehicles)
            queue_length = 8 + 6 * np.sin(st.session_state.current_step / 15 + 1) + np.random.normal(0, 2)
            st.session_state.metrics_history["queue_length"].append(max(0, int(queue_length)))
            
            # Throughput (vehicles/minute)
            throughput = 20 + 15 * np.sin(st.session_state.current_step / 20 + 2) + np.random.normal(0, 3)
            st.session_state.metrics_history["throughput"].append(max(0, int(throughput)))
            
            # Congestion index (0-1)
            congestion = 0.3 + 0.2 * np.sin(st.session_state.current_step / 30) + np.random.normal(0, 0.05)
            st.session_state.metrics_history["congestion_index"].append(max(0, min(1, congestion)))
    
    def create_metrics_cards(self):
        """Create cards for displaying current metrics"""
        cols = st.columns(4)
        
        # Only show metrics if we have data
        if len(st.session_state.timestamps) > 0:
            # Average waiting time
            with cols[0]:
                value = st.session_state.metrics_history["average_waiting_time"][-1]
                st.metric(
                    label="Average Waiting Time",
                    value=f"{value:.1f} sec",
                    delta=f"{value - st.session_state.metrics_history['average_waiting_time'][-2]:.1f}" 
                    if len(st.session_state.metrics_history['average_waiting_time']) > 1 else None
                )
            
            # Queue length
            with cols[1]:
                value = st.session_state.metrics_history["queue_length"][-1]
                st.metric(
                    label="Queue Length",
                    value=f"{value} vehicles",
                    delta=f"{value - st.session_state.metrics_history['queue_length'][-2]}" 
                    if len(st.session_state.metrics_history['queue_length']) > 1 else None
                )
            
            # Throughput
            with cols[2]:
                value = st.session_state.metrics_history["throughput"][-1]
                st.metric(
                    label="Throughput",
                    value=f"{value} veh/min",
                    delta=f"{value - st.session_state.metrics_history['throughput'][-2]}" 
                    if len(st.session_state.metrics_history['throughput']) > 1 else None
                )
            
            # Congestion index
            with cols[3]:
                value = st.session_state.metrics_history["congestion_index"][-1]
                st.metric(
                    label="Congestion Index",
                    value=f"{value:.2f}",
                    delta=f"{value - st.session_state.metrics_history['congestion_index'][-2]:.2f}" 
                    if len(st.session_state.metrics_history['congestion_index']) > 1 else None,
                    delta_color="inverse"  # Lower is better for congestion
                )
    
    def create_time_series_charts(self):
        """Create time series charts for metrics"""
        if len(st.session_state.timestamps) > 0:
            # Create DataFrame from metrics history
            df = pd.DataFrame({
                'time': st.session_state.timestamps,
                'average_waiting_time': st.session_state.metrics_history['average_waiting_time'],
                'queue_length': st.session_state.metrics_history['queue_length'],
                'throughput': st.session_state.metrics_history['throughput'],
                'congestion_index': st.session_state.metrics_history['congestion_index']
            })
            
            # Create two rows of charts
            col1, col2 = st.columns(2)
            
            # Waiting Time chart
            with col1:
                fig1 = px.line(
                    df, x='time', y='average_waiting_time', 
                    title='Average Waiting Time Over Time',
                    labels={'average_waiting_time': 'Waiting Time (sec)', 'time': 'Time'}
                )
                fig1.update_layout(height=300)
                st.plotly_chart(fig1, use_container_width=True)
            
            # Queue Length chart
            with col2:
                fig2 = px.line(
                    df, x='time', y='queue_length', 
                    title='Queue Length Over Time',
                    labels={'queue_length': 'Vehicles', 'time': 'Time'}
                )
                fig2.update_layout(height=300)
                st.plotly_chart(fig2, use_container_width=True)
            
            # Throughput chart
            with col1:
                fig3 = px.line(
                    df, x='time', y='throughput', 
                    title='Intersection Throughput Over Time',
                    labels={'throughput': 'Vehicles/min', 'time': 'Time'}
                )
                fig3.update_layout(height=300)
                st.plotly_chart(fig3, use_container_width=True)
            
            # Congestion Index chart
            with col2:
                fig4 = px.line(
                    df, x='time', y='congestion_index', 
                    title='Congestion Index Over Time',
                    labels={'congestion_index': 'Index (0-1)', 'time': 'Time'}
                )
                fig4.update_layout(height=300)
                st.plotly_chart(fig4, use_container_width=True)
    
    def create_intersection_map(self):
        """Create a map view of the intersection"""
        if self.show_map:
            st.subheader(f"Traffic Map: {st.session_state.selected_intersection}")
            
            # This would be a real map in a production application
            # For this prototype, we'll use a simple visualization
            
            fig = go.Figure()
            
            # Define intersection
            # Horizontal road
            fig.add_shape(type="rect", x0=-5, y0=-0.5, x1=5, y1=0.5, 
                          line=dict(color="gray", width=2), fillcolor="gray")
            
            # Vertical road
            fig.add_shape(type="rect", x0=-0.5, y0=-5, x1=0.5, y1=5, 
                          line=dict(color="gray", width=2), fillcolor="gray")
            
            # Add traffic lights
            light_positions = [
                {"pos": (0.75, 1.0), "dir": "north"},
                {"pos": (1.0, -0.75), "dir": "east"},
                {"pos": (-0.75, -1.0), "dir": "south"},
                {"pos": (-1.0, 0.75), "dir": "west"}
            ]
            
            # Randomly determine which direction has green
            import random
            green_dirs = ["north-south", "east-west"]
            current_green = random.choice(green_dirs)
            
            for light in light_positions:
                color = "green" if (
                    (current_green == "north-south" and light["dir"] in ["north", "south"]) or
                    (current_green == "east-west" and light["dir"] in ["east", "west"])
                ) else "red"
                
                fig.add_shape(
                    type="circle", x0=light["pos"][0]-0.2, y0=light["pos"][1]-0.2, 
                    x1=light["pos"][0]+0.2, y1=light["pos"][1]+0.2,
                    line=dict(color=color, width=2), fillcolor=color
                )
            
            # Add some vehicles
            if st.session_state.simulation_running and len(st.session_state.timestamps) > 0:
                # Number of vehicles based on congestion
                congestion = st.session_state.metrics_history["congestion_index"][-1]
                num_vehicles = int(30 * congestion)
                
                # Add vehicles as scatter points
                vehicle_x = []
                vehicle_y = []
                
                # Directions with more weight on congested approaches
                directions = ["north", "south", "east", "west"]
                vehicle_counts = {
                    "north": int(num_vehicles * (0.1 + 0.3 * (current_green != "north-south"))),
                    "south": int(num_vehicles * (0.1 + 0.3 * (current_green != "north-south"))),
                    "east": int(num_vehicles * (0.1 + 0.3 * (current_green != "east-west"))),
                    "west": int(num_vehicles * (0.1 + 0.3 * (current_green != "east-west")))
                }
                
                for direction in directions:
                    count = vehicle_counts[direction]
                    
                    if direction == "north":
                        # Vehicles coming from south to north
                        x_vals = np.random.uniform(-0.3, 0.3, count)
                        # Stagger y positions from -5 to -1
                        y_vals = np.linspace(-5, -1, count) + np.random.uniform(-0.2, 0.2, count)
                        vehicle_x.extend(x_vals)
                        vehicle_y.extend(y_vals)
                    elif direction == "south":
                        # Vehicles coming from north to south
                        x_vals = np.random.uniform(-0.3, 0.3, count)
                        # Stagger y positions from 5 to 1
                        y_vals = np.linspace(5, 1, count) + np.random.uniform(-0.2, 0.2, count)
                        vehicle_x.extend(x_vals)
                        vehicle_y.extend(y_vals)
                    elif direction == "east":
                        # Vehicles coming from west to east
                        y_vals = np.random.uniform(-0.3, 0.3, count)
                        # Stagger x positions from -5 to -1
                        x_vals = np.linspace(-5, -1, count) + np.random.uniform(-0.2, 0.2, count)
                        vehicle_x.extend(x_vals)
                        vehicle_y.extend(y_vals)
                    elif direction == "west":
                        # Vehicles coming from east to west
                        y_vals = np.random.uniform(-0.3, 0.3, count)
                        # Stagger x positions from 5 to 1
                        x_vals = np.linspace(5, 1, count) + np.random.uniform(-0.2, 0.2, count)
                        vehicle_x.extend(x_vals)
                        vehicle_y.extend(y_vals)
                
                fig.add_trace(go.Scatter(
                    x=vehicle_x,
                    y=vehicle_y,
                    mode="markers",
                    marker=dict(size=8, color="blue"),
                    name="Vehicles"
                ))
            
            # Configure the layout
            fig.update_layout(
                height=500,
                xaxis=dict(range=[-6, 6], showgrid=False, zeroline=False, visible=False),
                yaxis=dict(range=[-6, 6], showgrid=False, zeroline=False, visible=False),
                showlegend=False,
                margin=dict(l=0, r=0, t=0, b=0)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Add map legend/info
            col1, col2 = st.columns(2)
            with col1:
                st.info(f"Current signal phase: {'North-South Green' if current_green == 'north-south' else 'East-West Green'}")
            with col2:
                if st.session_state.simulation_running and len(st.session_state.timestamps) > 0:
                    st.info(f"Vehicles on approach: {sum(vehicle_counts.values())}")
    
    def render_dashboard(self):
        """Render the complete dashboard"""
        # Initialize if not done yet
        if not st.session_state.initialized:
            self.initialize_dashboard()
        
        # Update mock data if simulation is running
        self.mock_data_update()
        
        # Create intersection map
        self.create_intersection_map()
        
        # Create metrics section
        st.subheader("Performance Metrics")
        self.create_metrics_cards()
        
        # Create charts
        st.subheader("Trend Analysis")
        self.create_time_series_charts()
        
        # Add information panel about the intelligent system
        with st.expander("🧠 About the AI Traffic Management System"):
            st.markdown("""
            This intelligent traffic management system uses reinforcement learning to optimize signal timing based on real-time traffic conditions.
            
            **Key Features:**
            * Adaptive signal timing based on traffic demand
            * Integration with ANPR cameras for vehicle detection and counting
            * Special handling for emergency vehicles and unusual traffic patterns
            * Performance monitoring and optimization
            
            **How It Works:**
            1. Traffic cameras capture real-time data about vehicle density and flow
            2. The AI model processes this data and predicts optimal signal timing
            3. Traffic signals are adjusted to minimize waiting time and congestion
            4. The system continuously learns and improves through reinforcement learning
            """)
        
        # Add stats for the simulation
        if st.session_state.simulation_running and len(st.session_state.timestamps) > 0:
            st.sidebar.markdown("---")
            st.sidebar.subheader("Simulation Stats")
            st.sidebar.info(f"Steps: {st.session_state.current_step}")
            
            if len(st.session_state.timestamps) > 1:
                avg_waiting = np.mean(st.session_state.metrics_history["average_waiting_time"])
                st.sidebar.info(f"Avg. Waiting Time: {avg_waiting:.2f} sec")
                
                avg_congestion = np.mean(st.session_state.metrics_history["congestion_index"])
                st.sidebar.info(f"Avg. Congestion: {avg_congestion:.2f}")
                
                # Calculate improvement over baseline (just an example)
                if st.session_state.current_step > 30:
                    early_waiting = np.mean(st.session_state.metrics_history["average_waiting_time"][:10])
                    recent_waiting = np.mean(st.session_state.metrics_history["average_waiting_time"][-10:])
                    improvement = ((early_waiting - recent_waiting) / early_waiting) * 100
                    
                    if improvement > 0:
                        st.sidebar.success(f"Improvement: {improvement:.1f}% reduction in waiting time")
                    else:
                        st.sidebar.warning(f"Change: {improvement:.1f}% in waiting time")

def main():
    """Main function to run the dashboard"""
    dashboard = TrafficDashboard()
    dashboard.render_dashboard()
    
    # Auto-refresh the app every few seconds when simulation is running
    if st.session_state.simulation_running:
        time.sleep(1)
        st.rerun()

if __name__ == "__main__":
    main()
