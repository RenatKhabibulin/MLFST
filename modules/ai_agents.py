import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

from visualization import apply_firesafety_theme_to_plotly, FIRESAFETY_COLORS, FIRESAFETY_SEQUENTIAL, FIRESAFETY_CATEGORICAL
from utils import create_code_editor, create_module_quiz

def show_module_content():
    """Display AI Agents content for fire safety applications"""
    
    st.title("AI Agents in Fire Safety")
    
    # Introduction
    st.markdown("""
    ## Introduction to AI Agents
    
    AI Agents represent an advanced type of artificial intelligence system that can perceive its environment, 
    make decisions, and take actions to achieve specific goals. Unlike traditional AI models that perform 
    specific tasks in isolation, AI agents are designed to operate autonomously, adapt to changing conditions,
    and interact with their environment in a goal-directed manner.
    """)
    
    # What are AI Agents - with visual diagrams
    st.markdown("## What Are AI Agents?")
    
    # Create AI Agent architecture diagram
    fig = go.Figure()
    
    # Define the components
    components = ["Environment", "Sensors", "AI Agent Core", "Knowledge Base", "Decision Module", "Actuators"]
    y_positions = [0, 1, 0, -1, 0, 1]
    x_positions = [0, 1, 2, 2, 3, 4]
    colors = [FIRESAFETY_COLORS["light_text"], FIRESAFETY_COLORS["secondary"], 
              FIRESAFETY_COLORS["primary"], FIRESAFETY_COLORS["tertiary"], 
              FIRESAFETY_COLORS["primary"], FIRESAFETY_COLORS["secondary"]]
    
    # Add nodes
    for i, (comp, x, y, color) in enumerate(zip(components, x_positions, y_positions, colors)):
        fig.add_trace(go.Scatter(
            x=[x], y=[y],
            mode='markers+text',
            marker=dict(size=30, color=color, line=dict(width=2, color='white')),
            text=[comp],
            textposition="bottom center" if i in [0, 2, 4] else "top center",
            textfont=dict(size=12, color='white' if i in [2, 3, 4] else 'black'),
            name=comp,
            hoverinfo='name'
        ))
    
    # Add arrows/connections
    arrows = [
        (0, 1), # Environment to Sensors
        (1, 2), # Sensors to AI Agent Core
        (2, 4), # AI Agent Core to Decision Module
        (3, 2), # Knowledge Base to AI Agent Core
        (4, 5), # Decision Module to Actuators
        (5, 0)  # Actuators to Environment (completing the loop)
    ]
    
    for start, end in arrows:
        x_start, y_start = x_positions[start], y_positions[start]
        x_end, y_end = x_positions[end], y_positions[end]
        
        # Add arrow
        fig.add_shape(
            type="line",
            x0=x_start, y0=y_start,
            x1=x_end, y1=y_end,
            line=dict(color=FIRESAFETY_COLORS["text"], width=2),
            layer="below"
        )
        
        # Add arrowhead
        # Simple arrowhead by calculating midpoint and adding a small marker
        x_mid = (x_start + x_end) / 2
        y_mid = (y_start + y_end) / 2
        
        fig.add_trace(go.Scatter(
            x=[x_mid], y=[y_mid],
            mode='markers',
            marker=dict(
                symbol='triangle-right',
                size=10,
                color=FIRESAFETY_COLORS["text"],
                angle= -np.degrees(np.arctan2(y_end-y_start, x_end-x_start))
            ),
            showlegend=False,
            hoverinfo='none'
        ))
    
    # Styling
    fig = apply_firesafety_theme_to_plotly(
        fig, 
        title='AI Agent Architecture',
        height=400
    )
    
    fig.update_layout(
        showlegend=False,
        xaxis=dict(
            showticklabels=False,
            showgrid=False,
            zeroline=False,
            range=[-0.5, 4.5]
        ),
        yaxis=dict(
            showticklabels=False,
            showgrid=False,
            zeroline=False,
            range=[-1.5, 1.5]
        ),
        annotations=[
            dict(
                x=2.5, y=-1.4,
                xref="x", yref="y",
                text="AI Agent Perception-Action Cycle",
                showarrow=False,
                font=dict(
                    family="Arial",
                    size=14,
                    color=FIRESAFETY_COLORS["text"]
                )
            )
        ]
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Key Characteristics of AI Agents:**
        
        - **Autonomy**: Operate without direct human intervention
        - **Reactivity**: Respond to changes in the environment
        - **Proactivity**: Take initiative to achieve goals
        - **Social ability**: Interact with other agents and humans
        - **Learning capability**: Improve performance over time
        """)
        
    with col2:
        st.markdown("""
        **Components of AI Agents:**
        
        - **Sensors**: Collect data from the environment
        - **Knowledge Base**: Store and manage information
        - **Decision-making mechanisms**: Process information and choose actions
        - **Actuators**: Execute actions that affect the environment
        - **Learning modules**: Adapt and improve based on experience
        """)
    
    # Types of AI Agents
    st.markdown("## Types of AI Agents")
    
    # Simple vs. Learning Agents visualization
    agent_types_data = pd.DataFrame({
        'Agent Type': ['Simple Reflex', 'Model-Based', 'Goal-Based', 'Utility-Based', 'Learning'],
        'Complexity': [2, 4, 6, 8, 10],
        'Adaptability': [1, 3, 5, 7, 9],
        'Memory Required': [1, 4, 6, 7, 9],
        'Effectiveness': [3, 5, 7, 8, 9]
    })
    
    # Create radar chart
    categories = ['Complexity', 'Adaptability', 'Memory Required', 'Effectiveness']
    fig = go.Figure()
    
    for i, agent in enumerate(agent_types_data['Agent Type']):
        values = agent_types_data.iloc[i][categories].tolist()
        values.append(values[0])  # Close the loop
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories + [categories[0]],  # Close the loop
            fill='toself',
            name=agent,
            line_color=FIRESAFETY_CATEGORICAL[i % len(FIRESAFETY_CATEGORICAL)],
            opacity=0.8
        ))
    
    fig = apply_firesafety_theme_to_plotly(
        fig, 
        title='Comparison of AI Agent Types',
        height=500,
        legend_title='Agent Type'
    )
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 10]
            )
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""
    ### Types of AI Agents in Machine Learning
    
    1. **Simple Reflex Agents**: Act based on current perceptions, ignoring history. They use condition-action rules to respond to environmental inputs.
    
    2. **Model-Based Agents**: Maintain an internal model of the world, tracking aspects not visible in the current state and how the world evolves.
    
    3. **Goal-Based Agents**: Make decisions based on both the current state and a specific goal they aim to achieve, considering future implications.
    
    4. **Utility-Based Agents**: Evaluate actions based on a "utility function" that quantifies the desirability of different states.
    
    5. **Learning Agents**: Can improve their performance over time by learning from experience, adapting their behavior to better achieve their goals.
    """)
    
    # AI Agents in Fire Safety
    st.markdown("## Applications of AI Agents in Fire Safety")
    
    # Create application areas
    app_areas = st.tabs([
        "Early Detection & Prevention", 
        "Emergency Response", 
        "Building Management",
        "Post-Incident Analysis"
    ])
    
    with app_areas[0]:
        st.markdown("""
        ### Early Detection & Prevention
        
        AI agents can monitor buildings and environments to detect fire risks before they become active threats:
        
        - **Multi-Sensor Monitoring**: Integrate data from temperature, smoke, infrared, and gas sensors
        - **Pattern Recognition**: Identify unusual patterns that might indicate fire risk
        - **Predictive Maintenance**: Monitor fire safety equipment and predict failures before they occur
        - **Risk Assessment**: Continuously evaluate building areas for potential fire hazards
        
        #### Example System: Smart Fire Prevention Agent
        
        This system deploys an interconnected network of sensors monitored by an AI agent that:
        - Learns normal environmental patterns for different building areas
        - Alerts when patterns deviate from normal behavior
        - Recommends preventive actions based on detected anomalies
        - Integrates weather data to adjust fire risk calculations during high-risk conditions
        """)
        
        # Create a simple visualization of a fire detection agent
        st.image("https://raw.githubusercontent.com/replit/example-images/main/fire-detection-agent.jpg", 
                caption="Example of an AI agent monitoring multiple sensor feeds for fire detection")
        
    with app_areas[1]:
        st.markdown("""
        ### Emergency Response
        
        During an active fire emergency, AI agents can coordinate response efforts:
        
        - **Dynamic Evacuation Planning**: Calculate and update optimal evacuation routes based on fire spread
        - **Resource Allocation**: Direct fire suppression systems and recommend resource deployment
        - **Decision Support**: Provide situational awareness and decision options to firefighters
        - **Communication Management**: Coordinate communication between responders and affected people
        
        #### Example System: Evacuation Guidance Agent
        
        This agent system operates during emergencies to:
        - Map fire spread in real-time using sensor data and predictive modeling
        - Calculate personalized evacuation routes for building occupants
        - Communicate via mobile devices and building systems
        - Adjust strategies as conditions change
        """)
        
        # Evacuation route visualization
        fig = go.Figure()
        
        # Create a grid representing a building floor plan
        grid_size = 10
        walls = [(0,i) for i in range(grid_size)] + [(grid_size-1,i) for i in range(grid_size)]  # Outer walls
        walls += [(i,0) for i in range(grid_size)] + [(i,grid_size-1) for i in range(grid_size)]
        walls += [(3,i) for i in range(1,7)] + [(7,i) for i in range(3,9)]  # Inner walls
        
        exits = [(0,5), (grid_size-1,5)]  # Exit locations
        fire = [(2,2), (3,2), (4,3)]  # Fire location
        people = [(8,8), (5,6), (2,8)]  # People locations
        
        # Evacuation routes (pre-calculated for this example)
        routes = [
            [(8,8), (8,7), (8,6), (8,5), (8,4), (8,3), (9,3), (9,4), (9,5)],  # Person 1
            [(5,6), (5,5), (5,4), (5,3), (5,2), (4,2), (4,1), (3,1), (2,1), (1,1), (0,1)],  # Person 2
            [(2,8), (1,8), (1,7), (1,6), (1,5), (0,5)]  # Person 3
        ]
        
        # Add background
        x_bg = []
        y_bg = []
        color_bg = []
        
        for i in range(grid_size):
            for j in range(grid_size):
                pos = (i,j)
                x_bg.append(i)
                y_bg.append(j)
                
                if pos in walls:
                    color_bg.append('gray')
                elif pos in exits:
                    color_bg.append('green')
                elif pos in fire:
                    color_bg.append('red')
                else:
                    color_bg.append('lightblue')
        
        fig.add_trace(go.Scatter(
            x=x_bg, y=y_bg,
            mode='markers',
            marker=dict(
                size=25,
                color=color_bg,
                symbol='square',
                line=dict(width=1, color='white')
            ),
            showlegend=False,
            hoverinfo='none'
        ))
        
        # Add people
        fig.add_trace(go.Scatter(
            x=[p[0] for p in people],
            y=[p[1] for p in people],
            mode='markers',
            marker=dict(
                size=15,
                color='blue',
                symbol='circle',
                line=dict(width=1, color='white')
            ),
            name='Building Occupants',
            hoverinfo='name'
        ))
        
        # Add routes
        for i, route in enumerate(routes):
            fig.add_trace(go.Scatter(
                x=[p[0] for p in route],
                y=[p[1] for p in route],
                mode='lines',
                line=dict(
                    width=3, 
                    color=FIRESAFETY_CATEGORICAL[i % len(FIRESAFETY_CATEGORICAL)],
                    dash='dash'
                ),
                name=f'Evacuation Path {i+1}',
                hoverinfo='name'
            ))
        
        fig = apply_firesafety_theme_to_plotly(
            fig, 
            title='AI Agent Dynamic Evacuation Planning',
            height=500
        )
        
        fig.update_layout(
            xaxis=dict(
                showticklabels=False,
                showgrid=False,
                zeroline=False,
                range=[-0.5, grid_size-0.5]
            ),
            yaxis=dict(
                showticklabels=False,
                showgrid=False,
                zeroline=False,
                range=[-0.5, grid_size-0.5]
            ),
            annotations=[
                dict(
                    x=exits[0][0], y=exits[0][1],
                    xref="x", yref="y",
                    text="Exit A",
                    showarrow=True,
                    arrowhead=2,
                    ax=-30, ay=-30
                ),
                dict(
                    x=exits[1][0], y=exits[1][1],
                    xref="x", yref="y",
                    text="Exit B",
                    showarrow=True,
                    arrowhead=2,
                    ax=30, ay=-30
                )
            ]
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    with app_areas[2]:
        st.markdown("""
        ### Building Management
        
        AI agents can enhance fire safety in day-to-day building operations:
        
        - **Adaptive Monitoring**: Focus monitoring resources based on occupancy and risk levels
        - **Compliance Assistance**: Ensure building operations meet fire safety regulations
        - **Traffic Flow Optimization**: Manage people flow to reduce congestion risks
        - **Integration with Smart Building Systems**: Coordinate with HVAC, security, and other systems
        
        #### Example System: Integrated Building Safety Agent
        
        This management system:
        - Creates a digital twin of the building for simulation and testing
        - Monitors occupancy patterns and adjusts safety parameters accordingly
        - Identifies potential regulatory compliance issues before inspections
        - Optimizes energy use while maintaining fire safety standards
        """)
        
        # Create a visualization of building management using a heatmap
        fig = go.Figure()
        
        # Create a grid representation of a building floor
        grid_size = 15
        np.random.seed(42)
        
        # Create occupancy data
        occupancy = np.zeros((grid_size, grid_size))
        occupancy[3:7, 3:7] = np.random.uniform(0.7, 0.9, (4, 4))  # Meeting room
        occupancy[9:12, 2:6] = np.random.uniform(0.5, 0.8, (3, 4))  # Office area
        occupancy[2:5, 9:13] = np.random.uniform(0.6, 0.7, (3, 4))  # Kitchen
        occupancy[7:13, 8:13] = np.random.uniform(0.3, 0.5, (6, 5))  # Open plan
        
        # Create risk data - higher in kitchen, server room
        risk = np.zeros((grid_size, grid_size))
        risk[2:5, 9:13] = np.random.uniform(0.7, 0.9, (3, 4))  # Kitchen - high risk
        risk[9:11, 11:14] = np.random.uniform(0.6, 0.8, (2, 3))  # Server room
        risk[3:7, 3:7] = np.random.uniform(0.3, 0.4, (4, 4))  # Meeting room
        risk[9:12, 2:6] = np.random.uniform(0.2, 0.3, (3, 4))  # Office area
        
        # Create tabs for different views
        view_tabs = st.tabs(["Occupancy Heatmap", "Risk Assessment"])
        
        with view_tabs[0]:
            fig = go.Figure(data=go.Heatmap(
                z=occupancy,
                colorscale='YlOrRd',
                showscale=True,
                colorbar=dict(title="Occupancy Level")
            ))
            
            fig = apply_firesafety_theme_to_plotly(
                fig, 
                title='Real-time Building Occupancy Monitoring',
                height=500
            )
            
            # Add annotations for areas
            annotations = [
                dict(x=5, y=5, text="Meeting Room", showarrow=False, font=dict(color="black")),
                dict(x=10.5, y=4, text="Office Area", showarrow=False, font=dict(color="black")),
                dict(x=3.5, y=11, text="Kitchen", showarrow=False, font=dict(color="black")),
                dict(x=10, y=10, text="Open Plan", showarrow=False, font=dict(color="black")),
                dict(x=10, y=12.5, text="Server Room", showarrow=False, font=dict(color="black"))
            ]
            
            fig.update_layout(annotations=annotations)
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("""
            The AI agent continuously monitors occupancy levels throughout the building, allowing it to:
            - Identify unusually high-traffic areas that might need additional safety measures
            - Detect unauthorized access to restricted areas
            - Optimize evacuation planning based on real-time occupancy
            - Balance ventilation and other building systems according to actual usage
            """)
            
        with view_tabs[1]:
            fig = go.Figure(data=go.Heatmap(
                z=risk,
                colorscale='RdYlGn_r',  # Reversed so red is high risk
                showscale=True,
                colorbar=dict(title="Risk Level")
            ))
            
            fig = apply_firesafety_theme_to_plotly(
                fig, 
                title='AI Agent Fire Risk Assessment',
                height=500
            )
            
            # Add annotations for areas
            annotations = [
                dict(x=5, y=5, text="Meeting Room", showarrow=False, font=dict(color="black")),
                dict(x=10.5, y=4, text="Office Area", showarrow=False, font=dict(color="black")),
                dict(x=3.5, y=11, text="Kitchen", showarrow=False, font=dict(color="black")),
                dict(x=10, y=10, text="Open Plan", showarrow=False, font=dict(color="black")),
                dict(x=10, y=12.5, text="Server Room", showarrow=False, font=dict(color="white"))
            ]
            
            fig.update_layout(annotations=annotations)
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("""
            The AI agent performs continuous risk assessment by:
            - Evaluating each area's inherent risk factors (electronics, heat sources, etc.)
            - Combining risk factors with real-time sensor data
            - Adjusting risk levels based on time of day and occupancy
            - Recommending proactive measures for high-risk zones
            """)
        
    with app_areas[3]:
        st.markdown("""
        ### Post-Incident Analysis
        
        After fire incidents, AI agents can drive improvement in prevention and response:
        
        - **Forensic Analysis**: Analyze sensor data to determine fire cause and spread patterns
        - **Response Evaluation**: Assess effectiveness of detection and response systems
        - **Simulation and Training**: Create realistic scenarios based on past incidents
        - **Continuous Learning**: Update risk models and response protocols based on findings
        
        #### Example System: Incident Learning Agent
        
        This analytical system:
        - Consolidates data from all sensors and systems before, during, and after incidents
        - Identifies patterns across multiple incidents to discover systemic issues
        - Recommends specific improvements to prevention and response systems
        - Creates visualization tools to communicate key findings to stakeholders
        """)
        
        # Create visualization of fire incident analysis
        # Sample data
        months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        incidents = [3, 2, 5, 4, 6, 7, 8, 9, 5, 4, 3, 2]
        
        response_times = [8, 7, 6.5, 7, 6, 5.5, 5, 4.5, 4, 4, 3.5, 3]
        
        causes = ["Electrical", "Kitchen", "Heating", "Smoking", "Arson", "Other"]
        cause_counts = [18, 12, 8, 5, 3, 12]
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Incident count and response time
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=months,
                y=incidents,
                mode='lines+markers',
                name='Incidents',
                line=dict(color=FIRESAFETY_COLORS["primary"], width=3),
                marker=dict(size=8)
            ))
            
            fig.add_trace(go.Scatter(
                x=months,
                y=response_times,
                mode='lines+markers',
                name='Avg. Response Time (min)',
                line=dict(color=FIRESAFETY_COLORS["secondary"], width=3),
                marker=dict(size=8),
                yaxis="y2"
            ))
            
            fig = apply_firesafety_theme_to_plotly(
                fig, 
                title='Fire Incidents & AI-Enhanced Response Times',
                height=400
            )
            
            fig.update_layout(
                yaxis=dict(title="Incident Count"),
                yaxis2=dict(
                    title="Response Time (minutes)",
                    overlaying="y",
                    side="right",
                    showgrid=False
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            # Causes breakdown
            fig = px.pie(
                names=causes,
                values=cause_counts,
                title="Fire Incident Causes Analysis",
                color_discrete_sequence=FIRESAFETY_CATEGORICAL
            )
            
            fig = apply_firesafety_theme_to_plotly(
                fig, 
                title='Fire Incident Causes Analysis',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        The AI agent analyzes fire incidents over time and identifies:
        - **Trend improvements**: Decreasing response times due to AI-enhanced early detection
        - **Root causes**: Primary factors leading to fire incidents
        - **Seasonal patterns**: Higher incidents during summer months
        - **Effectiveness metrics**: Measuring performance of response systems
        
        These insights enable targeted improvements to both prevention strategies and response protocols.
        """)
    
    # AI Agent Development for Fire Safety
    st.markdown("## Implementing AI Agents in Fire Safety")
    
    st.markdown("""
    ### Key Technologies for Fire Safety AI Agents
    
    Developing effective AI agents for fire safety applications involves integrating multiple technologies:
    
    1. **Machine Learning Models**:
       - Classification algorithms for fire detection from sensor data
       - Reinforcement learning for optimizing response strategies
       - Anomaly detection to identify unusual patterns that may indicate fire risk
    
    2. **Natural Language Processing**:
       - Interactive communication with emergency responders
       - Processing and summarizing incident reports
       - Converting safety regulations into actionable rules
    
    3. **Computer Vision**:
       - Visual fire and smoke detection from security cameras
       - Occupancy monitoring and people counting
       - Identifying blocked exits or safety hazards
    
    4. **Sensor Integration**:
       - Smoke, heat, and gas sensor networks
       - Environmental monitoring (temperature, humidity)
       - Motion and occupancy detection
    
    5. **Decision Support Systems**:
       - Real-time risk assessment frameworks
       - Automated alarm verification systems
       - Dynamic evacuation route planning
    """)
    
    # Sample code for a simple fire detection agent
    st.markdown("### Sample Code: Simple Fire Detection Agent")
    
    code_sample = '''
# Simple Fire Detection Agent using Reinforcement Learning
import numpy as np

class FireDetectionAgent:
    def __init__(self, sensor_count, learning_rate=0.1, discount_factor=0.9):
        self.sensor_count = sensor_count
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        
        # Q-table: states x actions (raise alarm, investigate, ignore)
        self.q_table = np.zeros((2**sensor_count, 3))
        
    def get_state_index(self, sensor_readings):
        """Convert binary sensor readings to a state index"""
        state_index = 0
        for i, reading in enumerate(sensor_readings):
            if reading:  # If sensor is active
                state_index += 2**i
        return state_index
    
    def choose_action(self, state_index, exploration_rate=0.1):
        """Select action using epsilon-greedy policy"""
        if np.random.random() < exploration_rate:
            # Explore: choose a random action
            return np.random.choice(3)
        else:
            # Exploit: choose best action according to Q-table
            return np.argmax(self.q_table[state_index])
    
    def update_q_table(self, state_index, action, reward, next_state_index):
        """Update Q-value using Q-learning update rule"""
        # Get the maximum Q-value for the next state
        max_next_q = np.max(self.q_table[next_state_index])
        
        # Current Q-value
        current_q = self.q_table[state_index, action]
        
        # Apply Q-learning update formula
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - current_q
        )
        
        # Update Q-table
        self.q_table[state_index, action] = new_q
    
    def train(self, training_data, episodes=1000):
        """Train the agent using historical sensor data and outcomes"""
        for episode in range(episodes):
            # Sample a random training example
            sample_idx = np.random.randint(0, len(training_data))
            
            # Extract data
            sensor_readings, true_fire_status = training_data[sample_idx]
            
            # Get state index
            state_index = self.get_state_index(sensor_readings)
            
            # Choose action
            action = self.choose_action(state_index, exploration_rate=0.3)
            
            # Determine reward (example logic)
            if true_fire_status:  # There is actually a fire
                if action == 0:  # Raised alarm
                    reward = 10  # Correct alarm
                elif action == 1:  # Investigate
                    reward = 5   # Good but not optimal
                else:  # Ignored
                    reward = -15  # Missed fire (very bad)
            else:  # No fire
                if action == 0:  # Raised alarm
                    reward = -10  # False alarm
                elif action == 1:  # Investigate
                    reward = 0    # Neutral
                else:  # Ignored
                    reward = 2    # Correctly ignored
            
            # For simplicity, assume next state is the same but with updated knowledge
            next_state_index = state_index
            
            # Update Q-table
            self.update_q_table(state_index, action, reward, next_state_index)
    
    def detect_fire(self, sensor_readings):
        """Use the trained agent to detect fire"""
        state_index = self.get_state_index(sensor_readings)
        action = np.argmax(self.q_table[state_index])
        
        if action == 0:
            return "ALARM: Fire detected!"
        elif action == 1:
            return "WARNING: Investigate potential fire risk"
        else:
            return "NORMAL: No fire risk detected"

# Example usage
agent = FireDetectionAgent(sensor_count=4)

# Example training data: (sensor_readings, true_fire_status)
training_data = [
    ([True, False, False, False], False),  # Single sensor - false positive
    ([False, True, False, False], False),  # Single sensor - false positive
    ([True, True, False, False], True),    # Two sensors - actual fire
    ([True, True, True, False], True),     # Three sensors - actual fire
    ([False, False, False, False], False), # No sensors - no fire
    # Add more training examples...
]

# Train the agent
agent.train(training_data)

# Test the agent
test_readings = [True, True, False, True]
print(agent.detect_fire(test_readings))
'''
    
    create_code_editor(code_sample, "python")
    
    # Challenges and Ethical Considerations
    st.markdown("## Challenges and Ethical Considerations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### Technical Challenges
        
        - **Data Quality and Availability**: Fire incidents are relatively rare, making it difficult to collect sufficient training data
        
        - **Sensor Reliability**: Fire detection systems must operate in harsh environments with potential for sensor degradation
        
        - **Integration Complexity**: Connecting AI agents with existing building systems and emergency services
        
        - **Real-time Performance**: Fire detection and response require low-latency processing
        
        - **Generalization**: Creating systems that work across different building types and environments
        """)
        
    with col2:
        st.markdown("""
        ### Ethical Considerations
        
        - **False Alarms vs. Missed Detections**: Balancing the trade-off between sensitivity and specificity
        
        - **Privacy Concerns**: Monitoring systems may collect data about building occupants
        
        - **Transparency**: Ensuring AI decision-making is explainable, especially in emergency situations
        
        - **Responsibility**: Determining liability when AI systems are involved in safety-critical decisions
        
        - **Accessibility**: Ensuring AI-enhanced safety benefits are available to all communities
        """)
    
    # Future Trends section
    st.markdown("## Future Trends in AI Agents for Fire Safety")
    
    st.markdown("""
    ### Emerging Technologies and Approaches
    
    - **Multi-Agent Systems**: Networks of specialized agents working together for comprehensive fire safety
    
    - **Digital Twins**: Virtual replicas of buildings that allow for simulation and testing of fire scenarios
    
    - **Edge Computing**: Distributed processing that enables faster response times for fire detection
    
    - **Human-AI Collaboration**: Systems designed to augment human firefighters rather than replace them
    
    - **Adaptive Learning**: Agents that continue to improve their performance based on ongoing experiences
    """)
    
    # Module Quiz
    st.markdown("## Knowledge Assessment")
    
    questions = [
        {
            "question": "What key characteristic allows AI agents to operate without constant human input?",
            "options": ["Automation", "Autonomy", "Aggregation", "Anticipation"],
            "correct": "Autonomy"
        },
        {
            "question": "Which type of AI agent maintains an internal model of how the world evolves?",
            "options": ["Simple Reflex Agent", "Model-Based Agent", "Goal-Based Agent", "Learning Agent"],
            "correct": "Model-Based Agent"
        },
        {
            "question": "In fire safety applications, what is a primary advantage of AI agents over traditional fire detection systems?",
            "options": [
                "Lower cost of implementation", 
                "No need for physical sensors", 
                "Ability to adapt to changing conditions", 
                "Eliminating the need for human firefighters"
            ],
            "correct": "Ability to adapt to changing conditions"
        },
        {
            "question": "What technology would be most useful for an AI agent to identify blocked fire exits in a building?",
            "options": ["Natural Language Processing", "Computer Vision", "Reinforcement Learning", "Blockchain"],
            "correct": "Computer Vision"
        },
        {
            "question": "What is a key challenge in developing AI agents for fire safety?",
            "options": [
                "Excessive data availability", 
                "Limited integration options", 
                "Too many false alarms vs. missed detections", 
                "Sensor systems being too reliable"
            ],
            "correct": "Too many false alarms vs. missed detections"
        }
    ]
    
    create_module_quiz(questions, module_id=6)  # Assuming this is module 6
    
    # Resources for Further Learning
    st.markdown("## Resources for Further Learning")
    
    st.markdown("""
    ### Recommended Reading
    
    - **Artificial Intelligence: A Modern Approach** by Stuart Russell and Peter Norvig
      *Covers the fundamentals of AI agents and their applications*
    
    - **Reinforcement Learning: An Introduction** by Richard S. Sutton and Andrew G. Barto
      *Essential for understanding how AI agents learn from experience*
    
    - **Computer Vision for Visual Fire Recognition: A Comprehensive Review** by Saeed et al.
      *Focuses on computer vision techniques for fire detection*
    
    ### Online Courses
    
    - **AI for Public Safety** - Stanford University
    - **Machine Learning for IoT** - IBM Training
    - **Computer Vision for Emergency Response** - Microsoft Learn
    
    ### Tools and Frameworks
    
    - **TensorFlow Agents**: Library for building reinforcement learning agents
    - **OpenAI Gym**: Environment for developing and testing agents
    - **RASA**: Open-source framework for building conversational AI agents
    """)
    
    # Final thoughts
    st.markdown("""
    ## Conclusion
    
    AI agents represent a transformative technology for fire safety, offering capabilities that go far beyond traditional systems. By integrating perception, decision-making, and action in autonomous systems, AI agents can detect fires earlier, respond more effectively, and continuously learn from experience.
    
    However, successful implementation requires addressing technical challenges, ethical considerations, and integration with existing systems and protocols. The future of fire safety lies in creating intelligent agents that work alongside humans, enhancing our ability to prevent, detect, and respond to fire emergencies.
    """)