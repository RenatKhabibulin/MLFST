import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from utils import load_dataset, create_module_quiz

def show_module_content():
    """Display introduction to ML in fire safety content"""
    
    st.write("## Introduction to Machine Learning in Fire Safety")
    
    # Overview section
    st.write("""
    ### Overview
    
    Machine Learning (ML) has revolutionized many fields, including fire safety. In this module, 
    we'll explore how ML can be applied to enhance fire safety measures, predict fire risks, 
    and improve emergency response times.
    
    This course bridges two important domains:
    1. Machine Learning techniques and methodologies
    2. Fire safety applications and challenges
    
    By the end of this course, you'll understand how to apply various ML algorithms to solve 
    real-world fire safety problems.
    """)
    
    # Key concepts
    with st.expander("Key Concepts in Machine Learning"):
        st.write("""
        ### What is Machine Learning?
        
        Machine Learning is a subset of artificial intelligence that enables systems to learn and improve 
        from experience without being explicitly programmed. Instead of writing code with specific instructions 
        to accomplish a task, a machine learning system is trained on data to identify patterns and make decisions 
        with minimal human intervention.
        """)
        
        # Types of ML as text list instead of diagram
        st.write("### Types of Machine Learning")
        
        st.write("""
        Machine learning can be broadly categorized into three types:
        
        ### 1. Supervised Learning
        * Learning from labeled data where the "correct answers" are provided
        * The algorithm learns to map inputs to outputs based on example pairs
        * **Examples in Fire Safety**:
          * Predicting fire risk levels based on building characteristics
          * Classifying the cause of a fire from incident reports
          * Estimating response times based on location and time of day
        
        ### 2. Unsupervised Learning
        * Finding patterns and structures in unlabeled data
        * The algorithm discovers hidden patterns without explicit guidance
        * **Examples in Fire Safety**:
          * Clustering buildings by risk profile without predefined categories
          * Anomaly detection in sensor readings from fire alarm systems
          * Identifying unusual patterns in fire incident reports
        
        ### 3. Reinforcement Learning
        * Learning through interaction with an environment to maximize rewards
        * The algorithm learns optimal behavior through trial and error
        * **Examples in Fire Safety**:
          * Optimizing evacuation routes in dynamic emergency situations
          * Training virtual agents for fire response simulation
          * Autonomous drone navigation for fire monitoring
        """)
        
        # ML Process as detailed text list
        st.write("### The Machine Learning Process")
        
        st.write("""
        The machine learning process typically follows these steps:
        
        ### 1. Data Collection
        * **Process**: Gathering relevant fire safety data from multiple sources
        * **Fire Safety Sources**:
          * Historical fire incident reports and response logs
          * Building specifications and inspection records
          * Environmental and weather data
          * IoT sensor data from fire detection systems
          * Thermal imaging and video surveillance footage
        * **Challenges**: Data may be siloed across different departments or incomplete
        
        ### 2. Data Preparation
        * **Process**: Cleaning and preprocessing data to handle missing values and outliers
        * **Key Tasks**:
          * Removing duplicate records
          * Filling in missing values with appropriate methods
          * Normalizing or standardizing numerical features
          * Converting categorical data (e.g., building types) to numerical representations
          * Time series alignment for sensor data
        * **Importance**: Clean data is essential for accurate model predictions
        
        ### 3. Feature Engineering
        * **Process**: Creating meaningful features that help predict fire risks or outcomes
        * **Examples in Fire Safety**:
          * Calculating building risk scores from multiple parameters
          * Deriving response time windows from timestamp data
          * Creating seasonal fire risk indicators
          * Extracting spatial relationships between buildings and fire stations
        * **Impact**: Well-engineered features can significantly improve model performance
        
        ### 4. Model Training
        * **Process**: Teaching algorithms to recognize patterns in fire safety data
        * **Common Algorithms**:
          * Decision trees for interpretable fire risk classification
          * Neural networks for complex pattern recognition in sensor data
          * Random forests for robust prediction of fire outcomes
          * Support vector machines for binary classification tasks
        * **Considerations**: Model selection depends on the specific fire safety problem
        
        ### 5. Model Evaluation
        * **Process**: Testing model accuracy and performance in fire safety scenarios
        * **Metrics**:
          * Accuracy: Overall correctness of predictions
          * Precision: Proportion of positive identifications that were actually correct
          * Recall: Proportion of actual positives that were identified correctly
          * F1-score: Balance between precision and recall
        * **Validation**: Cross-validation to ensure model generalizes well to new data
        
        ### 6. Model Deployment
        * **Process**: Implementing models in real-world fire safety applications
        * **Deployment Options**:
          * Integration with building management systems
          * Mobile applications for field inspectors
          * Dashboard systems for fire departments
          * Automated alert systems for early warning
        * **Ongoing Maintenance**: Regular model updates with new data
        """)
    
    # ML in fire safety
    with st.expander("Machine Learning in Fire Safety: Historical Context"):
        st.write("""
        ### Evolution of Fire Safety Analytics
        
        Fire safety has evolved from simple rule-based systems to sophisticated data-driven approaches:
        
        * **1970s-80s**: Basic statistical analysis of fire incidents
        * **1990s-2000s**: Computer simulations and fire modeling
        * **2010s-present**: Machine learning algorithms for predictive fire safety
        
        ### Early Applications
        
        Early applications of data analysis in fire safety included:
        
        * Statistical analysis of fire incident reports
        * Computational fluid dynamics for fire spread modeling
        * Pattern recognition for fire detection in video feeds
        
        ### Modern Machine Learning Applications
        
        Today, machine learning is applied to numerous fire safety domains:
        
        * Predictive maintenance of fire protection systems
        * Real-time detection of fire hazards in buildings
        * Optimizing evacuation routes during emergencies
        * Fire risk assessment based on building characteristics
        * Resource allocation for fire departments
        """)
    
    # Practical example
    st.write("### Practical Example: Analyzing Fire Incident Data")
    
    # Load dataset
    df_incidents = load_dataset("fire_incidents.csv")
    
    if not df_incidents.empty:
        # Display sample data
        st.write("#### Sample Fire Incident Data")
        st.write(df_incidents.head())
        
        # Simple data analysis
        st.write("#### Descriptive Analysis")
        
        # Create visualizations
        st.write("##### Fire Incidents by Cause")
        
        # Bar chart of fire causes
        cause_counts = df_incidents['cause'].value_counts().reset_index()
        cause_counts.columns = ['Cause', 'Count']
        
        fig = px.bar(
            cause_counts, 
            x='Cause', 
            y='Count',
            title='Distribution of Fire Incidents by Cause',
            color='Count',
            color_continuous_scale=px.colors.sequential.Oranges
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Damage levels
        st.write("##### Fire Incidents by Damage Level")
        
        damage_counts = df_incidents['damage_level'].value_counts().reset_index()
        damage_counts.columns = ['Damage Level', 'Count']
        
        fig2 = px.pie(
            damage_counts, 
            values='Count', 
            names='Damage Level',
            title='Distribution of Fire Incidents by Damage Level',
            color_discrete_sequence=px.colors.sequential.Oranges
        )
        
        st.plotly_chart(fig2, use_container_width=True)
        
        # Interactive scatter plot
        st.write("##### Interactive Exploration")
        
        # Add a simple feature: time to arrive
        df_incidents['detection_time_mins'] = df_incidents['detection_time'].apply(
            lambda x: int(str(x).split(':')[0]) * 60 + int(str(x).split(':')[1]) if pd.notna(x) and isinstance(x, str) else 0
        )
        df_incidents['arrival_time_mins'] = df_incidents['arrival_time'].apply(
            lambda x: int(str(x).split(':')[0]) * 60 + int(str(x).split(':')[1]) if pd.notna(x) and isinstance(x, str) else 0
        )
        df_incidents['response_time_mins'] = df_incidents['arrival_time_mins'] - df_incidents['detection_time_mins']
        df_incidents.loc[df_incidents['response_time_mins'] < 0, 'response_time_mins'] += 24 * 60  # Fix day boundary
        
        # Scatter plot
        fig3 = px.scatter(
            df_incidents,
            x='response_time_mins',
            y='damage_level',
            color='cause',
            hover_data=['incident_id', 'date'],
            title='Relationship Between Response Time and Damage Level',
            labels={'response_time_mins': 'Response Time (minutes)', 'damage_level': 'Damage Level'},
            color_discrete_sequence=px.colors.qualitative.Set1
        )
        
        st.plotly_chart(fig3, use_container_width=True)
        
        # Insights
        st.write("""
        #### Key Insights from Descriptive Analysis
        
        From this simple analysis, we can draw several insights:
        
        1. **Common Causes**: Electrical issues are the most frequent cause of fires in our dataset.
        2. **Damage Distribution**: Minor damage is most common, but a significant portion of incidents result in severe damage.
        3. **Response Time Impact**: There appears to be a correlation between longer response times and more severe damage.
        
        This descriptive analysis is just the beginning. Machine learning can help us go beyond these observations to build predictive models.
        """)
    
    # Machine learning potential
    st.write("### The Potential of Machine Learning in Fire Safety")
    
    st.write("""
    Machine learning offers significant potential for improving fire safety in multiple domains:
    
    ### 1. Risk Assessment & Prediction
    * **Application**: Forecasting fire risks based on historical data and building characteristics
    * **Techniques**:
      * Regression models for risk scoring
      * Classification algorithms for risk categorization
      * Time series analysis for seasonal risk prediction
    * **Examples**:
      * Predicting high-risk periods based on weather patterns
      * Identifying buildings with highest fire probability
      * Forecasting urban areas with emerging fire risks
    
    ### 2. Early Detection & Monitoring
    * **Application**: Identifying fire hazards and early signs of fire through sensors and camera systems
    * **Techniques**:
      * Computer vision for smoke and flame detection
      * Anomaly detection for unusual temperature patterns
      * Sensor fusion algorithms for comprehensive monitoring
    * **Examples**:
      * Intelligent smoke detectors with reduced false alarms
      * Thermal imaging systems with automatic alert capabilities
      * Multi-sensor arrays with real-time hazard classification
    
    ### 3. Evacuation & Response Planning
    * **Application**: Optimizing evacuation routes and emergency response strategies
    * **Techniques**:
      * Pathfinding algorithms for optimal routes
      * Crowd simulation models for evacuation planning
      * Resource allocation algorithms for emergency services
    * **Examples**:
      * Dynamic evacuation route guidance based on real-time conditions
      * Personalized evacuation plans for individuals with mobility needs
      * Optimal placement of emergency responders based on predicted incidents
    
    ### 4. Predictive Maintenance
    * **Application**: Anticipating failures in fire safety equipment before they occur
    * **Techniques**:
      * Survival analysis for equipment lifetime prediction
      * Pattern recognition for identifying pre-failure signatures
      * Sequential models for maintenance scheduling
    * **Examples**:
      * Predicting sprinkler system failures before they happen
      * Scheduling fire alarm testing based on environmental stressors
      * Identifying degradation patterns in fire suppression systems
    
    ### 5. Resource Optimization
    * **Application**: Improving allocation of firefighting resources for maximum effectiveness
    * **Techniques**:
      * Optimization algorithms for resource placement
      * Demand forecasting for staffing levels
      * Multi-objective optimization for balancing coverage and cost
    * **Examples**:
      * Optimal fire station location planning
      * Dynamic staffing models based on predicted incident patterns
      * Equipment allocation optimization across multiple stations
    
    In the upcoming modules, we'll explore these applications in depth, learning both the technical aspects of machine learning algorithms and their practical implementation in fire safety contexts.
    """)
    
    # Module quiz
    st.write("### Module 1 Quiz")
    
    questions = [
        {
            'question': 'What is machine learning?',
            'options': [
                'A programming language specifically designed for fire safety applications',
                'A subset of artificial intelligence that enables systems to learn from data without explicit programming',
                'A simulation tool for predicting fire spread in buildings',
                'A database of historical fire incidents'
            ],
            'correct': 'A subset of artificial intelligence that enables systems to learn from data without explicit programming'
        },
        {
            'question': 'Which of the following is NOT a type of machine learning?',
            'options': [
                'Supervised Learning',
                'Unsupervised Learning',
                'Reinforcement Learning',
                'Descriptive Learning'
            ],
            'correct': 'Descriptive Learning'
        },
        {
            'question': 'What is the correct sequence in the machine learning process?',
            'options': [
                'Model Training → Data Collection → Data Preparation → Model Evaluation → Model Deployment',
                'Data Collection → Data Preparation → Model Training → Model Evaluation → Model Deployment',
                'Data Collection → Model Training → Data Preparation → Model Deployment → Model Evaluation',
                'Model Design → Data Collection → Model Training → Model Testing → Data Analysis'
            ],
            'correct': 'Data Collection → Data Preparation → Model Training → Model Evaluation → Model Deployment'
        },
        {
            'question': 'How can machine learning improve fire safety?',
            'options': [
                'By replacing human firefighters with robots',
                'By predicting fire risks, optimizing response strategies, and identifying hazards',
                'By eliminating the need for fire safety equipment',
                'By creating virtual simulations of fires for training purposes only'
            ],
            'correct': 'By predicting fire risks, optimizing response strategies, and identifying hazards'
        },
        {
            'question': 'What type of analysis were we performing in the practical example with fire incident data?',
            'options': [
                'Predictive analysis',
                'Prescriptive analysis',
                'Descriptive analysis',
                'Cognitive analysis'
            ],
            'correct': 'Descriptive analysis'
        },
        {
            'question': 'In the context of fire safety, what would be an example of supervised learning?',
            'options': [
                'Grouping buildings by their architectural similarities',
                'Predicting the fire risk level of a building based on historical fire data',
                'Identifying unusual patterns in smoke detector readings without prior examples',
                'Learning optimal evacuation routes through trial and error'
            ],
            'correct': 'Predicting the fire risk level of a building based on historical fire data'
        },
        {
            'question': 'Which of the following is a key challenge when applying machine learning to fire safety?',
            'options': [
                'Too much data available on fire incidents',
                'Fire incidents are too predictable',
                'The rarity of major fire events can lead to imbalanced datasets',
                'Machine learning algorithms are too simple for fire safety applications'
            ],
            'correct': 'The rarity of major fire events can lead to imbalanced datasets'
        },
        {
            'question': 'What was one of the earliest applications of data analysis in fire safety?',
            'options': [
                'Neural networks for fire prediction',
                'Statistical analysis of fire incident reports',
                'Deep learning for image recognition of fires',
                'Reinforcement learning for firefighting robots'
            ],
            'correct': 'Statistical analysis of fire incident reports'
        },
        {
            'question': 'When analyzing fire incident data, which of the following would NOT be considered a feature?',
            'options': [
                'Building age',
                'Presence of sprinkler systems',
                'The machine learning algorithm used',
                'Time to firefighter arrival'
            ],
            'correct': 'The machine learning algorithm used'
        },
        {
            'question': 'Which of these represents the correct progression of fire safety analytics over time?',
            'options': [
                'Machine learning → Statistical analysis → Computer simulations',
                'Computer simulations → Machine learning → Statistical analysis',
                'Statistical analysis → Computer simulations → Machine learning',
                'Statistical analysis → Machine learning → Computer simulations'
            ],
            'correct': 'Statistical analysis → Computer simulations → Machine learning'
        }
    ]
    
    create_module_quiz(questions, 0)
