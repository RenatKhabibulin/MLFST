import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from utils import load_dataset, create_module_quiz, create_code_editor
from visualization import plot_cnn_architecture
from ml_utils import create_cnn_model, create_lstm_model
# Temporarily commenting out TensorFlow imports to fix compatibility issues
# import tensorflow as tf

def show_module_content():
    """Display deep learning content for fire safety applications"""
    
    st.write("## Deep Learning for Fire Safety Applications")
    
    # Overview
    st.write("""
    ### Overview
    
    Deep learning is a subset of machine learning that uses neural networks with multiple layers 
    to progressively extract higher-level features from raw input. In fire safety, these powerful 
    techniques can process complex data like images, time series, and multidimensional sensor readings 
    to provide advanced detection, prediction, and analysis capabilities.
    """)
    
    # Tab-based navigation for different deep learning approaches
    tabs = st.tabs([
        "Neural Networks for Image Analysis", 
        "Recurrent Neural Networks",
        "Convolutional Neural Networks",
        "Transfer Learning"
    ])
    
    # Neural Networks for Image Analysis tab
    with tabs[0]:
        st.write("### Neural Networks for Image Analysis")
        
        st.write("""
        Neural networks can analyze visual data from cameras, thermal imaging, and satellite imagery 
        for fire detection, monitoring, and assessment.
        
        **Key Concepts:**
        - Multiple layers of interconnected neurons
        - Ability to learn complex visual patterns
        - Automatic feature extraction from raw images
        - Can process various image types (RGB, thermal, infrared)
        
        **Fire Safety Applications:**
        - Early fire and smoke detection from surveillance footage
        - Thermal hotspot identification for early warning
        - Post-fire damage assessment from aerial imagery
        - Wildfire detection and monitoring
        """)
        
        # Practical example
        st.write("#### Practical Example: Smoke and Fire Detection in Images")
        
        st.write("""
        Deep learning models can be trained to detect fire and smoke in images with high accuracy.
        This example demonstrates how a convolutional neural network (CNN) can be used for this purpose.
        """)
        
        # Show sample fire detection results
        st.write("##### Sample Fire Detection Results")
        
        # Create a sample grid showing detection results
        col1, col2 = st.columns(2)
        
        with col1:
            st.components.v1.html("""
            <div style="border: 2px solid #FF4B4B; border-radius: 5px; padding: 10px;">
                <div style="text-align: center; font-weight: bold; margin-bottom: 10px;">Fire Detected (98.7%)</div>
                <svg width="100%" height="200" xmlns="http://www.w3.org/2000/svg">
                    <rect width="100%" height="100%" fill="#000"/>
                    <path d="M50,180 C 50,100 80,120 70,60 C 90,100 100,80 100,140 C 120,100 120,160 130,120 C 150,160 140,180 160,160 C 160,180 140,200 120,180 C 100,200 80,180 50,180 Z" fill="#FF4B4B"/>
                    <path d="M70,170 C 70,120 90,130 85,90 C 95,110 100,100 100,140 C 110,120 110,150 115,130 C 125,150 120,170 130,160 C 130,170 120,180 110,170 C 100,180 90,170 70,170 Z" fill="#FFA500"/>
                    <path d="M85,160 C 85,130 95,140 92,120 C 97,130 100,120 100,140 C 105,130 105,150 107,140 C 112,150 110,160 115,155 C 115,160 110,165 107,160 C 103,165 98,160 85,160 Z" fill="#FFFF00"/>
                </svg>
                <div style="text-align: center; font-style: italic; margin-top: 5px;">Building fire with visible flames</div>
            </div>
            """, height=250)
        
        with col2:
            st.components.v1.html("""
            <div style="border: 2px solid #1E88E5; border-radius: 5px; padding: 10px;">
                <div style="text-align: center; font-weight: bold; margin-bottom: 10px;">No Fire Detected (2.3%)</div>
                <svg width="100%" height="200" xmlns="http://www.w3.org/2000/svg">
                    <rect width="100%" height="100%" fill="#87CEEB"/>
                    <path d="M0,130 C50,120 100,140 150,120 C200,140 250,120 300,130 L300,200 L0,200 Z" fill="#1E88E5"/>
                    <path d="M30,100 C40,95 50,100 60,95 C70,100 80,95 90,100 L90,130 L30,130 Z" fill="#FFFFFF"/>
                    <path d="M150,80 C165,75 180,80 195,75 C210,80 225,75 240,80 L240,110 L150,110 Z" fill="#FFFFFF"/>
                </svg>
                <div style="text-align: center; font-style: italic; margin-top: 5px;">Clear sky with no fire hazards</div>
            </div>
            """, height=250)
        
        # Explain the architecture of fire detection CNN
        st.write("##### Fire Detection Neural Network Architecture")
        
        st.write("""
        Fire detection neural networks typically use a convolutional architecture (CNN) that can recognize 
        patterns in images associated with fire and smoke. These networks consist of several key components:
        
        1. **Input Layer**: Receives the image (RGB or thermal)
        2. **Convolutional Layers**: Extract visual features like edges, textures, and patterns
        3. **Pooling Layers**: Reduce dimensions while preserving important information
        4. **Fully Connected Layers**: Interpret extracted features to make classification decisions
        5. **Output Layer**: Provides probability of fire/smoke presence
        """)
        
        # Visualize CNN architecture
        plot_cnn_architecture()
        
        # Image preprocessing section
        st.write("##### Image Preprocessing for Fire Detection")
        
        st.write("""
        Before feeding images to a deep learning model, several preprocessing steps are typically performed:
        
        1. **Resizing**: Standardize images to a fixed dimension (e.g., 224×224 pixels)
        2. **Normalization**: Scale pixel values to a range between 0 and 1
        3. **Augmentation**: Generate additional training samples through transformations:
           - Rotating images
           - Adjusting brightness and contrast
           - Horizontal flipping
           - Adding noise
        
        These preprocessing steps improve the model's ability to generalize and recognize fires in various conditions.
        """)
        
        # Code example for a fire detection model
        with st.expander("Python Code: Simple Fire Detection CNN"):
            st.code("""
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define image dimensions and parameters
img_width, img_height = 224, 224
batch_size = 32

# Create a data generator with augmentation for training
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Validation data generator (just rescaling)
validation_datagen = ImageDataGenerator(rescale=1./255)

# Load training data
train_generator = train_datagen.flow_from_directory(
    'fire_dataset/train',
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary'  # 'fire' or 'no_fire'
)

# Load validation data
validation_generator = validation_datagen.flow_from_directory(
    'fire_dataset/validation',
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary'
)

# Build the CNN model
model = Sequential([
    # First convolutional layer
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)),
    MaxPooling2D((2, 2)),
    
    # Second convolutional layer
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    
    # Third convolutional layer
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    
    # Flatten and dense layers
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),  # Reduce overfitting
    Dense(1, activation='sigmoid')  # Binary output (fire or no fire)
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=20,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size
)

# Save the model
model.save('fire_detection_model.h5')

# Evaluate the model
test_generator = validation_datagen.flow_from_directory(
    'fire_dataset/test',
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary'
)

test_loss, test_acc = model.evaluate(test_generator)
print(f'Test accuracy: {test_acc:.4f}')
""", language="python")
        
        # Performance metrics and challenges
        st.write("##### Performance and Challenges")
        
        st.write("""
        Modern fire detection neural networks can achieve high accuracy (95%+) in controlled environments, 
        but several challenges remain:
        
        1. **False Positives**: Distinguishing fires from fire-like objects (sunset, red lights)
        2. **Poor Visibility Conditions**: Detecting fires through smoke, fog, or at night
        3. **Computational Requirements**: Balancing accuracy with resource constraints for real-time detection
        4. **Data Scarcity**: Limited datasets of real fire incidents in various environments
        
        Ongoing research focuses on addressing these challenges through more sophisticated architectures, 
        better training techniques, and multimodal approaches combining RGB, thermal, and other sensor data.
        """)
    
    # Recurrent Neural Networks tab
    with tabs[1]:
        st.write("### Recurrent Neural Networks (RNNs)")
        
        st.write("""
        Recurrent Neural Networks are specialized for sequential data, making them ideal for time series analysis
        of sensor readings and temporal patterns in fire development.
        
        **Key Concepts:**
        - Process sequential data with variable length
        - Maintain memory of previous inputs
        - Capture temporal dependencies
        - Variants like LSTM and GRU handle long-term dependencies
        
        **Fire Safety Applications:**
        - Predicting fire spread dynamics over time
        - Early warning systems based on sensor time series
        - Forecasting evacuation time requirements
        - Modeling fire behavior under changing conditions
        """)
        
        # LSTM explanation
        st.write("#### Long Short-Term Memory (LSTM) Networks")
        
        st.write("""
        LSTM networks are a specialized type of RNN designed to learn long-term dependencies in sequence data. 
        They use a sophisticated memory cell structure that can selectively remember or forget information 
        over long sequences, making them ideal for fire progression analysis.
        
        **LSTM Architecture Components:**
        - **Input Gate**: Controls what new information to store in the cell state
        - **Forget Gate**: Controls what information to discard from the cell state
        - **Output Gate**: Controls what parts of the cell state to output
        - **Cell State**: Long-term memory that runs through the entire sequence
        """)
        
        # LSTM visual representation - New enhanced design
        st.components.v1.html("""
        <svg width="750" height="370" xmlns="http://www.w3.org/2000/svg">
            <!-- Background gradient -->
            <defs>
                <linearGradient id="bg_grad" x1="0%" y1="0%" x2="100%" y2="100%">
                    <stop offset="0%" style="stop-color:#FFF8F5;stop-opacity:1" />
                    <stop offset="100%" style="stop-color:#FFE8E0;stop-opacity:1" />
                </linearGradient>
                <linearGradient id="cell_grad" x1="0%" y1="0%" x2="100%" y2="100%">
                    <stop offset="0%" style="stop-color:#FFEBEB;stop-opacity:1" />
                    <stop offset="100%" style="stop-color:#FFC8C8;stop-opacity:1" />
                </linearGradient>
                <linearGradient id="gate_grad" x1="0%" y1="0%" x2="100%" y2="100%">
                    <stop offset="0%" style="stop-color:#E1F5FE;stop-opacity:1" />
                    <stop offset="100%" style="stop-color:#B3E5FC;stop-opacity:1" />
                </linearGradient>
                <filter id="shadow" x="-20%" y="-20%" width="140%" height="140%">
                    <feGaussianBlur in="SourceAlpha" stdDeviation="3" />
                    <feOffset dx="2" dy="2" result="offsetblur" />
                    <feComponentTransfer>
                        <feFuncA type="linear" slope="0.2" />
                    </feComponentTransfer>
                    <feMerge>
                        <feMergeNode />
                        <feMergeNode in="SourceGraphic" />
                    </feMerge>
                </filter>
            </defs>
            
            <!-- Main background -->
            <rect x="0" y="0" width="750" height="370" rx="10" fill="url(#bg_grad)" />
            <text x="375" y="30" text-anchor="middle" font-family="Arial" font-size="22" font-weight="bold" fill="#333">LSTM Architecture for Fire Progression Prediction</text>
            
            <!-- Time series data visualization -->
            <g transform="translate(60,50)">
                <rect x="0" y="0" width="120" height="80" rx="5" fill="#FFFFFF" stroke="#666666" stroke-width="1" />
                <text x="60" y="20" text-anchor="middle" font-family="Arial" font-size="12" font-weight="bold">Time Series Input</text>
                
                <!-- Temperature graph line -->
                <polyline points="10,60 25,45 40,50 55,30 70,35 85,25 100,20 110,30" 
                        stroke="#FF5722" stroke-width="2" fill="none" />
                <text x="60" y="75" text-anchor="middle" font-family="Arial" font-size="10">Temperature Data</text>
                
                <!-- Smoke sensor graph line -->
                <polyline points="10,50 25,55 40,60 55,65 70,50 85,35 100,25 110,20" 
                        stroke="#673AB7" stroke-width="2" fill="none" stroke-dasharray="2,2" />
            </g>
            
            <!-- LSTM Cell detailed architecture -->
            <g transform="translate(230,50)">
                <!-- LSTM Cell main container -->
                <rect x="0" y="0" width="400" height="270" rx="10" fill="url(#cell_grad)" stroke="#FF5722" stroke-width="2" filter="url(#shadow)" />
                <text x="200" y="25" text-anchor="middle" font-family="Arial" font-size="16" font-weight="bold">LSTM Cell Structure</text>
                
                <!-- Cell State line (long-term memory) -->
                <path d="M20,50 L380,50" stroke="#E91E63" stroke-width="3" />
                <text x="200" y="40" text-anchor="middle" font-family="Arial" font-size="12" font-weight="bold" fill="#E91E63">Cell State (Long-term Memory)</text>
                
                <!-- Input gate -->
                <circle cx="100" cy="120" r="30" fill="url(#gate_grad)" stroke="#2196F3" stroke-width="2" />
                <text x="100" y="123" text-anchor="middle" font-family="Arial" font-size="12" font-weight="bold">Input</text>
                <text x="100" y="138" text-anchor="middle" font-family="Arial" font-size="10">Gate</text>
                
                <!-- Forget gate -->
                <circle cx="200" cy="120" r="30" fill="url(#gate_grad)" stroke="#2196F3" stroke-width="2" />
                <text x="200" y="123" text-anchor="middle" font-family="Arial" font-size="12" font-weight="bold">Forget</text>
                <text x="200" y="138" text-anchor="middle" font-family="Arial" font-size="10">Gate</text>
                
                <!-- Output gate -->
                <circle cx="300" cy="120" r="30" fill="url(#gate_grad)" stroke="#2196F3" stroke-width="2" />
                <text x="300" y="123" text-anchor="middle" font-family="Arial" font-size="12" font-weight="bold">Output</text>
                <text x="300" y="138" text-anchor="middle" font-family="Arial" font-size="10">Gate</text>
                
                <!-- Connect gates to cell state -->
                <path d="M100,90 L100,50" stroke="#333" stroke-width="1.5" stroke-dasharray="4,2" />
                <path d="M200,90 L200,50" stroke="#333" stroke-width="1.5" stroke-dasharray="4,2" />
                <path d="M300,90 L300,50" stroke="#333" stroke-width="1.5" stroke-dasharray="4,2" />
                
                <!-- Input node -->
                <circle cx="50" cy="200" r="20" fill="#FFFFFF" stroke="#424242" stroke-width="1.5" />
                <text x="50" y="204" text-anchor="middle" font-family="Arial" font-size="12">X_t</text>
                
                <!-- Hidden state from previous time step -->
                <circle cx="50" cy="250" r="20" fill="#FFFFFF" stroke="#424242" stroke-width="1.5" />
                <text x="50" y="254" text-anchor="middle" font-family="Arial" font-size="12">h_t-1</text>
                
                <!-- Connect inputs to gates -->
                <path d="M70,200 L100,150" stroke="#333" stroke-width="1.5" />
                <path d="M70,200 L200,150" stroke="#333" stroke-width="1.5" />
                <path d="M70,200 L300,150" stroke="#333" stroke-width="1.5" />
                
                <path d="M70,250 L100,150" stroke="#333" stroke-width="1.5" stroke-dasharray="4,2" />
                <path d="M70,250 L200,150" stroke="#333" stroke-width="1.5" stroke-dasharray="4,2" />
                <path d="M70,250 L300,150" stroke="#333" stroke-width="1.5" stroke-dasharray="4,2" />
                
                <!-- Output/Hidden state -->
                <circle cx="350" cy="200" r="20" fill="#FFFFFF" stroke="#E64A19" stroke-width="2" />
                <text x="350" y="204" text-anchor="middle" font-family="Arial" font-size="12">h_t</text>
                
                <!-- Connect gates to output -->
                <path d="M300,150 L350,200" stroke="#333" stroke-width="1.5" />
                
                <!-- Output connections -->
                <path d="M350,220 L350,250" stroke="#E64A19" stroke-width="2" />
                <text x="350" y="270" text-anchor="middle" font-family="Arial" font-size="12">To Next Cell</text>
            </g>
            
            <!-- Prediction output -->
            <g transform="translate(650,150)">
                <rect x="0" y="0" width="80" height="60" rx="5" fill="#FFF3E0" stroke="#FF9800" stroke-width="2" />
                <text x="40" y="25" text-anchor="middle" font-family="Arial" font-size="12" font-weight="bold">Prediction</text>
                <text x="40" y="45" text-anchor="middle" font-family="Arial" font-size="12">Fire Spread</text>
                
                <!-- Fire icon -->
                <path d="M40,70 C40,50 45,55 42,40 C48,50 50,45 50,60 C55,50 55,65 57,55 C60,65 58,70 65,65 C60,75 55,80 52,75 C48,80 45,75 40,70 Z" fill="#FF5722" />
            </g>
            
            <!-- Connections -->
            <path d="M180,90 L230,90" stroke="#333" stroke-width="2" />
            <polygon points="225,85 235,90 225,95" fill="#333" />
            
            <path d="M630,180 L650,180" stroke="#333" stroke-width="2" />
            <polygon points="645,175 655,180 645,185" fill="#333" />
            
            <!-- Forget Gate -->
            <circle cx="200" cy="90" r="20" fill="#FFA500" />
            <text x="200" y="95" text-anchor="middle" font-family="Arial" font-size="16">×</text>
            <text x="200" y="125" text-anchor="middle" font-family="Arial" font-size="10">Forget</text>
            <text x="200" y="135" text-anchor="middle" font-family="Arial" font-size="10">Gate</text>
            
            <!-- Input Gate -->
            <circle cx="300" cy="90" r="20" fill="#66BB6A" />
            <text x="300" y="95" text-anchor="middle" font-family="Arial" font-size="16">+</text>
            <text x="300" y="125" text-anchor="middle" font-family="Arial" font-size="10">Input</text>
            <text x="300" y="135" text-anchor="middle" font-family="Arial" font-size="10">Gate</text>
            
            <!-- Cell Input -->
            <circle cx="250" cy="150" r="20" fill="#42A5F5" />
            <text x="250" y="155" text-anchor="middle" font-family="Arial" font-size="16">tanh</text>
            <path d="M250,130 L250,90" stroke="black" stroke-width="1" stroke-dasharray="4,2" />
            
            <!-- Output Gate -->
            <circle cx="450" cy="150" r="20" fill="#9575CD" />
            <text x="450" y="155" text-anchor="middle" font-family="Arial" font-size="16">×</text>
            <text x="450" y="185" text-anchor="middle" font-family="Arial" font-size="10">Output</text>
            <text x="450" y="195" text-anchor="middle" font-family="Arial" font-size="10">Gate</text>
            
            <!-- Final tanh -->
            <circle cx="400" cy="90" r="20" fill="#42A5F5" />
            <text x="400" y="95" text-anchor="middle" font-family="Arial" font-size="16">tanh</text>
            <path d="M400,110 L450,130" stroke="black" stroke-width="1" />
            
            <!-- Hidden State Line -->
            <path d="M450,150 L550,150" stroke="#FF7B4B" stroke-width="2" />
            <path d="M500,150 L500,220 L100,220 L100,150" stroke="#FF7B4B" stroke-width="2" stroke-dasharray="5,3" />
            <text x="300" y="240" text-anchor="middle" font-family="Arial" font-size="12">Hidden State Feedback</text>
        </svg>
        """, height=300)
        
        # Practical example
        st.write("#### Practical Example: Predicting Fire Spread Dynamics")
        
        st.write("""
        LSTMs can predict how a fire will progress based on time series data from multiple sensors. 
        This example demonstrates forecasting temperature changes in a building during a fire scenario.
        """)
        
        # Load sensor data for time series
        df_sensors = load_dataset("sensor_readings.csv")
        
        if not df_sensors.empty:
            # Prepare time series data
            df_sensors['timestamp'] = pd.to_datetime(df_sensors['timestamp'])
            df_sensors = df_sensors.sort_values('timestamp')
            
            # Get data for a specific sensor for demonstration
            sensor_id = df_sensors['sensor_id'].value_counts().index[0]
            df_sensor_ts = df_sensors[df_sensors['sensor_id'] == sensor_id].copy()
            
            # Display the data
            st.write("Temperature readings from a single sensor over time:")
            st.write(df_sensor_ts[['timestamp', 'temperature_celsius', 'smoke_density']].head())
            
            # Plot the time series
            fig = px.line(
                df_sensor_ts, 
                x='timestamp', 
                y='temperature_celsius',
                title=f'Temperature Readings from Sensor {sensor_id}',
                labels={'timestamp': 'Time', 'temperature_celsius': 'Temperature (°C)'}
            )
            
            fig.update_layout(
                xaxis_title="Time",
                yaxis_title="Temperature (°C)",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Add a second line for smoke density
            fig2 = go.Figure()
            
            fig2.add_trace(go.Scatter(
                x=df_sensor_ts['timestamp'],
                y=df_sensor_ts['temperature_celsius'],
                name='Temperature (°C)',
                line=dict(color='#FF4B4B', width=2)
            ))
            
            fig2.add_trace(go.Scatter(
                x=df_sensor_ts['timestamp'],
                y=df_sensor_ts['smoke_density'] * 50,  # Scale for visualization
                name='Smoke Density (scaled)',
                line=dict(color='#1E88E5', width=2, dash='dot')
            ))
            
            fig2.update_layout(
                title='Temperature and Smoke Density Over Time',
                xaxis_title='Time',
                yaxis_title='Values',
                height=400,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            st.plotly_chart(fig2, use_container_width=True)
            
            # Explain LSTM prediction approach
            st.write("#### LSTM Prediction Approach")
            
            st.write("""
            To predict fire spread using an LSTM, we follow these steps:
            
            1. **Prepare Sequences**: Create input sequences of sensor readings (temperature, smoke, CO) over a fixed time window
            2. **Define Prediction Target**: Set the target as future temperature values (1, 5, or 10 minutes ahead)
            3. **Model Architecture**: Build an LSTM network with appropriate layers for time series forecasting
            4. **Training**: Train the model on historical fire progression data
            5. **Evaluation**: Test prediction accuracy on held-out fire incidents
            6. **Deployment**: Implement the model in monitoring systems for real-time prediction
            """)
            
            # Show a visualization of LSTM prediction
            st.write("#### LSTM Prediction Results")
            
            # Generate some synthetic prediction data for visualization
            # (In a real application, this would come from an actual LSTM model)
            times = df_sensor_ts['timestamp'].iloc[-20:].reset_index(drop=True)
            actual_values = df_sensor_ts['temperature_celsius'].iloc[-20:].reset_index(drop=True)
            
            # Generate future times
            future_times = pd.date_range(
                start=times.iloc[-1], 
                periods=11, 
                freq=pd.infer_freq(times) or 'H'
            )[1:]  # Skip the first one as it's the last actual time
            
            # Combine times
            all_times = times.tolist() + future_times.tolist()
            
            # Generate "predicted" values (synthetic for this example)
            last_actual = actual_values.iloc[-1]
            
            # Create increasing prediction with some randomness
            predicted_future = [
                last_actual + (i+1) * 2.5 + np.random.normal(0, 1) 
                for i in range(len(future_times))
            ]
            
            # Create the plot
            fig_pred = go.Figure()
            
            # Actual values
            fig_pred.add_trace(go.Scatter(
                x=times,
                y=actual_values,
                name='Historical Data',
                line=dict(color='#1E88E5', width=2)
            ))
            
            # Predicted future values
            fig_pred.add_trace(go.Scatter(
                x=future_times,
                y=predicted_future,
                name='LSTM Prediction',
                line=dict(color='#FF4B4B', width=2)
            ))
            
            # Add confidence interval (synthetic for this example)
            upper_bound = [val + 5 + i/2 for i, val in enumerate(predicted_future)]
            lower_bound = [val - 5 - i/2 for i, val in enumerate(predicted_future)]
            
            fig_pred.add_trace(go.Scatter(
                x=future_times.tolist() + future_times.tolist()[::-1],
                y=upper_bound + lower_bound[::-1],
                fill='toself',
                fillcolor='rgba(255, 75, 75, 0.2)',
                line=dict(color='rgba(255, 255, 255, 0)'),
                name='Prediction Interval'
            ))
            
            # Update layout
            fig_pred.update_layout(
                title='LSTM Temperature Prediction with Confidence Interval',
                xaxis_title='Time',
                yaxis_title='Temperature (°C)',
                height=500,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                shapes=[
                    # Add vertical line at prediction start
                    dict(
                        type="line",
                        xref="x",
                        yref="paper",
                        x0=times.iloc[-1],
                        y0=0,
                        x1=times.iloc[-1],
                        y1=1,
                        line=dict(
                            color="Black",
                            width=2,
                            dash="dot",
                        )
                    )
                ],
                annotations=[
                    dict(
                        x=times.iloc[-1],
                        y=0.95,
                        xref="x",
                        yref="paper",
                        text="Prediction Start",
                        showarrow=True,
                        arrowhead=1,
                        ax=-40,
                        ay=-30
                    )
                ]
            )
            
            st.plotly_chart(fig_pred, use_container_width=True)
            
            # Code example for LSTM model
            with st.expander("Python Code: LSTM for Fire Spread Prediction"):
                st.code("""
import numpy as np
import pandas as pd
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# Prepare the dataset (assuming df_sensors is loaded)
# Get data for a specific sensor
sensor_id = 1
df_sensor = df_sensors[df_sensors['sensor_id'] == sensor_id].copy()
df_sensor = df_sensor.sort_values('timestamp')

# Select features for prediction
features = ['temperature_celsius', 'smoke_density', 'carbon_monoxide_ppm']
target = 'temperature_celsius'  # Predict future temperature

# Scale the data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df_sensor[features])

# Create sequences for LSTM (X: t to t+n-1, y: t+n)
def create_sequences(data, seq_length, pred_horizon):
    X, y = [], []
    for i in range(len(data) - seq_length - pred_horizon):
        X.append(data[i:(i + seq_length)])
        # Target is the temperature value 'pred_horizon' steps ahead
        y.append(data[i + seq_length + pred_horizon, 0])  # Temperature is at index 0
    return np.array(X), np.array(y)

# Parameters
sequence_length = 10  # Use 10 time steps for prediction
prediction_horizon = 5  # Predict 5 time steps ahead

# Create sequences
X, y = create_sequences(scaled_data, sequence_length, prediction_horizon)

# Split data into training and test sets
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Build LSTM model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(sequence_length, len(features))),
    Dropout(0.2),
    LSTM(50),
    Dropout(0.2),
    Dense(1)  # Single output for temperature prediction
])

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Train the model
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_test, y_test),
    verbose=1
)

# Make predictions
predictions = model.predict(X_test)

# Inverse transform to get actual temperature values
# Create a placeholder array matching the original data shape
temp_array = np.zeros((len(predictions), len(features)))
# Put predictions in the temperature column (index 0)
temp_array[:, 0] = predictions.flatten()
# Inverse transform
predicted_temperatures = scaler.inverse_transform(temp_array)[:, 0]

# Similarly for actual values
temp_array = np.zeros((len(y_test), len(features)))
temp_array[:, 0] = y_test
actual_temperatures = scaler.inverse_transform(temp_array)[:, 0]

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(actual_temperatures, predicted_temperatures))
print(f"Root Mean Squared Error: {rmse:.2f} degrees Celsius")

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(actual_temperatures, label='Actual Temperature')
plt.plot(predicted_temperatures, label='Predicted Temperature')
plt.title('LSTM Temperature Prediction')
plt.xlabel('Time Steps')
plt.ylabel('Temperature (°C)')
plt.legend()
plt.grid(True)
plt.show()
""", language="python")
            
            # Advantages and limitations
            st.write("#### Advantages and Limitations of RNNs for Fire Safety")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Advantages:**")
                st.write("""
                - Can capture temporal patterns in fire dynamics
                - Effective with multivariate time series from multiple sensors
                - Ability to forecast future fire conditions
                - Can handle variable-length sequences
                - Good for early warning systems
                """)
            
            with col2:
                st.write("**Limitations:**")
                st.write("""
                - Require substantial training data
                - May struggle with very long sequences
                - Computationally intensive
                - Need careful tuning to avoid vanishing/exploding gradients
                - Performance depends on quality of historical data
                """)
    
    # Convolutional Neural Networks tab
    with tabs[2]:
        st.write("### Convolutional Neural Networks (CNNs)")
        
        st.write("""
        Convolutional Neural Networks excel at processing grid-like data such as images and multi-channel 
        sensor readings. Their architecture is particularly suited for recognizing spatial patterns 
        relevant to fire detection and analysis.
        
        **Key Concepts:**
        - Specialized for grid-like data (images, spatial data)
        - Uses convolutional filters to detect features
        - Hierarchical feature learning
        - Parameter sharing reduces model complexity
        - Pooling layers reduce spatial dimensions
        
        **Fire Safety Applications:**
        - Thermal image analysis for hotspot detection
        - Multispectral fire detection
        - Fire scene recognition and classification
        - Smoke pattern recognition in CCTV footage
        """)
        
        # How CNNs work
        st.write("#### How Convolutional Neural Networks Work")
        
        st.write("""
        CNNs process images through a series of specialized layers:
        
        1. **Convolutional Layers**: Apply filters to detect features like edges, textures, and patterns
        2. **Activation Functions**: Introduce non-linearity (typically ReLU)
        3. **Pooling Layers**: Reduce spatial dimensions while preserving important features
        4. **Fully Connected Layers**: Interpret extracted features for classification
        
        For fire detection, early convolutional layers may detect simple features like edges and color patterns, 
        while deeper layers recognize complex patterns specific to fire and smoke.
        """)
        
        # Visual explanation of CNN layers
        st.write("#### CNN Layer Operations")
        
        st.components.v1.html("""
        <div style="display: flex; justify-content: space-between; margin-bottom: 20px;">
            <!-- Original Image -->
            <div style="text-align: center; width: 30%;">
                <div style="border: 1px solid #ddd; padding: 5px; margin-bottom: 5px;">
                    <svg width="100%" height="150" xmlns="http://www.w3.org/2000/svg">
                        <rect width="100%" height="100%" fill="#000"/>
                        <path d="M40,120 C 40,70 60,80 55,40 C 70,70 75,60 75,100 C 85,70 85,110 90,80 C 100,110 95,120 110,110 C 110,120 100,130 90,120 C 80,130 70,120 40,120 Z" fill="#FF4B4B"/>
                        <path d="M55,115 C 55,80 65,85 62,60 C 70,80 72,70 72,100 C 77,80 77,100 80,85 C 85,100 82,115 90,110 C 90,115 85,120 80,115 C 75,120 70,115 55,115 Z" fill="#FFA500"/>
                    </svg>
                </div>
                <div><strong>Input Image</strong></div>
                <div style="font-size: 0.8em; color: #666;">224×224×3</div>
            </div>
            
            <!-- Convolutional Layer -->
            <div style="text-align: center; width: 30%;">
                <div style="border: 1px solid #ddd; padding: 5px; margin-bottom: 5px;">
                    <svg width="100%" height="150" xmlns="http://www.w3.org/2000/svg">
                        <defs>
                            <pattern id="convPattern" patternUnits="userSpaceOnUse" width="20" height="20">
                                <rect width="10" height="10" fill="#FF4B4B" fill-opacity="0.5"/>
                                <rect x="10" y="0" width="10" height="10" fill="#FF7B4B" fill-opacity="0.3"/>
                                <rect x="0" y="10" width="10" height="10" fill="#FF7B4B" fill-opacity="0.3"/>
                                <rect x="10" y="10" width="10" height="10" fill="#FF4B4B" fill-opacity="0.4"/>
                            </pattern>
                        </defs>
                        <rect width="100%" height="100%" fill="url(#convPattern)"/>
                        <!-- Simplified visualization of filters -->
                        <rect x="20%" y="20%" width="20%" height="20%" stroke="#FF4B4B" stroke-width="2" fill="none"/>
                        <rect x="50%" y="40%" width="20%" height="20%" stroke="#FF4B4B" stroke-width="2" fill="none"/>
                    </svg>
                </div>
                <div><strong>Convolutional Layer</strong></div>
                <div style="font-size: 0.8em; color: #666;">Applies filters to detect features</div>
            </div>
            
            <!-- Pooling Layer -->
            <div style="text-align: center; width: 30%;">
                <div style="border: 1px solid #ddd; padding: 5px; margin-bottom: 5px;">
                    <svg width="100%" height="150" xmlns="http://www.w3.org/2000/svg">
                        <defs>
                            <pattern id="poolPattern" patternUnits="userSpaceOnUse" width="40" height="40">
                                <rect width="40" height="40" fill="#FFA500" fill-opacity="0.2"/>
                                <rect width="38" height="38" x="1" y="1" fill="#FF4B4B" fill-opacity="0.4"/>
                            </pattern>
                        </defs>
                        <rect width="100%" height="100%" fill="url(#poolPattern)"/>
                    </svg>
                </div>
                <div><strong>Pooling Layer</strong></div>
                <div style="font-size: 0.8em; color: #666;">Reduces dimensions while preserving features</div>
            </div>
        </div>
        """, height=200)
        
        # Practical example
        st.write("#### Practical Example: Thermal Image Analysis for Fire Detection")
        
        st.write("""
        CNNs can analyze thermal images to detect potential fire hazards before visible flames appear. 
        This example shows how CNN-based detection works on thermal imagery.
        """)
        
        # Create a visualization of thermal image analysis
        col1, col2 = st.columns(2)
        
        with col1:
            st.components.v1.html("""
            <div style="border: 2px solid #FF4B4B; border-radius: 5px; padding: 10px;">
                <div style="text-align: center; font-weight: bold; margin-bottom: 10px;">Thermal Hotspot Detected</div>
                <svg width="100%" height="200" xmlns="http://www.w3.org/2000/svg">
                    <!-- Thermal image visualization -->
                    <rect width="100%" height="100%" fill="#000"/>
                    
                    <!-- Cool to hot gradient (blue to red) areas -->
                    <circle cx="75" cy="75" r="50" fill="#0000FF" opacity="0.3"/>
                    <circle cx="150" cy="100" r="40" fill="#00FFFF" opacity="0.3"/>
                    <circle cx="200" cy="120" r="30" fill="#00FF00" opacity="0.3"/>
                    <circle cx="120" cy="150" r="25" fill="#FFFF00" opacity="0.3"/>
                    
                    <!-- Hot spot (potential fire) -->
                    <circle cx="180" cy="60" r="20" fill="#FF0000" opacity="0.8"/>
                    
                    <!-- Bounding box around hotspot -->
                    <rect x="155" y="35" width="50" height="50" stroke="#FF4B4B" stroke-width="2" fill="none"/>
                    <text x="155" y="30" font-family="Arial" font-size="10" fill="white">Temp: 115°C</text>
                </svg>
                <div style="text-align: center; font-style: italic; margin-top: 5px;">Thermal image with hotspot detection</div>
            </div>
            """, height=270)
        
        with col2:
            st.components.v1.html("""
            <div style="border: 2px solid #1E88E5; border-radius: 5px; padding: 10px;">
                <div style="text-align: center; font-weight: bold; margin-bottom: 10px;">Temperature Heatmap</div>
                <svg width="100%" height="200" xmlns="http://www.w3.org/2000/svg">
                    <!-- Heat map visualization -->
                    <defs>
                        <linearGradient id="heatGradient" x1="0%" y1="0%" x2="100%" y2="0%">
                            <stop offset="0%" style="stop-color:#0000FF;stop-opacity:1" />
                            <stop offset="25%" style="stop-color:#00FFFF;stop-opacity:1" />
                            <stop offset="50%" style="stop-color:#00FF00;stop-opacity:1" />
                            <stop offset="75%" style="stop-color:#FFFF00;stop-opacity:1" />
                            <stop offset="100%" style="stop-color:#FF0000;stop-opacity:1" />
                        </linearGradient>
                    </defs>
                    
                    <!-- Temperature scale -->
                    <rect x="20" y="170" width="260" height="10" fill="url(#heatGradient)"/>
                    <text x="20" y="190" font-family="Arial" font-size="8" fill="black">20°C</text>
                    <text x="150" y="190" font-family="Arial" font-size="8" fill="black">70°C</text>
                    <text x="280" y="190" font-family="Arial" font-size="8" fill="black">120°C</text>
                    
                    <!-- CNN detection confidence visualization -->
                    <rect x="50" y="20" width="200" height="30" rx="5" fill="#F5F5F5" stroke="#333" stroke-width="1"/>
                    <rect x="50" y="20" width="170" height="30" rx="5" fill="#FF4B4B"/>
                    <text x="150" y="40" font-family="Arial" font-size="14" text-anchor="middle" fill="white">Fire Risk: 85%</text>
                    
                    <!-- Temperature readings -->
                    <text x="50" y="80" font-family="Arial" font-size="12" fill="black">Peak Temperature: 115°C</text>
                    <text x="50" y="100" font-family="Arial" font-size="12" fill="black">Average Temp: 47°C</text>
                    <text x="50" y="120" font-family="Arial" font-size="12" fill="black">Rate of Change: +12°C/min</text>
                </svg>
                <div style="text-align: center; font-style: italic; margin-top: 5px;">CNN analysis results</div>
            </div>
            """, height=270)
        
        # CNN model for thermal image analysis
        st.write("#### CNN Architecture for Thermal Image Analysis")
        
        st.write("""
        A CNN for thermal image analysis typically includes:
        
        1. **Input Layer**: Thermal images (often single-channel grayscale)
        2. **Multiple Convolutional Blocks**: Each with convolution, activation, and pooling layers
        3. **Feature Maps**: Detecting temperature gradients and hotspots
        4. **Classification Head**: Identifying fire risks based on thermal patterns
        
        The model is trained on thousands of labeled thermal images to recognize the temperature patterns 
        associated with fire hazards in their early stages.
        """)
        
        # CNN-based fire detection system
        st.write("#### Implementing a CNN-based Fire Detection System")
        
        st.write("""
        A complete CNN-based fire detection system involves:
        
        1. **Data Collection**: Gathering diverse thermal images in various conditions
        2. **Data Preprocessing**: Normalizing and augmenting thermal images
        3. **Model Training**: Training the CNN on labeled images
        4. **Deployment**: Installing the model on thermal camera systems
        5. **Alert Integration**: Connecting detection results to alarm systems
        6. **Continuous Learning**: Updating the model with new data
        
        Such systems can detect potential fires up to several minutes before conventional smoke detectors,
        providing crucial extra time for evacuation and firefighting response.
        """)
        
        # Code example for thermal image CNN
        with st.expander("Python Code: CNN for Thermal Image Analysis"):
            st.code("""
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt

# Define model parameters
img_width, img_height = 224, 224
input_shape = (img_width, img_height, 1)  # Single channel for thermal images
batch_size = 32

# Data augmentation for thermal images
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Validation data generator
validation_datagen = ImageDataGenerator(rescale=1./255)

# Create data generators
# Note: For thermal images, you'd typically use grayscale mode
train_generator = train_datagen.flow_from_directory(
    'thermal_dataset/train',
    target_size=(img_width, img_height),
    color_mode='grayscale',
    batch_size=batch_size,
    class_mode='binary'  # 'fire_risk' or 'no_risk'
)

validation_generator = validation_datagen.flow_from_directory(
    'thermal_dataset/validation',
    target_size=(img_width, img_height),
    color_mode='grayscale',
    batch_size=batch_size,
    class_mode='binary'
)

# Create the CNN model
model = Sequential([
    # First convolutional block
    Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2)),
    
    # Second convolutional block
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2)),
    
    # Third convolutional block
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2)),
    
    # Fully connected layers
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Binary classification
])

# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
)

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=30,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size
)

# Save the model
model.save('thermal_fire_detection.h5')

# Visualize training history
plt.figure(figsize=(12, 4))

# Plot accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='lower right')

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper right')

plt.tight_layout()
plt.show()

# Function to analyze a new thermal image
def analyze_thermal_image(image_path, model):
    # Load and preprocess image
    img = tf.keras.preprocessing.image.load_img(
        image_path, target_size=(img_width, img_height), color_mode='grayscale'
    )
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Make prediction
    prediction = model.predict(img_array)[0][0]
    
    # Display results
    plt.figure(figsize=(8, 6))
    plt.imshow(np.squeeze(img_array), cmap='inferno')
    plt.title(f'Fire Risk Probability: {prediction:.2%}')
    plt.colorbar(label='Temperature')
    plt.axis('off')
    plt.show()
    
    return prediction
""", language="python")
    
    # Transfer Learning tab
    with tabs[3]:
        st.write("### Transfer Learning for Fire Safety")
        
        st.write("""
        Transfer learning leverages knowledge from pre-trained models to improve performance on fire safety 
        tasks, especially when limited labeled data is available.
        
        **Key Concepts:**
        - Uses models pre-trained on large datasets
        - Fine-tunes the models for specific fire safety tasks
        - Reduces training time and data requirements
        - Improves performance on small datasets
        - Enables rapid development of specialized models
        
        **Fire Safety Applications:**
        - Adapting general image classification models for fire detection
        - Using pre-trained models for thermal image analysis
        - Building on existing smoke detection systems
        - Creating specialized models for unique fire environments
        """)
        
        # Explain transfer learning
        st.write("#### How Transfer Learning Works")
        
        st.components.v1.html("""
        <svg width="750" height="280" xmlns="http://www.w3.org/2000/svg">
            <!-- Base Models -->
            <rect x="50" y="50" width="200" height="200" rx="10" fill="#F0F2F6" stroke="#262730" stroke-width="2"/>
            <text x="150" y="30" text-anchor="middle" font-family="Arial" font-size="16" font-weight="bold">Pre-trained Model</text>
            <text x="150" y="120" text-anchor="middle" font-family="Arial" font-size="14">Trained on</text>
            <text x="150" y="140" text-anchor="middle" font-family="Arial" font-size="14">ImageNet</text>
            <text x="150" y="160" text-anchor="middle" font-family="Arial" font-size="14">(1.4M Images)</text>
            <text x="150" y="180" text-anchor="middle" font-family="Arial" font-size="14">1000 Classes</text>
            
            <!-- Transfer Process -->
            <path d="M250,150 L350,150" stroke="#FF4B4B" stroke-width="3" />
            <polygon points="345,145 355,150 345,155" fill="#FF4B4B" />
            <text x="300" y="130" text-anchor="middle" font-family="Arial" font-size="14">Transfer</text>
            <text x="300" y="145" text-anchor="middle" font-family="Arial" font-size="14">Learned</text>
            <text x="300" y="160" text-anchor="middle" font-family="Arial" font-size="14">Features</text>
            
            <!-- Fine-tuned Model -->
            <rect x="350" y="50" width="200" height="200" rx="10" fill="#F0F2F6" stroke="#FF4B4B" stroke-width="2"/>
            <text x="450" y="30" text-anchor="middle" font-family="Arial" font-size="16" font-weight="bold">Fine-tuned Model</text>
            <text x="450" y="90" text-anchor="middle" font-family="Arial" font-size="14">Frozen Base</text>
            <text x="450" y="110" text-anchor="middle" font-family="Arial" font-size="14">Layers</text>
            <path d="M375,130 L525,130" stroke="#262730" stroke-width="1" stroke-dasharray="5,2" />
            <text x="450" y="150" text-anchor="middle" font-family="Arial" font-size="14">New Classification</text>
            <text x="450" y="170" text-anchor="middle" font-family="Arial" font-size="14">Layers</text>
            <text x="450" y="190" text-anchor="middle" font-family="Arial" font-size="14">("Fire" / "No Fire")</text>
            
            <!-- Training Process -->
            <path d="M550,150 L650,150" stroke="#FF4B4B" stroke-width="3" />
            <polygon points="645,145 655,150 645,155" fill="#FF4B4B" />
            <text x="600" y="130" text-anchor="middle" font-family="Arial" font-size="14">Fine-tune</text>
            <text x="600" y="145" text-anchor="middle" font-family="Arial" font-size="14">on</text>
            <text x="600" y="160" text-anchor="middle" font-family="Arial" font-size="14">Fire Dataset</text>
            
            <!-- Fire Dataset -->
            <rect x="650" y="50" width="100" height="200" rx="10" fill="#F0F2F6" stroke="#262730" stroke-width="2"/>
            <text x="700" y="30" text-anchor="middle" font-family="Arial" font-size="16" font-weight="bold">Fire Dataset</text>
            
            <!-- Small dataset visualization -->
            <rect x="670" y="70" width="60" height="40" fill="#FF4B4B" rx="5"/>
            <text x="700" y="90" text-anchor="middle" font-family="Arial" font-size="10" fill="white">Fire</text>
            <text x="700" y="105" text-anchor="middle" font-family="Arial" font-size="10" fill="white">Images</text>
            
            <rect x="670" y="120" width="60" height="40" fill="#1E88E5" rx="5"/>
            <text x="700" y="140" text-anchor="middle" font-family="Arial" font-size="10" fill="white">Non-Fire</text>
            <text x="700" y="155" text-anchor="middle" font-family="Arial" font-size="10" fill="white">Images</text>
            
            <text x="700" y="190" text-anchor="middle" font-family="Arial" font-size="14">500-1000</text>
            <text x="700" y="210" text-anchor="middle" font-family="Arial" font-size="14">Images</text>
        </svg>
        """, height=280)
        
        st.write("""
        Transfer learning involves two main steps:
        
        1. **Feature Extraction**: Using pre-trained models (like ResNet, VGG, or MobileNet) as fixed feature extractors
        2. **Fine-tuning**: Adapting some or all of the pre-trained model's parameters to the new task
        
        This approach is particularly valuable in fire safety applications where labeled data may be limited,
        such as fire detection in specialized environments (industrial settings, forests, etc.).
        """)
        
        # Practical example
        st.write("#### Practical Example: Transfer Learning for Wildfire Detection")
        
        st.write("""
        Transfer learning can be applied to detect wildfires from aerial or satellite imagery by 
        leveraging models pre-trained on general image datasets.
        """)
        
        # Sample code for transfer learning
        with st.expander("Python Code: Transfer Learning for Wildfire Detection"):
            st.code("""
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# Image dimensions and batch size
img_height, img_width = 224, 224
batch_size = 32

# Data augmentation for training
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Just rescaling for validation
valid_datagen = ImageDataGenerator(rescale=1./255)

# Load training and validation data
train_generator = train_datagen.flow_from_directory(
    'wildfire_dataset/train',
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary'  # 'wildfire' or 'no_wildfire'
)

validation_generator = valid_datagen.flow_from_directory(
    'wildfire_dataset/validation',
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary'
)

# Load pre-trained MobileNetV2 without the classification head
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(img_height, img_width, 3)
)

# Freeze the base model layers
base_model.trainable = False

# Add new classification head
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(1, activation='sigmoid')(x)

# Create the transfer learning model
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
)

# First training phase: train only the top layers
history_top = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=10,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size
)

# Fine-tuning phase: unfreeze some of the base model layers
# Unfreeze the last 23 layers (last block of MobileNetV2)
for layer in base_model.layers[-23:]:
    layer.trainable = True

# Recompile the model with a lower learning rate
model.compile(
    optimizer=Adam(learning_rate=0.00001),  # Lower learning rate for fine-tuning
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
)

# Continue training with fine-tuning
history_ft = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=20,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size
)

# Save the model
model.save('wildfire_detection_model.h5')

# Plot training history (combining both training phases)
def plot_training_history(history_top, history_ft):
    # Combine histories
    acc = history_top.history['accuracy'] + history_ft.history['accuracy']
    val_acc = history_top.history['val_accuracy'] + history_ft.history['val_accuracy']
    loss = history_top.history['loss'] + history_ft.history['loss']
    val_loss = history_top.history['val_loss'] + history_ft.history['val_loss']
    
    epochs = range(1, len(acc) + 1)
    
    plt.figure(figsize=(12, 5))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'b-', label='Training Accuracy')
    plt.plot(epochs, val_acc, 'r-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Add vertical line to mark the start of fine-tuning
    plt.axvline(x=10, color='green', linestyle='--')
    plt.text(10, min(acc), 'Start Fine-tuning', rotation=90, verticalalignment='bottom')
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'b-', label='Training Loss')
    plt.plot(epochs, val_loss, 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # Add vertical line to mark the start of fine-tuning
    plt.axvline(x=10, color='green', linestyle='--')
    plt.text(10, max(loss), 'Start Fine-tuning', rotation=90, verticalalignment='top')
    
    plt.tight_layout()
    plt.show()

# Plot the training history
plot_training_history(history_top, history_ft)
""", language="python")
        
        # Benefits of transfer learning
        st.write("#### Benefits of Transfer Learning in Fire Safety")
        
        st.write("""
        Transfer learning offers significant advantages for fire safety applications:
        
        1. **Efficient Use of Limited Data**: Fire incident imagery is often scarce, making it hard to train 
           deep models from scratch. Transfer learning leverages knowledge from pre-trained models to achieve 
           good performance with smaller datasets.
        
        2. **Faster Development**: Building on pre-trained models dramatically reduces training time and 
           computational requirements, enabling faster deployment of fire detection systems.
        
        3. **Better Generalization**: Models pre-trained on diverse datasets often generalize better to new 
           situations, which is crucial for robust fire detection across varying environments and conditions.
        
        4. **Lower Resource Requirements**: Transfer learning typically requires less computing power and time, 
           making it accessible for organizations with limited resources.
        """)
        
        # Transfer learning architectures
        st.write("#### Popular Transfer Learning Architectures for Fire Safety")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("**ResNet**")
            st.write("""
            - Deep residual networks
            - Good for detailed image analysis
            - Excellent feature extractors
            - Used for high-accuracy fire detection
            """)
        
        with col2:
            st.write("**MobileNet**")
            st.write("""
            - Lightweight architecture
            - Optimized for mobile devices
            - Fast inference time
            - Ideal for real-time fire monitoring
            """)
        
        with col3:
            st.write("**EfficientNet**")
            st.write("""
            - Optimized model scaling
            - Balanced depth/width/resolution
            - State-of-the-art performance
            - Good for resource-constrained systems
            """)
    
    # Module quiz
    st.write("### Module 5 Quiz")
    
    questions = [
        {
            'question': 'Which deep learning architecture is most appropriate for analyzing sequences of sensor readings over time?',
            'options': [
                'Convolutional Neural Network (CNN)',
                'Recurrent Neural Network (RNN/LSTM)',
                'Generative Adversarial Network (GAN)',
                'Transformer'
            ],
            'correct': 'Recurrent Neural Network (RNN/LSTM)'
        },
        {
            'question': 'What is the primary advantage of using CNNs for fire detection in images?',
            'options': [
                'They require very little training data',
                'They can automatically extract spatial features and patterns',
                'They are simpler to implement than traditional algorithms',
                'They consume less computational resources than other methods'
            ],
            'correct': 'They can automatically extract spatial features and patterns'
        },
        {
            'question': 'In the context of LSTM networks for fire prediction, what does the "forget gate" do?',
            'options': [
                'It completely resets the network when a prediction is wrong',
                'It controls what information to discard from the cell state',
                'It forgets previous fire incidents to focus only on current data',
                'It removes unnecessary sensors from the monitoring system'
            ],
            'correct': 'It controls what information to discard from the cell state'
        },
        {
            'question': 'What is transfer learning in the context of fire detection systems?',
            'options': [
                'Transferring fire detection models from one building to another',
                'Using a pre-trained model as a starting point for a new task',
                'Moving computational resources from one system to another',
                'Converting fire detection algorithms between programming languages'
            ],
            'correct': 'Using a pre-trained model as a starting point for a new task'
        },
        {
            'question': 'When analyzing thermal images for fire detection, what type of neural network layer is typically used to extract spatial features?',
            'options': [
                'Dense (Fully Connected) Layer',
                'Convolutional Layer',
                'Recurrent Layer',
                'Normalization Layer'
            ],
            'correct': 'Convolutional Layer'
        },
        {
            'question': 'Which of these is NOT a typical processing step in a CNN architecture?',
            'options': [
                'Convolution',
                'Activation',
                'Pooling',
                'Memory gates'
            ],
            'correct': 'Memory gates'
        },
        {
            'question': 'What is a key advantage of using LSTM networks over standard RNNs for fire spread prediction?',
            'options': [
                'They are simpler and have fewer parameters',
                'They better handle long-term dependencies in time series data',
                'They always require less training data',
                'They are specifically designed for fire modeling'
            ],
            'correct': 'They better handle long-term dependencies in time series data'
        },
        {
            'question': 'Which statement best describes the benefit of fine-tuning in transfer learning for fire detection?',
            'options': [
                'It eliminates the need for any fire-specific training data',
                'It adapts the pre-trained model to the specific characteristics of fire imagery',
                'It makes the model run faster on fire detection hardware',
                'It ensures the model works equally well for all types of fires'
            ],
            'correct': 'It adapts the pre-trained model to the specific characteristics of fire imagery'
        },
        {
            'question': 'What is a common challenge when applying deep learning to fire detection?',
            'options': [
                'Fires move too quickly for neural networks to process',
                'Limited availability of labeled fire incident data',
                'Neural networks cannot detect the color red accurately',
                'Fire detection is too simple for deep learning to be necessary'
            ],
            'correct': 'Limited availability of labeled fire incident data'
        },
        {
            'question': 'How can deep learning improve early fire detection compared to traditional smoke detectors?',
            'options': [
                'By eliminating the need for physical sensors entirely',
                'By analyzing visual patterns that may indicate fire before smoke is dense enough to trigger alarms',
                'By directly extinguishing fires once detected',
                'By being completely immune to false alarms'
            ],
            'correct': 'By analyzing visual patterns that may indicate fire before smoke is dense enough to trigger alarms'
        }
    ]
    
    create_module_quiz(questions, 4)

