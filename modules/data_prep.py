import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from utils import load_dataset, create_module_quiz, create_code_editor

def show_module_content():
    """Display data preparation and processing content"""

    st.write("## Data Preparation and Processing for Fire Safety ML Applications")

    # Overview
    st.write("""
    ### Overview

    High-quality data is the foundation of successful machine learning models. In fire safety, 
    data comes from various sources and often requires significant preparation before it can be 
    used for model training. This module covers essential techniques for data collection, 
    cleaning, and preprocessing in the context of fire safety applications.
    """)

    # Data sources in fire safety
    with st.expander("Data Sources in Fire Safety"):
        st.write("""
        ### Common Data Sources

        Fire safety data comes from numerous sources, each with its own characteristics and challenges:

        1. **Fire Incident Reports**
           - Historical records of fire incidents
           - Information about cause, damage, response time, etc.
           - Often maintained by fire departments and government agencies

        2. **Building Information**
           - Structural details, age, materials, occupancy
           - Fire protection systems (sprinklers, alarms)
           - Building code compliance records

        3. **Sensor Data**
           - Smoke detectors, heat sensors, carbon monoxide detectors
           - Real-time monitoring systems
           - IoT devices in smart buildings

        4. **Environmental Data**
           - Weather conditions (temperature, humidity, wind)
           - Seasonal variations
           - Climate data for long-term analysis

        5. **Human Behavior Data**
           - Evacuation patterns
           - Occupancy levels at different times
           - Human factors in fire incidents

        6. **Geographic Information**
           - Building locations
           - Proximity to fire stations
           - Urban vs. rural settings

        ### Data Collection Challenges

        Collecting fire safety data presents several challenges:

        - **Data Silos**: Information often exists in separate systems
        - **Standardization**: Different reporting formats across jurisdictions
        - **Privacy Concerns**: Balancing data needs with privacy requirements
        - **Historical Limitations**: Older data may be incomplete or paper-based
        - **Sensor Reliability**: Ensuring accurate readings from detection systems
        """)

    # Data cleaning and preprocessing
    with st.expander("Data Cleaning and Preprocessing"):
        st.write("""
        ### Common Data Issues

        Fire safety datasets often suffer from several issues that need addressing:

        1. **Missing Values**
           - Incomplete incident reports
           - Sensor failures or maintenance periods
           - Unreported building characteristics

        2. **Outliers**
           - Extreme fire events
           - Sensor malfunctions
           - Reporting errors

        3. **Inconsistent Formats**
           - Different units of measurement
           - Varied date formats
           - Inconsistent categorization schemes

        4. **Data Imbalance**
           - More data on minor incidents than major fires
           - Overrepresentation of certain building types
           - Seasonal imbalances

        ### Preprocessing Techniques

        Key techniques for preparing fire safety data:

        1. **Handling Missing Values**
           - Imputation based on similar incidents or buildings
           - Indicator variables for missingness patterns
           - Removal of entries with critical missing information

        2. **Outlier Treatment**
           - Statistical methods (z-scores, IQR)
           - Domain knowledge to identify valid extremes
           - Transformations to reduce outlier impact

        3. **Standardization and Normalization**
           - Converting measurements to consistent units
           - Scaling numerical features
           - Encoding categorical variables

        4. **Temporal Alignment**
           - Synchronizing time-series data from different sources
           - Creating consistent time windows for analysis
           - Handling time zones and daylight saving changes
        """)

    # Feature engineering
    with st.expander("Feature Engineering for Fire Safety"):
        st.write("""
        ### Feature Engineering Concepts

        Feature engineering transforms raw data into meaningful inputs for ML models:

        1. **Creating Derived Features**
           - Response time from detection to arrival
           - Building age from construction date
           - Fire load density from materials and contents

        2. **Temporal Features**
           - Time of day, day of week, season
           - Time since last inspection
           - Maintenance cycles of safety equipment

        3. **Spatial Features**
           - Distance to nearest fire station
           - Building density in the area
           - Access routes for emergency vehicles

        4. **Risk Indicators**
           - Aggregated risk scores
           - Compliance indices
           - Historical incident frequencies

        ### Domain-Specific Feature Creation

        Fire safety experts can guide feature creation:

        - **Fire Dynamics Features**: Variables related to fire physics
        - **Building Vulnerability Factors**: Structural characteristics affecting fire spread
        - **Occupancy Risk Factors**: How building usage affects fire risk
        - **Response Capability Measures**: Features describing firefighting resources
        """)

    # Practical example
    st.write("### Practical Example: Preprocessing Fire Sensor Data")

    # Load dataset
    df_sensors = load_dataset("sensor_readings.csv")

    if not df_sensors.empty:
        # Display original data
        st.write("#### Original Sensor Data")
        st.write(df_sensors.head())

        # Display data information
        st.write("#### Dataset Information")

        # Basic stats
        stats = pd.DataFrame({
            'Data Type': df_sensors.dtypes,
            'Non-Null Count': df_sensors.count(),
            'Null Count': df_sensors.isnull().sum(),
            'Unique Values': [df_sensors[col].nunique() for col in df_sensors.columns]
        })

        st.write(stats)

        # Step 1: Identify and handle missing values
        st.write("#### Step 1: Identify and Handle Missing Values")

        # Artificially introduce some missing values for the example
        df_clean = df_sensors.copy()
        np.random.seed(42)
        mask = np.random.random(df_clean.shape) < 0.05
        df_clean[mask] = np.nan

        # Display data with missing values
        st.write("Data with artificially introduced missing values (5%):")
        st.write(df_clean.head(10))

        # Show missing values count
        missing_values = df_clean.isnull().sum()
        st.write("Missing values count per column:")
        st.write(missing_values)

        # Handling missing values
        st.write("Handling missing values:")
        st.code("""
# Fill missing numerical values with column means
numerical_cols = ['temperature_celsius', 'smoke_density', 'carbon_monoxide_ppm']
for col in numerical_cols:
    df_clean[col] = df_clean[col].fillna(df_clean[col].mean())

# Fill missing categorical values with mode
categorical_cols = ['sensor_id', 'building_id', 'alarm_triggered']
for col in categorical_cols:
    df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0]).infer_objects()
        """)

        # Actually perform the imputation
        numerical_cols = ['temperature_celsius', 'smoke_density', 'carbon_monoxide_ppm']
        for col in numerical_cols:
            df_clean[col] = df_clean[col].fillna(df_clean[col].mean())

        categorical_cols = ['sensor_id', 'building_id', 'alarm_triggered']
        for col in categorical_cols:
            df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0]).infer_objects()

        # Show data after imputation
        st.write("Data after imputation:")
        st.write(df_clean.head(10))
        st.write(f"Missing values remaining: {df_clean.isnull().sum().sum()}")

        # Step 2: Detecting and handling outliers
        st.write("#### Step 2: Detecting and Handling Outliers")

        # Create a visualization of the distribution with outliers - с более яркими цветами
        fig = px.box(
            df_clean, 
            y=['temperature_celsius', 'smoke_density', 'carbon_monoxide_ppm'],
            title='Distribution of Sensor Readings with Outliers',
            color_discrete_sequence=['#FF5722', '#FF9800', '#F44336']
        )
        # Улучшаем видимость графика
        fig.update_layout(
            plot_bgcolor='rgba(240, 240, 240, 0.8)',
            font=dict(size=14),
            title_font=dict(size=20, color='#333333'),
            legend_title_font=dict(size=16),
            legend_font=dict(size=14)
        )

        st.plotly_chart(fig, use_container_width=True)

        # Show code for handling outliers
        st.write("Code for detecting and capping outliers:")
        st.code("""
# Define a function to cap outliers using IQR method
def cap_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Cap the outliers
    df[column] = np.where(
        df[column] < lower_bound,
        lower_bound,
        np.where(
            df[column] > upper_bound,
            upper_bound,
            df[column]
        )
    )
    return df

# Apply to each numerical column
for col in numerical_cols:
    df_clean = cap_outliers(df_clean, col)
        """)

        # Define the function to cap outliers
        def cap_outliers(df, column):
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            # Cap the outliers
            df[column] = np.where(
                df[column] < lower_bound,
                lower_bound,
                np.where(
                    df[column] > upper_bound,
                    upper_bound,
                    df[column]
                )
            )
            return df

        # Apply to each numerical column
        for col in numerical_cols:
            df_clean = cap_outliers(df_clean, col)

        # Show distribution after handling outliers - с более яркими цветами
        fig_after = px.box(
            df_clean, 
            y=['temperature_celsius', 'smoke_density', 'carbon_monoxide_ppm'],
            title='Distribution of Sensor Readings After Handling Outliers',
            color_discrete_sequence=['#FF5722', '#FF9800', '#F44336']
        )
        # Улучшаем видимость графика
        fig_after.update_layout(
            plot_bgcolor='rgba(240, 240, 240, 0.8)',
            font=dict(size=14),
            title_font=dict(size=20, color='#333333'),
            legend_title_font=dict(size=16),
            legend_font=dict(size=14)
        )

        st.plotly_chart(fig_after, use_container_width=True)

        # Step 3: Feature Engineering
        st.write("#### Step 3: Feature Engineering")

        st.write("""
        Let's create new features that might be useful for fire detection:

        1. **Alarm Ratio** - Ratio of alarms triggered to total readings for each sensor
        2. **Temperature-Smoke Ratio** - Relationship between temperature and smoke density
        3. **Combined Risk Score** - Weighted combination of all sensor readings
        """)

        # Show code for feature engineering
        st.code("""
# Create a sensor alarm ratio feature
sensor_alarm_counts = df_clean.groupby('sensor_id')['alarm_triggered'].agg(['sum', 'count'])
sensor_alarm_counts['alarm_ratio'] = sensor_alarm_counts['sum'] / sensor_alarm_counts['count']

# Merge back to main dataframe
df_clean = df_clean.merge(
    sensor_alarm_counts['alarm_ratio'], 
    left_on='sensor_id', 
    right_index=True
)

# Create temperature-smoke ratio
df_clean['temp_smoke_ratio'] = df_clean['temperature_celsius'] / (df_clean['smoke_density'] + 0.1)

# Create combined risk score (simplified example)
df_clean['risk_score'] = (
    (df_clean['temperature_celsius'] / df_clean['temperature_celsius'].max()) * 0.4 + 
    (df_clean['smoke_density'] / df_clean['smoke_density'].max()) * 0.3 + 
    (df_clean['carbon_monoxide_ppm'] / df_clean['carbon_monoxide_ppm'].max()) * 0.3
)
        """)

        # Actually create the features
        # Create a sensor alarm ratio feature
        sensor_alarm_counts = df_clean.groupby('sensor_id')['alarm_triggered'].agg(['sum', 'count'])
        sensor_alarm_counts['alarm_ratio'] = sensor_alarm_counts['sum'] / sensor_alarm_counts['count']

        # Merge back to main dataframe
        df_clean = df_clean.merge(
            sensor_alarm_counts['alarm_ratio'], 
            left_on='sensor_id', 
            right_index=True
        )

        # Create temperature-smoke ratio
        df_clean['temp_smoke_ratio'] = df_clean['temperature_celsius'] / (df_clean['smoke_density'] + 0.1)

        # Create combined risk score
        df_clean['risk_score'] = (
            (df_clean['temperature_celsius'] / df_clean['temperature_celsius'].max()) * 0.4 + 
            (df_clean['smoke_density'] / df_clean['smoke_density'].max()) * 0.3 + 
            (df_clean['carbon_monoxide_ppm'] / df_clean['carbon_monoxide_ppm'].max()) * 0.3
        )

        # Show data with new features
        st.write("Data with engineered features:")
        st.write(df_clean.head())

        # Visualize the new features
        st.write("Visualization of engineered features:")

        # Plot risk score vs alarm_triggered - с более яркими цветами
        fig_risk = px.scatter(
            df_clean, 
            x='risk_score', 
            y='temp_smoke_ratio',
            color='alarm_triggered',
            title='Risk Score vs Temperature-Smoke Ratio',
            color_discrete_sequence=['#2962FF', '#FF1744'],  # Более яркие насыщенные цвета
            labels={'alarm_triggered': 'Alarm Triggered'},
            opacity=0.8,
            size_max=15,
            template="plotly_white"
        )
        # Улучшаем видимость графика
        fig_risk.update_layout(
            plot_bgcolor='rgba(240, 240, 240, 0.8)',
            font=dict(size=14),
            title_font=dict(size=20, color='#333333'),
            legend_title_font=dict(size=16),
            legend_font=dict(size=14)
        )
        # Увеличиваем размер маркеров
        fig_risk.update_traces(marker=dict(size=12))

        st.plotly_chart(fig_risk, use_container_width=True)

        # Step 4: Feature scaling
        st.write("#### Step 4: Feature Scaling")

        st.write("""
        Before feeding data to machine learning models, we often need to scale the features to ensure they're on similar scales.
        Common scaling methods include:

        1. **Standardization (Z-score)**: Transforms features to have mean=0 and standard deviation=1
        2. **Normalization (Min-Max)**: Scales features to a fixed range, typically [0,1]
        """)

        # Show code for feature scaling
        st.code("""
# Standardization (Z-score normalization)
from sklearn.preprocessing import StandardScaler

# Select numerical features for scaling
features_to_scale = ['temperature_celsius', 'smoke_density', 'carbon_monoxide_ppm', 
                    'temp_smoke_ratio', 'risk_score']

# Initialize the scaler
scaler = StandardScaler()

# Apply standardization
df_clean[features_to_scale + '_scaled'] = scaler.fit_transform(df_clean[features_to_scale])

# Alternatively, min-max scaling
from sklearn.preprocessing import MinMaxScaler

# Initialize min-max scaler
min_max_scaler = MinMaxScaler()

# Apply min-max scaling
df_clean[features_to_scale + '_normalized'] = min_max_scaler.fit_transform(df_clean[features_to_scale])
        """)

        # Interactive code editor
        st.write("#### Try it yourself - Write code to handle missing values")

        code_sample = """# Complete the following code to handle missing values in the dataframe
import pandas as pd
import numpy as np

def handle_missing_values(df):
    # TODO: Fill missing values in numerical columns with the mean
    # TODO: Fill missing values in categorical columns with the mode

    # Return the cleaned dataframe
    return df

# Test your function
# Sample data (example only)
data = {
    'temperature': [25, np.nan, 30, 22, np.nan],
    'smoke_level': [0.1, 0.2, np.nan, 0.3, 0.4],
    'sensor_type': ['A', 'B', np.nan, 'A', 'C']
}
df = pd.DataFrame(data)
cleaned_df = handle_missing_values(df)
print(cleaned_df)
"""

        edited_code = create_code_editor(code_sample, "python")

        # Final data ready for modeling
        st.write("#### Final Preprocessed Data")
        st.write("Your preprocessed data is now ready for model training:")
        st.write(df_clean.head())

        # Exportable code
        with st.expander("Complete Data Preprocessing Pipeline Code"):
            st.code("""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def preprocess_fire_sensor_data(df):
    \"\"\"
    Complete preprocessing pipeline for fire sensor data.

    Parameters:
    -----------
    df : pandas.DataFrame
        Raw sensor data

    Returns:
    --------
    pandas.DataFrame
        Preprocessed data ready for modeling
    \"\"\"
    # Create a copy to avoid modifying the original
    df_clean = df.copy()

    # Step 1: Handle missing values
    numerical_cols = ['temperature_celsius', 'smoke_density', 'carbon_monoxide_ppm']
    for col in numerical_cols:
        df_clean[col] = df_clean[col].fillna(df_clean[col].mean())

    categorical_cols = ['sensor_id', 'building_id', 'alarm_triggered']
    for col in categorical_cols:
        df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0]).infer_objects()

    # Step 2: Handle outliers
    for col in numerical_cols:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        df_clean[col] = np.where(
            df_clean[col] < lower_bound,
            lower_bound,
            np.where(
                df_clean[col] > upper_bound,
                upper_bound,
                df_clean[col]
            )
        )

    # Step 3: Feature engineering
    # Create sensor alarm ratio
    sensor_alarm_counts = df_clean.groupby('sensor_id')['alarm_triggered'].agg(['sum', 'count'])
    sensor_alarm_counts['alarm_ratio'] = sensor_alarm_counts['sum'] / sensor_alarm_counts['count']
    df_clean = df_clean.merge(sensor_alarm_counts['alarm_ratio'], left_on='sensor_id', right_index=True)

    # Create temperature-smoke ratio
    df_clean['temp_smoke_ratio'] = df_clean['temperature_celsius'] / (df_clean['smoke_density'] + 0.1)

    # Create combined risk score
    df_clean['risk_score'] = (
        (df_clean['temperature_celsius'] / df_clean['temperature_celsius'].max()) * 0.4 + 
        (df_clean['smoke_density'] / df_clean['smoke_density'].max()) * 0.3 + 
        (df_clean['carbon_monoxide_ppm'] / df_clean['carbon_monoxide_ppm'].max()) * 0.3
    )

    # Step 4: Feature scaling
    features_to_scale = ['temperature_celsius', 'smoke_density', 'carbon_monoxide_ppm', 
                        'temp_smoke_ratio', 'risk_score']

    # Standardization
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df_clean[features_to_scale])

    # Create scaled features with new column names
    scaled_col_names = [f"{col}_scaled" for col in features_to_scale]
    df_scaled = pd.DataFrame(scaled_features, columns=scaled_col_names, index=df_clean.index)

    # Combine with original dataframe
    df_final = pd.concat([df_clean, df_scaled], axis=1)

    return df_final
            """)

    # Module quiz
    st.write("### Module 2 Quiz")

    questions = [
        {
            'question': 'Which of the following is NOT a common source of data in fire safety applications?',
            'options': [
                'Fire incident reports',
                'Building information records',
                'Social media opinions about fire departments',
                'Sensor data from smoke detectors'
            ],
            'correct': 'Social media opinions about fire departments'
        },
        {
            'question': 'What is a common challenge when collecting fire safety data?',
            'options': [
                'Data is too standardized across jurisdictions',
                'Too few sensors are available for monitoring',
                'Data exists in separate systems (data silos)',
                'Fire incidents are too frequent to record properly'
            ],
            'correct': 'Data exists in separate systems (data silos)'
        },
        {
            'question': 'Which method is commonly used to detect outliers in numerical data?',
            'options': [
                'Random sampling',
                'IQR (Interquartile Range) method',
                'Principal Component Analysis',
                'Regular expressions'
            ],
            'correct': 'IQR (Interquartile Range) method'
        },
        {
            'question': 'When handling missing values in fire sensor data, what approach might be inappropriate?',
            'options': [
                'Replacing missing temperature values with the sensor average',
                'Using the most common value (mode) for categorical variables',
                'Deleting entire rows with any missing values without consideration',
                'Using regression models to predict missing values based on other features'
            ],
            'correct': 'Deleting entire rows with any missing values without consideration'
        },
        {
            'question': 'Which of the following is an example of feature engineering in fire safety data?',
            'options': [
                'Converting all measurements to metric units',
                'Creating a "risk score" by combining temperature, smoke density, and CO levels',
                'Removing duplicate entries from the dataset',
                'Encrypting sensitive building information'
            ],
            'correct': 'Creating a "risk score" by combining temperature, smoke density, and CO levels'
        },
        {
            'question': 'Why is feature scaling important in machine learning for fire safety?',
            'options': [
                'It improves data security and privacy',
                'It makes the data more colorful in visualizations',
                'It ensures features with different scales contribute equally to the model',
                'It reduces the size of the dataset for faster processing'
            ],
            'correct': 'It ensures features with different scales contribute equally to the model'
        },
        {
            'question': 'What is a potential issue with fire incident data that might require special handling?',
            'options': [
                'Data imbalance due to the rarity of major fire events',
                'Too many fire events to process effectively',
                'Fire data is always perfectly recorded',
                'All fire departments use the same reporting format'
            ],
            'correct': 'Data imbalance due to the rarity of major fire events'
        },
        {
            'question': 'Which temporal feature might be valuable in fire risk prediction?',
            'options': [
                'The color of the building',
                'Time since last fire safety inspection',
                'The name of the building owner',
                'The energy efficiency rating'
            ],
            'correct': 'Time since last fire safety inspection'
        },
        {
            'question': 'When preprocessing fire sensor time-series data, which of the following is important?',
            'options': [
                'Ensuring all sensors have different calibrations',
                'Removing all data from sensors near windows',
                'Temporal alignment of data from different sources',
                'Converting all timestamps to hexadecimal format'
            ],
            'correct': 'Temporal alignment of data from different sources'
        },
        {
            'question': 'What does the term "feature engineering" refer to in the context of fire safety data?',
            'options': [
                'The physical design of fire sensors',
                'Creating new variables from existing data to improve model performance',
                'Engineering better fire prevention systems',
                'The process of hiring data scientists for fire departments'
            ],
            'correct': 'Creating new variables from existing data to improve model performance'
        }
    ]

    create_module_quiz(questions, 1)