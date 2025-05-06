import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
from utils import load_dataset, create_module_quiz, create_code_editor
from visualization import plot_linear_regression_example, plot_decision_tree_viz
from ml_utils import train_linear_regression, train_logistic_regression, train_random_forest, train_svm, train_naive_bayes

def show_module_content():
    """Display supervised learning algorithms content"""
    
    st.write("## Supervised Learning Algorithms for Fire Safety Applications")
    
    # Overview
    st.write("""
    ### Overview
    
    Supervised learning is a powerful approach where models learn from labeled examples to make predictions 
    or classify new data. In fire safety, supervised learning can help predict fire risks, classify building 
    safety levels, estimate response times, and much more. This module explores key supervised learning 
    algorithms and their applications in fire safety.
    """)
    
    # Tab-based navigation for different algorithms
    tabs = st.tabs([
        "Linear & Logistic Regression", 
        "Decision Trees & Random Forests",
        "Support Vector Machines",
        "Naive Bayes Methods"
    ])
    
    # Linear and Logistic Regression
    with tabs[0]:
        st.write("### Linear and Logistic Regression")
        
        st.write("""
        #### Linear Regression
        
        Linear regression models the relationship between a dependent variable and one or more independent 
        variables by fitting a linear equation to the data.
        
        **Key Concepts:**
        - Predicts continuous values
        - Assumes a linear relationship between features and target
        - Simple to understand and interpret
        - Computationally efficient
        
        **Fire Safety Applications:**
        - Predicting fire spread rate based on environmental factors
        - Estimating evacuation time based on building characteristics
        - Forecasting response time based on distance and traffic conditions
        - Modeling temperature rise during fire incidents
        """)
        
        # Practical example with fire data
        st.write("#### Practical Example: Predicting Fire Spread Time")
        
        # Load dataset
        df_incidents = load_dataset("fire_incidents.csv")
        
        if not df_incidents.empty:
            # Add a feature: time to extinguish from arrival
            df_incidents['detection_time_mins'] = df_incidents['detection_time'].apply(
                lambda x: int(str(x).split(':')[0]) * 60 + int(str(x).split(':')[1]) if pd.notna(x) and isinstance(x, str) else 0
            )
            df_incidents['arrival_time_mins'] = df_incidents['arrival_time'].apply(
                lambda x: int(str(x).split(':')[0]) * 60 + int(str(x).split(':')[1]) if pd.notna(x) and isinstance(x, str) else 0
            )
            df_incidents['extinguishing_time_mins'] = df_incidents['extinguishing_time'].apply(
                lambda x: int(str(x).split(':')[0]) * 60 + int(str(x).split(':')[1]) if pd.notna(x) and isinstance(x, str) else 0
            )
            
            # Calculate spread time (from detection to extinguishing)
            df_incidents['spread_time'] = df_incidents['extinguishing_time_mins'] - df_incidents['detection_time_mins']
            df_incidents.loc[df_incidents['spread_time'] < 0, 'spread_time'] += 24 * 60  # Fix day boundary
            
            # Calculate response time (from detection to arrival)
            df_incidents['response_time'] = df_incidents['arrival_time_mins'] - df_incidents['detection_time_mins']
            df_incidents.loc[df_incidents['response_time'] < 0, 'response_time'] += 24 * 60  # Fix day boundary
            
            # Display the prepared data
            st.write("First few rows of our prepared data:")
            st.write(df_incidents[['incident_id', 'response_time', 'spread_time', 'cause', 'damage_level']].head())
            
            # Linear regression to predict spread time based on response time
            X = df_incidents['response_time'].values.reshape(-1, 1)
            y = df_incidents['spread_time'].values
            
            # Train the model and get results
            model, y_pred, metrics = train_linear_regression(X, y)
            
            # Show the regression plot
            fig = px.scatter(
                x=df_incidents['response_time'], 
                y=df_incidents['spread_time'],
                labels={"x": "Response Time (minutes)", "y": "Fire Spread Time (minutes)"},
                title="Linear Regression: Response Time vs. Fire Spread Time"
            )
            
            # Add regression line
            x_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
            y_range = model.predict(x_range)
            
            fig.add_trace(
                go.Scatter(
                    x=x_range.flatten(), 
                    y=y_range, 
                    mode='lines', 
                    name=f'Regression Line (R² = {metrics["R2"]:.2f})',
                    line=dict(color='red', width=2)
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Display model information
            st.write("**Linear Regression Model Results:**")
            st.write(f"- Coefficient (slope): {model.coef_[0]:.4f}")
            st.write(f"- Intercept: {model.intercept_:.4f}")
            st.write(f"- R² Score: {metrics['R2']:.4f}")
            st.write(f"- Mean Absolute Error: {metrics['MAE']:.4f} minutes")
            st.write(f"- Root Mean Squared Error: {metrics['RMSE']:.4f} minutes")
            st.write(f"- Equation: Fire Spread Time = {model.coef_[0]:.4f} × Response Time + {model.intercept_:.4f}")
            
            # Interpretation
            st.write("""
            **Interpretation:**
            
            The model shows a positive relationship between response time and fire spread time. For every additional 
            minute in response time, the fire spread time increases by approximately the coefficient value in minutes.
            
            This demonstrates how slower emergency response correlates with longer fire incidents, emphasizing the 
            importance of quick response in limiting fire damage.
            """)
        
        # Logistic Regression
        st.write("""
        #### Logistic Regression
        
        Logistic regression models the probability of a binary outcome based on one or more predictors.
        
        **Key Concepts:**
        - Predicts categorical outcomes (usually binary)
        - Outputs probabilities between 0 and 1
        - Can be extended to multi-class classification
        - Provides interpretable coefficients
        
        **Fire Safety Applications:**
        - Predicting whether a fire alarm is true or false
        - Classifying buildings as high or low fire risk
        - Determining if evacuation will be successful
        - Predicting if a fire will reach critical status
        """)
        
        # Practical example with fire data
        st.write("#### Practical Example: Predicting Severe Fire Damage")
        
        if not df_incidents.empty:
            # Create a binary target variable: Is damage severe?
            df_incidents['is_severe'] = (df_incidents['damage_level'] == 'Severe').astype(int)
            
            # Features for prediction: response time and cause (encoded as dummy variables)
            X = pd.get_dummies(df_incidents[['response_time', 'cause']], drop_first=True)
            y = df_incidents['is_severe']
            
            # Train the model
            model, y_pred, metrics = train_logistic_regression(X.values, y)
            
            # Display confusion matrix
            st.write("**Confusion Matrix:**")
            
            cm = metrics['confusion_matrix']
            fig_cm, ax = plt.subplots(figsize=(5, 4))
            ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Oranges)
            ax.set_title('Confusion Matrix')
            
            # Add labels and values
            tick_marks = np.arange(2)
            ax.set_xticks(tick_marks)
            ax.set_yticks(tick_marks)
            ax.set_xticklabels(['Non-Severe', 'Severe'])
            ax.set_yticklabels(['Non-Severe', 'Severe'])
            
            # Add text annotations
            thresh = cm.max() / 2.
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, format(cm[i, j], 'd'),
                            ha="center", va="center",
                            color="white" if cm[i, j] > thresh else "black")
            
            ax.set_xlabel('Predicted label')
            ax.set_ylabel('True label')
            plt.tight_layout()
            
            st.pyplot(fig_cm)
            
            # Display model performance metrics
            st.write("**Logistic Regression Model Performance:**")
            st.write(f"- Accuracy: {metrics['accuracy']:.4f}")
            st.write(f"- Precision (Severe): {metrics['classification_report']['1']['precision']:.4f}")
            st.write(f"- Recall (Severe): {metrics['classification_report']['1']['recall']:.4f}")
            st.write(f"- F1-Score (Severe): {metrics['classification_report']['1']['f1-score']:.4f}")
            
            # Interpretation
            st.write("""
            **Interpretation:**
            
            This logistic regression model predicts whether a fire will cause severe damage based on the response time 
            and the cause of the fire. The confusion matrix shows how many predictions were correct (diagonal) versus 
            incorrect (off-diagonal).
            
            This type of model can help fire departments prioritize resources and response strategies based on the 
            predicted severity of a fire incident.
            """)
            
            # Interactive element
            st.write("#### Try it yourself!")
            
            response_time = st.slider("Response Time (minutes)", 
                                     min_value=1, 
                                     max_value=60, 
                                     value=15,
                                     help="Time between fire detection and firefighter arrival")
            
            cause = st.selectbox("Fire Cause", 
                                df_incidents['cause'].unique(),
                                help="The cause of the fire")
            
            # Create features array for prediction
            causes = df_incidents['cause'].unique()
            cause_cols = [f"cause_{c}" for c in causes[1:]]  # drop_first=True
            
            # Create dummy feature array
            X_input = np.zeros(1 + len(cause_cols))
            X_input[0] = response_time
            
            # Set the cause dummy variable
            if cause != causes[0]:  # If not the first cause
                cause_idx = np.where(causes == cause)[0][0]
                if cause_idx > 0:  # Ensure it's not the reference category
                    X_input[cause_idx] = 1
            
            # Make prediction
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X.values)
            X_input_scaled = scaler.transform(X_input.reshape(1, -1))
            
            model = LogisticRegression(max_iter=1000, random_state=42)
            model.fit(X_scaled, y)
            
            prob = model.predict_proba(X_input_scaled)[0, 1]
            pred_class = "Severe" if prob > 0.5 else "Non-Severe"
            
            # Display prediction
            st.write(f"**Prediction:** The model predicts **{pred_class}** damage with {prob:.1%} probability")
            
            # Visualization of the prediction
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = prob * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Probability of Severe Damage"},
                gauge = {
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "red"},
                    'steps': [
                        {'range': [0, 25], 'color': "lightgreen"},
                        {'range': [25, 50], 'color': "yellow"},
                        {'range': [50, 75], 'color': "orange"},
                        {'range': [75, 100], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 4},
                        'thickness': 0.75,
                        'value': 50
                    }
                }
            ))
            
            st.plotly_chart(fig_gauge, use_container_width=True)
    
    # Decision Trees and Random Forests
    with tabs[1]:
        st.write("### Decision Trees and Random Forests")
        
        st.write("""
        #### Decision Trees
        
        Decision trees create a model that predicts the value of a target variable by learning simple decision rules 
        inferred from the data features.
        
        **Key Concepts:**
        - Tree-like model of decisions
        - Simple to understand and interpret
        - Can handle both numerical and categorical data
        - Prone to overfitting with complex trees
        
        **Fire Safety Applications:**
        - Classification of fire risks in buildings
        - Predicting fire causes based on indicators
        - Determining optimal evacuation routes
        - Decision support for fire inspections
        """)
        
        # Visualization of a decision tree
        st.write("#### Visualization of a Decision Tree for Fire Risk Classification")
        plot_decision_tree_viz()
        
        st.write("""
        The above visualization shows a simplified decision tree for classifying buildings by fire risk level.
        The tree makes decisions based on factors like building age, presence of sprinklers, and number of floors.
        
        Note how the tree creates a hierarchical structure of if-then rules that are easy to interpret.
        """)
        
        st.write("""
        #### Random Forests
        
        Random forests combine multiple decision trees to improve prediction accuracy and control overfitting.
        
        **Key Concepts:**
        - Ensemble of decision trees
        - Each tree votes on the final prediction
        - More robust than individual decision trees
        - Can rank feature importance
        
        **Fire Safety Applications:**
        - Building fire risk classification with high accuracy
        - Identifying key factors influencing fire outcomes
        - Predicting multiple fire safety metrics simultaneously
        - Complex pattern recognition in fire incident data
        """)
        
        # Practical example with fire data
        st.write("#### Practical Example: Classifying Building Fire Risk")
        
        # Load building data
        df_buildings = load_dataset("building_characteristics.csv")
        
        if not df_buildings.empty:
            # Create a target variable: Building risk level (based on building characteristics)
            # This is a simplified risk calculation for demonstration
            df_buildings['risk_score'] = (
                (df_buildings['age_years'] / 10) +  # Older buildings have higher risk
                (df_buildings['floors'] / 2) -  # More floors increase risk
                (df_buildings['has_sprinkler_system'] * 3) -  # Sprinklers reduce risk
                (df_buildings['has_fire_alarm'] * 2)  # Alarms reduce risk
            )
            
            # Categorize risk
            risk_bins = [-np.inf, 0, 5, np.inf]
            risk_labels = ['Low', 'Medium', 'High']
            df_buildings['risk_level'] = pd.cut(df_buildings['risk_score'], bins=risk_bins, labels=risk_labels)
            
            # Display the prepared data
            st.write("First few rows of our building data with risk levels:")
            st.write(df_buildings[['building_id', 'construction_type', 'age_years', 'floors', 
                                  'has_sprinkler_system', 'has_fire_alarm', 'risk_level']].head())
            
            # Features for prediction
            X = pd.get_dummies(df_buildings[['construction_type', 'age_years', 'floors', 
                                           'has_sprinkler_system', 'has_fire_alarm', 'occupancy_type']], 
                              drop_first=True)
            y = df_buildings['risk_level']
            
            # Train the model
            model, y_pred, metrics = train_random_forest(X.values, y)
            
            # Display model performance
            st.write("**Random Forest Model Performance:**")
            st.write(f"- Overall Accuracy: {metrics['accuracy']:.4f}")
            
            # Classification report by class
            classification_df = pd.DataFrame(metrics['classification_report'])
            if 'support' in classification_df.columns:
                classification_df = classification_df.drop('support', axis=1)
            classification_df = classification_df.drop('accuracy', axis=1)
            
            st.write("**Performance Metrics by Risk Level:**")
            st.write(classification_df)
            
            # Feature importance
            feature_importance = pd.DataFrame({
                'Feature': X.columns,
                'Importance': metrics['feature_importance']
            }).sort_values('Importance', ascending=False)
            
            st.write("**Feature Importance:**")
            
            fig_importance = px.bar(
                feature_importance.head(10),
                x='Importance',
                y='Feature',
                orientation='h',
                title='Top 10 Features for Fire Risk Classification',
                color='Importance',
                color_continuous_scale=px.colors.sequential.Oranges
            )
            
            st.plotly_chart(fig_importance, use_container_width=True)
            
            # Interpretation
            st.write("""
            **Interpretation:**
            
            The random forest model classifies buildings into risk levels based on their characteristics. The feature 
            importance graph shows which building attributes most strongly influence the risk classification.
            
            Understanding these key factors can help focus fire safety improvements on the most impactful areas,
            such as installing sprinkler systems or fire alarms in older buildings.
            """)
            
            # Interactive element - prediction
            st.write("#### Try the model yourself!")
            
            # Input form for building characteristics
            col1, col2 = st.columns(2)
            with col1:
                construction = st.selectbox("Construction Type", 
                                          df_buildings['construction_type'].unique())
                age = st.slider("Building Age (years)", 
                               min_value=1, 
                               max_value=50, 
                               value=20)
                floors = st.slider("Number of Floors", 
                                  min_value=1, 
                                  max_value=10, 
                                  value=3)
            
            with col2:
                sprinkler = st.checkbox("Has Sprinkler System", value=True)
                alarm = st.checkbox("Has Fire Alarm System", value=True)
                occupancy = st.selectbox("Occupancy Type", 
                                        df_buildings['occupancy_type'].unique())
            
            # Create input for prediction
            input_data = pd.DataFrame({
                'construction_type': [construction],
                'age_years': [age],
                'floors': [floors],
                'has_sprinkler_system': [sprinkler],
                'has_fire_alarm': [alarm],
                'occupancy_type': [occupancy]
            })
            
            # Transform input to match training data format
            input_dummies = pd.get_dummies(input_data, drop_first=True)
            
            # Ensure all columns from training exist in input (add missing with 0s)
            for col in X.columns:
                if col not in input_dummies.columns:
                    input_dummies[col] = 0
            
            # Ensure columns are in the same order
            input_dummies = input_dummies[X.columns]
            
            # Make prediction
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X.values, y)
            
            prediction = model.predict(input_dummies.values)[0]
            prediction_probs = model.predict_proba(input_dummies.values)[0]
            
            # Display prediction results
            st.write(f"**Prediction:** The building has a **{prediction} Risk** level")
            
            # Visualize probabilities
            risk_levels = model.classes_
            prob_df = pd.DataFrame({
                'Risk Level': risk_levels,
                'Probability': prediction_probs
            })
            
            fig_probs = px.bar(
                prob_df,
                x='Risk Level',
                y='Probability',
                title='Probability of Each Risk Level',
                color='Probability',
                color_continuous_scale=px.colors.sequential.Oranges
            )
            
            st.plotly_chart(fig_probs, use_container_width=True)
    
    # Support Vector Machines
    with tabs[2]:
        st.write("### Support Vector Machines (SVM)")
        
        st.write("""
        Support Vector Machines are powerful supervised learning models used for classification and regression tasks.
        
        **Key Concepts:**
        - Creates a hyperplane to separate data into classes
        - Maximizes the margin between classes
        - Can use different kernel functions for non-linear data
        - Effective in high-dimensional spaces
        
        **Fire Safety Applications:**
        - Detecting anomalies in sensor readings
        - Classifying fire causes from multiple indicators
        - Identifying unusual fire development patterns
        - Optimizing resource allocation based on risk patterns
        """)
        
        # Visual explanation of SVM
        st.write("#### How SVM Works")
        
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/7/72/SVM_margin.png/600px-SVM_margin.png",
                 caption="Support Vector Machine - Maximizing the margin between classes", 
                 width=400)
        
        # Practical example
        st.write("#### Practical Example: Detecting Anomalies in Smoke Detector Readings")
        
        # Load sensor data
        df_sensors = load_dataset("sensor_readings.csv")
        
        if not df_sensors.empty:
            # Create a binary target: alarm triggered (1) or not (0)
            y = df_sensors['alarm_triggered'].astype(int)
            
            # Features: temperature and smoke density
            X = df_sensors[['temperature_celsius', 'smoke_density']].values
            
            # Train SVM model
            model, y_pred, metrics = train_svm(X, y)
            
            # Plot the results
            st.write("**SVM Classification of Sensor Readings:**")
            
            # Create a scatter plot of the data
            fig = px.scatter(
                df_sensors, 
                x='temperature_celsius', 
                y='smoke_density',
                color='alarm_triggered',
                title='SVM Classification of Fire Alarm Triggers',
                labels={
                    'temperature_celsius': 'Temperature (°C)',
                    'smoke_density': 'Smoke Density',
                    'alarm_triggered': 'Alarm Triggered'
                },
                color_discrete_map={
                    True: '#FF4B4B',
                    False: '#1E88E5'
                }
            )
            
            # Add decision boundary (simplified visualization)
            st.plotly_chart(fig, use_container_width=True)
            
            # Display model performance
            st.write("**SVM Model Performance:**")
            st.write(f"- Accuracy: {metrics['accuracy']:.4f}")
            st.write(f"- Precision (Alarm): {metrics['classification_report']['1']['precision']:.4f}")
            st.write(f"- Recall (Alarm): {metrics['classification_report']['1']['recall']:.4f}")
            st.write(f"- F1-Score (Alarm): {metrics['classification_report']['1']['f1-score']:.4f}")
            
            # Interpretation
            st.write("""
            **Interpretation:**
            
            The SVM model classifies sensor readings to determine whether they should trigger an alarm. The decision 
            boundary separates normal conditions (blue) from alarm conditions (red).
            
            This type of model is useful for creating more intelligent fire detection systems that can distinguish 
            between normal variations and actual fire indicators, potentially reducing false alarms while ensuring
            real fires are detected quickly.
            """)
            
            # Interactive prediction
            st.write("#### Try the model yourself!")
            
            col1, col2 = st.columns(2)
            with col1:
                temp = st.slider("Temperature (°C)", 
                                min_value=float(df_sensors['temperature_celsius'].min()),
                                max_value=float(df_sensors['temperature_celsius'].max()),
                                value=35.0,
                                step=0.5)
            
            with col2:
                smoke = st.slider("Smoke Density", 
                                 min_value=float(df_sensors['smoke_density'].min()),
                                 max_value=float(df_sensors['smoke_density'].max()),
                                 value=1.0,
                                 step=0.1)
            
            # Scale input (similar to how the model was trained)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            X_input = np.array([[temp, smoke]])
            X_input_scaled = scaler.transform(X_input)
            
            # Make prediction
            from sklearn.svm import SVC
            model = SVC(kernel='rbf', probability=True, random_state=42)
            model.fit(X_scaled, y)
            
            prediction = model.predict(X_input_scaled)[0]
            probability = model.predict_proba(X_input_scaled)[0][1]  # Probability of class 1
            
            # Display prediction
            alarm_status = "TRIGGERED" if prediction == 1 else "NOT TRIGGERED"
            
            if prediction == 1:
                st.error(f"Alarm Status: **{alarm_status}** (Probability: {probability:.1%})")
            else:
                st.success(f"Alarm Status: **{alarm_status}** (Probability: {1-probability:.1%})")
    
    # Naive Bayes
    with tabs[3]:
        st.write("### Naive Bayes Methods")
        
        st.write("""
        Naive Bayes methods are a family of simple probabilistic classifiers based on applying Bayes' theorem with 
        strong independence assumptions between the features.
        
        **Key Concepts:**
        - Based on Bayes' theorem of conditional probability
        - "Naive" because it assumes feature independence
        - Simple and fast for large datasets
        - Works well with high-dimensional data
        
        **Fire Safety Applications:**
        - Probabilistic assessment of fire causes
        - Text classification of fire incident reports
        - Spam filtering for fire alert systems
        - Quick preliminary assessment of fire risk factors
        """)
        
        # Bayes theorem explanation
        with st.expander("Bayes' Theorem Explained"):
            st.write("""
            **Bayes' Theorem:**
            
            P(A|B) = [P(B|A) × P(A)] / P(B)
            
            Where:
            - P(A|B) is the probability of hypothesis A given evidence B
            - P(B|A) is the probability of evidence B given hypothesis A
            - P(A) is the prior probability of hypothesis A
            - P(B) is the prior probability of evidence B
            
            **In Fire Safety Context:**
            
            For example, when determining the probability of a specific fire cause given certain evidence:
            
            P(Cause|Evidence) = [P(Evidence|Cause) × P(Cause)] / P(Evidence)
            
            This allows us to update our beliefs about fire causes based on new evidence.
            """)
        
        # Practical example
        st.write("#### Practical Example: Probabilistic Assessment of Fire Causes")
        
        # Load fire incidents data
        df_incidents = load_dataset("fire_incidents.csv")
        
        if not df_incidents.empty:
            # Create features from damage level and add a simplified time of day
            df_incidents['time_of_day'] = df_incidents['detection_time'].apply(
                lambda x: 'Morning' if pd.notna(x) and isinstance(x, str) and int(x.split(':')[0]) in range(5, 12) else
                         'Afternoon' if pd.notna(x) and isinstance(x, str) and int(x.split(':')[0]) in range(12, 18) else
                         'Evening' if pd.notna(x) and isinstance(x, str) and int(x.split(':')[0]) in range(18, 23) else
                         'Night'
            )
            
            # Features for prediction
            X = pd.get_dummies(df_incidents[['damage_level', 'time_of_day']], drop_first=True)
            y = df_incidents['cause']
            
            # Train Naive Bayes model
            model, y_pred, metrics = train_naive_bayes(X.values, y)
            
            # Display model performance
            st.write("**Naive Bayes Model Performance:**")
            st.write(f"- Overall Accuracy: {metrics['accuracy']:.4f}")
            
            # Create a heatmap of the confusion matrix
            cm = metrics['confusion_matrix']
            classes = np.unique(y)
            
            fig_cm, ax = plt.subplots(figsize=(10, 8))
            im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Oranges)
            ax.set_title('Confusion Matrix - Fire Cause Prediction')
            
            # Add labels and values
            tick_marks = np.arange(len(classes))
            ax.set_xticks(tick_marks)
            ax.set_yticks(tick_marks)
            ax.set_xticklabels(classes, rotation=45, ha='right')
            ax.set_yticklabels(classes)
            
            # Add text annotations
            thresh = cm.max() / 2.
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, format(cm[i, j], 'd'),
                            ha="center", va="center",
                            color="white" if cm[i, j] > thresh else "black")
            
            ax.set_xlabel('Predicted cause')
            ax.set_ylabel('True cause')
            plt.tight_layout()
            
            st.pyplot(fig_cm)
            
            # Interpretation
            st.write("""
            **Interpretation:**
            
            The Naive Bayes model predicts fire causes based on damage level and time of day. The confusion matrix 
            shows how well the model correctly identifies each cause (diagonal elements) versus misclassifications 
            (off-diagonal elements).
            
            This probabilistic approach is particularly useful when multiple factors might contribute to a fire, 
            and we want to estimate the most likely cause based on limited evidence.
            """)
            
            # Interactive prediction
            st.write("#### Try the model yourself!")
            
            col1, col2 = st.columns(2)
            with col1:
                damage = st.selectbox("Damage Level", df_incidents['damage_level'].unique())
            
            with col2:
                time = st.selectbox("Time of Day", ['Morning', 'Afternoon', 'Evening', 'Night'])
            
            # Create input features (matching training data format)
            input_data = pd.DataFrame({
                'damage_level': [damage],
                'time_of_day': [time]
            })
            
            input_dummies = pd.get_dummies(input_data, drop_first=True)
            
            # Add missing columns with 0s
            for col in X.columns:
                if col not in input_dummies.columns:
                    input_dummies[col] = 0
            
            # Ensure columns are in the same order
            input_dummies = input_dummies[X.columns]
            
            # Make prediction
            from sklearn.naive_bayes import GaussianNB
            model = GaussianNB()
            model.fit(X.values, y)
            
            prediction = model.predict(input_dummies.values)[0]
            probabilities = model.predict_proba(input_dummies.values)[0]
            
            # Display prediction and probabilities
            st.write(f"**Most Likely Cause: {prediction}**")
            
            # Create probability chart
            prob_df = pd.DataFrame({
                'Cause': model.classes_,
                'Probability': probabilities
            }).sort_values('Probability', ascending=False)
            
            fig_probs = px.bar(
                prob_df,
                x='Cause',
                y='Probability',
                title='Probability of Each Fire Cause',
                color='Probability',
                color_continuous_scale=px.colors.sequential.Oranges
            )
            
            st.plotly_chart(fig_probs, use_container_width=True)
    
    # Python code examples
    with st.expander("Python Code Examples"):
        st.write("#### Linear Regression Example")
        
        linear_reg_code = """
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load fire data (sample code)
df = pd.read_csv('fire_incidents.csv')

# Prepare features and target
X = df['response_time'].values.reshape(-1, 1)  # Feature
y = df['spread_time'].values  # Target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Coefficient: {model.coef_[0]:.4f}")
print(f"Intercept: {model.intercept_:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²: {r2:.4f}")

# Plot results
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color='blue', label='Actual data')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Regression line')
plt.xlabel('Response Time (minutes)')
plt.ylabel('Fire Spread Time (minutes)')
plt.title('Linear Regression: Response Time vs. Fire Spread Time')
plt.legend()
plt.grid(True)
plt.show()
"""
        
        st.code(linear_reg_code, language="python")
        
        st.write("#### Random Forest Example")
        
        rf_code = """
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load building data (sample code)
df = pd.read_csv('building_characteristics.csv')

# Create target variable (simplified example)
df['risk_score'] = (
    (df['age_years'] / 10) +
    (df['floors'] / 2) -
    (df['has_sprinkler_system'] * 3) -
    (df['has_fire_alarm'] * 2)
)

# Categorize risk
risk_bins = [-np.inf, 0, 5, np.inf]
risk_labels = ['Low', 'Medium', 'High']
df['risk_level'] = pd.cut(df['risk_score'], bins=risk_bins, labels=risk_labels)

# Prepare features and target
X = pd.get_dummies(df[['construction_type', 'age_years', 'floors', 
                      'has_sprinkler_system', 'has_fire_alarm', 'occupancy_type']], 
                  drop_first=True)
y = df['risk_level']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print("\\nClassification Report:")
print(report)

# Plot feature importance
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance.head(10))
plt.title('Top 10 Features for Fire Risk Classification')
plt.tight_layout()
plt.show()
"""
        
        st.code(rf_code, language="python")
    
    # Module quiz
    st.write("### Module 3 Quiz")
    
    questions = [
        {
            'question': 'Which supervised learning algorithm is most appropriate for predicting the exact temperature a fire will reach?',
            'options': [
                'Linear Regression',
                'Logistic Regression',
                'Decision Trees',
                'Naive Bayes'
            ],
            'correct': 'Linear Regression'
        },
        {
            'question': 'When would you use logistic regression instead of linear regression in fire safety applications?',
            'options': [
                'When predicting the exact temperature of a fire',
                'When estimating the time it will take firefighters to arrive',
                'When classifying whether a building is at high risk for fire or not',
                'When measuring the correlation between building age and fire frequency'
            ],
            'correct': 'When classifying whether a building is at high risk for fire or not'
        },
        {
            'question': 'What is a key advantage of random forests over single decision trees?',
            'options': [
                'Random forests are faster to train',
                'Random forests are more interpretable',
                'Random forests are more robust to overfitting',
                'Random forests require less data'
            ],
            'correct': 'Random forests are more robust to overfitting'
        },
        {
            'question': 'In a confusion matrix for a fire alarm prediction model, what does a false positive represent?',
            'options': [
                'The model correctly predicted a fire event',
                'The model correctly predicted there was no fire',
                'The model predicted a fire when there was none (false alarm)',
                'The model failed to predict a fire that occurred'
            ],
            'correct': 'The model predicted a fire when there was none (false alarm)'
        },
        {
            'question': 'Which metric would be most important to optimize in a model predicting whether a building will experience a catastrophic fire?',
            'options': [
                'Accuracy',
                'Precision',
                'Recall',
                'F1-Score'
            ],
            'correct': 'Recall'
        },
        {
            'question': 'What is the "naive" assumption in Naive Bayes classifiers?',
            'options': [
                'The target variable follows a normal distribution',
                'The features are independent of each other given the class',
                'The algorithm is simple and therefore less effective',
                'There is a linear relationship between features and the target'
            ],
            'correct': 'The features are independent of each other given the class'
        },
        {
            'question': 'Support Vector Machines (SVMs) work by:',
            'options': [
                'Creating a hierarchical tree of decisions',
                'Finding the hyperplane that maximizes the margin between classes',
                'Calculating conditional probabilities based on Bayes\' theorem',
                'Combining multiple weak learners into a strong predictor'
            ],
            'correct': 'Finding the hyperplane that maximizes the margin between classes'
        },
        {
            'question': 'Which of the following would NOT typically be a feature in a model predicting fire spread time?',
            'options': [
                'Building material type',
                'Distance to nearest fire station',
                'Humidity level',
                'The firefighter\'s name'
            ],
            'correct': 'The firefighter\'s name'
        },
        {
            'question': 'When evaluating a linear regression model for predicting fire damage costs, which metric indicates the proportion of variance explained by the model?',
            'options': [
                'Mean Absolute Error (MAE)',
                'Root Mean Squared Error (RMSE)',
                'R-squared (R²)',
                'Log Loss'
            ],
            'correct': 'R-squared (R²)'
        },
        {
            'question': 'Which supervised learning approach would be most appropriate for identifying the most important factors contributing to rapid fire spread?',
            'options': [
                'Logistic Regression with L1 regularization',
                'Random Forest with feature importance analysis',
                'Support Vector Machines with a linear kernel',
                'Naive Bayes classifier'
            ],
            'correct': 'Random Forest with feature importance analysis'
        }
    ]
    
    create_module_quiz(questions, 2)

