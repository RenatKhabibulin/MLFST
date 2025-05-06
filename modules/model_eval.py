import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, KFold
from utils import load_dataset, create_module_quiz, plot_confusion_matrix

def show_module_content():
    """Display model evaluation content"""
    
    st.write("## Model Evaluation for Fire Safety Machine Learning")
    
    # Overview
    st.write("""
    ### Overview
    
    Evaluating machine learning models is crucial to ensure they perform reliably in fire safety 
    applications. Proper evaluation helps understand a model's strengths and limitations, particularly 
    important when these models may influence life-safety decisions. This module covers key evaluation 
    metrics and techniques for different types of fire safety models.
    """)
    
    # Tab-based navigation for different evaluation approaches
    tabs = st.tabs([
        "Classification Metrics", 
        "Regression Metrics",
        "Cross-Validation",
        "Model Selection"
    ])
    
    # Classification Metrics tab
    with tabs[0]:
        st.write("### Metrics for Classification Models")
        
        st.write("""
        Classification models in fire safety are used for tasks such as predicting fire risk categories, 
        determining alarm validity, and classifying fire causes. Evaluating these models requires specific metrics.
        """)
        
        # Basic classification metrics
        st.write("#### Basic Classification Metrics")
        
        # Create a table for classification metrics
        metrics_df = pd.DataFrame({
            'Metric': [
                'Accuracy', 
                'Precision', 
                'Recall (Sensitivity)', 
                'Specificity',
                'F1-Score',
                'Area Under ROC Curve (AUC)'
            ],
            'Description': [
                'Proportion of all predictions that are correct',
                'Proportion of positive predictions that are actually positive',
                'Proportion of actual positives that are correctly identified',
                'Proportion of actual negatives that are correctly identified',
                'Harmonic mean of precision and recall',
                'Measure of model\'s ability to discriminate between classes'
            ],
            'Formula': [
                '(TP + TN) / (TP + TN + FP + FN)',
                'TP / (TP + FP)',
                'TP / (TP + FN)',
                'TN / (TN + FP)',
                '2 × (Precision × Recall) / (Precision + Recall)',
                'Area under the ROC curve'
            ],
            'Fire Safety Context': [
                'General measure, but can be misleading for imbalanced data',
                'When false alarms are costly (e.g., evacuation)',
                'When missing a fire event is dangerous',
                'Ability to correctly identify safe conditions',
                'Balance between precision and recall',
                'Overall discrimination, independent of threshold'
            ]
        })
        
        st.write(metrics_df)
        
        st.write("""
        Where:
        - **TP (True Positive)**: Model correctly predicts a fire/risk
        - **TN (True Negative)**: Model correctly predicts no fire/risk
        - **FP (False Positive)**: Model incorrectly predicts a fire/risk (false alarm)
        - **FN (False Negative)**: Model incorrectly predicts no fire/risk (missed detection)
        """)
        
        # Practical example: Confusion Matrix
        st.write("#### Practical Example: Evaluating a Fire Alarm Classification Model")
        
        # Load sensor data
        df_sensors = load_dataset("sensor_readings.csv")
        
        if not df_sensors.empty:
            # Create a binary classification scenario
            # Let's say we have a model that predicts whether an alarm should trigger
            # based on temperature and smoke density
            
            # First, create a synthetic ground truth and prediction
            # For demonstration purposes - in real scenario these would come from actual model predictions
            np.random.seed(42)  # For reproducibility
            
            # Create a copy of the data
            df_eval = df_sensors.copy()
            
            # Use the real 'alarm_triggered' column as ground truth
            ground_truth = df_eval['alarm_triggered'].astype(int).values
            
            # Create a synthetic model prediction (with deliberate errors for demonstration)
            # In reality, this would be your model's prediction
            prediction_probs = df_eval.apply(
                lambda x: (0.7 * (x['temperature_celsius'] / 100) + 
                          0.3 * (x['smoke_density'] / 3)) + 
                          np.random.normal(0, 0.1),  # Add some noise
                axis=1
            ).values
            
            # Convert probabilities to binary predictions (threshold = 0.5)
            predictions = (prediction_probs > 0.5).astype(int)
            
            # Compute confusion matrix
            cm = confusion_matrix(ground_truth, predictions)
            
            # Display confusion matrix
            st.write("##### Confusion Matrix")
            
            # Plot the confusion matrix
            fig_cm = plot_confusion_matrix(cm, ['No Alarm', 'Alarm'])
            st.pyplot(fig_cm)
            
            # Calculate and display metrics
            tn, fp, fn, tp = cm.ravel()
            
            total = tn + fp + fn + tp
            accuracy = (tp + tn) / total
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            st.write("##### Classification Metrics")
            
            # Create metrics columns
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Accuracy", f"{accuracy:.2f}")
                st.metric("Precision", f"{precision:.2f}")
            
            with col2:
                st.metric("Recall", f"{recall:.2f}")
                st.metric("Specificity", f"{specificity:.2f}")
            
            with col3:
                st.metric("F1 Score", f"{f1:.2f}")
                st.metric("False Alarm Rate", f"{fp/(fp+tn):.2f}")
            
            # ROC Curve
            st.write("##### ROC Curve")
            
            # Calculate ROC curve
            fpr, tpr, thresholds = roc_curve(ground_truth, prediction_probs)
            roc_auc = auc(fpr, tpr)
            
            # Plot ROC curve
            fig_roc = go.Figure()
            
            # Add ROC curve
            fig_roc.add_trace(go.Scatter(
                x=fpr, 
                y=tpr,
                mode='lines',
                name=f'ROC Curve (AUC = {roc_auc:.3f})',
                line=dict(color='#FF4B4B', width=2)
            ))
            
            # Add diagonal line (random classifier)
            fig_roc.add_trace(go.Scatter(
                x=[0, 1], 
                y=[0, 1],
                mode='lines',
                name='Random Classifier',
                line=dict(color='#262730', width=2, dash='dash')
            ))
            
            # Update layout
            fig_roc.update_layout(
                title='Receiver Operating Characteristic (ROC) Curve',
                xaxis_title='False Positive Rate',
                yaxis_title='True Positive Rate',
                legend=dict(
                    x=0.01,
                    y=0.99,
                    bgcolor='rgba(255, 255, 255, 0.5)'
                ),
                width=700,
                height=500
            )
            
            st.plotly_chart(fig_roc)
            
            # Precision-Recall Curve
            st.write("##### Precision-Recall Curve")
            
            # Calculate precision-recall curve
            precision_values, recall_values, _ = precision_recall_curve(ground_truth, prediction_probs)
            
            # Plot precision-recall curve
            fig_pr = go.Figure()
            
            # Add precision-recall curve
            fig_pr.add_trace(go.Scatter(
                x=recall_values, 
                y=precision_values,
                mode='lines',
                name='Precision-Recall Curve',
                line=dict(color='#FF7B4B', width=2)
            ))
            
            # Update layout
            fig_pr.update_layout(
                title='Precision-Recall Curve',
                xaxis_title='Recall',
                yaxis_title='Precision',
                legend=dict(
                    x=0.01,
                    y=0.99,
                    bgcolor='rgba(255, 255, 255, 0.5)'
                ),
                width=700,
                height=500
            )
            
            st.plotly_chart(fig_pr)
            
            # Threshold analysis
            st.write("##### Threshold Analysis")
            
            st.write("""
            In fire safety applications, choosing the right threshold for binary classification 
            is critical. A lower threshold increases sensitivity (catching more actual fires) 
            but may cause more false alarms. A higher threshold reduces false alarms but might 
            miss some actual fires.
            """)
            
            # Create data for threshold analysis
            threshold_data = []
            
            thresholds_to_analyze = np.linspace(0.0, 1.0, 100)
            for threshold in thresholds_to_analyze:
                binary_pred = (prediction_probs >= threshold).astype(int)
                tn, fp, fn, tp = confusion_matrix(ground_truth, binary_pred).ravel()
                
                precision_val = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall_val = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1_val = 2 * (precision_val * recall_val) / (precision_val + recall_val) if (precision_val + recall_val) > 0 else 0
                
                threshold_data.append({
                    'Threshold': threshold,
                    'Precision': precision_val,
                    'Recall': recall_val,
                    'F1 Score': f1_val,
                    'False Alarm Rate': fp / (fp + tn) if (fp + tn) > 0 else 0
                })
            
            threshold_df = pd.DataFrame(threshold_data)
            
            # Plot threshold analysis
            fig_threshold = px.line(
                threshold_df, 
                x='Threshold', 
                y=['Precision', 'Recall', 'F1 Score', 'False Alarm Rate'],
                title='Metric Values Across Different Thresholds',
                labels={'value': 'Metric Value', 'variable': 'Metric'}
            )
            
            # Add vertical line at typical threshold of 0.5
            fig_threshold.add_vline(
                x=0.5,
                line_dash="dash",
                line_color="black",
                annotation_text="Typical Threshold (0.5)",
                annotation_position="top right"
            )
            
            st.plotly_chart(fig_threshold, use_container_width=True)
            
            # Interpretation
            st.write("""
            ##### Interpretation for Fire Safety
            
            When evaluating a classification model for fire safety:
            
            1. **False Negatives (Missed Alarms)** are often more critical than false positives:
               - A missed fire detection could lead to catastrophic consequences
               - May need to optimize for high recall/sensitivity even at the cost of more false alarms
            
            2. **Cost of False Positives (False Alarms)**:
               - Frequent false alarms can lead to "alarm fatigue" and people ignoring real alarms
               - Each evacuation has economic and operational costs
            
            3. **Threshold Selection**:
               - Different environments may require different thresholds
               - Critical infrastructure might use lower thresholds (higher sensitivity)
               - Areas with frequent non-fire triggers might use slightly higher thresholds with additional verification
            
            4. **Beyond Accuracy**:
               - In imbalanced scenarios (fires are rare events), accuracy can be misleading
               - Focus on precision, recall, and F1-score to better understand model performance
            """)
    
    # Regression Metrics tab
    with tabs[1]:
        st.write("### Metrics for Regression Models")
        
        st.write("""
        Regression models in fire safety predict continuous values like fire spread time, 
        temperature development, evacuation duration, or property damage estimates. These models 
        require different evaluation metrics than classification models.
        """)
        
        # Basic regression metrics
        st.write("#### Basic Regression Metrics")
        
        # Create a table for regression metrics
        metrics_df = pd.DataFrame({
            'Metric': [
                'Mean Absolute Error (MAE)', 
                'Mean Squared Error (MSE)',
                'Root Mean Squared Error (RMSE)',
                'R-squared (R²)',
                'Mean Absolute Percentage Error (MAPE)',
                'Explained Variance'
            ],
            'Description': [
                'Average absolute difference between predicted and actual values',
                'Average squared difference between predicted and actual values',
                'Square root of MSE, in same units as the target variable',
                'Proportion of variance in the target variable explained by the model',
                'Average percentage difference between predicted and actual values',
                'Proportion of variance in the target variable that is explained by the model'
            ],
            'Formula': [
                '(1/n) × Σ|y_i - ŷ_i|',
                '(1/n) × Σ(y_i - ŷ_i)²',
                '√((1/n) × Σ(y_i - ŷ_i)²)',
                '1 - (Σ(y_i - ŷ_i)² / Σ(y_i - ȳ)²)',
                '(100%/n) × Σ|(y_i - ŷ_i)/y_i|',
                '1 - Var(y - ŷ) / Var(y)'
            ],
            'Fire Safety Context': [
                'Error in fire spread time prediction (in minutes)',
                'Penalizes larger errors more heavily',
                'Error in temperature prediction (in °C)',
                'How well the model explains variations in fire behavior',
                'Percentage error in evacuation time prediction',
                'Similar to R², but more robust to bias in the model'
            ]
        })
        
        st.write(metrics_df)
        
        # Practical example for regression evaluation
        st.write("#### Practical Example: Evaluating a Fire Spread Time Prediction Model")
        
        # Load fire incidents data
        df_incidents = load_dataset("fire_incidents.csv")
        
        if not df_incidents.empty:
            # Prepare data for regression example
            
            # Add derived time features for analysis
            df_incidents['detection_time_mins'] = df_incidents['detection_time'].apply(
                lambda x: int(str(x).split(':')[0]) * 60 + int(str(x).split(':')[1]) if pd.notna(x) and isinstance(x, str) else 0
            )
            df_incidents['arrival_time_mins'] = df_incidents['arrival_time'].apply(
                lambda x: int(str(x).split(':')[0]) * 60 + int(str(x).split(':')[1]) if pd.notna(x) and isinstance(x, str) else 0
            )
            df_incidents['extinguishing_time_mins'] = df_incidents['extinguishing_time'].apply(
                lambda x: int(str(x).split(':')[0]) * 60 + int(str(x).split(':')[1]) if pd.notna(x) and isinstance(x, str) else 0
            )
            
            # Calculate response time (detection to arrival)
            df_incidents['response_time'] = df_incidents['arrival_time_mins'] - df_incidents['detection_time_mins']
            df_incidents.loc[df_incidents['response_time'] < 0, 'response_time'] += 24 * 60  # Fix day boundary
            
            # Calculate total time (detection to extinguished)
            df_incidents['total_fire_time'] = df_incidents['extinguishing_time_mins'] - df_incidents['detection_time_mins']
            df_incidents.loc[df_incidents['total_fire_time'] < 0, 'total_fire_time'] += 24 * 60  # Fix day boundary
            
            # Prepare data for our regression scenario
            # We'll predict total fire time based on response time
            X = df_incidents['response_time'].values
            y_true = df_incidents['total_fire_time'].values
            
            # Generate synthetic model predictions
            # In a real scenario, these would come from your model
            np.random.seed(42)  # For reproducibility
            
            # Create a linear relationship with some noise
            y_pred = 2.5 * X + 20 + np.random.normal(0, 15, size=len(X))
            
            # Calculate regression metrics
            mae = mean_absolute_error(y_true, y_pred)
            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_true, y_pred)
            
            # Calculate MAPE (handling zero values)
            with np.errstate(divide='ignore', invalid='ignore'):
                mape = np.mean(np.abs((y_true - y_pred) / np.maximum(1e-10, y_true))) * 100
            
            # Display metrics
            st.write("##### Regression Metrics")
            
            # Create metrics columns
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("MAE", f"{mae:.2f} min")
                st.metric("MSE", f"{mse:.2f} min²")
            
            with col2:
                st.metric("RMSE", f"{rmse:.2f} min")
                st.metric("R²", f"{r2:.3f}")
            
            with col3:
                st.metric("MAPE", f"{mape:.2f}%")
            
            # Visualize predictions vs actual values
            st.write("##### Predicted vs. Actual Values")
            
            # Create scatter plot
            fig = px.scatter(
                x=y_true, 
                y=y_pred,
                labels={'x': 'Actual Total Fire Time (min)', 'y': 'Predicted Total Fire Time (min)'},
                title='Predicted vs. Actual Fire Duration'
            )
            
            # Add diagonal line (perfect predictions)
            fig.add_trace(
                go.Scatter(
                    x=[min(y_true), max(y_true)],
                    y=[min(y_true), max(y_true)],
                    mode='lines',
                    name='Perfect Prediction',
                    line=dict(color='black', dash='dash')
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Residual plot
            st.write("##### Residual Analysis")
            
            # Calculate residuals
            residuals = y_true - y_pred
            
            # Create residual plot
            fig_res = px.scatter(
                x=y_pred, 
                y=residuals,
                labels={'x': 'Predicted Values', 'y': 'Residuals'},
                title='Residual Plot'
            )
            
            # Add horizontal line at y=0
            fig_res.add_hline(
                y=0,
                line_dash="dash",
                line_color="black"
            )
            
            st.plotly_chart(fig_res, use_container_width=True)
            
            # Distribution of residuals
            st.write("##### Distribution of Residuals")
            
            # Create histogram of residuals
            fig_hist = px.histogram(
                residuals,
                nbins=30,
                labels={'value': 'Residual', 'count': 'Frequency'},
                title='Distribution of Residuals'
            )
            
            st.plotly_chart(fig_hist, use_container_width=True)
            
            # Interpretation of regression metrics
            st.write("""
            ##### Interpretation for Fire Safety
            
            When evaluating a regression model for fire safety:
            
            1. **Practical Interpretation of Errors**:
               - MAE of {mae:.2f} minutes means, on average, our predictions are off by about {mae:.1f} minutes
               - RMSE of {rmse:.2f} minutes gives extra weight to larger errors
               - An R² of {r2:.3f} indicates the model explains about {r2_percent:.1f}% of the variability in fire duration
            
            2. **Error Distribution**:
               - Unbiased models should have residuals centered around zero
               - Systematic patterns in residuals may indicate model deficiencies
            
            3. **Fire Safety Implications**:
               - Underestimating fire duration could lead to premature reentry or insufficient resources
               - Overestimating could waste resources but provides a safety margin
               - For critical applications, consider the worst-case error, not just the average
            
            4. **Context-Specific Evaluation**:
               - RMSE of 15 minutes may be acceptable for structural fires but not for flashover prediction
               - Consider the operational context when evaluating prediction errors
            """.format(mae=mae, rmse=rmse, r2=r2, r2_percent=r2*100))
    
    # Cross-Validation tab
    with tabs[2]:
        st.write("### Cross-Validation Techniques")
        
        st.write("""
        Cross-validation assesses how machine learning models generalize to independent data, helping to 
        detect overfitting and providing more reliable performance estimates.
        """)
        
        # Explain different cross-validation methods
        st.write("#### Common Cross-Validation Methods")
        
        # Visualization of different CV methods
        st.components.v1.html("""
        <svg width="700" height="400" xmlns="http://www.w3.org/2000/svg">
            <!-- Title -->
            <text x="350" y="30" font-family="Arial" font-size="16" font-weight="bold" text-anchor="middle">Cross-Validation Methods</text>
            
            <!-- K-Fold Cross-Validation -->
            <text x="350" y="70" font-family="Arial" font-size="14" font-weight="bold" text-anchor="middle">K-Fold Cross-Validation (k=5)</text>
            
            <!-- Fold 1 -->
            <rect x="100" y="90" width="500" height="30" fill="#F0F2F6" stroke="#262730" />
            <rect x="100" y="90" width="100" height="30" fill="#FF4B4B" />
            <text x="350" y="110" font-family="Arial" font-size="12" text-anchor="middle">Fold 1: Test on first 20%, train on rest</text>
            
            <!-- Fold 2 -->
            <rect x="100" y="130" width="500" height="30" fill="#F0F2F6" stroke="#262730" />
            <rect x="200" y="130" width="100" height="30" fill="#FF4B4B" />
            <text x="350" y="150" font-family="Arial" font-size="12" text-anchor="middle">Fold 2: Test on second 20%, train on rest</text>
            
            <!-- Fold 3 -->
            <rect x="100" y="170" width="500" height="30" fill="#F0F2F6" stroke="#262730" />
            <rect x="300" y="170" width="100" height="30" fill="#FF4B4B" />
            <text x="350" y="190" font-family="Arial" font-size="12" text-anchor="middle">Fold 3: Test on third 20%, train on rest</text>
            
            <!-- Fold 4 -->
            <rect x="100" y="210" width="500" height="30" fill="#F0F2F6" stroke="#262730" />
            <rect x="400" y="210" width="100" height="30" fill="#FF4B4B" />
            <text x="350" y="230" font-family="Arial" font-size="12" text-anchor="middle">Fold 4: Test on fourth 20%, train on rest</text>
            
            <!-- Fold 5 -->
            <rect x="100" y="250" width="500" height="30" fill="#F0F2F6" stroke="#262730" />
            <rect x="500" y="250" width="100" height="30" fill="#FF4B4B" />
            <text x="350" y="270" font-family="Arial" font-size="12" text-anchor="middle">Fold 5: Test on last 20%, train on rest</text>
            
            <!-- Legend -->
            <rect x="250" y="300" width="20" height="15" fill="#F0F2F6" stroke="#262730" />
            <text x="280" y="312" font-family="Arial" font-size="12" text-anchor="start">Training Data</text>
            
            <rect x="400" y="300" width="20" height="15" fill="#FF4B4B" />
            <text x="430" y="312" font-family="Arial" font-size="12" text-anchor="start">Test Data</text>
            
            <!-- Additional Text -->
            <text x="350" y="340" font-family="Arial" font-size="12" text-anchor="middle" font-style="italic">Final performance is the average of all folds</text>
            <text x="350" y="360" font-family="Arial" font-size="12" text-anchor="middle" font-style="italic">Helps ensure model works well across the entire dataset</text>
        </svg>
        """, height=400)
        
        # Describe cross-validation methods
        cv_methods = pd.DataFrame({
            'Method': [
                'K-Fold Cross-Validation', 
                'Stratified K-Fold',
                'Leave-One-Out (LOOCV)',
                'Time Series Split'
            ],
            'Description': [
                'Splits data into k equal folds, using each fold once as test set',
                'K-fold that preserves class distribution in each fold',
                'Special case of k-fold where k equals the number of samples',
                'Splits data with respect to time order (for temporal data)'
            ],
            'When to Use in Fire Safety': [
                'General fire classification/regression with sufficient data',
                'Imbalanced datasets (rare fire events vs. normal conditions)',
                'Very small datasets of unusual fire scenarios',
                'Fire progression models, temporal sensor data analysis'
            ]
        })
        
        st.write(cv_methods)
        
        # Practical example of cross-validation
        st.write("#### Practical Example: Cross-Validation for a Fire Risk Model")
        
        # Load building data
        df_buildings = load_dataset("building_characteristics.csv")
        
        if not df_buildings.empty:
            # Create a target variable: Building risk level
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
            
            # Convert to numeric for simplicity
            risk_mapping = {'Low': 0, 'Medium': 1, 'High': 2}
            df_buildings['risk_numeric'] = df_buildings['risk_level'].map(risk_mapping)
            
            # Features for prediction
            features = ['age_years', 'floors', 'has_sprinkler_system', 'has_fire_alarm']
            X = df_buildings[features].values
            y = df_buildings['risk_numeric'].values
            
            # Display data sample
            st.write("First few rows of our building data:")
            st.write(df_buildings[features + ['risk_level']].head())
            
            # Perform K-fold cross-validation for different models
            from sklearn.linear_model import LogisticRegression
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.svm import SVC
            
            # Define models to compare
            models = {
                'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
                'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
                'Support Vector Machine': SVC(kernel='rbf', random_state=42)
            }
            
            # Number of folds
            k_folds = 5
            
            # Create KFold object
            kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
            
            # Perform cross-validation for each model
            cv_results = {}
            
            for name, model in models.items():
                # Get cross-validation scores
                cv_scores = cross_val_score(model, X, y, cv=kf, scoring='accuracy')
                cv_results[name] = {
                    'scores': cv_scores,
                    'mean': cv_scores.mean(),
                    'std': cv_scores.std()
                }
            
            # Display cross-validation results
            st.write("##### Cross-Validation Results (Accuracy)")
            
            # Create a DataFrame for display
            cv_df = pd.DataFrame({
                'Model': list(cv_results.keys()),
                'Mean Accuracy': [cv_results[model]['mean'] for model in cv_results],
                'Std Dev': [cv_results[model]['std'] for model in cv_results]
            })
            
            # Add fold-specific columns
            for i in range(k_folds):
                cv_df[f'Fold {i+1}'] = [cv_results[model]['scores'][i] for model in cv_results]
            
            # Display the results
            st.write(cv_df)
            
            # Visualize cross-validation results
            st.write("##### Model Comparison with Cross-Validation")
            
            # Create data for visualization
            model_names = list(cv_results.keys())
            mean_scores = [cv_results[model]['mean'] for model in cv_results]
            std_scores = [cv_results[model]['std'] for model in cv_results]
            
            # Create error bars
            fig = go.Figure()
            
            # Add bar chart with error bars
            fig.add_trace(go.Bar(
                x=model_names,
                y=mean_scores,
                error_y=dict(
                    type='data',
                    array=std_scores,
                    visible=True
                ),
                marker_color='#FF4B4B'
            ))
            
            # Update layout
            fig.update_layout(
                title='Cross-Validation Accuracy by Model',
                xaxis_title='Model',
                yaxis_title='Accuracy',
                yaxis=dict(
                    range=[0.7, 1.0]  # Adjust as needed
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Fold-by-fold analysis for the best model
            st.write("##### Fold-by-Fold Analysis for Best Model")
            
            # Identify best model
            best_model_name = cv_df.iloc[cv_df['Mean Accuracy'].argmax()]['Model']
            best_model_scores = cv_results[best_model_name]['scores']
            
            # Create chart for fold analysis
            fig_folds = px.bar(
                x=[f'Fold {i+1}' for i in range(k_folds)],
                y=best_model_scores,
                labels={'x': 'Fold', 'y': 'Accuracy'},
                title=f'Accuracy by Fold for {best_model_name}'
            )
            
            # Add line for mean accuracy
            fig_folds.add_hline(
                y=best_model_scores.mean(),
                line_dash="dash",
                line_color="red",
                annotation_text=f"Mean: {best_model_scores.mean():.3f}",
                annotation_position="top right"
            )
            
            st.plotly_chart(fig_folds, use_container_width=True)
            
            # Interpretation
            st.write("""
            ##### Interpretation for Fire Safety
            
            Cross-validation provides several insights for fire safety models:
            
            1. **Model Stability**:
               - Standard deviation across folds shows how consistent the model is
               - Lower variance means more reliable predictions across different data subsets
            
            2. **Model Selection**:
               - Cross-validation helps select the most robust model for deployment
               - The best model should perform well consistently across all folds
            
            3. **Overfitting Detection**:
               - Large performance drops in certain folds may indicate overfitting
               - Particularly important when historical fire data is limited
            
            4. **Operational Confidence**:
               - Cross-validation helps quantify the expected performance variability
               - Critical for fire safety applications where reliable predictions are essential
            """)
            
            # Additional considerations
            st.write("""
            ##### Special Considerations for Fire Safety Data
            
            When using cross-validation with fire safety data:
            
            1. **Temporal Effects**: For time-series data (like sensor readings), use time series cross-validation 
               to respect the temporal order and avoid data leakage.
            
            2. **Rare Events**: For imbalanced data (like rare fire incidents), use stratified cross-validation 
               to maintain class distributions across folds.
            
            3. **Spatial Correlation**: For geographically correlated data (like wildfire spread), consider 
               spatial cross-validation that respects geographical relationships.
            
            4. **Critical Scenarios**: For high-consequence scenarios, consider worst-case performance 
               across folds, not just average performance.
            """)
    
    # Model Selection tab
    with tabs[3]:
        st.write("### Model Selection and Hyperparameter Tuning")
        
        st.write("""
        Selecting the right model and optimizing its hyperparameters are crucial steps in developing 
        reliable machine learning systems for fire safety applications.
        """)
        
        # Model selection process
        st.write("#### Model Selection Process")
        
        st.write("""
        Choosing the right model for a fire safety application involves several considerations:
        
        1. **Problem Characteristics**:
           - Classification vs. regression
           - Linear vs. non-linear relationships
           - Number of features and samples available
        
        2. **Model Properties**:
           - Interpretability requirements
           - Training and inference speed
           - Memory requirements
           - Handling of missing data
        
        3. **Fire Safety Considerations**:
           - Risk tolerance and consequence of errors
           - Regulatory requirements
           - Deployment environment constraints
        """)
        
        # Hyperparameter tuning
        st.write("#### Hyperparameter Tuning Methods")
        
        # Create a table for tuning methods
        tuning_df = pd.DataFrame({
            'Method': [
                'Grid Search', 
                'Random Search',
                'Bayesian Optimization',
                'Genetic Algorithms'
            ],
            'Description': [
                'Exhaustively tries all combinations of predefined hyperparameters',
                'Samples hyperparameter combinations randomly from defined distributions',
                'Uses Bayesian methods to intelligently search the hyperparameter space',
                'Evolves hyperparameter combinations using principles inspired by natural selection'
            ],
            'When to Use': [
                'Few hyperparameters with limited options',
                'Many hyperparameters with large ranges',
                'Computationally expensive models requiring efficient tuning',
                'Complex hyperparameter spaces with many interactions'
            ]
        })
        
        st.write(tuning_df)
        
        # Practical example of hyperparameter tuning
        st.write("#### Practical Example: Hyperparameter Tuning for Fire Detection")
        
        st.write("""
        Let's explore how hyperparameter tuning can improve a Random Forest model for building fire risk classification.
        """)
        
        # Load building data (we're reusing from previous tab)
        if 'df_buildings' in locals() and 'risk_numeric' in df_buildings.columns:
            # For simplicity, we'll simulate a grid search for a Random Forest
            # In a real application, you would use GridSearchCV or RandomizedSearchCV
            
            # Define hyperparameter grid (simplified for demonstration)
            param_grid = {
                'n_estimators': [10, 50, 100, 200],
                'max_depth': [None, 5, 10, 15],
                'min_samples_split': [2, 5, 10]
            }
            
            # Generate sample results
            np.random.seed(42)
            
            # Create a grid of all combinations
            from itertools import product
            
            # Generate all combinations
            all_params = list(product(
                param_grid['n_estimators'],
                param_grid['max_depth'],
                param_grid['min_samples_split']
            ))
            
            # Create synthetic results for demonstration
            tuning_results = []
            
            for params in all_params:
                n_estimators, max_depth, min_samples_split = params
                
                # Create a synthetic accuracy score
                # In reality, this would come from cross-validation
                base_score = 0.82
                
                # Add effects for each parameter (simplified model of their impact)
                estimator_effect = 0.05 * (1 - np.exp(-n_estimators / 100))  # Diminishing returns
                
                depth_effect = 0.0
                if max_depth is not None:
                    # Optimal depth around 10
                    depth_effect = 0.03 * (1 - abs(max_depth - 10) / 10)
                else:
                    depth_effect = 0.02  # None is generally good but can overfit
                
                split_effect = 0.01 * (1 - abs(min_samples_split - 5) / 10)  # Optimal around 5
                
                # Add a bit of randomness
                random_effect = np.random.normal(0, 0.01)
                
                # Combine effects
                accuracy = base_score + estimator_effect + depth_effect + split_effect + random_effect
                accuracy = min(accuracy, 0.99)  # Cap at reasonable values
                
                tuning_results.append({
                    'n_estimators': n_estimators,
                    'max_depth': 'None' if max_depth is None else max_depth,
                    'min_samples_split': min_samples_split,
                    'accuracy': accuracy
                })
            
            # Create DataFrame from results
            tuning_df = pd.DataFrame(tuning_results)
            
            # Display top results
            st.write("##### Top 10 Hyperparameter Combinations")
            st.write(tuning_df.sort_values('accuracy', ascending=False).head(10))
            
            # Visualize the effect of hyperparameters
            st.write("##### Effect of n_estimators on Accuracy")
            
            # Create a subset for visualization, fixing other parameters
            viz_df = tuning_df[tuning_df['max_depth'] == 'None']
            viz_df = viz_df[viz_df['min_samples_split'] == 2]
            
            # Plot the effect of n_estimators
            fig_ne = px.line(
                viz_df, 
                x='n_estimators', 
                y='accuracy',
                markers=True,
                title='Effect of n_estimators on Model Accuracy',
                labels={'n_estimators': 'Number of Trees', 'accuracy': 'Cross-Validation Accuracy'}
            )
            
            st.plotly_chart(fig_ne, use_container_width=True)
            
            # Compare max_depth
            st.write("##### Effect of max_depth on Accuracy")
            
            # Create a subset for visualization, fixing other parameters
            viz_df2 = tuning_df[tuning_df['n_estimators'] == 100]
            viz_df2 = viz_df2[viz_df2['min_samples_split'] == 2]
            
            # Convert 'None' to string for plotting
            viz_df2['max_depth'] = viz_df2['max_depth'].astype(str)
            
            # Plot the effect of max_depth
            fig_md = px.bar(
                viz_df2, 
                x='max_depth', 
                y='accuracy',
                title='Effect of max_depth on Model Accuracy',
                labels={'max_depth': 'Maximum Tree Depth', 'accuracy': 'Cross-Validation Accuracy'}
            )
            
            st.plotly_chart(fig_md, use_container_width=True)
            
            # Interactive parameter exploration
            st.write("##### Interactive Parameter Exploration")
            
            # Allow user to select parameters to visualize
            param1 = st.selectbox(
                "Select first parameter to visualize:",
                ['n_estimators', 'max_depth', 'min_samples_split']
            )
            
            param2 = st.selectbox(
                "Select second parameter to visualize:",
                ['max_depth', 'min_samples_split', 'n_estimators'],
                index=2 if param1 == 'max_depth' else 0
            )
            
            # Ensure parameters are different
            if param1 == param2:
                st.warning("Please select different parameters to compare.")
            else:
                # Determine the fixed parameter
                fixed_params = ['n_estimators', 'max_depth', 'min_samples_split']
                fixed_params.remove(param1)
                fixed_params.remove(param2)
                fixed_param = fixed_params[0]
                
                # Allow user to select value for fixed parameter
                unique_values = tuning_df[fixed_param].unique()
                fixed_value = st.selectbox(
                    f"Select value for {fixed_param}:",
                    unique_values
                )
                
                # Filter data based on fixed parameter
                heatmap_df = tuning_df[tuning_df[fixed_param] == fixed_value].copy()
                
                # Prepare for heatmap
                if 'max_depth' in [param1, param2]:
                    heatmap_df['max_depth'] = heatmap_df['max_depth'].astype(str)
                
                # Create pivot table for heatmap
                pivot_data = heatmap_df.pivot_table(
                    values='accuracy',
                    index=param1,
                    columns=param2
                )
                
                # Create heatmap
                fig_heatmap = px.imshow(
                    pivot_data,
                    labels=dict(x=param2, y=param1, color='Accuracy'),
                    title=f'Accuracy Heatmap: {param1} vs {param2} (with {fixed_param}={fixed_value})',
                    color_continuous_scale='Oranges'
                )
                
                st.plotly_chart(fig_heatmap, use_container_width=True)
            
            # Best model details
            st.write("##### Best Model Configuration")
            
            best_config = tuning_df.loc[tuning_df['accuracy'].idxmax()]
            
            st.write(f"""
            Based on our hyperparameter tuning, the best Random Forest configuration is:
            
            - **Number of Trees**: {best_config['n_estimators']}
            - **Maximum Depth**: {best_config['max_depth']}
            - **Minimum Samples Split**: {best_config['min_samples_split']}
            - **Expected Accuracy**: {best_config['accuracy']:.4f}
            """)
            
            # Final model training and validation
            st.write("""
            After selecting the best hyperparameters, the final model should be:
            
            1. Trained on the entire training dataset with the optimal hyperparameters
            2. Validated on a completely held-out test set that was not used during tuning
            3. Evaluated using appropriate metrics for the fire safety context
            4. Analyzed for potential biases or failure modes
            """)
        
        # Model selection considerations for fire safety
        st.write("""
        #### Model Selection Considerations for Fire Safety
        
        When selecting models for fire safety applications, consider these domain-specific factors:
        
        1. **Interpretability vs. Performance**:
           - Regulatory environments may require explainable models
           - Critical safety decisions might need transparent reasoning
           - Balance prediction accuracy with explainability requirements
        
        2. **False Positives vs. False Negatives**:
           - Evaluate models based on their error profiles
           - Some applications may tolerate false alarms but not missed detections
           - Custom evaluation metrics may be needed to reflect safety priorities
        
        3. **Computational Requirements**:
           - Real-time fire detection needs efficient models
           - Edge devices (like IoT sensors) may have limited computational resources
           - Cloud-based analysis can use more complex models
        
        4. **Uncertainty Quantification**:
           - Models that provide confidence intervals or prediction uncertainty
           - Probabilistic models may be preferred for risk assessment
           - Ensemble methods can provide multiple perspectives on predictions
        
        5. **Robustness to Noise and Outliers**:
           - Fire safety data often contains sensor noise
           - Models should be robust to environmental variations
           - Extreme but important events must be handled appropriately
        """)
    
    # Module quiz
    st.write("### Module 6 Quiz")
    
    questions = [
        {
            'question': 'Which evaluation metric is most important when developing a fire alarm system where missing a real fire would be catastrophic?',
            'options': [
                'Accuracy',
                'Precision',
                'Recall (Sensitivity)',
                'Specificity'
            ],
            'correct': 'Recall (Sensitivity)'
        },
        {
            'question': 'In a confusion matrix for a fire detection model, what does a false positive represent?',
            'options': [
                'The model correctly predicted a fire',
                'The model correctly predicted no fire',
                'The model incorrectly predicted a fire when there was none',
                'The model failed to predict a fire that occurred'
            ],
            'correct': 'The model incorrectly predicted a fire when there was none'
        },
        {
            'question': 'When evaluating a regression model that predicts fire spread time, which metric is in the same units as the prediction?',
            'options': [
                'Mean Squared Error (MSE)',
                'R-squared (R²)',
                'Root Mean Squared Error (RMSE)',
                'Mean Absolute Percentage Error (MAPE)'
            ],
            'correct': 'Root Mean Squared Error (RMSE)'
        },
        {
            'question': 'What is the main purpose of cross-validation in model evaluation?',
            'options': [
                'To make predictions faster',
                'To reduce the need for large datasets',
                'To assess how a model performs on independent data',
                'To eliminate the need for a test set'
            ],
            'correct': 'To assess how a model performs on independent data'
        },
        {
            'question': 'For a time-series based fire prediction model, which cross-validation approach is most appropriate?',
            'options': [
                'Standard k-fold cross-validation',
                'Leave-one-out cross-validation',
                'Time series split cross-validation',
                'Random shuffle cross-validation'
            ],
            'correct': 'Time series split cross-validation'
        },
        {
            'question': 'What does the area under the ROC curve (AUC-ROC) measure?',
            'options': [
                'The overall error rate of the model',
                'The model\'s ability to distinguish between classes',
                'The percentage of correct predictions',
                'The computational efficiency of the model'
            ],
            'correct': 'The model\'s ability to distinguish between classes'
        },
        {
            'question': 'When analyzing residuals in a fire temperature prediction model, what would a random scatter of residuals around zero indicate?',
            'options': [
                'The model has a systematic bias',
                'The model is likely overfitting',
                'The model fits the data well without systematic errors',
                'The model needs more complex features'
            ],
            'correct': 'The model fits the data well without systematic errors'
        },
        {
            'question': 'In hyperparameter tuning for a fire risk classification model, which approach tries all possible combinations of predefined hyperparameter values?',
            'options': [
                'Random Search',
                'Grid Search',
                'Bayesian Optimization',
                'Genetic Algorithm'
            ],
            'correct': 'Grid Search'
        },
        {
            'question': 'Why is stratified k-fold cross-validation particularly useful for fire incident prediction models?',
            'options': [
                'It reduces computational requirements',
                'It works better with regression problems',
                'It preserves the class distribution in imbalanced datasets',
                'It allows for faster model training'
            ],
            'correct': 'It preserves the class distribution in imbalanced datasets'
        },
        {
            'question': 'When selecting a model for a critical fire safety application, which of these factors is typically least important?',
            'options': [
                'The model\'s performance on relevant evaluation metrics',
                'The interpretability of the model\'s decisions',
                'The computational efficiency of the training process',
                'The model\'s robustness to noisy sensor data'
            ],
            'correct': 'The computational efficiency of the training process'
        }
    ]
    
    create_module_quiz(questions, 5)
    
    # Link to Practical Examples
    st.write("---")
    st.write("### Ready to see machine learning in action?")
    st.write("""
    Now that you've learned about model evaluation, you're ready to explore practical applications
    of machine learning in fire safety scenarios!
    """)
    
    if st.button("🔬 Go to Practical ML Examples", use_container_width=True):
        st.session_state.page = 'examples'
        st.rerun()

