import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Color palette for consistent visualization
FIRESAFETY_COLORS = {
    'primary': '#FF5722',       # Deep Orange (Fire)
    'secondary': '#4CAF50',     # Green (Safety)
    'tertiary': '#2196F3',      # Blue (Information)
    'accent': '#9C27B0',        # Purple (Accent)
    'background': '#FAFAFA',    # Light background
    'text': '#212121',          # Dark text
    'light_text': '#757575',    # Light text
    'warning': '#FFC107',       # Amber (Warning)
    'error': '#F44336',         # Red (Error)
    'success': '#4CAF50',       # Green (Success)
    'neutral': '#607D8B'        # Blue Grey (Neutral)
}

# Color sequences for charts
FIRESAFETY_SEQUENTIAL = ['#FFCCBC', '#FFAB91', '#FF8A65', '#FF7043', '#FF5722', '#F4511E', '#E64A19', '#D84315', '#BF360C']
FIRESAFETY_CATEGORICAL = ['#FF5722', '#4CAF50', '#2196F3', '#9C27B0', '#FFC107', '#607D8B', '#795548', '#009688', '#E91E63']

@st.cache_data
def apply_firesafety_theme_to_plotly(fig, title=None, height=None, legend_title=None):
    """
    Apply consistent Fire Safety theme to Plotly figures
    Uses caching to avoid recomputing the styling for the same input
    
    Parameters:
    - fig: A plotly figure object
    - title: Optional title override
    - height: Optional height override
    - legend_title: Optional legend title
    
    Returns:
    - The styled figure
    """
    if title:
        fig.update_layout(title=title)
    
    fig.update_layout(
        font=dict(family="Arial, sans-serif", size=14, color=FIRESAFETY_COLORS['text']),
        plot_bgcolor=FIRESAFETY_COLORS['background'],
        paper_bgcolor=FIRESAFETY_COLORS['background'],
        title_font=dict(size=18, color=FIRESAFETY_COLORS['primary'], family="Arial, sans-serif"),
        margin=dict(l=40, r=40, t=50, b=40),
        hoverlabel=dict(
            bgcolor="white", 
            font_size=14, 
            font_family="Arial, sans-serif"
        ),
        height=height or 450,
        legend=dict(
            title=legend_title,
            bordercolor=FIRESAFETY_COLORS['light_text'],
            borderwidth=1,
            font=dict(family="Arial, sans-serif", size=12)
        ),
        colorway=FIRESAFETY_CATEGORICAL
    )
    
    fig.update_xaxes(
        title_font=dict(size=14, family="Arial, sans-serif"),
        tickfont=dict(size=12, family="Arial, sans-serif"),
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(220, 220, 220, 0.2)',
        showline=True,
        linewidth=1,
        linecolor='rgba(220, 220, 220, 0.8)',
    )
    
    fig.update_yaxes(
        title_font=dict(size=14, family="Arial, sans-serif"),
        tickfont=dict(size=12, family="Arial, sans-serif"),
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(220, 220, 220, 0.2)',
        showline=True,
        linewidth=1,
        linecolor='rgba(220, 220, 220, 0.8)',
    )
    
    return fig

@st.cache_data
def apply_firesafety_theme_to_matplotlib(ax=None, title=None, figsize=(10, 6)):
    """
    Apply consistent Fire Safety theme to Matplotlib plots
    Uses caching to avoid recomputing the styling for the same input
    
    Parameters:
    - ax: A matplotlib axis (or None to create a new one)
    - title: Optional title for the plot
    - figsize: Figure size as tuple (width, height)
    
    Returns:
    - fig, ax: The figure and axis with styling applied
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()
    
    # Set the style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Set background color
    ax.set_facecolor(FIRESAFETY_COLORS['background'])
    fig.patch.set_facecolor(FIRESAFETY_COLORS['background'])
    
    # Grid styling
    ax.grid(color='lightgray', linestyle='--', linewidth=0.7, alpha=0.7)
    
    # Set title if provided
    if title:
        ax.set_title(title, fontsize=16, weight='bold', color=FIRESAFETY_COLORS['primary'])
    
    # Style spines
    for spine in ax.spines.values():
        spine.set_color(FIRESAFETY_COLORS['light_text'])
        spine.set_linewidth(0.5)
    
    # Style axes labels
    ax.xaxis.label.set_color(FIRESAFETY_COLORS['text'])
    ax.yaxis.label.set_color(FIRESAFETY_COLORS['text'])
    ax.xaxis.label.set_fontsize(12)
    ax.yaxis.label.set_fontsize(12)
    
    # Style tick labels
    ax.tick_params(axis='x', colors=FIRESAFETY_COLORS['text'], labelsize=10)
    ax.tick_params(axis='y', colors=FIRESAFETY_COLORS['text'], labelsize=10)
    
    # Add padding
    fig.tight_layout(pad=2.0)
    
    return fig, ax

def plot_fire_incidents_by_cause(df):
    """
    Create a bar chart showing fire incidents by cause
    
    Parameters:
    - df: pandas DataFrame with 'cause' column
    """
    if 'cause' not in df.columns:
        st.error("DataFrame must contain 'cause' column")
        return
    
    # Count incidents by cause
    cause_counts = df['cause'].value_counts().reset_index()
    cause_counts.columns = ['Cause', 'Count']
    
    # Create bar chart with new style
    fig = px.bar(
        cause_counts, 
        x='Cause', 
        y='Count',
        title='Fire Incidents by Cause',
        color='Count',
        color_continuous_scale=FIRESAFETY_SEQUENTIAL
    )
    
    # Apply our custom theme
    fig = apply_firesafety_theme_to_plotly(
        fig, 
        title='Fire Incidents by Cause',
        height=450
    )
    
    # Additional customizations specific to this chart
    fig.update_layout(
        xaxis_title='Cause of Fire',
        yaxis_title='Number of Incidents',
        coloraxis_showscale=False,
    )
    
    # Add tooltip information
    fig.update_traces(
        hovertemplate='<b>%{x}</b><br>Incidents: %{y}<extra></extra>'
    )
    
    st.plotly_chart(fig, use_container_width=True)

def plot_fire_damage_by_building_type(df):
    """
    Create a grouped bar chart showing damage levels by building type
    
    Parameters:
    - df: pandas DataFrame with 'construction_type' and 'damage_level' columns
    """
    if 'construction_type' not in df.columns or 'damage_level' not in df.columns:
        st.error("DataFrame must contain 'construction_type' and 'damage_level' columns")
        return
    
    # Count incidents by building type and damage level
    damage_by_type = pd.crosstab(df['construction_type'], df['damage_level'])
    
    # Create grouped bar chart with modern styling
    fig = px.bar(
        damage_by_type, 
        barmode='group',
        title='Damage Levels by Building Construction Type',
        color_discrete_sequence=FIRESAFETY_CATEGORICAL
    )
    
    # Apply our custom fire safety theme
    fig = apply_firesafety_theme_to_plotly(
        fig, 
        title='Damage Levels by Building Construction Type',
        height=450,
        legend_title='Damage Level'
    )
    
    # Additional customizations specific to this chart
    fig.update_layout(
        xaxis_title='Construction Type',
        yaxis_title='Number of Incidents',
        bargap=0.15,  # Gap between bars in the same group
        bargroupgap=0.1  # Gap between bar groups
    )
    
    # Add hover template
    fig.update_traces(
        hovertemplate='<b>%{x}</b><br>Damage Level: %{data.name}<br>Incidents: %{y}<extra></extra>'
    )
    
    st.plotly_chart(fig, use_container_width=True)

def plot_linear_regression_example(df):
    """
    Create a scatter plot with linear regression line
    
    Parameters:
    - df: pandas DataFrame with numerical features
    """
    if 'temperature_celsius' not in df.columns or 'smoke_density' not in df.columns:
        st.error("DataFrame must contain 'temperature_celsius' and 'smoke_density' columns")
        return
    
    # Prepare data
    X = df['temperature_celsius'].values.reshape(-1, 1)
    y = df['smoke_density'].values
    
    # Fit linear regression model
    model = LinearRegression()
    model.fit(X, y)
    
    # Generate predictions for line
    x_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    y_pred = model.predict(x_range)
    
    # Create scatter plot with regression line
    fig = go.Figure()
    
    # Add scatter points with more visually appealing style
    fig.add_trace(go.Scatter(
        x=df['temperature_celsius'],
        y=df['smoke_density'],
        mode='markers',
        name='Data Points',
        marker=dict(
            color=FIRESAFETY_COLORS['primary'],
            size=10,
            opacity=0.7,
            line=dict(width=1, color='white')
        )
    ))
    
    # Add regression line with improved styling
    fig.add_trace(go.Scatter(
        x=x_range.flatten(),
        y=y_pred,
        mode='lines',
        name=f'Regression Line (R² = {model.score(X, y):.2f})',
        line=dict(
            color=FIRESAFETY_COLORS['secondary'],
            width=3,
            dash='solid'
        )
    ))
    
    # Apply our custom fire safety theme
    fig = apply_firesafety_theme_to_plotly(
        fig, 
        title='Linear Regression: Temperature vs Smoke Density',
        height=480
    )
    
    # Additional customizations
    fig.update_layout(
        xaxis_title='Temperature (°C)',
        yaxis_title='Smoke Density',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Add confidence interval shading (95% confidence)
    # This is a simplified version for educational purposes
    from scipy import stats
    
    # Calculate confidence interval
    n = len(X)
    mean_x = np.mean(X)
    std_err = np.sqrt(np.sum((y - model.predict(X)) ** 2) / (n - 2)) * np.sqrt(1/n + (x_range - mean_x)**2 / np.sum((X - mean_x)**2))
    
    # Calculate upper and lower bounds 
    y_upper = y_pred + 1.96 * std_err
    y_lower = y_pred - 1.96 * std_err
    
    # Add confidence interval
    fig.add_trace(go.Scatter(
        x=np.concatenate([x_range.flatten(), x_range.flatten()[::-1]]),
        y=np.concatenate([y_upper.flatten(), y_lower.flatten()[::-1]]),
        fill='toself',
        fillcolor='rgba(76, 175, 80, 0.1)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo='skip',
        showlegend=False
    ))
    
    # Update trace ordering (confidence interval should be below the line)
    fig.data = [fig.data[2], fig.data[0], fig.data[1]]
    
    # Add hover templates
    fig.update_traces(
        hovertemplate='Temperature: %{x:.1f}°C<br>Smoke Density: %{y:.2f}<extra></extra>',
        selector=dict(mode='markers')
    )
    
    fig.update_traces(
        hovertemplate='Temperature: %{x:.1f}°C<br>Predicted Density: %{y:.2f}<extra></extra>',
        selector=dict(mode='lines')
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Create an attractive box for model information
    st.markdown("""
    <style>
    .model-info-box {
        background-color: #f8f9fa;
        border-left: 5px solid #FF5722;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0 20px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .model-info-box h4 {
        color: #FF5722;
        margin-top: 0;
        margin-bottom: 10px;
    }
    .model-info-box ul {
        list-style-type: none;
        padding-left: 5px;
    }
    .model-info-box li {
        padding: 5px 0;
        border-bottom: 1px dashed #e0e0e0;
    }
    .model-equation {
        background-color: #e8f5e9;
        padding: 8px 12px;
        border-radius: 4px;
        font-family: monospace;
        margin-top: 10px;
        font-weight: bold;
    }
    </style>
    
    <div class="model-info-box">
        <h4>Linear Regression Model Information</h4>
        <ul>
            <li><strong>Coefficient (slope):</strong> {:.4f}</li>
            <li><strong>Intercept:</strong> {:.4f}</li>
            <li><strong>R² Score:</strong> {:.4f}</li>
        </ul>
        <div class="model-equation">
            Smoke Density = {:.4f} × Temperature + {:.4f}
        </div>
    </div>
    """.format(
        model.coef_[0], 
        model.intercept_, 
        model.score(X, y),
        model.coef_[0],
        model.intercept_
    ), unsafe_allow_html=True)

def plot_decision_tree_viz():
    """
    Display a visualization of a decision tree for fire risk classification
    This is a static visualization for educational purposes
    """
    # Create a sample decision tree visualization (static SVG for demonstration)
    decision_tree_svg = """
    <svg width="800" height="400" xmlns="http://www.w3.org/2000/svg">
        <!-- Root node -->
        <rect x="350" y="20" width="100" height="50" rx="5" fill="#ff4b4b" stroke="black" />
        <text x="400" y="45" text-anchor="middle" font-family="Arial" font-size="12" fill="white">Building Age > 25?</text>
        <text x="400" y="60" text-anchor="middle" font-family="Arial" font-size="10" fill="white">samples = 100</text>
        
        <!-- Connecting lines -->
        <line x1="350" y1="45" x2="200" y2="110" stroke="black" stroke-width="2" />
        <line x1="450" y1="45" x2="600" y2="110" stroke="black" stroke-width="2" />
        <text x="260" y="80" text-anchor="middle" font-family="Arial" font-size="12">True</text>
        <text x="540" y="80" text-anchor="middle" font-family="Arial" font-size="12">False</text>
        
        <!-- Left internal node -->
        <rect x="150" y="110" width="100" height="50" rx="5" fill="#ff7b4b" stroke="black" />
        <text x="200" y="135" text-anchor="middle" font-family="Arial" font-size="12" fill="white">Has Sprinklers?</text>
        <text x="200" y="150" text-anchor="middle" font-family="Arial" font-size="10" fill="white">samples = 40</text>
        
        <!-- Right internal node -->
        <rect x="550" y="110" width="100" height="50" rx="5" fill="#ff7b4b" stroke="black" />
        <text x="600" y="135" text-anchor="middle" font-family="Arial" font-size="12" fill="white">Floors > 5?</text>
        <text x="600" y="150" text-anchor="middle" font-family="Arial" font-size="10" fill="white">samples = 60</text>
        
        <!-- Connecting lines level 2 -->
        <line x1="150" y1="135" x2="75" y2="200" stroke="black" stroke-width="2" />
        <line x1="250" y1="135" x2="325" y2="200" stroke="black" stroke-width="2" />
        <line x1="550" y1="135" x2="475" y2="200" stroke="black" stroke-width="2" />
        <line x1="650" y1="135" x2="725" y2="200" stroke="black" stroke-width="2" />
        
        <!-- Leaf nodes -->
        <rect x="25" y="200" width="100" height="50" rx="5" fill="#ffd700" stroke="black" />
        <text x="75" y="225" text-anchor="middle" font-family="Arial" font-size="12">High Risk</text>
        <text x="75" y="240" text-anchor="middle" font-family="Arial" font-size="10">samples = 25</text>
        
        <rect x="275" y="200" width="100" height="50" rx="5" fill="#90ee90" stroke="black" />
        <text x="325" y="225" text-anchor="middle" font-family="Arial" font-size="12">Low Risk</text>
        <text x="325" y="240" text-anchor="middle" font-family="Arial" font-size="10">samples = 15</text>
        
        <rect x="425" y="200" width="100" height="50" rx="5" fill="#90ee90" stroke="black" />
        <text x="475" y="225" text-anchor="middle" font-family="Arial" font-size="12">Low Risk</text>
        <text x="475" y="240" text-anchor="middle" font-family="Arial" font-size="10">samples = 40</text>
        
        <rect x="675" y="200" width="100" height="50" rx="5" fill="#ffd700" stroke="black" />
        <text x="725" y="225" text-anchor="middle" font-family="Arial" font-size="12">Medium Risk</text>
        <text x="725" y="240" text-anchor="middle" font-family="Arial" font-size="10">samples = 20</text>
        
        <!-- Legend -->
        <rect x="50" y="300" width="20" height="20" fill="#ffd700" stroke="black" />
        <text x="80" y="315" font-family="Arial" font-size="12">High Risk</text>
        
        <rect x="150" y="300" width="20" height="20" fill="#90ee90" stroke="black" />
        <text x="180" y="315" font-family="Arial" font-size="12">Low Risk</text>
        
        <rect x="250" y="300" width="20" height="20" fill="#ff7b4b" stroke="black" />
        <text x="280" y="315" font-family="Arial" font-size="12">Internal Node</text>
    </svg>
    """
    
    st.components.v1.html(decision_tree_svg, height=400)

def plot_clustering_example(df):
    """
    Demonstrate K-means clustering on sensor data
    
    Parameters:
    - df: pandas DataFrame with numerical features
    """
    if 'temperature_celsius' not in df.columns or 'smoke_density' not in df.columns:
        st.error("DataFrame must contain 'temperature_celsius' and 'smoke_density' columns")
        return
    
    # Select a subset of data for demonstration
    df_sample = df[['temperature_celsius', 'smoke_density']].sample(min(100, len(df)))
    
    # Standardize the data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(df_sample)
    
    # Perform K-means clustering
    n_clusters = 3
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(data_scaled)
    
    # Get cluster centers and transform back to original scale
    centers_scaled = kmeans.cluster_centers_
    centers = scaler.inverse_transform(centers_scaled)
    
    # Create a DataFrame with the original data and cluster labels
    df_clustered = df_sample.copy()
    
    # Sort clusters by temperature for consistent coloring
    cluster_temp_means = {}
    for i in range(n_clusters):
        mask = cluster_labels == i
        if np.any(mask):
            cluster_temp_means[i] = df_sample.loc[mask, 'temperature_celsius'].mean()
    
    # Sort clusters by temperature (low to high)
    sorted_clusters = sorted(cluster_temp_means.items(), key=lambda x: x[1])
    cluster_map = {old_idx: new_idx for new_idx, (old_idx, _) in enumerate(sorted_clusters)}
    
    # Map to new cluster labels
    df_clustered['Cluster'] = [f"Cluster {cluster_map[i]}" for i in cluster_labels]
    
    # Define colors based on risk level (low to high temp)
    cluster_colors = [
        FIRESAFETY_COLORS['secondary'],  # Green (Low risk/temp)
        FIRESAFETY_COLORS['warning'],     # Amber (Medium risk/temp)
        FIRESAFETY_COLORS['primary']      # Red (High risk/temp)
    ]
    
    # Create scatter plot with clusters
    fig = px.scatter(
        df_clustered, 
        x='temperature_celsius', 
        y='smoke_density',
        color='Cluster',
        title='K-means Clustering: Fire Risk Assessment from Sensor Data',
        color_discrete_sequence=cluster_colors,
        size_max=12,
        opacity=0.8
    )
    
    # Apply consistent theme
    fig = apply_firesafety_theme_to_plotly(
        fig, 
        title='K-means Clustering: Fire Risk Assessment from Sensor Data',
        height=480,
        legend_title='Risk Level'
    )
    
    # Add cluster centers with improved styling
    for i, (old_idx, _) in enumerate(sorted_clusters):
        center_x, center_y = centers[old_idx]
        fig.add_trace(go.Scatter(
            x=[center_x],
            y=[center_y],
            mode='markers',
            marker=dict(
                symbol='x',
                size=16,
                color=cluster_colors[i],
                line=dict(width=3, color='white')
            ),
            name=f'Center {i+1}'
        ))
    
    # Enhanced layout
    fig.update_layout(
        xaxis_title='Temperature (°C)',
        yaxis_title='Smoke Density',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Improved hover info
    fig.update_traces(
        hovertemplate='<b>%{customdata}</b><br>Temperature: %{x:.1f}°C<br>Smoke Density: %{y:.2f}<extra></extra>',
        customdata=["Risk Level " + str(i+1) for i in range(len(df_clustered))],
        selector=dict(mode='markers')
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Create card layout for cluster descriptions
    st.markdown("""
    <style>
    .cluster-cards {
        display: flex;
        flex-wrap: wrap;
        gap: 15px;
        margin-top: 20px;
        margin-bottom: 30px;
    }
    .cluster-card {
        flex: 1;
        min-width: 200px;
        border-radius: 8px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .cluster-card.low {
        background-color: rgba(76, 175, 80, 0.1);
        border-left: 5px solid #4CAF50;
    }
    .cluster-card.medium {
        background-color: rgba(255, 193, 7, 0.1);
        border-left: 5px solid #FFC107;
    }
    .cluster-card.high {
        background-color: rgba(255, 87, 34, 0.1);
        border-left: 5px solid #FF5722;
    }
    .cluster-card h4 {
        margin-top: 0;
        color: #212121;
    }
    .cluster-card-stats {
        margin: 10px 0;
        font-size: 0.9em;
    }
    </style>
    
    <h3>Cluster Interpretation</h3>
    <div class="cluster-cards">
    """, unsafe_allow_html=True)
    
    # Prepare cluster descriptions
    cluster_descs = [
        "Low temperature and low smoke density - Normal conditions",
        "Medium temperature and medium smoke density - Possible early fire warning",
        "High temperature and high smoke density - Likely fire conditions"
    ]
    
    # Sort clusters by temperature
    cluster_means = []
    for new_idx in range(n_clusters):
        cluster_name = f"Cluster {new_idx}"
        mean_temp = df_clustered[df_clustered['Cluster'] == cluster_name]['temperature_celsius'].mean()
        mean_smoke = df_clustered[df_clustered['Cluster'] == cluster_name]['smoke_density'].mean()
        count = len(df_clustered[df_clustered['Cluster'] == cluster_name])
        cluster_means.append((new_idx, mean_temp, mean_smoke, count))
    
    # Create cards for each cluster
    card_classes = ["low", "medium", "high"]
    for i, (cluster_idx, mean_temp, mean_smoke, count) in enumerate(cluster_means):
        st.markdown(f"""
        <div class="cluster-card {card_classes[i]}">
            <h4>Cluster {cluster_idx}: {cluster_descs[i].split(' - ')[1]}</h4>
            <p>{cluster_descs[i].split(' - ')[0]}</p>
            <div class="cluster-card-stats">
                <div><strong>Average Temperature:</strong> {mean_temp:.2f}°C</div>
                <div><strong>Average Smoke Density:</strong> {mean_smoke:.2f}</div>
                <div><strong>Number of Data Points:</strong> {count}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Add educational note
    st.info("""
    **About K-means Clustering**: K-means is an unsupervised learning algorithm that identifies patterns 
    in data by grouping similar data points into clusters. In fire safety applications, clustering 
    can help identify different risk levels or operating conditions based on sensor readings.
    """)

def plot_pca_example(df):
    """
    Demonstrate PCA dimensionality reduction
    
    Parameters:
    - df: pandas DataFrame with numerical features
    """
    # Select numerical columns only
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    
    if len(num_cols) < 3:
        st.error("DataFrame must contain at least 3 numerical columns")
        return
    
    # Select top 5 numerical columns to avoid too many features
    features = num_cols[:5]
    
    # Standardize the data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(df[features])
    
    # Apply PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(data_scaled)
    
    # Create a DataFrame with PCA results
    df_pca = pd.DataFrame(
        data=pca_result,
        columns=['Principal Component 1', 'Principal Component 2']
    )
    
    # Generate some pseudo-labels for demonstration (based on quadrants)
    df_pca['Group'] = 'Group 3'  # Default group
    df_pca.loc[(df_pca['Principal Component 1'] > 0) & (df_pca['Principal Component 2'] > 0), 'Group'] = 'Group 1'
    df_pca.loc[(df_pca['Principal Component 1'] < 0) & (df_pca['Principal Component 2'] > 0), 'Group'] = 'Group 2'
    df_pca.loc[(df_pca['Principal Component 1'] < 0) & (df_pca['Principal Component 2'] < 0), 'Group'] = 'Group 4'
    
    # Create scatter plot of PCA results with improved styling
    fig = px.scatter(
        df_pca,
        x='Principal Component 1',
        y='Principal Component 2',
        color='Group',
        title='PCA: Dimensionality Reduction of Fire Safety Data',
        color_discrete_sequence=FIRESAFETY_CATEGORICAL,
        opacity=0.8,
        size_max=10
    )
    
    # Apply our custom fire safety theme
    fig = apply_firesafety_theme_to_plotly(
        fig, 
        title='PCA: Dimensionality Reduction of Fire Safety Data',
        height=450,
        legend_title='Data Group'
    )
    
    # Add a origin crosshair lines
    fig.add_shape(
        type="line", line=dict(dash="dash", width=1, color="gray"),
        x0=df_pca['Principal Component 1'].min(), y0=0, 
        x1=df_pca['Principal Component 1'].max(), y1=0
    )
    fig.add_shape(
        type="line", line=dict(dash="dash", width=1, color="gray"),
        x0=0, y0=df_pca['Principal Component 2'].min(), 
        x1=0, y1=df_pca['Principal Component 2'].max()
    )
    
    # Improved hover info
    fig.update_traces(
        hovertemplate='<b>%{customdata}</b><br>PC1: %{x:.3f}<br>PC2: %{y:.3f}<extra></extra>',
        customdata=df_pca['Group'],
        selector=dict(mode='markers')
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Show explained variance with nicer visualization
    explained_variance = pca.explained_variance_ratio_
    
    # Create donut chart for explained variance
    variance_fig = go.Figure()
    variance_fig.add_trace(go.Pie(
        labels=['PC1', 'PC2', 'Remaining Variance'],
        values=[
            explained_variance[0] * 100,
            explained_variance[1] * 100,
            (1 - sum(explained_variance)) * 100
        ],
        hole=0.5,
        marker=dict(colors=[
            FIRESAFETY_COLORS['primary'],
            FIRESAFETY_COLORS['secondary'],
            FIRESAFETY_COLORS['light_text']
        ])
    ))
    
    # Apply our custom theme
    variance_fig = apply_firesafety_theme_to_plotly(
        variance_fig, 
        title='PCA Explained Variance (%)',
        height=350
    )
    
    # Update text info for the pie chart
    variance_fig.update_traces(
        textinfo='label+percent',
        textposition='outside',
        hoverinfo='label+percent'
    )
    
    # Add central annotation
    variance_fig.update_layout(
        annotations=[dict(
            text=f"{sum(explained_variance)*100:.1f}%<br>Total",
            x=0.5, y=0.5,
            font_size=20,
            showarrow=False
        )]
    )
    
    st.plotly_chart(variance_fig, use_container_width=True)
    
    # Feature importance in PCA with modernized visualization
    loadings = pca.components_
    
    # Create more visually appealing loadings heatmap
    loadings_df = pd.DataFrame(
        loadings.T,
        columns=['PC1', 'PC2'],
        index=features
    )
    
    # Sort features by absolute loading value on PC1
    loadings_df = loadings_df.reindex(
        loadings_df.abs().sort_values(by='PC1', ascending=False).index
    )
    
    # Create heatmap instead of bar chart
    loadings_fig = px.imshow(
        loadings_df,
        labels=dict(
            x="Principal Component",
            y="Feature",
            color="Loading Value"
        ),
        x=['PC1', 'PC2'],
        y=loadings_df.index,
        color_continuous_scale=FIRESAFETY_SEQUENTIAL,
        aspect="auto"
    )
    
    # Apply our custom theme
    loadings_fig = apply_firesafety_theme_to_plotly(
        loadings_fig, 
        title='Feature Loadings in Principal Components',
        height=400
    )
    
    # Show actual values in cells
    loadings_fig.update_traces(
        text=np.around(loadings_df.values, 2),
        texttemplate="%{text}",
        hovertemplate="Feature: %{y}<br>Component: %{x}<br>Loading: %{z:.3f}<extra></extra>"
    )
    
    st.plotly_chart(loadings_fig, use_container_width=True)
    
    # Add educational explanation
    st.info("""
    **About Principal Component Analysis (PCA)**  
    PCA is a dimensionality reduction technique that finds the directions (principal components) 
    of maximum variance in the data. In fire safety applications, PCA can help identify the most 
    important features or sensors that contribute to variance in the data, making it easier to 
    identify patterns or anomalies that might indicate fire risks.
    """)
    
    # Interpretation of results
    st.markdown("""
    <div style="background-color: #F5F5F5; padding: 15px; border-radius: 5px; margin-top: 20px;">
        <h3 style="margin-top: 0; color: #333;">Interpreting the PCA Results</h3>
        <ul>
            <li><strong>PC1 and PC2</strong> represent the two main dimensions that capture the most variance in the data.</li>
            <li>The <strong>heatmap</strong> shows how much each original feature contributes to each principal component.</li>
            <li>Features with <strong>larger absolute values</strong> in the heatmap have more influence on that component.</li>
            <li>The <strong>donut chart</strong> shows how much of the total variance is explained by each component.</li>
        </ul>
        <p>This analysis helps identify which sensor measurements or building characteristics are most important 
        for distinguishing different fire risk scenarios.</p>
    </div>
    """, unsafe_allow_html=True)

def plot_cnn_architecture():
    """Display a visualization of a CNN architecture for fire detection"""
    
    # Create a simplified CNN architecture visualization (static SVG)
    cnn_svg = """
    <svg width="800" height="300" xmlns="http://www.w3.org/2000/svg">
        <!-- Input Layer -->
        <rect x="50" y="100" width="100" height="100" fill="#FF4B4B" stroke="black" />
        <text x="100" y="90" text-anchor="middle" font-family="Arial" font-size="12">Input Image</text>
        <text x="100" y="155" text-anchor="middle" font-family="Arial" font-size="10" fill="white">224×224×3</text>
        
        <!-- Convolution Layer 1 -->
        <rect x="200" y="110" width="80" height="80" fill="#FF7B4B" stroke="black" />
        <text x="240" y="90" text-anchor="middle" font-family="Arial" font-size="12">Conv1</text>
        <text x="240" y="155" text-anchor="middle" font-family="Arial" font-size="10" fill="white">112×112×64</text>
        
        <!-- Arrow 1 -->
        <line x1="150" y1="150" x2="200" y2="150" stroke="black" stroke-width="2" />
        <polygon points="195,145 200,150 195,155" fill="black" />
        
        <!-- Convolution Layer 2 -->
        <rect x="330" y="120" width="60" height="60" fill="#FF7B4B" stroke="black" />
        <text x="360" y="90" text-anchor="middle" font-family="Arial" font-size="12">Conv2</text>
        <text x="360" y="155" text-anchor="middle" font-family="Arial" font-size="10" fill="white">56×56×128</text>
        
        <!-- Arrow 2 -->
        <line x1="280" y1="150" x2="330" y2="150" stroke="black" stroke-width="2" />
        <polygon points="325,145 330,150 325,155" fill="black" />
        
        <!-- Pooling Layer -->
        <rect x="440" y="130" width="40" height="40" fill="#FFA500" stroke="black" />
        <text x="460" y="90" text-anchor="middle" font-family="Arial" font-size="12">MaxPool</text>
        <text x="460" y="155" text-anchor="middle" font-family="Arial" font-size="10" fill="white">28×28×128</text>
        
        <!-- Arrow 3 -->
        <line x1="390" y1="150" x2="440" y2="150" stroke="black" stroke-width="2" />
        <polygon points="435,145 440,150 435,155" fill="black" />
        
        <!-- Fully Connected Layer -->
        <rect x="530" y="125" width="50" height="50" fill="#90EE90" stroke="black" />
        <text x="555" y="90" text-anchor="middle" font-family="Arial" font-size="12">FC</text>
        <text x="555" y="155" text-anchor="middle" font-family="Arial" font-size="10" fill="white">1024</text>
        
        <!-- Arrow 4 -->
        <line x1="480" y1="150" x2="530" y2="150" stroke="black" stroke-width="2" />
        <polygon points="525,145 530,150 525,155" fill="black" />
        
        <!-- Output Layer -->
        <rect x="630" y="135" width="30" height="30" fill="#ADD8E6" stroke="black" />
        <text x="645" y="90" text-anchor="middle" font-family="Arial" font-size="12">Output</text>
        <text x="645" y="155" text-anchor="middle" font-family="Arial" font-size="10" fill="black">2</text>
        
        <!-- Arrow 5 -->
        <line x1="580" y1="150" x2="630" y2="150" stroke="black" stroke-width="2" />
        <polygon points="625,145 630,150 625,155" fill="black" />
        
        <!-- Legend -->
        <rect x="50" y="230" width="20" height="20" fill="#FF4B4B" stroke="black" />
        <text x="80" y="245" font-family="Arial" font-size="12">Input Layer</text>
        
        <rect x="200" y="230" width="20" height="20" fill="#FF7B4B" stroke="black" />
        <text x="230" y="245" font-family="Arial" font-size="12">Convolutional Layers</text>
        
        <rect x="400" y="230" width="20" height="20" fill="#FFA500" stroke="black" />
        <text x="430" y="245" font-family="Arial" font-size="12">Pooling Layer</text>
        
        <rect x="550" y="230" width="20" height="20" fill="#90EE90" stroke="black" />
        <text x="580" y="245" font-family="Arial" font-size="12">Fully Connected Layer</text>
        
        <rect x="700" y="230" width="20" height="20" fill="#ADD8E6" stroke="black" />
        <text x="730" y="245" font-family="Arial" font-size="12">Output Layer</text>
    </svg>
    """
    
    st.components.v1.html(cnn_svg, height=300)
