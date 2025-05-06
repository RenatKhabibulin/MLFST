import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from utils import load_dataset, create_module_quiz, create_code_editor
from visualization import plot_clustering_example, plot_pca_example
from ml_utils import perform_kmeans_clustering, perform_pca

def show_module_content():
    """Display unsupervised learning algorithms content"""
    
    st.write("## Unsupervised Learning Algorithms for Fire Safety")
    
    # Overview of unsupervised learning
    st.write("""
    ### Overview
    
    Unsupervised learning algorithms find patterns in data without pre-existing labels. In fire safety, 
    these techniques can reveal hidden structures in data, group similar incidents or buildings, identify 
    anomalies, and reduce data complexity. This module explores key unsupervised learning approaches 
    and their applications in fire safety.
    """)
    
    # Tab-based navigation for different algorithms
    tabs = st.tabs([
        "Clustering", 
        "Dimensionality Reduction",
        "Anomaly Detection"
    ])
    
    # Clustering tab
    with tabs[0]:
        st.write("### Clustering Algorithms")
        
        st.write("""
        Clustering algorithms group similar data points together based on their features. These groups (clusters) 
        can reveal natural patterns in fire safety data.
        
        **Key Concepts:**
        - Groups similar data points without predefined labels
        - Discovers natural patterns and structures
        - Useful for segmentation and grouping
        - Different algorithms handle different cluster shapes
        
        **Common Clustering Algorithms:**
        
        **1. K-means Clustering**
        - **Mathematical Foundation**: Aims to minimize the within-cluster sum of squares (WCSS):
          $J = \sum_{j=1}^{k} \sum_{i=1}^{n} ||x_i^{(j)} - c_j||^2$
          where $x_i^{(j)}$ is a data point in cluster $j$ and $c_j$ is the centroid of cluster $j$
        - **Algorithm Steps**:
          1. Initialize $k$ centroids randomly
          2. Assign each data point to the nearest centroid
          3. Recalculate centroids based on the mean of all points in the cluster
          4. Repeat steps 2-3 until convergence (centroids stop moving significantly)
        - **Strengths**: Simple, efficient, works well with globular clusters
        - **Limitations**: Requires specifying $k$ in advance, sensitive to initial centroids, struggles with non-globular shapes
        - **Fire Safety Applications**: Grouping buildings with similar risk profiles, identifying common incident patterns
        
        **2. DBSCAN (Density-Based Spatial Clustering of Applications with Noise)**
        - **Mathematical Foundation**: Groups points based on density, using two parameters:
          - $\epsilon$ (eps): The radius of the neighborhood
          - MinPts: Minimum number of points required to form a dense region
        - **Algorithm Steps**:
          1. For each point, find all points within distance $\epsilon$
          2. Identify core points (those with at least MinPts neighbors)
          3. Form clusters by connecting core points that are within $\epsilon$ of each other
          4. Assign non-core points to clusters if they're neighbors with a core point
          5. Points not assigned to any cluster are considered noise
        - **Strengths**: Doesn't require specifying number of clusters, can find arbitrarily shaped clusters, robust to outliers
        - **Limitations**: Sensitive to parameters, struggles with varying density clusters
        - **Fire Safety Applications**: Identifying unusual incident patterns, detecting spatial clusters of fire occurrences
        
        **3. Agglomerative Hierarchical Clustering**
        - **Mathematical Foundation**: Builds a hierarchy of clusters by:
          1. Starting with each point as a separate cluster
          2. Merging the two most similar clusters at each step
          3. Continuing until all points are in a single cluster
        - **Distance Metrics**:
          - Single linkage: $d(C_i, C_j) = \min_{x \in C_i, y \in C_j} d(x, y)$
          - Complete linkage: $d(C_i, C_j) = \max_{x \in C_i, y \in C_j} d(x, y)$
          - Average linkage: $d(C_i, C_j) = \dfrac{1}{|C_i||C_j|} \sum_{x \in C_i} \sum_{y \in C_j} d(x, y)$
          - Ward's method: Minimizes the increase in total within-cluster variance
        - **Strengths**: Creates hierarchical structure (dendrogram), doesn't require pre-specifying number of clusters
        - **Limitations**: Computationally expensive $O(n^3)$, can't handle large datasets well
        - **Fire Safety Applications**: Creating hierarchical organization of building risks, structuring incident response protocols
        
        **Fire Safety Applications:**
        - Grouping buildings by similar fire risk profiles
        - Identifying common patterns in fire incidents
        - Segmenting geographic areas by fire occurrence patterns
        - Categorizing sensor readings for anomaly detection
        """)
        
        # K-means practical example
        st.write("### Practical Example: Clustering Buildings by Fire Risk Factors")
        
        # Load building data
        df_buildings = load_dataset("building_characteristics.csv")
        
        if not df_buildings.empty:
            # Select numerical features for clustering
            features = ['age_years', 'floors']
            
            # Add binary features
            df_buildings['has_sprinkler_system_num'] = df_buildings['has_sprinkler_system'].astype(int)
            df_buildings['has_fire_alarm_num'] = df_buildings['has_fire_alarm'].astype(int)
            
            features.extend(['has_sprinkler_system_num', 'has_fire_alarm_num'])
            
            # Display selected features
            st.write("First few rows of selected features:")
            st.write(df_buildings[features].head())
            
            # Perform K-means clustering
            X = df_buildings[features].values
            
            # Apply clustering
            model, labels, metrics = perform_kmeans_clustering(X, n_clusters=3)
            
            # Add cluster labels to the dataframe
            df_buildings['cluster'] = labels
            
            # Visualize the clusters (using the first two features for 2D plotting)
            st.write("#### Building Clusters Visualization")
            
            # Create a scatter plot of the clusters
            fig = px.scatter(
                df_buildings, 
                x='age_years', 
                y='floors',
                color='cluster',
                symbol='has_sprinkler_system',
                size='has_fire_alarm_num',
                hover_data=['building_id', 'construction_type', 'occupancy_type'],
                title='Building Clusters Based on Risk Factors',
                labels={
                    'age_years': 'Building Age (years)',
                    'floors': 'Number of Floors',
                    'cluster': 'Cluster',
                    'has_sprinkler_system': 'Has Sprinkler System',
                    'has_fire_alarm_num': 'Has Fire Alarm'
                },
                color_discrete_sequence=['#FF4B4B', '#FF7B4B', '#FFA500']
            )
            
            # Add cluster centers
            centers = metrics['cluster_centers']
            
            for i, center in enumerate(centers):
                fig.add_annotation(
                    x=center[0],
                    y=center[1],
                    text=f"Cluster {i} Center",
                    showarrow=True,
                    arrowhead=1,
                    arrowsize=1,
                    arrowwidth=2,
                    arrowcolor="black",
                    font=dict(size=12, color="black"),
                    bgcolor="white",
                    bordercolor="black",
                    borderwidth=1
                )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Cluster analysis
            st.write("#### Cluster Analysis")
            
            # Calculate cluster statistics
            cluster_stats = df_buildings.groupby('cluster').agg({
                'age_years': 'mean',
                'floors': 'mean',
                'has_sprinkler_system_num': 'mean',
                'has_fire_alarm_num': 'mean',
                'building_id': 'count'
            }).reset_index()
            
            cluster_stats.columns = ['Cluster', 'Average Age', 'Average Floors', 
                                   'Sprinkler System %', 'Fire Alarm %', 'Count']
            
            # Convert to percentages
            cluster_stats['Sprinkler System %'] = cluster_stats['Sprinkler System %'] * 100
            cluster_stats['Fire Alarm %'] = cluster_stats['Fire Alarm %'] * 100
            
            # Display cluster statistics
            st.write(cluster_stats)
            
            # Cluster characterization
            st.write("#### Cluster Descriptions")
            
            cluster_descs = [
                "**Cluster 0 - Older, Taller Buildings**: These buildings are older with more floors, but a moderate percentage have safety systems.",
                "**Cluster 1 - Modern, Well-Equipped Buildings**: These are newer buildings with good safety systems installed.",
                "**Cluster 2 - Mixed-Age, Basic Safety Buildings**: These buildings vary in age but typically have fewer floors and basic safety systems."
            ]
            
            for desc in cluster_descs:
                st.write(desc)
            
            # Application to fire safety
            st.write("""
            **Applications to Fire Safety:**
            
            Clustering buildings allows fire departments to:
            
            1. **Prioritize Inspections**: Focus more attention on high-risk clusters
            2. **Customize Prevention Strategies**: Develop tailored approaches for each building type
            3. **Resource Allocation**: Deploy resources more effectively based on cluster distribution
            4. **Risk Assessment**: Create more nuanced risk profiles beyond simple scoring
            
            This approach moves beyond one-size-fits-all safety policies to recognize different building profiles
            and their unique fire safety needs.
            """)
            
            # Interactive element - Choose features for clustering
            st.write("#### Try Different Features for Clustering")
            
            available_features = features + ['occupancy_type_numeric']
            
            # Create a numeric encoding for occupancy type
            occupancy_mapping = {ot: i for i, ot in enumerate(df_buildings['occupancy_type'].unique())}
            df_buildings['occupancy_type_numeric'] = df_buildings['occupancy_type'].map(occupancy_mapping)
            
            # Let user select features
            selected_features = st.multiselect(
                "Select features for clustering:",
                available_features,
                default=features[:2]  # Default to first two features for simplicity
            )
            
            # Let user select number of clusters
            n_clusters = st.slider("Number of clusters:", min_value=2, max_value=5, value=3)
            
            # If user has selected features, perform clustering
            if len(selected_features) >= 2:
                X_selected = df_buildings[selected_features].values
                
                # Perform clustering
                model, custom_labels, metrics = perform_kmeans_clustering(X_selected, n_clusters=n_clusters)
                
                # Add cluster labels to a copy of the dataframe
                df_clusters = df_buildings.copy()
                df_clusters['custom_cluster'] = custom_labels
                
                # Visualize the clusters using the first two selected features
                st.write(f"#### Clustering with {', '.join(selected_features)}")
                
                # Create a scatter plot
                fig_custom = px.scatter(
                    df_clusters, 
                    x=selected_features[0], 
                    y=selected_features[1],
                    color='custom_cluster',
                    hover_data=['building_id', 'construction_type', 'occupancy_type'],
                    title=f'Building Clusters Based on Selected Features',
                    color_discrete_sequence=['#FF4B4B', '#FF7B4B', '#FFA500', '#FFD700', '#FFFF00'][:n_clusters]
                )
                
                st.plotly_chart(fig_custom, use_container_width=True)
            else:
                st.info("Please select at least two features to perform clustering.")
            
            # Add a DBSCAN demonstration section
            st.write("### DBSCAN Clustering Example")
            st.write("""
            DBSCAN is particularly useful for finding clusters of irregular shapes and identifying outliers.
            Let's see how it works with fire safety data to identify irregular patterns.
            """)
            
            # Allow the user to set DBSCAN parameters
            st.write("#### Set DBSCAN Parameters")
            epsilon = st.slider("Epsilon (neighborhood distance):", 
                               min_value=0.1, max_value=5.0, value=1.0, step=0.1)
            min_samples = st.slider("Minimum samples in neighborhood:", 
                                   min_value=2, max_value=10, value=5)
            
            if st.button("Run DBSCAN Clustering"):
                # Import DBSCAN
                from sklearn.cluster import DBSCAN
                from sklearn.preprocessing import StandardScaler
                
                # Prepare data for DBSCAN (using all numeric features)
                X_all = df_buildings[features].values
                
                # Scale the data (important for DBSCAN)
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X_all)
                
                # Apply DBSCAN
                dbscan = DBSCAN(eps=epsilon, min_samples=min_samples)
                dbscan_labels = dbscan.fit_predict(X_scaled)
                
                # Add cluster labels to dataframe
                df_dbscan = df_buildings.copy()
                df_dbscan['dbscan_cluster'] = dbscan_labels
                
                # Count the number of clusters (excluding noise points labeled as -1)
                n_clusters_dbscan = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
                n_noise = list(dbscan_labels).count(-1)
                
                # Display DBSCAN results
                st.write(f"DBSCAN identified {n_clusters_dbscan} clusters and {n_noise} noise points.")
                
                # Create a scatter plot for DBSCAN results
                # Use the first two features for visualization
                color_map = {-1: '#808080'}  # Gray for noise points
                colors = ['#FF4B4B', '#4CAF50', '#2196F3', '#9C27B0', '#FF9800', '#795548']
                for i in range(n_clusters_dbscan):
                    color_map[i] = colors[i % len(colors)]
                
                fig_dbscan = px.scatter(
                    df_dbscan, 
                    x=features[0], 
                    y=features[1],
                    color='dbscan_cluster',
                    color_discrete_map=color_map,
                    title='DBSCAN Clustering Results',
                    labels={
                        features[0]: features[0].replace('_', ' ').title(),
                        features[1]: features[1].replace('_', ' ').title(),
                        'dbscan_cluster': 'Cluster'
                    }
                )
                
                # Customize hover info
                fig_dbscan.update_traces(
                    hovertemplate="<br>".join([
                        "Building ID: %{customdata[0]}",
                        f"{features[0]}: %{{x}}",
                        f"{features[1]}: %{{y}}",
                        "Cluster: %{marker.color}"
                    ]),
                    customdata=df_dbscan[['building_id']]
                )
                
                st.plotly_chart(fig_dbscan, use_container_width=True)
                
                # Display findings and interpretations
                if n_clusters_dbscan > 0:
                    st.write("#### DBSCAN Cluster Analysis")
                    
                    # Calculate statistics for each cluster
                    cluster_stats = []
                    for i in range(-1, n_clusters_dbscan):
                        cluster_data = df_dbscan[df_dbscan['dbscan_cluster'] == i]
                        cluster_size = len(cluster_data)
                        
                        if cluster_size > 0:
                            cluster_stats.append({
                                "Cluster": "Noise" if i == -1 else f"Cluster {i}",
                                "Size": cluster_size,
                                "% of Data": f"{(cluster_size / len(df_dbscan) * 100):.1f}%",
                                "Avg Building Age": f"{cluster_data['age_years'].mean():.1f}",
                                "Avg Floors": f"{cluster_data['floors'].mean():.1f}",
                                "% with Sprinklers": f"{(cluster_data['has_sprinkler_system_num'].mean() * 100):.1f}%",
                                "% with Alarms": f"{(cluster_data['has_fire_alarm_num'].mean() * 100):.1f}%"
                            })
                    
                    # Convert to dataframe and display
                    stats_df = pd.DataFrame(cluster_stats)
                    st.write(stats_df)
                    
                    # Interpretation
                    st.write("""
                    #### DBSCAN vs K-means
                    
                    **Advantages of DBSCAN in Fire Safety:**
                    
                    1. **Irregular Cluster Shapes**: DBSCAN can detect clusters with non-spherical shapes, which is valuable for
                       identifying complex risk patterns that follow geographical or structural features.
                       
                    2. **Noise Detection**: Buildings that don't fit into any cluster can be identified as outliers that may need
                       special attention or individualized risk assessment.
                       
                    3. **No Predefined Clusters**: The algorithm discovers the natural number of clusters, which helps identify
                       the intrinsic risk groupings without human bias.
                    
                    **Applications:**
                    
                    - Identifying unusual buildings that require special inspection protocols
                    - Finding natural geographical clusters of fire incidents
                    - Detecting anomalous sensor patterns that don't fit normal behavior
                    """)
                else:
                    st.warning("No clusters were found with the current parameters. Try adjusting epsilon or min_samples.")
            
            # Add hierarchical clustering demonstration
            st.write("### Hierarchical Clustering Example")
            st.write("""
            Hierarchical clustering builds a tree of clusters, allowing us to examine relationships
            at different levels of granularity. This is especially useful for organizing fire safety 
            protocols and understanding building risk relationships.
            """)
            
            if st.button("Run Hierarchical Clustering"):
                # Import necessary libraries
                from scipy.cluster.hierarchy import dendrogram, linkage
                from sklearn.preprocessing import StandardScaler
                import matplotlib.pyplot as plt
                
                # Prepare data for hierarchical clustering
                X_hier = df_buildings[features].values
                
                # Scale the data
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X_hier)
                
                # Compute the linkage matrix
                st.write("#### Hierarchical Clustering Dendrogram")
                st.write("""
                The dendrogram below shows how buildings are grouped hierarchically. 
                Each merge represents joining similar buildings or clusters.
                The height of each merge indicates the dissimilarity between merged groups.
                """)
                
                # Create figure for dendrogram
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Create the linkage matrix
                methods = {
                    "Ward": "ward",  # Minimizes variance within clusters
                    "Complete": "complete",  # Maximum distance between clusters
                    "Average": "average",  # Average distance between clusters
                    "Single": "single"  # Minimum distance between clusters
                }
                
                # Let user select the linkage method
                linkage_method = st.selectbox(
                    "Select linkage method:",
                    list(methods.keys()),
                    index=0
                )
                
                # Generate the linkage matrix
                Z = linkage(X_scaled, method=methods[linkage_method])
                
                # Plot the dendrogram
                dendrogram(
                    Z,
                    ax=ax,
                    truncate_mode='level',
                    p=5,  # Show only first p levels
                    leaf_rotation=90.,
                    leaf_font_size=10.,
                    labels=df_buildings['building_id'].astype(str).values if len(df_buildings) < 50 else None
                )
                
                ax.set_title(f'Hierarchical Clustering Dendrogram ({linkage_method} linkage)')
                ax.set_xlabel('Building ID or Cluster')
                ax.set_ylabel('Distance (dissimilarity)')
                
                # Display the plot
                st.pyplot(fig)
                
                # Display interpretation
                st.write("""
                #### Interpreting the Dendrogram:
                
                1. **Height of Joins**: Longer vertical lines indicate more dissimilar clusters.
                2. **Clusters**: Cutting the dendrogram horizontally at any level creates a clustering at that level of similarity.
                3. **Hierarchy**: Moving up the tree reveals broader groupings, while moving down shows finer distinctions.
                
                #### Applications in Fire Safety:
                
                - **Risk Hierarchy**: Organizing buildings into a risk hierarchy with multiple levels of detail
                - **Inspection Planning**: Creating tiered inspection protocols based on building similarity
                - **Resource Allocation**: Hierarchical allocation of prevention resources based on cluster relationships
                - **Response Planning**: Developing tiered response protocols based on building similarities
                
                #### Mathematical Intuition:
                
                - **${linkage_method} Linkage**: This method defines cluster distances based on {linkage_method.lower()} distances between points.
                - **Distance Measure**: The height in the dendrogram represents the dissimilarity between merged clusters.
                - **Optimal Clustering**: The number of clusters can be determined by looking for large vertical gaps.
                """.replace("{linkage_method}", linkage_method))
                
                # Let user try cutting the dendrogram
                n_clusters = st.slider("Number of clusters (cut dendrogram):", 
                                     min_value=2, max_value=10, value=3)
                
                # Apply hierarchical clustering with the specified number of clusters
                from scipy.cluster.hierarchy import fcluster
                hier_labels = fcluster(Z, n_clusters, criterion='maxclust') - 1  # Zero-based indexing
                
                # Add cluster labels to dataframe
                df_hier = df_buildings.copy()
                df_hier['hier_cluster'] = hier_labels
                
                # Visualize the hierarchical clusters
                fig_hier = px.scatter(
                    df_hier, 
                    x=features[0], 
                    y=features[1],
                    color='hier_cluster',
                    title=f'Hierarchical Clustering with {n_clusters} Clusters',
                    labels={
                        features[0]: features[0].replace('_', ' ').title(),
                        features[1]: features[1].replace('_', ' ').title(),
                        'hier_cluster': 'Cluster'
                    },
                    color_discrete_sequence=['#FF4B4B', '#4CAF50', '#2196F3', '#9C27B0', '#FF9800', 
                                           '#795548', '#607D8B', '#E91E63', '#03A9F4', '#8BC34A'][:n_clusters]
                )
                
                st.plotly_chart(fig_hier, use_container_width=True)
                
                # Calculate and display cluster statistics
                st.write("#### Hierarchical Cluster Statistics")
                
                hier_stats = df_hier.groupby('hier_cluster').agg({
                    'age_years': 'mean',
                    'floors': 'mean',
                    'has_sprinkler_system_num': 'mean',
                    'has_fire_alarm_num': 'mean',
                    'building_id': 'count'
                }).reset_index()
                
                hier_stats.columns = ['Cluster', 'Average Age', 'Average Floors', 
                                    'Sprinkler System %', 'Fire Alarm %', 'Count']
                
                # Convert to percentages
                hier_stats['Sprinkler System %'] = hier_stats['Sprinkler System %'] * 100
                hier_stats['Fire Alarm %'] = hier_stats['Fire Alarm %'] * 100
                
                st.write(hier_stats)
    
    # Dimensionality Reduction tab
    with tabs[1]:
        st.write("### Dimensionality Reduction")
        
        st.write("""
        Dimensionality reduction techniques transform high-dimensional data into a lower-dimensional space 
        while preserving important information. This helps with visualization, computational efficiency, 
        and revealing hidden patterns.
        
        **Key Concepts:**
        - Reduces data complexity by transforming to fewer dimensions
        - Helps visualize high-dimensional data
        - Removes redundant or less important features
        - Can improve machine learning model performance
        
        **Common Dimensionality Reduction Techniques:**
        1. **Principal Component Analysis (PCA)**: Transforms data to a new coordinate system to maximize variance
        2. **t-SNE**: Visualizes high-dimensional data by giving each datapoint a location in a 2D or 3D map
        3. **UMAP**: A newer technique that preserves both local and global structure
        
        **Fire Safety Applications:**
        - Visualizing complex relationships between fire risk factors
        - Identifying key factors that influence fire spread
        - Reducing noise in sensor data
        - Feature engineering for predictive models
        """)
        
        # PCA practical example
        st.write("### Practical Example: Analyzing Fire Incident Factors")
        
        # Load fire incident data
        df_incidents = load_dataset("fire_incidents.csv")
        
        if not df_incidents.empty:
            # Add derived features for analysis
            df_incidents['detection_time_mins'] = df_incidents['detection_time'].apply(
                lambda x: int(str(x).split(':')[0]) * 60 + int(str(x).split(':')[1]) if pd.notna(x) and isinstance(x, str) else 0
            )
            df_incidents['arrival_time_mins'] = df_incidents['arrival_time'].apply(
                lambda x: int(str(x).split(':')[0]) * 60 + int(str(x).split(':')[1]) if pd.notna(x) and isinstance(x, str) else 0
            )
            df_incidents['extinguishing_time_mins'] = df_incidents['extinguishing_time'].apply(
                lambda x: int(str(x).split(':')[0]) * 60 + int(str(x).split(':')[1]) if pd.notna(x) and isinstance(x, str) else 0
            )
            
            # Calculate time differences
            df_incidents['response_time'] = df_incidents['arrival_time_mins'] - df_incidents['detection_time_mins']
            df_incidents.loc[df_incidents['response_time'] < 0, 'response_time'] += 24 * 60  # Fix day boundary
            
            df_incidents['extinguish_time'] = df_incidents['extinguishing_time_mins'] - df_incidents['arrival_time_mins']
            df_incidents.loc[df_incidents['extinguish_time'] < 0, 'extinguish_time'] += 24 * 60  # Fix day boundary
            
            # Create numeric encoding for categorical variables
            cause_mapping = {cause: i for i, cause in enumerate(df_incidents['cause'].unique())}
            damage_mapping = {level: i for i, level in enumerate(df_incidents['damage_level'].unique())}
            
            df_incidents['cause_numeric'] = df_incidents['cause'].map(cause_mapping)
            df_incidents['damage_numeric'] = df_incidents['damage_level'].map(damage_mapping)
            
            # Select features for PCA
            features = [
                'detection_time_mins', 
                'response_time', 
                'extinguish_time',
                'cause_numeric',
                'damage_numeric'
            ]
            
            # Display selected features
            st.write("First few rows of selected features:")
            st.write(df_incidents[features].head())
            
            # Perform PCA
            X = df_incidents[features].values
            
            # Apply PCA
            model, X_pca, metrics = perform_pca(X, n_components=2)
            
            # Create dataframe with PCA results
            df_pca = pd.DataFrame(
                data=X_pca,
                columns=['Principal Component 1', 'Principal Component 2']
            )
            
            # Add original cause and damage level for coloring
            df_pca['Cause'] = df_incidents['cause'].values
            df_pca['Damage Level'] = df_incidents['damage_level'].values
            
            # Visualize PCA results
            st.write("#### PCA Visualization of Fire Incidents")
            
            # Create a scatter plot with color by cause
            fig_pca_cause = px.scatter(
                df_pca, 
                x='Principal Component 1', 
                y='Principal Component 2',
                color='Cause',
                title='PCA of Fire Incidents by Cause',
                color_discrete_sequence=px.colors.qualitative.Set1
            )
            
            st.plotly_chart(fig_pca_cause, use_container_width=True)
            
            # Create a scatter plot with color by damage level
            fig_pca_damage = px.scatter(
                df_pca, 
                x='Principal Component 1', 
                y='Principal Component 2',
                color='Damage Level',
                title='PCA of Fire Incidents by Damage Level',
                color_discrete_sequence=['#66BB6A', '#FFA726', '#EF5350']
            )
            
            st.plotly_chart(fig_pca_damage, use_container_width=True)
            
            # Show explained variance
            explained_variance = metrics['explained_variance_ratio']
            total_variance = sum(explained_variance)
            
            st.write("#### PCA Explained Variance")
            st.write(f"- Principal Component 1: {explained_variance[0]*100:.2f}%")
            st.write(f"- Principal Component 2: {explained_variance[1]*100:.2f}%")
            st.write(f"- Total Explained Variance: {total_variance*100:.2f}%")
            
            # Feature contribution to principal components
            components = metrics['components']
            feature_names = features
            
            # Create a dataframe of loadings
            loadings = pd.DataFrame(
                components,
                columns=feature_names
            )
            
            st.write("#### Feature Contributions to Principal Components")
            st.write("The values show how much each feature contributes to each principal component.")
            st.write(loadings)
            
            # Visualize feature contributions
            fig_loadings = px.bar(
                loadings.T.reset_index(),
                x='index',
                y=[0, 1],
                barmode='group',
                labels={'index': 'Features', 'value': 'Loading', 'variable': 'Component'},
                title='Feature Contributions to Principal Components',
                color_discrete_sequence=['#FF4B4B', '#FF7B4B']
            )
            
            fig_loadings.update_layout(
                xaxis_title='Features',
                yaxis_title='Loading Value'
            )
            
            st.plotly_chart(fig_loadings, use_container_width=True)
            
            # Interpretation
            st.write("""
            **Interpretation:**
            
            The PCA results show how fire incidents can be projected onto a 2D space that captures the maximum 
            variance in the data. The scatter plots reveal patterns in how different types of fires (by cause and 
            damage level) cluster together in this reduced dimensional space.
            
            The loadings indicate which features contribute most to each principal component:
            
            1. **Principal Component 1** appears to be related to fire timeline factors (detection, response, and extinguishing times).
            2. **Principal Component 2** seems to capture more of the fire characteristics (cause and damage level).
            
            These insights can help fire departments understand the underlying patterns in fire incidents and potentially
            identify factors that separate different types of fires.
            """)
            
            # Interactive PCA exploration
            st.write("#### Interactive PCA Feature Selection")
            
            # Add more potential features
            all_features = features + ['incident_id']
            
            # Let user select features
            selected_pca_features = st.multiselect(
                "Select features for PCA:",
                all_features,
                default=features[:3]
            )
            
            # If at least two features selected, perform PCA
            if len(selected_pca_features) >= 2:
                X_selected = df_incidents[selected_pca_features].values
                
                # Perform PCA
                model, X_pca_custom, metrics = perform_pca(X_selected, n_components=2)
                
                # Create dataframe with PCA results
                df_pca_custom = pd.DataFrame(
                    data=X_pca_custom,
                    columns=['Principal Component 1', 'Principal Component 2']
                )
                
                # Add original cause and damage level for coloring
                df_pca_custom['Cause'] = df_incidents['cause'].values
                
                # Visualize custom PCA results
                fig_pca_custom = px.scatter(
                    df_pca_custom, 
                    x='Principal Component 1', 
                    y='Principal Component 2',
                    color='Cause',
                    title=f'PCA with Selected Features: {", ".join(selected_pca_features)}',
                    color_discrete_sequence=px.colors.qualitative.Set1
                )
                
                st.plotly_chart(fig_pca_custom, use_container_width=True)
                
                # Show explained variance
                explained_variance = metrics['explained_variance_ratio']
                total_variance = sum(explained_variance)
                
                st.write(f"Total Explained Variance: {total_variance*100:.2f}%")
            else:
                st.info("Please select at least two features for PCA.")
    
    # Anomaly Detection tab
    with tabs[2]:
        st.write("### Anomaly Detection")
        
        st.write("""
        Anomaly detection identifies data points that deviate significantly from the majority of the data. 
        In fire safety, this is crucial for detecting unusual patterns that might indicate fire risks or 
        equipment malfunctions.
        
        **Key Concepts:**
        - Identifies outliers or unusual patterns in data
        - Can work with or without labeled examples of anomalies
        - Often uses statistical methods or machine learning
        - Crucial for early warning systems
        
        **Common Anomaly Detection Techniques:**
        1. **Statistical Methods**: Using z-scores, IQR, or parametric distributions
        2. **Isolation Forest**: Isolates observations by randomly selecting features
        3. **One-Class SVM**: Learns a boundary around normal data
        4. **Autoencoders**: Neural networks that learn to reconstruct normal data
        
        **Fire Safety Applications:**
        - Detecting abnormal sensor readings before fires start
        - Identifying unusual fire development patterns
        - Monitoring equipment for maintenance needs
        - Flagging suspicious fire incidents for investigation
        """)
        
        # Practical example
        st.write("### Practical Example: Detecting Anomalies in Fire Sensor Data")
        
        # Load sensor data
        df_sensors = load_dataset("sensor_readings.csv")
        
        if not df_sensors.empty:
            st.write("#### Statistical Anomaly Detection in Sensor Readings")
            
            # Display original data sample
            st.write("First few rows of sensor data:")
            st.write(df_sensors[['reading_id', 'temperature_celsius', 'smoke_density', 'carbon_monoxide_ppm']].head())
            
            # Apply statistical anomaly detection using Z-scores
            # Select features for anomaly detection
            anomaly_features = ['temperature_celsius', 'smoke_density', 'carbon_monoxide_ppm']
            
            # Create a copy for analysis
            df_anomaly = df_sensors.copy()
            
            # Calculate Z-scores for each feature
            for feature in anomaly_features:
                col_zscore = feature + '_zscore'
                df_anomaly[col_zscore] = (df_anomaly[feature] - df_anomaly[feature].mean()) / df_anomaly[feature].std()
            
            # Identify anomalies (absolute Z-score > 2.5 for any feature)
            df_anomaly['is_anomaly'] = (
                (df_anomaly['temperature_celsius_zscore'].abs() > 2.5) | 
                (df_anomaly['smoke_density_zscore'].abs() > 2.5) | 
                (df_anomaly['carbon_monoxide_ppm_zscore'].abs() > 2.5)
            )
            
            # Count anomalies
            anomaly_count = df_anomaly['is_anomaly'].sum()
            normal_count = len(df_anomaly) - anomaly_count
            
            st.write(f"Detected {anomaly_count} anomalies out of {len(df_anomaly)} readings ({anomaly_count/len(df_anomaly)*100:.1f}%)")
            
            # Visualize anomalies
            st.write("#### Anomaly Visualization")
            
            # Create scatter plot of temperature vs smoke density, highlighting anomalies
            fig_anomaly = px.scatter(
                df_anomaly, 
                x='temperature_celsius', 
                y='smoke_density',
                color='is_anomaly',
                hover_data=['reading_id', 'carbon_monoxide_ppm'],
                title='Anomaly Detection in Fire Sensor Readings',
                labels={
                    'temperature_celsius': 'Temperature (°C)',
                    'smoke_density': 'Smoke Density',
                    'is_anomaly': 'Is Anomaly'
                },
                color_discrete_map={
                    True: '#FF4B4B',
                    False: '#1E88E5'
                }
            )
            
            st.plotly_chart(fig_anomaly, use_container_width=True)
            
            # Visualize anomalies in 3D
            st.write("#### 3D Visualization of Anomalies")
            
            fig_3d = px.scatter_3d(
                df_anomaly, 
                x='temperature_celsius', 
                y='smoke_density', 
                z='carbon_monoxide_ppm',
                color='is_anomaly',
                hover_data=['reading_id'],
                title='3D View of Anomalies in Sensor Readings',
                labels={
                    'temperature_celsius': 'Temperature (°C)',
                    'smoke_density': 'Smoke Density',
                    'carbon_monoxide_ppm': 'CO (ppm)',
                    'is_anomaly': 'Is Anomaly'
                },
                color_discrete_map={
                    True: '#FF4B4B',
                    False: '#1E88E5'
                }
            )
            
            st.plotly_chart(fig_3d, use_container_width=True)
            
            # Detailed anomaly analysis
            st.write("#### Anomaly Analysis")
            
            # Display statistics for normal vs anomalous readings
            anomaly_stats = df_anomaly.groupby('is_anomaly')[anomaly_features].agg(['mean', 'std']).round(2)
            st.write("Statistics for Normal vs Anomalous Readings:")
            st.write(anomaly_stats)
            
            # Show some example anomalies
            st.write("Sample of detected anomalies:")
            st.write(df_anomaly[df_anomaly['is_anomaly']][anomaly_features + ['reading_id']].head(10))
            
            # Interpretation
            st.write("""
            **Interpretation:**
            
            The anomaly detection identifies sensor readings that significantly deviate from normal patterns. These outliers could indicate:
            
            1. **Potential Fire Conditions**: Unusually high temperature, smoke, or CO readings might signal a fire
            2. **Sensor Malfunctions**: Extremely abnormal readings might indicate sensor failures
            3. **Environmental Anomalies**: Unusual conditions like steam could trigger false readings
            4. **Data Recording Errors**: Some anomalies might be due to data logging issues
            
            Early detection of these anomalies is crucial for preventive maintenance of fire detection systems and for
            identifying potential fire risks before they develop into actual fires.
            """)
            
            # Interactive anomaly threshold adjustment
            st.write("#### Adjust Anomaly Detection Threshold")
            
            z_threshold = st.slider(
                "Z-score threshold for anomaly detection:",
                min_value=1.0,
                max_value=4.0,
                value=2.5,
                step=0.1,
                help="Lower values will detect more anomalies, higher values will be more selective"
            )
            
            # Recalculate anomalies with new threshold
            df_anomaly['is_custom_anomaly'] = (
                (df_anomaly['temperature_celsius_zscore'].abs() > z_threshold) | 
                (df_anomaly['smoke_density_zscore'].abs() > z_threshold) | 
                (df_anomaly['carbon_monoxide_ppm_zscore'].abs() > z_threshold)
            )
            
            # Count anomalies with new threshold
            custom_anomaly_count = df_anomaly['is_custom_anomaly'].sum()
            
            st.write(f"With threshold {z_threshold}, detected {custom_anomaly_count} anomalies ({custom_anomaly_count/len(df_anomaly)*100:.1f}%)")
            
            # Visualize with custom threshold
            fig_custom = px.scatter(
                df_anomaly, 
                x='temperature_celsius', 
                y='smoke_density',
                color='is_custom_anomaly',
                hover_data=['reading_id', 'carbon_monoxide_ppm'],
                title=f'Anomaly Detection with Z-score Threshold = {z_threshold}',
                labels={
                    'temperature_celsius': 'Temperature (°C)',
                    'smoke_density': 'Smoke Density',
                    'is_custom_anomaly': 'Is Anomaly'
                },
                color_discrete_map={
                    True: '#FF4B4B',
                    False: '#1E88E5'
                }
            )
            
            st.plotly_chart(fig_custom, use_container_width=True)
    
    # Python code examples
    with st.expander("Python Code Examples"):
        st.write("#### K-means Clustering Example")
        
        kmeans_code = """
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# Load building data (sample code)
df = pd.read_csv('building_characteristics.csv')

# Select and prepare features for clustering
features = ['age_years', 'floors', 'has_sprinkler_system', 'has_fire_alarm']
X = df[features].copy()

# Convert boolean columns to integers
X['has_sprinkler_system'] = X['has_sprinkler_system'].astype(int)
X['has_fire_alarm'] = X['has_fire_alarm'].astype(int)

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Determine optimal number of clusters using the elbow method
inertia = []
for k in range(1, 10):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# Plot elbow curve
plt.figure(figsize=(8, 5))
plt.plot(range(1, 10), inertia, 'o-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')
plt.grid(True)
plt.show()

# Apply K-means with the chosen number of clusters
n_clusters = 3  # Based on elbow curve
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
df['cluster'] = kmeans.fit_predict(X_scaled)

# Get cluster centers and transform back to original scale
centers = scaler.inverse_transform(kmeans.cluster_centers_)

# Analyze clusters
cluster_stats = df.groupby('cluster').agg({
    'age_years': 'mean',
    'floors': 'mean',
    'has_sprinkler_system': 'mean',
    'has_fire_alarm': 'mean',
    'building_id': 'count'
}).reset_index()

print("Cluster Statistics:")
print(cluster_stats)

# Visualize clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='age_years', y='floors', hue='cluster', palette='Set1')
plt.title('Building Clusters by Age and Floors')
plt.xlabel('Building Age (years)')
plt.ylabel('Number of Floors')

# Plot cluster centers
for i, center in enumerate(centers):
    plt.plot(center[0], center[1], 'o', markersize=10, marker='*', 
             color='black', label=f'Cluster {i} center' if i == 0 else "")
plt.legend()
plt.grid(True)
plt.show()
"""
        
        st.code(kmeans_code, language="python")
        
        st.write("#### PCA Example")
        
        pca_code = """
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Load fire incident data (sample code)
df = pd.read_csv('fire_incidents.csv')

# Prepare data - create numeric features
# Add derived time fields (pseudo-code, adapt to your actual data)
df['response_time'] = # time difference calculation
df['extinguish_time'] = # time difference calculation

# Create numeric encoding for categorical variables
cause_mapping = {cause: i for i, cause in enumerate(df['cause'].unique())}
damage_mapping = {level: i for i, level in enumerate(df['damage_level'].unique())}

df['cause_numeric'] = df['cause'].map(cause_mapping)
df['damage_numeric'] = df['damage_level'].map(damage_mapping)

# Select features for PCA
features = [
    'response_time', 
    'extinguish_time',
    'cause_numeric',
    'damage_numeric'
]

X = df[features].values

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Create dataframe with PCA results
df_pca = pd.DataFrame(
    data=X_pca,
    columns=['Principal Component 1', 'Principal Component 2']
)

# Add original cause and damage level for visualization
df_pca['Cause'] = df['cause'].values
df_pca['Damage Level'] = df['damage_level'].values

# Plot PCA results
plt.figure(figsize=(12, 10))

# First subplot - color by cause
plt.subplot(2, 1, 1)
sns.scatterplot(
    data=df_pca,
    x='Principal Component 1',
    y='Principal Component 2',
    hue='Cause',
    palette='Set1'
)
plt.title('PCA of Fire Incidents by Cause')
plt.grid(True)

# Second subplot - color by damage level
plt.subplot(2, 1, 2)
sns.scatterplot(
    data=df_pca,
    x='Principal Component 1',
    y='Principal Component 2',
    hue='Damage Level',
    palette='RdYlGn_r'
)
plt.title('PCA of Fire Incidents by Damage Level')
plt.grid(True)

plt.tight_layout()
plt.show()

# Print explained variance
print("Explained Variance Ratio:")
print(pca.explained_variance_ratio_)
print(f"Total Explained Variance: {sum(pca.explained_variance_ratio_):.2f}")

# Visualize feature contributions to principal components
components = pd.DataFrame(
    pca.components_,
    columns=features,
    index=['PC1', 'PC2']
)
print("\nPrincipal Components:")
print(components)

# Heatmap of component loadings
plt.figure(figsize=(10, 6))
sns.heatmap(components, annot=True, cmap='coolwarm', cbar=True)
plt.title('Feature Contributions to Principal Components')
plt.tight_layout()
plt.show()
"""
        
        st.code(pca_code, language="python")
        
        st.write("#### Anomaly Detection Example")
        
        anomaly_code = """
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import seaborn as sns

# Load sensor data (sample code)
df = pd.read_csv('sensor_readings.csv')

# Select features for anomaly detection
features = ['temperature_celsius', 'smoke_density', 'carbon_monoxide_ppm']

# Method 1: Statistical anomaly detection using Z-scores
df_stats = df.copy()

# Calculate Z-scores for each feature
for feature in features:
    col_zscore = feature + '_zscore'
    df_stats[col_zscore] = (df_stats[feature] - df_stats[feature].mean()) / df_stats[feature].std()

# Identify anomalies (absolute Z-score > 2.5 for any feature)
df_stats['is_anomaly_zscore'] = (
    (df_stats[features[0] + '_zscore'].abs() > 2.5) | 
    (df_stats[features[1] + '_zscore'].abs() > 2.5) | 
    (df_stats[features[2] + '_zscore'].abs() > 2.5)
)

# Method 2: Isolation Forest
X = df[features].values

# Initialize and fit the model
iso_forest = IsolationForest(contamination=0.05, random_state=42)
df['is_anomaly_iforest'] = iso_forest.fit_predict(X)

# Convert predictions to boolean (Isolation Forest returns -1 for anomalies, 1 for normal)
df['is_anomaly_iforest'] = df['is_anomaly_iforest'] == -1

# Compare results
zscore_anomalies = df_stats['is_anomaly_zscore'].sum()
iforest_anomalies = df['is_anomaly_iforest'].sum()

print(f"Z-score detected {zscore_anomalies} anomalies ({zscore_anomalies/len(df)*100:.1f}%)")
print(f"Isolation Forest detected {iforest_anomalies} anomalies ({iforest_anomalies/len(df)*100:.1f}%)")

# Visualize anomalies detected by Z-score method
plt.figure(figsize=(12, 10))

# First subplot - 2D view
plt.subplot(2, 1, 1)
sns.scatterplot(
    data=df_stats,
    x='temperature_celsius',
    y='smoke_density',
    hue='is_anomaly_zscore',
    palette={True: 'red', False: 'blue'},
    alpha=0.7
)
plt.title('Anomalies Detected by Z-score Method')
plt.grid(True)

# Second subplot - Isolation Forest results
plt.subplot(2, 1, 2)
sns.scatterplot(
    data=df,
    x='temperature_celsius',
    y='smoke_density',
    hue='is_anomaly_iforest',
    palette={True: 'red', False: 'blue'},
    alpha=0.7
)
plt.title('Anomalies Detected by Isolation Forest')
plt.grid(True)

plt.tight_layout()
plt.show()

# 3D visualization of anomalies
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Normal points
normal = df_stats[~df_stats['is_anomaly_zscore']]
ax.scatter(
    normal['temperature_celsius'],
    normal['smoke_density'],
    normal['carbon_monoxide_ppm'],
    c='blue',
    marker='o',
    alpha=0.5,
    label='Normal'
)

# Anomalies
anomalies = df_stats[df_stats['is_anomaly_zscore']]
ax.scatter(
    anomalies['temperature_celsius'],
    anomalies['smoke_density'],
    anomalies['carbon_monoxide_ppm'],
    c='red',
    marker='o',
    alpha=0.7,
    label='Anomaly'
)

ax.set_xlabel('Temperature (°C)')
ax.set_ylabel('Smoke Density')
ax.set_zlabel('CO (ppm)')
ax.set_title('3D View of Anomalies in Sensor Readings')
ax.legend()

plt.tight_layout()
plt.show()
"""
        
        st.code(anomaly_code, language="python")
    
    # Module quiz
    st.write("### Module 4 Quiz")
    
    questions = [
        {
            'question': 'Which unsupervised learning technique groups similar buildings together based on their fire safety characteristics?',
            'options': [
                'Linear Regression',
                'Support Vector Machines',
                'K-means Clustering',
                'Decision Trees'
            ],
            'correct': 'K-means Clustering'
        },
        {
            'question': 'What is the main purpose of Principal Component Analysis (PCA)?',
            'options': [
                'To predict a target variable based on features',
                'To group similar data points together',
                'To reduce dimensionality while preserving variance',
                'To generate synthetic data points'
            ],
            'correct': 'To reduce dimensionality while preserving variance'
        },
        {
            'question': 'In the context of fire sensor data, what would an anomaly typically represent?',
            'options': [
                'A normal sensor reading during routine operation',
                'A sensor reading that significantly deviates from normal patterns',
                'The average temperature reading over time',
                'The number of sensors in a building'
            ],
            'correct': 'A sensor reading that significantly deviates from normal patterns'
        },
        {
            'question': 'What information does the "explained variance ratio" provide in PCA?',
            'options': [
                'The accuracy of a classification model',
                'The proportion of the original variance retained by each principal component',
                'The number of clusters detected in the data',
                'The proportion of anomalies in the dataset'
            ],
            'correct': 'The proportion of the original variance retained by each principal component'
        },
        {
            'question': 'Which method is NOT typically used for anomaly detection?',
            'options': [
                'Statistical methods (z-scores, IQR)',
                'Isolation Forest',
                'One-Class SVM',
                'K-means Regression'
            ],
            'correct': 'K-means Regression'
        },
        {
            'question': 'What key assumption does K-means clustering make about the data?',
            'options': [
                'The data follows a normal distribution',
                'The clusters are spherical and equally sized',
                'There are no outliers in the data',
                'The features are independent of each other'
            ],
            'correct': 'The clusters are spherical and equally sized'
        },
        {
            'question': 'How could clustering be applied to optimize fire department resource allocation?',
            'options': [
                'By predicting the exact time of future fires',
                'By grouping similar buildings to develop targeted inspection strategies',
                'By directly extinguishing fires automatically',
                'By replacing human firefighters with robots'
            ],
            'correct': 'By grouping similar buildings to develop targeted inspection strategies'
        },
        {
            'question': 'In a Z-score based anomaly detection system for fire sensors, what threshold is commonly used to identify anomalies?',
            'options': [
                'Z-score > 0',
                'Z-score > 1',
                '|Z-score| > 2 or 3',
                'Z-score = 0.5'
            ],
            'correct': '|Z-score| > 2 or 3'
        },
        {
            'question': 'What would be a good application of dimensionality reduction in fire safety?',
            'options': [
                'Directly extinguishing fires',
                'Visualizing complex relationships between multiple fire risk factors',
                'Replacing smoke detectors',
                'Calculating insurance premiums'
            ],
            'correct': 'Visualizing complex relationships between multiple fire risk factors'
        },
        {
            'question': 'Which statement best describes the "curse of dimensionality" problem that dimensionality reduction techniques help solve?',
            'options': [
                'Having too few features to make accurate predictions',
                'The tendency of data to become sparse in high-dimensional spaces, making analysis difficult',
                'The curse that causes fires to spread in multiple dimensions',
                'The difficulty in visualizing more than three dimensions on a computer screen'
            ],
            'correct': 'The tendency of data to become sparse in high-dimensional spaces, making analysis difficult'
        }
    ]
    
    create_module_quiz(questions, 3)

