"""
Behavioral profiling utilities for network entities.
"""

import os
import numpy as np
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config


class EntityProfiler:
    """
    Class for profiling network entities based on their behavior.
    """
    
    def __init__(self, entity_column='Source IP', time_window=300):
        """
        Initialize the entity profiler.
        
        Args:
            entity_column (str): Column name for the entity identifier
            time_window (int): Time window size in seconds for aggregating behavior
        """
        self.entity_column = entity_column
        self.time_window = time_window
        self.entity_profiles = {}
        self.global_profile = None
        self.clustering_model = None
        self.pca_model = None
        self.scaler = None
    
    def create_entity_features(self, df, timestamp_column):
        """
        Create behavioral features for each entity.
        
        Args:
            df (pd.DataFrame): Input dataframe
            timestamp_column (str): Column name for the timestamp
            
        Returns:
            pd.DataFrame: Dataframe with entity features
        """
        print(f"Creating behavioral features for entities using {self.entity_column}...")
        
        # Ensure timestamp column is datetime
        if pd.api.types.is_string_dtype(df[timestamp_column]):
            df[timestamp_column] = pd.to_datetime(df[timestamp_column])
        
        # Create time windows
        df['time_window'] = df[timestamp_column].dt.floor(f'{self.time_window}S')
        
        # Define aggregation functions based on available columns
        agg_dict = {}
        
        # Common network traffic columns
        if 'Destination IP' in df.columns:
            agg_dict['Destination IP'] = ['nunique', 'count']
        
        if 'Destination Port' in df.columns:
            agg_dict['Destination Port'] = ['nunique', 'count']
        
        if 'Protocol' in df.columns:
            agg_dict['Protocol'] = ['nunique', 'count']
        
        # Flow-related columns
        flow_columns = [
            'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
            'Total Length of Fwd Packets', 'Total Length of Bwd Packets',
            'Fwd Packet Length Max', 'Fwd Packet Length Min', 'Fwd Packet Length Mean',
            'Bwd Packet Length Max', 'Bwd Packet Length Min', 'Bwd Packet Length Mean',
            'Flow Bytes/s', 'Flow Packets/s'
        ]
        
        for col in flow_columns:
            if col in df.columns:
                agg_dict[col] = ['mean', 'std', 'min', 'max', 'sum']
        
        # If no columns are available, raise an error
        if not agg_dict:
            raise ValueError("No suitable columns found for creating entity features")
        
        # Group by entity and time window
        entity_features = df.groupby([self.entity_column, 'time_window']).agg(agg_dict)
        
        # Flatten the column names
        entity_features.columns = ['_'.join(col).strip() for col in entity_features.columns.values]
        
        # Reset index to get back the entity and time_window columns
        entity_features = entity_features.reset_index()
        
        print(f"Created {len(entity_features.columns) - 2} behavioral features for {entity_features[self.entity_column].nunique()} entities")
        
        return entity_features
    
    def build_entity_profiles(self, entity_features):
        """
        Build behavioral profiles for each entity.
        
        Args:
            entity_features (pd.DataFrame): Dataframe with entity features
            
        Returns:
            dict: Dictionary of entity profiles
        """
        print("Building behavioral profiles for entities...")
        
        # Get feature columns (exclude entity and time window columns)
        feature_columns = [col for col in entity_features.columns 
                          if col != self.entity_column and col != 'time_window']
        
        # Build profile for each entity
        entity_profiles = {}
        
        for entity, group in entity_features.groupby(self.entity_column):
            # Calculate statistics for each feature
            profile = {}
            
            for feature in feature_columns:
                values = group[feature].dropna()
                
                if len(values) == 0:
                    continue
                
                profile[f"{feature}_mean"] = values.mean()
                profile[f"{feature}_std"] = values.std()
                profile[f"{feature}_min"] = values.min()
                profile[f"{feature}_max"] = values.max()
                profile[f"{feature}_median"] = values.median()
                profile[f"{feature}_q1"] = values.quantile(0.25)
                profile[f"{feature}_q3"] = values.quantile(0.75)
            
            # Add number of time windows
            profile['num_time_windows'] = len(group)
            
            # Store profile
            entity_profiles[entity] = profile
        
        self.entity_profiles = entity_profiles
        
        # Build global profile
        global_profile = {}
        
        for feature in feature_columns:
            values = entity_features[feature].dropna()
            
            if len(values) == 0:
                continue
            
            global_profile[f"{feature}_mean"] = values.mean()
            global_profile[f"{feature}_std"] = values.std()
            global_profile[f"{feature}_min"] = values.min()
            global_profile[f"{feature}_max"] = values.max()
            global_profile[f"{feature}_median"] = values.median()
            global_profile[f"{feature}_q1"] = values.quantile(0.25)
            global_profile[f"{feature}_q3"] = values.quantile(0.75)
        
        self.global_profile = global_profile
        
        print(f"Built profiles for {len(entity_profiles)} entities")
        
        return entity_profiles
    
    def detect_anomalous_entities(self, threshold=2.0):
        """
        Detect anomalous entities based on their deviation from the global profile.
        
        Args:
            threshold (float): Z-score threshold for considering a feature anomalous
            
        Returns:
            dict: Dictionary of anomalous entities with their anomaly scores
        """
        print(f"Detecting anomalous entities using threshold {threshold}...")
        
        if not self.entity_profiles or not self.global_profile:
            raise ValueError("Entity profiles not built. Call build_entity_profiles first.")
        
        # Calculate anomaly scores for each entity
        anomaly_scores = {}
        
        for entity, profile in self.entity_profiles.items():
            # Calculate z-scores for each feature
            feature_z_scores = {}
            
            for feature in profile:
                if feature == 'num_time_windows':
                    continue
                
                # Extract base feature name (remove _mean, _std, etc.)
                base_feature = '_'.join(feature.split('_')[:-1])
                stat = feature.split('_')[-1]
                
                # Skip if global profile doesn't have this feature
                if f"{base_feature}_mean" not in self.global_profile or f"{base_feature}_std" not in self.global_profile:
                    continue
                
                # Get global mean and std for this feature
                global_mean = self.global_profile[f"{base_feature}_mean"]
                global_std = self.global_profile[f"{base_feature}_std"]
                
                # Skip if std is zero or very small
                if global_std < 1e-10:
                    continue
                
                # Calculate z-score
                z_score = abs((profile[feature] - global_mean) / global_std)
                feature_z_scores[feature] = z_score
            
            # Count features with z-scores above threshold
            anomalous_features = {feature: z_score for feature, z_score in feature_z_scores.items() if z_score > threshold}
            
            # Calculate overall anomaly score
            if feature_z_scores:
                anomaly_score = sum(feature_z_scores.values()) / len(feature_z_scores)
                
                # Store entity if it has anomalous features
                if anomalous_features:
                    anomaly_scores[entity] = {
                        'anomaly_score': anomaly_score,
                        'anomalous_features': anomalous_features,
                        'num_anomalous_features': len(anomalous_features),
                        'total_features': len(feature_z_scores)
                    }
        
        # Sort entities by anomaly score
        sorted_anomalies = {k: v for k, v in sorted(anomaly_scores.items(), key=lambda item: item[1]['anomaly_score'], reverse=True)}
        
        print(f"Detected {len(sorted_anomalies)} anomalous entities")
        
        return sorted_anomalies
    
    def cluster_entities(self, n_clusters=5, method='kmeans'):
        """
        Cluster entities based on their behavioral profiles.
        
        Args:
            n_clusters (int): Number of clusters for KMeans
            method (str): Clustering method ('kmeans' or 'dbscan')
            
        Returns:
            pd.DataFrame: Dataframe with entity clusters
        """
        print(f"Clustering entities using {method}...")
        
        if not self.entity_profiles:
            raise ValueError("Entity profiles not built. Call build_entity_profiles first.")
        
        # Convert entity profiles to a dataframe
        profile_data = []
        
        for entity, profile in self.entity_profiles.items():
            row = {'entity': entity}
            row.update(profile)
            profile_data.append(row)
        
        profile_df = pd.DataFrame(profile_data)
        
        # Handle missing values
        profile_df = profile_df.fillna(0)
        
        # Extract features (exclude entity column)
        feature_columns = [col for col in profile_df.columns if col != 'entity']
        X = profile_df[feature_columns].values
        
        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Apply PCA for dimensionality reduction
        self.pca_model = PCA(n_components=min(10, len(feature_columns)))
        X_pca = self.pca_model.fit_transform(X_scaled)
        
        # Cluster entities
        if method == 'kmeans':
            self.clustering_model = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = self.clustering_model.fit_predict(X_pca)
        elif method == 'dbscan':
            self.clustering_model = DBSCAN(eps=0.5, min_samples=5)
            clusters = self.clustering_model.fit_predict(X_pca)
        else:
            raise ValueError(f"Unsupported clustering method: {method}")
        
        # Add cluster labels to the dataframe
        profile_df['cluster'] = clusters
        
        # Calculate cluster statistics
        cluster_stats = profile_df.groupby('cluster').agg({
            'entity': 'count'
        }).rename(columns={'entity': 'count'})
        
        print(f"Clustered entities into {cluster_stats.shape[0]} groups")
        print(f"Cluster sizes: {cluster_stats['count'].tolist()}")
        
        return profile_df
    
    def visualize_entity_clusters(self, clustered_df, output_dir=None):
        """
        Visualize entity clusters.
        
        Args:
            clustered_df (pd.DataFrame): Dataframe with entity clusters
            output_dir (str): Directory to save visualizations
            
        Returns:
            None
        """
        print("Visualizing entity clusters...")
        
        if self.pca_model is None:
            raise ValueError("PCA model not initialized. Call cluster_entities first.")
        
        # Extract features (exclude entity and cluster columns)
        feature_columns = [col for col in clustered_df.columns if col != 'entity' and col != 'cluster']
        X = clustered_df[feature_columns].values
        
        # Scale features and apply PCA
        X_scaled = self.scaler.transform(X)
        X_pca = self.pca_model.transform(X_scaled)
        
        # Create PCA plot
        plt.figure(figsize=(10, 8))
        
        # Get unique clusters and assign colors
        clusters = clustered_df['cluster'].unique()
        
        for cluster in clusters:
            # Skip cluster -1 (noise points in DBSCAN)
            if cluster == -1:
                plt.scatter(
                    X_pca[clustered_df['cluster'] == cluster, 0],
                    X_pca[clustered_df['cluster'] == cluster, 1],
                    s=50, c='black', marker='x', label='Noise'
                )
            else:
                plt.scatter(
                    X_pca[clustered_df['cluster'] == cluster, 0],
                    X_pca[clustered_df['cluster'] == cluster, 1],
                    s=50, label=f'Cluster {cluster}'
                )
        
        plt.title('Entity Clusters (PCA)')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.legend()
        plt.grid(True)
        
        # Save or show the plot
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(os.path.join(output_dir, 'entity_clusters.png'))
        else:
            plt.show()
        
        # Create feature importance plot
        plt.figure(figsize=(12, 6))
        
        # Get feature importance from PCA
        components = self.pca_model.components_
        feature_names = feature_columns
        
        # Plot feature importance for the first two components
        plt.figure(figsize=(12, 8))
        
        # First component
        plt.subplot(2, 1, 1)
        plt.bar(range(len(feature_names[:10])), abs(components[0, :10]))
        plt.xticks(range(len(feature_names[:10])), feature_names[:10], rotation=90)
        plt.title('Feature Importance - Principal Component 1')
        plt.tight_layout()
        
        # Second component
        plt.subplot(2, 1, 2)
        plt.bar(range(len(feature_names[:10])), abs(components[1, :10]))
        plt.xticks(range(len(feature_names[:10])), feature_names[:10], rotation=90)
        plt.title('Feature Importance - Principal Component 2')
        plt.tight_layout()
        
        # Save or show the plot
        if output_dir:
            plt.savefig(os.path.join(output_dir, 'feature_importance.png'))
        else:
            plt.show()
    
    def track_entity_behavior_over_time(self, entity_features, entity, output_dir=None):
        """
        Track the behavior of a specific entity over time.
        
        Args:
            entity_features (pd.DataFrame): Dataframe with entity features
            entity (str): Entity to track
            output_dir (str): Directory to save visualizations
            
        Returns:
            pd.DataFrame: Dataframe with entity behavior over time
        """
        print(f"Tracking behavior of entity {entity} over time...")
        
        # Filter data for the specified entity
        entity_data = entity_features[entity_features[self.entity_column] == entity].copy()
        
        if entity_data.empty:
            raise ValueError(f"No data found for entity {entity}")
        
        # Sort by time window
        entity_data = entity_data.sort_values('time_window')
        
        # Get feature columns (exclude entity and time window columns)
        feature_columns = [col for col in entity_data.columns 
                          if col != self.entity_column and col != 'time_window']
        
        # Select a subset of features for visualization
        if len(feature_columns) > 10:
            viz_features = feature_columns[:10]
        else:
            viz_features = feature_columns
        
        # Create time series plots
        plt.figure(figsize=(12, 8))
        
        for i, feature in enumerate(viz_features):
            plt.subplot(len(viz_features), 1, i+1)
            plt.plot(entity_data['time_window'], entity_data[feature])
            plt.title(feature)
            plt.tight_layout()
        
        # Save or show the plot
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(os.path.join(output_dir, f'entity_{entity}_time_series.png'))
        else:
            plt.show()
        
        return entity_data
    
    def compare_entities(self, entity_features, entities, output_dir=None):
        """
        Compare the behavior of multiple entities.
        
        Args:
            entity_features (pd.DataFrame): Dataframe with entity features
            entities (list): List of entities to compare
            output_dir (str): Directory to save visualizations
            
        Returns:
            dict: Dictionary with comparison statistics
        """
        print(f"Comparing behavior of {len(entities)} entities...")
        
        # Filter data for the specified entities
        entity_data = entity_features[entity_features[self.entity_column].isin(entities)].copy()
        
        if entity_data.empty:
            raise ValueError(f"No data found for the specified entities")
        
        # Get feature columns (exclude entity and time window columns)
        feature_columns = [col for col in entity_data.columns 
                          if col != self.entity_column and col != 'time_window']
        
        # Select a subset of features for visualization
        if len(feature_columns) > 5:
            viz_features = feature_columns[:5]
        else:
            viz_features = feature_columns
        
        # Create comparison plots
        plt.figure(figsize=(15, 10))
        
        for i, feature in enumerate(viz_features):
            plt.subplot(len(viz_features), 1, i+1)
            
            for entity in entities:
                entity_subset = entity_data[entity_data[self.entity_column] == entity]
                plt.plot(entity_subset['time_window'], entity_subset[feature], label=entity)
            
            plt.title(feature)
            plt.legend()
            plt.tight_layout()
        
        # Save or show the plot
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(os.path.join(output_dir, 'entity_comparison.png'))
        else:
            plt.show()
        
        # Calculate comparison statistics
        comparison_stats = {}
        
        for entity in entities:
            entity_subset = entity_data[entity_data[self.entity_column] == entity]
            
            stats = {}
            for feature in feature_columns:
                stats[feature] = {
                    'mean': entity_subset[feature].mean(),
                    'std': entity_subset[feature].std(),
                    'min': entity_subset[feature].min(),
                    'max': entity_subset[feature].max()
                }
            
            comparison_stats[entity] = stats
        
        return comparison_stats
    
    def save_profiles(self, output_path):
        """
        Save entity profiles to disk.
        
        Args:
            output_path (str): Path to save the profiles
            
        Returns:
            None
        """
        print(f"Saving entity profiles to {output_path}...")
        
        if not self.entity_profiles:
            raise ValueError("Entity profiles not built. Call build_entity_profiles first.")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Convert profiles to a dataframe
        profile_data = []
        
        for entity, profile in self.entity_profiles.items():
            row = {'entity': entity}
            row.update(profile)
            profile_data.append(row)
        
        profile_df = pd.DataFrame(profile_data)
        
        # Save to CSV
        profile_df.to_csv(output_path, index=False)
        
        print(f"Saved profiles for {len(self.entity_profiles)} entities")
    
    def load_profiles(self, input_path):
        """
        Load entity profiles from disk.
        
        Args:
            input_path (str): Path to load the profiles from
            
        Returns:
            dict: Dictionary of entity profiles
        """
        print(f"Loading entity profiles from {input_path}...")
        
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"File not found: {input_path}")
        
        # Load from CSV
        profile_df = pd.read_csv(input_path)
        
        # Convert to dictionary
        entity_profiles = {}
        
        for _, row in profile_df.iterrows():
            entity = row['entity']
            profile = row.drop('entity').to_dict()
            entity_profiles[entity] = profile
        
        self.entity_profiles = entity_profiles
        
        # Build global profile
        global_profile = {}
        
        for feature in profile_df.columns:
            if feature == 'entity':
                continue
            
            values = profile_df[feature].dropna()
            
            if len(values) == 0:
                continue
            
            global_profile[feature] = {
                'mean': values.mean(),
                'std': values.std(),
                'min': values.min(),
                'max': values.max()
            }
        
        self.global_profile = global_profile
        
        print(f"Loaded profiles for {len(entity_profiles)} entities")
        
        return entity_profiles


if __name__ == "__main__":
    # Example usage
    print("Behavioral profiling module created successfully") 