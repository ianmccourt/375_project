#!/usr/bin/env python3
"""
Demo script to run the UBA system with a synthetic dataset.
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from src.data.data_processor import NetworkDataProcessor
from src.models.anomaly_detector import AnomalyDetector, EnsembleAnomalyDetector
from src.utils.behavioral_profiling import EntityProfiler
from src.config import config


def parse_arguments():
    """
    Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description='Run the UBA system with a synthetic dataset')
    
    parser.add_argument('--generate_data', action='store_true',
                        help='Generate a synthetic dataset')
    
    parser.add_argument('--data_path', type=str, default=None,
                        help='Path to the dataset (if not generating)')
    
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Directory to save results')
    
    parser.add_argument('--model_type', type=str, default='ensemble',
                        choices=['isolation_forest', 'one_class_svm', 'local_outlier_factor', 'ensemble'],
                        help='Type of anomaly detection model')
    
    parser.add_argument('--enable_profiling', action='store_true',
                        help='Enable behavioral profiling')
    
    return parser.parse_args()


def generate_synthetic_dataset():
    """
    Generate a synthetic network traffic dataset for testing.
    
    Returns:
        str: Path to the generated dataset
    """
    print("Generating synthetic network traffic dataset...")
    
    # Import the dataset generator
    from src.data.download_dataset import generate_synthetic_dataset
    
    # Generate the dataset
    dataset_path = generate_synthetic_dataset()
    
    return dataset_path


def setup_output_directory(args):
    """
    Set up the output directory for saving results.
    
    Args:
        args (argparse.Namespace): Command line arguments
        
    Returns:
        str: Path to the output directory
    """
    # Create a timestamped directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(args.output_dir, f"demo_{args.model_type}_{timestamp}")
    
    # Create directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'plots'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'models'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'profiles'), exist_ok=True)
    
    return output_dir


def main():
    """
    Main function.
    """
    # Parse command line arguments
    args = parse_arguments()
    
    # Set up output directory
    output_dir = setup_output_directory(args)
    
    # Get dataset path
    if args.generate_data:
        dataset_path = generate_synthetic_dataset()
    elif args.data_path:
        dataset_path = args.data_path
    else:
        # Check if synthetic dataset exists
        synthetic_path = os.path.join(config.DATA_DIR, 'synthetic_network_traffic.csv')
        if os.path.exists(synthetic_path):
            dataset_path = synthetic_path
        else:
            print("No dataset specified and no synthetic dataset found. Generating one...")
            dataset_path = generate_synthetic_dataset()
    
    print(f"Using dataset: {dataset_path}")
    
    # Initialize data processor
    processor = NetworkDataProcessor(scaling_method='standard', handle_imbalance=True)
    
    # Load data
    df = processor.load_data(dataset_path, target_column='Label')
    
    # Preprocess data
    X_train, X_val, X_test, y_train, y_val, y_test = processor.preprocess_data(
        df, 
        categorical_encoding='onehot',
        handle_imbalance_method='smote'
    )
    
    # Train model
    print(f"\nTraining {args.model_type} model...")
    
    if args.model_type == 'ensemble':
        # Create an ensemble of models
        model = EnsembleAnomalyDetector(voting='soft')
        
        # Add individual models to the ensemble
        model.add_model(AnomalyDetector(model_type='isolation_forest', contamination=0.05).fit(X_train))
        model.add_model(AnomalyDetector(model_type='one_class_svm').fit(X_train))
        model.add_model(AnomalyDetector(model_type='local_outlier_factor', contamination=0.05).fit(X_train))
    else:
        # Create and fit a single model
        model = AnomalyDetector(model_type=args.model_type, contamination=0.05)
        model.fit(X_train)
    
    # Optimize threshold
    print("\nOptimizing detection threshold...")
    model.find_optimal_threshold(X_val, y_val, metric='f1', output_dir=os.path.join(output_dir, 'plots'))
    
    # Evaluate model
    print("\nEvaluating model...")
    metrics = model.evaluate(X_test, y_test, output_dir=os.path.join(output_dir, 'plots'))
    
    # Print metrics
    print("\nEvaluation metrics:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Perform behavioral profiling if enabled
    if args.enable_profiling:
        print("\nPerforming behavioral profiling...")
        
        # Initialize entity profiler
        profiler = EntityProfiler(entity_column='Source IP', time_window=300)
        
        # Create entity features
        timestamp_column = 'Timestamp' if 'Timestamp' in df.columns else None
        
        if timestamp_column:
            # Create entity features
            entity_features = profiler.create_entity_features(df, timestamp_column)
            
            # Build entity profiles
            entity_profiles = profiler.build_entity_profiles(entity_features)
            
            # Detect anomalous entities
            anomalous_entities = profiler.detect_anomalous_entities(threshold=2.0)
            
            # Print anomalous entities
            print(f"\nDetected {len(anomalous_entities)} anomalous entities")
            
            if anomalous_entities:
                # Create a dataframe with anomalous entities
                anomalies_df = pd.DataFrame([
                    {
                        'entity': entity,
                        'anomaly_score': data['anomaly_score'],
                        'num_anomalous_features': data['num_anomalous_features'],
                        'total_features': data['total_features']
                    }
                    for entity, data in anomalous_entities.items()
                ])
                
                # Save anomalous entities to file
                anomalies_df.to_csv(os.path.join(output_dir, 'profiles', 'anomalous_entities.csv'), index=False)
                
                # Print top 5 anomalous entities
                print("\nTop 5 anomalous entities:")
                for i, (entity, data) in enumerate(list(anomalous_entities.items())[:5]):
                    print(f"  {i+1}. {entity}: score={data['anomaly_score']:.4f}, anomalous features={data['num_anomalous_features']}")
            
            # Cluster entities
            clustered_entities = profiler.cluster_entities(n_clusters=5, method='kmeans')
            
            # Visualize entity clusters
            profiler.visualize_entity_clusters(clustered_entities, output_dir=os.path.join(output_dir, 'profiles'))
            
            # Save entity profiles
            profiler.save_profiles(os.path.join(output_dir, 'profiles', 'entity_profiles.csv'))
        else:
            print("No timestamp column found in the dataset. Skipping behavioral profiling.")
    
    # Save model
    model_path = os.path.join(output_dir, 'models', f"{args.model_type}.joblib")
    model.save_model(model_path)
    print(f"\nModel saved to {model_path}")
    
    # Save summary report
    with open(os.path.join(output_dir, 'summary.txt'), 'w') as f:
        f.write("=== UBA Anomaly Detection System Demo ===\n\n")
        f.write(f"Model: {args.model_type}\n")
        f.write(f"Dataset: {dataset_path}\n\n")
        f.write("=== Evaluation Metrics ===\n")
        for metric, value in metrics.items():
            f.write(f"{metric}: {value:.4f}\n")
    
    print(f"\nResults saved to {output_dir}")
    print("\nDemo completed successfully!")


if __name__ == "__main__":
    main() 