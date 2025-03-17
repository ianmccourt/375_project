"""
Main script for running the UBA anomaly detection system.
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from data.data_processor import NetworkDataProcessor
from models.anomaly_detector import AnomalyDetector, EnsembleAnomalyDetector
from utils.behavioral_profiling import EntityProfiler
from config import config


def parse_arguments():
    """
    Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description='UBA Anomaly Detection System')
    
    # Data arguments
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to the network traffic data file')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Directory to save results')
    
    # Model arguments
    parser.add_argument('--model_type', type=str, default='ensemble',
                        choices=['isolation_forest', 'one_class_svm', 'local_outlier_factor', 'ensemble'],
                        help='Type of anomaly detection model')
    parser.add_argument('--contamination', type=float, default=0.01,
                        help='Expected proportion of anomalies in the data')
    
    # Preprocessing arguments
    parser.add_argument('--scaling', type=str, default='standard',
                        choices=['standard', 'minmax', 'none'],
                        help='Method for scaling numerical features')
    parser.add_argument('--categorical_encoding', type=str, default='onehot',
                        choices=['onehot', 'label', 'target', 'binary'],
                        help='Method for encoding categorical features')
    parser.add_argument('--handle_imbalance', action='store_true',
                        help='Whether to handle class imbalance')
    parser.add_argument('--imbalance_method', type=str, default='smote',
                        choices=['smote', 'undersample', 'both'],
                        help='Method for handling class imbalance')
    
    # Behavioral profiling arguments
    parser.add_argument('--enable_profiling', action='store_true',
                        help='Whether to enable behavioral profiling')
    parser.add_argument('--entity_column', type=str, default='Source IP',
                        help='Column name for the entity identifier')
    parser.add_argument('--time_window', type=int, default=300,
                        help='Time window size in seconds for aggregating behavior')
    
    # Evaluation arguments
    parser.add_argument('--optimize_threshold', action='store_true',
                        help='Whether to optimize the anomaly detection threshold')
    parser.add_argument('--threshold_metric', type=str, default='f1',
                        choices=['f1', 'precision', 'recall', 'false_positive_rate'],
                        help='Metric to optimize when finding the optimal threshold')
    
    # Execution arguments
    parser.add_argument('--save_model', action='store_true',
                        help='Whether to save the trained model')
    parser.add_argument('--load_model', type=str, default=None,
                        help='Path to a saved model to load')
    
    return parser.parse_args()


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
    output_dir = os.path.join(args.output_dir, f"{args.model_type}_{timestamp}")
    
    # Create directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'plots'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'models'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'profiles'), exist_ok=True)
    
    return output_dir


def load_and_preprocess_data(args):
    """
    Load and preprocess the network traffic data.
    
    Args:
        args (argparse.Namespace): Command line arguments
        
    Returns:
        tuple: Preprocessed data splits and data processor
    """
    print("\n=== Loading and Preprocessing Data ===")
    
    # Initialize data processor
    scaling_method = None if args.scaling == 'none' else args.scaling
    processor = NetworkDataProcessor(scaling_method=scaling_method, handle_imbalance=args.handle_imbalance)
    
    # Load data
    df = processor.load_data(args.data_path)
    
    # Preprocess data
    X_train, X_val, X_test, y_train, y_val, y_test = processor.preprocess_data(
        df, 
        categorical_encoding=args.categorical_encoding,
        handle_imbalance_method=args.imbalance_method
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test, processor


def train_model(args, X_train, X_val, y_val):
    """
    Train the anomaly detection model.
    
    Args:
        args (argparse.Namespace): Command line arguments
        X_train (pd.DataFrame): Training data
        X_val (pd.DataFrame): Validation data
        y_val (pd.Series): Validation labels
        
    Returns:
        object: Trained model
    """
    print("\n=== Training Model ===")
    
    if args.load_model:
        print(f"Loading model from {args.load_model}")
        if args.model_type == 'ensemble':
            model = EnsembleAnomalyDetector.load_model(args.load_model)
        else:
            model = AnomalyDetector.load_model(args.load_model)
        return model
    
    if args.model_type == 'ensemble':
        # Create an ensemble of models
        model = EnsembleAnomalyDetector(voting='soft')
        
        # Add individual models to the ensemble
        model.add_model(AnomalyDetector(model_type='isolation_forest', contamination=args.contamination))
        model.add_model(AnomalyDetector(model_type='one_class_svm'))
        model.add_model(AnomalyDetector(model_type='local_outlier_factor', contamination=args.contamination))
        
        # Fit the ensemble
        model.fit(X_train, contamination=args.contamination)
    else:
        # Create and fit a single model
        model = AnomalyDetector(model_type=args.model_type, contamination=args.contamination)
        model.fit(X_train, contamination=args.contamination)
    
    # Optimize threshold if requested
    if args.optimize_threshold:
        print("\n=== Optimizing Detection Threshold ===")
        model.find_optimal_threshold(X_val, y_val, metric=args.threshold_metric)
    
    return model


def evaluate_model(model, X_test, y_test, output_dir):
    """
    Evaluate the model on test data.
    
    Args:
        model (object): Trained model
        X_test (pd.DataFrame): Test data
        y_test (pd.Series): Test labels
        output_dir (str): Directory to save evaluation results
        
    Returns:
        dict: Evaluation metrics
    """
    print("\n=== Evaluating Model ===")
    
    # Create plots directory
    plots_dir = os.path.join(output_dir, 'plots')
    
    # Evaluate the model
    metrics = model.evaluate(X_test, y_test, output_dir=plots_dir)
    
    # Save metrics to file
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(os.path.join(output_dir, 'metrics.csv'), index=False)
    
    return metrics


def perform_behavioral_profiling(args, df, output_dir):
    """
    Perform behavioral profiling on network entities.
    
    Args:
        args (argparse.Namespace): Command line arguments
        df (pd.DataFrame): Input dataframe
        output_dir (str): Directory to save profiling results
        
    Returns:
        EntityProfiler: Trained entity profiler
    """
    print("\n=== Performing Behavioral Profiling ===")
    
    # Initialize entity profiler
    profiler = EntityProfiler(entity_column=args.entity_column, time_window=args.time_window)
    
    # Create entity features
    timestamp_column = next((col for col in df.columns if 'time' in col.lower()), None)
    if not timestamp_column:
        print("Warning: No timestamp column found. Skipping behavioral profiling.")
        return None
    
    entity_features = profiler.create_entity_features(df, timestamp_column)
    
    # Build entity profiles
    entity_profiles = profiler.build_entity_profiles(entity_features)
    
    # Detect anomalous entities
    anomalous_entities = profiler.detect_anomalous_entities(threshold=2.0)
    
    # Save anomalous entities to file
    anomalies_df = pd.DataFrame([
        {
            'entity': entity,
            'anomaly_score': data['anomaly_score'],
            'num_anomalous_features': data['num_anomalous_features'],
            'total_features': data['total_features']
        }
        for entity, data in anomalous_entities.items()
    ])
    
    if not anomalies_df.empty:
        anomalies_df.to_csv(os.path.join(output_dir, 'profiles', 'anomalous_entities.csv'), index=False)
    
    # Cluster entities
    clustered_entities = profiler.cluster_entities(n_clusters=5, method='kmeans')
    
    # Visualize entity clusters
    profiler.visualize_entity_clusters(clustered_entities, output_dir=os.path.join(output_dir, 'profiles'))
    
    # Save entity profiles
    profiler.save_profiles(os.path.join(output_dir, 'profiles', 'entity_profiles.csv'))
    
    return profiler


def save_results(args, model, metrics, output_dir):
    """
    Save model and results.
    
    Args:
        args (argparse.Namespace): Command line arguments
        model (object): Trained model
        metrics (dict): Evaluation metrics
        output_dir (str): Directory to save results
    """
    print("\n=== Saving Results ===")
    
    # Save model if requested
    if args.save_model:
        model_path = os.path.join(output_dir, 'models', f"{args.model_type}.joblib")
        model.save_model(model_path)
        print(f"Model saved to {model_path}")
    
    # Save command line arguments
    with open(os.path.join(output_dir, 'args.txt'), 'w') as f:
        for arg, value in vars(args).items():
            f.write(f"{arg}: {value}\n")
    
    # Save summary report
    with open(os.path.join(output_dir, 'summary.txt'), 'w') as f:
        f.write("=== UBA Anomaly Detection System ===\n\n")
        f.write(f"Model: {args.model_type}\n")
        f.write(f"Data: {args.data_path}\n\n")
        f.write("=== Evaluation Metrics ===\n")
        for metric, value in metrics.items():
            f.write(f"{metric}: {value:.4f}\n")
    
    print(f"Results saved to {output_dir}")


def main():
    """
    Main function.
    """
    # Parse command line arguments
    args = parse_arguments()
    
    # Set up output directory
    output_dir = setup_output_directory(args)
    
    # Load and preprocess data
    X_train, X_val, X_test, y_train, y_val, y_test, processor = load_and_preprocess_data(args)
    
    # Train model
    model = train_model(args, X_train, X_val, y_val)
    
    # Evaluate model
    metrics = evaluate_model(model, X_test, y_test, output_dir)
    
    # Perform behavioral profiling if enabled
    if args.enable_profiling:
        # Load the original data again for profiling
        df = processor.load_data(args.data_path)
        profiler = perform_behavioral_profiling(args, df, output_dir)
    
    # Save results
    save_results(args, model, metrics, output_dir)
    
    print("\n=== Done ===")


if __name__ == "__main__":
    main() 