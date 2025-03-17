"""
Baseline anomaly detection models for network traffic analysis.
"""

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config


class AnomalyDetector:
    """
    Base class for anomaly detection models.
    """
    
    def __init__(self, model_type='isolation_forest', **kwargs):
        """
        Initialize the anomaly detector.
        
        Args:
            model_type (str): Type of anomaly detection model
            **kwargs: Additional parameters for the model
        """
        self.model_type = model_type
        self.model = None
        self.threshold = None
        self.model_params = kwargs if kwargs else config.MODEL_PARAMS.get(model_type, {})
        
        # Initialize the model
        self._initialize_model()
    
    def _initialize_model(self):
        """
        Initialize the anomaly detection model based on the specified type.
        """
        if self.model_type == 'isolation_forest':
            self.model = IsolationForest(**self.model_params)
        elif self.model_type == 'one_class_svm':
            self.model = OneClassSVM(**self.model_params)
        elif self.model_type == 'local_outlier_factor':
            self.model = LocalOutlierFactor(**self.model_params, novelty=True)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def fit(self, X_train, contamination=None):
        """
        Fit the anomaly detection model.
        
        Args:
            X_train (pd.DataFrame): Training data
            contamination (float): Contamination parameter for the model
            
        Returns:
            self: Fitted model
        """
        print(f"Fitting {self.model_type} model...")
        
        # Update contamination if provided
        if contamination is not None and hasattr(self.model, 'contamination'):
            self.model.contamination = contamination
        
        # Fit the model
        self.model.fit(X_train)
        
        return self
    
    def predict(self, X, threshold=None):
        """
        Predict anomalies in the data.
        
        Args:
            X (pd.DataFrame): Data to predict
            threshold (float): Custom threshold for anomaly detection
            
        Returns:
            np.ndarray: Binary predictions (1 for normal, -1 for anomaly)
        """
        # Get anomaly scores
        scores = self.decision_function(X)
        
        # Use custom threshold if provided, otherwise use the model's default
        if threshold is not None:
            self.threshold = threshold
            predictions = np.where(scores < threshold, -1, 1)
        else:
            predictions = self.model.predict(X)
        
        return predictions
    
    def decision_function(self, X):
        """
        Get anomaly scores for the data.
        
        Args:
            X (pd.DataFrame): Data to score
            
        Returns:
            np.ndarray: Anomaly scores
        """
        if self.model_type == 'local_outlier_factor':
            return -self.model.decision_function(X)  # Negate to make higher values more anomalous
        else:
            return -self.model.decision_function(X)  # Negate to make higher values more anomalous
    
    def evaluate(self, X_test, y_test, threshold=None, output_dir=None):
        """
        Evaluate the model on test data.
        
        Args:
            X_test (pd.DataFrame): Test data
            y_test (pd.Series): True labels (1 for normal, 0 for anomaly)
            threshold (float): Custom threshold for anomaly detection
            output_dir (str): Directory to save evaluation results
            
        Returns:
            dict: Evaluation metrics
        """
        print(f"Evaluating {self.model_type} model...")
        
        # Convert labels to match model output (1 for normal, -1 for anomaly)
        y_test_converted = np.where(y_test == 0, -1, 1)
        
        # Get anomaly scores
        scores = self.decision_function(X_test)
        
        # Predict using the specified threshold
        y_pred = self.predict(X_test, threshold=threshold)
        
        # Calculate metrics
        cm = confusion_matrix(y_test_converted, y_pred)
        report = classification_report(y_test_converted, y_pred, output_dict=True)
        
        # Calculate precision-recall curve
        precision, recall, pr_thresholds = precision_recall_curve(
            y_test_converted == -1, -scores
        )
        
        # Calculate ROC curve
        fpr, tpr, roc_thresholds = roc_curve(
            y_test_converted == -1, -scores
        )
        roc_auc = auc(fpr, tpr)
        
        # Calculate false positive rate
        tn, fp, fn, tp = cm.ravel()
        false_positive_rate = fp / (fp + tn)
        
        # Compile metrics
        metrics = {
            'accuracy': report['accuracy'],
            'precision': report['-1']['precision'],  # Precision for anomaly class
            'recall': report['-1']['recall'],  # Recall for anomaly class
            'f1': report['-1']['f1-score'],  # F1 for anomaly class
            'false_positive_rate': false_positive_rate,
            'roc_auc': roc_auc
        }
        
        print(f"Evaluation metrics: {metrics}")
        
        # Plot and save evaluation results if output directory is provided
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
            # Plot confusion matrix
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=['Normal', 'Anomaly'], 
                        yticklabels=['Normal', 'Anomaly'])
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title(f'Confusion Matrix - {self.model_type}')
            plt.savefig(os.path.join(output_dir, f'{self.model_type}_confusion_matrix.png'))
            
            # Plot ROC curve
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve - {self.model_type}')
            plt.legend(loc='lower right')
            plt.savefig(os.path.join(output_dir, f'{self.model_type}_roc_curve.png'))
            
            # Plot precision-recall curve
            plt.figure(figsize=(8, 6))
            plt.plot(recall, precision, color='blue', lw=2)
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title(f'Precision-Recall Curve - {self.model_type}')
            plt.savefig(os.path.join(output_dir, f'{self.model_type}_precision_recall_curve.png'))
            
            # Plot anomaly score distribution
            plt.figure(figsize=(10, 6))
            sns.histplot(scores[y_test_converted == 1], color='green', label='Normal', alpha=0.5, bins=50)
            sns.histplot(scores[y_test_converted == -1], color='red', label='Anomaly', alpha=0.5, bins=50)
            plt.axvline(x=threshold if threshold is not None else 0, color='black', linestyle='--', 
                        label=f'Threshold: {threshold if threshold is not None else "Default"}')
            plt.xlabel('Anomaly Score')
            plt.ylabel('Count')
            plt.title(f'Anomaly Score Distribution - {self.model_type}')
            plt.legend()
            plt.savefig(os.path.join(output_dir, f'{self.model_type}_score_distribution.png'))
        
        return metrics
    
    def find_optimal_threshold(self, X_val, y_val, metric='f1', output_dir=None):
        """
        Find the optimal threshold for anomaly detection.
        
        Args:
            X_val (pd.DataFrame): Validation data
            y_val (pd.Series): True labels (1 for normal, 0 for anomaly)
            metric (str): Metric to optimize ('f1', 'precision', 'recall', 'false_positive_rate')
            output_dir (str): Directory to save threshold analysis results
            
        Returns:
            float: Optimal threshold
        """
        print(f"Finding optimal threshold for {self.model_type} model...")
        
        # Convert labels to match model output (1 for normal, -1 for anomaly)
        y_val_converted = np.where(y_val == 0, -1, 1)
        
        # Get anomaly scores
        scores = self.decision_function(X_val)
        
        # Define threshold range
        thresholds = np.linspace(np.min(scores), np.max(scores), 100)
        
        # Initialize metrics
        metrics_list = []
        
        # Calculate metrics for each threshold
        for threshold in thresholds:
            y_pred = np.where(scores < threshold, -1, 1)
            cm = confusion_matrix(y_val_converted, y_pred)
            tn, fp, fn, tp = cm.ravel()
            
            # Calculate metrics
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
            
            metrics_list.append({
                'threshold': threshold,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'false_positive_rate': false_positive_rate
            })
        
        # Convert to dataframe
        metrics_df = pd.DataFrame(metrics_list)
        
        # Find optimal threshold based on the specified metric
        if metric == 'f1':
            optimal_idx = metrics_df['f1'].idxmax()
        elif metric == 'precision':
            optimal_idx = metrics_df['precision'].idxmax()
        elif metric == 'recall':
            optimal_idx = metrics_df['recall'].idxmax()
        elif metric == 'false_positive_rate':
            # Find the threshold with the lowest false positive rate that still has a recall above 0.5
            valid_metrics = metrics_df[metrics_df['recall'] >= 0.5]
            optimal_idx = valid_metrics['false_positive_rate'].idxmin() if not valid_metrics.empty else metrics_df['f1'].idxmax()
        else:
            raise ValueError(f"Unsupported metric: {metric}")
        
        optimal_threshold = metrics_df.loc[optimal_idx, 'threshold']
        optimal_metrics = metrics_df.loc[optimal_idx]
        
        print(f"Optimal threshold: {optimal_threshold}")
        print(f"Metrics at optimal threshold: {optimal_metrics.to_dict()}")
        
        # Plot and save threshold analysis results if output directory is provided
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
            # Plot metrics vs threshold
            plt.figure(figsize=(10, 6))
            plt.plot(metrics_df['threshold'], metrics_df['precision'], label='Precision')
            plt.plot(metrics_df['threshold'], metrics_df['recall'], label='Recall')
            plt.plot(metrics_df['threshold'], metrics_df['f1'], label='F1')
            plt.plot(metrics_df['threshold'], metrics_df['false_positive_rate'], label='False Positive Rate')
            plt.axvline(x=optimal_threshold, color='black', linestyle='--', 
                        label=f'Optimal Threshold: {optimal_threshold:.4f}')
            plt.xlabel('Threshold')
            plt.ylabel('Metric Value')
            plt.title(f'Metrics vs Threshold - {self.model_type}')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, f'{self.model_type}_threshold_analysis.png'))
        
        # Set the optimal threshold
        self.threshold = optimal_threshold
        
        return optimal_threshold
    
    def save_model(self, model_path):
        """
        Save the model to disk.
        
        Args:
            model_path (str): Path to save the model
        """
        print(f"Saving {self.model_type} model to {model_path}...")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Save model and metadata
        joblib.dump({
            'model': self.model,
            'model_type': self.model_type,
            'threshold': self.threshold,
            'model_params': self.model_params
        }, model_path)
    
    @classmethod
    def load_model(cls, model_path):
        """
        Load a model from disk.
        
        Args:
            model_path (str): Path to the saved model
            
        Returns:
            AnomalyDetector: Loaded model
        """
        print(f"Loading model from {model_path}...")
        
        # Load model and metadata
        model_data = joblib.load(model_path)
        
        # Create a new instance
        detector = cls(model_type=model_data['model_type'], **model_data['model_params'])
        
        # Set model and threshold
        detector.model = model_data['model']
        detector.threshold = model_data['threshold']
        
        return detector


class EnsembleAnomalyDetector:
    """
    Ensemble of multiple anomaly detection models.
    """
    
    def __init__(self, models=None, voting='hard'):
        """
        Initialize the ensemble anomaly detector.
        
        Args:
            models (list): List of AnomalyDetector instances
            voting (str): Voting method ('hard' or 'soft')
        """
        self.models = models if models else []
        self.voting = voting
    
    def add_model(self, model):
        """
        Add a model to the ensemble.
        
        Args:
            model (AnomalyDetector): Model to add
        """
        self.models.append(model)
    
    def fit(self, X_train, contamination=None):
        """
        Fit all models in the ensemble.
        
        Args:
            X_train (pd.DataFrame): Training data
            contamination (float): Contamination parameter for the models
            
        Returns:
            self: Fitted ensemble
        """
        print("Fitting ensemble of anomaly detection models...")
        
        for model in self.models:
            model.fit(X_train, contamination=contamination)
        
        return self
    
    def predict(self, X):
        """
        Predict anomalies using the ensemble.
        
        Args:
            X (pd.DataFrame): Data to predict
            
        Returns:
            np.ndarray: Binary predictions (1 for normal, -1 for anomaly)
        """
        if not self.models:
            raise ValueError("No models in the ensemble")
        
        if self.voting == 'hard':
            # Hard voting: majority vote of binary predictions
            predictions = np.array([model.predict(X) for model in self.models])
            ensemble_pred = np.sign(np.sum(predictions, axis=0))
            
            # Handle zeros (ties) by classifying as anomalies
            ensemble_pred[ensemble_pred == 0] = -1
            
        elif self.voting == 'soft':
            # Soft voting: average of anomaly scores
            scores = np.array([model.decision_function(X) for model in self.models])
            ensemble_scores = np.mean(scores, axis=0)
            
            # Use a threshold of 0 for the averaged scores
            ensemble_pred = np.where(ensemble_scores > 0, 1, -1)
            
        else:
            raise ValueError(f"Unsupported voting method: {self.voting}")
        
        return ensemble_pred
    
    def decision_function(self, X):
        """
        Get ensemble anomaly scores for the data.
        
        Args:
            X (pd.DataFrame): Data to score
            
        Returns:
            np.ndarray: Ensemble anomaly scores
        """
        if not self.models:
            raise ValueError("No models in the ensemble")
        
        # Average the anomaly scores from all models
        scores = np.array([model.decision_function(X) for model in self.models])
        ensemble_scores = np.mean(scores, axis=0)
        
        return ensemble_scores
    
    def evaluate(self, X_test, y_test, threshold=None, output_dir=None):
        """
        Evaluate the ensemble on test data.
        
        Args:
            X_test (pd.DataFrame): Test data
            y_test (pd.Series): True labels (1 for normal, 0 for anomaly)
            threshold (float): Custom threshold for anomaly detection
            output_dir (str): Directory to save evaluation results
            
        Returns:
            dict: Evaluation metrics
        """
        print("Evaluating ensemble model...")
        
        # Create a temporary AnomalyDetector to use its evaluation method
        temp_detector = AnomalyDetector(model_type='ensemble')
        temp_detector.model_type = 'ensemble'
        temp_detector.decision_function = lambda X: self.decision_function(X)
        temp_detector.predict = lambda X, threshold=None: self.predict(X) if threshold is None else np.where(self.decision_function(X) < threshold, -1, 1)
        
        return temp_detector.evaluate(X_test, y_test, threshold=threshold, output_dir=output_dir)
    
    def find_optimal_threshold(self, X_val, y_val, metric='f1', output_dir=None):
        """
        Find the optimal threshold for the ensemble.
        
        Args:
            X_val (pd.DataFrame): Validation data
            y_val (pd.Series): True labels (1 for normal, 0 for anomaly)
            metric (str): Metric to optimize ('f1', 'precision', 'recall', 'false_positive_rate')
            output_dir (str): Directory to save threshold analysis results
            
        Returns:
            float: Optimal threshold
        """
        print("Finding optimal threshold for ensemble model...")
        
        # Create a temporary AnomalyDetector to use its threshold finding method
        temp_detector = AnomalyDetector(model_type='ensemble')
        temp_detector.model_type = 'ensemble'
        temp_detector.decision_function = lambda X: self.decision_function(X)
        
        return temp_detector.find_optimal_threshold(X_val, y_val, metric=metric, output_dir=output_dir)
    
    def save_model(self, model_path):
        """
        Save the ensemble to disk.
        
        Args:
            model_path (str): Path to save the ensemble
        """
        print(f"Saving ensemble model to {model_path}...")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Save individual models
        model_paths = []
        for i, model in enumerate(self.models):
            model_dir = os.path.join(os.path.dirname(model_path), f"model_{i}")
            os.makedirs(model_dir, exist_ok=True)
            model_file = os.path.join(model_dir, f"{model.model_type}.joblib")
            model.save_model(model_file)
            model_paths.append(model_file)
        
        # Save ensemble metadata
        joblib.dump({
            'model_paths': model_paths,
            'voting': self.voting
        }, model_path)
    
    @classmethod
    def load_model(cls, model_path):
        """
        Load an ensemble from disk.
        
        Args:
            model_path (str): Path to the saved ensemble
            
        Returns:
            EnsembleAnomalyDetector: Loaded ensemble
        """
        print(f"Loading ensemble model from {model_path}...")
        
        # Load ensemble metadata
        ensemble_data = joblib.load(model_path)
        
        # Load individual models
        models = []
        for model_file in ensemble_data['model_paths']:
            model = AnomalyDetector.load_model(model_file)
            models.append(model)
        
        # Create a new ensemble
        ensemble = cls(models=models, voting=ensemble_data['voting'])
        
        return ensemble


if __name__ == "__main__":
    # Example usage
    print("Anomaly detection models module created successfully") 