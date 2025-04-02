"""
TII-SSRC-23 Dataset Analysis with UBA System - Colab Version

This script analyzes multiple CICIDS2018 dataset files to detect network anomalies.
It processes all CSV files in the specified directory, evaluates multiple models,
and generates combined performance metrics.
"""

# Setup Environment
# =========================
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from google.colab import drive
import gc
import psutil
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix, 
                           roc_auc_score, precision_recall_curve, f1_score, precision_score, recall_score)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
import glob
import warnings
import json
import traceback
warnings.filterwarnings('ignore')

# Mount Google Drive
drive.mount('/content/drive')

# Clone repository if needed - Fixed Jupyter magic commands
if not os.path.exists('375_project'):
    # Use subprocess instead of Jupyter magic
    import subprocess
    subprocess.run(["git", "clone", "https://github.com/ianmccourt/375_project.git"])
    os.chdir('375_project')
    subprocess.run(["pip", "install", "-r", "requirements.txt"])

# Add project directory to Python path
if '/content/375_project' not in sys.path:
    sys.path.append('/content/375_project')

# Try to import project modules but continue if they fail
try:
    from src.data.data_processor import NetworkDataProcessor
    from src.models.anomaly_detector import AnomalyDetector
    from src.utils.behavioral_profiling import EntityProfiler
    print("✅ Imports successful!")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Continuing without custom modules...")

# Update essential columns based on the CICIDS2018 dataset structure
essential_columns = [
    'Flow ID', 'Src IP', 'Src Port', 'Dst IP', 'Dst Port', 'Protocol',
    'Timestamp', 'Flow Duration', 'Tot Fwd Pkts', 'Tot Bwd Pkts',
    'TotLen Fwd Pkts', 'TotLen Bwd Pkts', 'Fwd Pkt Len Max', 'Fwd Pkt Len Min',
    'Fwd Pkt Len Mean', 'Fwd Pkt Len Std', 'Bwd Pkt Len Max', 'Bwd Pkt Len Min',
    'Bwd Pkt Len Mean', 'Bwd Pkt Len Std', 'Flow Byts/s', 'Flow Pkts/s',
    'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min',
    'Label'  # The target column
]

# Define helper functions
def load_dataset(file_path, columns=None, sample_size=100000):
    """Load dataset with better class balance"""
    print(f"\nLoading data from {file_path}")
    
    # Check file size
    file_size_gb = os.path.getsize(file_path) / (1024**3)
    print(f"File size: {file_size_gb:.2f} GB")
    
    # First, load a small sample to understand column structure and labels
    small_sample = pd.read_csv(file_path, nrows=10000)
    
    # Check available columns
    available_columns = small_sample.columns.tolist()
    print(f"Available columns: {len(available_columns)} total")
    
    # Filter columns if specified
    if columns is not None:
        # Check which requested columns are actually in the dataset
        found_columns = [col for col in columns if col in available_columns]
        missing_columns = [col for col in columns if col not in available_columns]
        if missing_columns:
            print(f"Warning: These requested columns are not in the dataset: {missing_columns}")
        columns = found_columns
    else:
        columns = available_columns
    
    # Check for label column
    if 'Label' in columns:
        label_counts = small_sample['Label'].value_counts()
        print("Label distribution in initial sample:")
        print(label_counts.head(10))  # Show only top 10 classes to avoid cluttering output
        
        # Identify normal and attack classes
        if 'Benign' in label_counts.index:
            normal_class = 'Benign'
        elif 'BENIGN' in label_counts.index:
            normal_class = 'BENIGN'
        else:
            # If neither is found, use the minority class as normal (heuristic)
            normal_class = label_counts.idxmin()
            print(f"Warning: No explicit normal class found. Using minority class '{normal_class}' as normal.")
        
        print(f"Using '{normal_class}' as normal traffic class")
        
        # Load full sample
        print("Loading full sample...")
        df = pd.read_csv(file_path, usecols=columns, nrows=sample_size)
        
        # Check if we actually loaded data
        if len(df) == 0:
            print("Warning: No data loaded. Trying without column filtering...")
            df = pd.read_csv(file_path, nrows=sample_size)
    else:
        # If no Label column, load regular sample
        df = pd.read_csv(file_path, nrows=sample_size)
    
    print(f"Loaded {len(df)} rows with {len(df.columns)} columns")
    
    # Verify we have data
    if len(df) == 0:
        raise ValueError("Failed to load any data. Please check the file path and format.")
        
    # After loading the dataset, clean it
    try:
        # Handle problematic columns - check if any column headers are in the data
        for col in df.columns:
            if df[col].dtype == object and len(df) > 0:
                # Check first 10 rows for column headers
                first_values = df[col].head(10).astype(str).tolist()
                if any(header in first_values for header in df.columns):
                    print(f"Warning: Column {col} contains headers as values. Attempting to fix.")
                    # Find the row with headers
                    header_rows = df.index[df[col].astype(str).isin(df.columns)].tolist()
                    if header_rows:
                        # Drop these rows - they're likely duplicated headers
                        df = df.drop(header_rows)
                        print(f"Dropped {len(header_rows)} rows containing headers")
        
        # Convert numeric columns
        for col in df.columns:
            if col not in ['Label', 'Protocol']:  # Keep categorical columns as is
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                except:
                    pass  # Keep as is if cannot convert
        
        df = df.fillna(0)  # Fill NaN values after conversion
        df = df.replace([np.inf, -np.inf], 0)  # Replace infinities
        
        print("Cleaned dataset of problematic values")
    except Exception as e:
        print(f"Error during initial cleaning: {e}")
        
    return df, normal_class

def find_column(df, possible_names):
    """Find the actual column name in the dataframe"""
    for name in possible_names:
        if name in df.columns:
            return name
    return None

def analyze_dataset(dataset_path, sample_size=25000):
    """
    Analyze a single dataset file and return the results
    
    Args:
        dataset_path: Path to the CSV file
        sample_size: Number of samples to load
        
    Returns:
        metrics_df: DataFrame with model performance metrics
        label_stats: Dict with statistics about labels
    """
    print(f"\n{'='*50}")
    print(f"ANALYZING: {os.path.basename(dataset_path)}")
    print(f"{'='*50}\n")
    
    try:
        # Force garbage collection at the start
        gc.collect()
        
        # Load the dataset
        df, normal_class = load_dataset(dataset_path, essential_columns, sample_size=sample_size)
        
        # Create binary labels
        label_stats = {}
        if 'Label' in df.columns:
            df['Label_binary'] = (df['Label'] != normal_class).astype(int)
            print("\nClass distribution in loaded data:")
            class_dist = df['Label_binary'].value_counts(normalize=True)
            print(class_dist)
            
            # Store original distribution
            label_stats['original_distribution'] = class_dist.to_dict()
            label_stats['attack_types'] = df['Label'].value_counts().to_dict()
            
            # Check if we have an imbalance issue
            anomaly_percent = df['Label_binary'].mean() * 100
            print(f"Anomaly percentage: {anomaly_percent:.2f}%")
            label_stats['anomaly_percent'] = anomaly_percent
            
            # If we have extreme imbalance, balance the classes
            if df['Label_binary'].value_counts(normalize=True).min() < 0.1:
                print("Extreme class imbalance detected. Creating a more balanced dataset...")
                
                # Get minority and majority class info
                minority_class = df['Label_binary'].value_counts().idxmin()
                minority_sample_count = df['Label_binary'].value_counts()[minority_class]
                
                # Only balance if we have some minimum number of minority samples
                if minority_sample_count >= 2:
                    # Keep all samples from minority class
                    minority_samples = df[df['Label_binary'] == minority_class]
                    
                    # Sample from majority class (equal to minority count × 3, at most)
                    majority_class = 1 - minority_class
                    majority_samples = df[df['Label_binary'] == majority_class].sample(
                        min(len(minority_samples)*3, sum(df['Label_binary'] == majority_class)),
                        random_state=42
                    )
                    
                    # Combine balanced dataset
                    df = pd.concat([minority_samples, majority_samples])
                    print("After balancing:")
                    balanced_dist = df['Label_binary'].value_counts(normalize=True)
                    print(balanced_dist)
                    label_stats['balanced_distribution'] = balanced_dist.to_dict()
                else:
                    print(f"Warning: Only {minority_sample_count} samples in minority class. Insufficient for balancing.")
                    print("Creating synthetic minority samples...")
                    
                    # Get original samples
                    minority_samples = df[df['Label_binary'] == minority_class]
                    
                    # Create synthetic samples based on the few existing ones
                    synthetic_samples = []
                    num_synthetic = min(10, 5 * minority_sample_count)  # Create a reasonable number
                    
                    for i in range(num_synthetic):
                        # Select a random minority sample as base
                        base_idx = np.random.choice(minority_samples.index)
                        base_sample = df.loc[base_idx].copy()
                        
                        # Add noise to features (except Label columns)
                        for col in df.columns:
                            if col not in ['Label', 'Label_binary'] and pd.api.types.is_numeric_dtype(df[col]):
                                # Add random noise (5-10%)
                                noise_factor = 1 + np.random.uniform(-0.1, 0.1)
                                base_sample[col] = base_sample[col] * noise_factor
                        
                        synthetic_samples.append(base_sample)
                    
                    # Create DataFrame from synthetic samples
                    synthetic_df = pd.DataFrame(synthetic_samples)
                    
                    # Add synthetic samples to dataset
                    df = pd.concat([df, synthetic_df], ignore_index=True)
                    
                    print(f"Added {len(synthetic_samples)} synthetic minority samples")
                    print("After adding synthetic samples:")
                    synthetic_dist = df['Label_binary'].value_counts(normalize=True)
                    print(synthetic_dist)
                    label_stats['synthetic_distribution'] = synthetic_dist.to_dict()
                    
                    # Now do the balancing with synthetic samples included
                    minority_samples = df[df['Label_binary'] == minority_class]
                    majority_class = 1 - minority_class
                    majority_samples = df[df['Label_binary'] == majority_class].sample(
                        min(len(minority_samples)*3, sum(df['Label_binary'] == majority_class)),
                        random_state=42
                    )
                    
                    # Combine balanced dataset
                    df = pd.concat([minority_samples, majority_samples])
                    print("After balancing with synthetic samples:")
                    balanced_dist = df['Label_binary'].value_counts(normalize=True)
                    print(balanced_dist)
                    label_stats['balanced_distribution'] = balanced_dist.to_dict()
        
        # Check if we have enough data to work with
        if len(df) < 10:  # Arbitrary minimum threshold
            print(f"Warning: Dataset has only {len(df)} rows after processing.")
            print("Consider using a larger sample_size or checking the file format.")
            
        if df['Label_binary'].nunique() < 2:
            print("Error: Dataset has only one class after processing. Need both normal and attack samples.")
            return pd.DataFrame(), {'error': 'insufficient_classes', 'loaded_rows': len(df)}

        # Check minimum class size    
        min_class_count = df['Label_binary'].value_counts().min()
        if min_class_count < 5:
            print(f"Warning: Minority class has only {min_class_count} samples - may not be enough for reliable model training.")
            
            # Try to enhance minority samples if extremely rare
            if min_class_count < 3:
                minority_class = df['Label_binary'].value_counts().idxmin()
                minority_samples = df[df['Label_binary'] == minority_class]
                
                # Create synthetic samples by adding small noise to existing ones
                print("Creating synthetic minority samples...")
                synthetic_samples = []
                
                for i in range(max(5 - min_class_count, 0)):
                    # Select a random minority sample as base
                    base_sample = minority_samples.sample(1).iloc[0].copy()
                    
                    # Add small random noise to numeric features
                    for col in base_sample.index:
                        if col not in ['Label', 'Label_binary'] and pd.api.types.is_numeric_dtype(base_sample[col]):
                            # Add small percent noise
                            base_sample[col] = base_sample[col] * (1 + np.random.normal(0, 0.05))
                            
                    synthetic_samples.append(base_sample)
                    
                if synthetic_samples:
                    # Add synthetic samples to dataset
                    synthetic_df = pd.DataFrame(synthetic_samples)
                    df = pd.concat([df, synthetic_df], ignore_index=True)
                    print(f"Added {len(synthetic_samples)} synthetic samples to minority class.")
                    
                    # Update class distribution
                    print("Updated class distribution:")
                    updated_class_dist = df['Label_binary'].value_counts(normalize=True)
                    print(updated_class_dist)
        
        # Check for column name differences and adapt feature engineering
        print("\nPerforming feature engineering...")
        
        # Define column name mappings - original name to possible alternatives
        column_mappings = {
            'Tot Fwd Pkts': ['Tot Fwd Pkts', 'Total Fwd Packet', 'Total Fwd Packets'],
            'Tot Bwd Pkts': ['Tot Bwd Pkts', 'Total Bwd packets', 'Total Backward Packets'],
            'TotLen Fwd Pkts': ['TotLen Fwd Pkts', 'Total Length of Fwd Packet'],
            'TotLen Bwd Pkts': ['TotLen Bwd Pkts', 'Total Length of Bwd Packet'],
            'Flow Byts/s': ['Flow Byts/s', 'Flow Bytes/s'],
            'Flow Pkts/s': ['Flow Pkts/s', 'Flow Packets/s']
        }
        
        # Create feature engineering with flexible column names
        fwd_pkts_col = find_column(df, column_mappings['Tot Fwd Pkts'])
        bwd_pkts_col = find_column(df, column_mappings['Tot Bwd Pkts'])
        fwd_len_col = find_column(df, column_mappings['TotLen Fwd Pkts'])
        bwd_len_col = find_column(df, column_mappings['TotLen Bwd Pkts'])
        flow_bytes_col = find_column(df, column_mappings['Flow Byts/s'])
        flow_pkts_col = find_column(df, column_mappings['Flow Pkts/s'])
        
        # Only create features if we have the necessary columns
        if all([fwd_pkts_col, bwd_pkts_col, fwd_len_col, bwd_len_col, flow_bytes_col, flow_pkts_col]):
            # Create a copy of the dataframe to avoid reference issues
            try:
                # Before feature engineering, add this data cleaning step
                for col in df.columns:
                    # Skip explicitly non-numeric columns
                    if col == 'Label' or col == 'Label_binary':
                        continue
                        
                    # Try to convert to numeric
                    if df[col].dtype == object:  # If it's a string or object type
                        try:
                            # First, handle any headers that were incorrectly included
                            if len(df) > 0 and isinstance(df[col].iloc[0], str) and any(df[col].iloc[0] == col_name for col_name in df.columns):
                                print(f"Warning: Column {col} contains header as value. Dropping column.")
                                df = df.drop(col, axis=1)
                                continue
                            
                            # Then try conversion
                            df[col] = pd.to_numeric(df[col], errors='coerce')
                            print(f"Converted column {col} to numeric")
                        except Exception as e:
                            print(f"Could not convert column {col}: {e}")
                
                # Clean up NaN values after conversion
                df = df.fillna(0)
                
                # Now create the engineered features
                df['bytes_per_packet_fwd'] = df[fwd_len_col] / (df[fwd_pkts_col] + 0.1)
                df['bytes_per_packet_bwd'] = df[bwd_len_col] / (df[bwd_pkts_col] + 0.1)
                df['packet_size_ratio'] = df['bytes_per_packet_fwd'] / (df['bytes_per_packet_bwd'] + 0.1)
                df['packet_rate_ratio'] = df[flow_pkts_col] / (df['Flow Duration'] + 0.1)
                df['bytes_per_second'] = df[flow_bytes_col]
                df['total_packets'] = df[fwd_pkts_col] + df[bwd_pkts_col]
                df['fwd_to_bwd_packet_ratio'] = df[fwd_pkts_col] / (df[bwd_pkts_col] + 0.1)
                
                # Variance-based features - use nan-safe operations
                df['byte_rate_variance'] = np.nanvar([df['bytes_per_packet_fwd'], df['bytes_per_packet_bwd']], axis=0)
                df['packet_length_std'] = np.sqrt(np.nanvar([
                    df[fwd_len_col] / (df[fwd_pkts_col] + 0.1), 
                    df[bwd_len_col] / (df[bwd_pkts_col] + 0.1)
                ], axis=0))
                
                # Use Flow IAT Mean if available
                if 'Flow IAT Mean' in df.columns:
                    df['flow_iat_mean'] = df['Flow IAT Mean']
                else:
                    df['flow_iat_mean'] = df['Flow Duration'] / (df['total_packets'] + 1)
                
                # Replace any remaining infinities and NaNs that might have occurred
                df = df.replace([np.inf, -np.inf], np.nan)
                df = df.fillna(0)
                
                print("✅ Created all engineered features")
            except Exception as e:
                print(f"Error during feature engineering: {e}")
                # Create basic features if there was an error
                df['total_packets'] = df[fwd_pkts_col] + df[bwd_pkts_col]
                print("⚠️ Created only basic features due to error")
        else:
            print("⚠️ Missing some required columns for feature engineering")
            missing_cols = []
            if not fwd_pkts_col: missing_cols.append("forward packets")
            if not bwd_pkts_col: missing_cols.append("backward packets")
            if not fwd_len_col: missing_cols.append("forward length")
            if not bwd_len_col: missing_cols.append("backward length")
            if not flow_bytes_col: missing_cols.append("flow bytes")
            if not flow_pkts_col: missing_cols.append("flow packets")
            print(f"Missing columns: {', '.join(missing_cols)}")
        
        # Handle timestamp
        if 'Timestamp' in df.columns:
            try:
                # Check if it's already a literal column name 'Timestamp' instead of containing timestamps
                if df['Timestamp'].dtype == object and df['Timestamp'].iloc[0] == 'Timestamp':
                    print("Warning: 'Timestamp' is a column header, not actual timestamp data. Dropping column.")
                    df = df.drop('Timestamp', axis=1)
                else:
                    # Try various timestamp formats
                    try:
                        with warnings.catch_warnings():
                            warnings.filterwarnings("ignore", category=UserWarning)
                            df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
                        
                        # Fill missing values and convert to Unix timestamp
                        df['Timestamp'] = df['Timestamp'].fillna(pd.Timestamp('2018-01-01'))  # Default timestamp
                        df['Timestamp'] = df['Timestamp'].astype(np.int64) // 10**9
                        print("Converted Timestamp to Unix epoch format")
                    except Exception as e:
                        print(f"Error converting timestamp: {e}")
                        # If timestamp conversion fails, create a dummy timestamp
                        print("Creating a sequence timestamp instead")
                        df['Timestamp'] = np.arange(len(df))
            except Exception as e:
                print(f"Error handling timestamp column: {e}")
                # If all else fails, create a sequence
                df['Timestamp'] = np.arange(len(df))
        
        # Handle categorical columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        categorical_cols = [col for col in categorical_cols if col not in ['Label', 'Label_binary']]
        categorical_cols = [col for col in categorical_cols if df[col].nunique() < 100]
        
        # Split features and target
        X = df.drop(['Label', 'Label_binary', 'Flow ID', 'Src IP', 'Dst IP'], axis=1, errors='ignore')
        y = df['Label_binary']
        
        # Drop identifier columns
        columns_to_drop = ['Flow ID', 'Src IP', 'Dst IP']
        X = X.drop(columns=columns_to_drop, errors='ignore')
        
        # Before train/test split - check if we have enough samples for stratification
        min_samples_per_class = y.value_counts().min()
        if min_samples_per_class < 5:
            print(f"Warning: Insufficient samples in minority class ({min_samples_per_class}). Using simple random split.")
            # Use simple random split instead of stratified
            X_train, X_temp = train_test_split(X, test_size=0.3, random_state=42)
            y_train = y.loc[X_train.index]
            y_temp = y.loc[X_temp.index]
            
            X_val, X_test = train_test_split(X_temp, test_size=0.5, random_state=42)
            y_val = y_temp.loc[X_val.index]
            y_test = y_temp.loc[X_test.index]
            
            # Check if we have at least one sample of each class in each split
            if y_train.nunique() < 2 or y_val.nunique() < 2 or y_test.nunique() < 2:
                print("Warning: Some splits don't have both classes. Using synthetic samples.")
                
                # Create synthetic samples to ensure both classes in all splits
                for split_y, split_name in [(y_train, "train"), (y_val, "validation"), (y_test, "test")]:
                    if split_y.nunique() < 2:
                        missing_class = 1 if 1 not in split_y.values else 0
                        print(f"Adding synthetic samples of class {missing_class} to {split_name} set")
                        
                        # Add synthetic samples (copies of existing ones with small noise)
                        if missing_class == 1 and 1 in y.values:
                            # Find a class 1 sample from original data
                            orig_sample_idx = y[y == 1].index[0]
                            orig_sample = X.loc[orig_sample_idx].copy()
                            
                            # Add to the appropriate split with small random noise
                            if split_name == "train":
                                X_train = pd.concat([X_train, pd.DataFrame([orig_sample * (1 + np.random.randn() * 0.01)], 
                                            index=[f"synthetic_{missing_class}"])])
                                y_train = pd.concat([y_train, pd.Series([missing_class], index=[f"synthetic_{missing_class}"])])
                            elif split_name == "validation":
                                X_val = pd.concat([X_val, pd.DataFrame([orig_sample * (1 + np.random.randn() * 0.01)], 
                                          index=[f"synthetic_{missing_class}_val"])])
                                y_val = pd.concat([y_val, pd.Series([missing_class], index=[f"synthetic_{missing_class}_val"])])
                            else:  # test
                                X_test = pd.concat([X_test, pd.DataFrame([orig_sample * (1 + np.random.randn() * 0.01)], 
                                           index=[f"synthetic_{missing_class}_test"])])
                                y_test = pd.concat([y_test, pd.Series([missing_class], index=[f"synthetic_{missing_class}_test"])])
                        
                        elif missing_class == 0 and 0 in y.values:
                            # Find a class 0 sample from original data
                            orig_sample_idx = y[y == 0].index[0]
                            orig_sample = X.loc[orig_sample_idx].copy()
                            
                            # Add to the appropriate split with small random noise
                            if split_name == "train":
                                X_train = pd.concat([X_train, pd.DataFrame([orig_sample * (1 + np.random.randn() * 0.01)], 
                                            index=[f"synthetic_{missing_class}"])])
                                y_train = pd.concat([y_train, pd.Series([missing_class], index=[f"synthetic_{missing_class}"])])
                            elif split_name == "validation":
                                X_val = pd.concat([X_val, pd.DataFrame([orig_sample * (1 + np.random.randn() * 0.01)], 
                                          index=[f"synthetic_{missing_class}_val"])])
                                y_val = pd.concat([y_val, pd.Series([missing_class], index=[f"synthetic_{missing_class}_val"])])
                            else:  # test
                                X_test = pd.concat([X_test, pd.DataFrame([orig_sample * (1 + np.random.randn() * 0.01)], 
                                           index=[f"synthetic_{missing_class}_test"])])
                                y_test = pd.concat([y_test, pd.Series([missing_class], index=[f"synthetic_{missing_class}_test"])])
        else:
            # Use stratified split as before
            X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
            X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
        
        # Print split sizes
        print(f"Train set: {len(X_train)} samples, Test set: {len(X_test)} samples, Val set: {len(X_val)} samples")
        print(f"Class distribution - Train: {y_train.value_counts().to_dict()}, Test: {y_test.value_counts().to_dict()}")
        
        # Handle categorical features
        categorical_cols = X_train.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            # One-hot encode categorical features
            X_train_cat = pd.get_dummies(X_train[categorical_cols], drop_first=True)
            X_val_cat = pd.get_dummies(X_val[categorical_cols], drop_first=True)
            X_test_cat = pd.get_dummies(X_test[categorical_cols], drop_first=True)
            
            # Ensure same columns in validation and test sets
            for col in X_train_cat.columns:
                if col not in X_val_cat.columns:
                    X_val_cat[col] = 0
                if col not in X_test_cat.columns:
                    X_test_cat[col] = 0
            
            # Align columns
            X_val_cat = X_val_cat[X_train_cat.columns]
            X_test_cat = X_test_cat[X_train_cat.columns]
        
        # Scale numerical features
        numerical_cols = X_train.select_dtypes(include=['int64', 'float64']).columns
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train[numerical_cols])
        X_val_scaled = scaler.transform(X_val[numerical_cols])
        X_test_scaled = scaler.transform(X_test[numerical_cols])
        
        # Combine numerical and categorical features
        if len(categorical_cols) > 0:
            X_train_final = np.hstack([X_train_scaled, X_train_cat])
            X_val_final = np.hstack([X_val_scaled, X_val_cat])
            X_test_final = np.hstack([X_test_scaled, X_test_cat])
        else:
            X_train_final = X_train_scaled
            X_val_final = X_val_scaled 
            X_test_final = X_test_scaled
        
        # Add these data cleaning steps before model training:
        try:
            # Clean feature matrices from infinities and very large values
            def clean_feature_matrix(X):
                if isinstance(X, np.ndarray):
                    # Replace inf with large but finite values
                    X = np.nan_to_num(X, nan=0, posinf=1e10, neginf=-1e10)
                    # Clip values to reasonable range to avoid numerical issues
                    X = np.clip(X, -1e10, 1e10)
                return X
            
            X_train_final = clean_feature_matrix(X_train_final)
            X_val_final = clean_feature_matrix(X_val_final)
            X_test_final = clean_feature_matrix(X_test_final)
            
            print("Cleaned feature matrices of extreme values")
        except Exception as e:
            print(f"Error during matrix cleaning: {e}")
        
        # Before training Random Forest, check X_train_final
        print(f"Training data shape: {X_train_final.shape}")
        if isinstance(X_train_final, np.ndarray) and (np.isnan(X_train_final).any() or np.isinf(X_train_final).any()):
            print("Warning: Training data contains NaN or inf. Fixing...")
            X_train_final = np.nan_to_num(X_train_final, nan=0, posinf=1e10, neginf=-1e10)
        
        # Before initializing metrics dataframe, check if we have enough data to train
        if len(X_train_final) < 10 or len(X_test_final) < 5:
            print(f"Insufficient data for reliable model training: Train={len(X_train_final)}, Test={len(X_test_final)}")
            print("Skipping model training and evaluation for this dataset")
            
            # Create a placeholder metrics DataFrame with warnings
            placeholder_metrics = {
                'Model': f"Warning_{os.path.basename(dataset_path)}",
                'accuracy': float('nan'),
                'precision': float('nan'),
                'recall': float('nan'),
                'f1': float('nan'),
                'false_positive_rate': float('nan'),
                'error': f"Insufficient data: Train={len(X_train_final)}, Test={len(X_test_final)}"
            }
            warning_df = pd.DataFrame([placeholder_metrics])
            
            # Return early with warning
            return warning_df, label_stats
        
        # Initialize metrics dataframe
        metrics_df = pd.DataFrame(columns=['Model', 'accuracy', 'precision', 'recall', 'f1', 'false_positive_rate'])
        
        # Define evaluate_model function
        def evaluate_model(model, X_test, y_test, model_name):
            y_pred = model.predict(X_test)
            
            metrics = {
                'Model': f"{model_name}_{os.path.basename(dataset_path)}",
                'accuracy': (y_pred == y_test).mean(),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred),
                'false_positive_rate': (y_pred[y_test==0] == 1).mean() if sum(y_test==0) > 0 else 0
            }
            
            return metrics
        
        # Train Random Forest
        print("\nTraining Random Forest...")
        rf = RandomForestClassifier(n_estimators=200, max_depth=15, class_weight='balanced', random_state=42)
        rf.fit(X_train_final, y_train)
        
        # Evaluate Random Forest
        y_pred_rf = rf.predict(X_test_final)
        rf_metrics = {
            'Model': f"RF_{os.path.basename(dataset_path)}",
            'accuracy': (y_pred_rf == y_test).mean(),
            'precision': precision_score(y_test, y_pred_rf),
            'recall': recall_score(y_test, y_pred_rf),
            'f1': f1_score(y_test, y_pred_rf),
            'false_positive_rate': (y_pred_rf[y_test==0] == 1).mean() if sum(y_test==0) > 0 else 0
        }
        metrics_df = pd.concat([metrics_df, pd.DataFrame([rf_metrics])], ignore_index=True)
        
        # Train Gradient Boosting
        print("\nTraining Gradient Boosting...")
        gb = GradientBoostingClassifier(n_estimators=200, max_depth=5, random_state=42)
        gb.fit(X_train_final, y_train)
        
        # Evaluate Gradient Boosting
        gb_metrics = evaluate_model(gb, X_test_final, y_test, 'GB')
        metrics_df = pd.concat([metrics_df, pd.DataFrame([gb_metrics])], ignore_index=True)
        
        # Create and evaluate ensemble model
        print("\nEvaluating Ensemble Model...")
        try:
            # Make sure the model can predict probabilities
            y_prob = gb.predict_proba(X_test_final)[:, 1]
            
            # Calculate precision-recall curve
            precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob)
            
            # Make sure we have valid data for F1 calculation
            if len(precisions) > 1 and len(recalls) > 1:
                # Calculate F1 scores
                f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-10)
                best_threshold = thresholds[np.argmax(f1_scores)]
                
                # Apply the threshold
                y_pred_custom = (y_prob >= best_threshold).astype(int)
                
                # Calculate metrics
                ensemble_metrics = {
                    'Model': f"Ensemble_{os.path.basename(dataset_path)}",
                    'accuracy': (y_pred_custom == y_test).mean(),
                    'precision': precision_score(y_test, y_pred_custom),
                    'recall': recall_score(y_test, y_pred_custom),
                    'f1': f1_score(y_test, y_pred_custom),
                    'false_positive_rate': (y_pred_custom[y_test==0] == 1).mean() if sum(y_test==0) > 0 else 0
                }
                
                # Convert to DataFrame with careful handling
                ensemble_df = pd.DataFrame([ensemble_metrics])
                
                # Safely concatenate
                if isinstance(metrics_df, pd.DataFrame) and len(metrics_df) > 0:
                    metrics_df = pd.concat([metrics_df, ensemble_df], ignore_index=True)
                else:
                    metrics_df = ensemble_df
            else:
                print("Warning: Not enough data points for precision-recall curve")
        except Exception as e:
            print(f"Error in ensemble model: {e}")
            traceback.print_exc()
        
        # Create results directory
        results_dir = 'results'
        os.makedirs(results_dir, exist_ok=True)
        
        # Save results for this dataset
        file_metrics_path = f"{results_dir}/{os.path.basename(dataset_path)}_metrics.csv"
        try:
            # Check if the metrics DataFrame is valid
            if not metrics_df.empty:
                # Convert to dictionary and back to ensure clean DataFrame
                metrics_dict = []
                for idx, row in metrics_df.iterrows():
                    metrics_dict.append(dict(row))
                
                # Create a fresh DataFrame
                new_metrics_df = pd.DataFrame(metrics_dict)
                
                # Save with robust error handling
                try:
                    new_metrics_df.to_csv(file_metrics_path, index=False)
                    print(f"Metrics saved to {file_metrics_path}")
                except Exception as e:
                    print(f"Error saving metrics CSV: {e}")
                    # Fallback to JSON format
                    with open(file_metrics_path.replace('.csv', '.json'), 'w') as f:
                        json.dump(metrics_dict, f)
                    print(f"Saved metrics as JSON instead")
            else:
                print("No metrics to save")
        except Exception as e:
            print(f"Error preparing metrics for saving: {e}")
        
        # Clean up memory
        del X_train_scaled, X_val_scaled, X_test_scaled
        if len(categorical_cols) > 0:
            del X_train_cat, X_val_cat, X_test_cat
        del X_train, X_val, X_test
        del X_train_final, X_val_final, X_test_final
        gc.collect()
        
        # After calling analyze_dataset:
        # Force memory cleanup between files
        gc.collect()
        print(f"Memory usage: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")
        
        return metrics_df, label_stats
    
    except Exception as e:
        print(f"Error analyzing {dataset_path}: {e}")
        print("Detailed error information:")
        traceback.print_exc()
        # Return empty results if analysis fails
        return pd.DataFrame(), {}

# Main execution
if __name__ == "__main__":
    # Define the dataset directory
    dataset_dir = '/content/drive/MyDrive/Datasets/archive/'
    
    # Process all files
    all_day_files = glob.glob(f"{dataset_dir}/*.csv")
    
    # Verify file paths exist before running
    existing_files = []
    for file_path in all_day_files:
        if os.path.exists(file_path):
            existing_files.append(file_path)
        else:
            print(f"Warning: File not found: {file_path}")
    
    print(f"Found {len(existing_files)} files to process")
    
    # Store results from all datasets
    all_metrics = []
    all_label_stats = {}
    
    # Process each dataset file
    for i, day_file in enumerate(existing_files):
        print(f"\nProcessing file {i+1} of {len(existing_files)}: {os.path.basename(day_file)}")
        
        # Analyze this dataset
        metrics, stats = analyze_dataset(day_file, sample_size=25000)
        
        # Only add to results if we got valid metrics
        if not metrics.empty:
            # Store results
            all_metrics.append(metrics)
            all_label_stats[os.path.basename(day_file)] = stats
            
            # Save incremental combined results
            if all_metrics:
                try:
                    # Convert all DataFrames to a list of dictionaries first
                    all_metrics_dicts = []
                    for metrics_df in all_metrics:
                        for _, row in metrics_df.iterrows():
                            all_metrics_dicts.append(dict(row))
                            
                    # Create a fresh DataFrame from dictionaries
                    combined_metrics = pd.DataFrame(all_metrics_dicts)
                    
                    # Save with error handling
                    try:
                        combined_metrics.to_csv("results/combined_metrics.csv", index=False)
                        print("✅ Saved combined metrics to CSV")
                    except Exception as e:
                        print(f"Error saving combined CSV: {e}")
                        # Fallback to JSON
                        with open("results/combined_metrics.json", "w") as f:
                            json.dump(all_metrics_dicts, f)
                        print("Saved combined metrics as JSON instead")
                    
                    # Save label statistics - this should be fine as is
                    with open("results/label_statistics.json", "w") as f:
                        json.dump(all_label_stats, f, indent=2)
                    
                except Exception as e:
                    print(f"Error saving combined results: {e}")
                    traceback.print_exc()
            
            print(f"✅ Successfully processed {os.path.basename(day_file)}")
        else:
            print(f"❌ No valid metrics for {os.path.basename(day_file)}")
            
        print(f"Estimated progress: {(i+1)/len(existing_files)*100:.1f}%")
    
    # Create final combined visualizations if we have results
    if all_metrics:
        try:
            combined_metrics = pd.concat(all_metrics)
            
            # Visualize combined results
            plt.figure(figsize=(15, 10))
            
            # Group by model type (RF, GB, Ensemble)
            metrics_to_plot = ['precision', 'recall', 'f1', 'false_positive_rate']
            
            for i, metric in enumerate(metrics_to_plot, 1):
                plt.subplot(2, 2, i)
                
                # Extract base model type from Model column
                combined_metrics['ModelType'] = combined_metrics['Model'].apply(
                    lambda x: x.split('_')[0] if '_' in x else x)
                
                sns.barplot(x='ModelType', y=metric, data=combined_metrics)
                plt.title(f"{metric.capitalize()} by Model Type")
                plt.ylim(0, 1)
            
            plt.tight_layout()
            plt.savefig("results/combined_performance.png")
            plt.show()
            
            print("\n==== SUMMARY OF ALL DATASETS ====\n")
            # Print overall model type performance
            summary = combined_metrics.groupby('ModelType')[
                ['accuracy', 'precision', 'recall', 'f1', 'false_positive_rate']
            ].mean().reset_index()
            
            print("Average performance by model type:")
            print(summary)
            
            # Print summary of attack types across all datasets
            print("\nAttack types by dataset:")
            for filename, stats in all_label_stats.items():
                if 'attack_types' in stats:
                    print(f"\n{filename}:")
                    attack_types = {k: v for k, v in stats['attack_types'].items() 
                                   if k != 'Benign' and k != 'BENIGN'}
                    for attack, count in attack_types.items():
                        print(f"  - {attack}: {count}")
        except Exception as e:
            print(f"Error generating final visualizations: {e}")
    else:
        print("No metrics collected from any dataset. Please check for errors.") 