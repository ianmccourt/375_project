"""
Data processing utilities for network traffic data.
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import category_encoders as ce

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config

import gc
import time
from google.colab import drive
import psutil


class NetworkDataProcessor:
    """
    Class for processing network traffic data for UBA analysis.
    """
    
    def __init__(self, scaling_method='standard', handle_imbalance=True):
        """
        Initialize the data processor.
        
        Args:
            scaling_method (str): Method for scaling numerical features ('standard', 'minmax', or None)
            handle_imbalance (bool): Whether to handle class imbalance
        """
        self.scaling_method = scaling_method
        self.handle_imbalance = handle_imbalance
        self.scaler = None
        self.categorical_encoder = None
        self.numerical_columns = None
        self.categorical_columns = None
        self.target_column = None
        
    def load_data(self, file_path, target_column='Label'):
        """
        Load data from a CSV file.
        
        Args:
            file_path (str): Path to the CSV file
            target_column (str): Name of the target column
            
        Returns:
            pd.DataFrame: Loaded dataframe
        """
        print(f"Loading data from {file_path}...")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Determine file type and load accordingly
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.parquet'):
            df = pd.read_parquet(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
        
        self.target_column = target_column
        print(f"Data loaded with shape: {df.shape}")
        return df
    
    def load_tii_ssrc_dataset(self, file_path, target_column='label'):
        """
        Load and preprocess TII-SSRC-23 dataset.
        
        Args:
            file_path (str): Path to the TII-SSRC-23 dataset
            target_column (str): Name of the target column (default is 'label' for TII-SSRC-23)
            
        Returns:
            pd.DataFrame: Preprocessed dataframe
        """
        print(f"Loading TII-SSRC-23 dataset from {file_path}...")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Load the dataset
        df = pd.read_csv(file_path)
        
        # Handle TII-SSRC-23 specific preprocessing
        
        # 1. Set the target column
        self.target_column = target_column
        
        # 2. Map the target/label column to binary format (if needed)
        # Check if the label is already binary or needs conversion
        if target_column in df.columns and not set(df[target_column].unique()).issubset({0, 1}):
            print(f"Converting {target_column} to binary format...")
            # Map attack categories to 0 (anomaly) and normal traffic to 1 (normal)
            # Assuming "Normal" or "Benign" labels for normal traffic, adjust as needed
            if df[target_column].dtype == 'object':
                normal_labels = ['Normal', 'Benign', 'normal', 'benign']
                df[target_column] = df[target_column].apply(lambda x: 1 if x in normal_labels else 0)
            # If it's numeric but not binary, map all non-zero values to 0 (anomaly)
            elif df[target_column].dtype in ['int64', 'float64']:
                # If the dataset uses 0 for normal and other values for attacks, invert the logic
                if 0 in df[target_column].unique() and df[target_column].nunique() > 2:
                    df[target_column] = df[target_column].apply(lambda x: 1 if x == 0 else 0)
        
        # 3. Map column names to standardized format
        column_mapping = self._get_tii_ssrc_column_mapping()
        
        # Apply mapping only for columns that exist in the dataframe
        valid_mapping = {k: v for k, v in column_mapping.items() if k in df.columns}
        if valid_mapping:
            df = df.rename(columns=valid_mapping)
        
        # 4. Ensure timestamp is in datetime format
        timestamp_col = next((col for col in df.columns if 'time' in col.lower()), None)
        if timestamp_col and pd.api.types.is_string_dtype(df[timestamp_col]):
            try:
                df[timestamp_col] = pd.to_datetime(df[timestamp_col])
            except:
                print(f"Warning: Could not convert {timestamp_col} to datetime format.")
        
        # 5. Add derived features if possible
        df = self._add_derived_features_tii_ssrc(df)
        
        print(f"TII-SSRC-23 dataset loaded and preprocessed with shape: {df.shape}")
        print(f"Class distribution: {df[self.target_column].value_counts().to_dict()}")
        
        return df
    
    def _get_tii_ssrc_column_mapping(self):
        """
        Get column mapping from TII-SSRC-23 dataset format to standardized format.
        
        Returns:
            dict: Column mapping dictionary
        """
        # Mapping from TII-SSRC-23 column names to standardized names
        # Adjust this mapping based on the actual column names in your dataset
        return {
            'src_ip': 'Source IP',
            'dst_ip': 'Destination IP',
            'src_port': 'Source Port',
            'dst_port': 'Destination Port',
            'protocol': 'Protocol',
            'timestamp': 'Timestamp',
            'duration': 'Flow Duration',
            'total_fwd_packets': 'Total Fwd Packets',
            'total_bwd_packets': 'Total Backward Packets',
            'total_length_of_fwd_packets': 'Total Length of Fwd Packets',
            'total_length_of_bwd_packets': 'Total Length of Bwd Packets',
            'fwd_packet_length_max': 'Fwd Packet Length Max',
            'fwd_packet_length_min': 'Fwd Packet Length Min',
            'fwd_packet_length_mean': 'Fwd Packet Length Mean',
            'bwd_packet_length_max': 'Bwd Packet Length Max', 
            'bwd_packet_length_min': 'Bwd Packet Length Min',
            'bwd_packet_length_mean': 'Bwd Packet Length Mean',
            'flow_bytes_per_s': 'Flow Bytes/s',
            'flow_packets_per_s': 'Flow Packets/s',
            'label': 'Label'
        }
    
    def _add_derived_features_tii_ssrc(self, df):
        """
        Add derived features specific to network traffic.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with added features
        """
        # Only add these features if we have the necessary columns
        try:
            # Calculate bytes per packet if we have both packets and bytes
            if 'Total Fwd Packets' in df.columns and 'Total Length of Fwd Packets' in df.columns:
                df['Fwd Bytes per Packet'] = df['Total Length of Fwd Packets'] / df['Total Fwd Packets'].replace(0, 1)
            
            if 'Total Backward Packets' in df.columns and 'Total Length of Bwd Packets' in df.columns:
                df['Bwd Bytes per Packet'] = df['Total Length of Bwd Packets'] / df['Total Backward Packets'].replace(0, 1)
            
            # Calculate packet ratio
            if 'Total Fwd Packets' in df.columns and 'Total Backward Packets' in df.columns:
                total_packets = df['Total Fwd Packets'] + df['Total Backward Packets']
                df['Fwd Packet Ratio'] = df['Total Fwd Packets'] / total_packets.replace(0, 1)
                df['Bwd Packet Ratio'] = df['Total Backward Packets'] / total_packets.replace(0, 1)
            
            # Identify scan-like behavior
            if 'Destination IP' in df.columns and 'Source IP' in df.columns:
                # Count distinct destinations per source
                if len(df) > 1:  # Only if we have more than one row
                    src_ip_counts = df.groupby('Source IP')['Destination IP'].nunique()
                    ip_counts_dict = src_ip_counts.to_dict()
                    df['Distinct Destinations Count'] = df['Source IP'].map(ip_counts_dict)
        except Exception as e:
            print(f"Warning: Could not add some derived features. Error: {e}")
        
        return df
    
    def identify_column_types(self, df):
        """
        Identify numerical and categorical columns in the dataframe.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            tuple: Lists of numerical and categorical columns
        """
        # Exclude the target column
        feature_columns = [col for col in df.columns if col != self.target_column]
        
        # Identify numerical and categorical columns
        numerical_columns = df[feature_columns].select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_columns = df[feature_columns].select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
        
        self.numerical_columns = numerical_columns
        self.categorical_columns = categorical_columns
        
        print(f"Identified {len(numerical_columns)} numerical columns and {len(categorical_columns)} categorical columns")
        return numerical_columns, categorical_columns
    
    def handle_missing_values(self, df):
        """
        Handle missing values in the dataframe.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with missing values handled
        """
        print("Handling missing values...")
        
        # Check for missing values
        missing_values = df.isnull().sum()
        columns_with_missing = missing_values[missing_values > 0].index.tolist()
        
        if not columns_with_missing:
            print("No missing values found")
            return df
        
        print(f"Found missing values in {len(columns_with_missing)} columns")
        
        # Handle missing values in numerical columns
        numerical_with_missing = [col for col in columns_with_missing if col in self.numerical_columns]
        if numerical_with_missing:
            imputer = SimpleImputer(strategy='median')
            df[numerical_with_missing] = imputer.fit_transform(df[numerical_with_missing])
        
        # Handle missing values in categorical columns
        categorical_with_missing = [col for col in columns_with_missing if col in self.categorical_columns]
        if categorical_with_missing:
            imputer = SimpleImputer(strategy='most_frequent')
            df[categorical_with_missing] = imputer.fit_transform(df[categorical_with_missing])
        
        return df
    
    def encode_categorical_features(self, df, method='onehot'):
        """
        Encode categorical features.
        
        Args:
            df (pd.DataFrame): Input dataframe
            method (str): Encoding method ('onehot', 'label', 'target', 'binary')
            
        Returns:
            pd.DataFrame: Dataframe with encoded categorical features
        """
        if not self.categorical_columns:
            print("No categorical columns to encode")
            return df
        
        print(f"Encoding {len(self.categorical_columns)} categorical columns using {method} encoding...")
        
        if method == 'onehot':
            self.categorical_encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
            encoded_features = self.categorical_encoder.fit_transform(df[self.categorical_columns])
            
            # Create a dataframe with the encoded features
            encoded_df = pd.DataFrame(
                encoded_features, 
                columns=self.categorical_encoder.get_feature_names_out(self.categorical_columns),
                index=df.index
            )
            
            # Drop original categorical columns and concatenate encoded features
            df_encoded = pd.concat([df.drop(columns=self.categorical_columns), encoded_df], axis=1)
            
        elif method == 'label':
            self.categorical_encoder = ce.OrdinalEncoder(cols=self.categorical_columns)
            df_encoded = self.categorical_encoder.fit_transform(df)
            
        elif method == 'target':
            if self.target_column not in df.columns:
                raise ValueError("Target column not found in dataframe")
            
            self.categorical_encoder = ce.TargetEncoder(cols=self.categorical_columns)
            df_encoded = self.categorical_encoder.fit_transform(df, df[self.target_column])
            
        elif method == 'binary':
            self.categorical_encoder = ce.BinaryEncoder(cols=self.categorical_columns)
            df_encoded = self.categorical_encoder.fit_transform(df)
            
        else:
            raise ValueError(f"Unsupported encoding method: {method}")
        
        print(f"Encoded dataframe shape: {df_encoded.shape}")
        return df_encoded
    
    def scale_numerical_features(self, df):
        """
        Scale numerical features.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with scaled numerical features
        """
        if not self.numerical_columns:
            print("No numerical columns to scale")
            return df
        
        if not self.scaling_method:
            print("Scaling method is None, skipping scaling")
            return df
        
        print(f"Scaling {len(self.numerical_columns)} numerical columns using {self.scaling_method} scaling...")
        
        if self.scaling_method == 'standard':
            self.scaler = StandardScaler()
        elif self.scaling_method == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unsupported scaling method: {self.scaling_method}")
        
        # Scale numerical features
        df[self.numerical_columns] = self.scaler.fit_transform(df[self.numerical_columns])
        
        return df
    
    def handle_class_imbalance(self, X, y, method='smote', sampling_strategy=0.5):
        """
        Handle class imbalance in the dataset.
        
        Args:
            X (pd.DataFrame): Feature dataframe
            y (pd.Series): Target series
            method (str): Method for handling imbalance ('smote', 'undersample', 'both')
            sampling_strategy (float): Sampling strategy for minority class
            
        Returns:
            tuple: Balanced X and y
        """
        if not self.handle_imbalance:
            print("Class imbalance handling is disabled")
            return X, y
        
        print(f"Handling class imbalance using {method}...")
        
        # Check class distribution
        class_counts = y.value_counts()
        print(f"Class distribution before balancing: {class_counts.to_dict()}")
        
        if method == 'smote':
            smote = SMOTE(sampling_strategy=sampling_strategy, random_state=config.EVALUATION['random_state'])
            X_balanced, y_balanced = smote.fit_resample(X, y)
            
        elif method == 'undersample':
            undersampler = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=config.EVALUATION['random_state'])
            X_balanced, y_balanced = undersampler.fit_resample(X, y)
            
        elif method == 'both':
            # First undersample the majority class, then oversample the minority class
            undersampler = RandomUnderSampler(sampling_strategy=0.1, random_state=config.EVALUATION['random_state'])
            X_temp, y_temp = undersampler.fit_resample(X, y)
            
            smote = SMOTE(sampling_strategy=sampling_strategy, random_state=config.EVALUATION['random_state'])
            X_balanced, y_balanced = smote.fit_resample(X_temp, y_temp)
            
        else:
            raise ValueError(f"Unsupported imbalance handling method: {method}")
        
        # Check new class distribution
        new_class_counts = pd.Series(y_balanced).value_counts()
        print(f"Class distribution after balancing: {new_class_counts.to_dict()}")
        
        return X_balanced, y_balanced
    
    def create_time_based_features(self, df, timestamp_column, window_size=300):
        """
        Create time-based features for behavioral analysis.
        
        Args:
            df (pd.DataFrame): Input dataframe
            timestamp_column (str): Name of the timestamp column
            window_size (int): Time window size in seconds
            
        Returns:
            pd.DataFrame: Dataframe with time-based features
        """
        print(f"Creating time-based features using window size of {window_size} seconds...")
        
        # Ensure timestamp column is datetime
        if pd.api.types.is_string_dtype(df[timestamp_column]):
            df[timestamp_column] = pd.to_datetime(df[timestamp_column])
        
        # Sort by timestamp
        df = df.sort_values(by=timestamp_column)
        
        # Create time windows
        df['time_window'] = df[timestamp_column].dt.floor(f'{window_size}S')
        
        # Group by source IP and time window to create behavioral features
        if 'Source IP' in df.columns:
            # Features for each source IP in each time window
            agg_features = df.groupby(['Source IP', 'time_window']).agg({
                'Destination IP': ['nunique', 'count'],
                'Destination Port': ['nunique', 'count'],
                'Protocol': ['nunique', 'count'],
                'Flow Duration': ['mean', 'std', 'min', 'max'],
                'Total Fwd Packets': ['mean', 'sum'],
                'Total Backward Packets': ['mean', 'sum'],
                'Total Length of Fwd Packets': ['mean', 'sum'],
                'Total Length of Bwd Packets': ['mean', 'sum']
            })
            
            # Flatten the column names
            agg_features.columns = ['_'.join(col).strip() for col in agg_features.columns.values]
            
            # Reset index to get back the Source IP and time_window columns
            agg_features = agg_features.reset_index()
            
            # Merge back with original dataframe
            df = pd.merge(df, agg_features, on=['Source IP', 'time_window'], how='left')
        
        return df
    
    def split_data(self, df):
        """
        Split data into train, validation, and test sets.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            tuple: X_train, X_val, X_test, y_train, y_val, y_test
        """
        print("Splitting data into train, validation, and test sets...")
        
        # Separate features and target
        X = df.drop(columns=[self.target_column])
        y = df[self.target_column]
        
        # First split: separate test set
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, 
            test_size=config.EVALUATION['test_size'],
            random_state=config.EVALUATION['random_state'],
            stratify=y
        )
        
        # Second split: separate validation set from training set
        val_size = config.EVALUATION['validation_size'] / (1 - config.EVALUATION['test_size'])
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val,
            test_size=val_size,
            random_state=config.EVALUATION['random_state'],
            stratify=y_train_val
        )
        
        print(f"Train set: {X_train.shape}, Validation set: {X_val.shape}, Test set: {X_test.shape}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def preprocess_data(self, df, categorical_encoding='onehot', handle_imbalance_method='smote'):
        """
        Preprocess the data with all necessary steps.
        
        Args:
            df (pd.DataFrame): Input dataframe
            categorical_encoding (str): Method for encoding categorical features
            handle_imbalance_method (str): Method for handling class imbalance
            
        Returns:
            tuple: Preprocessed data splits
        """
        print("Starting data preprocessing pipeline...")
        
        # Identify column types
        self.identify_column_types(df)
        
        # Handle missing values
        df = self.handle_missing_values(df)
        
        # Encode categorical features
        df = self.encode_categorical_features(df, method=categorical_encoding)
        
        # Scale numerical features
        df = self.scale_numerical_features(df)
        
        # Split data
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(df)
        
        # Handle class imbalance (only for training data)
        if self.handle_imbalance:
            X_train, y_train = self.handle_class_imbalance(X_train, y_train, method=handle_imbalance_method)
        
        print("Data preprocessing completed")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def transform_new_data(self, df):
        """
        Transform new data using the fitted preprocessor.
        
        Args:
            df (pd.DataFrame): New data to transform
            
        Returns:
            pd.DataFrame: Transformed data
        """
        print("Transforming new data...")
        
        # Handle missing values
        df = self.handle_missing_values(df)
        
        # Transform categorical features
        if self.categorical_encoder and self.categorical_columns:
            if isinstance(self.categorical_encoder, OneHotEncoder):
                encoded_features = self.categorical_encoder.transform(df[self.categorical_columns])
                
                # Create a dataframe with the encoded features
                encoded_df = pd.DataFrame(
                    encoded_features, 
                    columns=self.categorical_encoder.get_feature_names_out(self.categorical_columns),
                    index=df.index
                )
                
                # Drop original categorical columns and concatenate encoded features
                df = pd.concat([df.drop(columns=self.categorical_columns), encoded_df], axis=1)
            else:
                df = self.categorical_encoder.transform(df)
        
        # Transform numerical features
        if self.scaler and self.numerical_columns:
            df[self.numerical_columns] = self.scaler.transform(df[self.numerical_columns])
        
        return df


# Comprehensive approach for safely loading large CSV files in Google Colab
class LargeCSVHandler:
    def __init__(self, file_path, target_column='label'):
        self.file_path = file_path
        self.target_column = target_column
        self.memory_limit_gb = 4.0  # Use at most 4GB for dataset
        self.chunk_size = 100000
        
    def check_file(self):
        """Check if file exists and its size"""
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"File not found: {self.file_path}")
        
        self.file_size_gb = os.path.getsize(self.file_path) / (1024**3)
        print(f"File exists. Size: {self.file_size_gb:.2f} GB")
        
        # Get available memory
        self.available_memory_gb = psutil.virtual_memory().available / (1024**3)
        print(f"Available memory: {self.available_memory_gb:.2f} GB")
        
        # Determine if we need sampling
        if self.file_size_gb > self.memory_limit_gb:
            print(f"File is larger than memory limit ({self.memory_limit_gb:.1f} GB). Will use sampling.")
            return False
        else:
            print("File size is within memory limits. Can load complete file.")
            return True
    
    def analyze_columns(self):
        """Analyze column structure without loading entire file"""
        # Read just the header and first few rows
        header_sample = pd.read_csv(self.file_path, nrows=5)
        print(f"\nFile has {len(header_sample.columns)} columns.")
        
        # Check for label column
        if self.target_column not in header_sample.columns:
            potential_labels = [col for col in header_sample.columns 
                              if 'label' in col.lower() or 'class' in col.lower() or 'target' in col.lower()]
            if potential_labels:
                self.target_column = potential_labels[0]
                print(f"Target column not found. Using '{self.target_column}' instead.")
            else:
                print("Warning: No obvious target column found.")
        
        # Identify likely important columns
        ip_cols = [col for col in header_sample.columns if 'ip' in col.lower()]
        time_cols = [col for col in header_sample.columns if 'time' in col.lower()]
        feature_cols = [col for col in header_sample.columns 
                       if any(key in col.lower() for key in ['flow', 'packet', 'byte', 'port', 'protocol'])]
        
        print(f"IP columns: {ip_cols}")
        print(f"Time columns: {time_cols}")
        print(f"Sample feature columns: {feature_cols[:5]}...")
        
        return header_sample.columns.tolist()
    
    def count_rows(self):
        """Count rows in file without loading it completely"""
        try:
            # Fast row counting for smaller files
            if self.file_size_gb < 1:
                row_count = sum(1 for _ in open(self.file_path)) - 1  # Subtract header
            else:
                # Estimate from file size and first chunk
                first_chunk = pd.read_csv(self.file_path, nrows=self.chunk_size)
                chunk_size_bytes = first_chunk.memory_usage(deep=True).sum()
                estimated_rows = int((self.file_size_gb * 1024**3) / (chunk_size_bytes / len(first_chunk)))
                row_count = estimated_rows
                del first_chunk
                gc.collect()
            
            print(f"Estimated row count: {row_count:,}")
            self.row_count = row_count
            return row_count
            
        except Exception as e:
            print(f"Error estimating row count: {e}")
            self.row_count = None
            return None
    
    def determine_sample_size(self):
        """Determine appropriate sample size based on file size and memory"""
        # Starting with memory-based calculation
        if not hasattr(self, 'row_count') or self.row_count is None:
            self.count_rows()
        
        # Calculate rows we can safely load
        safe_rows = int((self.memory_limit_gb * 1024**3) / (self.file_size_gb * 1024**3 / self.row_count))
        
        # Cap at either row_count or 1 million rows
        sample_size = min(self.row_count, safe_rows, 1000000)
        
        # Calculate sampling ratio
        sample_ratio = sample_size / self.row_count
        
        print(f"\nSampling strategy:")
        print(f"- Maximum safe sample size: {sample_size:,} rows")
        print(f"- Sampling ratio: {sample_ratio:.2%}")
        
        self.sample_size = sample_size
        self.sample_ratio = sample_ratio
        return sample_size
        
    def load_data(self, use_sampling=None, columns=None):
        """Load data with appropriate strategy"""
        # Determine if sampling is needed
        if use_sampling is None:
            use_sampling = not self.check_file()
        
        start_time = time.time()
        
        # Strategy 1: Load full file if possible
        if not use_sampling:
            print("\nLoading complete dataset...")
            if columns:
                df = pd.read_csv(self.file_path, usecols=columns)
            else:
                df = pd.read_csv(self.file_path)
        
        # Strategy 2: Random sampling for medium-sized files
        elif self.file_size_gb < 2 and self.sample_ratio > 0.1:
            print("\nLoading random sample of rows...")
            self.determine_sample_size()
            skiprows = self._generate_skiprows(self.sample_ratio)
            
            if columns:
                df = pd.read_csv(self.file_path, skiprows=skiprows, usecols=columns)
            else:
                df = pd.read_csv(self.file_path, skiprows=skiprows)
            
        # Strategy 3: Chunked loading for very large files
        else:
            print("\nLoading data in chunks...")
            self.determine_sample_size()
            chunks = []
            total_rows = 0
            
            for chunk in pd.read_csv(self.file_path, chunksize=self.chunk_size, usecols=columns):
                # Sample from this chunk
                chunk_sample = chunk.sample(frac=self.sample_ratio)
                chunks.append(chunk_sample)
                
                # Update progress
                total_rows += len(chunk_sample)
                print(f"  Progress: {total_rows:,} rows loaded", end='\r')
                
                # Stop if we've reached our target sample size
                if total_rows >= self.sample_size:
                    break
                
                # Clean up memory
                del chunk
                gc.collect()
            
            print("\nCombining chunks...")
            df = pd.concat(chunks, ignore_index=True)
            del chunks
            gc.collect()
            
        duration = time.time() - start_time
        print(f"\nLoaded {len(df):,} rows with {len(df.columns):,} columns in {duration:.1f} seconds")
        
        # Verify target column is present
        if self.target_column not in df.columns:
            print(f"Warning: Target column '{self.target_column}' not found in data!")
            potential_targets = [col for col in df.columns 
                               if 'label' in col.lower() or 'class' in col.lower()]
            if potential_targets:
                self.target_column = potential_targets[0]
                print(f"Using '{self.target_column}' as target instead.")
        
        return df
    
    def _generate_skiprows(self, sample_ratio):
        """Generate rows to skip for efficient sampling"""
        # Generate a list of rows to skip
        if not hasattr(self, 'row_count') or self.row_count is None:
            self.count_rows()
            
        # Calculate how many rows to skip
        rows_to_keep = int(self.row_count * sample_ratio)
        rows_to_skip = self.row_count - rows_to_keep
        
        if rows_to_skip <= 0:
            return None
            
        # Randomly select rows to skip (but always keep header row)
        np.random.seed(42)  # For reproducibility
        skip_indices = np.random.choice(
            range(1, self.row_count + 1),  # +1 for header
            size=rows_to_skip, 
            replace=False
        )
        
        return lambda x: x > 0 and x in skip_indices


if __name__ == "__main__":
    # Example usage
    processor = NetworkDataProcessor(scaling_method='standard', handle_imbalance=True)
    
    # This is just a placeholder - in a real scenario, you would provide the actual file path
    # df = processor.load_data(os.path.join(config.DATA_DIR, "network_traffic.csv"))
    
    print("Data processor module created successfully")

    # Step 1: Mount Google Drive
    print("Mounting Google Drive...")
    drive.mount('/content/drive', force_remount=True)

    # Step 2: Define the dataset path
    dataset_path = '/content/drive/MyDrive/Datasets/TII-SSRC-23.csv'  # Adjust this path if needed

    # Step 4: Use the handler to load and process the data
    try:
        # Initialize the handler
        print("\n=== Initializing Dataset Handler ===")
        handler = LargeCSVHandler(dataset_path)
        
        # Analyze the file and determine loading strategy
        print("\n=== Analyzing Dataset ===")
        handler.check_file()
        all_columns = handler.analyze_columns()
        handler.count_rows()
        
        # Determine essential columns to reduce memory usage
        # Group 1: Core columns that we always need
        essential_cols = ['label', 'src_ip', 'dst_ip', 'timestamp', 'protocol']
        
        # Group 2: Use standard column mapping from your processor
        column_mapping = {
            'src_ip': 'Source IP',
            'dst_ip': 'Destination IP',
            'src_port': 'Source Port',
            'dst_port': 'Destination Port',
            'protocol': 'Protocol',
            'timestamp': 'Timestamp',
            'duration': 'Flow Duration',
            'total_fwd_packets': 'Total Fwd Packets',
            'total_bwd_packets': 'Total Backward Packets',
            'total_length_of_fwd_packets': 'Total Length of Fwd Packets',
            'total_length_of_bwd_packets': 'Total Length of Bwd Packets',
            'label': 'Label'
        }
        
        # Combine and filter to columns that exist in the dataset
        target_columns = list(set(essential_cols + list(column_mapping.keys())))
        target_columns = [col for col in target_columns if col in all_columns]
        
        # Add some additional feature columns if they exist
        for col in all_columns:
            if any(key in col.lower() for key in ['byte', 'packet', 'flow', 'port']):
                if col not in target_columns:
                    target_columns.append(col)
                    if len(target_columns) >= 30:  # Limit to ~30 total columns
                        break
        
        print(f"\nSelected {len(target_columns)} columns for loading:")
        print(target_columns)
        
        # Load the data with appropriate strategy
        print("\n=== Loading Dataset ===")
        df = handler.load_data(use_sampling=True, columns=target_columns)
        
        # Now process with your NetworkDataProcessor
        print("\n=== Processing Dataset ===")
        from src.data.data_processor import NetworkDataProcessor
        
        # Initialize processor
        processor = NetworkDataProcessor(scaling_method='standard', handle_imbalance=True)
        
        # Set target column
        processor.target_column = handler.target_column
        
        # Display basic dataset info
        print("\nDataset preview:")
        display(df.head())
        
        print("\nClass distribution:")
        display(df[handler.target_column].value_counts())
        
        # Process the data
        print("\nPreprocessing data...")
        X_train, X_val, X_test, y_train, y_val, y_test = processor.preprocess_data(
            df, 
            categorical_encoding='onehot',
            handle_imbalance_method='smote'
        )
        
        print("\nPreprocessing complete!")
        print(f"Training set: {X_train.shape}")
        print(f"Validation set: {X_val.shape}")
        print(f"Test set: {X_test.shape}")
        
        # Training data is now ready for your models
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc() 