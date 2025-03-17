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


if __name__ == "__main__":
    # Example usage
    processor = NetworkDataProcessor(scaling_method='standard', handle_imbalance=True)
    
    # This is just a placeholder - in a real scenario, you would provide the actual file path
    # df = processor.load_data(os.path.join(config.DATA_DIR, "network_traffic.csv"))
    
    print("Data processor module created successfully") 