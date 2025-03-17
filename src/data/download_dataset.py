"""
Script to download a sample dataset for testing the UBA system.
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
import requests
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config


def download_file(url, destination):
    """
    Download a file from a URL to a destination path with progress bar.
    
    Args:
        url (str): URL to download from
        destination (str): Path to save the file
    """
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 Kibibyte
    
    with open(destination, 'wb') as f, tqdm(
        desc=os.path.basename(destination),
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(block_size):
            size = f.write(data)
            bar.update(size)


def download_cicids2017_sample():
    """
    Download a sample of the CICIDS2017 dataset.
    """
    # Create data directory if it doesn't exist
    os.makedirs(config.DATA_DIR, exist_ok=True)
    
    # Check if the dataset already exists
    dataset_path = os.path.join(config.DATA_DIR, 'cicids2017_sample.csv')
    
    if os.path.exists(dataset_path):
        print(f"Dataset already exists at {dataset_path}")
        return dataset_path
    
    # URL for the sample dataset
    # Note: This is a placeholder URL. In a real scenario, you would provide the actual URL.
    url = "https://example.com/cicids2017_sample.csv"
    
    try:
        print(f"Downloading CICIDS2017 sample dataset to {dataset_path}...")
        download_file(url, dataset_path)
        print("Download completed!")
        return dataset_path
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("Generating synthetic dataset instead...")
        return generate_synthetic_dataset()


def generate_synthetic_dataset():
    """
    Generate a synthetic network traffic dataset for testing.
    
    Returns:
        str: Path to the generated dataset
    """
    print("Generating synthetic network traffic dataset...")
    
    # Create data directory if it doesn't exist
    os.makedirs(config.DATA_DIR, exist_ok=True)
    
    # Path for the synthetic dataset
    dataset_path = os.path.join(config.DATA_DIR, 'synthetic_network_traffic.csv')
    
    # Generate synthetic features and labels
    X, y = make_classification(
        n_samples=10000,
        n_features=20,
        n_informative=10,
        n_redundant=5,
        n_classes=2,
        weights=[0.95, 0.05],  # 5% anomalies
        random_state=42
    )
    
    # Create a dataframe
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    
    # Add labels (1 for normal, 0 for anomaly)
    df['Label'] = y
    
    # Add synthetic network traffic features
    n_samples = len(df)
    
    # Source and destination IPs
    source_ips = np.random.choice([f'192.168.1.{i}' for i in range(1, 101)], n_samples)
    dest_ips = np.random.choice([f'10.0.0.{i}' for i in range(1, 101)], n_samples)
    
    # Ports
    source_ports = np.random.randint(1024, 65535, n_samples)
    dest_ports = np.random.choice([80, 443, 22, 21, 25, 53], n_samples)
    
    # Protocol
    protocols = np.random.choice(['TCP', 'UDP', 'ICMP'], n_samples, p=[0.7, 0.2, 0.1])
    
    # Timestamps
    start_time = pd.Timestamp('2023-01-01')
    timestamps = [start_time + pd.Timedelta(seconds=i) for i in range(n_samples)]
    
    # Flow duration
    flow_durations = np.random.exponential(scale=1.0, size=n_samples) * 1000  # in ms
    
    # Packet counts
    fwd_packets = np.random.poisson(lam=10, size=n_samples)
    bwd_packets = np.random.poisson(lam=8, size=n_samples)
    
    # Packet lengths
    fwd_packet_lengths = np.random.exponential(scale=500, size=n_samples)
    bwd_packet_lengths = np.random.exponential(scale=400, size=n_samples)
    
    # Add the network traffic features to the dataframe
    df['Source IP'] = source_ips
    df['Destination IP'] = dest_ips
    df['Source Port'] = source_ports
    df['Destination Port'] = dest_ports
    df['Protocol'] = protocols
    df['Timestamp'] = timestamps
    df['Flow Duration'] = flow_durations
    df['Total Fwd Packets'] = fwd_packets
    df['Total Backward Packets'] = bwd_packets
    df['Total Length of Fwd Packets'] = fwd_packet_lengths * fwd_packets
    df['Total Length of Bwd Packets'] = bwd_packet_lengths * bwd_packets
    
    # Introduce some anomalous behavior for specific source IPs
    anomalous_ips = np.random.choice([f'192.168.1.{i}' for i in range(1, 101)], 5, replace=False)
    
    for ip in anomalous_ips:
        # Get indices for this IP
        indices = df[df['Source IP'] == ip].index
        
        if len(indices) > 0:
            # Modify some features to make them anomalous
            df.loc[indices, 'Destination Port'] = np.random.choice([1337, 4444, 8080], len(indices))
            df.loc[indices, 'Total Fwd Packets'] = np.random.poisson(lam=50, size=len(indices))
            df.loc[indices, 'Total Length of Fwd Packets'] = np.random.exponential(scale=2000, size=len(indices))
    
    # Save the dataset
    df.to_csv(dataset_path, index=False)
    
    print(f"Synthetic dataset generated and saved to {dataset_path}")
    print(f"Dataset shape: {df.shape}")
    print(f"Class distribution: {df['Label'].value_counts().to_dict()}")
    
    return dataset_path


def parse_arguments():
    """
    Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description='Download or generate a dataset for the UBA system')
    
    parser.add_argument('--dataset', type=str, default='synthetic',
                        choices=['cicids2017', 'synthetic'],
                        help='Dataset to download or generate')
    
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save the dataset (defaults to config.DATA_DIR)')
    
    return parser.parse_args()


def main():
    """
    Main function.
    """
    args = parse_arguments()
    
    if args.output_dir:
        config.DATA_DIR = args.output_dir
    
    if args.dataset == 'cicids2017':
        dataset_path = download_cicids2017_sample()
    else:
        dataset_path = generate_synthetic_dataset()
    
    print(f"Dataset ready at: {dataset_path}")


if __name__ == "__main__":
    main() 