# Network UBA (User and Entity Behavior Analytics) ML Model

A machine learning system for detecting malicious network traffic with reduced false positives using behavioral analytics.

## Project Overview

This project implements a User and Entity Behavior Analytics (UBA) system that uses machine learning to detect anomalous and potentially malicious network traffic. The system focuses on reducing false positives while maintaining high detection rates by establishing behavioral baselines and identifying deviations from normal patterns.

## Key Features

- Behavioral baseline establishment for network entities
- Anomaly detection with reduced false positives
- Multiple ML models including isolation forests, autoencoders, and ensemble methods
- Temporal analysis of network traffic patterns
- Explainable AI components for alert justification
- Adaptive thresholding to minimize false positives

## Project Structure

```
├── src/
│   ├── data/           # Data processing and loading modules
│   ├── models/         # ML model implementations
│   ├── utils/          # Utility functions
│   ├── visualization/  # Visualization tools
│   └── config/         # Configuration files
├── notebooks/          # Jupyter notebooks for exploration and analysis
├── requirements.txt    # Project dependencies
├── run_uba_demo.py     # Demo script to run the UBA system
└── README.md           # Project documentation
```

## Getting Started

### Prerequisites

- Python 3.8+
- pip package manager

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/375_project.git
   cd 375_project
   ```

2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

### Running the Demo

The project includes a demo script that generates synthetic network traffic data and runs the UBA system:

```
python run_uba_demo.py --generate_data --enable_profiling
```

Options:
- `--generate_data`: Generate synthetic network traffic data
- `--data_path PATH`: Use an existing dataset at PATH
- `--output_dir DIR`: Save results to DIR (default: results)
- `--model_type TYPE`: Use a specific model type (isolation_forest, one_class_svm, local_outlier_factor, ensemble)
- `--enable_profiling`: Enable behavioral profiling

### Using Your Own Data

To use your own network traffic data, ensure it has the following columns:
- Source IP: IP address of the source
- Destination IP: IP address of the destination
- Source Port: Port number of the source
- Destination Port: Port number of the destination
- Protocol: Network protocol (e.g., TCP, UDP)
- Timestamp: Time of the traffic
- Flow Duration: Duration of the flow in milliseconds
- Total Fwd Packets: Number of packets in forward direction
- Total Backward Packets: Number of packets in backward direction
- Total Length of Fwd Packets: Total size of packets in forward direction
- Total Length of Bwd Packets: Total size of packets in backward direction
- Label: Traffic label (1 for normal, 0 for anomaly)

Then run:
```
python run_uba_demo.py --data_path /path/to/your/data.csv --enable_profiling
```

## Implementation Details

### Anomaly Detection Models

The system includes several anomaly detection models:

1. **Isolation Forest**: Effective for detecting outliers in high-dimensional data
2. **One-Class SVM**: Good for complex decision boundaries
3. **Local Outlier Factor**: Identifies local deviations in density
4. **Ensemble Model**: Combines multiple models for improved accuracy

### Behavioral Profiling

The behavioral profiling component:

1. Creates time-based features for each network entity
2. Builds behavioral profiles based on historical patterns
3. Detects anomalous entities using statistical methods
4. Clusters entities with similar behavior patterns

### False Positive Reduction

The system reduces false positives through:

1. Adaptive thresholding based on validation data
2. Combining anomaly detection with behavioral profiling
3. Entity-based context for alerts
4. Temporal analysis of behavior patterns

## Results

The system outputs:
- Evaluation metrics (accuracy, precision, recall, F1, false positive rate)
- Visualizations of model performance
- List of detected anomalies
- Entity behavior profiles
- Entity clusters

Results are saved to the specified output directory.

## License

[Specify your license here]

## Acknowledgments

- [List any acknowledgments, datasets, or papers that influenced this work]