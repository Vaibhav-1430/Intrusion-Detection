import numpy as np
import pandas as pd
from pathlib import Path

def generate_sample_network_data(n_samples=10000):
    """Generate synthetic network traffic data for demonstration."""
    np.random.seed(42)
    
    # Generate basic network features
    data = {
        'src_ip': np.random.choice(['192.168.1.1', '192.168.1.2', '10.0.0.1', '10.0.0.2', '172.16.0.1'], n_samples),
        'dst_ip': np.random.choice(['192.168.1.100', '192.168.1.200', '10.0.0.100', '8.8.8.8', '1.1.1.1'], n_samples),
        'src_port': np.random.randint(1024, 65535, n_samples),
        'dst_port': np.random.choice([80, 443, 22, 21, 25, 53, 3389, 8080], n_samples),
        'protocol': np.random.choice(['TCP', 'UDP', 'ICMP'], n_samples),
        'packet_size': np.random.exponential(1000, n_samples).astype(int),
        'flags': np.random.choice(['SYN', 'ACK', 'FIN', 'RST', 'PSH'], n_samples),
        'duration': np.random.exponential(1.0, n_samples),
        'packet_count': np.random.poisson(10, n_samples),
        'byte_count': np.random.exponential(5000, n_samples).astype(int),
    }
    
    df = pd.DataFrame(data)
    
    # Create labels: 0 = normal, 1 = attack
    # Simulate some attack patterns
    attack_mask = np.random.random(n_samples) < 0.15  # 15% attacks
    
    # Make attacks have different characteristics
    df.loc[attack_mask, 'packet_size'] = np.random.exponential(200, attack_mask.sum()).astype(int)
    df.loc[attack_mask, 'duration'] = np.random.exponential(0.1, attack_mask.sum())
    df.loc[attack_mask, 'packet_count'] = np.random.poisson(100, attack_mask.sum())
    df.loc[attack_mask, 'dst_port'] = np.random.choice([22, 3389, 23], attack_mask.sum())  # SSH, RDP, Telnet
    
    df['label'] = attack_mask.astype(int)
    
    return df

def main():
    # Create data directory
    data_dir = Path("data/NSL-KDD")
    data_dir.mkdir(parents=True, exist_ok=True)
    print(f"Created directory: {data_dir}")
    
    # Generate sample data
    print("Generating sample network traffic data...")
    df = generate_sample_network_data(10000)
    print(f"Generated dataframe with shape: {df.shape}")
    
    # Split into train/test like NSL-KDD
    train_size = int(0.8 * len(df))
    train_df = df[:train_size]
    test_df = df[train_size:]
    
    # Save as CSV files
    train_path = data_dir / "KDDTrain+.csv"
    test_path = data_dir / "KDDTest+.csv"
    
    print(f"Saving train data to: {train_path}")
    train_df.to_csv(train_path, index=False)
    print(f"Saving test data to: {test_path}")
    test_df.to_csv(test_path, index=False)
    
    print(f"Generated {len(df)} samples ({len(train_df)} train, {len(test_df)} test)")
    print(f"Attack rate: {df['label'].mean():.2%}")
    print(f"Files saved successfully!")
    
    # Verify files exist
    print(f"Train file exists: {train_path.exists()}")
    print(f"Test file exists: {test_path.exists()}")

if __name__ == "__main__":
    main()
