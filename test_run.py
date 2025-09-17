import os
import pandas as pd
import numpy as np

print("Simple IDS Test Script")
print("=====================")

# Check if data exists
data_path = os.path.join("data", "NSL-KDD")
train_file = os.path.join(data_path, "KDDTrain+.csv")
test_file = os.path.join(data_path, "KDDTest+.csv")

if os.path.exists(train_file) and os.path.exists(test_file):
    print(f"✓ Data files found in {data_path}")
    
    # Load a sample of the data
    try:
        train_data = pd.read_csv(train_file, nrows=5)
        print("\nSample data preview:")
        print(train_data.head())
        print(f"\nTotal columns: {train_data.shape[1]}")
        
        # Simple analysis
        print("\nProject is ready to run!")
        print("You can now try the following commands:")
        print("1. python scripts/train_supervised.py - To train a model")
        print("2. python scripts/run_realtime.py - To run real-time detection")
        print("3. streamlit run app/dashboard.py - To launch the dashboard")
    except Exception as e:
        print(f"Error loading data: {e}")
else:
    print(f"✗ Data files not found in {data_path}")
    print("Running data generation script...")
    
    # Create directories if they don't exist
    os.makedirs(data_path, exist_ok=True)
    
    # Generate simple sample data
    print("Generating sample data...")
    
    # Create simple dataset with 1000 samples
    np.random.seed(42)
    n_samples = 1000
    n_features = 10
    
    # Generate features
    X = np.random.randn(n_samples, n_features)
    
    # Generate labels (0: normal, 1: attack)
    y = np.random.choice([0, 1], size=n_samples, p=[0.85, 0.15])
    
    # Create DataFrame
    columns = [f"feature_{i}" for i in range(n_features)]
    columns.append("label")
    
    # Combine features and labels
    data = np.column_stack((X, y))
    
    # Create DataFrame
    df = pd.DataFrame(data, columns=columns)
    
    # Split into train and test
    train_size = int(0.8 * n_samples)
    train_df = df[:train_size]
    test_df = df[train_size:]
    
    # Save to CSV
    train_df.to_csv(train_file, index=False)
    test_df.to_csv(test_file, index=False)
    
    print(f"✓ Sample data generated and saved to {data_path}")
    print(f"  - Training samples: {train_df.shape[0]}")
    print(f"  - Test samples: {test_df.shape[0]}")
    print(f"  - Features: {n_features}")
    print(f"  - Attack rate: {y.mean():.2%}")
    
    print("\nSample data preview:")
    print(train_df.head())
    
    print("\nProject is ready to run!")
    print("You can now try the following commands:")
    print("1. python scripts/train_supervised.py - To train a model")
    print("2. python scripts/run_realtime.py - To run real-time detection")
    print("3. streamlit run app/dashboard.py - To launch the dashboard")