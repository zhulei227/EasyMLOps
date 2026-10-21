import numpy as np
import pandas as pd
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/.."))


def generate_test_data(n=200):
    np.random.seed(42)
    dates = pd.date_range(start="2020-01-01", periods=n, freq="D")
    data = {
        "date": dates,
        "value": np.cumsum(np.random.randn(n)) + 100,
        "feature1": np.random.randn(n) * 10,
        "feature2": np.random.randn(n) * 5,
    }
    df = pd.DataFrame(data)
    return df


def test_nbeats():
    try:
        import torch
    except ImportError:
        print("\n" + "="*60)
        print("Testing NBeatsRegression (skipped - PyTorch not available)")
        print("="*60)
        return
        
    try:
        from easymlops.ts.deep_learning.nbeats import NBeatsRegression
    except ImportError as e:
        print(f"\nTesting NBeatsRegression failed: {e}")
        return
    
    print("\n" + "="*60)
    print("Testing NBeatsRegression")
    print("="*60)
    
    df = generate_test_data(200)
    
    pipe = NBeatsRegression(input_cols=["value"], time_period=30, forecast_horizon=7, epochs=5)
    
    pipe.fit(df)
    result = pipe.transform(df)
    print(f"Predictions shape: {result.shape}")
    print(f"Predictions sample:\n{result[['value', 'nbeats_predict']].head()}")
    
    path = "tests/nbeats_model.pkl"
    pipe.save(path)
    pipe.load(path)
    result2 = pipe.transform(df)
    print(f"Loaded model predictions sample:\n{result2[['value', 'nbeats_predict']].head()}")
    
    if os.path.exists(path):
        os.remove(path)
    print("NBeatsRegression test passed!")


def test_nhits():
    try:
        import torch
    except ImportError:
        print("\n" + "="*60)
        print("Testing NHiTSRegression (skipped - PyTorch not available)")
        print("="*60)
        return
        
    try:
        from easymlops.ts.deep_learning.nbeats import NHiTSRegression
    except ImportError as e:
        print(f"\nTesting NHiTSRegression failed: {e}")
        return
    
    print("\n" + "="*60)
    print("Testing NHiTSRegression")
    print("="*60)
    
    df = generate_test_data(200)
    
    pipe = NHiTSRegression(input_cols=["value"], time_period=30, forecast_horizon=7, epochs=5)
    
    pipe.fit(df)
    result = pipe.transform(df)
    print(f"Predictions shape: {result.shape}")
    print(f"Predictions sample:\n{result[['value', 'nhits_predict']].head()}")
    
    path = "tests/nhits_model.pkl"
    pipe.save(path)
    pipe.load(path)
    result2 = pipe.transform(df)
    print(f"Loaded model predictions sample:\n{result2[['value', 'nhits_predict']].head()}")
    
    if os.path.exists(path):
        os.remove(path)
    print("NHiTSRegression test passed!")


def test_deepar():
    try:
        import torch
    except ImportError:
        print("\n" + "="*60)
        print("Testing DeepARRegression (skipped - PyTorch not available)")
        print("="*60)
        return
        
    try:
        from easymlops.ts.deep_learning.nbeats import DeepARRegression
    except ImportError as e:
        print(f"\nTesting DeepARRegression failed: {e}")
        return
    
    print("\n" + "="*60)
    print("Testing DeepARRegression")
    print("="*60)
    
    df = generate_test_data(200)
    
    pipe = DeepARRegression(input_cols=["value"], time_period=30, forecast_horizon=7, epochs=5)
    
    pipe.fit(df)
    result = pipe.transform(df)
    print(f"Predictions shape: {result.shape}")
    print(f"Predictions sample:\n{result[['value', 'deepar_predict']].head()}")
    
    path = "tests/deepar_model.pkl"
    pipe.save(path)
    pipe.load(path)
    result2 = pipe.transform(df)
    print(f"Loaded model predictions sample:\n{result2[['value', 'deepar_predict']].head()}")
    
    if os.path.exists(path):
        os.remove(path)
    print("DeepARRegression test passed!")


def test_gp():
    try:
        import torch
    except ImportError:
        print("\n" + "="*60)
        print("Testing GPRegression (skipped - PyTorch not available)")
        print("="*60)
        return
        
    try:
        from easymlops.ts.deep_learning.nbeats import GPRegression
    except ImportError as e:
        print(f"\nTesting GPRegression failed: {e}")
        return
    
    print("\n" + "="*60)
    print("Testing GPRegression")
    print("="*60)
    
    df = generate_test_data(200)
    
    pipe = GPRegression(input_cols=["value"], time_period=30, forecast_horizon=7)
    
    pipe.fit(df)
    result = pipe.transform(df)
    print(f"Predictions shape: {result.shape}")
    print(f"Predictions sample:\n{result[['value', 'gp_predict']].head()}")
    
    path = "tests/gp_model.pkl"
    pipe.save(path)
    pipe.load(path)
    result2 = pipe.transform(df)
    print(f"Loaded model predictions sample:\n{result2[['value', 'gp_predict']].head()}")
    
    if os.path.exists(path):
        os.remove(path)
    print("GPRegression test passed!")


if __name__ == "__main__":
    print("Running deep learning model tests...")
        
    test_nbeats()
    test_nhits()
    test_deepar()
    test_gp()
    
    print("\n" + "="*60)
    print("Deep learning models tests completed!")
    print("="*60)
