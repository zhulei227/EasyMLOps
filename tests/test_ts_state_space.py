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


def test_deepstate():
    try:
        import torch
    except ImportError:
        print("\n" + "="*60)
        print("Testing DeepStateRegression (skipped - PyTorch not available)")
        print("="*60)
        return
        
    try:
        from easymlops.ts.state_space.state_space import DeepStateRegression
    except ImportError as e:
        print(f"\nTesting DeepStateRegression failed: {e}")
        return
    
    print("\n" + "="*60)
    print("Testing DeepStateRegression")
    print("="*60)
    
    df = generate_test_data(200)
    
    pipe = DeepStateRegression(input_cols=["value"], time_period=30, forecast_horizon=7, epochs=5)
    
    pipe.fit(df)
    result = pipe.transform(df)
    print(f"Predictions shape: {result.shape}")
    print(f"Predictions sample:\n{result[['value', 'deepstate_predict']].head()}")
    
    path = "tests/deepstate_model.pkl"
    pipe.save(path)
    pipe.load(path)
    result2 = pipe.transform(df)
    print(f"Loaded model predictions sample:\n{result2[['value', 'deepstate_predict']].head()}")
    
    if os.path.exists(path):
        os.remove(path)
    print("DeepStateRegression test passed!")


def test_mamba():
    try:
        import torch
    except ImportError:
        print("\n" + "="*60)
        print("Testing MambaRegression (skipped - PyTorch not available)")
        print("="*60)
        return
        
    try:
        from easymlops.ts.state_space.state_space import MambaRegression
    except ImportError as e:
        print(f"\nTesting MambaRegression failed: {e}")
        return
    
    print("\n" + "="*60)
    print("Testing MambaRegression")
    print("="*60)
    
    df = generate_test_data(200)
    
    pipe = MambaRegression(input_cols=["value"], time_period=30, forecast_horizon=7, epochs=5)
    
    pipe.fit(df)
    result = pipe.transform(df)
    print(f"Predictions shape: {result.shape}")
    print(f"Predictions sample:\n{result[['value', 'mamba_predict']].head()}")
    
    path = "tests/mamba_model.pkl"
    pipe.save(path)
    pipe.load(path)
    result2 = pipe.transform(df)
    print(f"Loaded model predictions sample:\n{result2[['value', 'mamba_predict']].head()}")
    
    if os.path.exists(path):
        os.remove(path)
    print("MambaRegression test passed!")


def test_liquids4():
    try:
        import torch
    except ImportError:
        print("\n" + "="*60)
        print("Testing LiquidS4Regression (skipped - PyTorch not available)")
        print("="*60)
        return
        
    try:
        from easymlops.ts.state_space.state_space import LiquidS4Regression
    except ImportError as e:
        print(f"\nTesting LiquidS4Regression failed: {e}")
        return
    
    print("\n" + "="*60)
    print("Testing LiquidS4Regression")
    print("="*60)
    
    df = generate_test_data(200)
    
    pipe = LiquidS4Regression(input_cols=["value"], time_period=30, forecast_horizon=7, epochs=5)
    
    pipe.fit(df)
    result = pipe.transform(df)
    print(f"Predictions shape: {result.shape}")
    print(f"Predictions sample:\n{result[['value', 'liquids4_predict']].head()}")
    
    path = "tests/liquids4_model.pkl"
    pipe.save(path)
    pipe.load(path)
    result2 = pipe.transform(df)
    print(f"Loaded model predictions sample:\n{result2[['value', 'liquids4_predict']].head()}")
    
    if os.path.exists(path):
        os.remove(path)
    print("LiquidS4Regression test passed!")


if __name__ == "__main__":
    print("Running state space model tests...")
        
    test_deepstate()
    test_mamba()
    test_liquids4()
    
    print("\n" + "="*60)
    print("State space models tests completed!")
    print("="*60)
