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


def test_vae():
    try:
        import torch
    except ImportError:
        print("\n" + "="*60)
        print("Testing VAERegression (skipped - PyTorch not available)")
        print("="*60)
        return
        
    try:
        from easymlops.ts.generative.generative import VAERegression
    except ImportError as e:
        print(f"\nTesting VAERegression failed: {e}")
        return
    
    print("\n" + "="*60)
    print("Testing VAERegression")
    print("="*60)
    
    df = generate_test_data(200)
    
    pipe = VAERegression(input_cols=["value"], time_period=30, forecast_horizon=7, epochs=5)
    
    pipe.fit(df)
    result = pipe.transform(df)
    print(f"Predictions shape: {result.shape}")
    print(f"Predictions sample:\n{result[['value', 'vae_predict']].head()}")
    
    path = "tests/vae_model.pkl"
    pipe.save(path)
    pipe.load(path)
    result2 = pipe.transform(df)
    print(f"Loaded model predictions sample:\n{result2[['value', 'vae_predict']].head()}")
    
    if os.path.exists(path):
        os.remove(path)
    print("VAERegression test passed!")


def test_normalizing_flow():
    try:
        import torch
    except ImportError:
        print("\n" + "="*60)
        print("Testing NormalizingFlowRegression (skipped - PyTorch not available)")
        print("="*60)
        return
        
    try:
        from easymlops.ts.generative.generative import NormalizingFlowRegression
    except ImportError as e:
        print(f"\nTesting NormalizingFlowRegression failed: {e}")
        return
    
    print("\n" + "="*60)
    print("Testing NormalizingFlowRegression")
    print("="*60)
    
    df = generate_test_data(200)
    
    pipe = NormalizingFlowRegression(input_cols=["value"], time_period=30, forecast_horizon=7, epochs=5)
    
    pipe.fit(df)
    result = pipe.transform(df)
    print(f"Predictions shape: {result.shape}")
    print(f"Predictions sample:\n{result[['value', 'normalizingflow_predict']].head()}")
    
    path = "tests/normalizingflow_model.pkl"
    pipe.save(path)
    pipe.load(path)
    result2 = pipe.transform(df)
    print(f"Loaded model predictions sample:\n{result2[['value', 'normalizingflow_predict']].head()}")
    
    if os.path.exists(path):
        os.remove(path)
    print("NormalizingFlowRegression test passed!")


def test_diffusion():
    try:
        import torch
    except ImportError:
        print("\n" + "="*60)
        print("Testing DiffusionRegression (skipped - PyTorch not available)")
        print("="*60)
        return
        
    try:
        from easymlops.ts.generative.generative import DiffusionRegression
    except ImportError as e:
        print(f"\nTesting DiffusionRegression failed: {e}")
        return
    
    print("\n" + "="*60)
    print("Testing DiffusionRegression")
    print("="*60)
    
    df = generate_test_data(200)
    
    pipe = DiffusionRegression(input_cols=["value"], time_period=30, forecast_horizon=7, epochs=5)
    
    pipe.fit(df)
    result = pipe.transform(df)
    print(f"Predictions shape: {result.shape}")
    print(f"Predictions sample:\n{result[['value', 'diffusion_predict']].head()}")
    
    path = "tests/diffusion_model.pkl"
    pipe.save(path)
    pipe.load(path)
    result2 = pipe.transform(df)
    print(f"Loaded model predictions sample:\n{result2[['value', 'diffusion_predict']].head()}")
    
    if os.path.exists(path):
        os.remove(path)
    print("DiffusionRegression test passed!")


if __name__ == "__main__":
    print("Running generative model tests...")
        
    test_vae()
    test_normalizing_flow()
    test_diffusion()
    
    print("\n" + "="*60)
    print("Generative models tests completed!")
    print("="*60)
