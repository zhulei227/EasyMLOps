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


def test_tft():
    try:
        import torch
    except ImportError:
        print("\n" + "="*60)
        print("Testing TFTRegression (skipped - PyTorch not available)")
        print("="*60)
        return
        
    try:
        from easymlops.ts.transformer.transformer import TFTRegression
    except ImportError as e:
        print(f"\nTesting TFTRegression failed: {e}")
        return
    
    print("\n" + "="*60)
    print("Testing TFTRegression")
    print("="*60)
    
    df = generate_test_data(200)
    
    pipe = TFTRegression(input_cols=["value"], time_period=30, forecast_horizon=7, epochs=5)
    
    pipe.fit(df)
    result = pipe.transform(df)
    print(f"Predictions shape: {result.shape}")
    print(f"Predictions sample:\n{result[['value', 'tft_predict']].head()}")
    
    path = "tests/tft_model.pkl"
    pipe.save(path)
    pipe.load(path)
    result2 = pipe.transform(df)
    print(f"Loaded model predictions sample:\n{result2[['value', 'tft_predict']].head()}")
    
    if os.path.exists(path):
        os.remove(path)
    print("TFTRegression test passed!")


def test_informer():
    try:
        import torch
    except ImportError:
        print("\n" + "="*60)
        print("Testing InformerRegression (skipped - PyTorch not available)")
        print("="*60)
        return
        
    try:
        from easymlops.ts.transformer.transformer import InformerRegression
    except ImportError as e:
        print(f"\nTesting InformerRegression failed: {e}")
        return
    
    print("\n" + "="*60)
    print("Testing InformerRegression")
    print("="*60)
    
    df = generate_test_data(200)
    
    pipe = InformerRegression(input_cols=["value"], time_period=30, forecast_horizon=7, epochs=5)
    
    pipe.fit(df)
    result = pipe.transform(df)
    print(f"Predictions shape: {result.shape}")
    print(f"Predictions sample:\n{result[['value', 'informer_predict']].head()}")
    
    path = "tests/informer_model.pkl"
    pipe.save(path)
    pipe.load(path)
    result2 = pipe.transform(df)
    print(f"Loaded model predictions sample:\n{result2[['value', 'informer_predict']].head()}")
    
    if os.path.exists(path):
        os.remove(path)
    print("InformerRegression test passed!")


def test_autoformer():
    try:
        import torch
    except ImportError:
        print("\n" + "="*60)
        print("Testing AutoformerRegression (skipped - PyTorch not available)")
        print("="*60)
        return
        
    try:
        from easymlops.ts.transformer.transformer import AutoformerRegression
    except ImportError as e:
        print(f"\nTesting AutoformerRegression failed: {e}")
        return
    
    print("\n" + "="*60)
    print("Testing AutoformerRegression")
    print("="*60)
    
    df = generate_test_data(200)
    
    pipe = AutoformerRegression(input_cols=["value"], time_period=30, forecast_horizon=7, epochs=5)
    
    pipe.fit(df)
    result = pipe.transform(df)
    print(f"Predictions shape: {result.shape}")
    print(f"Predictions sample:\n{result[['value', 'autoformer_predict']].head()}")
    
    path = "tests/autoformer_model.pkl"
    pipe.save(path)
    pipe.load(path)
    result2 = pipe.transform(df)
    print(f"Loaded model predictions sample:\n{result2[['value', 'autoformer_predict']].head()}")
    
    if os.path.exists(path):
        os.remove(path)
    print("AutoformerRegression test passed!")


def test_fedformer():
    try:
        import torch
    except ImportError:
        print("\n" + "="*60)
        print("Testing FEDformerRegression (skipped - PyTorch not available)")
        print("="*60)
        return
        
    try:
        from easymlops.ts.transformer.transformer import FEDformerRegression
    except ImportError as e:
        print(f"\nTesting FEDformerRegression failed: {e}")
        return
    
    print("\n" + "="*60)
    print("Testing FEDformerRegression")
    print("="*60)
    
    df = generate_test_data(200)
    
    pipe = FEDformerRegression(input_cols=["value"], time_period=30, forecast_horizon=7, epochs=5)
    
    pipe.fit(df)
    result = pipe.transform(df)
    print(f"Predictions shape: {result.shape}")
    print(f"Predictions sample:\n{result[['value', 'fedformer_predict']].head()}")
    
    path = "tests/fedformer_model.pkl"
    pipe.save(path)
    pipe.load(path)
    result2 = pipe.transform(df)
    print(f"Loaded model predictions sample:\n{result2[['value', 'fedformer_predict']].head()}")
    
    if os.path.exists(path):
        os.remove(path)
    print("FEDformerRegression test passed!")


def test_patchtst():
    try:
        import torch
    except ImportError:
        print("\n" + "="*60)
        print("Testing PatchTSTRegression (skipped - PyTorch not available)")
        print("="*60)
        return
        
    try:
        from easymlops.ts.transformer.transformer import PatchTSTRegression
    except ImportError as e:
        print(f"\nTesting PatchTSTRegression failed: {e}")
        return
    
    print("\n" + "="*60)
    print("Testing PatchTSTRegression")
    print("="*60)
    
    df = generate_test_data(200)
    
    pipe = PatchTSTRegression(input_cols=["value"], time_period=30, forecast_horizon=7, epochs=5)
    
    pipe.fit(df)
    result = pipe.transform(df)
    print(f"Predictions shape: {result.shape}")
    print(f"Predictions sample:\n{result[['value', 'patchtst_predict']].head()}")
    
    path = "tests/patchtst_model.pkl"
    pipe.save(path)
    pipe.load(path)
    result2 = pipe.transform(df)
    print(f"Loaded model predictions sample:\n{result2[['value', 'patchtst_predict']].head()}")
    
    if os.path.exists(path):
        os.remove(path)
    print("PatchTSTRegression test passed!")


def test_timesnet():
    try:
        import torch
    except ImportError:
        print("\n" + "="*60)
        print("Testing TimesNetRegression (skipped - PyTorch not available)")
        print("="*60)
        return
        
    try:
        from easymlops.ts.transformer.transformer import TimesNetRegression
    except ImportError as e:
        print(f"\nTesting TimesNetRegression failed: {e}")
        return
    
    print("\n" + "="*60)
    print("Testing TimesNetRegression")
    print("="*60)
    
    df = generate_test_data(200)
    
    pipe = TimesNetRegression(input_cols=["value"], time_period=30, forecast_horizon=7, epochs=5)
    
    pipe.fit(df)
    result = pipe.transform(df)
    print(f"Predictions shape: {result.shape}")
    print(f"Predictions sample:\n{result[['value', 'timesnet_predict']].head()}")
    
    path = "tests/timesnet_model.pkl"
    pipe.save(path)
    pipe.load(path)
    result2 = pipe.transform(df)
    print(f"Loaded model predictions sample:\n{result2[['value', 'timesnet_predict']].head()}")
    
    if os.path.exists(path):
        os.remove(path)
    print("TimesNetRegression test passed!")


def test_itransformer():
    try:
        import torch
    except ImportError:
        print("\n" + "="*60)
        print("Testing iTransformerRegression (skipped - PyTorch not available)")
        print("="*60)
        return
        
    try:
        from easymlops.ts.transformer.transformer import iTransformerRegression
    except ImportError as e:
        print(f"\nTesting iTransformerRegression failed: {e}")
        return
    
    print("\n" + "="*60)
    print("Testing iTransformerRegression")
    print("="*60)
    
    df = generate_test_data(200)
    
    pipe = iTransformerRegression(input_cols=["value"], time_period=30, forecast_horizon=7, epochs=5)
    
    pipe.fit(df)
    result = pipe.transform(df)
    print(f"Predictions shape: {result.shape}")
    print(f"Predictions sample:\n{result[['value', 'itransformer_predict']].head()}")
    
    path = "tests/itransformer_model.pkl"
    pipe.save(path)
    pipe.load(path)
    result2 = pipe.transform(df)
    print(f"Loaded model predictions sample:\n{result2[['value', 'itransformer_predict']].head()}")
    
    if os.path.exists(path):
        os.remove(path)
    print("iTransformerRegression test passed!")


if __name__ == "__main__":
    print("Running transformer model tests...")
        
    test_tft()
    test_informer()
    test_autoformer()
    test_fedformer()
    test_patchtst()
    test_timesnet()
    test_itransformer()
    
    print("\n" + "="*60)
    print("Transformer models tests completed!")
    print("="*60)
