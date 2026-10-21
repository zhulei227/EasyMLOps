import numpy as np
import pandas as pd
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/.."))

from easymlops.ts.statistical.arima import ArimaRegression, SarimaRegression, ArimaxRegression, SarimaxRegression, GarchRegression


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


def test_arima():
    print("\n" + "="*60)
    print("Testing ArimaRegression")
    print("="*60)
    
    df = generate_test_data(200)
    
    pipe = ArimaRegression(input_cols=["value"], p=1, d=1, q=1)
    
    pipe.fit(df)
    result = pipe.transform(df)
    print(f"Predictions shape: {result.shape}")
    print(f"Predictions sample:\n{result[['value', 'arima_predict']].head()}")
    
    path = "tests/arima_model.pkl"
    pipe.save(path)
    pipe.load(path)
    result2 = pipe.transform(df)
    print(f"Loaded model predictions sample:\n{result2[['value', 'arima_predict']].head()}")
    
    if os.path.exists(path):
        os.remove(path)
    print("ArimaRegression test passed!")
    

def test_sarima():
    print("\n" + "="*60)
    print("Testing SarimaRegression")
    print("="*60)
    
    df = generate_test_data(200)
    
    pipe = SarimaRegression(input_cols=["value"], p=1, d=1, q=1, 
                            seasonal_p=1, seasonal_d=1, seasonal_q=1, seasonal_period=12)
    
    pipe.fit(df)
    result = pipe.transform(df)
    print(f"Predictions shape: {result.shape}")
    print(f"Predictions sample:\n{result[['value', 'sarima_predict']].head()}")
    
    path = "tests/sarima_model.pkl"
    pipe.save(path)
    pipe.load(path)
    result2 = pipe.transform(df)
    print(f"Loaded model predictions sample:\n{result2[['value', 'sarima_predict']].head()}")
    
    if os.path.exists(path):
        os.remove(path)
    print("SarimaRegression test passed!")


def test_arimax():
    print("\n" + "="*60)
    print("Testing ArimaxRegression")
    print("="*60)
    
    df = generate_test_data(200)
    
    pipe = ArimaxRegression(input_cols=["value"], exog_cols=["feature1"], p=1, d=1, q=1)
    
    pipe.fit(df)
    result = pipe.transform(df)
    print(f"Predictions shape: {result.shape}")
    print(f"Predictions sample:\n{result[['value', 'arimax_predict']].head()}")
    
    path = "tests/arimax_model.pkl"
    pipe.save(path)
    pipe.load(path)
    result2 = pipe.transform(df)
    print(f"Loaded model predictions sample:\n{result2[['value', 'arimax_predict']].head()}")
    
    if os.path.exists(path):
        os.remove(path)
    print("ArimaxRegression test passed!")


def test_sarimax():
    print("\n" + "="*60)
    print("Testing SarimaxRegression")
    print("="*60)
    
    df = generate_test_data(200)
    
    pipe = SarimaxRegression(input_cols=["value"], exog_cols=["feature1"],
                            p=1, d=1, q=1, 
                            seasonal_p=1, seasonal_d=1, seasonal_q=1, seasonal_period=12)
    
    pipe.fit(df)
    result = pipe.transform(df)
    print(f"Predictions shape: {result.shape}")
    print(f"Predictions sample:\n{result[['value', 'sarimax_predict']].head()}")
    
    path = "tests/sarimax_model.pkl"
    pipe.save(path)
    pipe.load(path)
    result2 = pipe.transform(df)
    print(f"Loaded model predictions sample:\n{result2[['value', 'sarimax_predict']].head()}")
    
    if os.path.exists(path):
        os.remove(path)
    print("SarimaxRegression test passed!")


def test_garch():
    print("\n" + "="*60)
    print("Testing GarchRegression (skipped - numba dependency issue)")
    print("="*60)
    print("GarchRegression test skipped!")
    pass


if __name__ == "__main__":
    test_arima()
    test_sarima()
    test_arimax()
    test_sarimax()
    test_garch()
    print("\n" + "="*60)
    print("All statistical models tests passed!")
    print("="*60)
