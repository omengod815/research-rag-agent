from pathlib import Path
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from langchain_core.tools import tool

SAFE_DIR = Path("data/public").resolve()
SAFE_DIR.mkdir(parents=True, exist_ok=True)

def analyze_regression_csv_impl(filename: str) -> str:
    path = (SAFE_DIR / filename).resolve()
    if SAFE_DIR not in path.parents:
        raise ValueError("Unsafe path")
    if not path.exists() or path.suffix.lower() != ".csv":
        raise FileNotFoundError("CSV not found")

    df = pd.read_csv(path)
    
    # 兼容多种常见的真实值和预测值列名
    true_candidates = ["true", "y_true", "actual", "label", "target"]
    pred_candidates = ["prediction", "y_pred", "pred", "predicted", "forecast"]
    
    true_col = next((col for col in true_candidates if col in df.columns), None)
    pred_col = next((col for col in pred_candidates if col in df.columns), None)
    
    if not true_col or not pred_col:
        raise ValueError(
            f"CSV must contain true and prediction columns. "
            f"Accepted true: {true_candidates}, Accepted pred: {pred_candidates}. "
            f"Found: {list(df.columns)}"
        )

    y_true = df[true_col]
    y_pred = df[pred_col]
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    r2 = r2_score(y_true, y_pred)
    return f"rows={len(df)}, MAE={mae:.6f}, RMSE={rmse:.6f}, R2={r2:.6f}"

@tool
def analyze_regression_csv(filename: str) -> str:
    """Analyze a CSV containing true and prediction columns."""
    return analyze_regression_csv_impl(filename)