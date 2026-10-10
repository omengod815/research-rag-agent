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
    required = {"true", "prediction"}
    if not required.issubset(df.columns):
        raise ValueError("CSV must contain true and prediction columns")

    y_true = df["true"]
    y_pred = df["prediction"]
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    r2 = r2_score(y_true, y_pred)
    return f"rows={len(df)}, MAE={mae:.6f}, RMSE={rmse:.6f}, R2={r2:.6f}"

@tool
def analyze_regression_csv(filename: str) -> str:
    """Analyze a CSV containing true and prediction columns."""
    return analyze_regression_csv_impl(filename)