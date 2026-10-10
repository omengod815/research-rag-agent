import pytest
import pandas as pd
from pathlib import Path
import app.tools.data_tool as data_tool_module
from app.tools.data_tool import analyze_regression_csv_impl, analyze_regression_csv

@pytest.fixture
def safe_dir(tmp_path, monkeypatch):
    """将模块中的 SAFE_DIR 临时替换为 pytest 的临时目录，避免污染真实数据"""
    test_safe_dir = tmp_path / "data_public"
    test_safe_dir.mkdir()
    monkeypatch.setattr(data_tool_module, "SAFE_DIR", test_safe_dir)
    return test_safe_dir

@pytest.fixture
def valid_csv(safe_dir):
    """在临时安全目录中创建一个合法的 CSV 文件"""
    csv_path = safe_dir / "valid.csv"
    df = pd.DataFrame({
        "true": [1.0, 2.0, 3.0, 4.0, 5.0],
        "prediction": [1.1, 2.2, 2.8, 4.1, 4.9]
    })
    df.to_csv(csv_path, index=False)
    return "valid.csv"

# ================= 核心功能验证 =================

def test_analyze_regression_success(valid_csv):
    """成功标志：正常 CSV 应返回 MAE, RMSE, R2 且数值合理"""
    result = analyze_regression_csv_impl(valid_csv)
    
    assert "rows=5" in result
    assert "MAE=" in result
    assert "RMSE=" in result
    assert "R2=" in result
    
    # 简单校验数值逻辑（人工估算：误差很小，R2 应该接近 1）
    assert "R2=0.9" in result or "R2=1.0" in result or float(result.split("R2=")[1]) > 0.9

def test_langchain_tool_invocation(valid_csv):
    """验证 LangChain @tool 装饰器包装后是否能正常调用"""
    # LangChain tool 可以通过 .invoke() 调用
    result = analyze_regression_csv.invoke({"filename": valid_csv})
    assert "MAE=" in result

# ================= 安全与异常拦截验证 =================

def test_security_path_traversal(safe_dir):
    """安全验证：拦截 ../../ 等路径穿越攻击"""
    with pytest.raises(ValueError, match="Unsafe path"):
        analyze_regression_csv_impl("../../etc/passwd")
        
    with pytest.raises(ValueError, match="Unsafe path"):
        analyze_regression_csv_impl("../secret.csv")

def test_file_not_found(safe_dir):
    """异常验证：文件不存在"""
    with pytest.raises(FileNotFoundError, match="CSV not found"):
        analyze_regression_csv_impl("not_exist.csv")

def test_non_csv_extension(safe_dir):
    """异常验证：传入非 .csv 后缀的文件"""
    txt_file = safe_dir / "test.txt"
    txt_file.write_text("hello")
    
    with pytest.raises(FileNotFoundError, match="CSV not found"):
        analyze_regression_csv_impl("test.txt")

def test_missing_required_columns(safe_dir):
    """异常验证：CSV 缺少 true 或 prediction 列"""
    bad_csv = safe_dir / "bad.csv"
    pd.DataFrame({"a": [1, 2], "b": [3, 4]}).to_csv(bad_csv, index=False)
    
    with pytest.raises(ValueError, match="CSV must contain true and prediction columns"):
        analyze_regression_csv_impl("bad.csv")