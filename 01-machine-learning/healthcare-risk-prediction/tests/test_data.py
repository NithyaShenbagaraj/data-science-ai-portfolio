
import pandas as pd
import pytest

from src.data import TARGET_COLUMN, feature_types, load_dataset


def test_feature_types_separate_known_categorical_columns():
    frame = pd.DataFrame({
        "age": [45, 52],
        "chol": [220, 190],
        "cp": [1, 3],
        "thal": [3, 7],
    })

    numeric, categorical = feature_types(frame)

    assert numeric == ["age", "chol"]
    assert categorical == ["cp", "thal"]


def test_target_column_name_is_stable():
    assert TARGET_COLUMN == "target"


def test_load_dataset_normalizes_target_and_missing_values(tmp_path):
    frame = pd.DataFrame({
        "age": [45, 52],
        "cp": [1, 3],
        "ca": ["?", "2"],
        "target": [0, 2],
    })

    path = tmp_path / "sample.csv"
    frame.to_csv(path, index=False)

    X, y = load_dataset(path)

    assert y.tolist() == [0, 1]
    assert pd.isna(X.loc[0, "ca"])
    assert X.loc[1, "ca"] == 2
    assert X.index.equals(y.index)


def test_load_dataset_rejects_missing_target_column(tmp_path):
    frame = pd.DataFrame({
        "age": [45, 52],
        "cp": [1, 3],
    })

    path = tmp_path / "missing_target.csv"
    frame.to_csv(path, index=False)

    with pytest.raises(ValueError, match="Expected target column"):
        load_dataset(path)


def test_load_dataset_rejects_single_class_target(tmp_path):
    frame = pd.DataFrame({
        "age": [45, 52],
        "cp": [1, 3],
        "target": [0, 0],
    })

    path = tmp_path / "single_class.csv"
    frame.to_csv(path, index=False)

    with pytest.raises(ValueError, match="binary target"):
        load_dataset(path)
