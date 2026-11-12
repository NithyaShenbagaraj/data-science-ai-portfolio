import pandas as pd

from src.data import TARGET_COLUMN, feature_types


def test_feature_types_separate_known_categorical_columns():
    frame = pd.DataFrame(
        {
            "age": [45, 52],
            "chol": [220, 190],
            "cp": [1, 3],
            "thal": [3, 7],
        }
    )
    numeric, categorical = feature_types(frame)
    assert numeric == ["age", "chol"]
    assert categorical == ["cp", "thal"]


def test_target_column_name_is_stable():
    assert TARGET_COLUMN == "target"
