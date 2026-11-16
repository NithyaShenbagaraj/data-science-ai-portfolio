import pandas as pd
import pytest
from src.features import engineer_features, split_features_target


def sample_frame():
    return pd.DataFrame({
        "customerID": ["A", "B", "C"],
        "tenure": [3, 18, 60],
        "MonthlyCharges": [40.0, 70.0, 95.0],
        "TotalCharges": ["120.0", "1260.0", "5700.0"],
        "PhoneService": ["Yes", "No", "Yes"],
        "OnlineSecurity": ["No", "No internet service", "Yes"],
        "Churn": ["No", "Yes", "No"],
    })


def test_feature_engineering():
    features = engineer_features(sample_frame())
    assert features["TotalCharges"].dtype.kind in "fi"
    assert features["tenure_group"].tolist() == ["0-12", "13-24", "49-72"]
    assert features["service_count"].tolist() == [1, 0, 2]
    assert "customerID" not in features.columns
    assert "Churn" not in features.columns


def test_target_is_binary():
    X, y = split_features_target(sample_frame())
    assert y.tolist() == [0, 1, 0]
    assert len(X) == len(y)


def test_unknown_target_label_raises():
    frame = sample_frame()
    frame.loc[0, "Churn"] = "Maybe"
    with pytest.raises(ValueError, match="Unexpected target labels"):
        split_features_target(frame)
