"""Data Transfer Objects for API responses."""
from dataclasses import dataclass, asdict
from typing import Any
import numpy as np


def _convert_numpy_types(obj: Any) -> Any:
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {key: _convert_numpy_types(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        # Convert tuples to lists for JSON serialization
        return [_convert_numpy_types(item) for item in obj]
    return obj


@dataclass
class PredictResponseDTO:
    """DTO for prediction response."""

    label: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class EstimateResponseDTO:
    """DTO for estimate response."""

    label: str
    cluster: int
    predicted_value: float
    top_rows: list[dict[str, Any]]
    top_rows_data: list[dict[str, Any]]
    updated_input: list[float]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "label": self.label,
            "cluster": _convert_numpy_types(self.cluster),
            "predicted_value": _convert_numpy_types(self.predicted_value),
            "top_rows": _convert_numpy_types(self.top_rows),
            "top_rows_data": _convert_numpy_types(self.top_rows_data),
            "updated_input": _convert_numpy_types(self.updated_input),
        }
