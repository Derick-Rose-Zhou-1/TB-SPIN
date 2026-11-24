"""scPAS.prediction-compatible scoring facade."""
from .api import score

predict = score

__all__ = ["predict"]
