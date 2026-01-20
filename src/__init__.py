"""NeuroCardiac digital twin minimal package

Import and expose core submodules so package consumers and linters
can discover them via ``from src import ...`` and ``__all__``.
"""

# Re-export commonly used submodules for convenience and static analysis tools.
from . import simulator as simulator
from . import preprocessing as preprocessing
from . import feature_extraction as feature_extraction
from . import model as model

__all__ = [
	"simulator",
	"preprocessing",
	"feature_extraction",
	"model",
]