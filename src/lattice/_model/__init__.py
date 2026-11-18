"""Private. Users should not import from this subpackage.

``LatticeModel`` is wrapped by :class:`lattice.LatticeClassifier` and
:class:`lattice.LatticeRegressor`. ``InferenceConfig`` is re-exported at
:class:`lattice.InferenceConfig`.
"""

from .base_model import LatticeModel
from .inference_config import InferenceConfig
