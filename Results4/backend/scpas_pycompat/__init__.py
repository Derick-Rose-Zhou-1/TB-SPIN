"""Reference-compatible Python backend for frozen scPAS v0.2.0."""
from .genes import common_genes
from .normalize import quantile_normalize
from .sparse_cor import sparse_cor
from .network import build_network_from_correlation, compute_snn
from .scoring import scaling_stats, score_cells
from .permutation import permutation_statistics
from .stats import pnorm_upper_abs, bh_adjust, labels
from .api import DEFAULT_ALPHA, ScPASModel, ScPASResult, fit_binomial, score

__version__ = "0.2.0.0"
