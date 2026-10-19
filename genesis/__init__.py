"""Project Genesis: a measurable test of the fractal intent decoder.

The Recursive Intelligence Engine of the Bio-Recursive Mobility Network decodes motor intent as
fractal recursion signatures; the August 2026 restoration architecture projects a state into its
invariant subspace (occupancy Omega_c = 47/125), measures the residual and gates on occupancy. This
package implements both steps as scikit-learn transformers so they can be benchmarked against the
decoders that brain-computer interfaces use today.
"""
from .coherence import OMEGA_C, CoherenceProjection, contraction_power, fisher_scores
from .fractal import FractalIntentSignature, correlation_dimension, signature_names
from .pipelines import BASELINES, GENESIS, SFREQ, build_pipelines, genesis_pipeline

__all__ = ["OMEGA_C", "CoherenceProjection", "contraction_power", "fisher_scores", "FractalIntentSignature",
           "correlation_dimension", "signature_names", "BASELINES", "GENESIS", "SFREQ", "build_pipelines", "genesis_pipeline"]
__version__ = "0.1.0"
