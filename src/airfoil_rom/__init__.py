"""airfoil_rom: the pure functions of the two notebooks, importable.

Thin re-export module. The notebooks keep their own inline copies (their outputs are
the record of the runs); ``docs/REPO_MAP.md`` maps every function here to the cell it
came from. Modules that need TensorFlow, Gymnasium or Matplotlib import them lazily or
on first use of the submodule, so ``import airfoil_rom`` itself needs only NumPy,
pandas and scikit-learn.
"""
from .preprocessing import (  # noqa: F401
    PARAMETER_NAMES,
    channel_statistics,
    standardise_fields,
    destandardise_fields,
    to_channels_last,
    to_channels_first,
    prepare_parameters,
    make_candidate_grid,
    digits_to_chain_inputs,
    chain_inputs_to_stage3,
)
from .metrics import (  # noqa: F401
    REGION_BANDS,
    standardised_mse,
    per_channel_mse,
    rmse,
    coefficient_scores,
    error_by_region,
    nearest_distance,
    timed,
    episode_metrics,
)
from .models import (  # noqa: F401
    get_sampling_layer,
    build_beta_vae,
    beta_vae_loss,
    beta_schedule,
    build_latent_surrogate,
    build_aerodynamic_head,
    dense_stack,
    run_stack,
    NumpyChain,
)

__all__ = [name for name in dir() if not name.startswith("_")]
__version__ = "0.1.0"
