"""Canonical names and scientific roles for GenVR transport methods.

The repository historically used short artifact keys such as ``cfm``,
``oracle``, ``dense`` and ``mc``.  Those keys remain supported for old runs,
but they are not sufficiently precise scientific descriptions.  This module is
the single source of truth for new reports and machine-readable benchmark
metadata.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Iterable


METHOD_TAXONOMY_SCHEMA_VERSION = "1.0"

FLUX_FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "genvr_cfm_response": ("genvr_cfm_response", "cfm"),
    "projected_local_mc_response": ("projected_local_mc_response", "oracle"),
    "dense_sn_reference": ("dense_sn_reference", "dense"),
    "global_history_mc": ("global_history_mc", "mc"),
}

LEGACY_FLUX_KEYS: dict[str, str] = {
    "cfm": "genvr_cfm_response",
    "oracle": "projected_local_mc_response",
    "dense": "dense_sn_reference",
    "mc": "global_history_mc",
}

METHODS: dict[str, dict[str, Any]] = {
    "genvr_cfm_response": {
        "display_name": "GenVR/CFM projected response",
        "role": "learned candidate",
        "local_kernel": "conditional-flow-matching surrogate samples",
        "global_solver": "deterministic projected response-operator solve",
        "global_particle_histories": False,
        "finite_interface_projection": True,
        "finite_sampling": True,
        "continuum_truth": False,
        "allowed_claim": "Learned approximation to the projected local-response transport model.",
    },
    "projected_local_mc_response": {
        "display_name": "Projected local-MC response",
        "role": "same-discretization kernel reference",
        "local_kernel": "finite random-flight Monte Carlo samples in each local state",
        "global_solver": "deterministic projected response-operator solve",
        "global_particle_histories": False,
        "finite_interface_projection": True,
        "finite_sampling": True,
        "continuum_truth": False,
        "allowed_claim": "Separates learned-kernel error from the shared finite interface discretization.",
    },
    "dense_sn_reference": {
        "display_name": "Dense finite-angle diamond-difference S_N sweep",
        "role": "deterministic finite-grid diagnostic",
        "local_kernel": "diamond-difference discrete ordinates with conservative zero-flux fixup",
        "global_solver": "source iteration over a fixed spatial and angular mesh",
        "global_particle_histories": False,
        "finite_interface_projection": False,
        "finite_sampling": False,
        "continuum_truth": False,
        "allowed_claim": "Deterministic finite-discretization comparison that requires spatial and angular convergence checks; not an exact continuum solution.",
    },
    "global_history_mc": {
        "display_name": "Global history MC (implicit capture)",
        "role": "independent stochastic transport reference",
        "local_kernel": "direct collision and boundary tracking",
        "global_solver": "source-to-termination particle histories",
        "global_particle_histories": True,
        "finite_interface_projection": False,
        "finite_sampling": True,
        "continuum_truth": False,
        "allowed_claim": "Statistical reference with sampling uncertainty; not an exact finite-history truth.",
    },
    "global_cell_response_mc": {
        "display_name": "Global cell-response MC",
        "role": "non-neural GMC composition check",
        "local_kernel": "direct continuous random-flight cell response",
        "global_solver": "source-to-termination particle histories",
        "global_particle_histories": True,
        "finite_interface_projection": False,
        "finite_sampling": True,
        "continuum_truth": False,
        "allowed_claim": "Tests cell-response composition without neural-model error; remains Monte Carlo.",
    },
    "gmc_particle_transport": {
        "display_name": "Particle GMC with learned cell responses",
        "role": "accelerated stochastic candidate",
        "local_kernel": "learned cell-exit samples with optional direct-MC fallback",
        "global_solver": "source-to-termination particle histories",
        "global_particle_histories": True,
        "finite_interface_projection": False,
        "finite_sampling": True,
        "continuum_truth": False,
        "allowed_claim": "Accelerated particle transport whose bias depends on the learned kernel and guards.",
    },
    "diffusion_proxy": {
        "display_name": "Diffusion proxy",
        "role": "low-fidelity screening or adjoint importance",
        "local_kernel": "diffusion approximation",
        "global_solver": "deterministic diffusion solve",
        "global_particle_histories": False,
        "finite_interface_projection": False,
        "finite_sampling": False,
        "continuum_truth": False,
        "allowed_claim": "Cheap ranking/importance proxy; not a transport certification reference.",
    },
    "importance_vr_mc": {
        "display_name": "Importance/weight-window MC",
        "role": "variance-reduced stochastic estimator",
        "local_kernel": "direct transport with weight-preserving splitting and roulette",
        "global_solver": "source-to-termination weighted particle histories",
        "global_particle_histories": True,
        "finite_interface_projection": False,
        "finite_sampling": True,
        "continuum_truth": False,
        "allowed_claim": "Variance reduction is valid only when weight conservation and unbiasedness checks pass.",
    },
}


def get_method_taxonomy(method_ids: Iterable[str] | None = None) -> dict[str, Any]:
    """Return JSON-serializable method metadata for reports and manifests."""

    selected = METHODS if method_ids is None else {method_id: METHODS[method_id] for method_id in method_ids}
    return {
        "schema_version": METHOD_TAXONOMY_SCHEMA_VERSION,
        "legacy_flux_key_aliases": dict(LEGACY_FLUX_KEYS),
        "methods": deepcopy(selected),
    }
