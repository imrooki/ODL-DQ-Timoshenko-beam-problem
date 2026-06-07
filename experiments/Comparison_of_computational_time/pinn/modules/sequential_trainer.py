"""Sequential PINN trainer.

Bypasses ``run_dual_pseudo_transfer`` (which hard-rejects LBFGS in the
dual-supervision loop) and instead trains the **linear and nonlinear PINN
models sequentially**, each via the single-model entry point
``modules.solver.train_model`` (which fully supports LBFGS, including
strong-Wolfe line search and history_size).

Why a separate module:
* upstream's ``pseudo_trainer.run_dual_pseudo_transfer`` couples linear and
  nonlinear models inside one optimizer-step loop and explicitly raises
  on ``optimizer_type=='LBFGS'`` because the alternating ``optimizer.step()``
  scheme is incompatible with LBFGS's closure-based step.
* our use case (``use_pseudo_supervision=False``,
  ``transfer_freq=0``) does not actually need the dual coupling at all; a
  sequential `linear -> nonlinear` training is mathematically equivalent
  and unlocks LBFGS support.

The function returns a dict whose schema is **identical** to the one
returned by ``run_dual_pseudo_transfer``, so the downstream consumer in
``utils.training_core`` does not require any change beyond the optimizer-
type branch.

Author: local fork (not part of the upstream Pure-PINN distribution).
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import numpy as np
import torch

# Local imports (relative; fall back to absolute when run as script)
try:
    from .solver import build_model, train_model
except ImportError:  # pragma: no cover (script-mode fallback)
    from modules.solver import build_model, train_model  # type: ignore


# Train-model log columns:    [epoch, total, Pi_all, bc, Pi_str, Pi_str_T, Pi_e]
# Dual-trainer log columns:   [epoch, total, Pi_all, bc, Pi_str, Pi_str_T, Pi_w, Pi_e, pseudo]
# We pad the single-model log with a zero column at index 6 (Pi_w) and
# a zero column at index 8 (pseudo) so the downstream CSV writer (which
# expects the dual-trainer schema) can consume them unchanged.
def _pad_single_log_to_dual_schema(log_arr: np.ndarray) -> np.ndarray:
    if log_arr is None or len(log_arr) == 0:
        return log_arr
    n = log_arr.shape[0]
    zeros_col = np.zeros((n, 1), dtype=log_arr.dtype)
    # Original order: 0:epoch 1:total 2:Pi_all 3:bc 4:Pi_str 5:Pi_str_T 6:Pi_e
    # Target order:   0:epoch 1:total 2:Pi_all 3:bc 4:Pi_str 5:Pi_str_T 6:Pi_w(=0) 7:Pi_e 8:pseudo(=0)
    padded = np.hstack([
        log_arr[:, :6],                    # epoch ... Pi_str_T
        zeros_col,                         # Pi_w  = 0  (no foundation in PINN energy decomposition here)
        log_arr[:, 6:7],                   # Pi_e
        zeros_col,                         # pseudo = 0 (we never enable pseudo-supervision here)
    ])
    return padded


def run_sequential_training(
    *,
    coeffs,
    params_obj,
    bc,
    device,
    # network
    encoder_dims_shared=None,
    head_dims=None,
    in_dim: int = 1,
    # training
    epochs: int,
    N_train: int,
    lr: float,
    print_every: int,
    bc_weight: float,
    optimizer_type: str,
    lbfgs_max_iter: int,
    lbfgs_history_size: int,
    lbfgs_line_search_fn: Optional[str],
    adamw_weight_decay: float = 1e-4,
    # sampler / integrator
    sampler: str = "uniform",
    sampler_reuse: bool = False,
    integrator: str = "mc",
    agq_rule: str = "G10K21",
    agq_abs_tol: float = 1e-6,
    agq_rel_tol: float = 1e-4,
    agq_max_points: int = 4096,
    agq_max_depth: int = 100,
    agq_refine_every: int = 0,
    agq_fail_policy: str = "use_partial",
    # adaptive lr (ignored for LBFGS, kept for signature-compatibility)
    use_adaptive_lr: bool = False,
    lr_early_max: float = 1e-3,
    lr_early_min: float = 2e-4,
    lr_mid_max: float = 2e-4,
    lr_mid_min: float = 1e-4,
    lr_late_fixed: float = 1e-4,
    lr_patience: int = 500,
    lr_improvement_threshold: float = 1e-6,
    lr_decay_factor: float = 0.5,
    lr_verbose: bool = False,
    lr_warmup_epochs: int = 100,
    lr_early_ratio: float = 0.6,
    lr_mid_ratio: float = 0.85,
    lr_min_early_epochs: int = 1000,
    lr_min_mid_epochs: int = 5000,
    # pseudo / transfer (always disabled in sequential mode; accepted for
    # signature-compatibility with run_dual_pseudo_transfer)
    use_pseudo_supervision: bool = False,
    ps_w_non_start: float = 1.0,
    ps_w_lin_start: float = 0.5,
    ps_cut_ratio: float = 0.8,
    ps_use_phi: bool = False,
    transfer_alpha: float = 0.3,
    transfer_ratio: float = 0.7,
    transfer_freq: int = 0,
    transfer_cut_ratio: float = 0.2,
    # eval / callbacks
    x_eval: Optional[torch.Tensor] = None,
    epoch_callback: Optional[Callable[[int, Dict[str, Any]], None]] = None,
    # activation / lifting
    activation_type: str = "Tanh",
    siren_omega_0: float = 30.0,
    siren_omega_hidden: float = 30.0,
    lifting_basis: str = "poly",
    # local fork knobs
    patience: Optional[int] = None,
    restore_best: bool = True,
    seed: int = 42,
) -> Dict[str, Any]:
    """Train linear and nonlinear PINN models sequentially.

    The signature mirrors :func:`run_dual_pseudo_transfer` so the call site
    in ``training_core._run_dual_training`` only needs to switch which
    function is called based on ``optimizer_type``. Pseudo-supervision and
    transfer-learning kwargs are accepted but always treated as disabled
    (sequential training has no inter-model coupling).

    Returns
    -------
    dict
        Same schema as ``run_dual_pseudo_transfer``::

            {
              'fields_linear':       Dict[str, torch.Tensor]   (only if x_eval given)
              'fields_nonlinear':    Dict[str, torch.Tensor]   (only if x_eval given)
              'logs_linear':         np.ndarray (n_iter, 9)    (padded to dual schema)
              'logs_nonlinear':      np.ndarray (n_iter, 9)
              'best_linear_loss':    float
              'best_nonlinear_loss': float
              'best_linear_epoch':   int
              'best_nonlinear_epoch':int
              'model_linear':        EnergyPINNStatic
              'model_nonlinear':     EnergyPINNStatic
            }
    """
    print()
    print("=" * 70)
    print("[Sequential] LBFGS-compatible training (linear -> nonlinear)")
    print(f"[Sequential] Optimizer = {optimizer_type}, lr = {lr}, epochs = {epochs}")
    print(f"[Sequential] patience  = {patience}, restore_best = {restore_best}")
    print(f"[Sequential] seed      = {seed}")
    print("=" * 70)

    if use_pseudo_supervision or transfer_freq > 0:
        print("[Sequential] WARNING: pseudo/transfer requested but ignored "
              "(sequential mode does not couple the two models).")

    # Common kwargs for build_model
    build_kwargs = dict(
        coeffs=coeffs,
        params=params_obj,
        bc=bc,
        device=device,
        bc_weight=bc_weight,
        encoder_dims_shared=encoder_dims_shared,
        head_dims=head_dims,
        in_dim=in_dim,
        sampler=sampler,
        sampler_reuse=sampler_reuse,
        integrator=integrator,
        agq_rule=agq_rule,
        agq_abs_tol=agq_abs_tol,
        agq_rel_tol=agq_rel_tol,
        agq_max_points=agq_max_points,
        agq_max_depth=agq_max_depth,
        agq_refine_every=agq_refine_every,
        agq_fail_policy=agq_fail_policy,
        activation_type=activation_type,
        siren_omega_0=siren_omega_0,
        siren_omega_hidden=siren_omega_hidden,
        lifting_basis=lifting_basis,
    )

    # Common kwargs for train_model
    train_kwargs = dict(
        epochs=epochs,
        lr=lr,
        N_samples=N_train,
        print_every=print_every,
        best_model_path=None,  # restore_best handled in-memory by train_model
        optimizer_type=optimizer_type,
        lbfgs_max_iter=lbfgs_max_iter,
        lbfgs_history_size=lbfgs_history_size,
        lbfgs_line_search_fn=lbfgs_line_search_fn,
        adamw_weight_decay=adamw_weight_decay,
        patience=patience,
        restore_best=restore_best,
    )

    # ----------------- LINEAR MODEL -----------------
    print()
    print("[Sequential] === Stage 1/2: LINEAR PINN ===")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model_lin = build_model("linear", **build_kwargs)
    log_lin, best_lin_loss, best_lin_epoch = train_model(model_lin, **train_kwargs)
    print(f"[Sequential] Linear training done: "
          f"best_loss = {best_lin_loss:.4e} @ epoch {best_lin_epoch}, "
          f"actual epochs run = {len(log_lin) if log_lin is not None else 0}")

    # ----------------- NONLINEAR MODEL -----------------
    print()
    print("[Sequential] === Stage 2/2: NONLINEAR PINN ===")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model_nl = build_model("nonlinear", **build_kwargs)
    log_nl, best_nl_loss, best_nl_epoch = train_model(model_nl, **train_kwargs)
    print(f"[Sequential] Nonlinear training done: "
          f"best_loss = {best_nl_loss:.4e} @ epoch {best_nl_epoch}, "
          f"actual epochs run = {len(log_nl) if log_nl is not None else 0}")

    # ----------------- ASSEMBLE OUTPUT -----------------
    out: Dict[str, Any] = {}
    if x_eval is not None:
        model_lin.eval()
        model_nl.eval()
        out["fields_linear"] = model_lin.fields_and_grads(x_eval)
        out["fields_nonlinear"] = model_nl.fields_and_grads(x_eval)

    out["logs_linear"] = _pad_single_log_to_dual_schema(log_lin)
    out["logs_nonlinear"] = _pad_single_log_to_dual_schema(log_nl)
    out["best_linear_loss"] = float(best_lin_loss)
    out["best_nonlinear_loss"] = float(best_nl_loss)
    out["best_linear_epoch"] = int(best_lin_epoch)
    out["best_nonlinear_epoch"] = int(best_nl_epoch)
    out["model_linear"] = model_lin
    out["model_nonlinear"] = model_nl

    print()
    print("=" * 70)
    print(f"[Sequential] Both stages complete. "
          f"Linear best loss = {best_lin_loss:.4e}, "
          f"Nonlinear best loss = {best_nl_loss:.4e}")
    print("=" * 70)

    return out
