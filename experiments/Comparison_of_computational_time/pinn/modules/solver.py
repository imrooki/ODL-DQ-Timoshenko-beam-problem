"""
Timoshenko beam PINN solver and training module

Author: Yang
Version: 1.0

Responsibilities:
- EnergyPINNStatic: the core PINN solver class, integrating the network architecture, boundary conditions, and physical loss computation
- train_model: an efficient training loop supporting a hybrid training strategy with the Adam and LBFGS optimizers
- Provide complete forward propagation, loss computation, gradient back-propagation, and model saving functionality

Core features:
- Physics-constrained loss function based on the energy variational principle
- Automatic differentiation to compute high-order derivatives, ensuring numerical accuracy
- Flexible handling of soft and hard constraints for boundary conditions
- Automatic saving of the best model and training log recording
- Support for both linear and nonlinear Timoshenko beam problems

Technical implementation:
- A unified forward propagation interface returning a dictionary of displacement fields and derivative fields
- Weighted combined optimization of the energy loss and boundary loss
- A numerically stable training strategy and convergence criteria
- An API design fully compatible with existing code

Return format conventions:
- fields_and_grads dictionary keys: 'u','w','phi','ux','wx','phix'
- Maintain the integrity of the computation graph, supporting high-order automatic differentiation
- Compatible with both batch processing and single-point evaluation
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import time

# Compatible with running the script directly: try absolute imports when relative imports fail
try:
    from .nets import build_timoshenko_net
    from .data_types import MaterialCoeffs, PhysicalParams, BoundaryConditions
    from .bc import lifting, lifting_trig, get_lifting_function
    from .numerics import d_dx, sample_1d
    from .physics import EnergyLoss, WeightedEnergyLoss
except ImportError:
    import os, sys
    pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if pkg_root not in sys.path:
        sys.path.insert(0, pkg_root)
    from modules.nets import build_timoshenko_net
    from modules.data_types import MaterialCoeffs, PhysicalParams, BoundaryConditions
    from modules.bc import lifting, lifting_trig, get_lifting_function
    from modules.numerics import d_dx, sample_1d
    from modules.physics import EnergyLoss, WeightedEnergyLoss


class ConstantField:
    """Constant field: avoids defining a nested function inside as_fun."""

    def __init__(self, value: float, name: str = "") -> None:
        self.value = float(value)
        # Only used for readability during debugging
        self.__name__ = f"const_{name}" if name else "const_field"

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return torch.full_like(x, self.value)


def as_fun(val_or_fun, name: str = "") -> Callable[[torch.Tensor], torch.Tensor]:
    """Wrap a constant or callable object as a callable object (no nested function)."""

    if callable(val_or_fun):
        return val_or_fun
    return ConstantField(val_or_fun, name)


class EnergyPINNStatic(nn.Module):
    """Energy-based static Timoshenko beam PINN solver -- the core class of the project

    ================================================================================
                            Class design philosophy and innovations
    ================================================================================

    [Core innovation: PINN implementation of the energy variational principle]

    Unlike traditional PINNs that directly encode the strong-form partial differential equations, this class is implemented based on the Hamilton variational principle:

    1. Physical principle:
       Principle of minimum potential energy: delta-Pi = 0
       where the total potential energy: Pi = Pi_str + Pi_w - Pi_e

       This automatically derives the equilibrium equations of the Timoshenko beam, without explicitly encoding the complex PDE system

    2. Numerical advantages:
       - The integral form has better numerical stability than the differential form
       - The equilibrium equations are satisfied naturally, reducing constraint terms
       - Energy conservation is automatically guaranteed, with strong physical consistency
       - Boundary conditions are satisfied naturally through the variation (natural boundary conditions)

    [Design considerations of the SharedEncoder architecture]

    A shared-encoder + independent-decoder-head architecture is adopted, rather than a traditional end-to-end network:

    Network structure:
    x -> [shared encoder (128-64-32)] -> z -> [decoder head u (32-16-1)] -> u(x)
                                          -> [decoder head w (32-16-1)] -> w(x)
                                          -> [decoder head phi (32-16-1)] -> phi(x)

    Design advantages:
    1. Natural expression of physical coupling:
       - In Timoshenko theory, u, w, phi are coupled through the equilibrium equations
       - The shared encoder learns the common physical features of the three fields
       - The independent decoder heads allow field-specific nonlinear mappings

    2. Parameter efficiency:
       - The encoder parameters are shared by the three outputs, reducing the total parameter count by about 40%
       - A deeper feature-extraction layer can be built at the same parameter scale

    3. Training stability:
       - Avoids one field overfitting while the others underfit
       - Shared gradients help the fields converge cooperatively
       - Reduces the training difficulty of a multi-output network

    [Hybrid boundary condition handling strategy]

    Combines the advantages of hard constraints and soft constraints:

    1. Hard constraints (lifting function) handle Dirichlet boundaries:
       - The displacement boundary conditions (u=0, w=0, phi=0) are satisfied exactly through an analytical function
       - Avoids boundary loss terms, reducing hyperparameter tuning
       - Numerically stable, unaffected by the penalty-term weight

    2. Soft constraints (penalty method) handle Neumann boundaries:
       - The natural boundary condition (M=0) is satisfied progressively through a loss term
       - Flexibly adapts to different combinations of boundary conditions
       - The weight is adjustable, balancing accuracy and convergence speed

    [Numerical integration strategy]

    Supports multiple integration methods, selectable according to the problem characteristics:

    1. Monte Carlo (mc):
       - Random sampling, suitable for high-dimensional problems
       - Unbiased estimation, convergence rate O(1/sqrt(N))

    2. Gauss-Legendre (gauss):
       - Deterministic high-accuracy integration
       - Most efficient for smooth functions

    3. Clenshaw-Curtis (clenshaw):
       - Good nestedness, convenient for error estimation
       - More robust for non-smooth functions

    4. Adaptive Gauss quadrature (agq):
       - Automatically subdivides the interval to reach the target accuracy
       - Adapts well to singularities and sharp local variations
       - Estimates the error using a high-order/low-order rule pair (G21/G10)

    [Unified handling of linear/nonlinear problems]

    The two types of problems are unified elegantly through the is_nonlinear flag:

    - Linear: removes the geometric nonlinear term (partial w / partial x)^2
    - Nonlinear: the complete von Karman strain
    - Maximizes code reuse, convenient for comparison and validation

    ================================================================================

    Main methods:
    - __init__: initialize the network, loss function, and numerical parameters
    - fields_and_grads: compute the displacement fields (u,w,phi) and their derivatives (ux,wx,phix)
    - energies: compute the energy components (strain energy, foundation energy, external work)
    - loss: assemble the total loss (energy + boundary penalty)
    - _train_samples: generate training sampling points (supports cache reuse)
    """

    def __init__(
        self,
        coeffs: MaterialCoeffs,
        params: PhysicalParams,
        bc: BoundaryConditions,
        *,
        device: Optional[torch.device] = None,
        bc_type: str = "C-C",
        bc_weight: float = 10.0,
        is_nonlinear: bool = False,
        encoder_dims_shared: Optional[list] = None,
        head_dims: Optional[list] = None,
        in_dim: int = 1,
        sampler: str = "uniform",
        sampler_reuse: bool = False,
        integrator: str = "mc",
        # AGQ params (used when integrator_mode == 'agq')
        agq_rule: str = "G10K21",
        agq_abs_tol: float = 1e-6,
        agq_rel_tol: float = 1e-4,
        agq_max_points: int = 4096,
        agq_max_depth: int = 100,
        agq_refine_every: int = 0,
        agq_fail_policy: str = "use_partial",
        # Activation function parameters
        activation_type: str = "Tanh",
        siren_omega_0: float = 30.0,
        siren_omega_hidden: float = 30.0,
        # Lifting basis function parameter
        lifting_basis: str = "poly",
    ) -> None:
        super().__init__()
        self.coeffs = coeffs
        self.params = params
        # Merge the boundary conditions with the type (ensure .type matches the passed-in bc_type)
        self.bc = BoundaryConditions(
            type=bc_type,
            u_left=bc.u_left,
            u_right=bc.u_right,
            w_left=bc.w_left,
            w_right=bc.w_right,
            phi_left=bc.phi_left,
            phi_right=bc.phi_right,
        )
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Network construction (only the shared architecture is supported); the current static solver only supports x-only input
        if in_dim != 1:
            raise ValueError(
                f"The static EnergyPINNStatic currently only supports x-only input (in_dim=1), got in_dim={in_dim}. "
                "For (x,t) dynamics, please use the corresponding dynamic solver entry point."
            )
        self.net = build_timoshenko_net(
            in_dim=in_dim,
            encoder_dims_shared=encoder_dims_shared,
            head_dims=head_dims,
            activation_type=activation_type,
            siren_omega_0=siren_omega_0,
            siren_omega_hidden=siren_omega_hidden,
        ).to(self.device)

        # Store the lifting basis function type and get the corresponding function
        self.lifting_basis = str(lifting_basis)
        self._lifting_fn = get_lifting_function(self.lifting_basis)

        self.bc_type = bc_type
        self.bc_weight = float(bc_weight)
        self.is_nonlinear = bool(is_nonlinear)
        self.sampler_type = str(sampler)
        self.sampler_reuse = bool(sampler_reuse)
        self._cached_samples: Optional[torch.Tensor] = None
        self._cached_N: Optional[int] = None
        self.integrator_type = str(integrator)
        # AGQ config
        self.agq_rule = str(agq_rule)
        self.agq_abs_tol = float(agq_abs_tol)
        self.agq_rel_tol = float(agq_rel_tol)
        self.agq_max_points = int(agq_max_points)
        self.agq_max_depth = int(agq_max_depth)
        self.agq_refine_every = int(agq_refine_every)
        self.agq_fail_policy = str(agq_fail_policy)

        if self.params.lambda_val <= 0:
            raise ValueError(f"lambda_val must be greater than 0, current value: {self.params.lambda_val}")
        if self.device.type == "cuda" and not torch.cuda.is_available():
            print("[WARNING] CUDA specified but GPU not available, switching to CPU")
            self.device = torch.device("cpu")

        # Construct the energy and boundary losses
        energy_loss = EnergyLoss(self.coeffs, self.params, self.bc, self.device, is_nonlinear=self.is_nonlinear)

        # Use a lazy import to avoid a module-level circular import
        def _get_boundary_penalty_class():
            """Lazily import the boundary condition penalty class"""
            try:
                from .bc import BoundaryConditionPenalty
                return BoundaryConditionPenalty
            except ImportError:
                from modules.bc import BoundaryConditionPenalty
                return BoundaryConditionPenalty

        BoundaryConditionPenalty = _get_boundary_penalty_class()
        bc_penalty = BoundaryConditionPenalty(self.bc, self.coeffs, self.params, self.device, is_nonlinear=self.is_nonlinear)
        self.weighted_loss_func = WeightedEnergyLoss(energy_loss, bc_penalty, self.bc_weight)
        # Set the integrator and AGQ parameters
        self.weighted_loss_func.integrator = (self.integrator_type or 'mc')
        # AGQ configuration (used for integrator_mode='agq')
        self.weighted_loss_func.agq_rule = self.agq_rule
        self.weighted_loss_func.agq_abs_tol = self.agq_abs_tol
        self.weighted_loss_func.agq_rel_tol = self.agq_rel_tol
        self.weighted_loss_func.agq_max_points = self.agq_max_points
        self.weighted_loss_func.agq_max_depth = self.agq_max_depth
        self.weighted_loss_func.agq_refine_every = self.agq_refine_every
        self.weighted_loss_func.agq_fail_policy = self.agq_fail_policy

    def fields_and_grads(self, x_norm: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute the lifted field variables and their derivatives, returning the dictionary: {'u','w','phi','ux','wx','phix'}

        Processing flow:
        1. Neural network forward propagation: map the normalized coordinate x to the raw displacement fields (raw_u, raw_w, raw_phi)
        2. Boundary condition lifting: convert the raw outputs to displacement fields that satisfy the boundary conditions via the lifting function
        3. Automatic differentiation: use PyTorch's automatic differentiation mechanism to compute the spatial derivatives (ux, wx, phix)
        4. Return the complete displacement field dictionary, for subsequent energy integration computation

        Parameters:
        - x_norm: normalized coordinate tensor, shape=(N, 1), range [0, 1]

        Returns:
        - fields: dictionary of displacement fields and their derivatives
          * 'u': axial displacement field u(x)
          * 'w': transverse displacement field w(x) (deflection)
          * 'phi': rotation field phi(x)
          * 'ux': axial displacement derivative du/dx
          * 'wx': transverse displacement derivative dw/dx
          * 'phix': rotation derivative dphi/dx

        Technical notes:
        - Boundary condition lifting ensures the boundary constraints are satisfied analytically, reducing the influence of the boundary penalty term
        - Automatic differentiation ensures the accuracy of derivative computation, avoiding the truncation error of numerical differentiation
        - When differentiating the same x_norm repeatedly, the first two calls need retain_graph=True to keep the computation graph
        """

        raw = self.net(x_norm)
        # Ensure the network output dimension is correct (at least 3 outputs: u, w, phi)
        if raw.size(-1) < 3:
            raise ValueError(f"Insufficient network output dimension, expected at least 3 outputs, got {raw.size(-1)}")
        raw_u, raw_w, raw_phi = raw[:, [0]], raw[:, [1]], raw[:, [2]]
        # Use the configured lifting function (poly or trig)
        u, w, phi = self._lifting_fn(x_norm, raw_u, raw_w, raw_phi, self.bc)

        # Differentiating the same x_norm repeatedly: all calls keep the computation graph
        # Avoid the "Trying to backward through the graph a second time" error.
        # Note: in pseudo-supervised training, fields_and_grads is called multiple times within the same epoch:
        # 1. Called internally by compute_total_loss via field_eval (the AGQ integrator calls it multiple times)
        # 2. Called directly during pseudo-label computation
        # Therefore the computation graph must be kept in all grad calls, letting PyTorch's garbage collection manage memory automatically
        ux = torch.autograd.grad(
            u, x_norm,
            grad_outputs=torch.ones_like(u),
            create_graph=True,
            retain_graph=True,  # keep the computation graph
            only_inputs=True
        )[0]
        wx = torch.autograd.grad(
            w, x_norm,
            grad_outputs=torch.ones_like(w),
            create_graph=True,
            retain_graph=True,  # keep the computation graph
            only_inputs=True
        )[0]
        phix = torch.autograd.grad(
            phi, x_norm,
            grad_outputs=torch.ones_like(phi),
            create_graph=True,
            retain_graph=True,  # keep the computation graph (consistent with the first two)
            only_inputs=True
        )[0]
        return {"u": u, "w": w, "phi": phi, "ux": ux, "wx": wx, "phix": phix}

    def _train_samples(self, N_samples: int) -> torch.Tensor:
        """Return training samples, optionally reusing a cached set across epochs."""
        if self.sampler_reuse and self._cached_samples is not None and self._cached_N == int(N_samples):
            return self._cached_samples
        x = sample_1d(N_samples, self.device, sampler=self.sampler_type, dtype=torch.float32)
        if self.sampler_reuse:
            self._cached_samples = x
            self._cached_N = int(N_samples)
        return x

    def energies(self, x_norm: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the energy components: Pi_str, Pi_T, Pi_e, Pi_all."""

        fields = self.fields_and_grads(x_norm)
        # Call the energy function inside the weighting object (keep consistent with the old interface)
        energy = self.weighted_loss_func.energy_loss.compute_total_energy(x_norm, fields)
        return energy["Pi_str"], energy["Pi_str_T"], energy["Pi_e"], energy["Pi_all"]

    def bc_loss(self) -> torch.Tensor:
        """Boundary penalty value (excluding the weight)."""

        return self.weighted_loss_func.bc_penalty.compute(self.fields_and_grads)

    def loss(self, N_samples: int = 4096) -> Dict[str, torch.Tensor]:
        """Compute the training loss dictionary, consistent with the old interface."""
        x_norm = self._train_samples(N_samples)
        loss_comp = self.weighted_loss_func.compute_total_loss(x_norm, self.fields_and_grads)
        return {
            "Pi_str": loss_comp["Pi_str"],
            "Pi_T": loss_comp["Pi_str_T"],
            "Pi_e": loss_comp["Pi_e"],
            "Pi_all": loss_comp["Pi_all"],
            "bc": loss_comp["bc"],
            "total": loss_comp["total"],
        }


class LBFGSClosure:
    """LBFGS closure (avoids defining a nested function inside train_model)."""

    def __init__(self, model: "EnergyPINNStatic", optimizer: torch.optim.Optimizer, x_samples: torch.Tensor) -> None:
        self.model = model
        self.optimizer = optimizer
        self.x_samples = x_samples

    def __call__(self) -> torch.Tensor:
        self.optimizer.zero_grad()
        loss_comp = self.model.weighted_loss_func.compute_total_loss(self.x_samples, self.model.fields_and_grads)
        total_c = loss_comp["total"]
        if torch.isnan(total_c) or torch.isinf(total_c):
            return torch.tensor(float("inf"), device=total_c.device)
        total_c.backward()
        return total_c


def train_model(
    model: EnergyPINNStatic,
    epochs: int = 3000,
    lr: float = 1e-3,
    N_samples: int = 4096,
    print_every: int = 200,
    best_model_path: Optional[str] = None,
    optimizer_type: str = "Adam",
    lbfgs_max_iter: int = 20,
    lbfgs_history_size: int = 100,
    lbfgs_line_search_fn: Optional[str] = "strong_wolfe",
    adamw_weight_decay: float = 1e-4,
    # ---- Early-stopping & best/final selection ----
    patience: Optional[int] = None,
    restore_best: bool = True,
):
    """Main training loop for the PINN model

    Description:
    - Train the Timoshenko beam PINN model based on the energy minimization principle
    - Support multiple optimization algorithms to suit different convergence needs
    - Automatically save the best model weights with the minimum loss
    - Provide detailed training progress monitoring and GPU status tracking

    Training strategy:
    - Adam optimizer: adaptive learning rate, the most commonly used, stable convergence
    - AdamW optimizer: Adam + decoupled weight decay, suitable for regularization
    - RAdam optimizer: variance-corrected Adam, more stable training
    - NAdam optimizer: Nesterov momentum + Adam
    - Adamax optimizer: the L-infinity variant of Adam, suitable for sparse gradients
    - LBFGS optimizer: quasi-Newton method, suitable for fine optimization on small batches
    - Gradient clipping: prevents gradient explosion, ensuring training stability
    - NaN/Inf detection: automatically skips abnormal iterations, ensuring training continuity

    Parameters:
    - model: EnergyPINNStatic solver instance
    - epochs: number of training epochs, controls the number of optimization iterations
    - lr: learning rate, affects the convergence speed and stability
    - N_samples: number of sampling points per training epoch, affects the integration accuracy and computational cost
    - print_every: print frequency, controls the progress output interval
    - best_model_path: best model save path, not saved if None
    - optimizer_type: optimizer type ('Adam'/'AdamW'/'RAdam'/'NAdam'/'Adamax'/'LBFGS')
    - lbfgs_*: parameters specific to the LBFGS optimizer
    - adamw_weight_decay: AdamW weight decay coefficient

    Returns:
    - log_array: training log array, containing the loss record of each epoch
    """

    opt_upper = optimizer_type.upper()
    if opt_upper == "ADAM":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif opt_upper == "ADAMW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=adamw_weight_decay)
    elif opt_upper == "RADAM":
        optimizer = torch.optim.RAdam(model.parameters(), lr=lr)
    elif opt_upper == "NADAM":
        optimizer = torch.optim.NAdam(model.parameters(), lr=lr)
    elif opt_upper == "ADAMAX":
        optimizer = torch.optim.Adamax(model.parameters(), lr=lr)
    elif opt_upper == "LBFGS":
        optimizer = torch.optim.LBFGS(
            model.parameters(),
            lr=lr,
            max_iter=lbfgs_max_iter,
            history_size=lbfgs_history_size,
            line_search_fn=lbfgs_line_search_fn,
        )
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}. "
                        f"Options: ['Adam', 'AdamW', 'RAdam', 'NAdam', 'Adamax', 'LBFGS']")

    log_list = []
    best_loss = float("inf")
    best_epoch = 0
    # ---- Track best in-memory state + patience counter ----
    best_state_in_mem: Optional[dict] = None
    epochs_since_best = 0

    if patience is not None:
        print(f"[EarlyStop] Patience-based early stopping ENABLED "
              f"(patience={patience} epochs).")

    for epoch in range(1, epochs + 1):
        model.train()

        # Adam-family optimizers use the same training logic (Adam, AdamW, RAdam, NAdam, Adamax)
        if opt_upper in ("ADAM", "ADAMW", "RADAM", "NADAM", "ADAMAX"):
            optimizer.zero_grad(set_to_none=True)
            losses = model.loss(N_samples=N_samples)
            total = losses["total"]
            if torch.isnan(total) or torch.isinf(total):
                print(f"[WARNING] epoch {epoch} NaN/Inf loss detected, skipping update")
                continue
            total.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                print(f"[WARNING] epoch {epoch} NaN/Inf gradient detected, skipping update")
                continue
            optimizer.step()
            current_loss = float(total.item())
        else:
            # Generate fixed sampling points for the entire epoch to ensure LBFGS convergence consistency
            x_samples = model._train_samples(N_samples)

            closure = LBFGSClosure(model, optimizer, x_samples)
            optimizer.step(closure)
            # Compute the final loss using the same sampling points
            loss_comp = model.weighted_loss_func.compute_total_loss(x_samples, model.fields_and_grads)
            losses = {
                "Pi_str": loss_comp["Pi_str"],
                "Pi_T": loss_comp["Pi_str_T"],
                "Pi_e": loss_comp["Pi_e"],
                "Pi_all": loss_comp["Pi_all"],
                "bc": loss_comp["bc"],
                "total": loss_comp["total"],
            }
            current_loss = float(losses["total"].item())
            if torch.isnan(losses["total"]) or torch.isinf(losses["total"]):
                print(f"[WARNING] epoch {epoch} NaN/Inf loss detected, skipping logging")
                continue

        if current_loss < best_loss:
            best_loss = current_loss
            best_epoch = epoch
            # ---- Keep best state in memory (for restore_best) ----
            best_state_in_mem = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            epochs_since_best = 0
            if best_model_path:
                try:
                    torch.save(model.state_dict(), best_model_path)
                except Exception as e:
                    print(f"[WARNING] Failed to save model (epoch {epoch}): {e}")
        else:
            epochs_since_best += 1
            if patience is not None and epochs_since_best >= patience:
                print(f"[EarlyStop] No loss improvement for {patience} "
                      f"consecutive epochs (best @ epoch {best_epoch}, "
                      f"loss={best_loss:.4e}). Stopping at epoch {epoch} / "
                      f"{epochs}.")
                break

        if epoch % print_every == 0 or epoch == 1 or epoch == epochs:
            print("=" * 60)
            msg = (
                f"[{epoch:5d}/{epochs}] loss={current_loss:.4e} "
                f"(best: {best_loss:.4e} @ epoch: {best_epoch})  "
                f"Pi_str={losses['Pi_str'].item():.4e}  "
            )
            if "Pi_T" in losses and float(losses["Pi_T"].item()) != 0.0:
                msg += f"Pi_T={losses['Pi_T'].item():.4e}  "
            msg += (
                f"Pi_e={losses['Pi_e'].item():.4e}  "
                f"Pi_all={losses['Pi_all'].item():.4e}  "
                f"bc={losses['bc'].item():.4e}"
            )
            print(msg)

            # Print GPU status information every 1000 iterations
            if epoch % 1000 == 0 and epoch > 0:
                try:
                    # Lazily import the GPU status retrieval function
                    try:
                        from ..utils.gpu_monitor import get_gpu_status_string
                    except ImportError:
                        import sys, os
                        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                        from utils.gpu_monitor import get_gpu_status_string

                    gpu_status = get_gpu_status_string()
                    print(f"      [GPU Status] {gpu_status}")
                except Exception:
                    # Fall back to a simple GPU memory display
                    if torch.cuda.is_available():
                        gpu_id = torch.cuda.current_device()
                        gpu_name = torch.cuda.get_device_name(gpu_id)
                        gpu_memory_allocated = torch.cuda.memory_allocated(gpu_id) / 1024**3
                        gpu_memory_reserved = torch.cuda.memory_reserved(gpu_id) / 1024**3
                        print(f"      [GPU Status] {gpu_name}: Allocated {gpu_memory_allocated:.2f}GB / Reserved {gpu_memory_reserved:.2f}GB")

        # Log columns: [epoch, total, Pi_all, bc, Pi_str, Pi_str_T, Pi_e]
        log_list.append([
            epoch,
            current_loss,
            float(losses.get("Pi_all").item()) if isinstance(losses.get("Pi_all"), torch.Tensor) else float(losses.get("Pi_all", float("nan"))),
            float(losses.get("bc").item()) if isinstance(losses.get("bc"), torch.Tensor) else float(losses.get("bc", float("nan"))),
            float(losses.get("Pi_str").item()) if isinstance(losses.get("Pi_str"), torch.Tensor) else float(losses.get("Pi_str", float("nan"))),
            float(losses.get("Pi_T").item()) if isinstance(losses.get("Pi_T"), torch.Tensor) else float(losses.get("Pi_T", 0.0)),
            float(losses.get("Pi_e").item()) if isinstance(losses.get("Pi_e"), torch.Tensor) else float(losses.get("Pi_e", float("nan"))),
        ])

    # ---- Restore best-loss state if requested ----
    if restore_best and best_state_in_mem is not None:
        model.load_state_dict(best_state_in_mem)
        print(f"[ModelSelect] restore_best=True: reloaded best state "
              f"(loss={best_loss:.4e} @ epoch {best_epoch}).")
    elif not restore_best:
        print(f"[ModelSelect] restore_best=False: keeping FINAL-epoch state.")

    return np.array(log_list, dtype=np.float64), best_loss, best_epoch


# ========== Training and evaluation helper functions ==========


def build_model(
    problem: str,
    *,
    coeffs: MaterialCoeffs,
    params: PhysicalParams,
    bc: BoundaryConditions,
    device: torch.device,
    bc_weight: float,
    encoder_dims_shared: Optional[list] = None,
    head_dims: Optional[list] = None,
    in_dim: int = 1,
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
    # Activation function parameters
    activation_type: str = "Tanh",
    siren_omega_0: float = 30.0,
    siren_omega_hidden: float = 30.0,
    # Lifting basis function parameter
    lifting_basis: str = "poly",
) -> EnergyPINNStatic:
    """Build linear or nonlinear model."""

    is_nonlinear = problem.lower() == "nonlinear"
    model = EnergyPINNStatic(
        coeffs,
        params,
        bc,
        device=device,
        bc_type=bc.type,
        bc_weight=bc_weight,
        is_nonlinear=is_nonlinear,
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
    return model


__all__ = ["EnergyPINNStatic", "as_fun", "train_model", "build_model"]