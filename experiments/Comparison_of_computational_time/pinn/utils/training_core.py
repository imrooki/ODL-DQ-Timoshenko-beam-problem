#!/usr/bin/env python3
"""
Timoshenko Beam PINNs Training Core Module

Author: Yang
Version: 3.0

Function Description:
- Extract and encapsulate core training logic from main.py for code reuse and modularization
- Support unified calling from multiple entry points (main.py, sensitivity_analysis.py, etc.)
- Implement separate result storage for different tasks through script_name parameter
- Integrate complete parameter validation, training process, and result management functions

Core Functions:
1. Unified training process: Parameter calculation -> Model training -> Result saving -> Performance analysis
2. Multi-entry support: Support different calling modes for main program and sensitivity analysis
3. Path management: Automatically organize output directory structure based on script_name
4. Exception handling: Complete error capture, memory management, and resource cleanup
5. Result validation: Integrity check and quality assessment of training results

Technical Features:
- Parameter dictionary validation: Ensure integrity and validity of all required parameters
- GPU monitoring integration: Real-time monitoring of hardware resource usage during training
- Output management: Standardized file naming and directory structure management
- Memory optimization: Memory usage monitoring and optimization during training
- Logging: Detailed training process recording and performance analysis

Design Principles:
- Highly modular: Core logic decoupled from specific applications
- Unified interface: Provide consistent API calling methods
- Error-friendly: Detailed error messages and recovery suggestions
- Performance-first: Optimized training process and resource management
"""

import os
import sys
import numpy as np
import torch
from typing import Dict, Any, Optional, Tuple
from contextlib import nullcontext

# Allow loading OpenMP library on Windows to avoid duplicate library conflicts
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from modules.solver import as_fun
from modules.data_types import MaterialCoeffs, PhysicalParams
from modules.bc import make_bc_spec
from modules.pseudo_trainer import run_dual_pseudo_transfer

from utils.material_properties import (
    compute_material_params_for_solver,
    create_material_calculator,
)
from utils.experiment import (
    summarize_results,
    update_index_via_manager,
)
from utils.output_manager import OutputManager
from utils.common import set_seed, get_device, print_config
from utils.gpu_monitor import GPUMonitor


def run_training_core(
    params_dict: Dict[str, Any],
    script_name: str = "main",
    verbose: bool = True,
    base_dir: str = "results"
) -> Dict[str, Any]:
    """
    Core training function that can be shared and called from multiple entry points

    Parameters:
        params_dict: Dictionary containing all training parameters
            Required parameters: 'h', 'L', 'W_Gr', 'H_Gr', 'T', 'q', 'bc_type', 'distribution'
            Optional parameters: 'epochs', 'lr', 'optimizer_type', 'k1', 'k2', etc.
        script_name: Name of the calling script, which determines the output path (main/sensitivity_analysis, etc.)
            Key parameter that determines results are stored under {base_dir}/{script_name}/
        verbose: Whether to output detailed information
        base_dir: Output root directory, defaults to "results", can be customized to another path

    Returns:
        dict: Dictionary containing the training results
            - success: bool, whether the training succeeded
            - results: dict, displacement field results {"linear": (u,w,phi), "nonlinear": (u,w,phi)}
                each element is a numpy array of length 201 (default number of evaluation points)
            - logs: dict, training logs {"linear": array, "nonlinear": array}
                array format: [epoch, total_loss, Pi_all, bc_loss, Pi_str, Pi_str_T, Pi_w, Pi_e]
            - summary: dict, result summary (max, min, mean, standard deviation, etc.)
            - output_manager: OutputManager, the output manager instance
            - filename_prefix: str, filename prefix (formatted parameter combination)
            - material_params: dict, the computed material parameters
            - error: str, error message (present only when success=False)
    """

    if verbose:
        print(f"\nSTART: Starting training core (script: {script_name})")

    # Import exception handling system
    try:
        from utils.exceptions import ExceptionHandler, ConfigurationError, safe_execute
        from utils.memory_manager import get_memory_manager
        from utils.numerical_safety import check_nan_inf
        handler = ExceptionHandler()
        memory_mgr = get_memory_manager()
    except ImportError as e:
        if verbose:
            print(f"Warning: Unable to import safety tools: {e}")
        handler = None
        memory_mgr = None

    try:
        # Set random seed and computing device
        set_seed(params_dict.get('seed', 42))
        device = get_device()
        if verbose:
            print(f"Device: {device}")

        # Build configuration info dictionary
        integrator = params_dict.get("integrator", "mc")
        int_norm = str(integrator).lower().replace("_", "").replace("-", "").replace(" ", "")
        int_display = (
            f"{integrator} (max_points={params_dict.get('agq_max_points', 'NA')})"
            if int_norm == "agq"
            else f"{integrator} (N_train={params_dict.get('N_train', 'NA')})"
        )

        # Configure sampler display based on integrator mode
        if int_norm == "agq":
            sampler_display = "N/A (AGQ uses adaptive quadrature nodes)"
            sampler_reuse_display = "N/A (AGQ uses cached nodes)"
        else:
            sampler_display = params_dict.get("sampler", "uniform")
            sampler_reuse_display = params_dict.get("sampler_reuse", False)

        # Configure learning rate display based on adaptive lr setting
        use_adaptive_lr = params_dict.get('use_adaptive_lr', False)
        if use_adaptive_lr:
            lr_display = (
                f"adaptive (early: {params_dict.get('lr_early_max', 1e-3):.0e}-{params_dict.get('lr_early_min', 2e-4):.0e}, "
                f"mid: {params_dict.get('lr_mid_max', 2e-4):.0e}-{params_dict.get('lr_mid_min', 1e-4):.0e}, "
                f"late: {params_dict.get('lr_late_fixed', 1e-4):.0e} fixed)"
            )
        else:
            lr_display = f"{params_dict.get('lr', 1e-4):.0e} (fixed)"

        config_dict = {
            "Geometry": f"h={params_dict['h']}, L={params_dict['L']}",
            "Material": f"W_Gr={params_dict['W_Gr']}, H_Gr={params_dict['H_Gr']}, T={params_dict['T']}",
            "Load": f"q={params_dict['q']}",
            "BC": params_dict['bc_type'],
            "Training": f"epochs={params_dict.get('epochs', 30000)}, lr={lr_display}, optimizer={params_dict.get('optimizer_type', 'Adam')}",
            "Network": f"shared encoder={params_dict.get('encoder_dims_shared', [1, 32, 64, 128])}, head={params_dict.get('head_dims', [128, 64, 32, 1])}",
            "InputDim": params_dict.get("input_dim", 1),
            "Sampler": sampler_display,
            "SamplerReuse": sampler_reuse_display,
            "Integrator": int_display,
        }

        # Add adaptive lr details if enabled (v3.0 - pure dynamic threshold)
        if use_adaptive_lr:
            config_dict["AdaptiveLR"] = (
                f"dynamic thresholds (warmup={params_dict.get('lr_warmup_epochs', 100)}, "
                f"ratios={params_dict.get('lr_early_ratio', 0.6)}/{params_dict.get('lr_mid_ratio', 0.85)}, "
                f"min_epochs={params_dict.get('lr_min_early_epochs', 1000)}/{params_dict.get('lr_min_mid_epochs', 5000)}, "
                f"patience={params_dict.get('lr_patience', 500)})"
            )

        if int_norm == "agq":
            config_dict["AGQ"] = (
                f"rule={params_dict.get('agq_rule','G10K21')}, "
                f"abs_tol={params_dict.get('agq_abs_tol',1e-6)}, rel_tol={params_dict.get('agq_rel_tol',1e-4)}, "
                f"max_points={params_dict.get('agq_max_points',4096)}, max_depth={params_dict.get('agq_max_depth',100)}, "
                f"refine_every={params_dict.get('agq_refine_every',0)}, fail_policy={params_dict.get('agq_fail_policy','use_partial')}"
            )

        if verbose:
            print_config(config_dict)

        # Calculate material parameters
        if verbose:
            print("\n[INFO] Calculating material parameters...")

        material_params = compute_material_params_for_solver(
            h=params_dict['h'],
            L=params_dict['L'],
            num_layers=params_dict.get('num_layers', 10),
            W_Gr=params_dict['W_Gr'],
            H_Gr=params_dict['H_Gr'],
            T=params_dict['T'],
            distribution_type=params_dict['distribution'],
            q=params_dict['q'],
        )

        # Detailed material property summary (optional)
        if verbose:
            try:
                calculator = create_material_calculator()
                calculator.print_material_summary(
                    h=params_dict['h'],
                    L=params_dict['L'],
                    num_layers=params_dict.get('num_layers', 10),
                    W_Gr=params_dict['W_Gr'],
                    H_Gr=params_dict['H_Gr'],
                    T=params_dict['T'],
                    distribution_type=params_dict['distribution'],
                )

                # Print elastic foundation parameters
                print("\nElastic foundation parameters:")
                print(f"  k1 = {params_dict.get('k1', 0.0):.4f} (Winkler foundation stiffness)")
                print(f"  k2 = {params_dict.get('k2', 0.0):.4f} (Pasternak foundation stiffness)")

            except Exception as e:
                print(f"Warning: Material summary failed: {e}")

        # Create output manager (key: use passed script_name)
        param_folder = OutputManager.make_param_folder(
            params_dict['W_Gr'],
            params_dict['T'],
            params_dict['H_Gr'],
            params_dict['q'],
            material_params["lambda_val"],
            k1=params_dict.get('k1', 0.0),
            k2=params_dict.get('k2', 0.0),
            activation_type=params_dict.get('activation_type', 'Tanh'),
            siren_omega_0=params_dict.get('siren_omega_0', 30.0),
            siren_omega_hidden=params_dict.get('siren_omega_hidden', 30.0),
            lifting_basis=params_dict.get('lifting_basis', 'poly'),
        )

        output_manager = OutputManager(
            base_dir=base_dir,  # Use the passed-in base_dir parameter
            script_name=script_name,  # Key: use passed script_name
            bc_type=params_dict['bc_type'],
            distribution=params_dict['distribution'],
            param_folder=param_folder,
        )

        filename_prefix = output_manager.generate_filename(
            params_dict['W_Gr'],
            params_dict['T'],
            params_dict['H_Gr'],
            params_dict['q'],
            k1=params_dict.get('k1', 0.0),
            k2=params_dict.get('k2', 0.0),
            activation_type=params_dict.get('activation_type', 'Tanh'),
            siren_omega_0=params_dict.get('siren_omega_0', 30.0),
            siren_omega_hidden=params_dict.get('siren_omega_hidden', 30.0),
            lifting_basis=params_dict.get('lifting_basis', 'poly'),
        )

        # Material stiffness coefficients and physical parameters encapsulation
        coeffs = MaterialCoeffs(
            a11=as_fun(material_params["a11"], "a11"),
            b11=as_fun(material_params["b11"], "b11"),
            d11=as_fun(material_params["d11"], "d11"),
            a55=as_fun(material_params["a55"], "a55"),
        )

        params_obj = PhysicalParams(
            alpha_t=material_params["alpha_effective"],
            DeltaT=material_params["delta_T"],
            lambda_val=material_params["lambda_val"],
            q=material_params["q"],
            n_xT=material_params["n_xT"],
            m_xT=material_params["m_xT"],
            # Elastic foundation parameters
            k1=params_dict.get('k1', 0.0),
            k2=params_dict.get('k2', 0.0),
        )

        # Boundary condition specification and evaluation grid points
        bc = make_bc_spec(params_dict['bc_type'])
        x_eval = torch.linspace(0.0, 1.0, 201, device=device).reshape(-1, 1)
        x_eval.requires_grad_(True)

        results = {}
        logs = {}

        # Dual model pseudo-supervised training (with GPU monitoring)
        if verbose:
            print("\n[INFO] Starting dual model pseudo-supervised training..")

        gpu_log_path: Optional[str] = None
        monitor_context = nullcontext()
        monitor_active = False

        disable_gpu_monitor = (
            params_dict.get('disable_gpu_monitor', True)
            or str(os.environ.get('GPUMONITOR_DISABLE', '')).strip().lower() in {'1', 'true', 'yes'}
        )

        if device.type == "cuda" and not disable_gpu_monitor:
            gpu_log_path = os.path.join(output_manager.logs_dir, "gpu_usage.csv")
            try:
                monitor_context = GPUMonitor(
                    interval=2.0,
                    output_csv_path=gpu_log_path,
                    log_to_console=False,
                )
                monitor_active = True
            except Exception as exc:
                monitor_context = nullcontext()
                if verbose:
                    print(f"[WARN] GPU monitor unavailable: {exc}; monitoring disabled.")

        # Branch on optimizer_type: LBFGS bypasses the dual loop and goes
        # through sequential trainer (linear → nonlinear), since the upstream
        # pseudo_trainer hard-rejects LBFGS in dual-supervision mode. Adam
        # family keeps the original dual code path.
        _opt_type_branch = str(params_dict.get('optimizer_type', 'Adam')).upper()
        _use_sequential = _opt_type_branch == 'LBFGS'
        if _use_sequential:
            from modules.sequential_trainer import run_sequential_training

        def _run_dual_training():
            if _use_sequential:
                return run_sequential_training(
                    coeffs=coeffs,
                    params_obj=params_obj,
                    bc=bc,
                    device=device,
                    encoder_dims_shared=params_dict.get('encoder_dims_shared', [1, 32, 64, 128]),
                    head_dims=params_dict.get('head_dims', [128, 64, 32, 1]),
                    in_dim=params_dict.get("input_dim", 1),
                    epochs=params_dict.get('epochs', 30000),
                    N_train=params_dict.get('N_train', 10000),
                    lr=params_dict.get('lr', 1.0),
                    print_every=params_dict.get('print_every', 1000),
                    bc_weight=params_dict.get('bc_weight', 1000.0),
                    optimizer_type=params_dict.get('optimizer_type', 'LBFGS'),
                    lbfgs_max_iter=params_dict.get('lbfgs_max_iter', 20),
                    lbfgs_history_size=params_dict.get('lbfgs_history_size', 100),
                    lbfgs_line_search_fn=params_dict.get('lbfgs_line_search_fn', 'strong_wolfe'),
                    adamw_weight_decay=params_dict.get('adamw_weight_decay', 1e-4),
                    sampler=params_dict.get('sampler', 'uniform'),
                    sampler_reuse=params_dict.get('sampler_reuse', False),
                    integrator=params_dict.get('integrator', 'mc'),
                    agq_rule=params_dict.get('agq_rule', 'G10K21'),
                    agq_abs_tol=params_dict.get('agq_abs_tol', 1e-6),
                    agq_rel_tol=params_dict.get('agq_rel_tol', 1e-4),
                    agq_max_points=params_dict.get('agq_max_points', 4096),
                    agq_max_depth=params_dict.get('agq_max_depth', 100),
                    agq_refine_every=params_dict.get('agq_refine_every', 0),
                    agq_fail_policy=params_dict.get('agq_fail_policy', 'use_partial'),
                    x_eval=x_eval,
                    activation_type=params_dict.get('activation_type', 'Tanh'),
                    siren_omega_0=params_dict.get('siren_omega_0', 30.0),
                    siren_omega_hidden=params_dict.get('siren_omega_hidden', 30.0),
                    lifting_basis=params_dict.get('lifting_basis', 'poly'),
                    patience=params_dict.get('patience', None),
                    restore_best=params_dict.get('restore_best', True),
                    seed=int(params_dict.get('seed', 42)),
                )
            return run_dual_pseudo_transfer(
                coeffs=coeffs,
                params_obj=params_obj,
                bc=bc,
                device=device,
                encoder_dims_shared=params_dict.get('encoder_dims_shared', [1, 32, 64, 128]),
                head_dims=params_dict.get('head_dims', [128, 64, 32, 1]),
                in_dim=params_dict.get("input_dim", 1),
                epochs=params_dict.get('epochs', 30000),
                N_train=params_dict.get('N_train', 10000),
                lr=params_dict.get('lr', 1e-4),
                print_every=params_dict.get('print_every', 1000),
                bc_weight=params_dict.get('bc_weight', 1000.0),
                optimizer_type=params_dict.get('optimizer_type', 'Adam'),
                lbfgs_max_iter=params_dict.get('lbfgs_max_iter', 20),
                lbfgs_history_size=params_dict.get('lbfgs_history_size', 50),
                lbfgs_line_search_fn=params_dict.get('lbfgs_line_search_fn', None),
                adamw_weight_decay=params_dict.get('adamw_weight_decay', 1e-4),
                sampler=params_dict.get('sampler', 'uniform'),
                sampler_reuse=params_dict.get('sampler_reuse', False),
                integrator=params_dict.get('integrator', 'mc'),
                agq_rule=params_dict.get('agq_rule', 'G10K21'),
                agq_abs_tol=params_dict.get('agq_abs_tol', 1e-6),
                agq_rel_tol=params_dict.get('agq_rel_tol', 1e-4),
                agq_max_points=params_dict.get('agq_max_points', 4096),
                agq_max_depth=params_dict.get('agq_max_depth', 100),
                agq_refine_every=params_dict.get('agq_refine_every', 0),
                agq_fail_policy=params_dict.get('agq_fail_policy', 'use_partial'),
                # Pseudo-supervision toggle parameters
                use_pseudo_supervision=params_dict.get('use_pseudo_supervision', True),
                ps_w_non_start=params_dict.get('ps_w_non_start', 1.0),
                ps_w_lin_start=params_dict.get('ps_w_lin_start', 0.5),
                ps_cut_ratio=params_dict.get('ps_cut_ratio', 0.8),
                ps_use_phi=params_dict.get('ps_use_phi', False),
                transfer_alpha=params_dict.get('transfer_alpha', 0.3),
                transfer_ratio=params_dict.get('transfer_ratio', 0.7),
                transfer_freq=params_dict.get('transfer_freq', 500),
                transfer_cut_ratio=params_dict.get('transfer_cut_ratio', 0.2),
                # Adaptive learning rate parameters (v3.0 - pure dynamic threshold)
                use_adaptive_lr=params_dict.get('use_adaptive_lr', False),
                lr_early_max=params_dict.get('lr_early_max', 1e-3),
                lr_early_min=params_dict.get('lr_early_min', 2e-4),
                lr_mid_max=params_dict.get('lr_mid_max', 2e-4),
                lr_mid_min=params_dict.get('lr_mid_min', 1e-4),
                lr_late_fixed=params_dict.get('lr_late_fixed', 1e-4),
                lr_patience=params_dict.get('lr_patience', 500),
                lr_improvement_threshold=params_dict.get('lr_improvement_threshold', 1e-6),
                lr_decay_factor=params_dict.get('lr_decay_factor', 0.5),
                lr_verbose=params_dict.get('lr_verbose', False),
                # Dynamic threshold calibration parameters (v3.0)
                lr_warmup_epochs=params_dict.get('lr_warmup_epochs', 100),
                lr_early_ratio=params_dict.get('lr_early_ratio', 0.6),
                lr_mid_ratio=params_dict.get('lr_mid_ratio', 0.85),
                lr_min_early_epochs=params_dict.get('lr_min_early_epochs', 1000),
                lr_min_mid_epochs=params_dict.get('lr_min_mid_epochs', 5000),
                x_eval=x_eval,
                # epoch callback (for external snapshot saving, etc.)
                epoch_callback=params_dict.get('epoch_callback', None),
                # Activation function parameters
                activation_type=params_dict.get('activation_type', 'Tanh'),
                siren_omega_0=params_dict.get('siren_omega_0', 30.0),
                siren_omega_hidden=params_dict.get('siren_omega_hidden', 30.0),
                # Lifting basis function parameter
                lifting_basis=params_dict.get('lifting_basis', 'poly'),
                # ---- Early-stopping & best/final selection ----
                patience=params_dict.get('patience', None),
                restore_best=params_dict.get('restore_best', True),
            )

        try:
            with monitor_context:
                result = _run_dual_training()
        except Exception as exc:
            if monitor_active:
                if verbose:
                    print(f"[WARN] GPU monitor failed: {exc}; retrying without monitoring.")
                monitor_context = nullcontext()
                monitor_active = False
                result = _run_dual_training()
            else:
                raise

        # Extract training results
        fields_lin = result.get("fields_linear")
        fields_non = result.get("fields_nonlinear")
        logs["linear"] = result.get("logs_linear")
        logs["nonlinear"] = result.get("logs_nonlinear")

        if fields_lin is not None:
            u_lin = fields_lin["u"].detach().cpu().numpy().flatten()
            w_lin = fields_lin["w"].detach().cpu().numpy().flatten()
            phi_lin = fields_lin["phi"].detach().cpu().numpy().flatten()
            results["linear"] = (u_lin, w_lin, phi_lin)

        if fields_non is not None:
            u_non = fields_non["u"].detach().cpu().numpy().flatten()
            w_non = fields_non["w"].detach().cpu().numpy().flatten()
            phi_non = fields_non["phi"].detach().cpu().numpy().flatten()
            results["nonlinear"] = (u_non, w_non, phi_non)

        # Synchronize CUDA operations to ensure data transfer is complete
        if device.type == "cuda":
            torch.cuda.synchronize()

        # Enhanced tensor cleanup strategy (avoids memory leaks)
        # Only clear gradients for leaf tensors (avoids warnings on non-leaf .grad).
        if fields_lin is not None:
            for k in list(fields_lin.keys()):
                # Only leaf tensors (such as model parameters) have gradients that need to be cleared
                # Non-leaf tensors (such as intermediate results returned by fields_and_grads) do not retain gradients by default
                if hasattr(fields_lin[k], 'is_leaf') and fields_lin[k].is_leaf and hasattr(fields_lin[k], 'grad'):
                    fields_lin[k].grad = None
                fields_lin[k] = None
        if fields_non is not None:
            for k in list(fields_non.keys()):
                if hasattr(fields_non[k], 'is_leaf') and fields_non[k].is_leaf and hasattr(fields_non[k], 'grad'):
                    fields_non[k].grad = None
                fields_non[k] = None
        del fields_lin, fields_non

        # Force garbage collection
        import gc
        gc.collect()

        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        # Save models
        try:
            lin_model = result.get("model_linear")
            non_model = result.get("model_nonlinear")

            # Windows MAX_PATH limit (260 characters), reserve a safety margin
            MAX_PATH_LIMIT = 255

            # Helper function: generate the model save path, automatically handling overly long paths
            def get_model_path(models_dir: str, prefix: str, short_prefix: str, filename: str) -> str:
                """Generate the model path; automatically use a shortened prefix or hashed filename when the path is too long"""
                # Strategy 1: full prefix + full filename
                full_path = os.path.join(models_dir, f"{prefix}_{filename}.pth")
                abs_full_path = os.path.abspath(full_path)

                if len(abs_full_path) <= MAX_PATH_LIMIT:
                    return full_path

                # Strategy 2: shortened prefix + full filename
                short_path = os.path.join(models_dir, f"{short_prefix}_{filename}.pth")
                abs_short_path = os.path.abspath(short_path)

                if len(abs_short_path) <= MAX_PATH_LIMIT:
                    if verbose:
                        print(f"[INFO] Path too long ({len(abs_full_path)} chars), using short prefix: {short_prefix}")
                    return short_path

                # Strategy 3: shortened prefix + hashed filename (final fallback)
                import hashlib
                # Compute the MD5 hash of the filename, take the first 12 characters
                hash_str = hashlib.md5(filename.encode()).hexdigest()[:12]
                hashed_path = os.path.join(models_dir, f"{short_prefix}_{hash_str}.pth")

                if verbose:
                    print(f"[WARN] Path extremely long ({len(abs_short_path)} chars). Using hashed filename: {os.path.basename(hashed_path)}")
                    # For easier later correspondence, write the mapping to mapping.txt in the same directory
                    try:
                        mapping_file = os.path.join(models_dir, "filename_mapping.txt")
                        safe_mapping_file = output_manager.get_safe_file_path(mapping_file)
                        with open(safe_mapping_file, "a") as f:
                            f.write(f"{short_prefix}_{hash_str}.pth -> {prefix}_{filename}.pth\n")
                    except Exception:
                        pass # Ignore mapping-file write errors

                return hashed_path

            # Save linear model (move to CPU before saving).
            if lin_model is not None:
                linear_model_path = get_model_path(
                    output_manager.models_dir, "Linearw", "Lw", filename_prefix
                )

                # Ensure the models directory exists (use safe paths to handle the Windows long-path issue)
                safe_models_dir = output_manager.get_safe_file_path(output_manager.models_dir)
                os.makedirs(safe_models_dir, exist_ok=True)

                # Move to CPU (avoids GPU memory usage)
                lin_model_cpu = lin_model.cpu()
                safe_linear_path = output_manager.get_safe_file_path(linear_model_path)
                torch.save(lin_model_cpu.state_dict(), safe_linear_path)

                # Immediately clean up the CPU copy
                del lin_model_cpu
                gc.collect()

                if verbose:
                    print(f"[OK] linear model saved: {linear_model_path}")
            else:
                if verbose:
                    print("[WARN] linear model is None; skip saving")

            # Save nonlinear model (move to CPU before saving).
            if non_model is not None:
                nonlinear_model_path = get_model_path(
                    output_manager.models_dir, "Nonlinearw", "NLw", filename_prefix
                )

                # Ensure the models directory exists (use safe paths to handle the Windows long-path issue)
                safe_models_dir = output_manager.get_safe_file_path(output_manager.models_dir)
                os.makedirs(safe_models_dir, exist_ok=True)

                # Move to CPU (avoids GPU memory usage)
                non_model_cpu = non_model.cpu()
                safe_nonlinear_path = output_manager.get_safe_file_path(nonlinear_model_path)
                torch.save(non_model_cpu.state_dict(), safe_nonlinear_path)

                # Immediately clean up the CPU copy
                del non_model_cpu
                gc.collect()

                if verbose:
                    print(f"[OK] Nonlinear model saved: {nonlinear_model_path}")
            else:
                if verbose:
                    print("[WARN] Nonlinear model is None; skip saving")

        except Exception as e:
            error_msg = f"Failed to save model: {e}"
            print(f"[ERROR] {error_msg}")
            # Log serious errors even in non-verbose mode
            if verbose:
                import traceback
                print(f"Detailed error: {traceback.format_exc()}")
            # Continue execution but log error
        finally:
            # Clean up model references and GPU memory after saving
            del lin_model, non_model
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

        # Save calculation results and data
        if verbose:
            print("\n[SAVE] Saving results...")

        x_numpy = x_eval.detach().cpu().numpy().flatten()

        # Save displacement data
        try:
            # Ensure the data directory exists

            output_manager.save_displacement_data(
                x_numpy,
                linear_results=results.get("linear"),
                nonlinear_results=results.get("nonlinear"),
                filename=filename_prefix,
            )
            if verbose:
                print(f"[OK] Displacement data saved: w_{filename_prefix}.csv")
        except Exception as e:
            print(f"[ERROR] Failed to save displacement data: {e}")
            if verbose:
                import traceback
                print(f"Detailed error: {traceback.format_exc()}")

        # Save training logs
        try:
            # Ensure the logs directory exists

            output_manager.save_loss_log(
                linear_log=logs.get("linear"),
                nonlinear_log=logs.get("nonlinear"),
                filename=filename_prefix,
            )
            if verbose:
                print(f"[OK] Training log saved: {filename_prefix}_loss.csv")
        except Exception as e:
            print(f"[ERROR] Failed to save training log: {e}")
            if verbose:
                import traceback
                print(f"Detailed error: {traceback.format_exc()}")

        # Generate plots (if needed)
        if params_dict.get("generate_plots", True):
            # Ensure the plots directory exists

            output_manager.plot_training_curves(
                linear_log=logs.get("linear"),
                nonlinear_log=logs.get("nonlinear"),
                filename=filename_prefix,
            )
            output_manager.plot_displacement_comparison(
                x_numpy,
                linear_results=results.get("linear"),
                nonlinear_results=results.get("nonlinear"),
                filename=filename_prefix,
            )

        # Result summary
        summary_dict = summarize_results(results.get("linear"), results.get("nonlinear"))

        if logs.get("linear") is not None and summary_dict.get("linear") is not None:
            summary_dict["linear"]["final_loss"] = float(logs["linear"][-1, 1])

        if logs.get("nonlinear") is not None and summary_dict.get("nonlinear") is not None:
            summary_dict["nonlinear"]["final_loss"] = float(logs["nonlinear"][-1, 1])

        if verbose:
            output_manager.print_summary(summary_dict)

        # Update index
        try:
            import hashlib
            import json as _json

            def _to_jsonable(obj: Any) -> Any:
                if obj is None or isinstance(obj, (bool, int, float, str)):
                    return obj
                if isinstance(obj, (list, tuple)):
                    return [_to_jsonable(v) for v in obj]
                if isinstance(obj, dict):
                    return {str(k): _to_jsonable(v) for k, v in obj.items()}

                # numpy scalars/arrays
                try:
                    import numpy as _np

                    if isinstance(obj, _np.generic):
                        return obj.item()
                    if isinstance(obj, _np.ndarray):
                        return obj.tolist()
                except Exception:
                    pass

                # torch tensors
                try:
                    import torch as _torch

                    if isinstance(obj, _torch.Tensor):
                        return obj.detach().cpu().tolist()
                except Exception:
                    pass

                # paths
                try:
                    from pathlib import Path as _Path

                    if isinstance(obj, _Path):
                        return str(obj)
                except Exception:
                    pass

                return str(obj)

            full_params = {str(k): _to_jsonable(v) for k, v in params_dict.items()}
            hash_src = _json.dumps(full_params, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
            run_hash = hashlib.sha1(hash_src.encode("utf-8")).hexdigest()[:12]

            # Backward-compatible flat keys + richer audit payload.
            run_params = {
                "run_hash": run_hash,
                # Physical/material params
                "h": _to_jsonable(params_dict.get("h")),
                "L": _to_jsonable(params_dict.get("L")),
                "L_factor": _to_jsonable(params_dict.get("L_factor")),
                "num_layers": int(params_dict.get("num_layers", 10)),
                "W_Gr": _to_jsonable(params_dict.get("W_Gr")),
                "H_Gr": _to_jsonable(params_dict.get("H_Gr")),
                "T": _to_jsonable(params_dict.get("T")),
                "q": _to_jsonable(params_dict.get("q")),
                "bc_type": _to_jsonable(params_dict.get("bc_type")),
                "distribution": _to_jsonable(params_dict.get("distribution")),
                "k1": _to_jsonable(params_dict.get("k1", 0.0)),
                "k2": _to_jsonable(params_dict.get("k2", 0.0)),
                # Network params
                "network_arch": _to_jsonable(params_dict.get("network_arch", "shared")),
                "encoder_dims_shared": _to_jsonable(params_dict.get("encoder_dims_shared", [1, 32, 64, 128])),
                "head_dims": _to_jsonable(params_dict.get("head_dims", [128, 64, 32, 1])),
                "input_dim": _to_jsonable(params_dict.get("input_dim", params_dict.get("in_dim", 1))),
                "activation_type": _to_jsonable(params_dict.get("activation_type", "Tanh")),
                "siren_omega_0": _to_jsonable(params_dict.get("siren_omega_0", 30.0)),
                "siren_omega_hidden": _to_jsonable(params_dict.get("siren_omega_hidden", 30.0)),
                "lifting_basis": _to_jsonable(params_dict.get("lifting_basis", "poly")),
                # Training params (for audit)
                "epochs": _to_jsonable(params_dict.get("epochs")),
                "lr": _to_jsonable(params_dict.get("lr")),
                "seed": _to_jsonable(params_dict.get("seed", 42)),
                "bc_weight": _to_jsonable(params_dict.get("bc_weight")),
                "integrator": _to_jsonable(params_dict.get("integrator", "mc")),
                "sampler": _to_jsonable(params_dict.get("sampler", "uniform")),
                "sampler_reuse": _to_jsonable(params_dict.get("sampler_reuse", False)),
                "N_train": _to_jsonable(params_dict.get("N_train")),
                "agq_rule": _to_jsonable(params_dict.get("agq_rule")),
                "agq_max_points": _to_jsonable(params_dict.get("agq_max_points")),
                # Full raw params snapshot (source of truth)
                "full_params": full_params,
            }

            material_keys = [
                "a11",
                "b11",
                "d11",
                "a55",
                "alpha_effective",
                "delta_T",
                "lambda_val",
                "n_xT",
                "m_xT",
            ]
            run_params["material_summary"] = {
                k: _to_jsonable(material_params.get(k)) for k in material_keys if k in material_params
            }

            update_index_via_manager(
                output_manager,
                filename_prefix,
                run_params,
                logs,
                results,
                plots_enabled=params_dict.get("generate_plots", True),
            )
        except Exception as e:
            if verbose:
                print(f"Warning: Failed to update index: {e}")

        if verbose:
            print("\n[SUCCESS] Training completed!")

        return {
            'success': True,
            'results': results,
            'logs': logs,
            'summary': summary_dict,
            'output_manager': output_manager,
            'filename_prefix': filename_prefix,
            'material_params': material_params
        }

    except Exception as e:
        error_msg = f"Error during training: {e}"
        if verbose:
            print(f"\n[ERROR] {error_msg}")
            import traceback
            traceback.print_exc()

        if handler:
            handler.handle_exception(e)

        return {
            'success': False,
            'error': error_msg,
            'results': {},
            'logs': {},
            'summary': {},
            'output_manager': None,
            'filename_prefix': None
        }


def validate_params_dict(params_dict: Dict[str, Any]) -> bool:
    """
    Validate the completeness of the parameter dictionary

    Parameters:
        params_dict: Parameter dictionary

    Returns:
        bool: Whether parameters are valid
    """
    required_params = [
        'h', 'L', 'W_Gr', 'H_Gr', 'T', 'q', 'bc_type', 'distribution'
    ]

    for param in required_params:
        if param not in params_dict:
            print(f"[ERROR] Missing required parameter: {param}")
            return False

    return True
