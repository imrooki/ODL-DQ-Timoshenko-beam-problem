

import torch
from typing import Any, Dict, Tuple, Optional
import warnings
import os
import sys


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.dq_core import get_cached_dq_system, cheb_lobatto_nodes
from modules.method_factory import MethodFactory
from modules.residuals import TimoshenkoBeamResiduals
from utils.solver_utils import inject_dirichlet
from utils.gpu_monitor import create_gpu_monitor
from utils.memory_manager import TensorPool, MemoryOptimizer
from utils.logger import ODILLogger
from utils.tensor_ops import (
    create_boundary_mask,
    create_optimizer,
    apply_gradient_clipping,
    initialize_parameters
)

torch.set_default_dtype(torch.float64)


def _get_config_value(config: Optional[Any], name: str, default=None):
    
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(name, default)
    return getattr(config, name, default)


def _compute_soft_bc_loss(u: torch.Tensor, w: torch.Tensor, phi: torch.Tensor,
                          A: torch.Tensor, bc_type: str, material_params: Dict,
                          device: torch.device, dtype: torch.dtype,
                          is_nonlinear: bool = False) -> torch.Tensor:
    
    loss_bc = torch.tensor(0.0, device=device, dtype=dtype)

    
    loss_bc += (u[0] - 0.0)**2 + (u[-1] - 0.0)**2
    loss_bc += (w[0] - 0.0)**2 + (w[-1] - 0.0)**2

    
    ux = A @ u
    wx = A @ w
    phix = A @ phi

    b11 = material_params['b11']
    d11 = material_params['d11']
    lambda_val = material_params['lambda_val']
    m_xT = material_params.get('m_xT', 0.0)

    if bc_type in ['H-H', 'S-S']:
        
        from utils.solver_utils import compute_moment
        M0 = compute_moment(ux, phix, wx, b11, d11, lambda_val, 0, is_nonlinear, m_xT)
        ML = compute_moment(ux, phix, wx, b11, d11, lambda_val, -1, is_nonlinear, m_xT)
        loss_bc += M0**2 + ML**2

    elif bc_type in ['C-S', 'C-H']:
        
        from utils.solver_utils import compute_moment
        loss_bc += phi[0]**2
        ML = compute_moment(ux, phix, wx, b11, d11, lambda_val, -1, is_nonlinear, m_xT)
        loss_bc += ML**2
    
    return loss_bc


def odil_loss(params_dict: Dict[str, torch.Tensor], x: torch.Tensor,
             A: torch.Tensor, B: torch.Tensor,
             material_params: Dict, q: float, bcs: Dict, bc_type: str,
             weights: Tuple[float, float, float] = (1.0, 1.0, 1.0),
             reg: float = 1e-12, is_nonlinear: bool = False,
             bc_weight: float = 10.0,
             A_interior: torch.Tensor = None, B_interior: torch.Tensor = None,
             use_interior_computation: bool = False,
             should_use_hard_constraint: bool = True,
             bc_weight_factor: float = 1.0,
             foundation_params: Optional[Dict] = None) -> torch.Tensor:
    
    N = len(x)
    device = x.device
    dtype = x.dtype
    
    
    if bc_type == 'C-C':
        
        u = inject_dirichlet(params_dict['u_param'], bcs['u'][0], bcs['u'][1])
        w = inject_dirichlet(params_dict['w_param'], bcs['w'][0], bcs['w'][1])
        phi = inject_dirichlet(params_dict['phi_param'], bcs['phi'][0], bcs['phi'][1])
    else:
        
        u = params_dict['u_param']
        w = params_dict['w_param']
        phi = params_dict['phi_param']

    foundation_params = foundation_params or {'k1': 0.0, 'k2': 0.0}
    k1 = foundation_params.get('k1', 0.0)  
    k2 = foundation_params.get('k2', 0.0)  
    residual_calc = TimoshenkoBeamResiduals(material_params, q, k1, k2)

    if use_interior_computation and A_interior is not None and B_interior is not None:
        
        R1, R2, R3 = residual_calc.compute(u, w, phi, A_interior, B_interior, is_nonlinear)
        loss_pde = (weights[0] * R1.pow(2).mean() +
                    weights[1] * R2.pow(2).mean() +
                    weights[2] * R3.pow(2).mean())
    else:
        
        R1, R2, R3 = residual_calc.compute(u, w, phi, A, B, is_nonlinear)
        mask = create_boundary_mask(N, device, dtype)
        loss_pde = (weights[0] * (R1 * mask).pow(2).mean() +
                    weights[1] * (R2 * mask).pow(2).mean() +
                    weights[2] * (R3 * mask).pow(2).mean())

    if not should_use_hard_constraint or bc_type != 'C-C':
        loss_bc = _compute_soft_bc_loss(u, w, phi, A, bc_type, material_params, device, dtype, is_nonlinear)
        effective_bc_weight = bc_weight * bc_weight_factor
        loss_bc *= effective_bc_weight
    else:
        loss_bc = torch.tensor(0.0, device=device, dtype=dtype)

    
    reg_loss = torch.tensor(0.0, device=device, dtype=dtype)
    if reg > 0:
        for param in params_dict.values():
            reg_loss += reg * param.pow(2).mean()
    
    return loss_pde + loss_bc + reg_loss


class ODILSolver:
    
    
    def __init__(self, N: int = 11, material_params: Optional[Dict] = None,
                 device: Optional[torch.device] = None, dq_method: str = 'original',
                 bc_type: str = 'C-C', log_dir: Optional[str] = None,
                 print_every: int = None, gpu_monitor_interval: int = None,
                 config: Optional[Any] = None,
                 
                 fd_stencil_size: int = None, x_nodes: str = None,
                 fd_build_orders: tuple = None, fd_build_C_D: bool = None,
                 fd_B_from_A: bool = None):
        
        
        self.config = config
        method_type = _get_config_value(self.config, 'method', 'dq')
        if method_type == 'dq' and N > 21:
            warnings.warn(f"DQ method node count N={N} is too large, may cause numerical instability. DQ method recommends N≤21")
        elif method_type == 'taylor' and N > 31:
            warnings.warn(f"Taylor method node count N={N} is too large, may affect computational efficiency. Taylor method recommends N≤31")
        elif method_type == 'spline' and N > 51:
            warnings.warn(f"Spline method node count N={N} is too large, may affect computational efficiency. Spline method recommends N≤51")
        elif method_type == 'spline' and N < 11:
            warnings.warn(f"Spline method node count N={N} is too small, may affect accuracy. Spline method recommends N≥11")
        
        
        self.N = int(N.item() if torch.is_tensor(N) else N)
        self.material_params = material_params
        self.dq_method = dq_method
        self.bc_type = bc_type
        self.device = device if device is not None else \
                     torch.device("cuda" if torch.cuda.is_available() else "cpu")

        
        self.print_every = print_every if print_every is not None else _get_config_value(self.config, 'print_every', 50)
        self.gpu_monitor_interval = gpu_monitor_interval if gpu_monitor_interval is not None else _get_config_value(self.config, 'gpu_monitor_interval', 100)

        
        self.tensor_pool = TensorPool(self.device, torch.float64)
        self.memory_optimizer = MemoryOptimizer()

        
        log_mode = _get_config_value(self.config, 'log_mode', 'simple')
        log_level = _get_config_value(self.config, 'log_level', 'INFO')
        self.logger = ODILLogger(
            name=f'ODIL-{bc_type}',
            mode=log_mode,
            level=log_level,
            log_dir=log_dir,
            config=self.config
        )
        
        
        self.gpu_monitor = None
        if _get_config_value(self.config, 'enable_gpu_monitor', False):
            try:
                self.gpu_monitor = create_gpu_monitor()
                if self.gpu_monitor:
                    self.logger.info("[GPU Monitor] GPU performance monitoring enabled")
            except Exception as e:
                self.logger.warning(f"[GPU Monitor] GPU monitor initialization failed: {e}")
        
        
        self.fd_stencil_size = fd_stencil_size if fd_stencil_size is not None else _get_config_value(self.config, 'fd_stencil_size', 5)
        self.x_nodes = x_nodes if x_nodes is not None else _get_config_value(self.config, 'x_nodes', 'cheb')
        self.fd_build_orders = fd_build_orders if fd_build_orders is not None else _get_config_value(self.config, 'fd_build_orders', (1, 2))
        self.fd_build_C_D = fd_build_C_D if fd_build_C_D is not None else _get_config_value(self.config, 'fd_build_C_D', False)
        self.fd_B_from_A = fd_B_from_A if fd_B_from_A is not None else _get_config_value(self.config, 'fd_B_from_A', False)

        
        
        self.x = cheb_lobatto_nodes(N, a=0.0, b=1.0, device=self.device)

        
        method_type = _get_config_value(self.config, 'method', 'dq')  
        self.method_type = method_type

        try:
            
            self.A, self.B = MethodFactory.create_discretization(method_type, self.x, self.config)

            
            self.use_interior_computation = False
            self.should_use_hard_constraint = True  
            self.bc_weight_factor = 1.0  

            if method_type == 'spline':
                spline_type = _get_config_value(self.config, 'spline_type', 'cubic')
                spline_boundary = _get_config_value(self.config, 'spline_boundary', 'natural')
                extras = []
                spline_degree = _get_config_value(self.config, 'spline_degree', None)
                if spline_degree:
                    extras.append(f"degree={spline_degree}")
                spline_tension = _get_config_value(self.config, 'spline_tension', 0.0)
                if spline_type.lower() in {'tension', 'hermite'} and spline_tension:
                    extras.append(f"tension={spline_tension}")
                extra_str = f" [{' , '.join(extras)}]" if extras else ""
                self.logger.info(
                    f"[Spline] Using {spline_type.upper()} spline (boundary={spline_boundary.upper()}){extra_str}"
                )

            
            self.A_interior, self.B_interior = self.A, self.B

            
            self.dq_system = {
                'x': self.x,
                'A': self.A,
                'B': self.B,
                'method': method_type,
                'condition_info': {'stable': True}  
            }

        except Exception as e:
            self.logger.warning(f"Method factory creation failed, falling back to legacy DQ system: {e}")
            
            self.dq_system = get_cached_dq_system(
                N=N, a=0.0, b=1.0, device=self.device,
                dq_method=dq_method,
                
                x_nodes=self.x_nodes,
                fd_stencil_size=self.fd_stencil_size,
                fd_build_orders=self.fd_build_orders,
                fd_build_C_D=self.fd_build_C_D,
                fd_B_from_A=self.fd_B_from_A
            )
            self.x = self.dq_system['x']
            self.A = self.dq_system['A']
            self.B = self.dq_system['B']
        
        
        if self.dq_system.get('condition_info'):
            cond_info = self.dq_system['condition_info']
            if not cond_info['stable']:
                self.logger.warning(f"Derivative matrix condition number is high: A={cond_info['cond_A']:.2e}, B={cond_info['cond_B']:.2e}")

        
        self.loss_history = []
    
    def _initialize_solve_parameters(self, initial_guess: Optional[Dict] = None) -> Dict[str, torch.Tensor]:
        
        return initialize_parameters(self.N, self.bc_type, self.device, initial_guess)
    
    def _adam_optimization(self, params_dict: Dict, compute_loss, max_iter: int, 
                          lr: float, verbose: bool, print_every: int = None) -> None:
        
        params_opt = [params_dict['u_param'], params_dict['w_param'], params_dict['phi_param']]
        opt = torch.optim.Adam(params_opt, lr=lr)

        for it in range(max_iter):
            opt.zero_grad()
            loss = compute_loss()
            loss.backward()

            
            grad_norm = apply_gradient_clipping(params_opt, max_norm=100)

            opt.step()

            loss_val = float(loss.item())
            self.loss_history.append(loss_val)

            if verbose and (it % 50 == 0 or it == max_iter-1):
                self.logger.iteration(it, loss_val, optimizer='Adam',
                                     grad_norm=grad_norm.item() if hasattr(grad_norm, 'item') else None, lr=lr)

            
            if len(self.loss_history) > 10 and loss_val > 10 * self.loss_history[0]:
                if verbose:
                    self.logger.warning("Loss diverged, stopping optimization")
                break
    
    def _compute_residual_vector(self, params_dict: Dict, bcs: Dict, q: float,
                                 is_nonlinear: bool = False) -> torch.Tensor:
        
        N = self.N
        device = self.device
        dtype = torch.float64

        if self.bc_type == 'C-C':
            u = inject_dirichlet(params_dict['u_param'], bcs['u'][0], bcs['u'][1])
            w = inject_dirichlet(params_dict['w_param'], bcs['w'][0], bcs['w'][1])
            phi = inject_dirichlet(params_dict['phi_param'], bcs['phi'][0], bcs['phi'][1])
        else:
            u = params_dict['u_param']
            w = params_dict['w_param']
            phi = params_dict['phi_param']

        foundation_params = _get_config_value(self.config, 'foundation_params', {'k1': 0.0, 'k2': 0.0})
        k1 = foundation_params.get('k1', 0.0)  
        k2 = foundation_params.get('k2', 0.0)  
        residual_calc = TimoshenkoBeamResiduals(self.material_params, q, k1, k2)

        if getattr(self, 'use_interior_computation', False) and hasattr(self, 'A_interior'):
            
            R1, R2, R3 = residual_calc.compute(u, w, phi, self.A_interior, self.B_interior, is_nonlinear)
            R1_masked = R1
            R2_masked = R2
            R3_masked = R3
        else:
            
            R1, R2, R3 = residual_calc.compute(u, w, phi, self.A, self.B, is_nonlinear)
            mask = create_boundary_mask(N, device, dtype)
            R1_masked = R1 * mask
            R2_masked = R2 * mask
            R3_masked = R3 * mask

        
        if self.bc_type != 'C-C':
            
            bc_residuals = []

            
            bc_residuals.append(u[0] - 0.0)
            bc_residuals.append(u[-1] - 0.0)
            bc_residuals.append(w[0] - 0.0)
            bc_residuals.append(w[-1] - 0.0)

            
            ux = self.A @ u
            wx = self.A @ w
            phix = self.A @ phi

            b11 = self.material_params['b11']
            d11 = self.material_params['d11']
            lambda_val = self.material_params['lambda_val']
            m_xT = self.material_params.get('m_xT', 0.0)

            if self.bc_type in ['H-H', 'S-S']:
                
                from utils.solver_utils import compute_moment
                M0 = compute_moment(ux, phix, wx, b11, d11, lambda_val, 0, is_nonlinear, m_xT)
                ML = compute_moment(ux, phix, wx, b11, d11, lambda_val, -1, is_nonlinear, m_xT)
                bc_residuals.append(M0)
                bc_residuals.append(ML)
            elif self.bc_type in ['C-S', 'C-H']:
                
                bc_residuals.append(phi[0])
                from utils.solver_utils import compute_moment
                ML = compute_moment(ux, phix, wx, b11, d11, lambda_val, -1, is_nonlinear, m_xT)
                bc_residuals.append(ML)

            
            bc_residual_tensor = torch.stack(bc_residuals)

            
            residual_vector = torch.cat([R1_masked, R2_masked, R3_masked, bc_residual_tensor])
        else:
            
            residual_vector = torch.cat([R1_masked, R2_masked, R3_masked])

        return residual_vector
    
    def _compute_jacobian(self, params_dict: Dict, bcs: Dict, q: float,
                         is_nonlinear: bool = False) -> torch.Tensor:
        
        
        if self.bc_type == 'C-C':
            
            params_vec = torch.cat([
                params_dict['u_param'].reshape(-1),
                params_dict['w_param'].reshape(-1),
                params_dict['phi_param'].reshape(-1)
            ])
        else:
            
            params_vec = torch.cat([
                params_dict['u_param'].reshape(-1),
                params_dict['w_param'].reshape(-1),
                params_dict['phi_param'].reshape(-1)
            ])

        params_vec = params_vec.clone().detach().to(device=self.device, dtype=torch.float64)
        params_vec.requires_grad_(True)

        
        def residual_func(params):
            
            if self.bc_type == 'C-C':
                n_params = self.N - 2
            else:
                n_params = self.N
            
            u_param = params[:n_params]
            w_param = params[n_params:2*n_params]
            phi_param = params[2*n_params:3*n_params]
            
            params_dict_local = {
                'u_param': u_param,
                'w_param': w_param,
                'phi_param': phi_param
            }
            
            return self._compute_residual_vector(params_dict_local, bcs, q, is_nonlinear)

        
        jacobian = None
        try:
            jacobian = torch.autograd.functional.jacobian(
                residual_func,
                params_vec,
                create_graph=False,
                vectorize=True
            )
        except (TypeError, RuntimeError, AttributeError):
            try:
                jacobian = torch.autograd.functional.jacobian(
                    residual_func,
                    params_vec,
                    create_graph=False
                )
            except Exception:
                residual = residual_func(params_vec)
                n_residual = residual.shape[0]
                n_params = params_vec.shape[0]
                jacobian = torch.zeros(n_residual, n_params, device=self.device, dtype=torch.float64)
                for i in range(n_residual):
                    grad = torch.autograd.grad(
                        residual[i],
                        params_vec,
                        retain_graph=True,
                        create_graph=False,
                        allow_unused=False
                    )[0]
                    jacobian[i, :] = grad

        n_params = params_vec.shape[0]
        jacobian = jacobian.reshape(-1, n_params)

        return jacobian.detach()
    
    def _gauss_newton_optimization(self, params_dict: Dict, bcs: Dict, q: float,
                                  is_nonlinear: bool, max_iter: int,
                                  verbose: bool, tol: float = 1e-6,
                                  damping: float = 0.1, print_every: int = None) -> None:
        
        if verbose:
            self.logger.info("[Gauss-Newton] Starting optimization...")
            self.logger.info(f"[Params] max_iter: {max_iter}, tol: {tol:.2e}, damping: {damping:.2e}")

        
        def pack_params():
            if self.bc_type == 'C-C':
                return torch.cat([
                    params_dict['u_param'].flatten(),
                    params_dict['w_param'].flatten(),
                    params_dict['phi_param'].flatten()
                ])
            else:
                return torch.cat([
                    params_dict['u_param'].flatten(),
                    params_dict['w_param'].flatten(),
                    params_dict['phi_param'].flatten()
                ])
        
        def unpack_params(params_vec):
            if self.bc_type == 'C-C':
                n_params = self.N - 2
            else:
                n_params = self.N
            
            params_dict['u_param'].data = params_vec[:n_params].reshape_as(params_dict['u_param'])
            params_dict['w_param'].data = params_vec[n_params:2*n_params].reshape_as(params_dict['w_param'])
            params_dict['phi_param'].data = params_vec[2*n_params:3*n_params].reshape_as(params_dict['phi_param'])
        
        
        def line_search(params_vec, delta, residual_norm_sq):
            
            alpha = 1.0
            beta = 0.5
            c = 0.1

            for _ in range(20):  
                
                new_params = params_vec + alpha * delta
                unpack_params(new_params)

                
                new_residual = self._compute_residual_vector(params_dict, bcs, q, is_nonlinear)
                new_norm_sq = torch.norm(new_residual)**2

                
                if new_norm_sq < residual_norm_sq * (1 - c * alpha):
                    return alpha

                alpha *= beta

            return alpha

        
        for it in range(max_iter):
            
            residual = self._compute_residual_vector(params_dict, bcs, q, is_nonlinear)
            residual_norm = torch.norm(residual).item()

            
            self.loss_history.append(residual_norm**2)

            if verbose and (it % (print_every or self.print_every) == 0):
                self.logger.iteration(it, residual_norm**2, optimizer='Gauss-Newton')

            
            if residual_norm < tol:
                if verbose:
                    self.logger.convergence(
                        reason="residual converged",
                        final_loss=residual_norm**2,
                        iterations=it
                    )
                break

            
            J = self._compute_jacobian(params_dict, bcs, q, is_nonlinear)

            
            JTJ = J.T @ J
            JTr = J.T @ residual

            
            eye = torch.eye(JTJ.shape[0], device=self.device, dtype=torch.float64)
            JTJ_reg = JTJ + damping * torch.diag(JTJ.diagonal()) + 1e-10 * eye

            try:
                
                delta = torch.linalg.solve(JTJ_reg, -JTr)

                
                params_vec = pack_params()
                alpha = line_search(params_vec, delta, residual_norm**2)

                
                params_vec = params_vec + alpha * delta
                unpack_params(params_vec)

                
                new_residual = self._compute_residual_vector(params_dict, bcs, q, is_nonlinear)
                new_norm = torch.norm(new_residual).item()

                if new_norm < residual_norm:
                    damping *= 0.7  
                else:
                    damping *= 2.0  

            except torch.linalg.LinAlgError:
                if verbose:
                    self.logger.warning(f"[Iteration {it}] Linear system solve failed, increasing damping factor")
                damping *= 10
                continue

        if verbose and it == max_iter - 1:
            self.logger.warning(f"[Gauss-Newton] Reached maximum iterations {max_iter}")
    
    def _levenberg_marquardt_optimization(self, params_dict: Dict, bcs: Dict, q: float,
                                         is_nonlinear: bool, max_iter: int,
                                         verbose: bool, tol: float = 1e-8,
                                         damping_init: float = 1e-3,
                                         damping_factor: float = 10.0,
                                         print_every: int = None) -> None:
        
        if verbose:
            self.logger.info("[Levenberg-Marquardt] Starting optimization...")
            self.logger.info(f"[Params] max_iter: {max_iter}, tol: {tol:.2e}, initial damping: {damping_init:.2e}")

        damping = damping_init
        damping_min = 1e-10
        damping_max = 1e10

        
        def pack_params():
            if self.bc_type == 'C-C':
                return torch.cat([
                    params_dict['u_param'].flatten(),
                    params_dict['w_param'].flatten(),
                    params_dict['phi_param'].flatten()
                ])
            else:
                return torch.cat([
                    params_dict['u_param'].flatten(),
                    params_dict['w_param'].flatten(),
                    params_dict['phi_param'].flatten()
                ])
        
        def unpack_params(params_vec):
            if self.bc_type == 'C-C':
                n_params = self.N - 2
            else:
                n_params = self.N
            
            params_dict['u_param'].data = params_vec[:n_params].reshape_as(params_dict['u_param'])
            params_dict['w_param'].data = params_vec[n_params:2*n_params].reshape_as(params_dict['w_param'])
            params_dict['phi_param'].data = params_vec[2*n_params:3*n_params].reshape_as(params_dict['phi_param'])
        
        
        best_loss = float('inf')
        patience_counter = 0
        max_patience = 10

        for it in range(max_iter):
            
            residual = self._compute_residual_vector(params_dict, bcs, q, is_nonlinear)
            current_loss = 0.5 * torch.norm(residual)**2

            
            self.loss_history.append(current_loss.item())

            if verbose and (it % (print_every or self.print_every) == 0):
                self.logger.iteration(it, current_loss.item(), optimizer='LM',
                                     grad_norm=damping)

            
            if torch.norm(residual) < tol:
                if verbose:
                    self.logger.convergence(
                        reason="residual converged",
                        final_loss=current_loss.item(),
                        iterations=it
                    )
                break
            
            
            J = self._compute_jacobian(params_dict, bcs, q, is_nonlinear)

            
            g = J.T @ residual

            
            if torch.norm(g) < tol * 10:
                if verbose:
                    self.logger.convergence(
                        reason="gradient converged",
                        final_loss=current_loss.item(),
                        iterations=it
                    )
                break
            
            
            step_accepted = False
            params_vec = pack_params()

            for attempt in range(10):  
                
                JTJ = J.T @ J
                JTr = J.T @ residual

                
                eye = torch.eye(JTJ.shape[0], device=self.device, dtype=torch.float64)
                H = JTJ + damping * eye

                try:
                    
                    delta = torch.linalg.solve(H, -JTr)

                    
                    new_params = params_vec + delta
                    unpack_params(new_params)

                    
                    new_residual = self._compute_residual_vector(params_dict, bcs, q, is_nonlinear)
                    new_loss = 0.5 * torch.norm(new_residual)**2

                    
                    actual_reduction = current_loss - new_loss
                    predicted_reduction = -0.5 * (delta @ (2*JTr + JTJ @ delta))

                    if predicted_reduction > 0:
                        rho = actual_reduction / predicted_reduction
                    else:
                        rho = -1

                    
                    if rho > 0.75:  
                        damping = max(damping / 3, damping_min)
                        step_accepted = True
                    elif rho > 0.25:  
                        step_accepted = True
                    else:  
                        damping = min(damping * damping_factor, damping_max)
                        
                        unpack_params(params_vec)

                    if step_accepted:
                        
                        if new_loss < best_loss:
                            best_loss = new_loss
                            patience_counter = 0
                        else:
                            patience_counter += 1

                        if verbose and attempt > 0:
                            self.logger.debug(f"[LM] Step accepted after {attempt+1} attempts, ρ={rho:.3f}, λ={damping:.2e}")
                        break

                except torch.linalg.LinAlgError:
                    
                    damping = min(damping * damping_factor, damping_max)
                    if verbose:
                        self.logger.warning(f"[Iteration {it}] Linear system solve failed, increasing damping to {damping:.2e}")

            
            if not step_accepted:
                if verbose:
                    self.logger.warning(f"[Iteration {it}] No acceptable step found")
                if patience_counter > max_patience:
                    if verbose:
                        self.logger.convergence(
                            reason="patience limit reached",
                            final_loss=current_loss.item(),
                            iterations=it
                        )
                    break
        
        if verbose and it == max_iter - 1:
            self.logger.warning(f"[Levenberg-Marquardt] Reached maximum iterations {max_iter}")
    
    def _lbfgs_optimization(self, params_dict: Dict, compute_loss, max_iter: int, 
                           lr: float, verbose: bool,
                           print_every: int = None, gpu_monitor_interval: int = None) -> None:
        
        params_opt = [params_dict['u_param'], params_dict['w_param'], params_dict['phi_param']]

        
        perturbation_enabled = _get_config_value(self.config, 'perturbation_enabled', True)
        perturbation_threshold = _get_config_value(self.config, 'perturbation_threshold', 0.001)
        perturbation_patience = _get_config_value(self.config, 'perturbation_patience', 1000)
        perturbation_scale = _get_config_value(self.config, 'perturbation_scale', 0.01)
        perturbation_scale_increment = _get_config_value(self.config, 'perturbation_scale_increment', 0.5)

        
        self.lbfgs_iter_count = 0
        restart_count = 0  

        
        L0 = float(compute_loss().item())
        if verbose:
            self.logger.info(f"[L-BFGS] Initial loss = {L0:.3e}")
            if perturbation_enabled:
                self.logger.info(f"[Perturbation] Enabled - threshold: {perturbation_threshold*100:.2f}%, patience: {perturbation_patience}")
        
        
        while self.lbfgs_iter_count < max_iter:
            
            opt = create_optimizer(params_opt, "lbfgs", lr, max_iter - self.lbfgs_iter_count)

            
            flat_count = 0
            last_loss = float('inf')
            need_restart = False

            def closure():
                nonlocal flat_count, last_loss, need_restart
                opt.zero_grad()
                loss = compute_loss()
                loss.backward()

                
                apply_gradient_clipping(params_opt, max_norm=100)

                loss_val = float(loss.item())
                self.loss_history.append(loss_val)
                self.lbfgs_iter_count += 1

                
                if perturbation_enabled and len(self.loss_history) > 1:
                    relative_change = abs(loss_val - last_loss) / max(abs(last_loss), 1e-10)
                    if relative_change < perturbation_threshold:
                        flat_count += 1
                    else:
                        flat_count = 0

                    
                    if flat_count >= perturbation_patience:
                        need_restart = True
                        raise StopIteration("Need perturbation restart")

                last_loss = loss_val

                if verbose and (self.lbfgs_iter_count == 1 or self.lbfgs_iter_count % (print_every or self.print_every) == 0):
                    self.logger.iteration(self.lbfgs_iter_count, loss_val, optimizer='L-BFGS', lr=lr)
                    if self.gpu_monitor and self.lbfgs_iter_count % (gpu_monitor_interval or self.gpu_monitor_interval) == 0:
                        self.gpu_monitor.print_current_stats()

                
                if self.lbfgs_iter_count >= max_iter:
                    raise StopIteration(f"Reached maximum iterations: {max_iter}")

                return loss

            try:
                opt.step(closure)
                
                if verbose:
                    self.logger.convergence(
                        reason="L-BFGS converged",
                        final_loss=self.loss_history[-1] if self.loss_history else 0,
                        iterations=self.lbfgs_iter_count
                    )
                break
                
            except StopIteration as e:
                if need_restart and perturbation_enabled:
                    
                    restart_count += 1
                    current_scale = perturbation_scale * (1 + perturbation_scale_increment * (restart_count - 1))

                    if verbose:
                        self.logger.warning(f"[Restart #{restart_count}] Stagnation detected ({flat_count} consecutive), adding perturbation (scale: {current_scale:.4f})...")

                    
                    with torch.no_grad():
                        for key in ['u_param', 'w_param', 'phi_param']:
                            param = params_dict[key]
                            
                            param_std = torch.std(param).item()
                            noise_scale = current_scale * param_std if param_std > 0 else current_scale
                            noise = torch.randn_like(param) * noise_scale
                            param.data += noise

                    
                    continue

                elif "maximum iterations" in str(e):
                    
                    if verbose:
                        self.logger.convergence(
                            reason=str(e),
                            final_loss=self.loss_history[-1] if self.loss_history else 0,
                            iterations=self.lbfgs_iter_count
                        )
                    break
                else:
                    
                    if verbose:
                        self.logger.convergence(
                            reason="L-BFGS inner converged",
                            final_loss=self.loss_history[-1] if self.loss_history else 0,
                            iterations=self.lbfgs_iter_count
                        )
                    break

            except Exception as e:
                
                import traceback
                error_msg = f"L-BFGS optimization failed: {str(e)}"
                error_trace = traceback.format_exc()

                if verbose:
                    self.logger.error(error_msg)
                    self.logger.info("[Recovery] Automatically switching to Adam optimizer...")
                    self.logger.debug(f"Current iteration: {self.lbfgs_iter_count}, last loss: {self.loss_history[-1] if self.loss_history else 'N/A'}")

                
                try:
                    import datetime
                    with open("optimization_error_log.txt", "a", encoding='utf-8') as f:
                        f.write(f"\n{'='*60}\n")
                        f.write(f"Time: {datetime.datetime.now()}\n")
                        f.write(f"Boundary condition: {self.bc_type}\n")
                        f.write(f"Node count: {self.N}\n")
                        f.write(f"Error: {error_msg}\n")
                        f.write(f"Traceback:\n{error_trace}\n")
                except Exception:
                    pass  

                
                opt_adam = torch.optim.Adam(params_opt, lr=0.01)
                for it in range(min(max_iter - self.lbfgs_iter_count, 1000)):
                    opt_adam.zero_grad()
                    loss = compute_loss()
                    loss.backward()
                    apply_gradient_clipping(params_opt, max_norm=1e3)
                    opt_adam.step()
                    
                    loss_val = float(loss.item())
                    self.loss_history.append(loss_val)
                    self.lbfgs_iter_count += 1
                    
                    if verbose and (it % (print_every or self.print_every) == 0):
                        self.logger.iteration(self.lbfgs_iter_count, loss_val, optimizer='Adam-fallback', lr=0.01)
                break
        
        
        if verbose:
            Lf = float(compute_loss().item())
            self.logger.info(f"[Final] Loss = {Lf:.3e}")
            if restart_count > 0:
                self.logger.info(f"[Stats] Performed {restart_count} perturbation restarts in total")
    
    def _assemble_solution(self, params_dict: Dict, bcs: Dict) -> Dict:
        
        if self.bc_type == 'C-C':
            
            from utils.solver_utils import inject_dirichlet
            u = inject_dirichlet(params_dict['u_param'].detach(), bcs['u'][0], bcs['u'][1])
            w = inject_dirichlet(params_dict['w_param'].detach(), bcs['w'][0], bcs['w'][1])
            phi = inject_dirichlet(params_dict['phi_param'].detach(), bcs['phi'][0], bcs['phi'][1])
        else:
            
            u = params_dict['u_param'].detach()
            w = params_dict['w_param'].detach()
            phi = params_dict['phi_param'].detach()
        
        return {
            'x': self.x,
            'u': u,
            'w': w,
            'phi': phi,
            'A': self.A,
            'B': self.B,
            'material_params': self.material_params,
            'loss_history': self.loss_history,
            'final_loss': self.loss_history[-1] if self.loss_history else None,
            'gpu_stats': self.gpu_monitor.get_summary() if self.gpu_monitor else None,
            'memory_stats': self.tensor_pool.get_stats() if hasattr(self, 'tensor_pool') else None
        }
    
    def solve(self, bcs: Dict, q: float = 0.0, is_nonlinear: bool = False,
             optim_name: str = "lbfgs", max_iter: int = 200, lr: float = 0.8,
             loss_weights: Tuple[float, float, float] = (1.0, 1.0, 1.0),
             reg: float = 1e-12, verbose: bool = True,
             use_adaptive_weights: bool = False,  
             initial_guess: Optional[Dict] = None,
             bc_weight: float = 10.0,
             print_every: int = None, gpu_monitor_interval: int = None) -> Dict:
        
        params_dict = self._initialize_solve_parameters(initial_guess)
        
        if verbose:
            self.logger.optimization_start(
                mode='linear' if not is_nonlinear else 'nonlinear',
                bc_type=self.bc_type,
                N=self.N,
                optimizer=optim_name
            )
        
        
        self.loss_history = []

        
        foundation_params = _get_config_value(self.config, 'foundation_params', {'k1': 0.0, 'k2': 0.0})
        def compute_loss():
            return odil_loss(params_dict, self.x, self.A, self.B,
                           self.material_params, q, bcs, self.bc_type,
                           loss_weights, reg, is_nonlinear, bc_weight,
                           A_interior=getattr(self, 'A_interior', None),
                           B_interior=getattr(self, 'B_interior', None),
                           use_interior_computation=getattr(self, 'use_interior_computation', False),
                           should_use_hard_constraint=getattr(self, 'should_use_hard_constraint', True),
                           bc_weight_factor=getattr(self, 'bc_weight_factor', 1.0),
                           foundation_params=foundation_params)
        
        
        print(f"\n[Solver] Optimizer actually used: {optim_name}")

        
        if optim_name.lower() == "adam":
            self._adam_optimization(params_dict, compute_loss, max_iter,
                                  lr, verbose, print_every)
        elif optim_name.lower() == "lbfgs":
            self._lbfgs_optimization(params_dict, compute_loss, max_iter,
                                    lr, verbose, print_every, gpu_monitor_interval)
        elif optim_name.lower() in ["gauss-newton", "gauss_newton", "newton"]:
            
            
            self._gauss_newton_optimization(
                params_dict, bcs, q, is_nonlinear,
                max_iter, verbose,
                tol=_get_config_value(self.config, 'gauss_newton_tol', 1e-6),
                damping=_get_config_value(self.config, 'gauss_newton_damping', 0.1),
                print_every=print_every
            )
        elif optim_name.lower() in ["lm", "levenberg-marquardt", "levenberg_marquardt"]:
            
            self._levenberg_marquardt_optimization(
                params_dict, bcs, q, is_nonlinear,
                max_iter, verbose,
                tol=_get_config_value(self.config, 'lm_tol', 1e-8),
                damping_init=_get_config_value(self.config, 'lm_damping_init', 1e-3),
                damping_factor=_get_config_value(self.config, 'lm_damping_factor', 10.0),
                print_every=print_every
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optim_name}")
        
        
        solution = self._assemble_solution(params_dict, bcs)

        
        if self.gpu_monitor and verbose:
            self.gpu_monitor.print_summary()

        
        self.tensor_pool.clear()
        self.memory_optimizer.clear_cache()

        return solution
    

