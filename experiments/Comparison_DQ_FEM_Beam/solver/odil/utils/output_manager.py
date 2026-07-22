

import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
from typing import Any, Dict, Optional, List


def _format_float(value: float, precision: int = 6) -> str:
    
    if value is None:
        return '0'

    formatted = f"{value:.{precision}f}".rstrip('0').rstrip('.')
    if formatted in {'', '-'}:
        formatted = '0'
    return formatted


def _format_signed(value: float, precision: int = 6) -> str:
    
    if value is None:
        return '0'

    if abs(value) < 1e-12:
        return '0'

    return _format_float(value, precision=precision)


def _derive_length_ratio(lambda_val: Optional[float] = None,
                         L: Optional[float] = None,
                         h: Optional[float] = None) -> Optional[float]:
    
    if L is not None and h not in (None, 0):
        return L / h
    return lambda_val


def _cfg(config: Optional[Any], name: str, default=None):
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(name, default)
    return getattr(config, name, default)


def get_filename_base(W_Gr: float,
                      T: float,
                      H_Gr: float,
                      q: float,
                      lambda_val: Optional[float] = None,
                      L: Optional[float] = None,
                      h: Optional[float] = None,
                      **_: Dict) -> str:
    
    components = [
        f"W_{_format_float(W_Gr)}",
        f"T_{_format_float(T)}",
        f"H_{_format_float(H_Gr)}",
        f"q_{_format_signed(q)}"
    ]
    return "_".join(components)


def get_directory_name(W_Gr: float,
                       T: float,
                       H_Gr: float,
                       q: float,
                       lambda_val: Optional[float] = None,
                       L: Optional[float] = None,
                       h: Optional[float] = None,
                       foundation_params: Optional[Dict[str, float]] = None) -> str:
    
    components = [
        f"W_{_format_float(W_Gr)}",
        f"T_{_format_float(T)}",
        f"H_{_format_float(H_Gr)}",
        f"q_{_format_signed(q)}"
    ]

    ratio = _derive_length_ratio(lambda_val=lambda_val, L=L, h=h)
    if ratio is not None:
        components.append(f"L_{_format_float(ratio)}h")

    
    if foundation_params is None:
        foundation_params = {'k1': 0.0, 'k2': 0.0}
    k1 = foundation_params.get('k1', 0.0)
    k2 = foundation_params.get('k2', 0.0)
    components.append(f"k1_{_format_float(k1)}")
    components.append(f"k2_{_format_float(k2)}")

    return "_".join(components)


class OutputManager:
    
    
    def __init__(self, base_dir: str = 'results',
                 sub_dir: str = 'main',
                 bc_type: Optional[str] = None,
                 distr_type: Optional[str] = None,
                 W_Gr: Optional[float] = None,
                 T: Optional[float] = None,
                 H_Gr: Optional[float] = None,
                 q: Optional[float] = None,
                 lambda_val: Optional[float] = None,
                 h: Optional[float] = None,
                 L: Optional[float] = None,
                 foundation_params: Optional[Dict[str, float]] = None,
                 config: Optional[Any] = None):
        
        self.config = config

        
        if bc_type is None:
            bc_type = _cfg(self.config, 'bc_type', None)
        if distr_type is None:
            distr_type = _cfg(self.config, 'distr_type', None)

        
        self.bc_type = bc_type
        self.distr_type = distr_type
        self.W_Gr = W_Gr
        self.T = T
        self.H_Gr = H_Gr
        self.q = q
        self.lambda_val = lambda_val
        self.h = h if h is not None else _cfg(self.config, 'h', None)
        self.L = L if L is not None else _cfg(self.config, 'L', None)
        self.foundation_params = foundation_params or _cfg(self.config, 'foundation_params', None)
        self.file_base = None

        
        self._build_directory_structure(
            base_dir, sub_dir, bc_type, distr_type,
            W_Gr, T, H_Gr, q, lambda_val
        )

        
        self._ensure_directories()
    
    def _build_directory_structure(self, base_dir: str, sub_dir: str, 
                                  bc_type: Optional[str], distr_type: Optional[str],
                                  W_Gr: Optional[float], T: Optional[float], 
                                  H_Gr: Optional[float], q: Optional[float], 
                                  lambda_val: Optional[float]):
        

        
        is_special_dir = self._is_special_directory(sub_dir)

        if is_special_dir:
            
            
            sub_dir_normalized = sub_dir.replace('/', os.sep).replace('\\', os.sep)
            self.results_dir = os.path.join(base_dir, sub_dir_normalized)
        elif all(param is not None for param in [W_Gr, T, H_Gr, q, lambda_val]):
            
            folder_name = get_directory_name(
                W_Gr, T, H_Gr, q,
                lambda_val=lambda_val,
                L=self.L,
                h=self.h,
                foundation_params=self.foundation_params
            )
            self.file_base = get_filename_base(
                W_Gr, T, H_Gr, q,
                lambda_val=lambda_val,
                L=self.L,
                h=self.h,
                foundation_params=self.foundation_params
            )

            
            path_components = [base_dir, sub_dir]

            
            if bc_type:
                path_components.append(bc_type.upper())

            
            if distr_type:
                path_components.append(distr_type.upper())

            
            path_components.append(folder_name)

            self.results_dir = os.path.join(*path_components)
        else:
            
            path_components = [base_dir, sub_dir]
            if bc_type:
                path_components.append(bc_type.upper())
            if distr_type:
                path_components.append(distr_type.upper())
            self.results_dir = os.path.join(*path_components)
            if all(param is not None for param in [W_Gr, T, H_Gr, q]):
                self.file_base = get_filename_base(W_Gr, T, H_Gr, q,
                                                   lambda_val=lambda_val,
                                                   L=self.L,
                                                   h=self.h,
                                                   foundation_params=self.foundation_params)

        
        self.data_dir = os.path.join(self.results_dir, 'data')
        self.plots_dir = os.path.join(self.results_dir, 'plots')
        self.models_dir = os.path.join(self.results_dir, 'models')
        self.logs_dir = os.path.join(self.results_dir, 'logs')

    def _is_special_directory(self, sub_dir: str) -> bool:
        
        special_patterns = [
            'summary',
            'Summary_comparisons',
            'comparison',
            'archive',
        ]
        
        has_separator = '/' in sub_dir or '\\' in sub_dir
        has_special = any(pattern in sub_dir for pattern in special_patterns)
        return has_separator or has_special

    def _ensure_directories(self):
        
        for dir_path in [self.results_dir, self.data_dir, self.plots_dir, self.models_dir, self.logs_dir]:
            os.makedirs(dir_path, exist_ok=True)

    def get_logs_dir(self) -> str:
        
        return self.logs_dir
    
    def save_solution_data(self, result: Dict, mode: str, base_name: str) -> str:
        
        
        x = result['x'].cpu().numpy() if hasattr(result['x'], 'cpu') else result['x']
        u = result['u'].cpu().numpy() if hasattr(result['u'], 'cpu') else result['u']
        w = result['w'].cpu().numpy() if hasattr(result['w'], 'cpu') else result['w']
        phi = result['phi'].cpu().numpy() if hasattr(result['phi'], 'cpu') else result['phi']

        
        df = pd.DataFrame({
            'x': x,
            'u': u,
            'w': w,
            'phi': phi
        })

        
        
        csv_filename = f"{mode}_w_{base_name}.csv"
        csv_path = os.path.join(self.data_dir, csv_filename)

        
        df.to_csv(csv_path, index=False)
        print(f"  Data saved: {csv_filename}")

        return csv_path
    
    def save_loss_history(self, loss_history: List, mode: str, base_name: str) -> Optional[str]:
        
        if not loss_history:
            return None

        
        loss_df = pd.DataFrame({
            'iteration': range(len(loss_history)),
            'loss': loss_history
        })

        
        loss_csv = f"loss_{mode}_{base_name}.csv"
        loss_path = os.path.join(self.data_dir, loss_csv)

        
        loss_df.to_csv(loss_path, index=False)
        print(f"  Loss history saved: {loss_csv}")

        return loss_path
    
    def plot_single_solution(self, result: Dict, mode: str, base_name: str, 
                            plot_dpi: int = 300) -> str:
        
        
        x = result['x'].cpu().numpy() if hasattr(result['x'], 'cpu') else result['x']
        u = result['u'].cpu().numpy() if hasattr(result['u'], 'cpu') else result['u']
        w = result['w'].cpu().numpy() if hasattr(result['w'], 'cpu') else result['w']
        phi = result['phi'].cpu().numpy() if hasattr(result['phi'], 'cpu') else result['phi']

        
        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

        title_prefix = "Linear" if mode == "linear" else "Nonlinear"
        fig.suptitle(f"{title_prefix} Timoshenko Beam Solution", fontsize=14, fontweight='bold')

        
        axes[0].plot(x, u, 'b-', linewidth=2, marker='o', markersize=4,
                    markerfacecolor='white', markeredgecolor='blue')
        axes[0].set_ylabel('u(x)\nAxial Displacement', fontsize=11)
        axes[0].grid(True, alpha=0.3)
        axes[0].set_title('Axial Displacement', fontsize=10)

        
        axes[1].plot(x, w, 'r-', linewidth=2, marker='s', markersize=4,
                    markerfacecolor='white', markeredgecolor='red')
        axes[1].set_ylabel('w(x)\nTransverse Displacement', fontsize=11)
        axes[1].grid(True, alpha=0.3)
        axes[1].set_title('Transverse Displacement', fontsize=10)

        
        axes[2].plot(x, phi, 'g-', linewidth=2, marker='^', markersize=4,
                    markerfacecolor='white', markeredgecolor='green')
        axes[2].set_ylabel('φ(x)\nRotation Angle', fontsize=11)
        axes[2].set_xlabel('Position x/L', fontsize=11)
        axes[2].grid(True, alpha=0.3)
        axes[2].set_title('Rotation Angle', fontsize=10)

        plt.tight_layout()

        
        
        plot_filename = f"{title_prefix}w_{base_name}.png"
        plot_path = os.path.join(self.plots_dir, plot_filename)

        
        plt.savefig(plot_path, dpi=plot_dpi, bbox_inches='tight')
        plt.close()
        print(f"  Figure saved: {plot_filename}")

        return plot_path
    
    def plot_comparison(self, linear_result: Dict, nonlinear_result: Dict,
                       base_name: str, plot_dpi: int = 300) -> str:
        
        
        x = linear_result['x'].cpu().numpy() if hasattr(linear_result['x'], 'cpu') else linear_result['x']

        w_linear = linear_result['w'].cpu().numpy() if hasattr(linear_result['w'], 'cpu') else linear_result['w']
        w_nonlinear = nonlinear_result['w'].cpu().numpy() if hasattr(nonlinear_result['w'], 'cpu') else nonlinear_result['w']

        u_linear = linear_result['u'].cpu().numpy() if hasattr(linear_result['u'], 'cpu') else linear_result['u']
        u_nonlinear = nonlinear_result['u'].cpu().numpy() if hasattr(nonlinear_result['u'], 'cpu') else nonlinear_result['u']

        phi_linear = linear_result['phi'].cpu().numpy() if hasattr(linear_result['phi'], 'cpu') else linear_result['phi']
        phi_nonlinear = nonlinear_result['phi'].cpu().numpy() if hasattr(nonlinear_result['phi'], 'cpu') else nonlinear_result['phi']

        
        fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
        fig.suptitle('Linear vs Nonlinear Comparison', fontsize=14, fontweight='bold')

        
        axes[0].plot(x, u_linear, 'b-', linewidth=2, label='Linear', marker='o',
                    markersize=4, markerfacecolor='white', markeredgecolor='blue')
        axes[0].plot(x, u_nonlinear, 'b--', linewidth=2, label='Nonlinear', marker='s',
                    markersize=4, markerfacecolor='white', markeredgecolor='darkblue')
        axes[0].set_ylabel('u(x)\nAxial Displacement', fontsize=11)
        axes[0].legend(loc='best')
        axes[0].grid(True, alpha=0.3)

        
        axes[1].plot(x, w_linear, 'r-', linewidth=2, label='Linear', marker='o',
                    markersize=4, markerfacecolor='white', markeredgecolor='red')
        axes[1].plot(x, w_nonlinear, 'r--', linewidth=2, label='Nonlinear', marker='s',
                    markersize=4, markerfacecolor='white', markeredgecolor='darkred')
        axes[1].set_ylabel('w(x)\nTransverse Displacement', fontsize=11)
        axes[1].legend(loc='best')
        axes[1].grid(True, alpha=0.3)

        
        axes[2].plot(x, phi_linear, 'g-', linewidth=2, label='Linear', marker='o',
                    markersize=4, markerfacecolor='white', markeredgecolor='green')
        axes[2].plot(x, phi_nonlinear, 'g--', linewidth=2, label='Nonlinear', marker='s',
                    markersize=4, markerfacecolor='white', markeredgecolor='darkgreen')
        axes[2].set_ylabel('φ(x)\nRotation Angle', fontsize=11)
        axes[2].set_xlabel('Position x/L', fontsize=11)
        axes[2].legend(loc='best')
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()

        
        comparison_filename = f"comparison_w_{base_name}.png"
        comparison_path = os.path.join(self.plots_dir, comparison_filename)

        
        plt.savefig(comparison_path, dpi=plot_dpi, bbox_inches='tight')
        plt.close()
        print(f"  Comparison plot saved: {comparison_filename}")

        return comparison_path
    
    def plot_loss_history(self, loss_history: List, base_name: str, 
                         mode: Optional[str] = None, plot_dpi: int = 300) -> Optional[str]:
        
        if not loss_history:
            return None

        fig, ax = plt.subplots(figsize=(8, 6))

        iterations = range(len(loss_history))
        ax.semilogy(iterations, loss_history, 'b-', linewidth=2)
        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)

        title = 'Loss History'
        if mode:
            title = f'{mode.capitalize()} Problem - Loss History'
        ax.set_title(title, fontsize=14)
        ax.grid(True, alpha=0.3, which='both')

        
        final_loss = loss_history[-1]
        ax.annotate(f'Final: {final_loss:.2e}',
                   xy=(len(loss_history)-1, final_loss),
                   xytext=(0.7, 0.9), textcoords='axes fraction',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        plt.tight_layout()
        
        if mode:
            filename_core = f"loss_{mode}_{base_name}.png"
        else:
            filename_core = f"loss_{base_name}.png"

        loss_filename = filename_core
        loss_path = os.path.join(self.plots_dir, loss_filename)

        
        plt.savefig(loss_path, dpi=plot_dpi, bbox_inches='tight')
        plt.close()
        print(f"  Loss plot saved: {loss_filename}")

        return loss_path

    def plot_combined_loss_history(self,
                                  linear_loss_history: List,
                                  nonlinear_loss_history: List,
                                  base_name: str,
                                  plot_dpi: int = 300) -> Optional[str]:
        
        if not linear_loss_history and not nonlinear_loss_history:
            return None

        fig, ax = plt.subplots(figsize=(10, 6))

        
        if linear_loss_history:
            iterations_linear = range(len(linear_loss_history))
            ax.semilogy(iterations_linear, linear_loss_history, 'b-',
                       linewidth=2, label='Linear', alpha=0.8)

            
            final_loss_linear = linear_loss_history[-1]
            ax.plot(len(linear_loss_history)-1, final_loss_linear, 'bo',
                   markersize=8, markerfacecolor='white', markeredgewidth=2)

        
        if nonlinear_loss_history:
            iterations_nonlinear = range(len(nonlinear_loss_history))
            ax.semilogy(iterations_nonlinear, nonlinear_loss_history, 'r-',
                       linewidth=2, label='Nonlinear', alpha=0.8)

            
            final_loss_nonlinear = nonlinear_loss_history[-1]
            ax.plot(len(nonlinear_loss_history)-1, final_loss_nonlinear, 'ro',
                   markersize=8, markerfacecolor='white', markeredgewidth=2)

        
        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('Loss History Comparison: Linear vs Nonlinear', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, which='both')
        ax.legend(loc='best', fontsize=11)

        
        y_pos = 0.9
        if linear_loss_history:
            ax.text(0.98, y_pos, f'Linear Final: {final_loss_linear:.2e}', 
                   transform=ax.transAxes, fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7),
                   horizontalalignment='right')
            y_pos -= 0.08
        
        if nonlinear_loss_history:
            ax.text(0.98, y_pos, f'Nonlinear Final: {final_loss_nonlinear:.2e}', 
                   transform=ax.transAxes, fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7),
                   horizontalalignment='right')
        
        plt.tight_layout()

        
        loss_filename = f"loss_{base_name}.png"
        loss_path = os.path.join(self.plots_dir, loss_filename)

        plt.savefig(loss_path, dpi=plot_dpi, bbox_inches='tight')
        plt.close()
        print(f"  Combined loss plot saved: {loss_filename}")

        return loss_path
    
    def save_model(self, model_state: Dict, mode: str, base_name: str) -> Optional[str]:
        
        
        return None
    
    def save_all_results(self, result: Dict, mode: str, base_name: str,
                        dpi: int = 300, verbose: bool = True,
                        save_data: bool = True,
                        save_loss: bool = True) -> Dict[str, str]:
        
        if verbose:
            print(f"\n[Saving {mode} problem results]")

        saved_paths = {}

        
        if save_data:
            saved_paths['data'] = self.save_solution_data(result, mode, base_name)

        
        if save_loss and 'loss_history' in result and result['loss_history']:
            saved_paths['loss_data'] = self.save_loss_history(
                result['loss_history'], mode, base_name
            )
            saved_paths['loss_plot'] = self.plot_loss_history(
                result['loss_history'], base_name, mode, plot_dpi=dpi
            )

        
        saved_paths['solution_plot'] = self.plot_single_solution(
            result, mode, base_name, plot_dpi=dpi
        )

        if verbose:
            print(f"  All {mode} results have been saved")

        return saved_paths

    def save_combined_solution_data(self, linear_result: Dict, nonlinear_result: Dict, 
                                    base_name: str) -> Optional[str]:
        
        try:
            import pandas as pd
        except Exception:
            return None

        
        required = ['x', 'u', 'w', 'phi']
        if not all(k in linear_result for k in required):
            return None
        if not all(k in nonlinear_result for k in required):
            return None

        
        def to_np(t):
            return t.cpu().numpy() if hasattr(t, 'cpu') else t

        x = to_np(linear_result['x'])
        u_lin = to_np(linear_result['u'])
        w_lin = to_np(linear_result['w'])
        phi_lin = to_np(linear_result['phi'])
        u_nl = to_np(nonlinear_result['u'])
        w_nl = to_np(nonlinear_result['w'])
        phi_nl = to_np(nonlinear_result['phi'])

        
        df = pd.DataFrame({
            'x': x,
            'linear_u': u_lin,
            'linear_w': w_lin,
            'linear_phi': phi_lin,
            'nonlinear_u': u_nl,
            'nonlinear_w': w_nl,
            'nonlinear_phi': phi_nl,
        })

        csv_filename = f"w_{base_name}.csv"
        csv_path = os.path.join(self.data_dir, csv_filename)
        df.to_csv(csv_path, index=False)
        print(f"  Combined solution data saved: {csv_filename}")

        return csv_path

    def save_combined_loss_history(self, linear_loss_history: List, 
                                   nonlinear_loss_history: List,
                                   base_name: str) -> Optional[str]:
        
        if (not linear_loss_history) and (not nonlinear_loss_history):
            return None

        import pandas as pd
        len_lin = len(linear_loss_history) if linear_loss_history else 0
        len_nl = len(nonlinear_loss_history) if nonlinear_loss_history else 0
        n = max(len_lin, len_nl)

        
        iter_lin = list(range(len_lin)) + [None] * (n - len_lin)
        loss_lin = list(linear_loss_history) + [None] * (n - len_lin)
        iter_nl = list(range(len_nl)) + [None] * (n - len_nl)
        loss_nl = list(nonlinear_loss_history) + [None] * (n - len_nl)

        df = pd.DataFrame({
            'iteration_linear': iter_lin,
            'loss_linear': loss_lin,
            'iteration_nonlinear': iter_nl,
            'loss_nonlinear': loss_nl,
        })

        csv_filename = f"loss_{base_name}.csv"
        csv_path = os.path.join(self.data_dir, csv_filename)
        df.to_csv(csv_path, index=False)
        print(f"  Combined loss history saved: {csv_filename}")

        return csv_path
    
    def print_summary(self, linear_result: Optional[Dict] = None, 
                     nonlinear_result: Optional[Dict] = None):
        
        print("\n" + "="*60)
        print("Solution Result Summary")
        print("="*60)

        if linear_result:
            print("\nLinear problem:")
            if 'final_loss' in linear_result:
                print(f"  Final loss: {linear_result['final_loss']:.3e}")
            w_max = float(torch.max(torch.abs(linear_result['w'])))
            print(f"  Maximum deflection: {w_max:.6f}")

        if nonlinear_result:
            print("\nNonlinear problem:")
            if 'final_loss' in nonlinear_result:
                print(f"  Final loss: {nonlinear_result['final_loss']:.3e}")
            if 'w' in nonlinear_result:
                w_max = float(torch.max(torch.abs(nonlinear_result['w'])))
                print(f"  Maximum deflection: {w_max:.6f}")
            else:
                print(f"  Warning: no deflection data found in the nonlinear result")

        if linear_result and nonlinear_result:
            
            w_diff = float(torch.max(torch.abs(nonlinear_result['w'] - linear_result['w'])))
            w_linear_max = float(torch.max(torch.abs(linear_result['w'])))
            if w_linear_max > 0:
                relative_diff = w_diff / w_linear_max * 100
                print(f"\nMaximum deflection difference: {w_diff:.6f} ({relative_diff:.2f}%)")

        print("\nResult files have been saved to the results/ directory")
        print("="*60)



def create_output_manager(base_dir: str = 'results',
                         sub_dir: str = 'main',
                         bc_type: Optional[str] = None,
                         distr_type: Optional[str] = None,
                         auto_load_params: bool = True,
                         config: Optional[Any] = None,
                         **kwargs) -> OutputManager:
    
    
    if auto_load_params:
        if bc_type is None:
            bc_type = _cfg(config, 'bc_type', bc_type)
        if distr_type is None:
            distr_type = _cfg(config, 'distr_type', distr_type)

    return OutputManager(
        base_dir=base_dir,
        sub_dir=sub_dir,
        bc_type=bc_type,
        distr_type=distr_type,
        config=config,
        **kwargs
    )
