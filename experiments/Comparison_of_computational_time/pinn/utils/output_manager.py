"""
Output manager - handles result saving, visualization, and report generation

Author: Yang
Creation date: 2024-09-22
Version: 1.0

This module provides unified output management functionality, including model saving, data export,
visualization chart generation, and result reporting. It supports hierarchical directory structure management.
"""

import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional
from datetime import datetime


class OutputManager:
    """Unified management of all output operations

    Supports the new hierarchical directory structure:
    results/{script_name}/{bc_type}/{distribution}/{param_folder}/(data|logs|models|plots)
    and creates a summary_comparisons/ directory at the distribution level.
    """

    def __init__(self,
                 base_dir: str = "results",
                 script_name: Optional[str] = None,
                 bc_type: Optional[str] = None,
                 distribution: Optional[str] = None,
                 param_folder: Optional[str] = None):
        """
        Initialize the output manager

        Parameters:
            base_dir: top-level results directory (default results)
            script_name: script name (e.g. main)
            bc_type: boundary condition type (e.g. C-C, S-S)
            distribution: distribution type (e.g. X, O, U)
            param_folder: parameter combination folder name (e.g. W_..._lambda_...)
        """
        self.base_dir = base_dir  # top-level results directory
        self.script_name = script_name
        self.bc_type = bc_type
        self.distribution = distribution
        self.param_folder = param_folder

        # Compute the run root directory (fall back to the old structure under results/ if there is no context)
        if all(v is not None for v in (script_name, bc_type, distribution, param_folder)):
            # If script_name is an empty string, skip this level
            if script_name:
                self.run_root = os.path.join(base_dir, script_name, bc_type, distribution, param_folder)
                # Same-level summary directory (not inside the parameter directory)
                self.summary_dir = os.path.join(base_dir, script_name, bc_type, distribution, "summary_comparisons")
            else:
                # When script_name is an empty string, start directly from bc_type
                self.run_root = os.path.join(base_dir, bc_type, distribution, param_folder)
                self.summary_dir = os.path.join(base_dir, bc_type, distribution, "summary_comparisons")
        else:
            # Compatible with the old structure: output directly under results/
            self.run_root = base_dir
            self.summary_dir = os.path.join(base_dir, "summary_comparisons")

        # Specific subdirectories
        self.data_dir = os.path.join(self.run_root, "data")
        self.logs_dir = os.path.join(self.run_root, "logs")
        self.models_dir = os.path.join(self.run_root, "models")
        self.plots_dir = os.path.join(self.run_root, "plots")

        # Create the directory structure
        self._create_directories()

    def _create_directories(self):
        """Create the necessary directory structure (supports Windows long paths)"""
        for dir_path in [
            self.base_dir,  # top-level results
            os.path.dirname(self.run_root),  # at least create up to the distribution level
            self.run_root,
            self.data_dir,
            self.logs_dir,
            self.models_dir,
            self.plots_dir,
            self.summary_dir,
        ]:
            try:
                # On Windows, always use long-path handling to avoid the 260-character limit
                actual_path = self._get_safe_path(dir_path)
                os.makedirs(actual_path, exist_ok=True)
            except OSError as e:
                print(f"[WARN] Failed to create directory {dir_path}: {e}")

    def _get_safe_path(self, path: str) -> str:
        """Get a Windows-safe path (handles the long-path issue)

        The Windows path length limit is 260 characters; using the \\\\?\\ prefix supports up to 32767 characters.
        """
        if os.name == 'nt':
            abs_path = os.path.abspath(path)
            # If the path length exceeds 200 or is already very long, use the long-path prefix
            if len(abs_path) > 200 and not abs_path.startswith("\\\\?\\"):
                return "\\\\?\\" + abs_path
        return path

    def _ensure_dir_exists(self, dir_path: str) -> bool:
        """Ensure the directory exists; return whether it succeeded"""
        try:
            actual_path = self._get_safe_path(dir_path)
            os.makedirs(actual_path, exist_ok=True)
            return True
        except OSError as e:
            print(f"[WARN] Failed to create directory {dir_path}: {e}")
            return False

    def get_safe_file_path(self, file_path: str) -> str:
        """Get a Windows-safe path for a file

        Used to handle the long-path issue when saving files.
        """
        return self._get_safe_path(file_path)
    
    @staticmethod
    def _format_number(x: float) -> str:
        """
        Format a number as a string, removing redundant decimal points and zeros

        Args:
            x: the number to format

        Returns:
            str: the formatted string
        """
        s = f"{x:.3f}"
        while len(s) > 1 and s[-1] == '0':
            s = s[:-1]
        if s.endswith('.'):
            s += '0'
        return s

    @staticmethod
    def _format_q(q: float) -> str:
        """Format the load value as a compact string (sign denoted by q / qn)"""

        fmt = OutputManager._format_number
        if q >= 0:
            return f"q{fmt(q)}"
        return f"qn{fmt(abs(q))}"

    @staticmethod
    def build_filename_prefix(W_Gr: float, T: float, H_Gr: float, q: float,
                              k1: float = 0.0, k2: float = 0.0,
                              activation_type: str = 'Tanh',
                              siren_omega_0: float = 30.0,
                              siren_omega_hidden: float = 30.0,
                              lifting_basis: str = 'poly') -> str:
        """Generate a standardized file name prefix (no directory side effects, easy to reuse)

        Args:
            W_Gr: graphene mass fraction
            T: temperature
            H_Gr: graphene shape factor
            q: load
            k1, k2: elastic foundation parameters
            activation_type: activation function type ('Tanh', 'Sin', 'SIREN')
            siren_omega_0: SIREN first-layer frequency factor
            siren_omega_hidden: SIREN hidden-layer frequency factor
            lifting_basis: boundary-constraint lifting basis function type ('poly', 'trig', 'none')
        """

        fmt = OutputManager._format_number
        q_str = OutputManager._format_q(q)

        filename = f"W_{fmt(W_Gr)}_T_{fmt(T)}_H_{fmt(H_Gr)}_{q_str}"

        # Activation function parameters (added before the elastic foundation parameters)
        if activation_type == 'SIREN':
            # SIREN activation: includes omega parameters
            filename += f"_SIREN_w{fmt(siren_omega_0)}_{fmt(siren_omega_hidden)}"
        else:
            # Other activation functions: only add the type name
            filename += f"_{activation_type}"

        # lifting_basis parameter (added only when not the default value poly)
        if lifting_basis.lower() != 'poly':
            filename += f"_lift{lifting_basis}"

        # Elastic foundation parameters (added when non-zero)
        if k1 != 0 or k2 != 0:
            filename += f"_k1_{fmt(k1)}_k2_{fmt(k2)}"

        return filename

    def generate_filename(self, W_Gr: float, T: float, H_Gr: float, q: float,
                         k1: float = 0.0, k2: float = 0.0,
                         activation_type: str = 'Tanh',
                         siren_omega_0: float = 30.0,
                         siren_omega_hidden: float = 30.0,
                         lifting_basis: str = 'poly') -> str:
        """
        Generate a file name prefix, a simplified version to avoid excessive length
        The main parameters are in the folder name; the file name only contains the key distinguishing parameters

        Args:
            W_Gr: graphene mass fraction
            T: temperature
            H_Gr: graphene shape factor
            q: load
            k1, k2: elastic foundation parameters
            activation_type: activation function type ('Tanh', 'Sin', 'SIREN')
            siren_omega_0: SIREN first-layer frequency factor
            siren_omega_hidden: SIREN hidden-layer frequency factor
            lifting_basis: boundary-constraint lifting basis function type ('poly', 'trig', 'none')

        Returns:
            str: file name prefix string
        """
        return self.build_filename_prefix(
            W_Gr=W_Gr,
            T=T,
            H_Gr=H_Gr,
            q=q,
            k1=k1,
            k2=k2,
            activation_type=activation_type,
            siren_omega_0=siren_omega_0,
            siren_omega_hidden=siren_omega_hidden,
            lifting_basis=lifting_basis,
        )

    @staticmethod
    def make_param_folder(W_Gr: float, T: float, H_Gr: float, q: float, lambda_val: float,
                         k1: float = 0.0, k2: float = 0.0,
                         activation_type: str = 'Tanh',
                         siren_omega_0: float = 30.0,
                         siren_omega_hidden: float = 30.0,
                         lifting_basis: str = 'poly') -> str:
        """
        Generate a parameter combination folder name, intelligently handling the activation function and elastic foundation parameters

        Base format: W{W}-T{T}-H{H}-q{q}-L{L}h (supports negative load q -> qn)
        Activation function: -SIREN_w{omega0}_{omegah} (SIREN) or -{type} (others)
        lifting_basis: -lift{basis} (appended only when not poly)
        Elastic foundation: -k{k1}_{k2} (appended only when k1 or k2 is non-zero)

        Args:
            W_Gr: graphene mass fraction
            T: temperature
            H_Gr: graphene shape factor
            q: load
            lambda_val: beam length factor (L/h)
            k1, k2: elastic foundation parameters
            activation_type: activation function type ('Tanh', 'Sin', 'SIREN')
            siren_omega_0: SIREN first-layer frequency factor
            siren_omega_hidden: SIREN hidden-layer frequency factor
            lifting_basis: boundary-constraint lifting basis function type ('poly', 'trig', 'none')

        Returns:
            str: folder name string
        """
        # Use the unified formatting method
        fmt = OutputManager._format_number

        # Base parameters (required), supporting negative q values and the L_h format
        q_str = OutputManager._format_q(q)
        folder_name = f"W{fmt(W_Gr)}-T{fmt(T)}-H{fmt(H_Gr)}-{q_str}-L{int(lambda_val)}h"

        # Activation function parameters (added before the elastic foundation parameters)
        if activation_type == 'SIREN':
            # SIREN activation: includes omega parameters
            folder_name += f"-SIREN_w{fmt(siren_omega_0)}_{fmt(siren_omega_hidden)}"
        else:
            # Other activation functions: only add the type name
            folder_name += f"-{activation_type}"

        # lifting_basis parameter (added only when not the default value poly)
        if lifting_basis.lower() != 'poly':
            folder_name += f"-lift{lifting_basis}"

        # Elastic foundation parameters (added when non-zero)
        if k1 != 0 or k2 != 0:
            folder_name += f"-k{fmt(k1)}_{fmt(k2)}"

        return folder_name
    
    def save_model(self, model: torch.nn.Module, filename: str, is_linear: bool = True):
        """
        Save the model

        Parameters:
            model: PyTorch model
            filename: file name prefix
            is_linear: whether it is a linear model
        """
        # Ensure the models directory exists
        self._ensure_dir_exists(self.models_dir)

        prefix = "Linearw" if is_linear else "Nonlinearw"
        full_path = os.path.join(self.models_dir, f"{prefix}_{filename}.pth")
        safe_full_path = self.get_safe_file_path(full_path)
        torch.save(model.state_dict(), safe_full_path)
        print(f"Model saved: {full_path}")
    
    def save_displacement_data(self, x: np.ndarray,
                              linear_results: Optional[Tuple] = None,
                              nonlinear_results: Optional[Tuple] = None,
                              filename: str = None):
        """
        Save displacement data to CSV

        Args:
            x: position coordinate array
            linear_results: linear result tuple (u, w, phi)
            nonlinear_results: nonlinear result tuple (u, w, phi)
            filename: file name prefix; a timestamp is used when empty
        """
        if filename is None:
            filename = f"displacement_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Prepare the data
        data_dict = {'x': x}
        headers = ['x']

        if linear_results is not None:
            u_lin, w_lin, phi_lin = linear_results
            data_dict.update({
                'linear_u': u_lin,
                'linear_w': w_lin,
                'linear_phi': phi_lin
            })
            headers.extend(['linear_u', 'linear_w', 'linear_phi'])
        
        if nonlinear_results is not None:
            u_non, w_non, phi_non = nonlinear_results
            data_dict.update({
                'nonlinear_u': u_non,
                'nonlinear_w': w_non,
                'nonlinear_phi': phi_non
            })
            headers.extend(['nonlinear_u', 'nonlinear_w', 'nonlinear_phi'])

        # Combine the data and save
        data_array = np.column_stack([data_dict[h] for h in headers])

        # Ensure the data directory exists
        self._ensure_dir_exists(self.data_dir)

        csv_path = os.path.join(self.data_dir, f"w_{filename}.csv")
        safe_csv_path = self.get_safe_file_path(csv_path)
        np.savetxt(safe_csv_path, data_array, delimiter=',',
                  header=','.join(headers), comments='')
        print(f"Data saved: {csv_path}")

    def save_loss_log(self,
                      linear_log: Optional[np.ndarray] = None,
                      nonlinear_log: Optional[np.ndarray] = None,
                      filename: Optional[str] = None):
        """
        Save the training loss log to CSV (alongside the displacement fields, for unified archiving)

        Parameters:
            linear_log: linear model training log, of shape (N, 9) = [epoch, total, Pi_all, bc, Pi_str, Pi_str_T, Pi_w, Pi_e, pseudo]
            nonlinear_log: nonlinear model training log, of shape (N, 9)
            filename: file name prefix (a timestamp is used if empty)
        Notes:
            - If both types of logs exist, they are written to the same file aligned row by row. When the lengths differ, they are padded with NaN.
            - The output column names follow the 'linear_*' and 'nonlinear_*' prefixes.
            - Supports adaptive column counts: 4 columns (basic) / 8 columns (extended) / 9 columns (with pseudo-supervision)
        """
        if linear_log is None and nonlinear_log is None:
            return

        if filename is None:
            filename = f"loss_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Compute the output length
        len_lin = int(linear_log.shape[0]) if linear_log is not None else 0
        len_non = int(nonlinear_log.shape[0]) if nonlinear_log is not None else 0
        max_len = max(len_lin, len_non)

        # Pre-allocate the arrays and fill with NaN
        cols = ["epoch"]
        data = []

        # Take the epoch column (prefer linear, otherwise nonlinear, otherwise 1..max_len)
        if linear_log is not None:
            epochs = linear_log[:, 0]
        elif nonlinear_log is not None:
            epochs = nonlinear_log[:, 0]
        else:
            epochs = np.arange(1, max_len + 1)

        # Truncate or pad to max_len
        if epochs.shape[0] < max_len:
            epochs = np.concatenate([epochs, np.full((max_len - epochs.shape[0],), np.nan)])
        else:
            epochs = epochs[:max_len]

        data.append(epochs)

        # Linear log columns (adaptive column count)
        if linear_log is not None:
            # Expected extended format: [epoch, total, Pi_all, bc, Pi_str, Pi_str_T, Pi_w, Pi_e, pseudo]
            cols.extend(["linear_total", "linear_Pi_all", "linear_bc"])
            lin_total = linear_log[:, 1]
            lin_Pi_all = linear_log[:, 2]
            lin_bc = linear_log[:, 3]
            series = [lin_total, lin_Pi_all, lin_bc]
            if linear_log.shape[1] >= 5:
                cols.append("linear_Pi_str")
                series.append(linear_log[:, 4])
            if linear_log.shape[1] >= 6:
                cols.append("linear_Pi_str_T")
                series.append(linear_log[:, 5])
            if linear_log.shape[1] >= 7:
                cols.append("linear_Pi_w")
                series.append(linear_log[:, 6])
            if linear_log.shape[1] >= 8:
                cols.append("linear_Pi_e")
                series.append(linear_log[:, 7])
            if linear_log.shape[1] >= 9:
                cols.append("linear_pseudo")
                series.append(linear_log[:, 8])
            for arr in series:
                if arr.shape[0] < max_len:
                    pad = np.full((max_len - arr.shape[0],), np.nan)
                    data.append(np.concatenate([arr, pad]))
                else:
                    data.append(arr[:max_len])

        # Nonlinear log columns (adaptive column count)
        if nonlinear_log is not None:
            # Expected extended format: [epoch, total, Pi_all, bc, Pi_str, Pi_str_T, Pi_w, Pi_e, pseudo]
            cols.extend(["nonlinear_total", "nonlinear_Pi_all", "nonlinear_bc"])
            non_total = nonlinear_log[:, 1]
            non_Pi_all = nonlinear_log[:, 2]
            non_bc = nonlinear_log[:, 3]
            series = [non_total, non_Pi_all, non_bc]
            if nonlinear_log.shape[1] >= 5:
                cols.append("nonlinear_Pi_str")
                series.append(nonlinear_log[:, 4])
            if nonlinear_log.shape[1] >= 6:
                cols.append("nonlinear_Pi_str_T")
                series.append(nonlinear_log[:, 5])
            if nonlinear_log.shape[1] >= 7:
                cols.append("nonlinear_Pi_w")
                series.append(nonlinear_log[:, 6])
            if nonlinear_log.shape[1] >= 8:
                cols.append("nonlinear_Pi_e")
                series.append(nonlinear_log[:, 7])
            if nonlinear_log.shape[1] >= 9:
                cols.append("nonlinear_pseudo")
                series.append(nonlinear_log[:, 8])
            for arr in series:
                if arr.shape[0] < max_len:
                    pad = np.full((max_len - arr.shape[0],), np.nan)
                    data.append(np.concatenate([arr, pad]))
                else:
                    data.append(arr[:max_len])

        # Assemble and save
        out = np.column_stack(data)

        # Ensure the logs directory exists
        self._ensure_dir_exists(self.logs_dir)

        csv_path = os.path.join(self.logs_dir, f"loss_{filename}.csv")
        safe_csv_path = self.get_safe_file_path(csv_path)
        np.savetxt(safe_csv_path, out, delimiter=",", header=",".join(cols), comments="")
        print(f"Loss log saved: {csv_path}")
    
    def plot_training_curves(self, linear_log: Optional[np.ndarray] = None,
                           nonlinear_log: Optional[np.ndarray] = None,
                           filename: str = None):
        """
        Plot the training loss curves

        Parameters:
            linear_log: linear model training log
            nonlinear_log: nonlinear model training log
            filename: file name prefix
        """
        if linear_log is None and nonlinear_log is None:
            return

        plt.figure(figsize=(10, 6))

        if linear_log is not None:
            epochs_lin = linear_log[:, 0]
            loss_lin = linear_log[:, 1]  # training loss
            plt.plot(epochs_lin, loss_lin, 'b-', label='Linear Model', alpha=0.7)

        if nonlinear_log is not None:
            epochs_non = nonlinear_log[:, 0]
            loss_non = nonlinear_log[:, 1]  # training loss
            plt.plot(epochs_non, loss_non, 'r-', label='Nonlinear Model', alpha=0.7)

        plt.xlabel('Epoch')
        plt.ylabel('Training Loss')
        plt.title('Training Progress')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log')

        if filename:
            # Ensure the plots directory exists
            self._ensure_dir_exists(self.plots_dir)

            save_path = os.path.join(self.plots_dir, f"{filename}_loss.png")
            safe_save_path = self.get_safe_file_path(save_path)
            plt.savefig(safe_save_path, dpi=150, bbox_inches='tight')
            print(f"Training curves saved: {save_path}")
        
        plt.close()
    
    def plot_displacement_comparison(self, x: np.ndarray,
                                    linear_results: Optional[Tuple] = None,
                                    nonlinear_results: Optional[Tuple] = None,
                                    filename: str = None):
        """
        Plot the displacement comparison figures

        Parameters:
            x: position coordinate
            linear_results: linear results (u, w, phi)
            nonlinear_results: nonlinear results (u, w, phi)
            filename: file name prefix
        """
        if linear_results is None and nonlinear_results is None:
            return

        labels = ['Axial displacement (u)', 'Deflection (w)', 'Rotation (phi)']

        for i, label in enumerate(labels):
            plt.figure(figsize=(10, 6))

            if linear_results is not None:
                y_lin = linear_results[i]
                plt.plot(x, y_lin, 'b-', label='Linear', linewidth=2)

            if nonlinear_results is not None:
                y_non = nonlinear_results[i]
                plt.plot(x, y_non, 'r--', label='Nonlinear', linewidth=2)

            plt.xlabel('Position x/L')
            plt.ylabel(label)
            plt.title(f'{label} Comparison')
            plt.legend()
            plt.grid(True, alpha=0.3)

            if filename:
                # Ensure the plots directory exists
                self._ensure_dir_exists(self.plots_dir)

                field_name = ['u', 'w', 'phi'][i]
                save_path = os.path.join(self.plots_dir, f"{filename}_{field_name}.png")
                safe_save_path = self.get_safe_file_path(save_path)
                plt.savefig(safe_save_path, dpi=150, bbox_inches='tight')
                print(f"Comparison plot saved: {save_path}")
            
            plt.close()
    
    def print_summary(self, results: Dict):
        """
        Print a summary of the results

        Parameters:
            results: result dictionary
        """
        print("\n" + "=" * 60)
        print("Solution Results Summary")
        print("=" * 60)

        if 'linear' in results:
            print("\nLinear Model:")
            self._print_model_summary(results['linear'])

        if 'nonlinear' in results:
            print("\nNonlinear Model:")
            self._print_model_summary(results['nonlinear'])

        print("=" * 60)
    
    def _print_model_summary(self, model_results: Dict):
        """Print summary for single model"""
        if 'max_w' in model_results:
            print(f"  Max deflection: {model_results['max_w']:.6f}")
        if 'final_loss' in model_results:
            print(f"  Final loss: {model_results['final_loss']:.6e}")
        if 'training_time' in model_results:
            print(f"  Training time: {model_results['training_time']:.2f} sec")

    def update_index(self,
                     prefix: str,
                     params: Dict,
                     linear_meta: Optional[Dict] = None,
                     nonlinear_meta: Optional[Dict] = None,
                     data_files: Optional[Dict] = None) -> None:
        """
        Update the results/index.json metadata index

        Parameters:
            prefix: file name prefix (W_Gr_T_H_Gr_ or q or )
            params: run parameter dictionary (geometry, material, boundary, network, etc.)
            linear_meta: linear model metadata (best_epoch, best_loss, final_loss, max_w, model_path, plots)
            nonlinear_meta: nonlinear model metadata (same as above)
            data_files: data file paths (displacement CSV, loss CSV, etc.)
        """
        # The index file is written to the logs directory under the current run directory
        index_path = os.path.join(self.logs_dir, "index.json")
        safe_index_path = self.get_safe_file_path(index_path)
        runs = []
        if os.path.exists(safe_index_path):
            try:
                with open(safe_index_path, "r", encoding="utf-8") as f:
                    obj = json.load(f)
                    runs = obj.get("runs", [])
            except Exception:
                # If the index is corrupted, start from empty
                runs = []

        entry = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "prefix": prefix,
            "params": params,
            "linear": linear_meta,
            "nonlinear": nonlinear_meta,
            "data": data_files or {},
        }

        # Deduplicate: replace the old record with the same prefix
        runs = [e for e in runs if e.get("prefix") != prefix]
        runs.append(entry)

        # Ensure the logs directory exists
        self._ensure_dir_exists(self.logs_dir)

        with open(safe_index_path, "w", encoding="utf-8") as f:
            # Provide both the single run entry and the history list, for flexible reading by upper-layer tools
            json.dump({"run": entry, "runs": runs}, f, ensure_ascii=False, indent=2)
        print(f"Index updated: {index_path}")
