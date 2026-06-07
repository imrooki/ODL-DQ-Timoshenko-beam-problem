"""
ODIL-DQ project unified log manager

Provides two logging modes:
1. simple mode: keeps the existing simple output format
2. full mode: provides structured detailed logs
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Any, Optional


def _cfg(config: Optional[Any], name: str, default=None):
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(name, default)
    return getattr(config, name, default)


class ODILLogger:
    """
    ODIL-DQ project unified log manager

    Supports two modes:
    - simple: similar to the existing print output, keeps backward compatibility
    - full: structured detailed logs, providing more information
    """
    
    def __init__(self, name: str = 'ODIL-DQ', mode: str = 'simple',
                 level: str = 'INFO', verbose: bool = True, log_dir: Optional[str] = None,
                 config: Optional[Any] = None):
        """Initialize the log manager."""
        self.name = name
        self.mode = mode
        self.verbose = verbose
        self.log_dir = log_dir
        self.config = config
        self.iteration_count = 0
        self.start_time = None
        self.loss_history = []
        
        # Create logger
        self.logger = logging.getLogger(name)
        self.logger.handlers.clear()  # Clear any pre-existing old handlers

        # Set the log level
        level_map = {
            'DEBUG': logging.DEBUG,
            'INFO': logging.INFO,
            'WARNING': logging.WARNING,
            'ERROR': logging.ERROR
        }
        self.logger.setLevel(level_map.get(level, logging.INFO))
        
        if mode == 'simple':
            self._setup_simple_handler()
        else:
            self._setup_full_handlers()
    
    def _setup_simple_handler(self):
        """Set up the simple-mode handler (similar to the current print)"""
        handler = logging.StreamHandler(sys.stdout)
        # Simple format, similar to the existing output
        formatter = logging.Formatter('[%(name)s] %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def _setup_full_handlers(self):
        """Set up the full-mode handlers"""
        # Console output
        console_handler = logging.StreamHandler(sys.stdout)
        console_format = logging.Formatter(
            '[%(name)s-%(levelname)5s] %(message)s'
        )
        console_handler.setFormatter(console_format)

        # File output - unified path handling logic
        # Determine the log directory (prefer the provided log_dir)
        if self.log_dir:
            log_dir_path = Path(self.log_dir).resolve()
        else:
            default_dir = _cfg(self.config, 'archived_logs_dir', 'archived_logs')
            log_dir_path = Path.cwd() / default_dir
            self.logger.warning(f"No log directory provided, logs will be saved to the default location: {log_dir_path}")

        # Create the directory and log file
        log_dir_path.mkdir(parents=True, exist_ok=True)
        log_filename = log_dir_path / f'odil_dq_{datetime.now():%Y%m%d_%H%M%S}.log'
            
        file_handler = logging.FileHandler(str(log_filename), encoding='utf-8')
        file_format = logging.Formatter(
            '%(asctime)s | %(name)s | %(levelname)s | %(funcName)s | %(message)s'
        )
        file_handler.setFormatter(file_format)
        
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)
    
    # ============ Core logging methods ============

    def optimization_start(self, mode: str, bc_type: str, N: int,
                         optimizer: str, constraint_type: str = None):
        """Record the start of optimization"""
        if not self.verbose:
            return

        self.start_time = datetime.now()

        if self.mode == 'simple':
            # Simple mode (keep the existing format)
            constraint = "hard constraint" if bc_type == 'C-C' else "soft constraint"
            self.logger.info(f"  Boundary condition: {bc_type} - using {constraint}")
            if bc_type == 'C-C':
                self.logger.info(f"  Parameter dimensions: u={N-2}, w={N-2}, phi={N-2}")
            else:
                self.logger.info(f"  Parameter dimensions: u={N}, w={N}, phi={N}")
        else:
            # Full mode
            self.logger.info("="*60)
            self.logger.info(f"Starting {mode} solving")
            self.logger.info(f"Boundary condition: {bc_type} | Number of nodes: {N} | Optimizer: {optimizer}")
            constraint = "hard constraint" if bc_type == 'C-C' else "soft constraint"
            self.logger.info(f"Constraint type: {constraint}")
            self.logger.info("="*60)
    
    def iteration(self, iter_num: int, loss: float, optimizer: str = 'L-BFGS',
                 grad_norm: Optional[float] = None, lr: Optional[float] = None):
        """Record iteration information"""
        if not self.verbose:
            return

        self.iteration_count = iter_num
        self.loss_history.append(loss)

        if self.mode == 'simple':
            # Simple mode (keep the existing format)
            self.logger.info(f"[{optimizer}] {'Epoch' if optimizer=='L-BFGS' else 'iter'} {iter_num:4d}  loss={loss:.3e}")
        else:
            # Full mode
            trend = self._get_trend()
            msg = f"Iter {iter_num:5d} | Loss: {loss:.3e} {trend}"
            if grad_norm is not None:
                msg += f" | Grad: {grad_norm:.2e}"
            if lr is not None:
                msg += f" | LR: {lr:.3f}"
            self.logger.info(msg)
    
    def convergence(self, reason: str, final_loss: float, iterations: int):
        """Record optimization convergence"""
        if not self.verbose:
            return

        if self.mode == 'simple':
            # Simple mode
            self.logger.info(f"[Final] loss = {final_loss:.3e}")
        else:
            # Full mode
            if self.start_time:
                elapsed = (datetime.now() - self.start_time).total_seconds()
            else:
                elapsed = 0

            self.logger.info("-"*60)
            self.logger.info(f"Optimization converged: {reason}")
            self.logger.info(f"Final loss: {final_loss:.3e}")
            self.logger.info(f"Total iterations: {iterations}")
            if elapsed > 0:
                self.logger.info(f"Elapsed time: {elapsed:.2f} s")
            self.logger.info("="*60)
    
    def warning(self, message: str):
        """Record a warning message"""
        if not self.verbose:
            return

        if self.mode == 'simple':
            self.logger.info(f"[Warning] {message}")
        else:
            self.logger.warning(message)

    def error(self, message: str, exception: Optional[Exception] = None):
        """Record an error message"""
        if self.mode == 'simple':
            self.logger.error(f"[Error] {message}")
            if exception:
                self.logger.error(f"  Details: {str(exception)}")
        else:
            self.logger.error(message)
            if exception:
                self.logger.error(f"Exception: {exception}", exc_info=True)

    def info(self, message: str):
        """Record general information"""
        if not self.verbose:
            return
        self.logger.info(message)

    def debug(self, message: str):
        """Record debug information"""
        if not self.verbose:
            return
        self.logger.debug(message)

    # ============ Helper methods ============

    def _get_trend(self) -> str:
        """Get the loss trend indicator"""
        if len(self.loss_history) < 2:
            return ""
        
        current = self.loss_history[-1]
        previous = self.loss_history[-2]
        
        try:
            # Try to use Unicode characters
            if current < previous * 0.99:
                return "↓"  # Decreasing
            elif current > previous * 1.01:
                return "↑"  # Increasing
            else:
                return "→"  # Steady
        except:
            # The Windows console may not support it, use ASCII instead
            if current < previous * 0.99:
                return "v"  # Decreasing
            elif current > previous * 1.01:
                return "^"  # Increasing
            else:
                return "-"  # Steady
    
# Convenience function
