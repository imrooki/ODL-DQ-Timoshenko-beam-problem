

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
    
    
    def __init__(self, name: str = 'ODIL-DQ', mode: str = 'simple',
                 level: str = 'INFO', verbose: bool = True, log_dir: Optional[str] = None,
                 config: Optional[Any] = None):
        
        self.name = name
        self.mode = mode
        self.verbose = verbose
        self.log_dir = log_dir
        self.config = config
        self.iteration_count = 0
        self.start_time = None
        self.loss_history = []
        
        
        self.logger = logging.getLogger(name)
        self.logger.handlers.clear()  

        
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
        
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter('[%(name)s] %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def _setup_full_handlers(self):
        
        console_handler = logging.StreamHandler(sys.stdout)
        console_format = logging.Formatter(
            '[%(name)s-%(levelname)5s] %(message)s'
        )
        console_handler.setFormatter(console_format)

        
        if self.log_dir:
            log_dir_path = Path(self.log_dir).resolve()
        else:
            default_dir = _cfg(self.config, 'archived_logs_dir', 'archived_logs')
            log_dir_path = Path.cwd() / default_dir
            self.logger.warning(f"No log directory provided, logs will be saved to the default location: {log_dir_path}")

        
        log_dir_path.mkdir(parents=True, exist_ok=True)
        log_filename = log_dir_path / f'odil_dq_{datetime.now():%Y%m%d_%H%M%S}.log'
            
        file_handler = logging.FileHandler(str(log_filename), encoding='utf-8')
        file_format = logging.Formatter(
            '%(asctime)s | %(name)s | %(levelname)s | %(funcName)s | %(message)s'
        )
        file_handler.setFormatter(file_format)
        
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)
    
    def optimization_start(self, mode: str, bc_type: str, N: int,
                         optimizer: str, constraint_type: str = None):
        
        if not self.verbose:
            return

        self.start_time = datetime.now()

        if self.mode == 'simple':
            constraint = "hard constraint" if bc_type == 'C-C' else "soft constraint"
            self.logger.info(f"  Boundary condition: {bc_type} - using {constraint}")
            if bc_type == 'C-C':
                self.logger.info(f"  Parameter dimensions: u={N-2}, w={N-2}, phi={N-2}")
            else:
                self.logger.info(f"  Parameter dimensions: u={N}, w={N}, phi={N}")
        else:
            self.logger.info("="*60)
            self.logger.info(f"Starting {mode} solving")
            self.logger.info(f"Boundary condition: {bc_type} | Number of nodes: {N} | Optimizer: {optimizer}")
            constraint = "hard constraint" if bc_type == 'C-C' else "soft constraint"
            self.logger.info(f"Constraint type: {constraint}")
            self.logger.info("="*60)
    
    def iteration(self, iter_num: int, loss: float, optimizer: str = 'L-BFGS',
                 grad_norm: Optional[float] = None, lr: Optional[float] = None):
        
        if not self.verbose:
            return

        self.iteration_count = iter_num
        self.loss_history.append(loss)

        if self.mode == 'simple':
            self.logger.info(f"[{optimizer}] {'Epoch' if optimizer=='L-BFGS' else 'iter'} {iter_num:4d}  loss={loss:.3e}")
        else:
            trend = self._get_trend()
            msg = f"Iter {iter_num:5d} | Loss: {loss:.3e} {trend}"
            if grad_norm is not None:
                msg += f" | Grad: {grad_norm:.2e}"
            if lr is not None:
                msg += f" | LR: {lr:.3f}"
            self.logger.info(msg)
    
    def convergence(self, reason: str, final_loss: float, iterations: int):
        
        if not self.verbose:
            return

        if self.mode == 'simple':
            self.logger.info(f"[Final] loss = {final_loss:.3e}")
        else:
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
        
        if not self.verbose:
            return

        if self.mode == 'simple':
            self.logger.info(f"[Warning] {message}")
        else:
            self.logger.warning(message)

    def error(self, message: str, exception: Optional[Exception] = None):
        
        if self.mode == 'simple':
            self.logger.error(f"[Error] {message}")
            if exception:
                self.logger.error(f"  Details: {str(exception)}")
        else:
            self.logger.error(message)
            if exception:
                self.logger.error(f"Exception: {exception}", exc_info=True)

    def info(self, message: str):
        
        if not self.verbose:
            return
        self.logger.info(message)

    def debug(self, message: str):
        
        if not self.verbose:
            return
        self.logger.debug(message)

    def _get_trend(self) -> str:
        
        if len(self.loss_history) < 2:
            return ""
        
        current = self.loss_history[-1]
        previous = self.loss_history[-2]
        
        try:
            if current < previous * 0.99:
                return "↓"
            elif current > previous * 1.01:
                return "↑"
            else:
                return "→"
        except:
            
            if current < previous * 0.99:
                return "v"
            elif current > previous * 1.01:
                return "^"
            else:
                return "-"
    

