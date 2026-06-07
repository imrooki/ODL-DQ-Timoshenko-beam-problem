"""
ODIL-Timoshenko beam solver core module package

Contains all core computation modules:
- material_properties: material property computation
- solver_odil: ODIL framework solver
- dq_core: DQ differential quadrature method
- taylor_core: Taylor/Fornberg discretization method
- spline_core: spline collocation method
- method_factory: unified discretization method factory
- residuals: PDE residual computation module

Supports combinations of three discretization methods (DQ, Taylor, Spline) with four optimizers
"""

from .material_properties import MaterialCalculator, MaterialConstants

__all__ = [
    'MaterialCalculator',
    'MaterialConstants'
]