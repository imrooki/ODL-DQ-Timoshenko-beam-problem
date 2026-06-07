"""
Timoshenko beam PINNs core module aggregation package

Author: Yang
Version: 1.0

Description:
- Provides a unified public API exporting network architectures, data types, solvers, and loss functions
- Manages inter-module dependencies to avoid circular import problems
- Supports compatibility handling for both relative and absolute imports
- Organizes exported interfaces by function: type definitions, numerical tools, network construction, boundary conditions, physical losses, solvers

Module layer hierarchy:
1. Bottom: data_types (data type definitions)
2. Foundation: numerics (numerical computation tools)
3. Network: nets (neural network architectures)
4. Boundary: bc (boundary condition handling)
5. Physics: physics (energy loss computation)
6. Top: solver (PINNs solver)
"""

# Compatibility for running the script directly: fall back to absolute imports when relative imports fail
try:
    # Import from bottom to top following the dependency order, to avoid circular dependencies

    # Type definitions (single source)
    from .data_types import (
        MaterialCoeffs,
        PhysicalParams,
        BoundaryConditions,
        BoundaryConditionType,
        DistributionType,
    )

    # Numerical tools (base dependency)
    from .numerics import d_dx, compute_derivatives, safe_divide

    # Networks and builders
    from .nets import (
        SharedEncoderMultiDecoder,
        build_timoshenko_net,
    )

    # Boundary condition handling
    from .bc import BoundaryConditionPenalty, lifting

    # Energy loss computation
    from .physics import (
        EnergyLoss,
        WeightedEnergyLoss,
        create_loss_function,
    )

    # Solver and training (top-level dependency)
    from .solver import EnergyPINNStatic, as_fun, train_model

except ImportError:
    import os, sys
    pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if pkg_root not in sys.path:
        sys.path.insert(0, pkg_root)

    # Import from bottom to top following the dependency order, to avoid circular dependencies
    from modules.data_types import (
        MaterialCoeffs,
        PhysicalParams,
        BoundaryConditions,
        BoundaryConditionType,
        DistributionType,
    )
    from modules.numerics import d_dx, compute_derivatives, safe_divide
    from modules.nets import (
        SharedEncoderMultiDecoder,
        build_timoshenko_net,
    )
    from modules.bc import BoundaryConditionPenalty, lifting
    from modules.physics import (
        EnergyLoss,
        WeightedEnergyLoss,
        create_loss_function,
    )
    from modules.solver import EnergyPINNStatic, as_fun, train_model

__all__ = [
    # Types and enums
    "MaterialCoeffs",
    "PhysicalParams",
    "BoundaryConditions",
    "BoundaryConditionType",
    "DistributionType",
    # Networks and builders
    "SharedEncoderMultiDecoder",
    "build_timoshenko_net",
    # Solver
    "EnergyPINNStatic",
    "as_fun",
    "train_model",
    # Losses and boundaries
    "EnergyLoss",
    "WeightedEnergyLoss",
    "create_loss_function",
    "BoundaryConditionPenalty",
    "lifting",
    # Numerical tools
    "d_dx",
    "compute_derivatives",
    "safe_divide",
]
