"""
Data types and enumeration definitions module for the Timoshenko beam PINN

Author: Yang
Version: 1.0

Responsibilities:
- Centrally manage data structures and enumerations as the single source of truth across modules
- Define data types for material coefficients, physical parameters, boundary conditions, and other configurations
- Provide type aliases compatible with existing code for smooth migration and extension
- Support all physical quantities and boundary condition types in Timoshenko beam theory

Core data types:
- BoundaryConditionType: boundary condition enumeration (C-C, S-S, H-H, etc.)
- DistributionType: material distribution type enumeration (X-type, O-type, U-type)
- MaterialCoeffs: material stiffness coefficient data class
- PhysicalParams: physical parameter data class
- BoundaryConditions: boundary condition configuration data class

Design principles:
- This module contains no numerical algorithms; it only defines data structures and types
- All function-type coefficients support spatial variation (function-form input/output)
- Equal emphasis on type safety and runtime validation
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional, Dict

import torch



class BoundaryConditionType(str, Enum):
    """Boundary condition type enumeration"""

    CLAMPED_CLAMPED = "C-C"  # clamped-clamped
    SIMPLE_SIMPLE = "S-S"  # simply supported-simply supported
    HINGED_HINGED = "H-H"  # hinged-hinged
    CLAMPED_SIMPLE = "C-S"  # clamped-simply supported
    CLAMPED_HINGED = "C-H"  # clamped-hinged
    CLAMPED_FREE = "C-F"  # clamped-free


# Only the shared network architecture is used.


class DistributionType(str, Enum):
    """Material distribution type enumeration"""

    X_TYPE = "X"  # X-type distribution
    O_TYPE = "O"  # O-type distribution
    U_TYPE = "U"  # U-type distribution


@dataclass
class MaterialCoeffs:
    """Material stiffness coefficients (function form, supports spatially varying coefficients)"""

    a11: Callable[[torch.Tensor], torch.Tensor]  # axial stiffness
    b11: Callable[[torch.Tensor], torch.Tensor]  # extension-bending coupling
    d11: Callable[[torch.Tensor], torch.Tensor]  # bending stiffness
    a55: Callable[[torch.Tensor], torch.Tensor]  # shear stiffness


@dataclass
class PhysicalParams:
    """Physical parameters (dimensionless)"""

    lambda_val: float = 1.0  # slenderness ratio lambda = L/h
    q: float = 0.0  # distributed load (dimensionless)
    n_xT: float = 0.0  # dimensionless thermal axial force
    m_xT: float = 0.0  # dimensionless thermal bending moment (kept, currently not directly used)
    alpha_t: float = 0.0  # backward compatibility (thermal expansion coefficient)
    DeltaT: float = 0.0  # backward compatibility (temperature change)

    # Elastic foundation parameters
    k1: float = 0.0  # elastic foundation Winkler stiffness coefficient (dimensionless)
    k2: float = 0.0  # elastic foundation Pasternak stiffness coefficient (dimensionless)


@dataclass
class BoundaryConditions:
    """Boundary condition specification (a Dirichlet value of None denotes free)"""

    type: str = "C-C"
    u_left: Optional[float] = 0.0
    u_right: Optional[float] = None
    w_left: Optional[float] = 0.0
    w_right: Optional[float] = 0.0
    phi_left: Optional[float] = 0.0
    phi_right: Optional[float] = 0.0


__all__ = [
    # Data types
    "MaterialCoeffs",
    "PhysicalParams",
    "BoundaryConditions",
    # Enumerations
    "BoundaryConditionType",
    "DistributionType",
]
