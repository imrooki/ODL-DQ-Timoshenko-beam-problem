"""
Boundary Condition Handling Module
===================

This module implements various boundary conditions for the Timoshenko beam, supporting hard- and soft-constraint strategies of the ODIL framework:

Supported boundary condition types:
1. C-C: Clamped-Clamped - hard constraint
2. S-S: Simply-Supported - soft constraint
3. C-S: Clamped-Simply - soft constraint
4. C-H: Clamped-Hinged - soft constraint
5. H-H: Hinged-Hinged - soft constraint

Constraint strategies:
- The C-C boundary condition uses a hard constraint (inject_dirichlet)
- Other boundary conditions use soft constraints (added to the loss function)
- Compatible with all discretization methods (DQ, Taylor, Spline)
"""

import torch
from typing import Dict, Optional

def get_boundary_conditions(bc_type: str, device: Optional[torch.device] = None) -> Dict:
    """
    Return the corresponding boundary values according to the boundary condition type

    Parameters:
        bc_type: boundary condition type string
        device: compute device

    Returns:
        Dictionary containing the u, w, phi boundary values
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Helper function to create a zero tensor
    def zero():
        return torch.tensor(0.0, device=device, dtype=torch.float64)

    # Definitions of all boundary condition types
    bc_dict = {
        'C-C': {  # Clamped-Clamped
            'u': (zero(), zero()),    # u(0)=0, u(L)=0
            'w': (zero(), zero()),    # w(0)=0, w(L)=0
            'phi': (zero(), zero()),  # φ(0)=0, φ(L)=0
            'description': 'Clamped-Clamped'
        },
        'S-S': {  # Simply-Supported
            'u': (zero(), zero()),    # u(0)=0, u(L)=0
            'w': (zero(), zero()),    # w(0)=0, w(L)=0
            'phi': (None, None),      # φ free, but the M=0 condition is required
            'description': 'Simply-Supported',
            'additional': 'M(0)=0, M(L)=0'  # bending moment is zero
        },
        'H-H': {  # Hinged-Hinged (similar to simply supported)
            'u': (zero(), zero()),    # u(0)=0, u(L)=0
            'w': (zero(), zero()),    # w(0)=0, w(L)=0
            'phi': (None, None),      # φ free
            'description': 'Hinged-Hinged',
            'additional': 'M(0)=0, M(L)=0'
        },
        'C-S': {  # Clamped-Simply
            'u': (zero(), zero()),    # u(0)=0, u(L)=0
            'w': (zero(), zero()),    # w(0)=0, w(L)=0
            'phi': (zero(), None),    # φ(0)=0, φ(L) free but M(L)=0
            'description': 'Clamped-Simply',
            'additional': 'M(L)=0'
        },
        'C-H': {  # Clamped-Hinged
            'u': (zero(), zero()),    # u(0)=0, u(L)=0
            'w': (zero(), zero()),    # w(0)=0, w(L)=0
            'phi': (zero(), None),    # φ(0)=0, φ(L) free
            'description': 'Clamped-Hinged',
            'additional': 'M(L)=0'
        }
    }

    if bc_type not in bc_dict:
        raise ValueError(f"Unsupported boundary condition type: {bc_type}. "
                        f"Supported types: {list(bc_dict.keys())}")

    bc = bc_dict[bc_type].copy()

    # Handle boundary conditions that need special treatment (e.g., bending moment conditions)
    # Note: for H-H and S-S, phi is handled specially in the solver; here it is just a placeholder
    if bc_type in ['H-H', 'S-S']:
        # For these boundary conditions, φ is determined by M=0 and no hard constraint is used
        bc['phi'] = (zero(), zero())  # placeholder value, not actually used
    elif bc_type in ['C-S', 'C-H']:
        # Left end fixed, right end determined by M=0
        bc['phi'] = (zero(), zero())  # placeholder value for the right end

    # Return the boundary values and type information
    result = {
        'u': bc['u'],
        'w': bc['w'],
        'phi': bc['phi'],
        'type': bc_type
    }
    
    return result



def print_boundary_info(bc_type: str):
    """
    Print detailed information about the boundary condition

    Parameters:
        bc_type: boundary condition type
    """
    info = {
        'C-C': {
            'name': 'Clamped-Clamped',
            'conditions': [
                'Left boundary: u(0)=0, w(0)=0, φ(0)=0',
                'Right boundary: u(L)=0, w(L)=0, φ(L)=0'
            ],
            'physical': 'Both ends fully fixed, no displacement or rotation'
        },
        'S-S': {
            'name': 'Simply-Supported',
            'conditions': [
                'Left boundary: u(0)=0, w(0)=0, M(0)=0',
                'Right boundary: u(L)=0, w(L)=0, M(L)=0'
            ],
            'physical': 'Both ends can rotate but cannot translate, bending moment is zero'
        },
        'C-S': {
            'name': 'Clamped-Simply',
            'conditions': [
                'Left boundary: u(0)=0, w(0)=0, φ(0)=0',
                'Right boundary: u(L)=0, w(L)=0, M(L)=0'
            ],
            'physical': 'Left end fixed, right end can rotate'
        },
        'C-H': {
            'name': 'Clamped-Hinged',
            'conditions': [
                'Left boundary: u(0)=0, w(0)=0, φ(0)=0',
                'Right boundary: u(L)=0, w(L)=0, M(L)=0'
            ],
            'physical': 'Left end fixed, right end hinged'
        },
        'H-H': {
            'name': 'Hinged-Hinged',
            'conditions': [
                'Left boundary: u(0)=0, w(0)=0, M(0)=0',
                'Right boundary: u(L)=0, w(L)=0, M(L)=0'
            ],
            'physical': 'Both ends hinged, free to rotate'
        }
    }

    if bc_type in info:
        bc_info = info[bc_type]
        print(f"\nBoundary condition: {bc_info['name']}")
        print("-" * 40)
        for condition in bc_info['conditions']:
            print(f"  - {condition}")
        print(f"Physical meaning: {bc_info['physical']}")
    else:
        print(f"Unknown boundary condition type: {bc_type}")

def get_all_boundary_types() -> list:
    """
    Get all supported boundary condition types

    Returns:
        List of boundary condition types
    """
    return ['C-C', 'S-S', 'C-S', 'C-H', 'H-H']