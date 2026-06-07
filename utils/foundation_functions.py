"""
Elastic foundation parameter helpers
=====================================

This module provides validation and description utilities for the elastic
foundation parameters used by the Timoshenko beam solver:

- k1: Winkler foundation stiffness (dimensionless)
- k2: Pasternak foundation stiffness (dimensionless)

Author: ODIL-Timoshenko project team
"""


def validate_foundation_parameters(k1: float, k2: float) -> bool:
    """
    Validate the elastic foundation parameters

    Parameters:
        k1: Winkler foundation stiffness (dimensionless)
        k2: Pasternak foundation stiffness (dimensionless)

    Returns:
        whether it is a valid parameter combination
    """
    # Check parameter ranges
    if not (0 <= k1 <= 200):
        print(f"Warning: Winkler foundation stiffness k1={k1} is outside the recommended range [0, 200]")
        return False

    if not (0 <= k2 <= 200):
        print(f"Warning: Pasternak foundation stiffness k2={k2} is outside the recommended range [0, 200]")
        return False

    return True


def get_foundation_description(k1: float, k2: float) -> str:
    """
    Generate the descriptive text for the elastic foundation parameters

    Parameters:
        k1: Winkler foundation stiffness
        k2: Pasternak foundation stiffness

    Returns:
        descriptive text
    """
    if k1 == 0 and k2 == 0:
        return "No elastic foundation (free beam)"

    descriptions = []
    if k1 > 0:
        descriptions.append(f"Winkler foundation (k1={k1})")
    if k2 > 0:
        descriptions.append(f"Pasternak foundation (k2={k2})")

    return " + ".join(descriptions)
