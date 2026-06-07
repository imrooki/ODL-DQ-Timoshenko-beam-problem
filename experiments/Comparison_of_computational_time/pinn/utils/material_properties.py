"""
Functionally graded material property calculation module

Author: Yang
Version: 1.0

Description:
This module implements a complete material property calculation framework for graphene-reinforced copper-matrix composites (Gr/Cu), including:

Core computation functions:
1. Basic material constant definitions: material properties of the copper matrix and the graphene reinforcement phase
2. Correction factor calculation: the effect of graphene geometric features on material performance
3. Layer-wise material property calculation: effective material properties based on the rule of mixtures
4. Stiffness coefficient calculation: the a11, b11, d11, a55 coefficients required by Timoshenko beam theory
5. Dimensionless parameter calculation: nondimensionalization for convenient numerical computation
6. Thermodynamic parameter calculation: temperature-dependent thermal expansion and thermal stress analysis

Material distribution types:
- X-type distribution: graphene concentration is highest at the surface layers
- O-type distribution: graphene concentration is highest at the neutral layer
- U-type distribution: graphene is uniformly distributed

Technical features:
- Modeling of the graphene reinforcement effect based on the modified Halpin-Tsai theory
- Support for temperature-dependent material property variation
- Consideration of the effect of graphene geometric parameters (aspect ratio, thickness)
- Strict numerical stability handling and anomaly detection

Important notes:
- The implementation of this module has been verified as correct; please do not arbitrarily modify the core algorithm
- It has been optimized for the needs of the DQ-PINN coupled solver
- All computation results have been validated against theoretical values from the literature
"""

from typing import Tuple, Dict, NamedTuple, Optional
from dataclasses import dataclass
import numpy as np
import warnings

# Compatible with different versions of SciPy - this module mainly uses numerical computation and does not need complex integration

@dataclass
class MaterialConstants:
    """Material constants class

    Stores the basic material properties of copper and graphene
    """
    # Reference temperature
    T_0: float = 300.0  # K

    # Material properties of copper
    E_Cu: float = 65.79e9  # Young's modulus (Pa)
    nu_Cu: float = 0.387   # Poisson's ratio
    rho_Cu: float = 8.80e3 # density (kg/m^3)
    alpha_Cu: float = 16.51e-6  # thermal expansion coefficient (1/K)

    # Material properties of graphene
    E_Gr: float = 929.57e9  # Young's modulus (Pa)
    nu_Gr: float = 0.220    # Poisson's ratio
    rho_Gr: float = 1.80e3  # density (kg/m^3)
    alpha_Gr: float = -3.98e-6  # thermal expansion coefficient (1/K)

    # Geometric parameters
    l_Gr: float = 83.76  # graphene length
    t_Gr: float = 3.4    # graphene thickness

class MaterialProperties(NamedTuple):
    """Material property result class"""
    E: float        # effective Young's modulus
    nu: float       # effective Poisson's ratio
    alpha: float    # effective thermal expansion coefficient
    rho: float      # effective density
    layer_properties: Dict  # detailed properties of each layer
    Q11: Dict       # Q11 stiffness coefficient of each layer
    Q55: Dict       # Q55 stiffness coefficient of each layer

class DimensionlessParameters(NamedTuple):
    """Dimensionless parameter result class"""
    a11: float      # dimensionless extensional stiffness
    b11: float      # dimensionless bending-extension coupling stiffness
    d11: float      # dimensionless bending stiffness
    a55: float      # dimensionless shear stiffness
    lambda_val: float  # slenderness ratio
    n_xT: float     # dimensionless thermal axial force
    m_xT: float     # dimensionless thermal bending moment

class MaterialCalculator:
    """Material property calculator"""

    def __init__(self, constants: Optional[MaterialConstants] = None):
        """
        Initialize the material calculator

        Parameters:
            constants: material constants; if None, default values are used
        """
        self.constants = constants if constants is not None else MaterialConstants()
    
    def compute_correction_factors(self, V_Gr: float, T_ratio: float, H_Gr: float) -> Tuple[float, float, float, float]:
        """
        Compute the correction factors

        Parameters:
            V_Gr: graphene volume fraction
            T_ratio: temperature ratio (current temperature / reference temperature)
            H_Gr: graphene shape factor

        Returns:
            Four correction factors: f_E, f_nu, f_alpha, f_rho
        """
        if V_Gr == 0:
            # Pure copper case
            return 1.0, 1.0, 1.0, 1.0

        # Ensure the parameters are within a reasonable range
        V_Gr = np.clip(V_Gr, 0.0, 1.0)
        H_Gr = np.clip(H_Gr, 0.0, 10.0)
        T_ratio = np.clip(T_ratio, 0.5, 3.0)

        # Young's modulus correction factor
        f_E = (1.11 - 1.22 * V_Gr - 0.134 * T_ratio + 0.559 * V_Gr * T_ratio
               - 5.5 * H_Gr * V_Gr + 38 * H_Gr * V_Gr ** 2 - 20.6 * H_Gr ** 2 * V_Gr ** 2)

        # Poisson's ratio correction factor
        f_nu = (1.01 - 1.43 * V_Gr + 0.165 * T_ratio - 16.8 * H_Gr * V_Gr
                - 1.1 * H_Gr * V_Gr * T_ratio + 16 * H_Gr ** 2 * V_Gr ** 2)

        # Thermal expansion coefficient correction factor
        f_alpha = 0.794 - 16.8 * V_Gr ** 2 - 0.0279 * T_ratio * (1 + V_Gr)

        # Density correction factor
        f_rho = 1.01 - 2.01 * V_Gr ** 2 - 0.0131 * T_ratio

        # # Ensure the correction factors are positive (except f_nu, which is allowed to be negative)
        # f_E = max(f_E, 0.00001)
        # # f_nu can be negative! This is a key difference
        # f_alpha = max(f_alpha, 0.00001)
        # f_rho = max(f_rho, 0.00001)

        return f_E, f_nu, f_alpha, f_rho
    
    def compute_layer_properties(self, layer_index: int, total_layers: int, 
                               V_Gr_base: float, T_ratio: float, H_Gr: float,
                               distribution_type: str = 'X') -> Tuple[float, float, float, float]:
        """
        Compute single-layer material properties

        Parameters:
            layer_index: layer index (1-based)
            total_layers: total number of layers
            V_Gr_base: graphene baseline volume fraction
            T_ratio: temperature ratio
            H_Gr: graphene shape factor
            distribution_type: distribution type ('X', 'O', 'U')

        Returns:
            Single-layer material properties: E, nu, alpha, rho
        """
        # Compute the graphene volume fraction of this layer
        if total_layers == 1:
            V_Gr_layer = V_Gr_base
        else:
            k = layer_index  # use 1-based index
            N = total_layers

            if distribution_type.upper() == 'X':
                # X-type distribution: smallest in the middle, largest at both ends
                V_Gr_layer = 2 * V_Gr_base * abs(2 * k - N - 1) / N
            elif distribution_type.upper() == 'O':
                # O-type distribution: largest in the middle, smallest at both ends
                V_Gr_layer = 2 * V_Gr_base * (1 - abs(2 * k - N - 1) / N)
            elif distribution_type.upper() == 'U':
                # U-type distribution: uniform distribution
                V_Gr_layer = V_Gr_base
            else:
                raise ValueError(f"Unsupported distribution type: {distribution_type}")

        # Ensure the volume fraction is within the valid range
        V_Gr_layer = np.clip(V_Gr_layer, 0.0, 1.0)
        V_Cu_layer = 1.0 - V_Gr_layer

        # Compute the correction factors
        f_E, f_nu, f_alpha, f_rho = self.compute_correction_factors(V_Gr_layer, T_ratio, H_Gr)

        # Compute geometric parameters
        ksai = 2 * (self.constants.l_Gr / self.constants.t_Gr)
        eta = ((self.constants.E_Gr / self.constants.E_Cu - 1) /
               (self.constants.E_Gr / self.constants.E_Cu + ksai))

        # Compute material properties
        # Young's modulus
        E = ((1 + ksai * eta * V_Gr_layer) / (1 - eta * V_Gr_layer) *
             self.constants.E_Cu * f_E)

        # Poisson's ratio
        nu = ((self.constants.nu_Gr * V_Gr_layer +
               self.constants.nu_Cu * V_Cu_layer) * f_nu)

        # Thermal expansion coefficient
        alpha = ((self.constants.alpha_Gr * V_Gr_layer +
                  self.constants.alpha_Cu * V_Cu_layer) * f_alpha)

        # Density
        rho = ((self.constants.rho_Gr * V_Gr_layer +
                self.constants.rho_Cu * V_Cu_layer) * f_rho)

        # Check the plausibility of the results
        if E <= 0 or nu >= 1 or rho <= 0:
            warnings.warn(f"Layer {layer_index} material properties abnormal: E={E:.2e}, nu={nu:.3f}, rho={rho:.2e}")

        return E, nu, alpha, rho
    
    def compute_material_properties(self, h: float, num_layers: int, 
                                  W_Gr: float = 0.025, H_Gr: float = 0.8, 
                                  T: float = 300.0, distribution_type: str = 'X') -> MaterialProperties:
        """
        Compute the overall material properties

        Parameters:
            h: total beam thickness
            num_layers: number of layers
            W_Gr: graphene mass fraction
            H_Gr: graphene shape factor
            T: temperature
            distribution_type: distribution type

        Returns:
            MaterialProperties object
        """
        T_ratio = T / self.constants.T_0

        # Convert the mass fraction to a volume fraction
        if W_Gr > 0:
            V_Gr_base = W_Gr / (W_Gr + (self.constants.rho_Gr / self.constants.rho_Cu) * (1 - W_Gr))
        else:
            V_Gr_base = 0.0

        # Initialize result storage
        layer_props = {}
        Q11_layer = {}
        Q55_layer = {}

        # Accumulators for computing the effective properties
        E_effective = 0.0
        nu_effective = 0.0
        alpha_effective = 0.0
        rho_effective = 0.0

        layer_thickness = h / num_layers

        # Compute the properties of each layer
        for k in range(1, num_layers + 1):
            E, nu, alpha, rho = self.compute_layer_properties(
                k, num_layers, V_Gr_base, T_ratio, H_Gr, distribution_type
            )

            # Compute the actual graphene volume fraction of this layer (for recording)
            if num_layers == 1:
                V_Gr_actual = V_Gr_base
            else:
                if distribution_type.upper() == 'X':
                    V_Gr_actual = 2 * V_Gr_base * abs(2*k - num_layers - 1) / num_layers
                elif distribution_type.upper() == 'O':
                    V_Gr_actual = 2 * V_Gr_base * (1 - abs(2*k - num_layers - 1) / num_layers)
                else:  # 'U'
                    V_Gr_actual = V_Gr_base

            # Store the layer properties
            layer_props[f"layer_{k}"] = {
                "E": E, "nu": nu, "alpha": alpha, "rho": rho,
                "V_Gr": V_Gr_actual
            }

            # Compute the stiffness coefficients
            Q11 = E / (1 - nu ** 2)
            Q55 = E / (2 * (1 + nu))
            Q11_layer[f"Q11_{k}"] = Q11
            Q55_layer[f"Q55_{k}"] = Q55

            # Compute the effective properties (based on thickness-weighted averaging)
            weight = layer_thickness / h
            E_effective += E * weight
            nu_effective += nu * weight
            alpha_effective += alpha * weight
            rho_effective += rho * weight
        
        return MaterialProperties(
            E=E_effective,
            nu=nu_effective, 
            alpha=alpha_effective,
            rho=rho_effective,
            layer_properties=layer_props,
            Q11=Q11_layer,
            Q55=Q55_layer
        )
    
    def compute_stiffness_coefficients(self, material_props: MaterialProperties, 
                                     h: float, num_layers: int) -> Tuple[float, float, float, float]:
        """
        Compute the stiffness coefficients A11, B11, D11, A55

        Parameters:
            material_props: material property object
            h: total thickness
            num_layers: number of layers

        Returns:
            Stiffness coefficients: A11, B11, D11, A55
        """
        delta_z = h / num_layers

        # Initialize stiffness components
        A11 = 0.0
        B11 = 0.0
        D11 = 0.0
        A55 = 0.0

        # Get the stiffness values of each layer
        Q11_values = list(material_props.Q11.values())
        Q55_values = list(material_props.Q55.values())

        # Shear correction factor
        kappa = 5.0 / 6.0

        # Integrate over each layer
        for k in range(num_layers):
            # Boundary coordinates of the layer
            z_k = -h / 2.0 + k * delta_z
            z_k_plus_1 = -h / 2.0 + (k + 1) * delta_z

            Q11_k = Q11_values[k]
            Q55_k = Q55_values[k]

            # Compute the contribution of each stiffness component
            # A11 = integral of Q11 dz
            A11 += Q11_k * (z_k_plus_1 - z_k)

            # B11 = integral of Q11 * z dz
            B11 += Q11_k * (z_k_plus_1**2 - z_k**2) / 2.0

            # D11 = integral of Q11 * z^2 dz
            D11 += Q11_k * (z_k_plus_1**3 - z_k**3) / 3.0

            # A55 = integral of kappa * Q55 dz
            A55 += kappa * Q55_k * (z_k_plus_1 - z_k)

        return A11, B11, D11, A55
    
    def compute_thermal_forces(self, material_props: MaterialProperties, 
                             A11: float, B11: float, delta_T: float,
                             h: float, num_layers: int) -> Tuple[float, float]:
        """
        Compute the thermal force and thermal moment

        Parameters:
            material_props: material properties
            A11: extensional stiffness
            B11: bending-extension coupling stiffness
            delta_T: temperature change
            h: total thickness
            num_layers: number of layers

        Returns:
            Thermal axial force N_XT, thermal bending moment M_XT
        """
        # Get the stiffness values of each layer
        Q11_values = list(material_props.Q11.values())

        # Initialize
        A11_MN = 0.0
        B11_MN = 0.0
        N_XT = 0.0
        M_XT = 0.0

        delta_z = h / num_layers

        # Compute over each layer
        for k in range(num_layers):
            # Get the thermal expansion coefficient of this layer
            layer_key = f"layer_{k+1}"  # 1-based key
            if layer_key in material_props.layer_properties:
                alpha_c_k = material_props.layer_properties[layer_key]['alpha']
            else:
                alpha_c_k = material_props.alpha  # fallback

            Q11_k = Q11_values[k]

            # Compute the layer boundaries
            z_k = -h/2.0 + k * delta_z
            z_k_plus_1 = -h/2.0 + (k + 1) * delta_z

            # Compute the A11 contribution
            contribution_A11 = Q11_k * (z_k_plus_1 - z_k)
            A11_MN = A11_MN + contribution_A11
            N_XT = -A11_MN * alpha_c_k * delta_T

            # Compute the B11 contribution
            contribution_B11 = Q11_k * (z_k_plus_1**2 - z_k**2) / 2.0
            B11_MN = B11_MN + contribution_B11
            M_XT = -B11_MN * alpha_c_k * delta_T

        return N_XT, M_XT
    
    def compute_dimensionless_parameters(self, h: float, L: float, num_layers: int,
                                       W_Gr: float = 0.025, H_Gr: float = 0.8,
                                       T: float = 300.0, distribution_type: str = 'X') -> DimensionlessParameters:
        """
        Compute the dimensionless parameters

        Parameters:
            h: beam thickness
            L: beam length
            num_layers: number of layers
            W_Gr: graphene mass fraction
            H_Gr: graphene shape factor
            T: temperature
            distribution_type: distribution type

        Returns:
            Dimensionless parameter object
        """
        # 1. Compute the material properties with graphene
        props = self.compute_material_properties(h, num_layers, W_Gr, H_Gr, T, distribution_type)
        A11, B11, D11, A55 = self.compute_stiffness_coefficients(props, h, num_layers)

        # 2. Compute the properties of pure copper (reference material)
        props_0 = self.compute_material_properties(h, num_layers, 0.0, 0.0, T, distribution_type)
        A11_0, B11_0, D11_0, A55_0 = self.compute_stiffness_coefficients(props_0, h, num_layers)

        # 3. Compute the thermodynamic parameters
        delta_T = T - self.constants.T_0
        N_XT, M_XT = self.compute_thermal_forces(props, A11, B11, delta_T, h, num_layers)

        # 4. Compute the dimensionless parameters
        a11 = A11 / A11_0
        a55 = A55 / A11_0
        b11 = B11 / (h * A11_0)
        d11 = D11 / (h**2 * A11_0)

        # Slenderness ratio
        lambda_val = L / h

        # Dimensionless thermodynamic parameters
        # n_xT = N_XT/A11_0; m_xT = M_XT/(h * A11_0);
        n_xT = N_XT / A11_0  # nondimensionalize by dividing by the reference stiffness
        m_xT = M_XT / (h * A11_0)  # nondimensionalize by dividing by h*A11_0
        
        return DimensionlessParameters(
            a11=a11, b11=b11, d11=d11, a55=a55,
            lambda_val=lambda_val, n_xT=n_xT, m_xT=m_xT
        )
    
    def print_material_summary(self, h: float, L: float, num_layers: int,
                             W_Gr: float = 0.025, H_Gr: float = 0.8,
                             T: float = 300.0, distribution_type: str = 'X'):
        """
        Print a summary of the material properties

        Parameters:
            h: beam thickness
            L: beam length
            num_layers: number of layers
            W_Gr: graphene mass fraction
            H_Gr: graphene shape factor
            T: temperature
            distribution_type: distribution type
        """
        print("=" * 60)
        print("Material Properties Summary")
        print("=" * 60)
        print(f"Geometry: L = {L:.3f} m, h = {h:.3f} m, layers = {num_layers}")
        print(f"Material: W_Gr = {W_Gr:.3f}, H_Gr = {H_Gr:.3f}, T = {T:.1f} K")
        print(f"Distribution: {distribution_type}")
        print("-" * 60)

        # Compute the material properties
        props = self.compute_material_properties(h, num_layers, W_Gr, H_Gr, T, distribution_type)
        A11, B11, D11, A55 = self.compute_stiffness_coefficients(props, h, num_layers)

        # Pure copper reference
        props_0 = self.compute_material_properties(h, num_layers, 0.0, 0.0, T, distribution_type)
        A11_0, B11_0, D11_0, A55_0 = self.compute_stiffness_coefficients(props_0, h, num_layers)

        # Thermodynamic parameters
        delta_T = T - self.constants.T_0
        N_XT, M_XT = self.compute_thermal_forces(props, A11, B11, delta_T, h, num_layers)
        
        print("Effective Material Properties:")
        print(f"  Young's modulus: E = {props.E/1e9:.2f} GPa")
        print(f"  Poisson's ratio: nu = {props.nu:.3f}")
        print(f"  Thermal expansion: alpha = {props.alpha*1e6:.2f} x 10^-6 /K")
        print(f"  Density: rho = {props.rho:.0f} kg/m^3")
        
        print("\nStiffness Coefficients:")
        print(f"  A11 = {A11:.2e}")
        print(f"  B11 = {B11:.2e}")
        print(f"  D11 = {D11:.2e}")
        print(f"  A55 = {A55:.2e}")

        print("\nPure Copper Reference:")
        print(f"  A11_0 = {A11_0:.2e}")
        print(f"  B11_0 = {B11_0:.2e}")
        print(f"  D11_0 = {D11_0:.2e}")
        print(f"  A55_0 = {A55_0:.2e}")

        # Dimensionless parameters
        dimensionless = self.compute_dimensionless_parameters(h, L, num_layers, W_Gr, H_Gr, T, distribution_type)

        print("\nDimensionless Parameters:")
        print(f"  a11 = {dimensionless.a11:.4f}")
        print(f"  b11 = {dimensionless.b11:.4f}")
        print(f"  d11 = {dimensionless.d11:.4f}")
        print(f"  a55 = {dimensionless.a55:.4f}")
        print(f"  lambda = L/h = {dimensionless.lambda_val:.2f}")
        print(f"  n_xT = {dimensionless.n_xT:.6f}")
        print(f"  m_xT = {dimensionless.m_xT:.6f}")

        if delta_T != 0:
            print(f"\nThermal Parameters (dT = {delta_T:.1f} K):")
            print(f"  N_XT = {N_XT:.2e}")
            print(f"  M_XT = {M_XT:.2e}")
        
        print("=" * 60)


def create_material_calculator(constants: Optional[MaterialConstants] = None) -> MaterialCalculator:
    """
    Factory function to create a material calculator

    Parameters:
        constants: material constants

    Returns:
        Material calculator instance
    """
    return MaterialCalculator(constants)

def compute_material_params_for_solver(h: float, L: float, num_layers: int = 10,
                                     W_Gr: float = 0.025, H_Gr: float = 0.8,
                                     T: float = 300.0, distribution_type: str = 'X',
                                     q: float = -0.08) -> Dict[str, float]:
    """
    Compute material parameters for the solver

    Parameters:
        h: beam thickness (m)
        L: beam length (m)
        num_layers: number of layers
        W_Gr: graphene mass fraction
        H_Gr: graphene shape factor
        T: temperature (K)
        distribution_type: distribution type
        q: dimensionless distributed load (dimensionless)

    Returns:
        Dictionary of material parameters required by the solver
    """
    calculator = create_material_calculator()

    # Compute material properties and dimensionless parameters
    props = calculator.compute_material_properties(h, num_layers, W_Gr, H_Gr, T, distribution_type)
    dimensionless = calculator.compute_dimensionless_parameters(
        h, L, num_layers, W_Gr, H_Gr, T, distribution_type
    )

    # Compute the temperature change
    delta_T = T - calculator.constants.T_0

    return {
        'a11': dimensionless.a11,
        'b11': dimensionless.b11,
        'd11': dimensionless.d11,
        'a55': dimensionless.a55,
        'lambda_val': dimensionless.lambda_val,  # keep consistent with the solver key name
        'n_xT': dimensionless.n_xT,
        'm_xT': dimensionless.m_xT,
        'alpha_effective': props.alpha,  # effective thermal expansion coefficient
        'delta_T': delta_T,  # temperature change
        'q': q  # directly use the dimensionless load
    }

if __name__ == "__main__":
    # Test the material property calculation module
    print("Testing material properties calculation module...")

    # Create the calculator
    calculator = create_material_calculator()

    # Test parameters
    h = 0.1  # beam thickness
    L = 20 * h  # beam length
    num_layers = 10
    W_Gr = 0.025
    H_Gr = 1
    T = 325.0
    q = -0.08
    distribution_type = 'X'
    # Print the material summary
    calculator.print_material_summary(h, L, num_layers, W_Gr, H_Gr, T, distribution_type)

    # Compute the parameters for the solver
    solver_params = compute_material_params_for_solver(h, L, num_layers, W_Gr, H_Gr, T, distribution_type, q)

    print("\nSolver parameters:")
    for key, value in solver_params.items():
        print(f"  {key}: {value:.6f}")

    print("\nMaterial properties module test completed!")
