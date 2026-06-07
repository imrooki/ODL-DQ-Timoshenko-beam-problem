"""
Graphene-reinforced functionally graded material (GPL-FGM) property calculation module

This module implements property calculations for graphene platelet (GPL) reinforced copper-matrix composites.

Core functionality:
1. Material constant definitions: basic properties of copper (Cu) and graphene (Graphene)
2. Correction factor computation: based on semi-empirical formulas accounting for temperature and shape factor effects
3. Layer-wise property computation: using a modified rule of mixtures to compute the equivalent properties of each layer
4. Stiffness coefficient computation: computing A11, B11, D11, A55 via laminate theory
5. Non-dimensionalization: non-dimensionalized with respect to pure copper properties as reference
6. Thermodynamic parameters: computing the axial force N_xT and bending moment M_xT induced by thermal loading

Material distribution types:
- X-type: more GPL at the two ends and less in the middle, enhancing bending resistance
- U-type: GPL uniformly distributed, identical properties across layers
- O-type: more GPL in the middle and less at the two ends, enhancing shear resistance

Based on Timoshenko beam theory (FSDT), adapted to the ODIL solver framework
"""

from typing import Tuple, Dict, NamedTuple, Optional
from dataclasses import dataclass
import numpy as np
import warnings

# Compatible with different SciPy versions - this module mainly uses numerical computation, no complex integration needed

@dataclass
class MaterialConstants:
    """Basic material constants class

    Defines the basic properties of the matrix material (copper) and the reinforcement phase (graphene).
    All values are based on experimental data and literature reports.
    """
    # Reference temperature (room temperature, used to compute temperature change)
    T_0: float = 300.0  # K

    # Material properties of copper (Cu) - matrix material
    E_Cu: float = 65.79e9    # Young's modulus (Pa) - describes the extensional stiffness of the material
    nu_Cu: float = 0.387     # Poisson's ratio - ratio of transverse strain to longitudinal strain
    rho_Cu: float = 8.80e3   # density (kg/m³) - affects inertial force
    alpha_Cu: float = 16.51e-6  # thermal expansion coefficient (1/K) - strain induced by temperature change

    # Material properties of graphene (Graphene) - reinforcement phase
    E_Gr: float = 929.57e9   # Young's modulus (Pa) - far higher than copper, provides reinforcement effect
    nu_Gr: float = 0.220     # Poisson's ratio - lower than copper
    rho_Gr: float = 1.80e3   # density (kg/m³) - far lower than copper, favorable for lightweighting
    alpha_Gr: float = -3.98e-6  # thermal expansion coefficient (1/K) - negative value, contracts when heated and expands when cooled

    # GPL (graphene platelet) geometric parameters - used for the Halpin-Tsai model
    l_Gr: float = 83.76  # GPL length (dimensionless) - affects aspect ratio
    t_Gr: float = 3.4    # GPL thickness (dimensionless) - affects reinforcement efficiency

class MaterialProperties(NamedTuple):
    """Material properties result class"""
    E: float        # effective Young's modulus
    nu: float       # effective Poisson's ratio
    alpha: float    # effective thermal expansion coefficient
    rho: float      # effective density
    layer_properties: Dict  # detailed properties of each layer
    Q11: Dict       # Q11 stiffness coefficient of each layer
    Q55: Dict       # Q55 stiffness coefficient of each layer

class DimensionlessParameters(NamedTuple):
    """Dimensionless parameters result class"""
    a11: float      # dimensionless extensional stiffness
    b11: float      # dimensionless bending-extension coupling stiffness
    d11: float      # dimensionless bending stiffness
    a55: float      # dimensionless shear stiffness
    lambda_val: float  # length-to-height ratio
    n_xT: float     # dimensionless axial thermal force
    m_xT: float     # dimensionless thermal bending moment

class MaterialCalculator:
    """Material property calculator"""

    def __init__(self, constants: Optional[MaterialConstants] = None):
        """
        Initialize the material calculator

        Parameters:
            constants: material constants; uses default values if None
        """
        self.constants = constants if constants is not None else MaterialConstants()
    
    def compute_correction_factors(self, V_Gr: float, T_ratio: float, H_Gr: float) -> Tuple[float, float, float, float]:
        """
        Compute the correction factors for the material properties

        Based on semi-empirical formulas, accounting for the effects of temperature and GPL shape on the composite properties.
        These formulas are derived from fitting experimental data and molecular dynamics simulations.

        Parameters:
            V_Gr: GPL volume fraction (0-1)
            T_ratio: temperature ratio T/T₀ (used to account for temperature effects)
            H_Gr: GPL shape factor (describes the aspect ratio and orientation of GPL)

        Returns:
            f_E: Young's modulus correction factor
            f_nu: Poisson's ratio correction factor
            f_alpha: thermal expansion coefficient correction factor
            f_rho: density correction factor
        """
        if V_Gr == 0:
            # Pure copper case
            return 1.0, 1.0, 1.0, 1.0

        # Ensure parameters are within a reasonable range
        V_Gr = np.clip(V_Gr, 0.0, 1.0)
        H_Gr = np.clip(H_Gr, 0.0, 10.0)
        T_ratio = np.clip(T_ratio, 0.5, 3.0)

        # Young's modulus correction factor - accounts for the coupled effects of V_Gr, T and H_Gr
        f_E = (1.11 - 1.22 * V_Gr - 0.134 * T_ratio + 0.559 * V_Gr * T_ratio
               - 5.5 * H_Gr * V_Gr + 38 * H_Gr * V_Gr ** 2 - 20.6 * H_Gr ** 2 * V_Gr ** 2)

        # Poisson's ratio correction factor - may be negative, which is physically reasonable
        f_nu = (1.01 - 1.43 * V_Gr + 0.165 * T_ratio - 16.8 * H_Gr * V_Gr
                - 1.1 * H_Gr * V_Gr * T_ratio + 16 * H_Gr ** 2 * V_Gr ** 2)

        # Thermal expansion coefficient correction factor - reflects the negative thermal expansion characteristic of GPL
        f_alpha = 0.794 - 16.8 * V_Gr ** 2 - 0.0279 * T_ratio * (1 + V_Gr)

        # Density correction factor - usually close to 1, because the GPL density is far lower than copper
        f_rho = 1.01 - 2.01 * V_Gr ** 2 - 0.0131 * T_ratio

        # # Ensure correction factors are positive (except f_nu, which is allowed to be negative)
        # f_E = max(f_E, 0.00001)
        # # f_nu can be negative! This is the key difference
        # f_alpha = max(f_alpha, 0.00001)
        # f_rho = max(f_rho, 0.00001)
        
        return f_E, f_nu, f_alpha, f_rho
    
    def compute_layer_properties(self, layer_index: int, total_layers: int,
                               V_Gr_base: float, T_ratio: float, H_Gr: float,
                               distribution_type: str = 'X') -> Tuple[float, float, float, float]:
        """
        Compute the material properties of layer number layer_index

        Adjusts the GPL volume fraction of each layer according to the distribution type, realizing a functionally graded distribution.
        Uses a modified rule of mixtures and geometric parameters to compute the equivalent material properties.

        Parameters:
            layer_index: layer number (starting from 1, from bottom to top)
            total_layers: total number of layers (usually 10)
            V_Gr_base: average GPL volume fraction
            T_ratio: temperature ratio T/T₀
            H_Gr: GPL shape factor
            distribution_type: distribution type
                'X': GPL-rich at the two ends (bending-resistance optimized)
                'O': GPL-rich in the middle (shear-resistance optimized)
                'U': uniform distribution

        Returns:
            equivalent properties of the layer: E, nu, alpha, rho
        """
        # Compute the graphene volume fraction of this layer
        if total_layers == 1:
            V_Gr_layer = V_Gr_base
        else:
            k = layer_index  # use 1-based index
            N = total_layers

            if distribution_type.upper() == 'X':
                # X-type distribution: GPL-rich at the two ends, GPL-poor near the neutral plane
                # Favorable for increasing the bending stiffness D11
                V_Gr_layer = 2 * V_Gr_base * abs(2 * k - N - 1) / N
            elif distribution_type.upper() == 'O':
                # O-type distribution: GPL-rich near the neutral plane, GPL-poor at the two ends
                # Favorable for increasing the shear stiffness A55
                V_Gr_layer = 2 * V_Gr_base * (1 - abs(2 * k - N - 1) / N)
            elif distribution_type.upper() == 'U':
                # U-type distribution: identical GPL content in each layer
                # Serves as the reference baseline
                V_Gr_layer = V_Gr_base
            else:
                raise ValueError(f"Unsupported distribution type: {distribution_type}")

        # Ensure the volume fraction is within the valid range
        V_Gr_layer = np.clip(V_Gr_layer, 0.0, 1.0)
        V_Cu_layer = 1.0 - V_Gr_layer

        # Compute correction factors
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

        # Check the reasonableness of the result
        if E <= 0 or nu >= 1 or rho <= 0:
            warnings.warn(f"Abnormal material properties in layer {layer_index}: E={E:.2e}, nu={nu:.3f}, rho={rho:.2e}")
        
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

        # Convert mass fraction to volume fraction
        if W_Gr > 0:
            V_Gr_base = W_Gr / (W_Gr + (self.constants.rho_Gr / self.constants.rho_Cu) * (1 - W_Gr))
        else:
            V_Gr_base = 0.0

        # Initialize result storage
        layer_props = {}
        Q11_layer = {}
        Q55_layer = {}

        # Accumulators used to compute effective properties
        E_effective = 0.0
        nu_effective = 0.0
        alpha_effective = 0.0
        rho_effective = 0.0

        layer_thickness = h / num_layers

        # Compute properties of each layer
        for k in range(1, int(num_layers) + 1):
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

            # Store layer properties
            layer_props[f"layer_{k}"] = {
                "E": E, "nu": nu, "alpha": alpha, "rho": rho,
                "V_Gr": V_Gr_actual
            }

            # Compute stiffness coefficients
            Q11 = E / (1 - nu ** 2)
            Q55 = E / (2 * (1 + nu))
            Q11_layer[f"Q11_{k}"] = Q11
            Q55_layer[f"Q55_{k}"] = Q55

            # Compute effective properties (based on thickness-weighted average)
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
            material_props: material properties object
            h: total thickness
            num_layers: number of layers

        Returns:
            stiffness coefficients: A11, B11, D11, A55
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
        for k in range(int(num_layers)):
            # Boundary coordinates of the layer
            z_k = -h / 2.0 + k * delta_z
            z_k_plus_1 = -h / 2.0 + (k + 1) * delta_z

            Q11_k = Q11_values[k]
            Q55_k = Q55_values[k]

            # Compute the contribution of each stiffness component
            # A11 = ∫ Q11 dz
            A11 += Q11_k * (z_k_plus_1 - z_k)

            # B11 = ∫ Q11 * z dz
            B11 += Q11_k * (z_k_plus_1**2 - z_k**2) / 2.0

            # D11 = ∫ Q11 * z² dz
            D11 += Q11_k * (z_k_plus_1**3 - z_k**3) / 3.0

            # A55 = ∫ κ * Q55 dz
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
            thermal axial force N_XT, thermal bending moment M_XT
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
        for k in range(int(num_layers)):
            # Get the thermal expansion coefficient of this layer
            layer_key = f"layer_{k+1}"  # 1-based key
            if layer_key in material_props.layer_properties:
                alpha_c_k = material_props.layer_properties[layer_key]['alpha']
            else:
                alpha_c_k = material_props.alpha  # fallback

            Q11_k = Q11_values[k]

            # Compute layer boundaries
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
            dimensionless parameters object
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

        # Length-to-height ratio
        lambda_val = L / h

        # Dimensionless thermodynamic parameters
        # n_xT = N_XT/A11_0; m_xT = M_XT/(h * A11_0);
        n_xT = N_XT / A11_0  # divided by the reference stiffness for non-dimensionalization
        m_xT = M_XT / (h * A11_0)  # divided by h*A11_0 for non-dimensionalization
        
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
        print("Material property calculation summary")
        print("=" * 60)
        print(f"Geometric parameters: L = {L:.3f} m, h = {h:.3f} m, number of layers = {num_layers}")
        print(f"Material parameters: W_Gr = {W_Gr:.3f}, H_Gr = {H_Gr:.3f}, T = {T:.1f} K")
        print(f"Distribution type: {distribution_type}")
        print("-" * 60)

        # Compute material properties
        props = self.compute_material_properties(h, num_layers, W_Gr, H_Gr, T, distribution_type)
        A11, B11, D11, A55 = self.compute_stiffness_coefficients(props, h, num_layers)

        # Pure copper reference
        props_0 = self.compute_material_properties(h, num_layers, 0.0, 0.0, T, distribution_type)
        A11_0, B11_0, D11_0, A55_0 = self.compute_stiffness_coefficients(props_0, h, num_layers)

        # Thermodynamic parameters
        delta_T = T - self.constants.T_0
        N_XT, M_XT = self.compute_thermal_forces(props, A11, B11, delta_T, h, num_layers)

        print("Effective material properties:")
        print(f"  Young's modulus: E = {props.E/1e9:.2f} GPa")
        print(f"  Poisson's ratio: ν = {props.nu:.3f}")
        print(f"  Thermal expansion coefficient: α = {props.alpha*1e6:.2f} × 10⁻⁶ /K")
        print(f"  Density: ρ = {props.rho:.0f} kg/m³")

        print("\nStiffness coefficients:")
        print(f"  A11 = {A11:.2e}")
        print(f"  B11 = {B11:.2e}")
        print(f"  D11 = {D11:.2e}")
        print(f"  A55 = {A55:.2e}")

        print("\nPure copper reference values:")
        print(f"  A11_0 = {A11_0:.2e}")
        print(f"  B11_0 = {B11_0:.2e}")
        print(f"  D11_0 = {D11_0:.2e}")
        print(f"  A55_0 = {A55_0:.2e}")
        
        # Dimensionless parameters
        dimensionless = self.compute_dimensionless_parameters(h, L, num_layers, W_Gr, H_Gr, T, distribution_type)

        print("\nDimensionless parameters:")
        print(f"  a11 = {dimensionless.a11:.4f}")
        print(f"  b11 = {dimensionless.b11:.4f}")
        print(f"  d11 = {dimensionless.d11:.4f}")
        print(f"  a55 = {dimensionless.a55:.4f}")
        print(f"  λ = L/h = {dimensionless.lambda_val:.2f}")
        print(f"  n_xT = {dimensionless.n_xT:.6f}")
        print(f"  m_xT = {dimensionless.m_xT:.6f}")

        if delta_T != 0:
            print(f"\nThermodynamic parameters (ΔT = {delta_T:.1f} K):")
            print(f"  N_XT = {N_XT:.2e}")
            print(f"  M_XT = {M_XT:.2e}")
        
        print("=" * 60)


def create_material_calculator(constants: Optional[MaterialConstants] = None) -> MaterialCalculator:
    """
    Factory function to create a material calculator

    Parameters:
        constants: material constants

    Returns:
        material calculator instance
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
        dictionary of material parameters required by the solver
    """
    calculator = create_material_calculator()
    dimensionless = calculator.compute_dimensionless_parameters(
        h, L, num_layers, W_Gr, H_Gr, T, distribution_type
    )
    
    return {
        'a11': dimensionless.a11,
        'b11': dimensionless.b11,
        'd11': dimensionless.d11,
        'a55': dimensionless.a55,
        'lambda_val': dimensionless.lambda_val,  # consistent with the solver's key name
        'n_xT': dimensionless.n_xT,
        'm_xT': dimensionless.m_xT,
        'q': q  # directly use the dimensionless load
    }

if __name__ == "__main__":
    # Test the material property calculation module
    print("Testing the material property calculation module...")

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

    # Compute parameters for the solver
    solver_params = compute_material_params_for_solver(h, L, num_layers, W_Gr, H_Gr, T, distribution_type, q)

    print("\nSolver parameters:")
    for key, value in solver_params.items():
        print(f"  {key}: {value:.6f}")

    print("\nMaterial property calculation module test complete!")
