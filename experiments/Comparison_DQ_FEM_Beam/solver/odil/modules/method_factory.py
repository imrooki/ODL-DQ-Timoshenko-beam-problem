

import torch
from typing import Any, Dict, Optional, Tuple
import warnings


def _cfg(config: Optional[Any], name: str, default=None):
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(name, default)
    return getattr(config, name, default)


torch.set_default_dtype(torch.float64)


class MethodFactory:
    

    @staticmethod
    def create_discretization(method: str, x: torch.Tensor, params: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        
        
        fallback_order = {
            'spline': ['taylor', 'dq'],
            'taylor': ['dq'],
            'dq': []
        }

        try:
            if method == 'dq':
                return MethodFactory._create_dq_matrices(x, params)

            elif method == 'taylor':
                return MethodFactory._create_taylor_matrices(x, params)

            elif method == 'spline':
                return MethodFactory._create_spline_matrices(x, params)

            else:
                raise ValueError(f"Unsupported discretization method: {method}")

        except Exception as e:
            warnings.warn(f"Method factory creation failed, falling back to the traditional DQ system: {e}")

            fallback_methods = fallback_order.get(method, ['dq'])

            for fallback_method in fallback_methods:
                try:
                    warnings.warn(f"Trying fallback method: {fallback_method}")

                    if fallback_method == 'dq':
                        return MethodFactory._create_dq_matrices(x, params)
                    elif fallback_method == 'taylor':
                        return MethodFactory._create_taylor_matrices(x, params)

                except Exception as fallback_error:
                    warnings.warn(f"Fallback method {fallback_method} also failed: {fallback_error}")
                    continue

            warnings.warn("All fallback methods failed, using the basic DQ method")
            try:
                from modules.dq_core import weighting_coefficients
                A, B, _, _ = weighting_coefficients(x)
                return A, B
            except Exception as final_error:
                raise RuntimeError(f"All discretization methods failed: primary method({method}): {e}, final error: {final_error}")

    @staticmethod
    def _create_dq_matrices(x: torch.Tensor, params: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        
        from modules.dq_core import get_cached_dq_system

        if hasattr(params, 'dq_type'):
            dq_type = params.dq_type
        elif hasattr(params, 'dq_method'):
            
            dq_type = params.dq_method
        else:
            dq_type = 'original'

        N = len(x)
        
        a = float(x[0].item())
        b = float(x[-1].item())

        dq_system = get_cached_dq_system(N, a, b, x.device, dq_type)
        A = dq_system['A']
        B = dq_system['B']
        return A, B

    @staticmethod
    def _create_taylor_matrices(x: torch.Tensor, params: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        
        from modules.taylor_core import TaylorFornbergCore

        stencil_size = _cfg(params, 'taylor_stencil_size', 9)
        sparse_format = _cfg(params, 'taylor_sparse_format', 'dense')  

        taylor_core = TaylorFornbergCore(
            x=x,
            stencil_size=stencil_size,
            sparse_format=sparse_format,
            device=x.device
        )

        A = taylor_core.build_derivative_matrix(order=1)
        B = taylor_core.build_derivative_matrix(order=2)

        return A, B

    @staticmethod
    def _create_spline_matrices(x: torch.Tensor, params: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        
        from modules.spline_core import SplineCore

        spline_type = _cfg(params, 'spline_type', 'cubic')
        spline_boundary = _cfg(params, 'spline_boundary', 'natural')
        bspline_degree = _cfg(params, 'spline_degree', None)
        tension = _cfg(params, 'spline_tension', 0.0)
        derivative_values = _cfg(params, 'spline_derivative_values', None)

        core = SplineCore(
            x=x,
            spline_type=spline_type,
            bc_type=spline_boundary,
            device=x.device,
            bspline_degree=bspline_degree,
            tension=tension,
            derivative_values=derivative_values,
        )

        try:
            return core.compute_derivative_matrices()
        except Exception as exc:
            warnings.warn(f"Spline matrix construction failed ({exc}); falling back to Taylor method.")

            from modules.taylor_core import TaylorFornbergCore

            taylor_core = TaylorFornbergCore(x, stencil_size=7, device=x.device)
            A = taylor_core.build_derivative_matrix(order=1)
            B = taylor_core.build_derivative_matrix(order=2)

            return A, B

    @staticmethod
    def get_method_info(method: str, params: Any) -> Dict[str, Any]:
        
        info = {'method': method}

        if method == 'dq':
            info['dq_type'] = _cfg(params, 'dq_type', 'original')
            info['nodes'] = _cfg(params, 'N', 11)

        elif method == 'taylor':
            info['nodes'] = _cfg(params, 'N', 13)
            info['stencil_size'] = _cfg(params, 'taylor_stencil_size', 9)
            info['node_type'] = _cfg(params, 'taylor_nodes', 'cheb')

        elif method == 'spline':
            info['nodes'] = _cfg(params, 'N', 21)
            info['spline_type'] = _cfg(params, 'spline_type', 'cubic')
            info['boundary_type'] = _cfg(params, 'spline_boundary', 'natural')
            info['spline_degree'] = _cfg(params, 'spline_degree', None)
            info['tension'] = _cfg(params, 'spline_tension', 0.0)
            spline_type_lower = str(info['spline_type']).lower()
            if spline_type_lower == 'quintic':
                info['spline_degree'] = 5
            elif spline_type_lower in {'b-spline', 'bspline', 'b_spline'} and info['spline_degree'] is None:
                info['spline_degree'] = 3

        return info


def print_method_info(method: str, params: Any):
    
    info = MethodFactory.get_method_info(method, params)

    print(f"  Discretization method: {method.upper()}")

    if method == 'dq':
        print(f"    - DQ type: {info['dq_type']}")
        print(f"    - Number of nodes: {info['nodes']}")

    elif method == 'taylor':
        print(f"    - Number of nodes: {info['nodes']}")
        print(f"    - Stencil size: {info['stencil_size']}")
        print(f"    - Node type: {info['node_type']}")

    elif method == 'spline':
        print(f"    - Number of nodes: {info['nodes']}")
        print(f"    - Spline type: {info['spline_type'].upper()}")
        print(f"    - Boundary condition: {info['boundary_type'].upper()}")
        spline_type_lower = info.get('spline_type', '').lower()
        if spline_type_lower in {'b-spline', 'bspline', 'b_spline', 'quintic'}:
            degree = info.get('spline_degree', None)
            if degree is not None:
                print(f"    - B-spline degree: {degree}")
        if spline_type_lower in {'tension', 'hermite'}:
            tension = info.get('tension', 0.0)
            print(f"    - Tension parameter: {tension}")
