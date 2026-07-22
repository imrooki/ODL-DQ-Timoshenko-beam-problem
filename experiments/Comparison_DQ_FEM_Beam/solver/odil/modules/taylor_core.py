

import torch
import warnings
from typing import Tuple, Optional, Dict, List, Union


torch.set_default_dtype(torch.float64)

class TaylorFornbergCore:
    

    def __init__(self, x: torch.Tensor, stencil_size: int = 5,
                 sparse_format: str = 'coo', device: Optional[torch.device] = None,
                 enable_gpu_acceleration: bool = True):
        
        self.x = x.to(device) if device else x
        self.N = len(x)
        self.stencil_size = stencil_size
        self.sparse_format = sparse_format.lower()
        self.device = self.x.device
        self.enable_gpu_acceleration = enable_gpu_acceleration and self.device.type == 'cuda'

        
        assert stencil_size % 2 == 1, f"Stencil size must be an odd number, got {stencil_size}"
        assert stencil_size <= self.N, f"Stencil size {stencil_size} cannot exceed the number of nodes {self.N}"
        assert sparse_format in ['coo', 'csr', 'dense'], f"Unsupported sparse format: {sparse_format}"

        print(f"Taylor/Fornberg core initialization:")
        print(f"  - Number of nodes: {self.N}")
        print(f"  - Stencil size: {self.stencil_size}")
        print(f"  - Storage format: {self.sparse_format}")
        print(f"  - Device: {self.device}")

    def fd_weights_fornberg_optimized(self, x0: torch.Tensor, x_local: torch.Tensor,
                                    m: int) -> torch.Tensor:
        
        n = len(x_local)
        device = x_local.device
        dtype = x_local.dtype

        
        if not isinstance(x0, torch.Tensor):
            x0 = torch.tensor(x0, device=device, dtype=dtype)
        elif x0.dim() > 0:
            x0 = x0.item()
            x0 = torch.tensor(x0, device=device, dtype=dtype)

        
        w = torch.zeros((n, m+1), device=device, dtype=dtype)
        w[0, 0] = 1.0

        c1 = torch.tensor(1.0, device=device, dtype=dtype)
        c4 = x_local[0] - x0

        
        for i in range(1, n):
            mn = min(i, m)
            c2 = torch.tensor(1.0, device=device, dtype=dtype)
            c5 = c4
            c4 = x_local[i] - x0

            for j in range(i):
                c3 = x_local[i] - x_local[j]

                
                if torch.abs(c3) < 1e-15:
                    warnings.warn(f"Nearly duplicate nodes found in the Fornberg algorithm: {x_local[i]:.1e}, {x_local[j]:.1e}")
                    c3 = torch.sign(c3) * 1e-15

                c2 = c2 * c3

                if j == i - 1:
                    
                    for k in range(mn, 0, -1):
                        w[i, k] = c1 * (k * w[i-1, k-1] - c5 * w[i-1, k]) / c2
                    w[i, 0] = -c1 * c5 * w[i-1, 0] / c2

                
                for k in range(mn, 0, -1):
                    w[j, k] = (c4 * w[j, k] - k * w[j, k-1]) / c3
                w[j, 0] = c4 * w[j, 0] / c3

            c1 = c2

        return w[:, m]

    def select_local_stencil(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        
        h = (self.stencil_size - 1) // 2

        if i < h:
            
            indices = torch.arange(0, self.stencil_size, device=self.device, dtype=torch.long)
        elif i > self.N - 1 - h:
            
            indices = torch.arange(self.N - self.stencil_size, self.N,
                                 device=self.device, dtype=torch.long)
        else:
            
            indices = torch.arange(i - h, i + h + 1, device=self.device, dtype=torch.long)

        x_local = self.x[indices]
        return indices, x_local

    def build_sparse_derivative_matrix_gpu_accelerated(self, order: int) -> Union[torch.Tensor, torch.sparse.FloatTensor]:
        
        if not self.enable_gpu_acceleration or not torch.cuda.is_available():
            return self.build_sparse_derivative_matrix(order)

        
        batch_size = min(256, self.N)  
        row_indices = []
        col_indices = []
        values = []

        for batch_start in range(0, self.N, batch_size):
            batch_end = min(batch_start + batch_size, self.N)
            batch_rows, batch_cols, batch_vals = self._compute_fornberg_weights_batch(
                batch_start, batch_end, order)

            row_indices.extend(batch_rows)
            col_indices.extend(batch_cols)
            values.extend(batch_vals)

        
        row_indices = torch.tensor(row_indices, device=self.device, dtype=torch.long)
        col_indices = torch.tensor(col_indices, device=self.device, dtype=torch.long)
        values = torch.tensor(values, device=self.device, dtype=torch.float64)

        
        if self.sparse_format == 'coo':
            indices = torch.stack([row_indices, col_indices])
            return torch.sparse_coo_tensor(indices, values, (self.N, self.N), device=self.device)
        elif self.sparse_format == 'csr':
            indices = torch.stack([row_indices, col_indices])
            sparse_coo = torch.sparse_coo_tensor(indices, values, (self.N, self.N), device=self.device)
            return sparse_coo.to_sparse_csr()
        else:  
            matrix = torch.zeros((self.N, self.N), device=self.device, dtype=torch.float64)
            matrix[row_indices, col_indices] = values
            return matrix

    def _compute_fornberg_weights_batch(self, batch_start: int, batch_end: int,
                                      order: int) -> Tuple[List[int], List[int], List[float]]:
        
        batch_rows = []
        batch_cols = []
        batch_vals = []

        for i in range(batch_start, batch_end):
            
            indices, x_local = self.select_local_stencil(i)

            weights = self.fd_weights_fornberg_optimized(self.x[i], x_local, order)

            
            for j, (col_idx, weight) in enumerate(zip(indices, weights)):
                if torch.abs(weight) > 1e-15:
                    batch_rows.append(i)
                    batch_cols.append(col_idx.item())
                    batch_vals.append(weight.item())

        return batch_rows, batch_cols, batch_vals

    def build_sparse_derivative_matrix(self, order: int) -> Union[torch.Tensor, torch.sparse.FloatTensor]:
        
        
        row_indices = []
        col_indices = []
        values = []

        for i in range(self.N):
            
            indices, x_local = self.select_local_stencil(i)

            
            weights = self.fd_weights_fornberg_optimized(self.x[i], x_local, order)

            
            for j, (col_idx, weight) in enumerate(zip(indices, weights)):
                if torch.abs(weight) > 1e-15:  
                    row_indices.append(i)
                    col_indices.append(col_idx.item())
                    values.append(weight.item())

        
        row_indices = torch.tensor(row_indices, device=self.device, dtype=torch.long)
        col_indices = torch.tensor(col_indices, device=self.device, dtype=torch.long)
        values = torch.tensor(values, device=self.device, dtype=torch.float64)

        
        if self.sparse_format == 'coo':
            indices = torch.stack([row_indices, col_indices])
            return torch.sparse_coo_tensor(indices, values, (self.N, self.N), device=self.device)
        elif self.sparse_format == 'csr':
            indices = torch.stack([row_indices, col_indices])
            sparse_coo = torch.sparse_coo_tensor(indices, values, (self.N, self.N), device=self.device)
            return sparse_coo.to_sparse_csr()
        else:  
            matrix = torch.zeros((self.N, self.N), device=self.device, dtype=torch.float64)
            matrix[row_indices, col_indices] = values
            return matrix

    def build_derivative_matrix(self, order: int) -> Union[torch.Tensor, torch.sparse.FloatTensor]:
        
        if self.enable_gpu_acceleration:
            return self.build_sparse_derivative_matrix_gpu_accelerated(order)
        else:
            return self.build_sparse_derivative_matrix(order)

    def compute_derivative_matrices(self, orders: Tuple[int, ...] = (1, 2)) -> Dict[str, torch.Tensor]:
        
        print(f"Constructing order-{orders} derivative matrices...")

        matrices = {}
        matrix_names = ['', 'A', 'B', 'C', 'D', 'E', 'F']

        for order in orders:
            if order < len(matrix_names):
                name = matrix_names[order]
                print(f"  Constructing order-{order} derivative matrix ({name})...")
                matrices[name] = self.build_sparse_derivative_matrix(order)
            else:
                matrices[f'D{order}'] = self.build_sparse_derivative_matrix(order)

        return matrices

    def analyze_sparsity(self, matrix: torch.Tensor) -> Dict[str, float]:
        
        if matrix.is_sparse:
            total_elements = matrix.shape[0] * matrix.shape[1]
            nonzero_elements = matrix._nnz()
            sparsity = 1.0 - nonzero_elements / total_elements
        else:
            total_elements = matrix.numel()
            nonzero_elements = torch.count_nonzero(matrix).item()
            sparsity = 1.0 - nonzero_elements / total_elements

        return {
            'sparsity_ratio': sparsity,
            'nonzero_elements': nonzero_elements,
            'total_elements': total_elements,
            'memory_reduction': sparsity
        }

    def matrix_vector_multiply(self, matrix: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
        
        if matrix.is_sparse:
            return torch.sparse.mm(matrix, vector.unsqueeze(-1)).squeeze(-1)
        else:
            return matrix @ vector

    def benchmark_performance(self, matrix: torch.Tensor, n_tests: int = 1000) -> Dict[str, float]:
        
        test_vector = torch.randn(self.N, device=self.device, dtype=torch.float64)

        
        for _ in range(10):
            _ = self.matrix_vector_multiply(matrix, test_vector)

        
        torch.cuda.synchronize() if self.device.type == 'cuda' else None
        start_time = torch.cuda.Event(enable_timing=True) if self.device.type == 'cuda' else None
        end_time = torch.cuda.Event(enable_timing=True) if self.device.type == 'cuda' else None

        if self.device.type == 'cuda':
            start_time.record()
            for _ in range(n_tests):
                _ = self.matrix_vector_multiply(matrix, test_vector)
            end_time.record()
            torch.cuda.synchronize()
            elapsed_time = start_time.elapsed_time(end_time) / 1000.0  
        else:
            import time
            start = time.time()
            for _ in range(n_tests):
                _ = self.matrix_vector_multiply(matrix, test_vector)
            elapsed_time = time.time() - start

        return {
            'total_time': elapsed_time,
            'average_time': elapsed_time / n_tests,
            'operations_per_second': n_tests / elapsed_time
        }


def create_taylor_fornberg_system(x: torch.Tensor, stencil_size: int = 5,
                                orders: Tuple[int, ...] = (1, 2),
                                sparse_format: str = 'dense',
                                device: Optional[torch.device] = None) -> Dict:
    
    
    core = TaylorFornbergCore(x, stencil_size, sparse_format, device)

    
    matrices = core.compute_derivative_matrices(orders)

    
    sparsity_analysis = {}
    performance_analysis = {}

    for name, matrix in matrices.items():
        if matrix is not None:
            sparsity_analysis[name] = core.analyze_sparsity(matrix)
            
            performance_analysis[name] = core.benchmark_performance(matrix, n_tests=100)

    
    system = {
        'x': x,
        'matrices': matrices,
        'core': core,
        'sparsity_analysis': sparsity_analysis,
        'performance_analysis': performance_analysis,
        'system_info': {
            'N': core.N,
            'stencil_size': stencil_size,
            'sparse_format': sparse_format,
            'device': str(device or x.device),
            'orders': orders
        }
    }

    return system


def validate_taylor_accuracy(x: torch.Tensor, system: Dict, test_function: str = 'sin') -> Dict:
    
    matrices = system['matrices']

    
    if test_function == 'sin':
        f = torch.sin(torch.pi * x)
        f1_exact = torch.pi * torch.cos(torch.pi * x)
        f2_exact = -torch.pi**2 * torch.sin(torch.pi * x)
    elif test_function == 'polynomial':
        
        f = x**4 - 2*x**3 + 3*x**2 - x + 1
        f1_exact = 4*x**3 - 6*x**2 + 6*x - 1
        f2_exact = 12*x**2 - 12*x + 6
    else:
        raise ValueError(f"Unsupported test function: {test_function}")

    
    core = system['core']
    errors = {}

    if 'A' in matrices:
        f1_numerical = core.matrix_vector_multiply(matrices['A'], f)
        error_1 = torch.max(torch.abs(f1_numerical - f1_exact)).item()
        l2_error_1 = torch.norm(f1_numerical - f1_exact).item() / torch.norm(f1_exact).item()
        errors['first_derivative'] = {'max_error': error_1, 'l2_relative_error': l2_error_1}

    if 'B' in matrices:
        f2_numerical = core.matrix_vector_multiply(matrices['B'], f)
        error_2 = torch.max(torch.abs(f2_numerical - f2_exact)).item()
        l2_error_2 = torch.norm(f2_numerical - f2_exact).item() / torch.norm(f2_exact).item()
        errors['second_derivative'] = {'max_error': error_2, 'l2_relative_error': l2_error_2}

    return {
        'test_function': test_function,
        'errors': errors,
        'validation_passed': all(
            err['max_error'] < 1e-8 for err in errors.values()
        )
    }


if __name__ == "__main__":
    
    print("Taylor/Fornberg core module demonstration")
    print("-" * 40)

    
    from modules.dq_core import cheb_lobatto_nodes

    N = 11
    x = cheb_lobatto_nodes(N, 0.0, 1.0)
    print(f"Number of test nodes: {N}")

    
    system = create_taylor_fornberg_system(
        x, stencil_size=5, orders=(1, 2), sparse_format='dense'
    )

    
    validation = validate_taylor_accuracy(x, system, 'sin')

    print("\nAccuracy validation result:")
    for deriv, error_info in validation['errors'].items():
        print(f"  {deriv}: max error={error_info['max_error']:.2e}, L2 relative error={error_info['l2_relative_error']:.2e}")

    print(f"\nValidation passed: {validation['validation_passed']}")

    
    print("\nSparsity analysis:")
    for name, analysis in system['sparsity_analysis'].items():
        print(f"  {name} matrix: sparsity={analysis['sparsity_ratio']:.1%}, memory savings={analysis['memory_reduction']:.1%}")