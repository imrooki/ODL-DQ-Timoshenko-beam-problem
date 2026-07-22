

import torch
from typing import Dict, Tuple


class TimoshenkoBeamResiduals:
    
    
    def __init__(self, material_params: Dict, q: float = 0.0, k1: float = 0.0, k2: float = 0.0):
        
        self.a11 = material_params['a11']
        self.b11 = material_params['b11']
        self.d11 = material_params['d11']
        self.a55 = material_params['a55']
        self.lambda_val = material_params['lambda_val']
        self.n_xT = material_params['n_xT']
        self.m_xT = material_params.get('m_xT', 0.0)
        self.q = q
        self.k1 = k1  
        self.k2 = k2  
    
    def compute_linear(self, u: torch.Tensor, w: torch.Tensor, phi: torch.Tensor,
                      A: torch.Tensor, B: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        
        ux = A @ u      
        wx = A @ w      
        phix = A @ phi  

        uxx = B @ u     
        wxx = B @ w     
        phixx = B @ phi 

        
        interior_mode = uxx.shape[0] != u.shape[0]

        
        if interior_mode:
            field_slice = slice(1, -1)
            u_field = u[field_slice]
            w_field = w[field_slice]
            phi_field = phi[field_slice]
        else:
            u_field = u
            w_field = w
            phi_field = phi

        

        
        R1 = self.a11 * uxx + self.b11 * phixx

        
        R2 = self.a55 * (wxx + self.lambda_val * phix) + self.q
        
        R2 = R2 - self.n_xT * wxx
        
        R2 = R2 - self.k1 * w_field + self.k2 * wxx

        
        R3 = self.b11 * uxx + self.d11 * phixx - self.a55 * self.lambda_val * (wx + self.lambda_val * phi_field)

        return R1, R2, R3
    
    def compute_nonlinear(self, u: torch.Tensor, w: torch.Tensor, phi: torch.Tensor,
                         A: torch.Tensor, B: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        
        ux = A @ u
        wx = A @ w
        phix = A @ phi

        uxx = B @ u
        wxx = B @ w
        phixx = B @ phi

        
        interior_mode = uxx.shape[0] != u.shape[0]

        if interior_mode:
            field_slice = slice(1, -1)
            u_field = u[field_slice]
            w_field = w[field_slice]
            phi_field = phi[field_slice]
        else:
            u_field = u
            w_field = w
            phi_field = phi

        

        
        R1 = self.a11 * uxx + self.b11 * phixx + (self.a11 / self.lambda_val) * wx * wxx

        
        
        term_a = (self.a11 / self.lambda_val) * (
            uxx * wx + ux * wxx + (3.0 / (2.0 * self.lambda_val)) * wxx * (wx ** 2)
        )
        term_b = (self.b11 / self.lambda_val) * (phixx * wx + phix * wxx)

        R2 = term_a + term_b + self.a55 * (wxx + self.lambda_val * phix) + self.q
        
        R2 = R2 - self.n_xT * wxx
        
        R2 = R2 - self.k1 * w_field + self.k2 * wxx

        
        R3 = self.b11 * uxx + self.d11 * phixx + (self.b11 / self.lambda_val) * wx * wxx - \
             self.a55 * self.lambda_val * (wx + self.lambda_val * phi_field)

        return R1, R2, R3
    
    def compute(self, u: torch.Tensor, w: torch.Tensor, phi: torch.Tensor,
               A: torch.Tensor, B: torch.Tensor, is_nonlinear: bool = False
               ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        if is_nonlinear:
            return self.compute_nonlinear(u, w, phi, A, B)
        else:
            return self.compute_linear(u, w, phi, A, B)
