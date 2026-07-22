

import torch
from typing import Dict, Optional

def get_boundary_conditions(bc_type: str, device: Optional[torch.device] = None) -> Dict:
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
    def zero():
        return torch.tensor(0.0, device=device, dtype=torch.float64)

    
    bc_dict = {
        'C-C': {  
            'u': (zero(), zero()),    
            'w': (zero(), zero()),    
            'phi': (zero(), zero()),  
            'description': 'Clamped-Clamped'
        },
        'S-S': {  
            'u': (zero(), zero()),    
            'w': (zero(), zero()),    
            'phi': (None, None),      
            'description': 'Simply-Supported',
            'additional': 'M(0)=0, M(L)=0'  
        },
        'H-H': {  
            'u': (zero(), zero()),    
            'w': (zero(), zero()),    
            'phi': (None, None),      
            'description': 'Hinged-Hinged',
            'additional': 'M(0)=0, M(L)=0'
        },
        'C-S': {  
            'u': (zero(), zero()),    
            'w': (zero(), zero()),    
            'phi': (zero(), None),    
            'description': 'Clamped-Simply',
            'additional': 'M(L)=0'
        },
        'C-H': {  
            'u': (zero(), zero()),    
            'w': (zero(), zero()),    
            'phi': (zero(), None),    
            'description': 'Clamped-Hinged',
            'additional': 'M(L)=0'
        }
    }

    if bc_type not in bc_dict:
        raise ValueError(f"Unsupported boundary condition type: {bc_type}. "
                        f"Supported types: {list(bc_dict.keys())}")

    bc = bc_dict[bc_type].copy()

    
    
    if bc_type in ['H-H', 'S-S']:
        
        bc['phi'] = (zero(), zero())  
    elif bc_type in ['C-S', 'C-H']:
        
        bc['phi'] = (zero(), zero())  

    
    result = {
        'u': bc['u'],
        'w': bc['w'],
        'phi': bc['phi'],
        'type': bc_type
    }
    
    return result



def print_boundary_info(bc_type: str):
    
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
    
    return ['C-C', 'S-S', 'C-S', 'C-H', 'H-H']