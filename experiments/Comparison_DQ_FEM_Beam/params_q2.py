


h = 0.1                 
L = 20 * h              
num_layers = 10
W_Gr = 0.025            
H_Gr = 0.8             
T = 300.0              
distribution = 'X'     
q = -0.08              
bc_type = 'C-C'        


N = 13                  


SCENARIOS = [
    {"name": "with_foundation", "k1": 0.01, "k2": 0.001},
    {"name": "no_foundation",   "k1": 0.0,  "k2": 0.0},
]



REFERENCE_W_MID = {
    "with_foundation": {"linear": -0.46834, "nonlinear": -0.42895},
    "no_foundation":   {"linear": -0.52113, "nonlinear": -0.46520},
}





INIT_GUESS = "random"     
INIT_SEED  = 0            
INIT_SCALE = 0.01         
INIT_SWEEP = ["random", "gauss", "zero", "linear"]   
