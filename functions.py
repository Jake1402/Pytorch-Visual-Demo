import numpy as np

class functions():
    def __init__(self):
        self.name = ""
        self.lamb_func = lambda x : x
    
    def __call__(self, X):
        return self.lamb_func(X)
    
class quad_func(functions):
    def __init__(self):
        super().__init__()
        self.name = "Quadratic Function"
        self.lamb_func = lambda x : -(0.2*(x**2) - (0.5*x) + 1.5)

class sinx_func(functions):
    def __init__(self):
        super().__init__()
        self.name = "Function of Sin(x)"
        self.lamb_func = lambda x : np.sin(x)

class cosx_func(functions):
    def __init__(self):
        super().__init__()
        self.name = "Function of Cos(x)"
        self.lamb_func = lambda x : np.cos(x)