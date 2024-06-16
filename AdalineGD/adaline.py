import numpy as np 
class AdalineGD:
    def __init__(self, eta = 0.01, n_iter = 50, random_state = 1):
        self.eta = eta
        self.n_iter= n_iter
        self.random_state = random_state
    
    def fit(self, X, y):
        rng = np.random.default_rng(self.random_state)