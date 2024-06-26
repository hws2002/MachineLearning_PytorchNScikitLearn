import numpy as np 
class AdalineGD:
    def __init__(self, eta = 0.01, n_iter = 50, random_state = 1):
        self.eta = eta
        self.n_iter= n_iter
        self.random_state = random_state
    
    def fit(self, X, y):
        rng = np.random.default_rng(self.random_state)
        self.w_ = rng.normal(loc = 0.0, scale = 0.01, size = X.shape[1])
        self.b_ = np.float_(0.0)
        self.losses_ = []
        
        for _ in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            error = ( y - output)
            self.w_ += self.eta * 2.0 * np.dot(X.T, error) / X.shape[0] 
            self.b_ += self.eta * 2.0 * error.mean()
            loss = (error ** 2).mean()
            self.losses_.append(loss)
            
        
        
    def net_input(self, X):
        return np.dot(X, self.w_) + self.b_
    
    def activation(self, X):
        return X
    
    def predict(self, X):
        return np.where(  self.activation(self.net_input(X)) >= 0.5 , 1, 0 )