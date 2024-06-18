import numpy as np

class Perceptron :
    def __init__(self, eta = 0.01, n_iter = 50, random_state = 1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
    
    def fit(self, X, y):
        # 훈련 데이터 학습
        rng = np.random.default_rng(self.random_state)
        self.w_ = rng.normal(loc = 0.0, scale = 0.1, size = X.shape[1])
        self.b_ = np.float_(0.)
        self.errors_ = []
        
        for _ in range (self.n_iter):
            error = 0
            for xi, target in zip(X,y):
                update = self.eta * (target - self.predict(xi))
                self.w_ += update*xi
                self.b_ += update
                error += int(update != 0.0)
            self.errors_.append(error)
        return self

    def net_input(self, X):
        return np.dot(X , self.w_)  + self.b_
    
    def predict(self, X):
        return np.where( self.net_input(X) >= 0.0 , 1 , 0)
