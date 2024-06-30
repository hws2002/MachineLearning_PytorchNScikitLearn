import numpy as np

class LogisticRegressionGD:
    """
        완전 배치 경사 하강법으로 구현한 logistic regression classifier
    """
    def __init__(self, eta = 0.01, n_iter = 50,random_state = 1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state_state
    
    def fit(self, X, y):
        rng = np.random.default_rng(self.random_state)
        self.w_ = rng.normal(loc = 0.0, scale = 0.01, size = X.shape[1])
        self.b_ = np.float(0.0)
        self.losses_ = []
        
        # TODO
        for _ in range( self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            self.w_ = self.eta * 
            self.b_ = self.eta * 
            loss = 
            self.losses_.append(loss)
        
        return self

    def activation(self, X):
        """
            로지스틱 시그모이드 활성화 계산
        """
        return 1.0 / ( 1.0 + np.exp(-X))
    
    def net_input(self, X):
        return np.dot(X, self.w_) + self.b_

    def predict(self, X):
        return np.where( self.activation(self.net_input(X)) >= 0.5, 1, 0)