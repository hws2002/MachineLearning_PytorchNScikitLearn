import numpy as np

class LogisticRegressionGD:
    """
        완전 배치 경사 하강법으로 구현한 logistic regression classifier
        ! 이진 분류에만 적용할 수 있음
    """
    def __init__(self, eta = 0.01, n_iter = 50,random_state = 1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
    
    def fit(self, X, y):
        rng = np.random.default_rng(self.random_state)
        self.w_ = rng.normal(loc = 0.0, scale = 0.01, size = X.shape[1])
        self.b_ = np.float_(0.0)
        self.losses_ = []
        
        # TODO
        for _ in range( self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = (y - output)
            self.w_ += self.eta * 2.0 * np.dot(X.T, errors) / X.shape[0]
            self.b_ += self.eta * 2.0 * np.mean(errors)
            loss = np.mean(-y*np.log(output) - (1-y) * np.log(1-output))
            self.losses_.append(loss)
        
        return self

    def activation(self, z):
        """
            로지스틱 시그모이드 활성화 계산
        """
        return 1.0 / ( 1.0 + np.exp( np.clip(-z,-250, 250) ) )
    
    def net_input(self, X):
        return np.dot(X, self.w_) + self.b_

    def predict(self, X):
        return np.where( self.activation(self.net_input(X)) >= 0.5, 1, 0)