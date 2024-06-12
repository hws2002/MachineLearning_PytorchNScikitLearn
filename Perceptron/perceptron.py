import numpy as np

class Perceptron:
    """퍼셉트론 분류기
    
    매개변수
    ----------------
    eta : float
        학습률( 0.0 ~ 1.0)
    n_iter : int
        훈련 데이터셋 반복 횟수
    random_state : int
        가중치 무작위 초기화를 위한 난수 생성기 시드
    
    속성
    ----------------
    w_ : 1d-array
        학습된 가중치
    b_ : 스칼라
        학습된 절편 유닛
    
    errors_ : list
        에포크마다 누저도니 분류 오류
    
    """
    
    def __init__(self, eta = 0.01, n_iter = 50, random_state = 1):
        self.eat = eta
        self.n_iter = n_iter
        self.random_state = random_state
    
    def fit(self, X, y):
        """훈련 데이터 학습
        
        매개변수
        ----------------
        X : {array-like}, shape = [n_samples, n_features]
            n_samples개의 샘플과 n_features개의 특성으로 이루어진 훈련 데이터
        
        y : array-like, shape = [n_samples]
            타깃 값
        
        반환값
        ----------------
        self : object
        
        """
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc = 0.0, scale = 0.01, size = X.shape[1])
        
        self.b_ = np.float_(0.)
        self.errors_ = []
        
        def net_input(self, X);
            return np.dot(X,self.w_) + self.b_
            
        def predict(self, X):
            return np.where( self.net_input(X) >= 0.0 ,1, 0)