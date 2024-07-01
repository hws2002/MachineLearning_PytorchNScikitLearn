import numpy as np
import pandas as pd
import sklearn
from sklearn import dataset

def load_iris_from_skl():
    iris = dataset.load_iris()
    X = iris.data
    y = iris.target
    return X,y
