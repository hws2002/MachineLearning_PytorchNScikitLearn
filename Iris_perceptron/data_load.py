import os 
import pandas as pd

s = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'

df = pd.read_csv(s,
                 header = None,
                 encoding = 'utf-8')
df.tail()