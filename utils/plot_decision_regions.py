import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

def plot_decision_boundary(X, y, classifier, resolution = 0.01):
    markers = ('o', 's', '^', 'v', '<')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    # decesion boundary 그리고
    x1_min, x1_max = X[:,0].min() -1, X[:,0].max() + 1
    x2_min, x2_max = X[:,1].min() -1, X[:,1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    x = np.array([xx1.ravel(), xx2.ravel()]).T
    lab = classifier.predict(x)
    lab = lab.reshape(xx1.shape)
    plt.contourf(xx1,xx2,lab, alpha = 0.3, cmap = cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # scatter plot 그리기
    for idx ,cl in enumerate(np.unique(y)):
      mask = (y == cl)
      plt.scatter(X[mask,0],
                  X[mask,1],
                  marker = markers[idx],
                  alpha = 0.8,
                  c = colors[idx],
                  label = f'Class {cl}',
                  edgecolor = 'black' 
      )
