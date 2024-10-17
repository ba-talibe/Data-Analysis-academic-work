import numpy as np
from your_DBSCAN import my_DBSCAN
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt



iris = load_iris()
X = iris.data
y = iris.target
eps =  np.array([ 0.3, 0.5, 0.7])
minpts = np.array([ 3, 5, 7])

colors =['k','r','b','g','c','m',]
n_colors = 6

plt.subplots(3, 3, figsize=(15, 15))

for ie, epsi in enumerate(eps):
    for im, minptsi in enumerate(minpts):
        print("eps = ", epsi, " minpts = ", minptsi)
        y = my_DBSCAN(X, epsi, minptsi)
        statistiques = np.unique(y,return_counts=True)
        K = len(statistiques[0])-(1 if -1 in statistiques[0] else 0)
        print("clusters = ", len(set(y)))
        print("rejetés = ", len([i for i in y if i == -1]))
        plt.subplot(3, 3, ie*3 + im + 1)
        for k in range(1,K+1):
            plt.plot(X[y==k, 0], X[y==k, 1], colors[k%n_colors]+'o')
        plt.plot(X[y==-1, 0], X[y==-1, 1], 'kv')
        plt.show()
        plt.title(f"eps = {epsi}, minpts = {minptsi}")
        print("")