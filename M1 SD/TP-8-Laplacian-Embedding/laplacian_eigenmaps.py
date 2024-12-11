from sklearn.neighbors import kneighbors_graph
from sklearn.datasets import load_iris
# import matplotlib.pyplot as plt
from scipy import sparse 
import numpy as np

def laplacian_eigenmaps(X,y, n_neighbors, normilize=True):
    N = X.shape[0]  
    kng = kneighbors_graph(X, n_neighbors, mode='distance')
    W = 0.5 * (kng + kng.T)

    #W = np.where(W > 0, 1, 0)
    D = sparse.diags(np.asarray(W.sum(axis=1)).flatten())

    if normilize:
        L = np.eye(X.shape[0]) - D**-0.5 @ W @ D**-0.5
    else:
        L = D - W

    [yl, YL] = sparse.linalg.eigsh(L, k=n_neighbors, which='SM')
   
    print(YL)
    return YL
    # plt.figure(figsize=(8,8))
    # plt.scatter(YL[:,2], YL[:,3], c=y[0, N], cmap=plt.cm.Set1)
    # plt.show()

   
if __name__ == '__name__':
    iris = load_iris()
    X = iris.data
    y = iris.target
    print(laplacian_eigenmaps(X, y, 4, True))