from sklearn.neighbors import kneighbors_graph
import numpy as np

def laplacian_eigenmaps(X, n_neighbors, n_components, normilize=True):
    kng = kneighbors_graph(X, n_neighbors, mode='distance')
    kng = 0.5 * (kng + kng.T)
    W =  kng.toarray()
    D = np.diag(W.sum(axis=1))

    if normilize:
        L = np.eye(X.shape[0]) - np.linalg.inv(D) @ W
   