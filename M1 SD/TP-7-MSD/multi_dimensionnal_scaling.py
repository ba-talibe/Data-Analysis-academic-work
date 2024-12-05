import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.neighbors import kneighbors_graph
from scipy.sparse.csgraph import shortest_path
from operator import itemgetter
from sklearn import manifold

def TriVP(Valp,Vectp):
    # trie dans l'ordre décroisant les valeurs propres
    # en cas de valeurs propres complexes on trie  selon leu module
    liste1 = Vectp.tolist()
    liste2 = Valp.tolist()
    norme = np.abs(Valp)
    liste3 = norme.tolist()

    result = zip(liste1, liste2,liste3)
    result_trie =sorted(result,key =itemgetter(2), reverse=True)
    liste1, liste2, liste3 =  zip(*result_trie)
    Vectp = np.asarray(liste1)
    Valp = np.asarray(liste2)
    
    return Valp,Vectp


def multi_dimendional_scalling(data, n_neighbors=50, n_components=2):
    """
    This function takes a dataframe and returns a dataframe with n_components
    columns that are the result of the multi-dimensional scalling.
    """

    # Import the necessary libraries
    kng = kneighbors_graph(data, n_neighbors=n_neighbors, mode='distance')
    D = shortest_path(kng, directed=True)

    m = D.shape[0]
    Id = np.eye(m)

    ones = np.ones(m)

    # B = -.5*(1 - 1/n_neighbors)* (Id - 1/m*ones)@(D**2)@(Id - 1/m*ones)
    B = -.5* (Id - (1/m)*ones)@(D**2)@(Id - (1/m)*ones)

    v, V = np.linalg.eig(B)

    idx = np.argsort(v)[::-1]
    val_max = v[idx]
    vec_max = V[:, idx]
    Y = vec_max[:, :n_components]@np.diag(val_max[:n_components]**(-.5))
    return Y

def plot_mds(Y, n_neighbors=39, n_components=2):
    """
    This function takes a dataframe and the labels of the data and plots the
    data in 2D using the multi-dimensional scalling.
    """

    # Apply the multi-dimensional scalling
    plt.figure(figsize=(8, 8))
    plt.scatter(Y[:, 0], Y[:, 1], c=y_mnist, cmap=plt.cm.Set1,edgecolor='k')
    plt.xlabel('CP 1')
    plt.ylabel('CP 2')
    plt.title('myISO des données IRIS')
    plt.show()


    X_iso = manifold.Isomap(n_neighbors=39, n_components=2, path_method='D').fit_transform(X_wine)

    plt.figure(figsize=(8, 8))
    plt.scatter(X_iso[:, 0], X_iso[:, 1], c=y_mnist, cmap=plt.cm.Set1,edgecolor='k')
    plt.xlabel('CP 1')
    plt.ylabel('CP 2')
    plt.title('sklearn ISO des données IRIS')
    plt.show()


if __name__ == '__main__':
    # Load the iris dataset
    iris = datasets.load_iris()
    wine = datasets.load_wine()
    X_wine = wine.data
    y_wine = wine.target
    X_iris = iris.data
    y_iris = iris.target
    mnist = datasets.load_digits()
    X_mnist = mnist.data
    y_mnist = mnist.target

    # Apply the multi-dimensional scalling
    Y = multi_dimendional_scalling(X_mnist, n_neighbors=50, n_components=2)
    plot_mds(Y, n_neighbors=39, n_components=2)