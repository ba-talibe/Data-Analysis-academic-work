#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  8 09:38:57 2022

@author: Thierry Paquet
"""

#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
from sklearn import datasets
from sklearn.cluster import DBSCAN
from numpy.linalg import norm
import matplotlib.pyplot as plt

import sklearn.cluster

colors =['k','r','b','g','c','m',]
n_colors = 6

###########################################################################
def EpsilonVoisinage(i,X,Dist,eps):
    N,p =np.shape(X)
    # Voisins = [v for v in range(N) if ( i != v and Dist[v,i] < eps)]
    Voisins = [v for v in range(N) if  Dist[v,i] < eps]
    return Voisins

###########################################################################
def etendre_cluster(X, y, Dist, Cluster, no_cluster, Voisins, Visite, eps, minpts):

    for v in Voisins:
        #### VOTRE CODE ICI
        if not Visite[v]:
            Visite[v] = True
            if y[v] == -1:
                y[v] = no_cluster
                Cluster.append(v)

            Voisins_v = EpsilonVoisinage(v,X,Dist,eps)
            if len(Voisins_v) >= minpts:
                for vv in Voisins_v:
                    if vv not in Voisins:
                        Voisins.append(vv)

    return Cluster, y, Visite


##########################################################################
#              MY DBSCAN
def my_DBSCAN(X, eps=None, minpts=None, Visualisation = False):
    N,pp =np.shape(X)
    no_cluster = 0
    
    # on pré-calcule toutes les distances entre points
    Dist = np.reshape(norm(X - X[0,:],axis=1),(N,1))
    for n in range(1,N):
        D = np.reshape(norm(X - X[n,:],axis=1),(N,1))
        Dist = np.concatenate((Dist,D),axis=1)
    
    if eps is None:
        eps = estime(Dist)
    
    if minpts is None:
        minpts = estime_minpts(X, Dist, eps)

    Visite = [False for _ in range(N)]
    
    y = - np.ones(N)  # tableau des labels des données, initialisé bruit (-1)
    Clusters = []
    
    for p in range(N):
        ######### VOTRE CODE ICI
        if not Visite[p]:
            Visite[p] = True
            Voisins = EpsilonVoisinage(p,X,Dist,eps)
            if len(Voisins) >= minpts:
                no_cluster += 1
                y[p] = no_cluster
                Cluster = [p]
                Cluster, y, Visite = etendre_cluster(X, y, Dist, Cluster, no_cluster, Voisins, Visite, eps, minpts)
                Clusters.append(Cluster) 

    if Visualisation :
        print(len(Clusters),' clusters trouvés', no_cluster)
        print("Clusters =",Clusters)
        for cluster in Clusters:
            print('effectif cluster ',len(cluster))
                       
        Bruit = [n for n in range(N) if y[n] == -1]
        print('effectif  bruit',len(Bruit))

    return y

def estime(Dist):

    N = np.shape(Dist)[0]
    Diag =  np.eye(N)*np.max(Dist)*2
    EPS = np.percentile(np.min(Dist+Diag, axis=0), 95)
    return EPS

def estime_minpts(X, Dist, eps):
    NVoisins = []
    N, p = np.shape(X)
    for p in range(N):
        NVoisins += [len(EpsilonVoisinage(p, X, Dist, eps))]
    return np.ceil(np.percentile(np.array(NVoisins, dtype=np.float64), 5))

def plot_iris(iris):
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    fig = plt.figure(2, figsize=(8, 6))
    plt.clf()
    plt.scatter(X[0:50, 0], X[0:50, 1], cmap=plt.cm.Set1,edgecolor='k',label=iris.target_names[0])
    plt.scatter(X[50:100, 0], X[50:100, 1], cmap=plt.cm.Set1,edgecolor='k',label=iris.target_names[1])
    plt.scatter(X[100:150, 0], X[100:150, 1], cmap=plt.cm.Set1,edgecolor='k',label=iris.target_names[2])
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')
    plt.legend(scatterpoints=1)

def visualize_dbscan(X, y_pred, title="DBSCAN"):
    statistiques = np.unique(y_pred,return_counts=True)
    K = len(statistiques[0])-(1 if -1 in statistiques[0] else 0)
    Bruit = [p for p in range(len(y_pred)) if y_pred[p]==-1]
    
    fig = plt.figure(figsize=(8, 6))

    for k in range(1,K+1):
        plt.plot(X[y_pred==k, 0], X[y_pred==k, 1], colors[k%n_colors]+'o')
    plt.plot(X[y_pred==-1, 0], X[y_pred==-1, 1], 'kv')

    plt.title(f'{title}:'+str(K)+' clusters, '+str(len(Bruit))+' noise')
    plt.show()

if __name__ == '__main__':

#########################################################

    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    plot_iris(iris)

    eps = 0.5
    minpts = 5
    
    my_y = my_DBSCAN(X, Visualisation=True)
    visualize_dbscan(X, my_y, title="My DBSCAN")

    # comparaison avec DBSCAN de scikit learn
    yy = DBSCAN(eps=eps,min_samples=minpts).fit_predict(X)
    visualize_dbscan(X, yy, title="scikit learn DBSCAN")