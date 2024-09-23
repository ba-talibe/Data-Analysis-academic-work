#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# M1 Science et Ingénieurie des données
# Université de Rouen Normandie
# T. Paquet
import matplotlib
import matplotlib.pyplot as plt
from sklearn import datasets
import numpy as np
from numpy.linalg import norm

colors =['r','b','g','c','m','o']
n_colors = 6
def my_kmedoides(X,K,Visualisation=False,Seuil=0.001,Max_iterations = 100):
    
    N,p = np.shape(X)
    iteration = 0        
    Normes=np.zeros((K,N))

    J=np.zeros(Max_iterations+1)
    variance_explained = np.zeros(Max_iterations)

    # Initialisation des clusters
    # par tirage de K exemples, pour tomber dans les données     
    Index_init = np.random.choice(N, K,replace = False)
    C = np.zeros((K,p))
    for k in range(K):
        C[k,:] = X[Index_init[k],:]


    while iteration < Max_iterations:
        iteration += 1
        print("Iteration :",iteration)
        #################################################################
        #          affectation des données aux médoïde le plus proches
        for k in range(K):
            Normes[k,:] = np.linalg.norm(X-C[k,:], axis=1)**2
        r  = np.zeros((N, K))
        
        y = np.argmin(Normes,axis=0)

        
        #################################################################
        # Calcul des meilleurs médoïdes
   
        for k in range(K):
            cluster_points = X[y == k]
            if len(cluster_points) == 0:
                continue
            distances = np.sum(np.linalg.norm(cluster_points[:, np.newaxis] - cluster_points, axis=2), axis=1)
            medoid_index = np.argmin(distances)
            C[k,:] = cluster_points[medoid_index]
        
        

        J[iteration] = (1/N)*np.sum(np.min(Normes[y, :],axis=0))
        Ik=np.array([np.sum((X[y==k]-C[k,:])**2)/len(X[y==k]) for k in range(C.shape[0])])
        Iw=np.sum([(len(X[y==k])*Ik[k])/len(X) for k in range(k)])
        Ib=np.sum([(len(X[y==k])/len(X))*(X.mean(axis=0)-C[k,:])**2 for k in range(k)])
        It=Iw+Ib
        variance_explained[iteration] = 100*(1-(Iw/It))
        #################################################################
        # test du critère d'arrêt l'évolution du critère est inférieure 
        # au Seuil en pour cent
        if abs(J[iteration-1]-J[iteration])/J[iteration-1] < Seuil:
            break

    if Visualisation:
        plt.subplots(1, 3, figsize=(19, 7))
        plt.subplot(1, 3, 1)

        # fig = plt.figure(3, figsize=(8, 6))
        for k in range(K):
            plt.plot(X[y==k, 0], X[y==k, 1], colors[k%n_colors]+'o')
        plt.plot(C[:, 0], C[:, 1],'kx')
        plt.title('K medoïdes ('+str(K)+')')

        # plt.show()
        
        plt.subplot(1, 3, 2)
        plt.plot(J[:iteration], 'o-')
        plt.xlabel('Iteration')
        plt.ylabel('Mean Squared Error (MSE)')
        plt.title('Evolution du critère')
        # plt.show()

        # fig = plt.figure(figsize=(8, 6))
        # print(variance_explique[:iter])    
        plt.subplot(1, 3, 3)
        plt.plot(variance_explained[:iteration], 'o-')
        plt.xlabel('Iteration')
        plt.ylabel('Variance expliquée (%)')
        plt.title('Evolution de la variance expliquée')
        plt.show()

    return C, y, J[:iteration+1], variance_explained[:iteration]


if __name__ == '__main__':

#########################################################
#''' K means '''
    iris = datasets.load_iris()
    X = iris.data#[:, :2]  # we only take the first two features.
    y = iris.target
    K= 3


    fig = plt.figure(2, figsize=(8, 6))
    plt.clf()
    plt.scatter(X[0:50, 0], X[0:50, 1], cmap=plt.cm.Set1,edgecolor='k',label=iris.target_names[0])
    plt.scatter(X[50:100, 0], X[50:100, 1], cmap=plt.cm.Set1,edgecolor='k',label=iris.target_names[1])
    plt.scatter(X[100:150, 0], X[100:150, 1], cmap=plt.cm.Set1,edgecolor='k',label=iris.target_names[2])
    
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')
    plt.legend(scatterpoints=1)


    Cluster, y, Critere, variance_explique = my_kmedoides(X, K, Visualisation = True)
    
    
    # fig = plt.figure(3, figsize=(8, 6))
    # for k in range(K):
    #     plt.plot(X[y==k, 0], X[y==k, 1], colors[k%n_colors]+'o')
    # plt.plot(Cluster[:,0], Cluster[:,1],'kx')
    # plt.title('K medoïdes ('+str(K)+')')
    # plt.show()
    
    # fig = plt.figure(figsize=(8, 6))
    # plt.plot(Critere, 'o-')
    # plt.xlabel('Iteration')
    # plt.ylabel('Mean Squared Error (MSE)')
    # plt.title('Critère des k-médoïdes')
    # plt.show()
    
    # print("Critere:",Critere)

    # fig = plt.figure(figsize=(8, 6))
    # print(variance_explique)    
    # plt.plot(variance_explique, 'o-')
    # plt.xlabel('Iteration')
    # plt.ylabel('Variance expliquée (%)')
    # plt.title('Evolution de la variance expliquée')
    # plt.show()
    # #
        
