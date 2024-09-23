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

###########################################################################
# initialisation ++ du kmeans
#
def initPlusPlus(X,K):
    N,p = np.shape(X)
    C = np.zeros((p,K))
    generator = np.random.default_rng()
    
    index = np.random.choice(N, 1,replace = False)
    liste_index = [index]
    C[:,0] = X[index,:]
    X = np.delete(X,index,0)
    #print("k=0 C[k]=",C[:,0],"index=",index)
    k=1
    
    while k < K:
        
        # calcul des distances
        NN = X.shape[0]
        dist = np.zeros(NN)
        
        for n in range(NN):
            D = C[:, :k] - np.repeat(X[n, :], k).reshape(p,k)  
            D = np.diag(D.T @ D)

            dist[n] = np.min(D)



        # calcul des probabilités
        proba = dist/np.sum(dist)
        
        
        
        # tirage aléatoire selon proba

        rand_val = generator.random((1))[0]
        intervals = np.cumsum(proba)
        index = 0
        while index < NN:
            if intervals[index] > rand_val:
                break
            index += 1

        C[:,k] = X[index,:]
        X = np.delete(X,index,0)
        k +=1
    return C

# initialisation
#  k = 1, choisir un premier centroïd �! par tirage aléatoire
# Tant que � < � répéter
#  Déterminer le centroïde �,$ le plus proche de chaque exemple
#  Affecter à chaque exemple une probabilité proportionnelle à �$ − �,$
# %
#  k=k+1
#  Tirer aléatoirement le centroïde �! parmi les points selon leur probabilité


def my_kmeans(X,K,Visualisation=False,Seuil=0.001,Max_iterations = 1000, kpp= False):
    
    N,p = np.shape(X)
    iteration = 0        
    Dist=np.zeros((K,N)) # distance au carré entre les points et les centres
    J=np.zeros(Max_iterations+1)
    variance_explained = np.zeros(Max_iterations)
    J[0] = 10000000
    r = np.zeros((N, K))
    
    # Initialisation des clusters
    # par tirage de K exemples, pour tomber dans les données     
 
    Index_init = np.random.choice(N, K,replace = False)
    if kpp:
        C = initPlusPlus(X,K)
    else:
        C = np.zeros((p,K))
        for k in range(K):
            C[:,k] = X[Index_init[k],:].T 
        
        
        
    while iteration < Max_iterations:
        iteration +=1
        #################################################################
        # E step : estimation des données manquantes 
        #          affectation des données aux clusters les plus proches
        # for n in range(N):
        for k in range(K):
            Dist[k,:] = np.linalg.norm(X-C[:,k], axis=1)**2
        
        
        
        #################################################################
        # M Step : calcul des meilleurs centres
        y = np.argmin(Dist,axis=0)          
        for n in range(N):
            r[n,y[n]] = 1

        for k in range(K):
            C[:,k] = np.mean(X[y==k,:],axis=0)
        
        J[iteration] = (1/N)*np.sum(np.min(Dist[y, :],axis=0))
        Ik=np.array([np.sum((X[y==k]-C[:,k])**2)/len(X[y==k]) for k in range(len(C[0]))])
        Iw=np.sum([(len(X[y==k])*Ik[k])/len(X) for k in range(k)])
        Ib=np.sum([(len(X[y==k])/len(X))*(X.mean(axis=0)-C[:,k])**2 for k in range(k)])
        It=Iw+Ib
        variance_explained[iteration] =  100*(1-(Iw/It))
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
            plt.plot(X[y==k, 0], X[y==k, 1], colors[k%n_colors]+'o', label=f"laberl : {k}")
        plt.plot(C[0, :], C[1, :],'kx')
        plt.title('K moyennes ('+str(K)+')')
        plt.legend()
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

    print("Nombre d'itérations :",iteration)
    return C, y, J[1:iteration], variance_explained[:iteration]


if __name__ == '__main__':

#########################################################
#''' K means '''
    iris = datasets.load_iris()
    X = iris.data[:, :2]  # we only take the first two features.
    y = iris.target
    K = 4


    fig = plt.figure(2, figsize=(8, 6))
    plt.clf()
    plt.scatter(X[0:50, 0], X[0:50, 1], cmap=plt.cm.Set1,edgecolor='k',label=iris.target_names[0])
    plt.scatter(X[50:100, 0], X[50:100, 1], cmap=plt.cm.Set1,edgecolor='k',label=iris.target_names[1])
    plt.scatter(X[100:150, 0], X[100:150, 1], cmap=plt.cm.Set1,edgecolor='k',label=iris.target_names[2])
    
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')
    plt.legend(scatterpoints=1)


    Cluster, y, Critere, variance_explique = my_kmeans(iris.data, K, Visualisation = True)
    

    # Cluster, y, Critere = my_kmeans(iris.data,K,Visualisation = False)
    
    # fig = plt.figure(3, figsize=(8, 6))
    # for k in range(K):
    #     plt.plot(X[y==k, 0], X[y==k, 1], colors[k%n_colors]+'o')
    # plt.plot(Cluster[0, :], Cluster[1, :],'kx')
    # plt.title('K moyennes ('+str(K)+')')
    # plt.show()
    
    # fig = plt.figure(figsize=(8, 6))
    # print(Critere)
    # plt.plot(Critere, 'o-')
    # plt.xlabel('Iteration')
    # plt.ylabel('Mean Squared Error (MSE)')
    # plt.title('Evolution du critère')
    # plt.show()

    # fig = plt.figure(figsize=(8, 6))
    # print(variance_explique)    
    # plt.plot(variance_explique, 'o-')
    # plt.xlabel('Iteration')
    # plt.ylabel('Variance expliquée (%)')
    # plt.title('Evolution de la variance expliquée')
    # plt.show()
        
