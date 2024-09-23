import numpy as np
import matplotlib.pyplot as plt

def RadialVisualization(X,target,target_names,feature_names, dataset_name="data",r=1):
    plt.figure(figsize=(7,7))
    n_colors = 7
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    offset = 0.02
    # Etape 1: tracer le cercle de rayon r
    p = len(feature_names)
    n = len(target_names)

    theta = np.linspace(0, 2*np.pi, 200)
    circle_x = r*np.cos(theta)
    circle_y = r*np.sin(theta)
    plt.plot(circle_x,circle_y)
    plt.axis('equal')
    #plt.set_aspect(1)
    plt.grid(linestyle='--')
    plt.title("Radial Visualization of "+dataset_name, fontsize=8)
    # Etape 2: calculer les coordonnées des n ancres et les tracer sur le cercle
   
    phi = np.linspace(0, 2*np.pi, p+1)
    ancre_x = r*np.cos(phi)
    ancre_y = r*np.sin(phi)
    plt.plot(ancre_x,ancre_y,'ro')


    for i in range(p):
        plt.text(ancre_x[i]+offset,ancre_y[i]+offset,feature_names[i])
    # Etape 3: calculer les coordonnées des points et les tracer à l’intérieur du cercle
    Norm = np.sum(X, axis=1)
    X1 = np.dot(X, ancre_x[:p].T)/Norm
    X2 = np.dot(X, ancre_y[:p].T)/Norm
    for i in range(n):
        plt.plot(X1[target == i], X2[target == i], 'x', color=colors[i])
    
    plt.show()