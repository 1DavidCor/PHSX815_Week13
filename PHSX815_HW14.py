# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 20:05:48 2021

@author: d338c921
"""

#Generate data from a "mixture model"
#Implement the K-means clustering algorithm and apply it to your data
#Some potential points to evaluate: How similar are the clusters to the "true" mixture? Does this depend on the amount of data? 
#How does the model change with the number of mixture components (keeping the number fixed in the generating model)? How well can you visualize your data and algorithm?

# imports of external packages to use in our code
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans

# main function for our coin toss Python code
if __name__ == "__main__":
    # if the user includes the flag -h or --help print the options
    if '-h' in sys.argv or '--help' in sys.argv:
        print ("Usage: \n -N [number of clusters] \n -n [sample size of each cluster] \n")
        print
        sys.exit(1)

    
    #default number of clusters:
    N = 5
    
    #default sample size in each cluster
    n = 100

    # read the user-provided values from the command line (if there)
    if '-N' in sys.argv:
        p = sys.argv.index('-N')
        N = int((sys.argv[p+1]))
    if '-n' in sys.argv:
        p = sys.argv.index('-n')
        n = int((sys.argv[p+1]))

    #genrate the "preliminary" coordinates of the n cluster centers
    x = np.random.uniform(0, 10, size = N)
    y = np.random.uniform(0, 10, size = N)
    
    #generate random points around the cluster centers
    data_x = []
    data_y = []
    
    for i in range(N):
        mu = np.random.uniform(0, 5)
        sig = np.random.uniform(0, 1)
        data_x = np.append(data_x, x[i] + np.random.normal(mu, sig, n))
        data_y = np.append(data_y, y[i] + np.random.normal(mu, sig, n))
    plt.figure()
    plt.title("Randomly Generated Mixture Model")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.scatter(data_x, data_y)
    #plt.scatter(x, y, marker = "x")
    plt.show()
    
    #K-Means clustering algorithm implementation: https://www.analyticsvidhya.com/blog/2019/10/gaussian-mixture-models-clustering/
    #train the k-means model
    data = np.vstack((data_x, data_y)).T #shape = (N*n, 2)
    kmeans = KMeans(n_clusters = N)
    kmeans.fit(data)
    
    #predictions from kmeans
    pred = kmeans.predict(data)
    frame = pd.DataFrame(data)
    frame['cluster'] = pred
    frame.columns = ['X', 'Y', 'cluster']
    
    #plotting cluster results
    plt.figure()
    plt.title("K-Means Clustering")
    plt.xlabel("X")
    plt.ylabel("Y")
    for k in range(N):
        data = frame[frame["cluster"]==k]
        plt.scatter(data["X"],data["Y"])
    plt.show()