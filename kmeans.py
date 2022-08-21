import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.datasets import load_digits

# read the datasets.csv
student_evaluation = pd.read_excel('/Users/me-mac/Desktop/study/AI/Moheb/student-evaluation_generic.xlsx')

# usage PCA to decrease the numbers of dimensions to visualize the results by using a 2D Scatter plot.
pca = PCA(2)
#Transform the data
df = pca.fit_transform(student_evaluation)
df.shape


# define the class object for kmean
kmeans = KMeans(n_clusters= 3)

#predict the labels of clusters.
label = kmeans.fit_predict(df)

#Getting unique labels
u_labels = np.unique(label)

#Getting the Centroids
centroids = kmeans.cluster_centers_

#ploting
for i in u_labels:
    plt.scatter(df[label == i , 0] , df[label == i , 1] , label = i)
plt.scatter(centroids[:,0] , centroids[:,1] , s = 80, color = 'black')
plt.legend()
plt.show()


#same thing with different methods 
 
# start loading data by using the pca
data = load_digits().data
pca = PCA(2)
 
# use fit to transform the data
df = pca.fit_transform(data)
df.shape


# define the class object for kmean
kmeans = KMeans(n_clusters= 3) 
#predict the labels of clusters.
label = kmeans.fit_predict(df)



#get the Centroids
centroids = kmeans.cluster_centers_
u_labels = np.unique(label)
 
#plotting:
 
for i in u_labels:
    plt.scatter(df[label == i , 0] , df[label == i , 1] , label = i)
plt.scatter(centroids[:,0] , centroids[:,1] , s = 80, color = 'k')
plt.legend()
plt.show()




