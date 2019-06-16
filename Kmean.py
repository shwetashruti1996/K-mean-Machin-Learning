import numpy as np
import pandas as pd
from copy import deepcopy
from matplotlib import pyplot as plt 

def KMeans():
	#set three centers, the model should predict similar result
	center_1 = np.array([1,1])
	print(center_1)

	center_2 = np.array([5,5])
	print(center_2)

	center_3 = np.array([8,1])
	print(center_3)

	#Generate random data and center it to the three centers
	data_1 = np.random.randn(7,2) + center_1
	print("Elements of first cluster with size "+str(len(data_1)))
	print(data_1)

	data_2 = np.random.randn(7,2) + center_2
	print("Elements of second cluster with size "+str(len(data_2)))
	print(data_2)

	data_3 = np.random.randn(7,2) + center_3
	print("Elements of third cluster with size "+str(len(data_3)))
	print(data_3)

	data = np.concatenate((data_1,data_2,data_3),axis = 0)
	print("Size of complete data set"+str(len(data)))

	plt.scatter(data[:,0],data[:,1],s=7)
	plt.title('Input Dataset')
	plt.show()

	#Number of clusters
	k=3

	#Number of training data
	n=data.shape[0]
	print("Total number of elements are : ",n)

	#Number of features in the data
	c = data.shape[1]
	print("Total number of features are : ",c)

	#Generate random centers, here we use sigma and mean to ensure it represent the whole data
	mean = np.mean(data, axis = 0)
	print("Value of mean : ",mean)

	#calculate standard deviation
	std = np.std(data, axis = 0)
	print("Value of std : ",std)

	centers = np.random.randn(k,c)*std + mean
	print("Random points are : ",centers)

	#Plot the data and the centers generated as random
	plt.scatter(data[:,0],data[:,1],c='r',s=7)
	plt.scatter(centers[:,0],centers[:,1],marker='*',c='g',s=150)
	plt.title('Input Dataset with random centroid *')
	plt.show()

	centers_old = np.zeros(centers.shape) #to store old centers
	centers_new = deepcopy(centers) #to new centers

	print("Values of old centroids")
	print(centers_old)

	print("Values of new centroids")
	print(centers_new)

	data.shape
	clusters = np.zeros(n)
	distances = np.zeros((n,k))

	print("Initial distances are")
	print(distances)

	error = np.linalg.norm(centers_new - centers_old)

	#When, after an update, the estimate of that center stays the same, exit loop
	while  error != 0:
		print("Measure the distance to every center")
		for i in range(k):
			print("Iteration number",i)
			distances[:,i] = np.linalg.norm(data - centers[i],axis=1)

		#Assign all training data to closest center
		clusters = np.argmin(distances, axis = 1)
		centers_old = deepcopy(centers_new)

		#Calculate mean for every cluster and update the center
		for i in range(k):
                        centers_new[i] = np.mean(data[clusters == i], axis=0)
		error = np.linalg.norm(centers_new - centers_old)
	#end of while
	centers_new

	#plt the data and the centers generated as random
	plt.scatter(data[:,0],data[:,1],s=7)
	plt.scatter(centers_new[:,0], centers_new[:,1], marker='*',c='g', s=150)
	plt.title('Final data with centroid')
	plt.show()

def main():
	KMeans()

if __name__ == '__main__':
	main()

			
