import pandas as pd
import copy
import numpy as np

def logistic_regression(data, labels, weights, num_epochs, learning_rate): # do not change the heading of the function

    data = np.insert(data, 0, np.ones(data.shape[0]), axis=-1)

    for i in range(num_epochs):

    	h_t = (1/(1+np.exp(-np.dot(data,weights))))
    	a = h_t-labels
    	#a1 = np.ones(6000)
    	b = data.transpose()
    	d=(a*b)
    	#d = np.dot(a,b)
    	#a3 = np.stack((a,a1,a1),axis=-1)
    	#c = np.dot(b,a3)
    	
    	#c = a3.transpose()*b
    	
    	c = np.sum(d.transpose(),axis=0)
    	print(a)
    	weights = weights-learning_rate * c
    	break
    	
    return weights

data_file='a.csv'
raw_data = pd.read_csv(data_file, sep=',')
#print(raw_data.head())
raw_data = pd.read_csv(data_file, sep=',')
labels=raw_data['Label'].values
data=np.stack((raw_data['Col1'].values,raw_data['Col2'].values), axis=-1)
weights = np.zeros(3) # We compute the weight for the intercept as well...
num_epochs = 50000
learning_rate = 50e-5

coefficients=logistic_regression(data, labels, weights, num_epochs, learning_rate)
print(coefficients)
