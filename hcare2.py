from sklearn import tree
from sklearn.metrics import accuracy_score
from scipy.spatial import distance
import scipy
from pylab import *
import numpy as np
#from sklearn.neighbors import KNeighborsClassifier

def euc(a,b):	#calculate distance
	return distance.euclidean(a,b)
	
class MyKNN(): #classifier
	def fit(self, x_train, y_train): #function to map training data
		self.x_train = x_train
		self.y_train = y_train
	def predict(self, x_test): #returns predicted output
		for row in x_test:
			label = self.closest(row)
			predictions = label
		return predictions
        def calcaccuracy(self,x_test): 
		caccuracy = []
		for row in x_test:
			label = self.closest(row)
			caccuracy.append(label)
		return caccuracy
		
	def closest(self, row):#calculation of distance between datapoints
		best_dist = euc(row, self.x_train[0])
		best_index = 0
		for i in range(1, len(self.x_train)):
			dist = euc(row, self.x_train[i])
			if dist < best_dist:
				best_dist = dist
				best_index = i
		return self.y_train[best_index]		


x = np.loadtxt("train2.csv",delimiter=",")#load data
y =np.loadtxt("label.csv",delimiter=",")#load labels

clf = MyKNN() #object of created classifier

from sklearn.cross_validation import train_test_split  #split dataset in tesing and training part
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = .3)# x_train-training data, y_train-labels, x_test-testing data, y_test

clf.fit(x_train,y_train)#map with labels in training data
cross_valid=clf.calcaccuracy(x_test)#accuracy with cross validation
clf.fit(x, y)# map data with labels
test = np.loadtxt("contacts.csv",delimiter=",")#load actual testing data (input)
test = test.reshape(1, -1)
predictions = clf.predict(test)#output class of testing data

if (predictions == 0):
 f1='0'
if (predictions == 1):
 f1='1'
#write result and accuracy to output file
fd = file('result.csv','w')
fd.write(f1)
fd.close()

print predictions
accuracy = accuracy_score(y_test, cross_valid)
accuracy = accuracy * 100
accuracy = str(accuracy)
fd1 = file('resultac.csv','w')
fd1.write(accuracy)
fd1.close()
print cross_valid
print accuracy


