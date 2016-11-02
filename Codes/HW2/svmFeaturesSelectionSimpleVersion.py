import csv
import numpy as np
from sklearn import svm
from sklearn.feature_selection import RFE
import time
from sklearn.model_selection import cross_val_score

def loadDataset( filename ):
	with open(filename) as csvfile:
		lines = csv.reader(csvfile)
		dataSet = []
		for row in lines:
			dataSet.append( row )
		return dataSet


def processData(dataSet):
	"""
	:param dataSet: list( list(float) )
	:return: list( list )
	"""
	processedData = []
	labels = []
	for sample in dataSet:
		processedData.append( list(map(float, sample[:-1])) )
		labels.append( int( sample[-1] ) )
	return processedData, labels


def filterDataWithTopFeatures(data, colNames, ranks, k):
	"""
	:param data: list( samples )
	:param colNames: list( string )
	:param ranks: list( float )
	:return: list( samples with less features )
	"""
	colRanks = [x for (y,x) in sorted(zip(ranks,colNames))]
	selectedFeatureNames = colRanks[:k]

	data = np.array( data )
	ranks = np.array( ranks )
	ind = np.argsort( ranks )
	sortedData = data[:, ind]
	dataWithSelectedFeatures = sortedData[:,:k].tolist()

	return dataWithSelectedFeatures, selectedFeatureNames

# # test selectTopFeatures
# data = [[4,5,6],
#      [1,2,3]]
# colNames = ["a", "b", "c"]
# ranks = [3,1,2]
# k = 1
# dataWithSelectedFeatures, selectedFeatureNames = filterDataWithTopFeatures(data,colNames,ranks,k)
# print(dataWithSelectedFeatures)
# print(selectedFeatureNames)


def rankTopkFeaturesLinear( data, labels, nf):
	"""
	:param data: list( samples )
	:param labels: list( int )
	:param kernel: string
	:param nf: (number of features to select) int
	:return:
	"""
	X = data
	y = labels
	m, n = X.shape
	scores = [0] * n
	for ind in range(n):
		start = time.time()
		Xi = X[:, ind].reshape(-1, 1)
		svc = svm.SVC(kernel='linear').fit(Xi, y)
		print(ind, " time:", time.time() - start)
		scores[ind] = svc.score(Xi, y)
	maxInds = np.argpartition(scores, -nf)[-nf:]
	return maxInds

# # test rankTopKFeatures
# data = [[1,2,1,1,12,2], [3,5,2,3,1,4],
#         [2,1,1,4,5,3], [4,2,12,3,4,8]]
# labels = [1,1,-1,-1]
# SVMclassifier = rankTopkFeatures( data, labels, 5 )
# print( SVMclassifier.ranking_ )
# print( SVMclassifier.score( np.array(data), np.array(labels) ) )

def removeOneFeature( data, ind ):
	"""
	remove colume int from data
	:param data: np.array( matrix )
	:param ind: int
	:return: np.array( matrix )
	"""
	data = np.delete(data,  ind , axis=1)
	return data
# # test removeOneFeature
# data = np.array( [[1,2,3],
#                   [4,5,6]] )
# ind = 2
# print( removeOneFeature(data, ind) )


def rankTopFeaturesPolyKernel( data, labels, nf ):
	"""
	find out the best feature
	:param data: list( sample )
	:param labels: list( int )
	:return: int
	"""
	X = data
	y = labels
	m, n = X.shape
	scores = [0]*n
	for ind in range(n):
		# print(ind)
		start = time.time()
		Xi = X[:, ind].reshape(-1,1)
		svc = svm.SVC(kernel='poly', degree=2).fit(Xi, y)
		print(ind, " time:", time.time() - start)
		scores[ind] = svc.score(Xi,y)
	maxInds = np.argpartition(scores, -nf)[-nf:]
	return maxInds
# # test rankTopFeaturesPolyKernel
# data = np.array([[1,2,1,1,12,2], [3,5,2,3,1,4],
#         [2,1,1,4,5,3], [4,2,12,3,4,8]])
# # data = [[1,2],
# #         [1,4],
# #         [1,9],
# #         [1,10]]
# labels = np.array([1,1,-1,-1])
# print( rankTopFeaturesPolyKernel(data, labels, 3) )


def generateKfeaturesData(data,  inds):
	"""
	:param data: np.array( matrix )
	:param labels: np.array()
	:param nf: number of features to select
	:return: list( sample ) with nf features
	"""

	data = data[:, inds]
	return data

# test generateKfeaturesData
# data = np.array([[1,2,1,1,12,2], [3,5,2,3,1,4],
#         [2,1,1,4,5,3], [4,2,12,3,4,8]])
# labels = np.array([1,1,-1,-1])
# print( generateKfeaturesData(data,labels,3) )


def runPolySVMSelector( data, labels, nf ):
	inds = rankTopFeaturesPolyKernel(data, labels, nf)
	kFeaturesData = generateKfeaturesData(data, inds)
	return kFeaturesData


def runLinearSVMSelector( data, labels, nf):
	inds = rankTopkFeaturesLinear(data, labels, nf)
	kFeaturesData = generateKfeaturesData(data, inds)
	return kFeaturesData


def evaluateFeatures( data, labels ):
	X = data
	y = labels
	svcLinear = svm.SVC(kernel='linear')
	scoresLinear = cross_val_score(svcLinear, X, y, cv=5)
	svcPoly = svm.SVC(kernel='poly', degree=2)
	scoresPoly = cross_val_score(svcPoly, X, y, cv=5)
	return scoresLinear, scoresPoly


def linearFeaturesSelection( data, labels, nf ):
	X = data
	y = labels
	svcLinear = svm.SVC(kernel='linear').fit(X, y)
	weights = svcLinear.coef_
	weights = np.absolute(weights)
	weights = weights[0]
	maxInds = np.argpartition(weights, -nf)[-nf:]
	kFeaturesData = generateKfeaturesData( data, maxInds )
	return kFeaturesData

"""
---------------------------------------------------------------
"""

def main():
	filename = "allCases.csv"
	nf = 200
	loadedDataset = loadDataset(filename)
	data, labels = processData(loadedDataset[1:])
	data = np.array( data )
	labels = np.array( labels )
	kFeaturesData_L = runLinearSVMSelector( data, labels, nf )
	kFeaturesData_P = runPolySVMSelector( data, labels, nf )
	eval0 = evaluateFeatures( data, labels )
	eval1 = evaluateFeatures( kFeaturesData_L, labels )
	eval2 = evaluateFeatures( kFeaturesData_P, labels )
	print( eval0 )
	print( eval1 )
	print( eval2 )

# main()

def singleLinearMain():
	filename = "allCases.csv"
	nf = 200
	loadedDataset = loadDataset(filename)
	data, labels = processData(loadedDataset[1:])
	data = np.array(data)
	labels = np.array(labels)
	kFeaturesData = linearFeaturesSelection( data, labels, nf )
	eval = evaluateFeatures( kFeaturesData, labels )
	print( eval )

# singleLinearMain()


