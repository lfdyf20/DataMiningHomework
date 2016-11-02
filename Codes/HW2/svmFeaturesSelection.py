import csv
import numpy as np
from sklearn import svm
from sklearn.feature_selection import RFE
import time

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


def rankTopkFeatures( data, labels, nf):
	"""
	:param data: list( samples )
	:param labels: list( int )
	:param kernel: string
	:param nf: (number of features to select) int
	:return:
	"""
	X = data
	y = labels
	# if kernel == "linear":
	# 	svc = svm.SVC(kernel="linear")
	# else:
	# 	# svc = svm.SVC(kernel="poly", degree=2)
	start = time.time()
	print( "start" )
	svc = svm.SVC(kernel="linear")
	svcNF = RFE(estimator=svc, n_features_to_select=nf, step=1).fit(X, y)
	print( time.time() - start )
	return svcNF

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


def rankTopFeaturesPolyKernel( data, labels ):
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
		print(ind)
		start = time.time()
		Xi = removeOneFeature( X, ind )
		svc = svm.SVC(kernel='poly', degree=2).fit(Xi, y)
		print(ind, " time: ", time.time() - start)
		scores[ind] = svc.score(Xi,y)
	maxInd = np.argmin( np.array(scores) )
	return maxInd

# # test rankTopFeaturesPolyKernel
# data = [[1,2,1,1,12,2], [3,5,2,3,1,4],
#         [2,1,1,4,5,3], [4,2,12,3,4,8]]
# # data = [[1,2],
# #         [1,4],
# #         [1,9],
# #         [1,10]]
# labels = [1,1,-1,-1]
# print( rankTopFeaturesPolyKernel(data, labels) )


def generateKfeaturesData(data, labels, nf):
	"""
	:param data: np.array( matrix )
	:param labels: np.array()
	:param nf: number of features to select
	:return: list( sample ) with nf features
	"""
	kFeaturesData = []
	ct = 0
	while ct < nf:
		ct += 1
		ind = rankTopFeaturesPolyKernel( data, labels )
		kFeaturesData.append( data[ :, ind] )
		data = removeOneFeature( data, ind )
	kFeaturesData = np.array( kFeaturesData ).transpose()
	return kFeaturesData

# # test generateKfeaturesData
# data = [[1,2,1,1,12,2], [3,5,2,3,1,4],
#         [2,1,1,4,5,3], [4,2,12,3,4,8]]
# labels = [1,1,-1,-1]
# print( generateKfeaturesData(data,labels,3) )

def mainPoly():
	filename = "allCases.csv"
	nf = 1
	loadedDataset = loadDataset(filename)
	data, labels = processData(loadedDataset[1:])
	data = np.array( data )
	labels = np.array( labels )
	kFeaturesData = generateKfeaturesData(data, labels, nf)
	print(kFeaturesData)

def mainLinear():
	filename = "allCases.csv"
	nf = 1
	loadedDataset = loadDataset(filename)
	data, labels = processData(loadedDataset[1:])
	data = np.array(data)
	labels = np.array(labels)
	svmClassifier = rankTopkFeatures(data, labels, nf)

mainLinear()



