import csv
import numpy as np
from scipy.spatial import distance
from collections import defaultdict
import random
from itertools import combinations, product
import matplotlib.pyplot as plt


def loadDataset( filename ):
	with open(filename) as csvfile:
		lines = csv.reader(csvfile)
		dataSet = []
		for row in lines:
			dataSet.append( row )
		return dataSet


def processData(dataSet):
	"""
	:param dataSet: list( list )
	:return: list( list )
	"""
	processedData = []
	for sample in dataSet:
		processedData.append( list(map(float, sample[:-1])) )
	return processedData

def computeDistance( x, y ):
	"""
	:param x: list( float )
	:param y: list( float )
	:return: distance float
	"""
	x = np.array( x )
	y = np.array( y )
	return distance.euclidean( x, y )

def computeDotProduct( x, y ):
	"""
	:param x: list( float )
	:param y: list( float )
	:return: distance float
	"""
	x = np.array(x)
	y = np.array(y)
	return np.dot( x, y )

# # test computerDistance & computeDotProduct
# dis = computeDistance([1,2], [3,4])
# dis = computeDotProduct([1,2], [3,4])
# print( dis )

def getCentroids( datasetDic ):
	"""
	:param datasetDic: dic{ k: set( samples ) }
	:return:
	"""
	centroidsDic = defaultdict( list )
	for k in datasetDic:
		kClusterSamples = np.array(datasetDic[k])
		centroidsDic[k] = kClusterSamples.mean(axis=0).tolist()
	return centroidsDic

# # test getCentroids
# datasetDic = {1: [[6, 6], [10, 10]], 2: [[1, 1], [2, 2], [4, 4]]}
# print( getCentroids(datasetDic) )

def clusterData( dataset, centroidsDic ):
	"""
	:param dataset: list( list(float) )
	:param centroidsDic:  dic( k: list( float ) )
	:param k: int
	:return: dic( k: list( list( float ) ) )
	"""
	datasetDic = defaultdict( list )
	for sample in dataset:
		# dist = [ computeDistance(sample, centroidsDic[k]) for k in centroidsDic ]
		dist = [ computeDotProduct(sample, centroidsDic[k]) for k in centroidsDic]
		i = np.array( dist ).argmin()
		datasetDic[i+1].append( sample )
	return datasetDic

# # test clusterData
# dataset = [[1,1],
#            [3,3],
#            [5,5]]
# centroids = {1:[2,2],
# 			 2:[4,4]}
# print( clusterData(dataset, centroids) )

def generateInitialCentroids( k, dataset ):
	"""
	:param k: int
	:param dataset: list( list( float ) )
	:return: dic( k: list( float ) )
	"""
	candidates = random.sample( dataset, k )
	centroidsDic = defaultdict( list )
	for i, pos in enumerate(candidates):
		centroidsDic[i+1] = pos
	return centroidsDic

# # test generateInitialCentroids
# k = 3
# dataset = [[1,1],
#            [3,3],
#            [5,5]]
# print( generateInitialCentroids(k, dataset) )

def kmeans( dataset, k ):
	"""
	:param dataset: list( list( float ) )
	:param k: int
	:return: dic( k: list( list( float ) ) )
	"""
	clusteredDataset = defaultdict(list)
	centroidsDic = generateInitialCentroids(k, dataset)
	# print("cen:", centroidsDic)
	datasetDic = clusterData(dataset, centroidsDic)
	# print(datasetDic)
	count = 1
	while count < 100:
		count += 1
		centroidsDic = getCentroids( datasetDic )
		newDatasetDic = clusterData( dataset, centroidsDic )
		if newDatasetDic == datasetDic:
			print(count)
			return centroidsDic, newDatasetDic
		datasetDic = newDatasetDic.copy()
	return centroidsDic, newDatasetDic

# # test kmeans
# loadedDataset = loadDataset("allCases200features.csv")
# dataset = processData( loadedDataset[1:] )
# centroidsDic, datasetDic = kmeans(dataset, 2)
# print(centroidsDic)
# print(datasetDic)

def countClusteredSamples( datasetDic ):
	"""
	:param datasetDic: dic( k: list( list( float ) ) )
	:return: dic( k: int )
	"""
	clusterCountsDic = defaultdict()
	for k in datasetDic:
		clusterCountsDic[k] = len( datasetDic[k] )
	return clusterCountsDic
# # test countClusteredSamples
# loadedDataset = loadDataset("allCases200features.csv")
# dataset = processData( loadedDataset[1:] )
# centroidsDic, datasetDic = kmeans(dataset, 2)
# print( countClusteredSamples( datasetDic ) )

def linkDistance( datesetDic, k ):
	"""
	:param datesetDic: dic( k: list( list( float ) ) )
	:return: single, complete, centroid, average
	"""
	dists = []
	clusterCombinations = combinations( [ datesetDic[i] for i in datesetDic ], 2 )
	for c1, c2 in clusterCombinations:
		for sample1, sample2 in product(c1, c2):
			# dists.append( computeDistance(sample1, sample2) )
			dists.append(computeDotProduct(sample1, sample2))
	minDist = min( dists )
	maxDist = max( dists )
	aveDist = np.array( dists ).mean()

	centroidsDic = getCentroids( datesetDic )
	cenDists = []
	for c1, c2 in combinations( [centroidsDic[i] for i in centroidsDic], 2 ):
		# cenDists.append( computeDistance(c1, c2) )
		cenDists.append(computeDotProduct(c1, c2))

	return minDist, maxDist, aveDist, cenDists

# # test linkDistance
# loadedDataset = loadDataset("allCases200features.csv")
# dataset = processData( loadedDataset[1:] )
# centroidsDic, datasetDic = kmeans(dataset, 3)
# print( linkDistance( datasetDic, 3 ) )


def main(  ):
	filename = "allCases200features.csv"
	k = 4
	loadedDataset = loadDataset(filename)
	dataset = processData( loadedDataset[1:] )
	centroidsDic, datasetDic = kmeans(dataset, k)
	clusterCounts = countClusteredSamples( datasetDic )
	dists = linkDistance(datasetDic, k)
	print("counts: ", clusterCounts)
	print("dists: ", dists  )

main()

