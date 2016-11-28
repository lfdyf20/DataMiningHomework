import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
from sklearn import preprocessing


def readData( filename ):
	return pd.read_csv( filename )

# test readData
# filename = "samples.csv"
filename = "allCases.csv"
df = readData( filename )
# print(df)

def preprocessData( df ):
	df["Classes"] = df["Classes"].replace([2,1], [1,0])
	data = df.values
	return data

# test preprocessData
data = preprocessData( df )
# print(data)

def normalizeData( data ):
	return preprocessing.normalize(data, norm='l2')
# test normalizeData
data, labels = data[:, :-1], data[:,-1]
data = normalizeData( data )
print(data)


def trainHCBU( data, label ):
	X, y = data, label
	model = AgglomerativeClustering(linkage="average", affinity="cosine")
	predictions = model.fit_predict( X )
	return predictions

# test trainHCBU
predictions = trainHCBU( data, labels )
print("labels:      ",labels.astype(int))
print("predictions: ",np.array(predictions))


def getTwoClusterData( data,labels, predictions ):
	diff = ( labels==predictions ).astype(int)
	print(diff)
	return sum(diff)

# test getTwoClusterData
sumdiff = getTwoClusterData(data,labels, predictions)
print( sumdiff )
print( data.shape[0] )


# def main():
# 	filename


