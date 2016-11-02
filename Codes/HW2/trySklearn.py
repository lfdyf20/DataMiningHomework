from sklearn.cluster import  KMeans
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
from sklearn import svm
from sklearn.feature_selection import RFE

X = np.array([[1,2,1,1,12,2], [3,5,2,3,1,4],
              [2,1,1,4,5,3], [4,2,12,3,4,8]])
y = np.array([1,1,-1,-1])
# y = KMeans( n_clusters=2, random_state=0 ).fit_predict(X)
# d = KMeans( n_clusters=2, random_state=0 ).fit_transform(X)
# print(y)
# print(d)

# kmeans = KMeans( n_clusters=2, random_state=0 ).fit(X)
# print( kmeans.cluster_centers_, kmeans.labels_, )
#
# data = X.tolist()
# label = kmeans.labels_.tolist()
# print(data, label)
# svc = svm.SVC(kernel='linear')
# rfe = RFE(estimator=svc, n_features_to_select=1, step=1).fit(X,y)
# ranking = rfe.ranking_
# print(ranking)
# print(rfe.score(X,y))




svc = svm.SVC( kernel='linear' ).fit( X, y )
print(svc.coef_)


