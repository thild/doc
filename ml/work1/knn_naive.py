#!/usr/bin/python

import sys
import math
import random
from sklearn.metrics.pairwise import euclidean_distances
import pandas as pd
import numpy as np
import operator
from functools import reduce

class KnnClassifier(object):
    model = []
    numclasses = 0

    def __init__(self, k, numclasses):
        self.k = k
        self.numclasses = numclasses
    
    # Euclidean distance between two points
    def euclidean_distance(self, data1, data2):
        distance = 0
        for x in range(len(data1)):
            distance += np.square(data1[x] - data2[x])
        return np.sqrt(distance)

    # Model fit
    def fit(self, trainset, labelset):
        self.model = list(zip(labelset, trainset))

    # Predict with test set
    def predict(self, testset):
        distances = {}
        # Calculating euclidean distance and sorting and extracting k closest neighbors
        neighbors = {}
        for i, te in enumerate(testset):
            distances[i] = []
            for me in self.model:
                dist = self.euclidean_distance(te, me[1])
                distances[i].append((me[0], dist))
            neighbors[i] = sorted(distances[i], key=lambda tup: tup[1])[0:self.k]
        
        classVotes = {}
        
        # Voting for most frequency neighbors
        for k, v in neighbors.items():
            classVotes[k] = []
            for i in range(0, self.numclasses):
                classVotes[k].append((i, reduce((lambda x, y: x + 1), (x[0] for x in v if x[0] == i), 0)))
            classVotes[k] = sorted(classVotes[k], key=lambda tup: tup[1], reverse=True)[0]

        return classVotes
        
    def score(self, testset, labelset):
        pred = [x[0] for x in self.predict(testset).values()]
        matchs = reduce((lambda x, y: x + 1), (x for x in zip(pred, labelset) if x[0] == x[1]), 0)
        return matchs / len(labelset)

    def confusion_matrix(self, labelset, predset):
        import pandas as pd
        pred = [x[0] for x in predset.values()]
        y_actu = pd.Series(labelset, name='Actual')
        y_pred = pd.Series(pred, name='Predicted')
        df_confusion = pd.crosstab(y_actu, y_pred, rownames=['Actual'], colnames=['Predicted'], margins=True)
        return df_confusion / df_confusion.sum(axis=1)
    