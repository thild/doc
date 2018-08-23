# gensim modules
from gensim import utils
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec
from sources import sources

import string

# numpy
import numpy

# shuffle
from random import shuffle

# logging
import logging
import os.path
import sys
import _pickle as pickle


log = logging.getLogger()
log.setLevel(logging.INFO)

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
log.addHandler(ch)

class LabeledLineSentence(object):

    
    def __init__(self, sources):
        self.sources = sources

        flipped = {}

        # make sure that keys are unique
        for key, value in sources.items():
            if value not in flipped:
                flipped[value] = [key]
            else:
                raise Exception('Non-unique prefix encountered')

    def __iter__(self):
        for source, prefix in self.sources.items():
            with utils.smart_open('data/' + source) as fin:
                for item_no, line in enumerate(fin):
                    yield TaggedDocument(utils.to_unicode(line).split(), [prefix + '_%s' % item_no])

    def to_array(self):
        self.sentences = []
        for source, prefix in self.sources.items():
            with utils.smart_open('data/' + source) as fin:
                for item_no, line in enumerate(fin):
                    self.sentences.append(TaggedDocument(
                        utils.to_unicode(line).split(), [prefix + '_%s' % item_no]))
        return self.sentences

    def sentences_perm(self):
        shuffle(self.sentences)
        return self.sentences

from pathlib import Path

if not Path("./imdb.d2v").is_file():
    # file exists 
    sentences = LabeledLineSentence(sources)
    model = Doc2Vec(min_count=1, window=10, size=100, sample=1e-4, negative=5, workers=16)
    model.build_vocab(sentences.to_array())

    for epoch in range(20):
        log.info('Epoch %d' % epoch)
        model.train(sentences.sentences_perm(),
                    total_examples=model.corpus_count,
                    epochs=model.iter,
        )

    model.save('./imdb.d2v')

model = Doc2Vec.load('./imdb.d2v')

log.info('Sentiment')
train_size = 100
half_train_size = int(train_size/ 2 )

train_arrays = numpy.zeros((train_size, 100))
train_labels = numpy.zeros(train_size)


for i in range(half_train_size):
    prefix_train_pos = 'TRAIN_POS_' + str(i)
    prefix_train_neg = 'TRAIN_NEG_' + str(i)
    train_arrays[i] = model.docvecs[prefix_train_pos]
    train_arrays[half_train_size + i] = model.docvecs[prefix_train_neg]
    train_labels[i] = 1
    train_labels[half_train_size + i] = 0

test_size = 100
half_test_size = int(test_size/ 2 )

test_arrays = numpy.zeros((test_size, 100))
test_labels = numpy.zeros(test_size)

for i in range(half_test_size):
    prefix_test_pos = 'TEST_POS_' + str(i)
    prefix_test_neg = 'TEST_NEG_' + str(i)
    test_arrays[i] = model.docvecs[prefix_test_pos]
    test_arrays[half_test_size + i] = model.docvecs[prefix_test_neg]
    test_labels[i] = 1
    test_labels[half_test_size + i] = 0

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.datasets import load_svmlight_file
from sklearn import preprocessing
import pylab as pl
from sklearn.metrics import classification_report

from knn_naive import KnnClassifier
knn = KnnClassifier(9, 2)
knn.fit(train_arrays, train_labels)
pred = knn.predict(test_arrays)
print (knn.score(test_arrays, test_labels))
print (knn.confusion_matrix(test_labels, pred))

# cria um kNN
neigh = KNeighborsClassifier(n_neighbors=9, metric='euclidean')

print ('Fitting knn...')
neigh.fit(train_arrays, train_labels)

# predicao do classificador
print ('Predicting...')
y_pred = neigh.predict(test_arrays)

# mostra o resultado do classificador na base de teste
print ('Score:')
print (neigh.score(test_arrays, test_labels))

# cria a matriz de confusao
print ('Confusion matrix:')
cm = confusion_matrix(test_labels, y_pred)
print (cm)
# print ('Report:')
# print (classification_report(test_labels, y_pred))
# pl.matshow(cm)
# pl.colorbar()
# pl.show()
