import pandas as pd
import numpy as np
import math
import pickle
from sklearn.pipeline import Pipeline, FeatureUnion, make_union, make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer 
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.preprocessing import RobustScaler, Normalizer
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.dummy import DummyClassifier
from nltk import wordpunct_tokenize
from nltk.corpus import stopwords
from sklearn.externals import joblib

class ItemSelector(BaseEstimator, TransformerMixin):

    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, data_dict):
        return data_dict[self.key]
    
class ArrayCaster(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self

    def transform(self, data):
        print data.shape
        print np.transpose(np.matrix(data)).shape
        return np.transpose(np.matrix(data))
    