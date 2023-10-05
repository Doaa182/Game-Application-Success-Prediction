import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import time
import datetime
# import nltk
import re
# from nltk.corpus import stopwords
from sklearn.svm import SVC
import os
import pickle
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.metrics import mean_squared_error
from category_encoders import *
import warnings
warnings.filterwarnings('ignore')



# read csv File
dataa=pd.read_csv('games-classification-dataset.csv')

X=dataa.iloc[:,0:17]
Y=dataa.iloc[:,17:18]

import Ms2_Functions

X=Ms2_Functions.fill_nullls(X)
X=Ms2_Functions.encoding_Xtest(X)
Y=Ms2_Functions.encoding_Ytest(Y) 


####################################
X = X.drop('ID', axis=1)

X = X.drop('Icon URL', axis=1)

X = X.drop('URL', axis=1)

X = X.drop('Subtitle', axis=1)

X = X.drop('Description', axis=1)

X = X.drop('Original Release Date', axis=1)

X = X.drop('Current Version Release Date', axis=1)

X=X.drop('Name',axis=1)

Ms2_Functions.get_pickled_LogisticRegression(X,Y)
 

Ms2_Functions.get_pickled_RandomForest(X,Y)

# # def call_svm(X,Y):
Ms2_Functions.get_pickled_SVM(X,Y)
    

