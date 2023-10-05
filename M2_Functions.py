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
import nltk
import re
from nltk.corpus import stopwords
from sklearn.svm import SVC
import os
import pickle
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.metrics import mean_squared_error
from category_encoders import *
import warnings
warnings.filterwarnings('ignore')

def fill_nullls(X):
        
    #replace missing values with mode as it is categorical
    mode_value = X['Languages'].mode()[0]
    X['Languages'].fillna(mode_value, inplace=True)
    X["In-app Purchases"].fillna("0", inplace=True)
    return X
def encoding_Xtest(X_test):
    
    df2=pd.DataFrame()
    #URL
    df2=pd.concat([df2,X_test['URL']], axis=1)
    X_test =X_test.drop('URL',axis=1)

    #subtitle
    df2=pd.concat([df2,X_test['Subtitle']], axis=1)
    X_test = X_test.drop(columns=["Subtitle"],axis=1)
    
    # In-app Purchases    
    X_test['In-app Purchases'] = X_test['In-app Purchases'].apply(lambda x: x.split(',') if isinstance(x, str) else [x])
    X_test['In-app Purchases'] = X_test['In-app Purchases'].apply(lambda x: np.sum(np.array(x).astype(float)))

    
    df2=pd.concat([df2,X_test['In-app Purchases']], axis=1)
    X_test = X_test.drop(columns=["In-app Purchases"],axis=1)
    
    # Age Rating   
    # X_test['Age Rating'] = X_test['Age Rating'].apply(lambda x :int(x.strip('+')))
    
    # X_test['Age Rating'] = X_test['Age Rating'].apply(lambda x :int(x.strip('+')))  
    X_test['Age Rating'] = X_test['Age Rating'].astype(str).apply(lambda x: x.strip('+')) 
    
    df2=pd.concat([df2,X_test['Age Rating']], axis=1)
    X_test = X_test.drop(columns=['Age Rating'],axis=1)
    
    # Create dummy variables for genres in X_train
    Languages_dummies_test = X_test["Languages"].str.get_dummies(sep=',')
    # Reorder columns of genre_dummies_test to match those of genre_dummies_train
    Languages_dummies_test = Languages_dummies_test.reindex(columns=Languages_dummies_test.columns, fill_value=0)
   
    # Concatenate the dummy variables with X_train
    
    df2= pd.concat([df2, Languages_dummies_test], axis=1)
    # Drop the original "Genres" column from X_train
    X_test = X_test.drop(columns=["Languages"])
    # Create dummy variables for genres in X_train
    genre_dummies_test = X_test["Genres"].str.get_dummies(sep=',')
    genre_dummies_test = genre_dummies_test.reindex(columns=genre_dummies_test.columns, fill_value=0)
   
    # Concatenate the dummy variables with X_train
    df2 = pd.concat([df2, genre_dummies_test], axis=1)
    # Drop the original "Genres" column from X_train
    X_test = X_test.drop(columns=["Genres"],axis=1)
    
    # Original Release Date
    X_test['Original Release Date'] = X_test['Original Release Date'].apply(lambda x : time.mktime(datetime.datetime.strptime(x, '%d/%M/%Y').timetuple()))
    df2=pd.concat([df2,X_test['Original Release Date']], axis=1)
    X_test = X_test.drop(columns=['Original Release Date'],axis=1)
    
    # Current Version Release Date
    X_test['Current Version Release Date'] = X_test['Current Version Release Date'].apply(lambda x : time.mktime(datetime.datetime.strptime(x, '%d/%M/%Y').timetuple()))
    df2=pd.concat([df2,X_test['Current Version Release Date']], axis=1)
    X_test = X_test.drop(columns=['Current Version Release Date'],axis=1)
   
    file2 = open("enconding_X.sav",'rb')
    loaded_encoder = pickle.load(file2)
    file2.close()

    X_test=loaded_encoder.transform(X_test)
    X_test=pd.concat([X_test,df2],axis=1)
    return X_test 

def encoding_Ytest(Y_test):
  file2 = open("enconding_Y.sav",'rb')
  loaded_encoder = pickle.load(file2)
  file2.close() 
  Y_test=loaded_encoder.transform(Y_test)
  return Y_test


def get_model_evaluation(Y_test, y_pred):
    conf_matrix=confusion_matrix(Y_test, y_pred)
    # print ("Confusion matrix = ")
    # print(conf_matrix)

    class_report=classification_report(Y_test,y_pred)
    # print('classification = ')
    # print(class_report)

    accuracy=accuracy_score(Y_test, y_pred)
    print('Model Accuracy = ',accuracy)
    print('---------------------------------------------------------------')



def get_pickled_RandomForest(X_test,Y_test):
      file_name_rf='random_forest_model.sav'
      print("Loading Random Forest Model")
      print("we find best accuracy when (n_estimators=100, max_depth=5, random_state=4)")
      # load the model from disk
      loaded_model = pickle.load(open(file_name_rf, 'rb'))
      y_pred = loaded_model.predict(X_test)
      get_model_evaluation(Y_test, y_pred)
 
     
def get_pickled_LogisticRegression(X_test,Y_test):
      file_name_lr='LogisticRegression_model.sav'
      print("Loading Logistic Regression Model")
      print("we find best accuracy when(C=1.0, solver='lbfgs', random_state=42)")
      # load the model from disk
      loaded_model = pickle.load(open(file_name_lr, 'rb'))
      y_pred = loaded_model.predict(X_test)
      get_model_evaluation(Y_test, y_pred)
  
      
def get_pickled_SVM(X_test,Y_test):
      file_name_svm='SVM_model.sav'
      print("Loading SVM Model")
      print("we find best accuracy when  (kernel='poly',C=4,degree=7)")
      # load the model from disk
      loaded_model = pickle.load(open(file_name_svm, 'rb'))
      y_pred = loaded_model.predict(X_test)
      get_model_evaluation(Y_test, y_pred)
  
