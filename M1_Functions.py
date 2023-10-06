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



def fill_null(data):
    mode_value = data['Languages'].mode()[0]
    data['Languages'].fillna(mode_value, inplace=True)

    #replace missing values with 0
    data["In-app Purchases"].fillna("0", inplace=True)
    return data

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



def lasso_reg_model(X_test_selected_lss,Y_test):
    file_name_lassoLinear='lassoLinear_model.sav'
    loaded_model_lassolinear = pickle.load(open(file_name_lassoLinear, 'rb'))
    Y_pred_selected_lss = loaded_model_lassolinear.predict(X_test_selected_lss)
    mse_selected_lasso_reg = mean_squared_error(Y_test, Y_pred_selected_lss)
    print("............................................")
    print("Selected features:", selected_features_lasso_reg)
    print("Mean squared error using selected features for Lasso linear regression: %.2f" % mse_selected_lasso_reg)
    
def ridge_model(X_test_selected_rdg,Y_test):
    file_name_ridgeLinear='ridgeLinear_model.sav'
    loaded_model_ridgelinear = pickle.load(open(file_name_ridgeLinear, 'rb'))
    Y_pred_selected_rdg = loaded_model_ridgelinear.predict(X_test_selected_rdg)
    # mean squared error
    mse_selected_ridge_reg = mean_squared_error(Y_test, Y_pred_selected_rdg)
    # print the selected features and the mean squared error
    print("............................................")
    print("Selected features:", selected_features_ridge_reg)
    print("Mean squared error using selected features for Ridge linear regression: %.2f" % mse_selected_ridge_reg)

def poly_lasso_model(X_test_poly,Y_test):
    file_name_poly_lasso='lasso_reg_model.sav'
    # Predict on the test data using the polynomial features
    loaded_model_lasso = pickle.load(open(file_name_poly_lasso, 'rb'))
    Y_pred_poly = loaded_model_lasso.predict(X_test_poly)
    # Compute the mean squared error of the predictions using the polynomial features
    mse_poly = mean_squared_error(Y_test, Y_pred_poly)
    # Print the selected features and the mean squared error
    print("............................................")
    print("Selected features:", X.columns[selector_lasso_reg.get_support()])
    print("Mean squared error using polynomial features for Lasso Polynomial regression: %.2f" % mse_poly)

def ridge_reg(X_test_poly,Y_test):
    file_name_poly='ridge_reg_model.sav'
    # Predict on the test data using the polynomial features
    loaded_model = pickle.load(open(file_name_poly, 'rb'))
    Y_pred_poly = loaded_model.predict(X_test_poly)
    # Y_pred_poly = ridge_poly.predict(X_test_poly)
    # Compute the mean squared error of the predictions using the polynomial features
    mse_poly = mean_squared_error(Y_test, Y_pred_poly)
    # Print the selected features and the mean squared error
    print("............................................")
    print("Selected features:", X.columns[selector_ridge_reg.get_support()])
    print("Mean squared error using polynomial features for Ridge Polynomial regression: %.2f" % mse_poly)


    