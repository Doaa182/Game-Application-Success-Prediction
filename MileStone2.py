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
###############################################################################
###############################################################################

#[1] import data
data = pd.read_csv('games-classification-dataset.csv')

###############################################################################
###############################################################################

#[2] rename ID col
data = data.rename(columns = {"ID":"id"})


#######SPLITTING#######

X = data.iloc[:, 0:17]
Y=data['Rate']

# Split the data into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y,  test_size = 0.20,shuffle=True,random_state=10)


###############################################################################
###############################################################################
#[3] Cleaning data
#[3.1] drop Dublicate rows

DuplicateNumber=X_train.duplicated().sum()
#print(DuplicateNumber)
duplicates = X_train[X_train.duplicated(keep=False)]
duplicate_indices = duplicates.index
# print(duplicate_indices)
X_train=X_train.drop_duplicates()
indices_to_remove=[1479, 4467, 3156, 3423, 4847, 2967,
               776, 4563, 1606, 4245,  323,
            4038, 3595, 3266,  847, 3398, 2299,
             319, 1474, 2921, 4325, 3394
             , 1415, 1222,  314, 4442]

# DuplicateNumber=X_train.duplicated().sum()
# print(DuplicateNumber)
Y_train = Y_train.drop(indices_to_remove)

######################
######################
#[3.2] cleaning missing values
#there is null or not
NullNumber=X_train.isnull().sum()
NullNumber=X_test.isnull().sum()
# NullNumber=data.isnull().sum()
# print(NullNumber)

#replace missing values with mode as it is categorical
mode_value = X_train['Languages'].mode()[0]
X_train['Languages'].fillna(mode_value, inplace=True)
X_test['Languages'].fillna(mode_value, inplace=True)
# mode_value = data['Languages'].mode()[0]
# data['Languages'].fillna(mode_value, inplace=True)

#replace missing values with 0
X_train["In-app Purchases"].fillna("0", inplace=True)
X_test["In-app Purchases"].fillna("0", inplace=True)
# data["In-app Purchases"].fillna("0", inplace=True)

#there is null or not
NullNumber=X_train.isnull().sum()
NullNumber=X_test.isnull().sum()
# NullNumber=X_test.isnull().sum()
# print(NullNumber)


###############################################################################
##############################ENCODING#########################################

df = pd.DataFrame() 
data2 = pd.DataFrame() 
#URL
df=pd.concat([df,X_train['URL']], axis=1)
data2=pd.concat([data2,X['URL']], axis=1)
X_train = X_train.drop(columns=["URL"])
X = X.drop(columns=["URL"])

#subtitle
df=pd.concat([df,X_train['Subtitle']], axis=1)
X_train = X_train.drop(columns=["Subtitle"])
data2=pd.concat([data2,X['Subtitle']], axis=1)
X = X.drop(columns=["Subtitle"])

# In-app Purchases    
X_train['In-app Purchases'] = X_train['In-app Purchases'].apply(lambda x: x.split(','))
X_train['In-app Purchases'] = X_train['In-app Purchases'].apply(lambda x: np.sum(np.array(x).astype(float)))

df=pd.concat([df,X_train['In-app Purchases']], axis=1)
X_train = X_train.drop(columns=["In-app Purchases"])
data2=pd.concat([data2,X['In-app Purchases']], axis=1)
X = X.drop(columns=["In-app Purchases"])


# Age Rating   
X_train['Age Rating']= X_train['Age Rating'].apply(lambda x :int(x.strip('+')))       
df=pd.concat([df,X_train['Age Rating']], axis=1)
X_train = X_train.drop(columns=["Age Rating"])
data2=pd.concat([data2,X['Age Rating']], axis=1)
X = X.drop(columns=["Age Rating"])



# Create dummy variables for genres in X_train
Languages_dummies_train = X_train["Languages"].str.get_dummies(sep=',')
# Concatenate the dummy variables with X_train
df= pd.concat([df, Languages_dummies_train], axis=1)
# # Drop the original "Genres" column from X_train
X_train = X_train.drop(columns=["Languages"])
data2=pd.concat([data2,X['Languages']], axis=1)
X = X.drop(columns=["Languages"])



# Create dummy variables for genres in X_train
genre_dummies_train = X_train["Genres"].str.get_dummies(sep=',')
# Concatenate the dummy variables with X_train
df= pd.concat([df, genre_dummies_train], axis=1)
# # Drop the original "Genres" column from X_train
X_train = X_train.drop(columns=["Genres"])
data2=pd.concat([data2,X['Genres']], axis=1)
X = X.drop(columns=["Genres"])

# Original Release Date
X_train['Original Release Date']= X_train['Original Release Date'].apply(lambda x : time.mktime(datetime.datetime.strptime(x, '%d/%M/%Y').timetuple()))
df=pd.concat([df,X_train['Original Release Date']], axis=1)
X_train = X_train.drop(columns=["Original Release Date"])
data2=pd.concat([data2,X['Original Release Date']], axis=1)
X = X.drop(columns=["Original Release Date"])

# Current Version Release Date
X_train['Current Version Release Date'] = X_train['Current Version Release Date'].apply(lambda x : time.mktime(datetime.datetime.strptime(x, '%d/%M/%Y').timetuple()))
df=pd.concat([df,X_train['Current Version Release Date']], axis=1)
X_train = X_train.drop(columns=["Current Version Release Date"])
data2=pd.concat([data2,X['Current Version Release Date']], axis=1)
X = X.drop(columns=["Current Version Release Date"])

encoder=OrdinalEncoder(handle_unknown='ignore')
# label_encoder=LabelEncoder()

encoder.fit(X)
X_train=encoder.transform(X_train)
X_train=pd.concat([X_train,df],axis=1)
# data=pd.concat([data,data2],axis=1)

# dump X_encoder by pickle
file_X = open("enconding_X.sav","wb")
pickle.dump(encoder,file_X)
file_X.close()

###########

encoderY=OrdinalEncoder(handle_unknown='ignore') 
encoderY.fit(Y_train)
Y_train=encoderY.transform(Y_train)

# dump Y_encoder by pickle
file_Y = open("enconding_Y.sav","wb")
pickle.dump(encoderY,file_Y)
file_Y.close()

# load encoder by pickle to apply it on test 

def encoding_Xtest(X_test):
    
    df2=pd.DataFrame()
    #URL
    df2=pd.concat([df2,X_test['URL']], axis=1)
    X_test =X_test.drop(columns=["URL"])

    #subtitle
    df2=pd.concat([df2,X_test['Subtitle']], axis=1)
    X_test = X_test.drop(columns=["Subtitle"])
    
    # In-app Purchases    
    # X_test['In-app Purchases'] = X_test['In-app Purchases'].apply(lambda x: x.split(','))
    # X_test['In-app Purchases'] = X_test['In-app Purchases'].apply(lambda x: np.sum(np.array(x).astype(float)))
    
    
    
    X_test['In-app Purchases'] = X_test['In-app Purchases'].apply(lambda x: x.split(',') if isinstance(x, str) else [x])
    X_test['In-app Purchases'] = X_test['In-app Purchases'].apply(lambda x: np.sum(np.array(x).astype(float)))

    
    df2=pd.concat([df2,X_test['In-app Purchases']], axis=1)
    X_test = X_test.drop(columns=["In-app Purchases"])
    
    # Age Rating   
    # X_test['Age Rating'] = X_test['Age Rating'].apply(lambda x :int(x.strip('+')))
    
    # X_test['Age Rating'] = X_test['Age Rating'].apply(lambda x :int(x.strip('+')))  
    X_test['Age Rating'] = X_test['Age Rating'].astype(str).apply(lambda x: x.strip('+')) 
    
    df2=pd.concat([df2,X_test['Age Rating']], axis=1)
    X_test = X_test.drop(columns=['Age Rating'])
    
    # Create dummy variables for genres in X_train
    Languages_dummies_test = X_test["Languages"].str.get_dummies(sep=',')
    # Reorder columns of genre_dummies_test to match those of genre_dummies_train
    Languages_dummies_test = Languages_dummies_test.reindex(columns=Languages_dummies_train.columns, fill_value=0)
   
    # Concatenate the dummy variables with X_train
    
    df2= pd.concat([df2, Languages_dummies_test], axis=1)
    # Drop the original "Genres" column from X_train
    X_test = X_test.drop(columns=["Languages"])
    # Create dummy variables for genres in X_train
    genre_dummies_test = X_test["Genres"].str.get_dummies(sep=',')
    genre_dummies_test = genre_dummies_test.reindex(columns=genre_dummies_train.columns, fill_value=0)
   
    # Concatenate the dummy variables with X_train
    df2 = pd.concat([df2, genre_dummies_test], axis=1)
    # Drop the original "Genres" column from X_train
    X_test = X_test.drop(columns=["Genres"])
    
    # Original Release Date
    X_test['Original Release Date'] = X_test['Original Release Date'].apply(lambda x : time.mktime(datetime.datetime.strptime(x, '%d/%M/%Y').timetuple()))
    df2=pd.concat([df2,X_test['Original Release Date']], axis=1)
    X_test = X_test.drop(columns=['Original Release Date'])
    
    # Current Version Release Date
    X_test['Current Version Release Date'] = X_test['Current Version Release Date'].apply(lambda x : time.mktime(datetime.datetime.strptime(x, '%d/%M/%Y').timetuple()))
    df2=pd.concat([df2,X_test['Current Version Release Date']], axis=1)
    X_test = X_test.drop(columns=['Current Version Release Date'])
    
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

X_test = encoding_Xtest(X_test)
Y_test=encoding_Ytest(Y_test)
NullNumber=X_test.isnull().sum()
print(NullNumber)

###############################################################################
###########################URL & SUBTITLE COLUMNS##############################

# split the URL column into separate columns based on the forward slash character
url_parts = X_train['URL'].str.rsplit('/', n=-1, expand=True)

# concatenate the original DataFrame with the new columns
data_after_sep = pd.concat([X_train, url_parts], axis=1)

# drop all columns in url_parts except the last two columns
url_parts_cols_to_keep = url_parts.columns[-2:]
url_parts = url_parts.loc[:, url_parts_cols_to_keep]

url_parts_to_drop = set(url_parts.columns) - set(url_parts_cols_to_keep)
url_parts = url_parts.drop(columns=url_parts_to_drop)
url_parts = url_parts.drop(columns=url_parts_to_drop)

# concatenate the original DataFrame with the remaining columns in url_parts
X_train = pd.concat([X_train, url_parts], axis=1)

####################################
####################################

# Subtitle has 3749 Nulls
# fill missing values in subtitles column with values from name column
X_train['Subtitle'] = X_train['Subtitle'].fillna(X_train[5])
X_train['Subtitle'] = X_train['Subtitle'].str.lower()

# function that removes  punctuation and special characters using regular expressions
def remove_punctuation(text):
    pattern = r'[^\w\s]'
    filtered_text = re.sub(pattern, ' ', text)
    return filtered_text

# remove punctuation and special characters from title, genres
X_train['Subtitle'] = X_train['Subtitle'].apply(remove_punctuation)

###############################################################################
###########################URL & SUBTITLE COLUMNS (for Test)###################

# split the URL column into separate columns based on the forward slash character
url_parts = X_test['URL'].str.rsplit('/', n=-1, expand=True)

# concatenate the original DataFrame with the new columns
data_after_sep = pd.concat([X_test, url_parts], axis=1)

# drop all columns in url_parts except the last two columns
url_parts_cols_to_keep = url_parts.columns[-2:]
url_parts = url_parts.loc[:, url_parts_cols_to_keep]

url_parts_to_drop = set(url_parts.columns) - set(url_parts_cols_to_keep)
url_parts = url_parts.drop(columns=url_parts_to_drop)
url_parts = url_parts.drop(columns=url_parts_to_drop)

# concatenate the original DataFrame with the remaining columns in url_parts
X_test = pd.concat([X_test, url_parts], axis=1)

####################################
####################################

# Subtitle has 3749 Nulls
# fill missing values in subtitles column with values from name column
X_test['Subtitle'] = X_test['Subtitle'].fillna(X_test[5])
X_test['Subtitle'] = X_test['Subtitle'].str.lower()

# function that removes  punctuation and special characters using regular expressions
def remove_punctuation(text):
    pattern = r'[^\w\s]'
    filtered_text = re.sub(pattern, ' ', text)
    return filtered_text

# remove punctuation and special characters from title, genres
X_test['Subtitle'] = X_test['Subtitle'].apply(remove_punctuation)

###############################################################################
#############################DESCRIPTION COLUMN################################

# # Lowercase the description column
# X_train['Description'] = X_train['Description'].str.lower()
# X_test['Description'] = X_test['Description'].str.lower()

# ####################################
# ####################################

# # Define a function to remove stopwords using NLTK
# stopwords = set(stopwords.words('english'))
# def remove_stopwords(text):
#     tokens = nltk.word_tokenize(text)
#     filtered_tokens = [token for token in tokens if token.lower() not in stopwords]
#     filtered_text = ' '.join(filtered_tokens)
#     return filtered_text

# # Remove stopwords from the description column
# X_train['Description'] = X_train['Description'].apply(remove_stopwords)
# X_test['Description'] = X_test['Description'].apply(remove_stopwords)

# ####################################
# ####################################

# # Define a function to remove punctuation and special characters using regular expressions
# def remove_punctuation(text):
#     pattern = r'[^\w\s]'
#     filtered_text = re.sub(pattern, '', text)
#     return filtered_text

# # Remove punctuation and special characters from the description column
# X_train['Description'] = X_train['Description'].apply(remove_punctuation)
# X_test['Description'] = X_test['Description'].apply(remove_punctuation)

# ####################################
# ####################################

# # Define a function to remove numerical values using regular expressions
# def remove_numbers(text):
#     pattern = r'\d+'
#     filtered_text = re.sub(pattern, '', text)
#     return filtered_text

# # Remove numerical values from the description column
# X_train['Description'] = X_train['Description'].apply(remove_numbers)
# X_test['Description'] = X_test['Description'].apply(remove_numbers)

# ####################################
# ####################################

# # Define a tokenizer function using NLTK
# def tokenize_text(text):
#     return nltk.word_tokenize(text)

# # Tokenize the description column
# X_train['Description'] = X_train['Description'].apply(tokenize_text)
# X_test['Description'] = X_test['Description'].apply(tokenize_text)

###############################################################################
###############################################################################



X_train = X_train.drop('id', axis=1)
X_test = X_test.drop('id', axis=1)

# X_train = X_train.drop('Name', axis=1)
# X_test = X_test.drop('Name', axis=1)

X_train = X_train.drop('Icon URL', axis=1)
X_test = X_test.drop('Icon URL', axis=1)

# X_train = X_train.drop('Developer', axis=1)
# X_test = X_test.drop('Developer', axis=1)

X_train = X_train.drop('URL', axis=1)
X_test = X_test.drop('URL', axis=1)

X_train = X_train.drop('Subtitle', axis=1)
X_test = X_test.drop('Subtitle', axis=1)

X_train = X_train.drop('Description', axis=1)
X_test = X_test.drop('Description', axis=1)

X_train = X_train.drop('Original Release Date', axis=1)
X_test = X_test.drop('Original Release Date', axis=1)

X_train = X_train.drop('Current Version Release Date', axis=1)
X_test = X_test.drop('Current Version Release Date', axis=1)

X_train = X_train.drop(5, axis=1)
X_test = X_test.drop(5, axis=1)

X_train = X_train.drop(6, axis=1)
X_test = X_test.drop(6, axis=1)

###############################################################################
############################NUMERIC COLUMNS####################################

numerical_cols=X_train.select_dtypes(include=np.number)
#print(numerical_cols.mean())

#drop all non numeric columns and store only numeric cols 
data_numeric=X_train.applymap(lambda x: pd.to_numeric(x, errors='coerce')).dropna(axis=1)

numerical_col_names=data_numeric.columns.tolist()
# print(numerical_col_names)

####################################
####################################
numerical_cols3=data.select_dtypes(include=np.number)
#print(numerical_cols.mean())

#drop all non numeric columns and store only numeric cols 
data_numeric3=data.applymap(lambda x: pd.to_numeric(x, errors='coerce')).dropna(axis=1)

numerical_col_names3=data_numeric3.columns.tolist()
# print(numerical_col_names)

####################################
####################################

numerical_cols2=X_test.select_dtypes(include=np.number)
#print(numerical_cols2.mean())

#drop all non numeric columns and store only numeric cols 
data_numeric2=X_test.applymap(lambda x: pd.to_numeric(x, errors='coerce')).dropna(axis=1)

numerical_col_names2=data_numeric2.columns.tolist()
# print(numerical_col_names2)

##############################################################################
###########################OUTLIERS###########################################

#check shape dataframe
# print("Old Shape: " ,X_train.shape)

# function returns list of index of outliers
def detect_outliers(df, ft):
    Q1 = X_train[ft].quantile(0.25)
    Q3 = X_train[ft].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outlier_lst = df.index[(df[ft] < lower_bound) | (df[ft] > upper_bound)]
    return outlier_lst

# create list to store the output indices from multiple col
outlier_indices_list = []

for feature in numerical_col_names:
    outlier_indices_list.extend(detect_outliers(X_train , feature))
#print(outlier_indices_list)

# function to return the cleaned dataframe
def remove_outlier(df , lst):
    lst = sorted(set(lst))
    df = df.drop(lst)
    return df


data_without_outliers = remove_outlier(X_train ,outlier_indices_list)
# print("New Shape: ", data_without_outliers.shape)

####################################
####################################

#save the cleaned dataset to new csv file
data_without_outliers.to_csv('games-regression-dataset_cleaned.csv' , index = False)

###############################################################################
#################################SCALING#######################################
 
# # Scale the data using StandardScaler
# X_train =StandardScaler().fit_transform(X_train)
# X_test = StandardScaler().fit_transform(X_test)
# # X = StandardScaler().fit_transform(X)

# Scale the data using Normalization
X_train =MinMaxScaler().fit_transform(X_train)
X_test = MinMaxScaler().fit_transform(X_test)
# X = MinMaxScaler().fit_transform(X)

# NullNumber=X_train.isnull().sum()
# print('-------',NullNumber)
###############################################################################
############################FILTER(CORRELATION)################################

#all cols except the last col
X_train_df = pd.DataFrame(X_train)
Y_train_df = pd.DataFrame(Y_train, columns=['Rate'])

v3 = pd.concat([X_train_df, Y_train_df], axis=1)
cols_cor = v3

####################################
####################################

#Feature Selection (FILTER(CORRELATION))
#correlation between features
cor = cols_cor.corr()


#Top Correlation training features  is very weak approx equals 0
top_feature_cor = cor.index[abs(cor['Rate'])>0.07]


#Correlation plot
plt.subplots(figsize=(12, 8))
top_corr2 = cols_cor[top_feature_cor].corr()
sns.heatmap(top_corr2, annot=True)
plt.show()
top_feature_cor = top_feature_cor.delete(-1)
selected_features_cor = cols_cor[top_feature_cor]


###############################################################################
######################## Function to get evaluation of applied Model###########
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

###############################################################################
###################RANDOM FOREST MODEL (CLASSIFICATION)########################

#*********** Change Hyper Paramters of RANDOM FOREST  **************#

# random_forest=RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
# random_forest.fit(X_train, Y_train)
# y_pred = random_forest.predict(X_test)
# print('At Random Forest when n_estimators=100, max_depth=5, random_state=42  ')
# get_model_evaluation(Y_test, y_pred)

# random_forest2=RandomForestClassifier(n_estimators=150, max_depth=7, random_state=46)
# random_forest2.fit(X_train, Y_train)
# y_pred = random_forest2.predict(X_test)
# print('At Random Forest when n_estimators=150, max_depth=7, random_state=46')
# get_model_evaluation(Y_test, y_pred)

# random_forest3=RandomForestClassifier(n_estimators=200, max_depth=9, random_state=50)
# random_forest3.fit(X_train, Y_train)
# y_pred = random_forest3.predict(X_test)
# print('At Random Forest when n_estimators=200, max_depth=9, random_state=50')
# get_model_evaluation(Y_test, y_pred)


file_name_rf='random_forest_model.sav'
if os.path.exists('random_forest_model.sav'):
      print("Loading Random Forest Model")
      print("we find best accuracy when (n_estimators=100, max_depth=5, random_state=4)")
      # load the model from disk
      loaded_model = pickle.load(open(file_name_rf, 'rb'))
      y_pred = loaded_model.predict(X_test)
      get_model_evaluation(Y_test, y_pred)
else:
     print("Creating and a new Random Forset model")
     # save the model
     #numOf trees --> n_estimators=100
     random_forest=RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
     random_forest.fit(X_train, Y_train)
     pickle.dump(random_forest, open(file_name_rf, 'wb'))
     # load the model from disk
     loaded_model = pickle.load(open(file_name_rf, 'rb'))
     y_pred = loaded_model.predict(X_test)
     get_model_evaluation(Y_test, y_pred)
   

# ###############################################################################
# ###################LOGISTIC REGRESSION MODEL (CLASSIFICATION)##################

#*********** Change Hyper Paramters ofLOGISTIC REGRESSION  **************#

# lr =LogisticRegression(C=1.0, solver='lbfgs', random_state=42)
# lr.fit(X_train, Y_train)
# y_pred = lr.predict(X_test)
# print("Logistic Regression when (C=1.0, solver='lbfgs', random_state=42)")
# get_model_evaluation(Y_test, y_pred)


# lr1=LogisticRegression (C=2.0, solver='lbfgs', random_state=44)
# lr1.fit(X_train, Y_train)
# y_pred = lr1.predict(X_test)
# print("Logistic Regression when (C=2.0, solver='lbfgs', random_state=44)")
# get_model_evaluation(Y_test, y_pred)


# lr2=LogisticRegression(C=3.0, solver='lbfgs', random_state=45)
# lr2.fit(X_train, Y_train)
# y_pred = lr2.predict(X_test)
# print("Logistic Regression when (C=3.0, solver='lbfgs', random_state=45)")
# get_model_evaluation(Y_test, y_pred)



file_name_lr='LogisticRegression_model.sav'
if os.path.exists('LogisticRegression_model.sav'):
      print("Loading Logistic Regression Model")
      print("we find best accuracy when(C=1.0, solver='lbfgs', random_state=42)")
      # load the model from disk
      loaded_model = pickle.load(open(file_name_lr, 'rb'))
      y_pred = loaded_model.predict(X_test)
      get_model_evaluation(Y_test, y_pred)
else :
      print("Creating and a new Logistic Regression model")
      # save the model
      lr =LogisticRegression(C=1.0, solver='lbfgs', random_state=42)
      lr.fit(X_train, Y_train)
      pickle.dump(lr, open(file_name_lr, 'wb'))
      # load the model from disk
      loaded_model = pickle.load(open(file_name_lr, 'rb'))
      y_pred = loaded_model.predict(X_test)
      get_model_evaluation(Y_test, y_pred)


# ###############################################################################
# #############SUPPORT VECTOR MACHINE (SVM) MODEL (CLASSIFICATION)###############

#*********** Change Hyper Paramters of SVM **************#

# SVCModel = SVC(kernel='poly',C=2,degree= 5)
# SVCModel.fit(X_train, Y_train)
# y_pred = SVCModel.predict(X_test)
# print("SVC when (kernel='poly',C=2,degree= 5)")
# get_model_evaluation(Y_test, y_pred)


# SVCModel1= SVC(kernel='poly',C=3,degree=6)
# SVCModel1.fit(X_train, Y_train)
# y_pred = SVCModel1.predict(X_test)
# print("SVC when (kernel='poly',C=3,degree= 6)")
# get_model_evaluation(Y_test, y_pred)


# SVCModel = SVC(kernel='poly',C=4,degree=7)
# SVCModel.fit(X_train, Y_train)
# y_pred = SVCModel.predict(X_test)
# print("SVC when (kernel='poly',C=4,degree=7)")
# get_model_evaluation(Y_test, y_pred)



file_name_svm='SVM_model.sav'
if os.path.exists('SVM_model.sav'):
      print("Loading SVM Model")
      print("we find best accuracy when  (kernel='poly',C=4,degree=7)")
      # load the model from disk
      loaded_model = pickle.load(open(file_name_svm, 'rb'))
      y_pred = loaded_model.predict(X_test)
      get_model_evaluation(Y_test, y_pred)
else :
      print("Creating and a new SVM model")
      # save the model
      SVCModel = SVC(kernel='poly',C=4,degree=7)
      SVCModel.fit(X_train, Y_train)
      pickle.dump(SVCModel, open(file_name_svm, 'wb'))
      # load the model from disk
      loaded_model = pickle.load(open(file_name_svm, 'rb'))
      y_pred = loaded_model.predict(X_test)
      get_model_evaluation(Y_test, y_pred)


# # #Forward (wrapper)
# # # linear_forward_reg=LinearRegression()

# # sfs=SFS(SVCModel,k_features=4,forward=True,floating=False,scoring='r2',cv=0,n_jobs=-1)
# # # sfs=sfs.fit(X,Y)
# # sfs=sfs.fit(X_train,Y_train)
# # selected_features_forward = numerical_cols.columns[list(sfs.k_feature_idx_)]


# # # sfs_results=pd.DataFrame(sfs.subsets_).transpose()
# # # print(sfs_results)

# # Forward_X=data[selected_features_forward]
# # # linear_forward_reg.fit(Forward_X,Y)
# # SVCModel.fit(Forward_X,Y_train)

# # prediction=SVCModel.predict(Forward_X)
# # print("............................................")
# # print(selected_features_forward)
# # print("wrapper mean square error for SVM %.2f" % mean_squared_error(Y_test, prediction))
