import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import SelectFromModel
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
import time
import datetime
import nltk
import re
from nltk.corpus import stopwords
import statistics
from sklearn import datasets
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import*
from sklearn import linear_model
from sklearn import metrics
###############################################################################
###############################################################################

#[1]import data
data = pd.read_csv('games-regression-dataset.csv')

###############################################################################
###############################################################################

#[2]id , price & User Rating Count plot

data = data.rename(columns = {"ID":"id"})
plt.scatter(data['id'], data['Average User Rating'])
plt.title("id Scatter Plot")
# Setting the X and Y labels
plt.xlabel('id')
plt.ylabel('Average User Rating')
plt.show()


plt.scatter(data['Price'], data['Average User Rating'])
plt.title("Price Scatter Plot")
# Setting the X and Y labels
plt.xlabel('Price')
plt.ylabel('Average User Rating')
plt.show()


plt.scatter(data['User Rating Count'], data['Average User Rating'])
plt.title("User Rating Scatter Plot")
# Setting the X and Y labels
plt.xlabel('User Rating Count')
plt.ylabel('Average User Rating')
plt.show()

###############################################################################
###############################################################################

#[3] Cleaning data
#[3.1] cleaning missing values

#there is null or not
NullNumber=data.isnull().sum()
print(NullNumber)

#replace missing values with mode as it is categorical
mode_value = data['Languages'].mode()[0]
data['Languages'].fillna(mode_value, inplace=True)

#replace missing values with 0
data["In-app Purchases"].fillna("0", inplace=True)


###############################################################################
###############################################################################

#[3.2] drop Dublicate rows
#there are duplicate rows or not
DuplicateNumber=data.duplicated().sum()
print(DuplicateNumber)

data=data.drop_duplicates()

DuplicateNumber=data.duplicated().sum()
#print(DuplicateNumber)

###############################################################################
###############################################################################
#[4] Transformation data
# to know columns data type
data_type=data.dtypes
print(data_type)

label_encoder=LabelEncoder()

#[4.1] Encoding Name
data["Name"]=label_encoder.fit_transform(data["Name"])

plt.scatter(data['Name'], data['Average User Rating'])
plt.title("Name Scatter Plot")
 
# Setting the X and Y labels
plt.xlabel('Name')
plt.ylabel('Average User Rating')
plt.show()

#[4.2] Encoding Icon URL   
data["Icon URL"]=label_encoder.fit_transform(data["Icon URL"])              

plt.scatter(data['Icon URL'], data['Average User Rating'])
plt.title("Icon URL Scatter Plot")
 
# Setting the X and Y labels
plt.xlabel('Icon URL')
plt.ylabel('Average User Rating')
plt.show()

#[4.3] Encoding In-app Purchases    
data['In-app Purchases'] = data['In-app Purchases'].apply(lambda x: x.split(','))
data['In-app Purchases'] = data['In-app Purchases'].apply(lambda x: np.sum(np.array(x).astype(float)))

plt.scatter(data['In-app Purchases'], data['Average User Rating'])
plt.title("In-app Purchases Scatter Plot")
 
# Setting the X and Y labels
plt.xlabel('In-app Purchases')
plt.ylabel('Average User Rating')
plt.show()

       
    

#[4.4] Encoding Developer    
data["Developer"]=label_encoder.fit_transform(data["Developer"]) 
       
plt.scatter(data['Developer'], data['Average User Rating'])
plt.title("Developer Scatter Plot")
 
# Setting the X and Y labels
plt.xlabel('Developer')
plt.ylabel('Average User Rating')
plt.show()

#[4.5] Encoding Age Rating   
data['Age Rating'] = data['Age Rating'].apply(lambda x :int(x.strip('+')))           

plt.scatter(data['Age Rating'], data['Average User Rating'])
plt.title("Age Rating Scatter Plot")
 
# Setting the X and Y labels
plt.xlabel('Age Rating')
plt.ylabel('Average User Rating')
plt.show()

#[4.6] Encoding Languages  
language_dummies=data["Languages"].str.get_dummies(sep = ',')
data=pd.concat([data,language_dummies ], axis=1)
data=data.drop(columns=['Languages'])   

#[4.7] Encoding Primary Genre  
data["Primary Genre"]=label_encoder.fit_transform(data["Primary Genre"])   

plt.scatter(data['Primary Genre'], data['Average User Rating'])
plt.title("Primary Genre Scatter Plot")

#[4.8] Encoding Genres     
genre_dummies=data["Genres"].str.get_dummies(sep = ',')
data=pd.concat([data,genre_dummies ], axis=1)
data=data.drop(columns=['Genres'])               

#[4.9] Encoding Original Release Date
data['Original Release Date'] = data['Original Release Date'].apply(lambda x : time.mktime(datetime.datetime.strptime(x, '%d/%M/%Y').timetuple()))

plt.scatter(data['Original Release Date'], data['Average User Rating'])
plt.title("Original Release Date Scatter Plot")
 
# Setting the X and Y labels
plt.xlabel('Original Release Date')
plt.ylabel('Average User Rating')

plt.show()

#[4.10] Encoding Current Version Release Date
data['Current Version Release Date'] = data['Current Version Release Date'].apply(lambda x : time.mktime(datetime.datetime.strptime(x, '%d/%M/%Y').timetuple()))

plt.scatter(data['Current Version Release Date'], data['Average User Rating'])
plt.title("Current Version Release Date Scatter Plot")
 
# Setting the X and Y labels
plt.xlabel('Current Version Release Date')
plt.ylabel('Average User Rating')
 
plt.show()


# to know columns data type
data_type=data.dtypes
# print(data_type)

###############################################################################
###############################################################################

data = data.drop('id', axis=1)
data = data.drop('Name', axis=1)
data = data.drop('Icon URL', axis=1)
data = data.drop('Developer', axis=1)

# data = data.drop('Original Release Date', axis=1)
# data = data.drop('Current Version Release Date', axis=1)
###############################################################################
###############################################################################

# split the URL column into separate columns based on the forward slash character
url_parts = data['URL'].str.rsplit('/', n=-1, expand=True)

# concatenate the original DataFrame with the new columns
data_after_sep = pd.concat([data, url_parts], axis=1)

# drop all columns in url_parts except the last two columns
url_parts_cols_to_keep = url_parts.columns[-2:]
url_parts = url_parts.loc[:, url_parts_cols_to_keep]

url_parts_to_drop = set(url_parts.columns) - set(url_parts_cols_to_keep)
url_parts = url_parts.drop(columns=url_parts_to_drop)
url_parts = url_parts.drop(columns=url_parts_to_drop)

# concatenate the original DataFrame with the remaining columns in url_parts
data = pd.concat([data, url_parts], axis=1)

# # rename the 'old_name' column to 'new_name'
# data = data.rename(columns={'5': 'splitted url name'})
# data = data.rename(columns={'6': 'splitted url id'})
###############################################################################
###############################################################################

# Subtitle has 3749 Nulls
# fill missing values in subtitles column with values from name column
data['Subtitle'] = data['Subtitle'].fillna(data[5])

###############################################################################
###############################################################################

numerical_cols=data.select_dtypes(include=np.number)
#print(numerical_cols.mean())

#drop all non numeric columns and store only numeric cols 
data_numeric=data.applymap(lambda x: pd.to_numeric(x, errors='coerce')).dropna(axis=1)

numerical_col_names=data_numeric.columns.tolist()
# print(numerical_col_names)

###############################################################################
###############################################################################

#check shape dataframe
print("Old Shape: " ,data.shape)

# function returns list of index of outliers
def detect_outliers(df, ft):
    Q1 = data[ft].quantile(0.25)
    Q3 = data[ft].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outlier_lst = df.index[(df[ft] < lower_bound) | (df[ft] > upper_bound)]
    return outlier_lst


# create list to store the output indices from multiple col
outlier_indices_list = []

for feature in numerical_col_names:
    outlier_indices_list.extend(detect_outliers(data , feature))
#print(outlier_indices_list)

# function to return the cleaned dataframe
def remove_outlier(df , lst):
    lst = sorted(set(lst))
    df = df.drop(lst)
    return df

data_without_outliers = remove_outlier(data ,outlier_indices_list)
print("New Shape: ", data_without_outliers.shape)

#save the cleaned dataset to new csv file
data_without_outliers.to_csv('games-regression-dataset_cleaned.csv' , index = False)


###############################################################################
###############################################################################

# z_scores = np.abs((numerical_cols - numerical_cols.mean()) / numerical_cols.std())
# numerical_col_names = numerical_cols[z_scores < 3].dropna()

###############################################################################
###############################################################################

X = numerical_cols.loc[:, numerical_cols.columns != 'Average User Rating']
Y=data['Average User Rating']

# Split the data into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y,  test_size = 0.20,shuffle=True,random_state=10)

###############################################################################
###############################################################################
 
# # Scale the data using StandardScaler
# X_train =StandardScaler().fit_transform(X_train)
# X_test = StandardScaler().fit_transform(X_test)


# Scale the data using Normalization
X_train =MinMaxScaler().fit_transform(X_train)
X_test = MinMaxScaler().fit_transform(X_test)

##############################################################################
##############################################################################

#all cols except the last col
cols_cor = numerical_cols

#Feature Selection (FILTER(CORRELATION))
#correlation between features
cor = cols_cor.corr()

#Top Correlation training features  is very weak approx equals 0
top_feature_cor = cor.index[abs(cor['Average User Rating'])>0.1]

#Correlation plot
plt.subplots(figsize=(12, 8))
top_corr2 = data[top_feature_cor].corr()
sns.heatmap(top_corr2, annot=True)
plt.show()
top_feature_cor = top_feature_cor.delete(-1)
selected_features_cor = cols_cor[top_feature_cor]

###############################################################################
##############################################################################

# Lasso regression model
lasso_reg = Lasso(alpha = 0.01).fit(X_train,Y_train)

# Use Lasso regularization with SelectFromModel to perform feature selection
selector_lasso_reg = SelectFromModel(lasso_reg).fit(X_train,Y_train)

# Extract the selected features
selected_features_lasso_reg = X.columns[selector_lasso_reg.get_support()]

# train and test score for lasso regression
train_score_lss =lasso_reg.score(X_train,Y_train)
test_score_lss =lasso_reg.score(X_test,Y_test)

# Transform the training and testing sets to include only the selected features
X_train_selected_lss = selector_lasso_reg.transform(X_train)
X_test_selected_lss = selector_lasso_reg.transform(X_test)

# Fit the model to the training data using the selected features
linear_lasso_reg = LinearRegression().fit(X_train_selected_lss, Y_train)

# Predict on the test data using the selected features
Y_pred_selected_lss = linear_lasso_reg.predict(X_test_selected_lss)

# Compute the accuracy of the predictions using the selected features
mse_selected_lasso_reg = mean_squared_error(Y_test, Y_pred_selected_lss)

# Print the selected features and the mean squared error
print("............................................")
print("Selected features:", selected_features_lasso_reg)
print("Mean squared error using selected features for Lasso linear regression: %.2f" % mse_selected_lasso_reg)
# print("The train score for Lasso model is {}".format(train_score_lss))
# print("The test score for Lasso model is {}".format(test_score_lss))

###############################################################################
##############################################################################

# Ridge regression model
ridge_reg = Ridge(alpha = 0.01).fit(X_train,Y_train)

# Use Ridge regularization with SelectFromModel to perform feature selection
selector_ridge_reg = SelectFromModel(ridge_reg).fit(X_train,Y_train)

# Extract the selected features
selected_features_ridge_reg = X.columns[selector_ridge_reg.get_support()]

# train and test score for Ridge regression
train_score_rdg =ridge_reg.score(X_train,Y_train)
test_score_rdg =ridge_reg.score(X_test,Y_test)

# Transform the training and testing sets to include only the selected features
X_train_selected_rdg = selector_ridge_reg.transform(X_train)
X_test_selected_rdg = selector_ridge_reg.transform(X_test)

# Fit the model to the training data using the selected features
linear_ridge_reg = LinearRegression().fit(X_train_selected_rdg, Y_train)

# Predict on the test data using the selected features
Y_pred_selected_rdg = linear_ridge_reg.predict(X_test_selected_rdg)

# mean squared error
mse_selected_ridge_reg = mean_squared_error(Y_test, Y_pred_selected_rdg)

# print the selected features and the mean squared error
print("............................................")
print("Selected features:", selected_features_ridge_reg)
print("Mean squared error using selected features for Ridge linear regression: %.2f" % mse_selected_ridge_reg)
# print("The train score for Ridge model is {}".format(train_score_rdg))
# print("The test score for Ridge model is {}".format(test_score_rdg))

###############################################################################
###############################################################################

#Forward (wrapper)
linear_forward_reg=LinearRegression()

sfs=SFS(linear_forward_reg,k_features=4,forward=True,floating=False,scoring='r2',cv=0,n_jobs=-1)
# sfs=sfs.fit(X,Y)
sfs=sfs.fit(X_train,Y_train)
selected_features_forward = numerical_cols.columns[list(sfs.k_feature_idx_)]


# sfs_results=pd.DataFrame(sfs.subsets_).transpose()
# print(sfs_results)

Forward_X=data[selected_features_forward]
# linear_forward_reg.fit(Forward_X,Y)
linear_forward_reg.fit(Forward_X,Y)

prediction=linear_forward_reg.predict(Forward_X)
print("............................................")
print(selected_features_forward)
print("wrapper mean square error for linear regression %.2f" % mean_squared_error(Y, prediction))

###############################################################################
###############################################################################

# # multiple linear regression model
# multiple_reg = LinearRegression()
# multiple_reg.fit(X_train, Y_train)
# # lasso_reg = Lasso(alpha = 0.01).fit(X_train,Y_train)

# # Use SelectFromModel to perform feature selection
# selector_multiple_reg = SelectFromModel(multiple_reg).fit(X_train, Y_train)

# # Extract the selected features
# selected_features_multiple_reg = X.columns[selector_multiple_reg.get_support()]

# # train and test score for linear regression
# train_score_mreg =multiple_reg.score(X_train,Y_train)
# test_score_mreg  =multiple_reg.score(X_test,Y_test)

# # Transform the training and testing sets to include only the selected features
# X_train_selected_mreg = selector_multiple_reg.transform(X_train)
# X_test_selected_mreg = selector_multiple_reg.transform(X_test)

# # Fit the model to the training data using the selected features
# linear_multiple_reg = LinearRegression().fit(X_train_selected_mreg, Y_train)

# # Predict on the test data using the selected features
# Y_pred_selected_mreg = linear_multiple_reg.predict(X_test_selected_mreg)

# # Compute the accuracy of the predictions using the selected features
# mse_selected_multiple_reg = mean_squared_error(Y_test, Y_pred_selected_mreg)

# # Print the selected features and the mean squared error
# print("............................................")
# # print("Selected features:", selected_features_multiple_reg)
# print('Mean squared error of multiple regression : %.3f' % mse_selected_multiple_reg)
# print("The train score for Multiple Regression model is {}".format(train_score_mreg))
# print("The test score for Multiple Regression model is {}".format(test_score_mreg))


###############################################################################
###############################################################################


# Fit a Lasso regression model to the training data
lasso_reg = Lasso(alpha=0.01)
lasso_reg.fit(X_train, Y_train)

# Use Lasso regularization with SelectFromModel to perform feature selection
selector_lasso_reg = SelectFromModel(lasso_reg)
selector_lasso_reg.fit(X_train, Y_train)

# Extract the selected features
X_train_selected = selector_lasso_reg.transform(X_train)
X_test_selected = selector_lasso_reg.transform(X_test)

# Create polynomial features of degree 2 using the selected features
poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train_selected)
X_test_poly = poly.transform(X_test_selected)

# Fit a linear regression model to the training data using the polynomial features
linear_reg = LinearRegression()
linear_reg.fit(X_train_poly, Y_train)

# Predict on the test data using the polynomial features
Y_pred_poly = linear_reg.predict(X_test_poly)

# Compute the mean squared error of the predictions using the polynomial features
mse_poly = mean_squared_error(Y_test, Y_pred_poly)

# Print the selected features and the mean squared error
print("............................................")
print("Selected features:", X.columns[selector_lasso_reg.get_support()])
print("Mean squared error using polynomial features for Lasso Polynomial regression: %.2f" % mse_poly)

###############################################################################
###############################################################################

# Fit a Ridge regression model to the training data
ridge_reg = Ridge(alpha=10000)
ridge_reg.fit(X_train, Y_train)

# Use Ridge regularization with SelectFromModel to perform feature selection
selector_ridge_reg = SelectFromModel(ridge_reg)
selector_ridge_reg.fit(X_train, Y_train)

# Extract the selected features
X_train_selected = selector_ridge_reg.transform(X_train)
X_test_selected = selector_ridge_reg.transform(X_test)

# Create polynomial features of degree 2 using the selected features
poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train_selected)
X_test_poly = poly.transform(X_test_selected)

# Fit a linear regression model with Ridge regularization to the training data using the polynomial features
ridge_poly = Ridge(alpha=10000)
ridge_poly.fit(X_train_poly, Y_train)

# Predict on the test data using the polynomial features
Y_pred_poly = ridge_poly.predict(X_test_poly)

# Compute the mean squared error of the predictions using the polynomial features
mse_poly = mean_squared_error(Y_test, Y_pred_poly)

# Print the selected features and the mean squared error
print("............................................")
print("Selected features:", X.columns[selector_ridge_reg.get_support()])
print("Mean squared error using polynomial features for Ridge Polynomial regression: %.2f" % mse_poly)


###############################################################################
###############################################################################

# # Set degree for polynomial features
# degree = 2

# # Create polynomial features object
# poly = PolynomialFeatures(degree)

# # Fit and transform the data to polynomial features
# X_poly = poly.fit_transform(X)


# # Create Linear Regression object
# linear_poly_reg = LinearRegression()

# # Create Sequential Feature Selector object
# sfs = SFS(linear_poly_reg, k_features=4, forward=True, floating=False, scoring='r2', cv=0, n_jobs=-1)

# # Fit the SFS object to the data
# sfs = sfs.fit(X_poly, Y)

# # Get the selected features
# selected_features_forward = numerical_cols.columns[list(sfs.k_feature_idx_)]

# # Print the selected features
# print(selected_features_forward)

# # Create a new DataFrame with the selected features
# Forward_X = data[selected_features_forward]

# # Fit the linear regression model to the selected features
# linear_poly_reg.fit(poly.fit_transform(Forward_X), Y)

# # Predict the target variable using the selected features
# prediction = linear_poly_reg.predict(poly.fit_transform(Forward_X))

# # Print the mean squared error
# print("Wrapper mean square error for polynomial regression: ", mean_squared_error(Y, prediction))

###############################################################################
###############################################################################

# Lowercase the description column
data['Description'] = data['Description'].str.lower()

# Define a function to remove stopwords using NLTK
stopwords = set(stopwords.words('english'))
def remove_stopwords(text):
    tokens = nltk.word_tokenize(text)
    filtered_tokens = [token for token in tokens if token.lower() not in stopwords]
    filtered_text = ' '.join(filtered_tokens)
    return filtered_text

# Remove stopwords from the description column
data['Description'] = data['Description'].apply(remove_stopwords)

# Define a function to remove punctuation and special characters using regular expressions
def remove_punctuation(text):
    pattern = r'[^\w\s]'
    filtered_text = re.sub(pattern, '', text)
    return filtered_text

# Remove punctuation and special characters from the description column
data['Description'] = data['Description'].apply(remove_punctuation)

# Define a function to remove numerical values using regular expressions
def remove_numbers(text):
    pattern = r'\d+'
    filtered_text = re.sub(pattern, '', text)
    return filtered_text

# Remove numerical values from the description column
data['Description'] = data['Description'].apply(remove_numbers)

# Define a tokenizer function using NLTK
def tokenize_text(text):
    return nltk.word_tokenize(text)

# Tokenize the description column
data['Description'] = data['Description'].apply(tokenize_text)

###############################################################################
###############################################################################

fig, ax = plt.subplots(1,2, figsize=(15,5))
for idx, group in enumerate([('Train', Y_train), ('Test', Y_test)]):
    data_graph = group[1].value_counts()
    sns.barplot(ax=ax[idx], x=data_graph.index, y=data_graph.values)
    ax[idx].set_title(f'{group[0]} Label Count')
    ax[idx].set_xlabel(f'{group[0]} Labels')
    ax[idx].set_ylabel('Label Count')

plt.show()

###############################################################################
###############################################################################

# # concatenate the X and Y data into a single DataFrame
# data = pd.concat([X, Y], axis=1)

# # draw separate regression lines for each group
# sns.lmplot(x="X", y="Average User Rating", hue="group", data=data, height=5)









