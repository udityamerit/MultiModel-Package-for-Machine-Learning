import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# from google.colab import drive

# drive.mount('/content/drive')

# df = pd.read_csv('/content/drive/MyDrive/Datasets/diabetes.csv')
df = pd.read_csv('..\\Dataset\\diabetes.csv')

def preprocessing(df, verbose=True):
    # Declaring the Global variables for used throughout the code
    global X, y, y_test,y_train, X_test_scaled, X_train_scaled, y_pred_logi_test, y_pred_logi_train

    if verbose:
        df.head()

        col = list(df.columns)
        col

        # getting the info about the dataset
        df.info()

        df.isnull().cumsum()

        df.describe()

        

    # Selecting the Targeted columns and Non-Targeted columns
    X = df.drop(df[['Outcome']], axis=1)
    # or you can write it as X = df.drop(['Outcomes'], axis = 1)
    y = df['Outcome']

    # Checking the Columns of X for taking input features
    features = list(X.columns)
    features

    # Spliting the values in Testing and Traing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)
    
    if verbose:
        print('Data spliting is done')

        # Getting the Training and Testing dataset size
        print(f'Size of Training dataset: {len(X_train)} ')
        print(f'Size of Testing dataset: {len(X_test)}')

        # Normalization the data
    sc = StandardScaler()
    X_train_scaled = sc.fit_transform(X_train)
    X_test_scaled = sc.transform(X_test)

    if verbose:
        print('Feature scaling is done')


    # Calling the model
    model_logistic = LogisticRegression()
    model_logistic.fit(X_train_scaled, y_train)

    if verbose:
        print("Model training is done")

    y_pred_logi_test = model_logistic.predict(X_test_scaled)
    y_pred_logi_train = model_logistic.predict(X_train_scaled)

    if verbose:
        print("Model prediction is done")

# Defining the functions of different models
def Regression_model():
    model1 = LogisticRegression()
    model1.fit(X_train_scaled, y_train)
    y_pred_logi_test = model1.predict(X_test_scaled)
    logistic_accuracy = accuracy_score(y_test, y_pred_logi_test)
    print(f'Logistic Regression Accuracy: {logistic_accuracy*100} % ')
    print(classification_report(y_test, y_pred_logi_test))


def DecisionTree_model():
    model1 = DecisionTreeClassifier()
    model1.fit(X_train_scaled, y_train)
    y_pred_logi_test = model1.predict(X_test_scaled)
    logistic_accuracy = accuracy_score(y_test, y_pred_logi_test)
    print(f'DecisionTree Accuracy: {logistic_accuracy*100} % ')
    print(classification_report(y_test, y_pred_logi_test))


def KNN_model():
    model1 = KNeighborsClassifier( n_neighbors= 10)
    model1.fit(X_train_scaled, y_train)
    y_pred_logi_test = model1.predict(X_test_scaled)
    logistic_accuracy = accuracy_score(y_test, y_pred_logi_test)
    print(f'KNN Accuracy: {logistic_accuracy*100} % ')
    print(classification_report(y_test, y_pred_logi_test))


def pre_processing():
    print(preprocessing(df, verbose=True))

def model_list():
    preprocessing(df, verbose=False)
    print([Regression_model(), DecisionTree_model(), KNN_model()])


if __name__=='__main__':
    model_list()
    pre_processing()
    
    