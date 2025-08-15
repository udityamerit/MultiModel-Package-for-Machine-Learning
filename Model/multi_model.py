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
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from process import X_y
X,y = X_y()

# print(X.columns)


# Defining the functions of different models
def Regression_model(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    sc = StandardScaler()
    X_train_scaled = sc.fit_transform(X_train)
    X_test_scaled = sc.transform(X_test)

    model1 = LogisticRegression()
    model1.fit(X_train_scaled, y_train)
    y_pred_logi_test = model1.predict(X_test_scaled)
    logistic_accuracy = accuracy_score(y_test, y_pred_logi_test)
    # print(f'Logistic Regression Accuracy: {logistic_accuracy*100} % ')
    # print("Regression Model: ")
    reg_classificaiton = classification_report(y_test, y_pred_logi_test)
    return reg_classificaiton

def Support_vector_model(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    sc = StandardScaler()
    X_train_scaled = sc.fit_transform(X_train)
    X_test_scaled = sc.transform(X_test)
    svc = SVC(kernel='linear')
    svc.fit(X_train_scaled, y_train)
    y_pred_svc = svc.predict(X_test_scaled)
    svc_accuracy = accuracy_score(y_test, y_pred_svc)
    svc_classification_report = classification_report(y_test,y_pred_svc)
    # print("Support Vector Model: ")
    return svc_classification_report


def DecisionTree_model(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    sc = StandardScaler()
    X_train_scaled = sc.fit_transform(X_train)
    X_test_scaled = sc.transform(X_test)

    model1 = DecisionTreeClassifier()
    model1.fit(X_train_scaled, y_train)
    y_pred_logi_test = model1.predict(X_test_scaled)
    logistic_accuracy = accuracy_score(y_test, y_pred_logi_test)
    # print(f'DecisionTree Accuracy: {logistic_accuracy*100} % ')
    # print("Decision Tree Model: ")
    decision = classification_report(y_test, y_pred_logi_test)
    return decision


def KNN_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    sc = StandardScaler()
    X_train_scaled = sc.fit_transform(X_train)
    X_test_scaled = sc.transform(X_test)

    model1 = KNeighborsClassifier( n_neighbors= 10)
    model1.fit(X_train_scaled, y_train)
    y_pred_logi_test = model1.predict(X_test_scaled)
    logistic_accuracy = accuracy_score(y_test, y_pred_logi_test)
    # print(f'KNN Accuracy: {logistic_accuracy*100} % ')
    # print("KNN Model: ")
    knn = classification_report(y_test, y_pred_logi_test)
    return knn

def Naive_Bayes_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    sc = StandardScaler()
    X_train_scaled = sc.fit_transform(X_train)
    X_test_scaled = sc.transform(X_test)

    model1 = GaussianNB()
    model1.fit(X_train_scaled, y_train)
    y_pred_logi_test = model1.predict(X_test_scaled)
    logistic_accuracy = accuracy_score(y_test, y_pred_logi_test)
    # print(f'KNN Accuracy: {logistic_accuracy*100} % ')

    naive = classification_report(y_test, y_pred_logi_test)

    return naive


def run_all_model(X, y):
    models = [
        ("Regression model", Regression_model(X, y)),
        ("Support vector model", Support_vector_model(X, y)),
        ("Decision Tree model", DecisionTree_model(X, y)),
        ("KNN model", KNN_model(X, y)),
        ("Naive Bayes model", Naive_Bayes_model(X, y))
    ]

    for name, model in models:
        print(f"{name}:\n {model}")




if __name__=='__main__':
    run_all_model(X, y)
    
    