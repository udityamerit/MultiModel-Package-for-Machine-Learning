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
from sklearn.metrics import accuracy_score, classification_report

from process import X_y

class MultiModelClassifier:

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def Regression_model(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.3, random_state=42)
        sc = StandardScaler()
        X_train_scaled = sc.fit_transform(X_train)
        X_test_scaled = sc.transform(X_test)

        model = LogisticRegression()
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        report = classification_report(y_test, y_pred)
        return report

    def Support_vector_model(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.3, random_state=42)
        sc = StandardScaler()
        X_train_scaled = sc.fit_transform(X_train)
        X_test_scaled = sc.transform(X_test)

        svc = SVC(kernel='linear')
        svc.fit(X_train_scaled, y_train)
        y_pred = svc.predict(X_test_scaled)
        report = classification_report(y_test, y_pred)
        return report

    def DecisionTree_model(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.3, random_state=42)
        sc = StandardScaler()
        X_train_scaled = sc.fit_transform(X_train)
        X_test_scaled = sc.transform(X_test)

        model = DecisionTreeClassifier()
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        report = classification_report(y_test, y_pred)
        return report

    def KNN_model(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.3, random_state=42)
        sc = StandardScaler()
        X_train_scaled = sc.fit_transform(X_train)
        X_test_scaled = sc.transform(X_test)

        model = KNeighborsClassifier(n_neighbors=10)
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        report = classification_report(y_test, y_pred)
        return report

    def Naive_Bayes_model(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.3, random_state=42)
        sc = StandardScaler()
        X_train_scaled = sc.fit_transform(X_train)
        X_test_scaled = sc.transform(X_test)

        model = GaussianNB()
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        report = classification_report(y_test, y_pred)
        return report

    def run_all_models(self):
        models = [
            ('Regression model', self.Regression_model()),
            ('Support vector model', self.Support_vector_model()),
            ('Decision Tree model', self.DecisionTree_model()),
            ('KNN model', self.KNN_model()),
            ('Naive Bayes model', self.Naive_Bayes_model())
        ]

        for name, report in models:
            print(f"{name}:\n{report}\n")

if __name__ == '__main__':
    X, y = X_y()
    ml_models = MultiModelClassifier(X, y)
    ml_models.run_all_models()
