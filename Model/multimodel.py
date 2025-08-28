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
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

class MultiModelClassifier:

    def __init__(self, X, y, test_size=0.3):
        self.X = X
        self.y = y
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=test_size, random_state=42)
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(X_train)
        self.X_test_scaled = self.scaler.transform(X_test)
        self.y_train = y_train
        self.y_test = y_test
    
    @staticmethod
    def evaluate_model(true, predicted):
        report = classification_report(true, predicted)
        matrix = confusion_matrix(true, predicted)
        accuracy = accuracy_score(true, predicted)
        return report, matrix, accuracy
        
    def Regression_model(self):
        model = LogisticRegression()
        model.fit(self.X_train_scaled, self.y_train)
        y_pred = model.predict(self.X_test_scaled)

        return self.evaluate_model(self.y_test, y_pred)

    def Support_vector_model(self):
        svc = SVC(kernel='linear')
        svc.fit(self.X_train_scaled, self.y_train)
        y_pred = svc.predict(self.X_test_scaled)

        return self.evaluate_model(self.y_test, y_pred)

    def DecisionTree_model(self):
        model = DecisionTreeClassifier()
        model.fit(self.X_train_scaled, self.y_train)
        y_pred = model.predict(self.X_test_scaled)

        return self.evaluate_model(self.y_test, y_pred)

    def KNN_model(self):
        model = KNeighborsClassifier(n_neighbors=10)
        model.fit(self.X_train_scaled, self.y_train)
        y_pred = model.predict(self.X_test_scaled)

        return self.evaluate_model(self.y_test, y_pred)

    def Naive_Bayes_model(self):
        model = GaussianNB()
        model.fit(self.X_train_scaled, self.y_train)
        y_pred = model.predict(self.X_test_scaled)

        return self.evaluate_model(self.y_test, y_pred)

    def run_all_models(self):
        models = [
            ('Logistic Regression model', *self.Regression_model()),
            ('Support vector model', *self.Support_vector_model()),
            ('Decision Tree model', *self.DecisionTree_model()),
            ('KNN model', *self.KNN_model()),
            ('Naive Bayes model', *self.Naive_Bayes_model())
        ]
        return models

    def get_summary(self, models):
        best_model = max(models, key=lambda m: m[3]) if models else None

        for name, report, matrix, accuracy in models:
            print(f"\n{'='*40}\n{name}\n\nClassification Report:\n{report}\n\nAccuracy: {accuracy}\n")

            plt.figure(figsize=(6, 4))
            sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues')
            plt.title(f'Confusion Matrix for {name}')
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.show()

        if best_model:
            name, report, matrix, accuracy = best_model
            print(f"\n{'='*40}\nBest Model: {name}\n\nClassification Report:\n{report}\n\nAccuracy: {accuracy}\n")

            plt.figure(figsize=(6, 4))
            sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues')
            plt.title(f'Confusion Matrix for {name}')
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.show()

    def plot_comparison(self, models):
        model_names = [str(m[0]) for m in models]
        accuracies = [float(m[3]) for m in models]

        cmap = plt.get_cmap('tab10')  
        colors = [cmap(i % 10) for i in range(len(model_names))]

        plt.figure(figsize=(10, 6))
        bars = plt.bar(model_names, accuracies, color=colors)

        plt.title("Models Accuracy Comparison")
        plt.xlabel("Models")
        plt.ylabel("Accuracy")
        plt.ylim(0, 1)
        plt.xticks(rotation=25)
        plt.tight_layout()

        for bar, acc in zip(bars, accuracies):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"{acc:.2f}", ha='center', va='bottom')

        plt.show()


if __name__ == '__main__':
    MultiModelClassifier()
