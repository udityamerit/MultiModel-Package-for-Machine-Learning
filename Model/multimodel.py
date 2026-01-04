import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression, Lasso, Ridge
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, SVR
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_absolute_error, mean_squared_error, r2_score, roc_auc_score, roc_curve

class MultiModelClassifier:

    def __init__(self, X, y, test_size=0.3, scale = False):
        self.X = X
        self.y = y
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=test_size, random_state=42)
        self.scaler = StandardScaler()
        if scale==True:
            self.X_train_scaled = self.scaler.fit_transform(X_train)
            self.X_test_scaled = self.scaler.transform(X_test)
        else:
            self.X_train_scaled = X_train
            self.X_test_scaled = X_test
        self.y_train = y_train
        self.y_test = y_test
    
    @staticmethod
    def evaluate_model(model, X_test, y_true):
        predicted = model.predict(X_test)
        report = classification_report(y_true, predicted)
        matrix = confusion_matrix(y_true, predicted)
        accuracy = accuracy_score(y_true, predicted)
        
        if hasattr(model, "predict_proba"):
            y_pred_proba = model.predict_proba(X_test)[:, 1]
        else: 
            y_pred_proba = model.decision_function(X_test)
        
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = roc_auc_score(y_true, y_pred_proba)
        
        return report, matrix, accuracy, fpr, tpr, roc_auc
        

    def Regression_model(self):
        model = LogisticRegression()
        model.fit(self.X_train_scaled, self.y_train)
        return self.evaluate_model(model, self.X_test_scaled, self.y_test)

    def Support_vector_model(self):
        
        svc = SVC(kernel='linear', probability=True)
        svc.fit(self.X_train_scaled, self.y_train)
        return self.evaluate_model(svc, self.X_test_scaled, self.y_test)

    def DecisionTree_model(self):
        model = DecisionTreeClassifier()
        model.fit(self.X_train_scaled, self.y_train)
        return self.evaluate_model(model, self.X_test_scaled, self.y_test)

    def KNN_model(self):
        model = KNeighborsClassifier(n_neighbors=10)
        model.fit(self.X_train_scaled, self.y_train)
        return self.evaluate_model(model, self.X_test_scaled, self.y_test)

    def Naive_Bayes_model(self):
        model = GaussianNB()
        model.fit(self.X_train_scaled, self.y_train)
        return self.evaluate_model(model, self.X_test_scaled, self.y_test)

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

        for name, report, matrix, accuracy, fpr, tpr, roc_auc in models:
            print(f"\n{'='*40}\n{name}\n\nClassification Report:\n{report}\n\nAccuracy: {accuracy:.4f}\nROC AUC: {roc_auc:.4f}\n")

          
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            fig.suptitle(f'Metrics for {name}', fontsize=16)

            
            sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', ax=ax1)
            ax1.set_title('Confusion Matrix')
            ax1.set_xlabel('Predicted Label')
            ax1.set_ylabel('True Label')

           
            ax2.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
            ax2.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            ax2.set_title('ROC Curve')
            ax2.set_xlabel('False Positive Rate')
            ax2.set_ylabel('True Positive Rate')
            ax2.legend(loc="lower right")
            
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.show()

        if best_model:
            name, report, matrix, accuracy, _, _, roc_auc = best_model
            print(f"\n{'='*60}\n Best Model Found: {name}\n{'='*60}\n\nAccuracy: {accuracy:.4f}\nROC AUC: {roc_auc:.4f}\n\nClassification Report:\n{report}\n")

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            fig.suptitle(f'Metrics for {name}', fontsize=16)

            
            sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', ax=ax1)
            ax1.set_title('Confusion Matrix')
            ax1.set_xlabel('Predicted Label')
            ax1.set_ylabel('True Label')

           
            ax2.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
            ax2.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            ax2.set_title('ROC Curve')
            ax2.set_xlabel('False Positive Rate')
            ax2.set_ylabel('True Positive Rate')
            ax2.legend(loc="lower right")
            
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.show()

    def plot_comparison(self, models):
        model_names = [m[0] for m in models]
        accuracies = [m[3] for m in models]
        roc_aucs = [m[6] for m in models]
        
        # Plot 1: Accuracy Comparison
        plt.figure(figsize=(8, 6))
        sns.barplot(x=model_names, y=accuracies, palette='Blues')
        plt.title("Model Accuracy Comparison")
        plt.ylabel("Accuracy Score")
        plt.ylim(0, 1.1)
        plt.xticks(rotation=45)
        for i, acc in enumerate(accuracies):
            plt.text(i, acc + 0.02, f"{acc:.2f}", ha='center')
        plt.tight_layout()
        plt.show()
        
        # plot 2 Roc curve 
        plt.figure(figsize=(10, 8))
        for name, report, matrix, accuracy, fpr, tpr, roc_auc in models:
            plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.2f})')
        
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Chance')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Comparison of Receiver Operating Characteristic (ROC) Curves')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.show()

class MultiModelRegressior:

    def __init__(self, X, y, test_size=0.3, scale = False):
        self.X = X
        self.y = y
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=test_size, random_state=42)
        self.scaler = StandardScaler()
        if scale==True:
            self.X_train_scaled = self.scaler.fit_transform(X_train)
            self.X_test_scaled = self.scaler.transform(X_test)
        else:
            self.X_train_scaled = X_train
            self.X_test_scaled = X_test
        self.y_train = y_train
        self.y_test = y_test




if __name__ == '__main__':
    MultiModelClassifier()