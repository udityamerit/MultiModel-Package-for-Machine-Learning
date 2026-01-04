import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.multiclass import type_of_target

# Classification Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier

# Regression Models
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor

# Metrics
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, 
    mean_absolute_error, mean_squared_error, r2_score, 
    roc_auc_score, roc_curve, precision_score, recall_score, f1_score
)

# Set professional plotting style
sns.set_style("whitegrid")
plt.rcParams.update({'font.size': 12})

class MultiModelClassifier:

    def __init__(self, X, y, test_size=0.3, scaled_data=False):
        self.X = X
        self.y = y
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=test_size, random_state=42)
        
        # Initialize Scaler
        self.scaler = StandardScaler()
        
        # Logic updated to use 'scaled_data' flag
        if scaled_data:
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
        
        precision = precision_score(y_true, predicted, average='weighted')
        recall = recall_score(y_true, predicted, average='weighted')
        f1 = f1_score(y_true, predicted, average='weighted')
        
        if hasattr(model, "predict_proba"):
            y_pred_proba = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            y_pred_proba = model.decision_function(X_test)
        else:
            y_pred_proba = predicted
        
        try:
            fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
            roc_auc = roc_auc_score(y_true, y_pred_proba)
        except ValueError:
            fpr, tpr, roc_auc = [0], [0], 0.5

        return report, matrix, accuracy, precision, recall, f1, fpr, tpr, roc_auc
        
    def Logistic_model(self):
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

    def RandomForest_model(self):
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(self.X_train_scaled, self.y_train)
        return self.evaluate_model(model, self.X_test_scaled, self.y_test)

    def GradientBoosting_model(self):
        model = GradientBoostingClassifier(n_estimators=100, random_state=42)
        model.fit(self.X_train_scaled, self.y_train)
        return self.evaluate_model(model, self.X_test_scaled, self.y_test)

    def AdaBoost_model(self):
        model = AdaBoostClassifier(n_estimators=50, random_state=42)
        model.fit(self.X_train_scaled, self.y_train)
        return self.evaluate_model(model, self.X_test_scaled, self.y_test)

    def run_all_models(self):
        models = [
            ('Logistic Regression', *self.Logistic_model()),
            ('SVM', *self.Support_vector_model()),
            ('Decision Tree', *self.DecisionTree_model()),
            ('KNN', *self.KNN_model()),
            ('Naive Bayes', *self.Naive_Bayes_model()),
            ('Random Forest', *self.RandomForest_model()),
            ('Gradient Boosting', *self.GradientBoosting_model()),
            ('AdaBoost', *self.AdaBoost_model())
        ]
        return models
 
    def get_summary(self, models):
        best_model = max(models, key=lambda m: m[3]) if models else None

        for name, report, matrix, accuracy, precision, recall, f1, fpr, tpr, roc_auc in models:
            print(f"\n{'='*40}\n{name}\n{'='*40}")
            print(f"Accuracy : {accuracy:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f} | AUC: {roc_auc:.4f}")
            print(f"\nClassification Report:\n{report}\n")
          
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            fig.suptitle(f'Performance Metrics: {name}', fontsize=16, fontweight='bold')
            
            sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', ax=ax1, cbar=False)
            ax1.set_title('Confusion Matrix')
            ax1.set_xlabel('Predicted Label')
            ax1.set_ylabel('True Label')
           
            ax2.plot(fpr, tpr, color='#ff7f0e', lw=2.5, label=f'ROC (AUC = {roc_auc:.2f})')
            ax2.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            ax2.fill_between(fpr, tpr, alpha=0.1, color='#ff7f0e')
            ax2.set_title('ROC Curve')
            ax2.set_xlabel('False Positive Rate')
            ax2.set_ylabel('True Positive Rate')
            ax2.legend(loc="lower right")
            
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.show()

        if best_model:
            name, _, _, accuracy, _, _, _, _, _, roc_auc = best_model
            print(f"\n{'='*60}\n ⭐ BEST CLASSIFIER: {name} | Accuracy: {accuracy:.4f} | AUC: {roc_auc:.4f}\n{'='*60}\n")

    def plot_comparison(self, models):
        model_names = [m[0] for m in models]
        accuracy = [m[3] for m in models]
        precision = [m[4] for m in models]
        recall = [m[5] for m in models]
        f1 = [m[6] for m in models]
        
        data = {
            'Model': model_names * 4,
            'Score': accuracy + precision + recall + f1,
            'Metric': ['Accuracy']*len(models) + ['Precision']*len(models) + ['Recall']*len(models) + ['F1 Score']*len(models)
        }
        df_plot = pd.DataFrame(data)

        plt.figure(figsize=(14, 8))
        ax = sns.barplot(x='Model', y='Score', hue='Metric', data=df_plot, palette="viridis")
        plt.title("Comprehensive Model Comparison", fontsize=18, pad=20)
        plt.ylim(0, 1.1)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(rotation=45, ha='right')
        for container in ax.containers:
            ax.bar_label(container, fmt='%.2f', padding=3, fontsize=9)
        plt.tight_layout()
        plt.show()

class MultiModelRegressior:
    def __init__(self, X, y, test_size=0.3, scaled_data=False):
        self.X = X
        self.y = y
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=test_size, random_state=42)
        
        # Initialize Scaler
        self.scaler = StandardScaler()
        
        # Logic updated to use 'scaled_data' flag
        if scaled_data:
            self.X_train_scaled = self.scaler.fit_transform(X_train)
            self.X_test_scaled = self.scaler.transform(X_test)
        else:
            self.X_train_scaled = X_train
            self.X_test_scaled = X_test
            
        self.y_train = y_train
        self.y_test = y_test

    @staticmethod
    def evaluate_model(model, X_test, y_true):
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        return mae, mse, rmse, r2, y_pred

    def LinearRegression_model(self): return self.evaluate_model(LinearRegression().fit(self.X_train_scaled, self.y_train), self.X_test_scaled, self.y_test)
    def Lasso_model(self): return self.evaluate_model(Lasso(alpha=0.1).fit(self.X_train_scaled, self.y_train), self.X_test_scaled, self.y_test)
    def Ridge_model(self): return self.evaluate_model(Ridge(alpha=1.0).fit(self.X_train_scaled, self.y_train), self.X_test_scaled, self.y_test)
    def SVR_model(self): return self.evaluate_model(SVR(kernel='rbf').fit(self.X_train_scaled, self.y_train), self.X_test_scaled, self.y_test)
    def DecisionTree_model(self): return self.evaluate_model(DecisionTreeRegressor(random_state=42).fit(self.X_train_scaled, self.y_train), self.X_test_scaled, self.y_test)
    def RandomForest_model(self): return self.evaluate_model(RandomForestRegressor(n_estimators=100, random_state=42).fit(self.X_train_scaled, self.y_train), self.X_test_scaled, self.y_test)
    def GradientBoosting_model(self): return self.evaluate_model(GradientBoostingRegressor(n_estimators=100, random_state=42).fit(self.X_train_scaled, self.y_train), self.X_test_scaled, self.y_test)

    def run_all_models(self):
        return [
            ('Linear Regression', *self.LinearRegression_model()),
            ('Lasso Regression', *self.Lasso_model()),
            ('Ridge Regression', *self.Ridge_model()),
            ('SVR', *self.SVR_model()),
            ('Decision Tree Regressor', *self.DecisionTree_model()),
            ('Random Forest Regressor', *self.RandomForest_model()),
            ('Gradient Boosting Regressor', *self.GradientBoosting_model())
        ]

    def get_summary(self, models):
        best_model = max(models, key=lambda m: m[4]) if models else None
        for name, mae, mse, rmse, r2, y_pred in models:
            print(f"\n{'='*40}\n{name}\nMAE: {mae:.4f} | MSE: {mse:.4f} | RMSE: {rmse:.4f} | R2: {r2:.4f}\n")
        if best_model:
            print(f"\n{'='*60}\n ⭐ BEST REGRESSOR: {best_model[0]} | R2 Score: {best_model[4]:.4f}\n{'='*60}\n")

    def plot_comparison(self, models):
        model_names = [m[0] for m in models]
        r2_scores = [m[4] for m in models]
        plt.figure(figsize=(10, 6))
        sns.barplot(x=model_names, y=r2_scores, palette='viridis')
        plt.title("Regressor R2 Score Comparison")
        plt.ylabel("R2 Score")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

# =============================================================================
#  SMART MAIN BLOCK: AUTOMATIC DETECTION
# =============================================================================
if __name__ == '__main__':
    print("Initializing Multi-Model Analysis...")


    if 'X' in locals() and 'y' in locals():
        print(f"Data Loaded. X Shape: {X.shape}, y Shape: {y.shape}")
        
        
        target_type = type_of_target(y)
        unique_values = len(np.unique(y))
        
        if 'continuous' in target_type or (unique_values > 20 and target_type != 'multiclass'):
            print(f"\n[INFO] Detected REGRESSION problem (Target type: {target_type})")
            print("Running MultiModelRegressior...")
            
            # Updated argument 'scaled_data=True'
            regressor = MultiModelRegressior(X, y, scaled_data=True)
            results = regressor.run_all_models()
            regressor.get_summary(results)
            regressor.plot_comparison(results)
            
        else:
            print(f"\n[INFO] Detected CLASSIFICATION problem (Target type: {target_type})")
            print("Running MultiModelClassifier...")
            
            # Updated argument 'scaled_data=True'
            classifier = MultiModelClassifier(X, y, scaled_data=True)
            results = classifier.run_all_models()
            classifier.get_summary(results)
            classifier.plot_comparison(results)
            
    else:
        print("Error: X and y are not defined. Please define them.")