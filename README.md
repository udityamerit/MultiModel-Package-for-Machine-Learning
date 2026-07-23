<div align="center">

# 🚀 MultiModel Analysis

### Automatic Machine Learning Model Comparison & Benchmarking Library

<p align="center">

Train • Evaluate • Compare • Visualize • Benchmark

Multiple Machine Learning Models with **just a few lines of Python**

</p>

<p align="center">

[![PyPI Version](https://img.shields.io/pypi/v/multimodel-analysis.svg)](https://pypi.org/project/multimodel-analysis/)
[![Python](https://img.shields.io/pypi/pyversions/multimodel-analysis.svg)](https://pypi.org/project/multimodel-analysis/)
[![Downloads](https://img.shields.io/pypi/dm/multimodel-analysis)](https://pypi.org/project/multimodel-analysis/)
[![License](https://img.shields.io/badge/License-Apache%202.0-success.svg)](LICENSE)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)]()
[![Scikit-Learn](https://img.shields.io/badge/Built%20With-Scikit--Learn-orange)]()
[![Maintenance](https://img.shields.io/badge/Maintained-Yes-success)]()

</p>

---

### ⭐ Simplifying Machine Learning Model Comparison for Everyone

**MultiModel Analysis** is an open-source Python package that automates the process of training, evaluating, comparing, and visualizing multiple Machine Learning models.

Instead of manually writing repetitive code for each estimator, metric, and visualization, this library provides an elegant and beginner-friendly API that enables users to benchmark multiple models with just a few lines of Python.

Designed for:

🎓 Students

📊 Data Scientists

🤖 Machine Learning Engineers

🧪 Researchers

💼 Developers

---

## 📖 Table of Contents

- Overview
- Why MultiModel Analysis?
- Features
- Installation
- Quick Start
- Supported Models
- Evaluation Metrics
- Visualizations
- Examples
- API Documentation
- Roadmap
- Contributing
- License

---

# 📖 Overview

Machine Learning projects often require testing multiple algorithms before selecting the best-performing model.

Traditionally, developers need to:

- Import every estimator
- Train each model individually
- Generate predictions
- Calculate evaluation metrics
- Plot comparison graphs
- Compare the results manually

This process results in repetitive and time-consuming code.

**MultiModel Analysis** automates the entire workflow.

With a single API call, the package can:

- Train multiple ML algorithms
- Evaluate performance
- Compare every model
- Generate professional visualizations
- Recommend the best-performing model

This significantly reduces development time while improving reproducibility and consistency.

---

# ✨ Features

## 🚀 Model Training

✔ Train multiple models simultaneously

✔ Classification support

✔ Regression support

✔ Automatic train-test split

✔ Optional feature scaling

✔ Unified API

---

## 📊 Performance Evaluation

Automatically computes:

### Classification

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

### Regression

- Mean Absolute Error
- Mean Squared Error
- Root Mean Squared Error
- R² Score

---

## 📈 Visualization

Built-in professional visualizations

- Confusion Matrix
- ROC Curve
- Accuracy Comparison
- Precision Comparison
- Recall Comparison
- F1 Comparison
- Regression Comparison
- True vs Predicted Plot

---

## ⚡ Developer Friendly

- Minimal Code
- Clean API
- Beginner Friendly
- Research Ready
- Production Friendly

---

# 🚀 Why MultiModel Analysis?

Instead of writing hundreds of lines like this:

```python
model1.fit(X_train, y_train)

model2.fit(X_train, y_train)

model3.fit(X_train, y_train)

model4.fit(X_train, y_train)

pred1=model1.predict(X_test)

pred2=model2.predict(X_test)

pred3=model3.predict(X_test)

pred4=model4.predict(X_test)

accuracy_score(...)

precision_score(...)

recall_score(...)

f1_score(...)
```

Simply write:

```python
from multimodel_analysis import MultiModelClassifier

classifier = MultiModelClassifier(X, y)

results = classifier.run_all_models()

classifier.show_tabular_report(results)
```

That's it.

The package automatically handles:

- Data Splitting
- Model Training
- Prediction
- Evaluation
- Comparison
- Visualization

---

# 📦 Installation

Install directly from PyPI

```bash
pip install multimodel-analysis
```

Upgrade to the latest version

```bash
pip install --upgrade multimodel-analysis
```

Install from source

```bash
git clone https://github.com/udityamerit/multimodel-analysis.git

cd multimodel-analysis

pip install -e .
```

---

# 📚 Requirements

| Library | Version |
|----------|----------|
| Python | 3.9+ |
| NumPy | Latest |
| Pandas | Latest |
| Matplotlib | Latest |
| Seaborn | Latest |
| Scikit-Learn | Latest |

---

# 🚀 Quick Start

## Classification

```python
import pandas as pd

from multimodel_analysis import MultiModelClassifier

df = pd.read_csv("data.csv")

X = df.drop("target", axis=1)

y = df["target"]

classifier = MultiModelClassifier(

X,

y,

test_size=0.20,

scaled_data=True

)

results = classifier.run_all_models()

classifier.show_tabular_report(results)
```

---

## Regression

```python
import pandas as pd

from multimodel_analysis import MultiModelRegressior

df = pd.read_csv("housing.csv")

X = df.drop("Price", axis=1)

y = df["Price"]

regressor = MultiModelRegressior(

X,

y,

scaled_data=True

)

results = regressor.run_all_models()

regressor.show_tabular_report(results)
```

---

# 🤖 Supported Classification Models

| Model | Supported |
|---------|-----------|
| Logistic Regression | ✅ |
| Decision Tree | ✅ |
| Random Forest | ✅ |
| KNN | ✅ |
| Support Vector Machine | ✅ |
| Gaussian Naive Bayes | ✅ |
| AdaBoost | ✅ |
| Gradient Boosting | ✅ |

---

# 📈 Supported Regression Models

| Model | Supported |
|---------|-----------|
| Linear Regression | ✅ |
| Ridge Regression | ✅ |
| Lasso Regression | ✅ |
| Decision Tree Regressor | ✅ |
| Random Forest Regressor | ✅ |
| Gradient Boosting Regressor | ✅ |
| Support Vector Regressor | ✅ |

---

# 📊 Evaluation Metrics

## Classification

| Metric | Description |
|---------|-------------|
| Accuracy | Overall prediction correctness |
| Precision | Positive prediction quality |
| Recall | Ability to detect positives |
| F1 Score | Harmonic mean of Precision & Recall |
| ROC-AUC | Classification ranking quality |

---

## Regression

| Metric | Description |
|---------|-------------|
| MAE | Mean Absolute Error |
| MSE | Mean Squared Error |
| RMSE | Root Mean Squared Error |
| R² Score | Goodness of Fit |

---

# 📊 Built-in Visualizations

The package automatically generates:

✅ Confusion Matrix

✅ ROC Curve

✅ Model Comparison Charts

✅ Accuracy Comparison

✅ Precision Comparison

✅ Recall Comparison

✅ F1 Comparison

✅ True vs Predicted

✅ Regression Performance Comparison

---


# 📚 API Documentation

MultiModel Analysis provides two primary classes:

- `MultiModelClassifier`
- `MultiModelRegressior`

Both classes are designed with a consistent API, making it easy to switch between classification and regression tasks.

---

# 🧠 MultiModelClassifier

The `MultiModelClassifier` class automates the training and evaluation of multiple classification algorithms.

## Constructor

```python
MultiModelClassifier(
    X,
    y,
    test_size=0.2,
    random_state=42,
    scaled_data=False
)
```

### Parameters

| Parameter | Type | Description |
|------------|------|-------------|
| X | DataFrame / ndarray | Feature matrix |
| y | Series / ndarray | Target variable |
| test_size | float | Fraction of data used for testing |
| random_state | int | Random seed for reproducibility |
| scaled_data | bool | Automatically apply StandardScaler |

---

## Example

```python
from multimodel_analysis import MultiModelClassifier

classifier = MultiModelClassifier(
    X,
    y,
    test_size=0.25,
    scaled_data=True
)
```

---

# Available Methods

## 1. Train All Models

```python
results = classifier.run_all_models()
```

This function:

- Splits the dataset
- Trains every supported classifier
- Generates predictions
- Computes evaluation metrics
- Stores all results

Returns:

```python
dict
```

---

## 2. Display Report

```python
classifier.show_tabular_report(results)
```

Displays a formatted comparison table containing:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

Example Output

| Model | Accuracy | Precision | Recall | F1 |
|---------|-----------|------------|---------|-----|
| Random Forest | 98.7% | 98.5% | 98.6% | 98.5% |
| SVM | 97.8% | 97.5% | 97.4% | 97.4% |
| Logistic Regression | 96.9% | 96.8% | 96.6% | 96.7% |

---

## 3. Plot Model Comparison

```python
classifier.plot_comparison(results)
```

Generates a professional comparison chart of all trained models.

Useful for quickly identifying the best-performing classifier.

---

## 4. Plot Confusion Matrices

```python
classifier.plot_confusion_matrices(results)
```

Generates confusion matrices for every classification model.

Benefits

- Understand prediction quality
- Identify False Positives
- Identify False Negatives

---

## 5. Plot ROC Curves

```python
classifier.plot_roc_curves(results)
```

Automatically creates ROC curves for all supported classifiers.

This visualization helps compare classification performance beyond simple accuracy.

---

# 📊 Workflow

```
Dataset
     │
     ▼
Train-Test Split
     │
     ▼
Feature Scaling (Optional)
     │
     ▼
Train Multiple Models
     │
     ▼
Predictions
     │
     ▼
Evaluation Metrics
     │
     ▼
Visualization
     │
     ▼
Best Model Recommendation
```

---

# 📉 MultiModelRegressior

The `MultiModelRegressior` class provides a unified interface for comparing regression algorithms.

---

## Constructor

```python
MultiModelRegressior(
    X,
    y,
    test_size=0.2,
    random_state=42,
    scaled_data=False
)
```

---

## Example

```python
from multimodel_analysis import MultiModelRegressior

regressor = MultiModelRegressior(
    X,
    y,
    scaled_data=True
)
```

---

# Available Methods

## Train All Models

```python
results = regressor.run_all_models()
```

Automatically trains every supported regression algorithm.

Returns

```python
dict
```

---

## Display Performance Report

```python
regressor.show_tabular_report(results)
```

Displays

| Model | MAE | RMSE | R² |
|---------|------|--------|------|
| Random Forest | 2.31 | 3.15 | 0.96 |
| Gradient Boosting | 2.56 | 3.42 | 0.95 |
| Linear Regression | 3.74 | 4.18 | 0.91 |

---

## Compare Models

```python
regressor.plot_comparison(results)
```

Creates a comparison chart based on R² Score.

---

## True vs Predicted

```python
regressor.plot_true_vs_predicted(results)
```

Displays prediction accuracy visually.

Useful for

- Error analysis
- Model diagnostics
- Performance inspection

---



# 🤝 Contributing

Contributions are welcome and greatly appreciated!

## How to Contribute

### 1. Fork the Repository

```bash
git clone https://github.com/udityamerit/multimodel-analysis.git
```

### 2. Create a New Branch

```bash
git checkout -b feature/your-feature-name
```

### 3. Install Development Dependencies

```bash
pip install -r requirements.txt
```

### 4. Make Your Changes

Add your feature, enhancement, or bug fix.

### 5. Commit

```bash
git commit -m "Add: New Feature"
```

### 6. Push

```bash
git push origin feature/your-feature-name
```

### 7. Open a Pull Request

Describe the motivation, implementation details, and expected behavior.

---

# 📝 Changelog

## v1.0.0

- Initial Release
- Classification Support
- Regression Support
- Built-in Visualizations
- Automatic Performance Comparison
- PyPI Package Release

---

# 📚 Frequently Asked Questions

### Does the package support both Classification and Regression?

✅ Yes.

---

### Is feature scaling automatic?

Yes. Enable it using:

```python
scaled_data=True
```

---

### Can I use my own dataset?

Absolutely.

Any Pandas DataFrame or NumPy array is supported.

---

### Is it based on Scikit-Learn?

Yes.

The package is built on top of the Scikit-Learn ecosystem while providing a simplified interface.

---

### Is it suitable for beginners?

Definitely.

The API is intentionally designed to be beginner-friendly while remaining useful for experienced practitioners.

---

# 📖 Citation

If you use MultiModel Analysis in your research or academic work, please consider citing it.

```bibtex
@software{multimodel_analysis,
  author = {Uditya Narayan Tiwari},
  title = {MultiModel Analysis},
  year = {2026},
  url = {https://github.com/udityamerit/multimodel-analysis}
}
```

---

# 📜 License

Licensed under the **Apache License 2.0**.

You are free to use, modify, and distribute this project in accordance with the license terms.

See the `LICENSE` file for more information.

---

# 👨‍💻 Author

## **Uditya Narayan Tiwari**

**AI & Machine Learning Engineer**

Passionate about building practical AI solutions, open-source tools, and educational resources for the Machine Learning community.

### 🌐 Connect With Me

- 💼 **Portfolio:** https://udityanarayantiwari.netlify.app/
- 💻 **GitHub:** https://github.com/udityamerit
- 📦 **PyPI:** https://pypi.org/project/multimodel-analysis/
- 🔗 **LinkedIn:** https://www.linkedin.com/in/uditya-narayan-tiwari-562332289/

---

# 🌟 Support the Project

If you find this project useful:

⭐ Star the GitHub repository

🍴 Fork the repository

🐞 Report bugs

💡 Suggest new features

📢 Share it with your friends and colleagues

Every contribution helps improve the project and supports the open-source community.

---

# 🙏 Acknowledgements

This project is built upon the incredible Python ecosystem.

Special thanks to the maintainers and contributors of:

- Scikit-Learn
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Python Software Foundation

Their work makes projects like this possible.

---

# ❤️ Vision

The long-term vision of **MultiModel Analysis** is to become a lightweight yet powerful machine learning benchmarking toolkit that enables anyone—from beginners to professionals—to compare models, analyze results, and build better ML solutions with minimal effort.

Future versions aim to include advanced AutoML capabilities, explainable AI, model deployment utilities, and seamless integration with modern machine learning workflows.

---

<div align="center">

## ⭐ If this project helped you, please consider giving it a Star ⭐

### Built with ❤️ by **Uditya Narayan Tiwari**

### Empowering Developers, Students, and Researchers with Better Machine Learning Tools

**Happy Coding! 🚀**

</div>