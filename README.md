# 🔬 Multi-Class Benchmark Classification Pipeline (Iris / Wine)

An end-to-end supervised classification pipeline built with Scikit-Learn, Pandas, Seaborn, and Matplotlib.

## 📋 Overview

This project implements a complete classification pipeline on the **Iris** and **Wine** benchmark datasets, covering:

- ✅ Data loading (Iris from CSV · Wine from sklearn)
- ✅ Exploratory Data Analysis with Seaborn pairplots
- ✅ Feature scaling with `StandardScaler`
- ✅ Stratified 80/20 train/test split
- ✅ Logistic Regression & Decision Tree classifiers
- ✅ Benchmark metrics: Accuracy · Precision · Recall · F1-Score
- ✅ Confusion matrix heatmaps
- ✅ PCA 2-D decision boundary visualization
- ✅ Decision Tree structure diagram
- ✅ Feature importances (Gini)

## 📊 Results

| Dataset | Model               | Accuracy | Precision | Recall | F1-Score |
|---------|---------------------|----------|-----------|--------|----------|
| Iris    | Logistic Regression | 0.9333   | 0.9333    | 0.9333 | 0.9333   |
| Iris    | Decision Tree       | 0.9333   | 0.9333    | 0.9333 | 0.9333   |
| Wine    | Logistic Regression | 0.9722   | 0.9741    | 0.9722 | 0.9722   |
| Wine    | Decision Tree       | 0.9444   | 0.9508    | 0.9444 | 0.9453   |

## 🗂️ Project Structure

```
classification-pipeline/
├── classification_pipeline.ipynb   # Main Jupyter notebook
├── iris.csv                        # Iris dataset (user-supplied)
├── iris_pairplot.png
├── wine_pairplot.png
├── confusion_matrices.png
├── metrics_comparison.png
├── iris_decision_boundaries.png
├── wine_decision_boundaries.png
├── iris_decision_tree.png
├── wine_decision_tree.png
├── feature_importances.png
└── README.md
```

## 🚀 How to Run

```bash
pip install scikit-learn pandas seaborn matplotlib jupyter
jupyter notebook classification_pipeline.ipynb
```

## 🛠️ Tech Stack

| Library       | Purpose                          |
|---------------|----------------------------------|
| scikit-learn  | Models, scaling, metrics         |
| pandas        | Data loading & manipulation      |
| seaborn       | Pairplots & heatmaps             |
| matplotlib    | All custom visualizations        |
| jupyter       | Interactive notebook environment |
