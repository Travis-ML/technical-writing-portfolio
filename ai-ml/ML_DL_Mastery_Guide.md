# Machine Learning & Deep Learning Mastery Guide

**Author:** Travis Lelle ([travis@travisml.ai](mailto:travis@travisml.ai))  
**Published:** November 2025

---

## Introduction

This comprehensive guide is designed for students and practitioners who want to rapidly acquire the essential knowledge and skills needed to build, train, and deploy machine learning and deep learning systems. Whether you're catching up with peers or starting fresh, this guide provides a structured path to mastery.

### Our Approach

- **80/20 Rule:** Focus on the 20% of concepts that drive 80% of results
- **Practical First:** Learn by building, not just reading
- **Production Ready:** All code examples follow industry best practices
- **Spaced Repetition:** Integrated flashcard system for retention
- **Progressive Complexity:** Start with foundations, build to advanced topics

### Prerequisites

Basic Python programming knowledge, understanding of functions, loops, and data structures. Familiarity with basic mathematics (algebra, basic calculus) is helpful but not required.

---

## Table of Contents

1. [Part 1: Essential Python Packages](#part-1-essential-python-packages)
   - [NumPy - Numerical Computing](#1-numpy---numerical-computing-foundation)
   - [Pandas - Data Manipulation](#2-pandas---data-manipulation--analysis)
   - [Matplotlib & Seaborn - Visualization](#3-matplotlib--seaborn---data-visualization)
   - [Scikit-learn - Traditional ML](#4-scikit-learn---traditional-machine-learning)
   - [PyTorch - Deep Learning](#5-pytorch---deep-learning-framework)
   - [TorchVision - Computer Vision](#6-torchvision---computer-vision)
   - [Additional Libraries](#7-additional-essential-libraries)
2. [Part 2: Study Methods & Flashcards](#part-2-study-methods--flashcard-system)
3. [Part 3: Code Blueprint](#part-3-code-blueprint)
4. [Part 4: Four-Week Study Plan](#part-4-the-4-week-intensive-plan)

---

# Part 1: Essential Python Packages

## 1. NumPy - Numerical Computing Foundation

NumPy is the fundamental package for scientific computing in Python. Every machine learning framework (PyTorch, TensorFlow, Scikit-learn) uses NumPy arrays under the hood. Mastering NumPy operations is essential for understanding how ML algorithms work at a low level.

### 1.1 Array Creation

```python
import numpy as np

# Basic creation
arr = np.array([1, 2, 3, 4, 5])
matrix = np.array([[1, 2, 3], [4, 5, 6]])

# Special arrays
zeros = np.zeros((3, 4))           # All zeros
ones = np.ones((2, 3))             # All ones
identity = np.eye(4)                # Identity matrix
random = np.random.randn(3, 3)      # Random normal distribution
uniform = np.random.uniform(0, 1, (2, 3))  # Uniform distribution

# Ranges
arange = np.arange(0, 10, 2)        # [0, 2, 4, 6, 8]
linspace = np.linspace(0, 1, 5)     # 5 evenly spaced values from 0 to 1
```

**Key Insight:** Always prefer vectorized NumPy operations over Python loops. NumPy operations are 10-100x faster because they're implemented in C and use SIMD instructions.

### 1.2 Array Properties

```python
matrix = np.array([[1, 2, 3], [4, 5, 6]])

print(matrix.shape)      # (2, 3)
print(matrix.ndim)       # 2
print(matrix.size)       # 6
print(matrix.dtype)      # dtype('int64')
```

### 1.3 Reshaping & Manipulation

```python
arr = np.arange(12)
reshaped = arr.reshape(3, 4)         # Shape: (3, 4)
flattened = reshaped.flatten()       # Back to 1D
transposed = reshaped.T              # Transpose

# Adding/removing dimensions
expanded = arr[:, np.newaxis]        # Add dimension: (12,) -> (12, 1)
squeezed = expanded.squeeze()        # Remove dimensions of size 1

# Concatenation
arr1 = np.array([[1, 2], [3, 4]])
arr2 = np.array([[5, 6], [7, 8]])
vstack = np.vstack([arr1, arr2])     # Stack vertically
hstack = np.hstack([arr1, arr2])     # Stack horizontally
concat = np.concatenate([arr1, arr2], axis=0)  # Same as vstack
```

### 1.4 Indexing & Boolean Masking

Boolean indexing is one of NumPy's most powerful features and is extensively used in data preprocessing and filtering operations.

```python
arr = np.arange(10)
print(arr[2:5])          # [2, 3, 4]
print(arr[::2])          # [0, 2, 4, 6, 8] - every 2nd element
print(arr[::-1])         # Reverse array

# 2D indexing
matrix = np.arange(12).reshape(3, 4)
print(matrix[0, :])      # First row
print(matrix[:, 1])      # Second column
print(matrix[0:2, 1:3])  # Submatrix

# Boolean indexing (VERY IMPORTANT for data filtering)
arr = np.array([1, 2, 3, 4, 5])
mask = arr > 3
print(arr[mask])         # [4, 5]
print(arr[arr > 3])      # Same thing, one line

# Fancy indexing
indices = [0, 2, 4]
print(arr[indices])      # [1, 3, 5]
```

### 1.5 Mathematical Operations

```python
# Element-wise operations (broadcasting)
arr = np.array([1, 2, 3, 4])
print(arr + 10)          # [11, 12, 13, 14]
print(arr * 2)           # [2, 4, 6, 8]
print(arr ** 2)          # [1, 4, 9, 16]

# Array operations
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])
print(arr1 + arr2)       # [5, 7, 9]
print(arr1 * arr2)       # [4, 10, 18] - element-wise multiplication

# Matrix multiplication (CRITICAL for neural networks)
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
print(A @ B)             # Matrix multiplication (preferred)
print(np.dot(A, B))      # Same thing
print(A * B)             # Element-wise (different!)
```

**Critical Distinction:** Matrix multiplication (A @ B) is fundamentally different from element-wise multiplication (A * B). Neural networks rely on matrix multiplication for the forward pass: `output = input @ weights + bias`.

### 1.6 Statistics & Aggregations

```python
data = np.random.randn(100, 5)  # 100 samples, 5 features

# Basic stats
print(np.mean(data))             # Mean
print(np.std(data))              # Standard deviation
print(np.var(data))              # Variance
print(np.median(data))           # Median
print(np.min(data), np.max(data))

# Along specific axis
print(np.mean(data, axis=0))     # Mean of each column (5 values)
print(np.mean(data, axis=1))     # Mean of each row (100 values)

# Sums
print(np.sum(data))
print(np.cumsum(data))           # Cumulative sum
```

**Understanding Axis:** In a 2D array, `axis=0` operates down columns (resulting in one value per column), while `axis=1` operates across rows (one value per row). This is crucial for batch processing in neural networks.

### 1.7 Linear Algebra

Linear algebra operations are the mathematical foundation of machine learning.

```python
# Dot product
v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])
dot_product = np.dot(v1, v2)     # 1*4 + 2*5 + 3*6 = 32

# Matrix operations
A = np.random.randn(3, 3)
A_inv = np.linalg.inv(A)         # Inverse
det = np.linalg.det(A)           # Determinant
eigenvalues, eigenvectors = np.linalg.eig(A)  # Eigendecomposition

# Solving linear systems (Ax = b)
A = np.array([[3, 1], [1, 2]])
b = np.array([9, 8])
x = np.linalg.solve(A, b)        # Solution to Ax = b

# Norms (important for regularization)
v = np.array([3, 4])
l2_norm = np.linalg.norm(v)      # sqrt(3^2 + 4^2) = 5
l1_norm = np.linalg.norm(v, 1)   # |3| + |4| = 7
```

### 1.8 Broadcasting

Broadcasting is NumPy's method of performing operations on arrays of different shapes. It's extensively used in ML for efficient batch operations.

```python
# Example 1: Add scalar to array
arr = np.array([[1, 2, 3], [4, 5, 6]])
result = arr + 10                 # Adds 10 to every element

# Example 2: Add 1D to 2D
matrix = np.array([[1, 2, 3], [4, 5, 6]])
row = np.array([10, 20, 30])
result = matrix + row             # Adds row to each row of matrix

# Example 3: Normalize data (common preprocessing)
data = np.random.randn(100, 5)
mean = np.mean(data, axis=0)      # Shape: (5,)
std = np.std(data, axis=0)        # Shape: (5,)
normalized = (data - mean) / std  # Broadcasting works!
```

**Broadcasting Rules:**
1. Arrays with fewer dimensions are padded with ones on the left
2. Arrays with size 1 along a dimension are stretched to match the other array
3. If sizes don't match and neither is 1, broadcasting fails

### 1.9 Common ML Operations

```python
# Activation functions
def sigmoid(x):
    """Sigmoid activation function"""
    return 1 / (1 + np.exp(-x))

def relu(x):
    """ReLU activation function"""
    return np.maximum(0, x)

def softmax(x):
    """Softmax for multi-class classification"""
    exp_x = np.exp(x - np.max(x))  # Numerical stability
    return exp_x / np.sum(exp_x, axis=0)

# One-hot encoding
def one_hot_encode(y, num_classes):
    """One-hot encoding for labels"""
    one_hot = np.zeros((len(y), num_classes))
    one_hot[np.arange(len(y)), y] = 1
    return one_hot

# Neural network forward pass
def forward_pass(X, W1, b1, W2, b2):
    """
    X: Input (n_samples, n_features)
    W1: First layer weights (n_features, n_hidden)
    b1: First layer bias (n_hidden,)
    W2: Second layer weights (n_hidden, n_output)
    b2: Second layer bias (n_output,)
    """
    # Layer 1
    z1 = X @ W1 + b1
    a1 = relu(z1)
    
    # Layer 2 (output)
    z2 = a1 @ W2 + b2
    a2 = softmax(z2)
    
    return a2
```

### 1.10 Useful Functions

```python
# Clip values (prevent extreme values)
arr = np.array([-5, 2, 8, -1, 10])
clipped = np.clip(arr, 0, 5)      # [0, 2, 5, 0, 5]

# Where (conditional selection)
arr = np.array([1, 2, 3, 4, 5])
result = np.where(arr > 3, arr, 0)  # [0, 0, 0, 4, 5]

# Unique values
arr = np.array([1, 2, 2, 3, 3, 3])
unique = np.unique(arr)           # [1, 2, 3]
values, counts = np.unique(arr, return_counts=True)

# Sorting
arr = np.array([3, 1, 4, 1, 5])
sorted_arr = np.sort(arr)
indices = np.argsort(arr)         # Indices that would sort the array

# Random sampling (for train/test split)
np.random.seed(42)                # Reproducibility
shuffled = np.random.permutation(10)  # Shuffled [0, 1, ..., 9]
choice = np.random.choice(10, size=5, replace=False)  # 5 random samples
```

### NumPy Summary - When to Use What

- **Arrays**: Store data, perform vectorized operations
- **Matrix multiplication**: Neural network layers
- **Broadcasting**: Normalize data, apply operations efficiently
- **Random**: Initialize weights, shuffle data
- **Linear algebra**: PCA, SVD, solving equations

---

## 2. Pandas - Data Manipulation & Analysis

Pandas is essential for loading, cleaning, and preprocessing data. In real-world ML projects, you'll spend 60-80% of your time on data preparation.

### 2.1 Creating DataFrames

```python
import pandas as pd
import numpy as np

# From dictionary
data = {
    'name': ['Alice', 'Bob', 'Charlie', 'David'],
    'age': [25, 30, 35, 28],
    'salary': [50000, 60000, 75000, 55000]
}
df = pd.DataFrame(data)

# From CSV (most common)
df = pd.read_csv('data.csv')
df = pd.read_csv('data.csv', index_col=0)  # Use first column as index
df = pd.read_csv('data.csv', na_values=['?', 'NA'])  # Custom missing values

# From NumPy array
arr = np.random.randn(100, 5)
df = pd.DataFrame(arr, columns=['A', 'B', 'C', 'D', 'E'])
```

### 2.2 Basic Operations

```python
# Viewing data
print(df.head())        # First 5 rows
print(df.tail(10))      # Last 10 rows
print(df.sample(5))     # 5 random rows

# Info about the DataFrame
print(df.shape)         # (rows, columns)
print(df.info())        # Column types, non-null counts
print(df.describe())    # Statistics for numeric columns
print(df.columns)       # Column names
print(df.dtypes)        # Data types of each column
```

### 2.3 Selecting Data

```python
# Single column (returns Series)
ages = df['age']

# Multiple columns (returns DataFrame)
subset = df[['name', 'age']]

# Row selection
row = df.iloc[0]         # First row by position
rows = df.iloc[0:5]      # First 5 rows

# Conditional selection (VERY IMPORTANT!)
young = df[df['age'] < 30]
high_salary = df[df['salary'] > 60000]

# Multiple conditions
filtered = df[(df['age'] > 25) & (df['salary'] > 55000)]
filtered = df[(df['age'] < 30) | (df['salary'] > 70000)]

# Using query
filtered = df.query('age > 25 and salary > 55000')
```

### 2.4 Handling Missing Data

```python
# Check for missing values
print(df.isnull())           # Boolean DataFrame
print(df.isnull().sum())     # Count per column
print(df.isnull().any())     # True if any nulls in column

# Drop missing values
df_clean = df.dropna()                    # Drop rows with any NaN
df_clean = df.dropna(subset=['age'])      # Drop if 'age' is NaN
df_clean = df.dropna(axis=1)              # Drop columns with NaN

# Fill missing values
df_filled = df.fillna(0)                  # Fill with 0
df_filled = df.fillna(df.mean())          # Fill with column mean
df_filled = df.fillna(method='ffill')     # Forward fill

# Fill specific column
df['age'].fillna(df['age'].median(), inplace=True)
```

### 2.5 Data Transformation

```python
# Create new columns
df['age_squared'] = df['age'] ** 2
df['salary_k'] = df['salary'] / 1000

# Apply function
df['age_category'] = df['age'].apply(lambda x: 'Young' if x < 30 else 'Old')

# Map values
status_map = {25: 'Junior', 30: 'Mid', 35: 'Senior'}
df['status'] = df['age'].map(status_map)

# Binning
df['age_bin'] = pd.cut(df['age'], bins=[0, 30, 40, 100], 
                        labels=['Young', 'Middle', 'Senior'])

# One-hot encoding (CRITICAL for ML)
df_encoded = pd.get_dummies(df, columns=['age_bin'], prefix='age')
```

### 2.6 Date Features

```python
# Convert to datetime
df['date'] = pd.to_datetime(df['date'])

# Extract components
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day_of_week'] = df['date'].dt.dayofweek
df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

# Time series operations
df = df.set_index('date')
monthly = df.resample('M').mean()      # Monthly average
df['rolling_mean'] = df['value'].rolling(window=7).mean()

# Lag features
df['prev_value'] = df['value'].shift(1)
```

### 2.7 Grouping & Aggregation

```python
# Group by
grouped = df.groupby('category')
print(grouped.mean())
print(grouped.sum())

# Multiple aggregations
agg_result = df.groupby('category').agg({
    'sales': ['mean', 'max', 'min'],
    'quantity': 'sum'
})

# Custom aggregation
df.groupby('category')['value'].agg(['mean', 'std', lambda x: x.max() - x.min()])
```

### 2.8 Merging & Joining

```python
# Merge (like SQL JOIN)
merged = pd.merge(df1, df2, on='id', how='inner')  # Inner join
merged = pd.merge(df1, df2, on='id', how='left')   # Left join
merged = pd.merge(df1, df2, on='id', how='outer')  # Outer join

# Concatenate
df_concat = pd.concat([df1, df2], axis=0)  # Stack vertically
df_concat = pd.concat([df1, df2], axis=1)  # Stack horizontally
```

### 2.9 String Operations

```python
# String methods via .str
df['name_upper'] = df['name'].str.upper()
df['name_length'] = df['name'].str.len()
df['contains_a'] = df['name'].str.contains('a')

# Split strings
df[['first', 'last']] = df['full_name'].str.split(' ', expand=True)
```

### 2.10 Complete Preprocessing Pipeline

```python
def prepare_ml_data(df, target_col):
    """Complete preprocessing pipeline"""
    
    # 1. Handle missing values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    df[categorical_cols] = df[categorical_cols].fillna('Unknown')
    
    # 2. Encode categorical variables
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    # 3. Separate features and target
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    return X, y
```

### Pandas Summary - When to Use What

- **read_csv**: Load data (starting point for 95% of projects)
- **groupby**: Analyze patterns by category
- **merge**: Combine data from multiple sources
- **get_dummies**: Encode categorical variables for ML
- **fillna/dropna**: Handle missing data
- **apply**: Custom transformations

---

## 3. Matplotlib & Seaborn - Data Visualization

### Essential Visualizations

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Training curves (CRITICAL for ML!)
def plot_training_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy
    ax2.plot(history['train_acc'], label='Train Acc')
    ax2.plot(history['val_acc'], label='Val Acc')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

# Correlation heatmap (for feature selection)
plt.figure(figsize=(10, 8))
correlation = df.corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix')
plt.show()

# Distribution plot
sns.histplot(data=df, x='feature', hue='category', kde=True, bins=20)

# Box plot (outlier detection)
sns.boxplot(data=df, x='category', y='value')

# Confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()
```

### When to Use What

- **Line plots**: Training curves, time series
- **Scatter plots**: Relationships between variables
- **Histograms**: Distribution of single variable
- **Heatmaps**: Correlation matrices, confusion matrices
- **Box plots**: Outlier detection

---

## 4. Scikit-learn - Traditional Machine Learning

### Data Splitting

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,
    random_state=42,
    stratify=y  # Maintain class distribution
)
```

### Preprocessing

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Standardization (mean=0, std=1) - MOST COMMON
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Min-Max Scaling (0 to 1)
minmax = MinMaxScaler()
X_scaled = minmax.fit_transform(X)
```

### Linear Regression

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

reg = LinearRegression()
reg.fit(X_train, y_train)

y_pred = reg.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
```

### Logistic Regression

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

log_reg = LogisticRegression(max_iter=1000, random_state=42)
log_reg.fit(X_train, y_train)

y_pred = log_reg.predict(X_test)
y_pred_proba = log_reg.predict_proba(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(classification_report(y_test, y_pred))
```

### Random Forest

```python
from sklearn.ensemble import RandomForestClassifier

rf_clf = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)
rf_clf.fit(X_train, y_train)

# Feature importance
importances = rf_clf.feature_importances_
```

### Cross-Validation

```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(rf_clf, X, y, cv=5, scoring='accuracy')
print(f"Mean accuracy: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
```

### Hyperparameter Tuning

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)
grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")
```

### Pipeline (Best Practice!)

```python
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=0.95)),
    ('classifier', RandomForestClassifier(random_state=42))
])

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
```

### When to Use Which Algorithm

- **Linear/Logistic Regression**: Baseline, interpretability needed
- **Decision Trees**: Interpretable, handles non-linear relationships
- **Random Forest**: General-purpose, robust, good feature importance
- **Gradient Boosting**: Highest accuracy (often wins competitions)
- **SVM**: High-dimensional data
- **KNN**: Simple, non-parametric, good for small datasets

---

## 5. PyTorch - Deep Learning Framework

### Tensor Basics

```python
import torch

# Creating tensors
x = torch.tensor([1, 2, 3])
x = torch.zeros(3, 4)
x = torch.randn(2, 3)

# Tensor properties
print(x.shape)
print(x.dtype)
print(x.device)
```

### Tensor Operations

```python
# Element-wise
c = a + b
c = a * b

# Matrix multiplication
c = a @ b
c = torch.matmul(a, b)

# Reshaping
x = torch.arange(12)
x = x.view(3, 4)
x = x.reshape(3, 4)
```

### Autograd (Automatic Differentiation)

```python
x = torch.tensor([2.0], requires_grad=True)
y = x ** 2

y.backward()
print(x.grad)  # dy/dx = 2x = 4
```

### Neural Network Layers

```python
import torch.nn as nn

# Linear layer
linear = nn.Linear(in_features=10, out_features=5)

# Convolutional layer
conv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)

# Pooling
maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

# Batch Normalization
bn = nn.BatchNorm2d(64)

# Dropout
dropout = nn.Dropout(p=0.5)
```

### Building Neural Networks

```python
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
```

### CNN Architecture

```python
class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
```

### Loss Functions

```python
# Cross-Entropy (classification)
criterion = nn.CrossEntropyLoss()

# Binary Cross-Entropy
criterion = nn.BCEWithLogitsLoss()

# MSE (regression)
criterion = nn.MSELoss()
```

### Optimizers

```python
import torch.optim as optim

# Adam (most common)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

# SGD
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# AdamW
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
```

### Learning Rate Schedulers

```python
# Step LR
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# Reduce on plateau
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5
)

# Cosine annealing
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
```

### Dataset & DataLoader

```python
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

dataset = CustomDataset(X, y)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
```

### Training Loop

```python
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    
    return total_loss / len(dataloader), 100. * correct / total
```

### Evaluation Loop

```python
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    return total_loss / len(dataloader), 100. * correct / total
```

### Saving & Loading

```python
# Save
torch.save(model.state_dict(), 'model.pth')

# Save checkpoint
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
}, 'checkpoint.pth')

# Load
model.load_state_dict(torch.load('model.pth'))
model.eval()
```

### Transfer Learning

```python
import torchvision.models as models

resnet = models.resnet50(pretrained=True)

# Freeze all layers
for param in resnet.parameters():
    param.requires_grad = False

# Replace final layer
num_features = resnet.fc.in_features
resnet.fc = nn.Linear(num_features, 10)
```

### GPU Usage

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
x = x.to(device)
```

---

## 6. TorchVision - Computer Vision

### Data Augmentation

```python
import torchvision.transforms as transforms

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])
```

### Loading Datasets

```python
import torchvision.datasets as datasets

# MNIST
mnist = datasets.MNIST(root='./data', train=True, download=True, 
                       transform=transforms.ToTensor())

# CIFAR-10
cifar = datasets.CIFAR10(root='./data', train=True, download=True,
                         transform=train_transform)

# ImageFolder
dataset = datasets.ImageFolder(root='./data/train', transform=train_transform)
```

### Pretrained Models

```python
import torchvision.models as models

# ResNet
resnet18 = models.resnet18(pretrained=True)
resnet50 = models.resnet50(pretrained=True)

# EfficientNet
efficientnet = models.efficientnet_b0(pretrained=True)

# MobileNet
mobilenet = models.mobilenet_v2(pretrained=True)
```

---

## 7. Additional Essential Libraries

### PIL/Pillow - Image Processing

```python
from PIL import Image

img = Image.open('image.jpg')
img_resized = img.resize((224, 224))
img_gray = img.convert('L')
```

### tqdm - Progress Bars

```python
from tqdm import tqdm

for i in tqdm(range(100), desc="Processing"):
    # Your code here
    pass
```

---

# Part 2: Study Methods & Flashcard System

## The 80/20 Approach

- **80% time coding/implementing**
- **20% time on theory/formulas**
- Learn by doing, not just reading

## Daily Routine

1. **Morning (30 min)**: Review 20 flashcards (10 old, 10 new)
2. **Midday (2 hours)**: Code implementation
3. **Evening (1 hour)**: Read documentation/papers

## Spaced Repetition Schedule

- **Day 1**: Learn concept
- **Day 2**: Review
- **Day 4**: Review
- **Day 7**: Review
- **Day 14**: Review
- **Day 30**: Review

## Flashcard Examples

### NumPy Cards

**Front:** "What's the difference between np.dot() and element-wise multiplication?"

**Back:** 
```
np.dot(A,B) = matrix multiplication
A * B = element-wise multiplication
Example: [[1,2]]·[[3],[4]] = 11 vs [[1,2]]*[[3,4]] = [[3,8]]
```

**Front:** "How to normalize data with NumPy?"

**Back:**
```python
normalized = (data - np.mean(data, axis=0)) / np.std(data, axis=0)
```

### Pandas Cards

**Front:** "How to handle missing values?"

**Back:**
```
df.dropna() - remove rows
df.fillna(value) - fill with value
df.fillna(df.mean()) - fill with mean
```

**Front:** "Difference between .loc and .iloc?"

**Back:**
```
.loc - label-based: df.loc['row_label', 'col_label']
.iloc - integer-based: df.iloc[0, 1]
```

### PyTorch Cards

**Front:** "Difference between model.eval() and torch.no_grad()?"

**Back:**
```
model.eval() - disables dropout/batchnorm training behavior
torch.no_grad() - disables gradient computation
Use BOTH during validation!
```

**Front:** "Why optimizer.zero_grad()?"

**Back:**
```
PyTorch accumulates gradients by default
Must zero before each backward() pass
Otherwise gradients add up!
```

---

# Part 3: Code Blueprint

## Complete ML/DL Project Template

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ========================================
# 1. CONFIGURATION
# ========================================
class Config:
    # Data
    data_path = 'data.csv'
    train_split = 0.7
    val_split = 0.15
    test_split = 0.15
    
    # Model
    input_dim = 784
    hidden_dims = [256, 128, 64]
    output_dim = 10
    dropout_rate = 0.3
    
    # Training
    batch_size = 64
    learning_rate = 0.001
    num_epochs = 100
    weight_decay = 1e-5
    
    # Optimization
    patience = 10
    model_save_path = 'best_model.pth'

# ========================================
# 2. DATASET
# ========================================
class CustomDataset(Dataset):
    def __init__(self, X, y, transform=None):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
        self.transform = transform
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ========================================
# 3. MODEL
# ========================================
class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, dropout_rate):
        super(NeuralNetwork, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

# ========================================
# 4. TRAINING
# ========================================
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    
    return total_loss / len(dataloader), 100. * correct / total

# ========================================
# 5. EVALUATION
# ========================================
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    return total_loss / len(dataloader), 100. * correct / total

# ========================================
# 6. MAIN TRAINING LOOP
# ========================================
def main():
    config = Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load and preprocess data
    # ... (data loading code)
    
    # Create model
    model = NeuralNetwork(
        config.input_dim, 
        config.hidden_dims, 
        config.output_dim,
        config.dropout_rate
    ).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate,
                          weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(config.num_epochs):
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = evaluate(
            model, val_loader, criterion, device
        )
        
        scheduler.step(val_loss)
        
        print(f'Epoch {epoch+1}/{config.num_epochs}:')
        print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), config.model_save_path)
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= config.patience:
            break

if __name__ == "__main__":
    main()
```

## Quick Start Checklist

Before coding any model, ask yourself:

1. ✅ What's my problem type? (Classification/Regression)
2. ✅ What's my input shape?
3. ✅ What's my output shape?
4. ✅ How much data do I have?
5. ✅ What's my baseline?
6. ✅ What metrics matter?
7. ✅ Do I need normalization? (YES for neural networks)
8. ✅ How will I prevent overfitting?

---

# Part 4: The 4-Week Intensive Plan

## Week 1: Foundations

- **Day 1-2**: NumPy deep dive (implement from scratch)
- **Day 3-4**: Pandas mastery (clean 3 datasets)
- **Day 5-6**: Matplotlib/Seaborn (visualize datasets)
- **Day 7**: Mini-project combining all three

## Week 2: Traditional ML

- **Day 1-2**: Scikit-learn basics (Linear/Logistic Regression)
- **Day 3-4**: Tree-based methods (RF, Gradient Boosting)
- **Day 5-6**: Complete ML pipeline
- **Day 7**: Kaggle competition

## Week 3: Deep Learning Basics

- **Day 1-2**: PyTorch fundamentals
- **Day 3-4**: Build MLP, train on MNIST
- **Day 5-6**: CNNs, train on CIFAR-10
- **Day 7**: Transfer learning project

## Week 4: Advanced & Integration

- **Day 1-2**: Advanced architectures
- **Day 3-4**: Hyperparameter tuning
- **Day 5-6**: End-to-end project
- **Day 7**: Portfolio presentation

---

## Success Metrics

- Can you implement backpropagation from scratch?
- Can you explain every line of your code?
- Can you debug your own models?
- Can you achieve >90% on MNIST, >70% on CIFAR-10?

---

## Final Recommendations

### Daily Routine

1. **Morning (30 min)**: Flashcard review (Anki app)
2. **Midday (2 hours)**: Code implementation
3. **Evening (1 hour)**: Read documentation

### Resources

- **Documentation**: Official docs first, always
- **Courses**: fast.ai, Andrew Ng's ML course
- **Practice**: Kaggle competitions
- **Community**: r/MachineLearning, Papers with Code

### Key Principles

- **Code first, optimize later**
- **Always visualize**
- **Start simple**
- **Debug systematically**
- **Read error messages**

---

## Conclusion

This guide provides a comprehensive foundation in machine learning and deep learning. The key to mastery is **consistent, deliberate practice**. Code every single day, even if just for 30 minutes.

The field of ML/DL is vast and constantly evolving. Use this guide as your foundation, but continue learning through documentation, research papers, and hands-on projects.

Remember: every expert was once a beginner. Stay curious, stay persistent, and keep building.

---

## Contact

**Travis Lelle**  
Email: [travis@travisml.ai](mailto:travis@travisml.ai)

*Keep learning, keep building, keep shipping.*

---

## License

This guide is provided for educational purposes. Feel free to share and adapt with attribution.
