# ML-Tools for MATLAB Projects

This repository provides a collection of machine learning tools implemented in MATLAB, covering classification, regression, clustering, and dimensionality reduction techniques.

## 📁 Structure

ml-tools/
├── +kmeans
├── +knn
├── +pca
├── +plotter
├── +regression
├── +svm
└── README.md

## 🚀 Usage Examples

### 1. Linear Regression

```matlab
% Prepare data
X = [engine_size, vehicle_weight];
y = fuel_consumption;

% Create and fit model
model = regression.LinearModel(X, y);
model.fit();

% Predict new values
y_pred = model.predict([2.0, 1500]);

% Evaluate
rmse = model.computeRMSE(X_test, y_test);
r2 = model.r2(X_val, y_val);

% Polynomial features
poly_model = model.createPolyFeatures(1, [1 2 3]); % x, x², x³
```

### 2. Logistic Regression

```matlab
% Binary classification
model = regression.BinaryLogisticModel(X, y);

% Feature scaling and training
model.scaleInputs();
model.trainFminunc(400); % Using fminunc optimizer

% Evaluate
[accuracy, recall, precision] = model.accuracy(X_test, y_test);

% Polynomial decision boundary
Plotter.plotDecisionBoundary(model.B, X, y, 3); % Degree 3 polynomial
```

### 3. K-Nearest Neighbors (KNN)

```matlab
% Create classifier
knn = KNNClassifier(X_train, y_train, 5); % K=5

% Predict
y_pred = knn.predict(X_test);

% Evaluate accuracy
acc = knn.accuracy(X_test, y_test);

% Change K value
knn.setK(7); % Update to 7 neighbors
```

### 4. Support Vector Machine (SVM)

```matlab
% Create SVM model
svm = svmModel(X_train, y_train);
svm.kernelFunction = 'rbf';
svm.boxConstraint = 1;

% Train and optimize
svm.train();
[bestC, bestScale] = svm.optimizeHyperparameters();

% Visualize
svm.plotBoundary();
svm.plotsv();

% Evaluate
[error_rate, recognition_rate] = svm.accuracy(X_test, y_test);
```

### 5. K-Means Clustering

```matlab
% Create and fit model
kmeans = KMeansModel(X, 3); % 3 clusters
kmeans.fit();

% Visualize clusters
kmeans.plotClusters();

% Elbow method to find optimal K
[bestK, costs] = KMeansModel.elbow(X, 1:10, 5); % Test K=1..10 with 5 runs each
```

### 6. Principal Component Analysis (PCA)

```matlab
% Perform PCA
pca = PCA(X);

% Get reduced dimensions (2 components)
Z = pca.getReduced(2);

% Automatic component selection
k = pca.chooseK(0.95); % Keep 95% variance

% Visualize eigenvalues
pca.plotEigenvalues();
```

### 7. Plotting Tools

```matlab
% Scatter plot
Plotter.plotScatter(X(:,1), X(:,2), y);

% Decision boundary
Plotter.plotDecisionBoundary(model.B, X, y);

% Custom polynomial boundary
Plotter.plotDecisionBoundary(beta, X, y, 3, xmin, xmax, ymin, ymax);
```

## ✅ Requirements

* MATLAB (tested on R2021b and later)
* Optimization Toolbox (for fminunc)
* Statistics and Machine Learning Toolbox (for SVM)

## 🧰 Integration

Add the toolbox to your MATLAB path:

```matlab
addpath(genpath('ml-tools'));
```

## 📚 Features

* **Regression**: Linear models with polynomial features
* **Classification**: KNN, SVM, and Logistic Regression
* **Clustering**: K-Means with elbow method
* **Dimensionality Reduction**: PCA with automatic component selection
* **Visualization**: Decision boundaries and cluster plots
