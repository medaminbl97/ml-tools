# ML-Tools for MATLAB Projects

This repository provides a compact set of tools for working with linear and logistic regression in MATLAB. The code is designed to be easily integrated into your existing machine learning projects, especially for practical courses like *Machine Learning Praktikum*.

## 📁 Structure

```
ml-tools/
├── plotter/
└── regression/
```

* `regression/` contains implementations of:

  * `LinearModel` – for simple and multivariate linear regression
  * `BinaryLogisticModel` – for binary classification using logistic regression

* `plotter/` provides:

  * `Plotter` – a helper class to visualize decision boundaries and data scatter plots

## 🚀 Usage

### 1. Linear Regression

```matlab
% Prepare data (bias will be handled automatically)
X = [x3, x4];
y = fuel_consumption;

% Create model
model = regression.LinearModel(X, y, zeros(size(X,2) + 1, 1));

% Fit using normal equation
model.fit();

% Predict and evaluate (include bias term manually in evaluation vector)
y_hat = model.evaluate([1, 100, 1500]);
r2_score = model.r2();
```

### 2. Logistic Regression

```matlab
% Assume X has 2 features (bias will be added automatically)
model = regression.BinaryLogisticModel(X, y);

% Feature scaling
model.scaleInputs();

% Train with gradient descent
model.trainGradientDescent(0.1, 1000);

% Or train with fminunc
final_cost = model.trainFminunc(400, true);  % with logging enabled

% Accuracy on training data
acc = model.accuracy(model.X, model.y);
```

### 3. Training with `fminunc`

The `BinaryLogisticModel` class includes a convenient method `trainFminunc` that uses MATLAB’s `fminunc` optimizer internally to minimize the logistic regression cost function.

#### Example:

```matlab
model = regression.BinaryLogisticModel(X, y);
model.scaleInputs();
final_cost = model.trainFminunc(400, true);
```

This method:

* Automatically sets up `fminunc` with trust-region algorithm
* Uses `computeCost` for both cost and gradient
* Returns the final cost and updates `model.B`

There is no need to call `fminunc` manually unless you want full control of the optimization process.

### 4. Plotting

For 2D data (2 input features), use the `Plotter`:

```matlab
Plotter.plotScatter(X(:,1), X(:,2), y);
Plotter.plotDecisionBoundary(model.B, model.X, model.y);
```

## ✅ Requirements

* MATLAB (tested on R2021b and later)
* Optimization Toolbox (for `fminunc`)

## 🧰 Integration

Place the `ml-tools/` folder into your MATLAB project and add it to your path:

```matlab
addpath(genpath('ml-tools'));
```

You can now use all regression models and plotting tools directly in your project scripts or functions.

---

Let us know if you need help extending this for polynomial regression, regularization, or neural networks!
