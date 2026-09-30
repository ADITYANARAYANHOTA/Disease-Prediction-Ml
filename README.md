# Disease Prediction Using Machine Learning

A machine learning classification project using the **Breast Cancer Wisconsin Diagnostic dataset** to classify tumors as **Benign** or **Malignant**. The notebook explores the dataset, preprocesses the features, trains multiple classification models, compares their performance, analyzes feature importance, and saves the trained models for later use.

> **Important:** This is an educational machine learning project. The predictions are model outputs and must not be used as a medical diagnosis or as a substitute for professional clinical evaluation.

## Project Overview

The project builds binary classification models for breast-cancer diagnosis using 30 numerical diagnostic features.

The target labels are defined in the notebook as:

- `0` → **Benign**
- `1` → **Malignant**

The notebook compares four machine learning algorithms:

1. Logistic Regression
2. Support Vector Machine (SVM)
3. Random Forest
4. XGBoost

The models are evaluated using Accuracy, Precision, Recall, F1-Score, and ROC-AUC.

## Objectives

- Load and inspect the Breast Cancer Wisconsin Diagnostic dataset.
- Explore the target distribution and feature relationships.
- Visualize the dataset using statistical plots and a correlation matrix.
- Split the data into training and testing sets.
- Standardize numerical features where required.
- Train multiple classification algorithms.
- Compare model performance using standard evaluation metrics.
- Plot confusion matrices and ROC curves.
- Examine Random Forest feature importance.
- Demonstrate predictions on a sample patient record.
- Identify the model with the highest ROC-AUC.
- Save the trained models as `.pkl` files.

## Dataset

The notebook uses the `load_breast_cancer()` dataset from `sklearn.datasets`.

### Dataset dimensions

- **Total samples:** 569
- **Features:** 30
- **Target:** Binary classification
- **Training samples:** 455
- **Testing samples:** 114
- **Test size:** 20%
- **Train/test split:** `random_state=42`
- **Stratification:** Yes

### Target distribution

After the target labels are converted to the project's convention:

| Target | Meaning | Count |
|---:|---|---:|
| `0` | Benign | 357 |
| `1` | Malignant | 212 |

### Feature groups

The dataset contains measurements grouped into three categories:

- **Mean features**
- **Standard error features**
- **Worst features**

Examples include:

- `mean radius`
- `mean texture`
- `mean perimeter`
- `mean area`
- `mean smoothness`
- `mean compactness`
- `mean concavity`
- `mean concave points`
- `mean symmetry`
- `mean fractal dimension`
- `radius error`
- `texture error`
- `perimeter error`
- `area error`
- `smoothness error`
- `compactness error`
- `concavity error`
- `concave points error`
- `symmetry error`
- `fractal dimension error`
- `worst radius`
- `worst texture`
- `worst perimeter`
- `worst area`
- `worst smoothness`
- `worst compactness`
- `worst concavity`
- `worst concave points`
- `worst symmetry`
- `worst fractal dimension`

## Data Preparation

The notebook performs the following steps:

1. Loads the dataset using `load_breast_cancer()`.
2. Converts the feature matrix into a Pandas DataFrame.
3. Creates the target series.
4. Reverses the original target coding so that:
   - `0 = Benign`
   - `1 = Malignant`
5. Combines the features and target into a single DataFrame for exploration.
6. Checks the dataset shape and information.
7. Examines descriptive statistics.
8. Checks for missing values and duplicate records.
9. Examines the target distribution.
10. Creates a correlation matrix.
11. Splits the data into training and testing sets using stratification.

## Feature Scaling

`StandardScaler` is used for the models that require standardized input:

- Logistic Regression
- SVM

The scaler is fitted only on the training data and then applied to the test data:

```python
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

Random Forest and XGBoost are trained using the original feature values.

## Machine Learning Models

### 1. Logistic Regression

The notebook trains Logistic Regression with:

```python
LogisticRegression(max_iter=5000)
```

### 2. Support Vector Machine

The SVM uses an RBF kernel:

```python
SVC(
    kernel="rbf",
    probability=True,
    random_state=42
)
```

### 3. Random Forest

The Random Forest model uses:

```python
RandomForestClassifier(
    n_estimators=200,
    random_state=42
)
```

### 4. XGBoost

The notebook installs and uses XGBoost with the following configuration:

```python
XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42
)
```

## Model Performance

The results reported by the notebook are:

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9649 | 0.9750 | 0.9286 | 0.9512 | 0.9960 |
| SVM | 0.9737 | 1.0000 | 0.9286 | 0.9630 | 0.9947 |
| Random Forest | 0.9649 | 1.0000 | 0.9048 | 0.9500 | 0.9942 |
| XGBoost | 0.9737 | 1.0000 | 0.9286 | 0.9630 | 0.9957 |

The notebook identifies **Logistic Regression** as the model with the highest ROC-AUC:

```text
ROC-AUC = 0.9960
```

This is a description of the notebook's model-selection criterion and result, not a claim that the model is clinically appropriate for deployment.

## Classification Results

The notebook also generates classification reports for all four models.

### Logistic Regression

- Benign precision: 0.96
- Benign recall: 0.99
- Malignant precision: 0.97
- Malignant recall: 0.93
- Accuracy: 0.96

### SVM

- Benign precision: 0.96
- Benign recall: 1.00
- Malignant precision: 1.00
- Malignant recall: 0.93
- Accuracy: 0.97

### Random Forest

- Benign precision: 0.95
- Benign recall: 1.00
- Malignant precision: 1.00
- Malignant recall: 0.90
- Accuracy: 0.96

### XGBoost

- Benign precision: 0.96
- Benign recall: 1.00
- Malignant precision: 1.00
- Malignant recall: 0.93
- Accuracy: 0.97

## Exploratory Data Analysis

The notebook includes several exploratory analyses and visualizations:

- Dataset shape and structure
- Descriptive statistics
- Missing-value checks
- Duplicate checks
- Target-class distribution
- Feature distributions
- Correlation matrix
- Model accuracy comparison
- Model F1-score comparison
- Model ROC-AUC comparison
- ROC curve comparison

## Confusion Matrix

A confusion matrix is generated for the Random Forest model using:

```python
ConfusionMatrixDisplay(
    confusion_matrix=cm_rf,
    display_labels=["Benign", "Malignant"]
)
```

This helps visualize correct and incorrect predictions for each class.

## ROC Curve Analysis

ROC curves are generated for all four models:

- Logistic Regression
- SVM
- Random Forest
- XGBoost

The curves use the predicted probabilities from each classifier and are compared in a single visualization.

## Feature Importance

The notebook calculates Random Forest feature importance:

```python
feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf_model.feature_importances_
})
```

The 15 most important features are extracted and visualized in a horizontal bar chart.

This provides a model-specific view of which features contributed most to the Random Forest predictions.

## Sample Prediction

The notebook selects one patient from the test set and sends the same sample through all four trained models.

The recorded predictions were:

| Model | Prediction |
|---|---|
| Logistic Regression | Benign |
| SVM | Benign |
| Random Forest | Benign |
| XGBoost | Benign |

This example demonstrates how the trained models can be used to generate predictions on unseen observations.

## Model Selection

The notebook selects the model with the highest ROC-AUC using:

```python
best_model = results.loc[
    results["ROC-AUC"].idxmax(),
    "Model"
]
```

The resulting selection is:

```text
Model: Logistic Regression
ROC-AUC: 0.9960
```

## Saved Models

All four trained models are saved using `joblib`:

```text
logistic_regression_model.pkl
svm_model.pkl
random_forest_model.pkl
xgboost_model.pkl
```

These files allow the trained estimators to be loaded later without retraining.

> If you publish the repository with these model files, make sure their size is compatible with GitHub's repository/file limits. Git LFS can be considered for large model artifacts.

## Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- XGBoost
- Joblib
- Jupyter Notebook

## Installation

Clone the repository:

```bash
git clone https://github.com/<your-username>/<your-repository>.git
cd <your-repository>
```

Install the required Python packages:

```bash
pip install numpy pandas matplotlib scikit-learn xgboost joblib jupyter
```

Alternatively:

```bash
pip install -r requirements.txt
```

## Running the Notebook

Start Jupyter Notebook:

```bash
jupyter notebook
```

Open the project notebook:

```text
ba6988b3-74a7-443e-9907-00a4bc370af4.ipynb
```

Run the cells from top to bottom.

The notebook automatically loads the Breast Cancer Wisconsin Diagnostic dataset through scikit-learn, so a separate CSV dataset file is not required for the current implementation.

## Recommended Repository Structure

```text
breast-cancer-classification/
│
├── ba6988b3-74a7-443e-9907-00a4bc370af4.ipynb
├── logistic_regression_model.pkl
├── svm_model.pkl
├── random_forest_model.pkl
├── xgboost_model.pkl
├── requirements.txt
└── README.md
```

You may rename the notebook to something more descriptive before publishing, for example:

```text
breast_cancer_classification.ipynb
```

## Example: Loading a Saved Model

A saved model can be loaded with Joblib:

```python
import joblib

model = joblib.load("logistic_regression_model.pkl")
```

For Logistic Regression and SVM, remember that new input data must undergo the same scaling procedure used during training. The current notebook saves the classifiers but does not separately save the fitted `StandardScaler`.

## Evaluation Metrics

The project uses:

- **Accuracy** — proportion of all test samples classified correctly.
- **Precision** — proportion of predicted positive cases that are actually positive.
- **Recall** — proportion of actual positive cases detected by the model.
- **F1-Score** — harmonic mean of precision and recall.
- **ROC-AUC** — evaluates discrimination between the two classes using prediction probabilities.

For this dataset, recall for the malignant class is particularly relevant to interpret because missed malignant cases and false positives have different consequences. The notebook reports the metric values but does not perform clinical threshold optimization or cost-sensitive analysis.

## Limitations

This project has several important limitations:

- It uses a relatively small benchmark dataset of 569 samples.
- The evaluation is based on a single 80/20 train-test split.
- No external validation dataset is used.
- The notebook does not perform hyperparameter optimization or cross-validation for the final comparison.
- The saved Logistic Regression and SVM models depend on feature scaling, but the fitted scaler is not saved as a separate artifact.
- Model performance on this benchmark dataset should not be interpreted as clinical performance.
- No clinical workflow, regulatory validation, calibration study, or prospective testing is included.
- The sample prediction is illustrative and must not be used for medical decision-making.

## Future Improvements

Possible extensions include:

- Add stratified k-fold cross-validation.
- Perform systematic hyperparameter tuning.
- Save the fitted `StandardScaler` together with models that require scaling.
- Build complete preprocessing-and-model pipelines.
- Evaluate calibration of predicted probabilities.
- Test the models on an independent external dataset.
- Add explainability methods such as SHAP.
- Perform threshold analysis based on application-specific costs.
- Create a simple web application or API for demonstration purposes.
- Add automated model and data validation.

## Project Workflow

```text
Load Dataset
     ↓
Explore Dataset
     ↓
Prepare Target Labels
     ↓
Check Data Quality
     ↓
Exploratory Data Analysis
     ↓
Train/Test Split
     ↓
Feature Scaling
     ↓
Train Multiple Models
     ↓
Generate Predictions
     ↓
Evaluate Models
     ↓
Compare Metrics
     ↓
ROC / Confusion Matrix Analysis
     ↓
Feature Importance Analysis
     ↓
Sample Prediction
     ↓
Save Trained Models
```

## Conclusion

This project demonstrates a complete machine learning classification workflow for the Breast Cancer Wisconsin Diagnostic dataset. Four classification algorithms—Logistic Regression, SVM, Random Forest, and XGBoost—are trained and compared using multiple evaluation metrics.

The notebook reports the highest ROC-AUC for Logistic Regression at **0.9960**, while SVM and XGBoost both report an accuracy of **0.9737** on the test set.

The project is intended as an educational demonstration of supervised classification, model evaluation, comparison, and model persistence.
