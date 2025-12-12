# Multi-class Classification Using SVM, Random Forest, and Ensemble Methods

## Overview

This project implements and evaluates multiple machine learning classifiers for multi-class classification on the **Dry Bean Dataset**. The project compares Softmax Regression (Multinomial Logistic Regression), Support Vector Machines (SVM), Random Forest, and an Ensemble method, with comprehensive hyperparameter tuning and evaluation metrics.

## Dataset

- **Dataset**: Dry Bean Dataset (Excel format)
- **Total Samples**: 13,611 beans
- **Features**: 16 geometric and shape features
- **Classes**: 7 bean varieties
  - BARBUNYA
  - BOMBAY
  - CALI
  - DERMASON
  - HOROZ
  - SEKER
  - SIRA

### Features

The dataset includes geometric features such as:
- Area, Perimeter, MajorAxisLength, MinorAxisLength
- AspectRation, Eccentricity, ConvexArea
- EquivDiameter, Extent, Solidity
- Roundness, Compactness
- ShapeFactor1, ShapeFactor2, ShapeFactor3, ShapeFactor4

### Data Splits
- **Training Set**: 60% (8,166 samples)
- **Validation Set**: 20% (2,722 samples)
- **Test Set**: 20% (2,723 samples)
- **Stratification**: Applied to maintain class distribution across splits

## Methodology

### 1. Exploratory Data Analysis

#### Feature Analysis
- **Statistical Summary**: Descriptive statistics for all features
- **Distribution Analysis**: Histograms for feature distributions
- **Missing Values**: No missing values detected

#### Feature-Feature Relationships
- **Correlation Matrix**: Pearson correlation heatmap showing inter-feature relationships
- **Key Finding**: Some features are highly correlated, while others show negative correlations, indicating clear class-identifying characteristics

#### Feature-Label Relationships
- **ANOVA F-test**: Statistical test to measure feature-label associations
- **Top Features** (by F-score):
  1. Area (F=17,424)
  2. ConvexArea (F=17,411)
  3. EquivDiameter (F=15,250)
  4. Perimeter (F=14,575)
  5. MinorAxisLength (F=13,586)
- **Visualization**: Feature importance bar plots and scatter plots (class-colored)

### 2. Data Preprocessing

- **Standardization**: StandardScaler applied to all features
- **Label Encoding**: Text labels encoded to numerical values
- **Train/Val/Test Split**: Stratified split to maintain class balance

### 3. Classification Models

#### A. Softmax Regression (Multinomial Logistic Regression)

**Hyperparameter Tuning:**
- **C (Regularization)**: [0.01, 0.1, 1, 10]
- **Solver**: ['lbfgs', 'saga']
- **Max Iterations**: [100, 200, 500, 700, 1000]

**Best Configuration:**
- C: 10
- Solver: 'saga'
- Max Iterations: 100
- **Best Validation F1-Score**: 0.9255

**Key Insights:**
- Higher C values (1, 10) performed better, allowing more model flexibility
- 'lbfgs' solver converged faster with stable scores
- Max iterations beyond 100 showed no improvement

#### B. Support Vector Machine (SVM)

**Hyperparameter Tuning:**
- **C (Regularization)**: [0.1, 1, 10, 100]
- **Kernel**: ['linear', 'rbf', 'poly']
- **Gamma**: ['scale', 'auto', 0.001, 0.01, 0.1] (for RBF/poly)

**Best Configuration:**
- Selected through grid search on validation set
- Evaluated using weighted F1-score

**Key Insights:**
- Different kernels capture different decision boundaries
- RBF kernel often captures non-linear relationships effectively
- C parameter balances margin width and classification errors

#### C. Random Forest Classifier

**Hyperparameter Tuning:**
- **N Estimators**: [50, 100, 200, 300]
- **Max Depth**: [10, 20, 30, None]
- **Min Samples Split**: [2, 5, 10]
- **Min Samples Leaf**: [1, 2, 4]

**Best Configuration:**
- Selected through grid search on validation set
- Evaluated using weighted F1-score

**Feature Importance:**
- Random Forest provides intrinsic feature importance scores
- Top features align with ANOVA F-test results

**Key Insights:**
- Ensemble of decision trees reduces overfitting
- Feature importance reveals most discriminative attributes
- Deeper trees may capture complex patterns but risk overfitting

#### D. Ensemble Method (Voting Classifier)

**Configuration:**
- Combines best models from Softmax, SVM, and Random Forest
- **Voting Strategy**: Hard voting or soft voting
- Trained on combined training and validation sets
- Final evaluation on test set

**Key Insights:**
- Ensemble leverages strengths of individual classifiers
- Often achieves better generalization than individual models
- Reduces variance through model diversity

### 4. Model Evaluation

#### Metrics Used
- **Accuracy**: Overall classification accuracy
- **Precision**: Weighted precision score
- **Recall**: Weighted recall score
- **F1-Score**: Weighted F1-score (handles class imbalance)

#### Evaluation Strategy
1. **Training Set**: Model performance on training data
2. **Validation Set**: Hyperparameter selection and model comparison
3. **Test Set**: Final unbiased performance evaluation

#### Visualizations
- **Confusion Matrices**: For train, validation, and test sets
- **Classification Reports**: Detailed per-class metrics
- **Feature Importance Plots**: Random Forest feature contributions

## Technologies Used

- **Python 3.x**
- **NumPy & Pandas**: Data manipulation and analysis
- **Scikit-learn**:
  - `LogisticRegression`: Softmax regression
  - `SVC`: Support Vector Machine
  - `RandomForestClassifier`: Random Forest
  - `VotingClassifier`: Ensemble method
  - `StandardScaler`: Feature scaling
  - `LabelEncoder`: Label encoding
  - `train_test_split`: Data splitting
  - `f_classif`: ANOVA F-test
  - `ConfusionMatrixDisplay`: Confusion matrix visualization
- **Matplotlib & Seaborn**: Data visualization
- **SciPy**: Statistical functions

## Installation

```bash
# Clone the repository
git clone https://github.com/prabha-07/Multi-class-classification-using-SVM-RF-Ensemble-and-their-evaluation.git

# Install required packages
pip install numpy pandas scikit-learn matplotlib seaborn scipy openpyxl
```

## Usage

1. **Prepare your dataset**: Ensure your Excel file follows the same structure
2. **Update the file path**: Modify the path in `pd.read_excel()` to point to your dataset
3. **Run the notebook**: Execute cells sequentially to:
   - Load and explore the data
   - Perform feature analysis
   - Train classification models
   - Evaluate and compare performance

## Key Results

### Feature Analysis
- **Top Discriminative Features**: Area, ConvexArea, EquivDiameter, Perimeter
- **Strong Feature-Label Associations**: All features show significant F-scores (p < 0.05)
- **Feature Correlations**: Clear patterns indicating class separability

### Model Performance
- **Softmax Regression**: Validation F1-Score: 0.9255
- **SVM**: Performance evaluated through comprehensive grid search
- **Random Forest**: Feature importance analysis and performance evaluation
- **Ensemble**: Combined model performance on test set

## Key Contributions

- Comprehensive comparison of three different classification paradigms
- Rigorous hyperparameter tuning for each model
- Statistical feature analysis using ANOVA F-test
- Ensemble method combining multiple classifiers
- Detailed evaluation with confusion matrices and classification reports
- Feature importance analysis

## Evaluation Highlights

1. **Statistical Rigor**: ANOVA F-test for feature selection
2. **Hyperparameter Optimization**: Grid search with validation set
3. **Comprehensive Metrics**: Accuracy, Precision, Recall, F1-Score
4. **Visual Analysis**: Confusion matrices, feature importance plots
5. **Ensemble Learning**: Voting classifier combining multiple models

## Future Enhancements

- Deep learning models (Neural Networks)
- Advanced ensemble methods (Stacking, Boosting)
- Feature engineering and selection techniques
- Cross-validation for more robust evaluation
- Model interpretation techniques (SHAP, LIME)
- Handling class imbalance with SMOTE or class weights

## License

This project is open source and available for educational purposes.

## Author

**prabha-07**

---

*For questions or contributions, please open an issue or submit a pull request.*

