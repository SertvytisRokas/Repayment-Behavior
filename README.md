# DS.v2.5.3.4.1
### Module 3: Machine Learning v2.5
### Machine Learning Capstone Project

# Repayment Behavior Patterns
The relevant project is located in repayment_behavior_predictions.ipynb.

To run the notebook, first install dependencies using `pip`:
```
pip install -r requirements.txt
```
Alternatively, some IDEs, such as VS Code, offer built-in environment management features, allowing you to specify the same requirements.txt file for automatic package installation.


## Introduction
This project aims to develop a machine learning model for predicting loan repayment behavior, focusing on categorizing borrowers into three distinct groups:
- early payers
- on-time payers
- late payers


## Data sources
- `installments_payments`: Contains historical payment data, used to derive target values by comparing scheduled vs. actual payment dates.
- `application_train` and `application_test`: Provide biographical and financial information about borrowers, essential for training and validating the repayment prediction models.

The datasets required for this project can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1dOLUb4B4ZbuvVURcQ7YZIFloaRma8NMA). After downloading, extract the .csv files and place them in the `data/` subdirectory.


## Project structure
- Exploratory data analysis
- Correlation analysis
- Missing values analysis
- Outlier analysis
- Categorical feature encoding
- Numerical data normalization
- Model training
- Hyperparameter tuning
- Feature impact (SHAP) calculation
- Feature reduction
- Missing feature imputation via kNN
- Statistical inference
- Model ensemble
- Model evaluation
- Conclusion
- Potential improvements


## Modelling techniques
- XGBoost (eXtreme Gradient Boosting)
- LightGBM (Light Gradient Boosting)
- Hyperparameter optimization (via Optuna)
- kNN (k-nearest neighbors)
- SMOTE (Synthetic Minority Over-sampling)
- Ensemble learning (via Voting Regressor)
