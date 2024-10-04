# Essential libraries
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import joblib
from tqdm import tqdm
import math
import re
from IPython.display import display

# Profiling and data exploration
from ydata_profiling import ProfileReport

# Statistical methods
from scipy.stats import chi2_contingency, pearsonr, f_oneway, ks_2samp

# Preprocessing
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer
from sklearn.compose import ColumnTransformer

# Modelling
import optuna
import xgboost as xgb
import lightgbm as lgb
from optuna.integration import XGBoostPruningCallback
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import VotingRegressor
from sklearn.base import RegressorMixin, BaseEstimator
from lightgbm import LGBMRegressor

# Imputation
from sklearn.impute import KNNImputer

# Resampling for imbalanced data
from imblearn.over_sampling import SMOTE

# Distributed computing
import dask.dataframe as dd
from dask.diagnostics import ProgressBar

# SHAP for model interpretability
import shap

from functions import *

# Suppress warnings
warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.ERROR)


def clean_column_names(df):
    """Removes or replaces special characters in column names."""
    df.columns = df.columns.str.replace(r'[^\w\s]', '', regex=True)
    df.columns = df.columns.str.replace(' ', '_')
    return df


def calculate_iqr_outliers(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    return outliers, lower_bound, upper_bound


def cap_outliers(df, columns, percentile=0.99):
    for column in columns:
        cap_value = df[column].quantile(percentile)
        df[column] = np.where(df[column] > cap_value, cap_value, df[column])
        print(f"Capped {column} at {cap_value}")
    
    return df

def plot_histograms_grid(df, columns):
    n_cols = 4
    n_rows = int(np.ceil(len(columns) / n_cols))
    
    plt.figure(figsize=(16, 4 * n_rows))
    
    for i, col in enumerate(columns):
        plt.subplot(n_rows, n_cols, i + 1)
        sns.histplot(df[col], kde=True, bins=50, color='blue')
        plt.title(f'Histogram of {col}')
        plt.xlabel(col)
        plt.ylabel('Count')
    
    plt.tight_layout()
    plt.show()


def summarize_numerical_columns(df, numerical_columns):
    summary_stats = []

    for col in numerical_columns:
        col_skewness = df[col].skew()
        col_kurtosis = df[col].kurtosis()
        col_min = df[col].min()
        col_max = df[col].max()
        col_mean = df[col].mean()
        col_std = df[col].std()

        summary_stats.append({
            'Column': col,
            'Skewness': col_skewness,
            'Kurtosis': col_kurtosis,
            'Min': col_min,
            'Max': col_max,
            'Mean': col_mean,
            'Std': col_std
        })

    summary_df = pd.DataFrame(summary_stats)
    return summary_df


def normalize_data(X, scalers, columns_to_scale):
    """
    Normalize the data using different scalers for specified columns.
    
    Parameters:
    - X: DataFrame containing the data to normalize (can be either training or testing)
    - scalers: Dictionary mapping scaler names to scaler objects (e.g., StandardScaler)
    - columns_to_scale: Dictionary mapping scaler names to lists of columns to scale with each scaler
    
    Returns:
    - X: Normalized data
    """
    for scaler_name, columns in columns_to_scale.items():
        scaler = scalers[scaler_name]
        columns_to_normalize = [col for col in columns if col in X.columns]
        X[columns_to_normalize] = scaler.fit_transform(X[columns_to_normalize])
    
    return X


def train_test_split_data(X, y, test_size=0.3, random_state=42):
    """Split the data into train and test sets."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def calculate_metrics(y_true, y_pred):
    """Calculate RMSE and R² metrics."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return rmse, r2


def train_model(model_type: str, X_train, y_train, X_test, y_test, params: dict, num_boost_round: int = 100):
    if model_type == 'xgb':
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)
        model = xgb.train(params, dtrain, num_boost_round=num_boost_round, evals=[(dtest, 'eval')], early_stopping_rounds=10, verbose_eval=False)
        y_pred = model.predict(dtest)
    
    elif model_type == 'lgb':
        dtrain = lgb.Dataset(X_train, label=y_train)
        dvalid = lgb.Dataset(X_test, label=y_test, reference=dtrain)
        model = lgb.train(params, dtrain, num_boost_round=num_boost_round, valid_sets=[dvalid])
        y_pred = model.predict(X_test)

    return model, y_pred


def optimize_with_optuna(model_type: str, X_train, y_train, n_trials: int = 50, baseline_rmse=None):
    def objective(trial):
        if model_type == 'xgb':
            param = {
                'objective': 'reg:squarederror',
                'eval_metric': 'rmse',
                'tree_method': 'hist',
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True),
                'nthread': 4
            }
            
            dtrain = xgb.DMatrix(X_train, label=y_train)
            model = xgb.train(param, dtrain, num_boost_round=100)
            y_pred = model.predict(dtrain)
        
        elif model_type == 'lgb':
            param = {
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'gbdt',
                'num_leaves': trial.suggest_int('num_leaves', 20, 150),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True),
                'nthread': 4
            }
            
            model, _ = train_model(model_type, X_train, y_train, X_train, y_train, param, num_boost_round=100)
            y_pred = model.predict(X_train)
        
        rmse = np.sqrt(mean_squared_error(y_train, y_pred))
        
        if baseline_rmse and rmse >= baseline_rmse * 1.10:
            raise optuna.exceptions.TrialPruned()
        
        return rmse

    study = optuna.create_study(direction='minimize', pruner=optuna.pruners.MedianPruner())
    
    for _ in tqdm(range(n_trials), desc="Trials", position=0, leave=True):
        study.optimize(objective, n_trials=1)

    return study.best_params


def display_results_table(metrics: dict):
    metrics_df = pd.DataFrame.from_dict(metrics, orient='index').reset_index()
    metrics_df.columns = ['Target', 'RMSE', 'R2']
    display(metrics_df)


def analyze_missing_data(df):
    missing_summary = [
        {'Column': column, 
         'Missing_Count': df[column].isnull().sum(), 
         'Missing_Percentage': (df[column].isnull().sum() / df.shape[0]) * 100}
        for column in df.columns
    ]
    missing_summary_df = pd.DataFrame(missing_summary)
    return missing_summary_df[missing_summary_df['Missing_Count'] > 0].sort_values(by='Missing_Percentage', ascending=False)


def plot_missing_histogram(df, threshold=0.95):
    missing_per_observation = df.isnull().sum(axis=1)
    threshold_95 = missing_per_observation.quantile(threshold)

    plt.figure(figsize=(12, 6))
    plt.hist(missing_per_observation, bins=30, edgecolor='k', alpha=0.7)
    plt.axvline(threshold_95, color='r', linestyle='dashed', linewidth=2, label=f'95th Percentile: {threshold_95:.2f}')
    plt.xlabel('Number of Missing Features')
    plt.ylabel('Number of Observations')
    plt.title('Distribution of Missing Features per Observation')
    plt.xticks(np.arange(0, missing_per_observation.max() + 1, 5))
    plt.legend()
    plt.show()


def filter_missing_data(X, y, threshold=40):
    initial_total_observations = X.shape[0]
    initial_avg_missing_percentage = (X.isnull().mean().mean()) * 100
    X_filtered = X[X.isnull().sum(axis=1) < threshold]
    y_filtered = y.loc[X_filtered.index]
    
    final_total_observations = X_filtered.shape[0]
    final_avg_missing_percentage = (X_filtered.isnull().mean().mean()) * 100
    percentage_dropped = ((initial_total_observations - final_total_observations) / initial_total_observations) * 100

    print(f"Initial total number of observations: {initial_total_observations}")
    print(f"Final total number of observations: {final_total_observations}")
    print(f"Percentage of data dropped: {percentage_dropped:.2f}%")
    print(f"Average percentage of missing features (initial): {initial_avg_missing_percentage:.2f}%")
    print(f"Average percentage of missing features (after filtering): {final_avg_missing_percentage:.2f}%")
    
    return X_filtered.reset_index(drop=True), y_filtered.reset_index(drop=True)


def optimize_knn(df, k_range=range(1, 11)):
    X_sampled, _ = train_test_split(df, test_size=0.95, random_state=42)
    errors = []
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for k in k_range:
        continuous_error_sum = 0
        for train_index, test_index in kf.split(X_sampled):
            train_data, test_data = X_sampled.iloc[train_index], X_sampled.iloc[test_index]
            imputer = KNNImputer(n_neighbors=k)
            imputer.fit(train_data)
            imputed_data = pd.DataFrame(imputer.transform(test_data), columns=test_data.columns, index=test_data.index)
            continuous_error = np.nanmean((test_data - imputed_data) ** 2)
            continuous_error_sum += continuous_error

        errors.append((k, continuous_error_sum / kf.get_n_splits()))

    errors_df = pd.DataFrame(errors, columns=['k', 'continuous_error'])
    optimal_k = errors_df.loc[errors_df['continuous_error'].idxmin(), 'k']
    print(f"Optimal number of neighbors: {optimal_k}")
    return optimal_k


def knn_impute_dask(df, n_partitions=16, optimal_k=1):
    df_dask = dd.from_pandas(df, npartitions=n_partitions)

    def knn_impute(data_chunk):
        imputer = KNNImputer(n_neighbors=optimal_k)
        return pd.DataFrame(imputer.fit_transform(data_chunk), columns=data_chunk.columns)

    with ProgressBar():
        with tqdm(total=df_dask.npartitions) as pbar:
            def update_progress(future):
                pbar.update()

            df_imputed_dask = df_dask.map_partitions(knn_impute, meta=df).persist()
            df_imputed_dask.compute(callback=update_progress)
    return df_imputed_dask.compute()


def evaluate_imputation_effectiveness(original_df, imputed_df):
    evaluation_summary = []
    skipped_features = []

    def identify_feature_type(column_data):
        unique_values = column_data.nunique()
        if unique_values == 2 and set(column_data.unique()).issubset({0, 1}):
            return 'binary'
        elif unique_values > 20:
            return 'continuous'
        else:
            return 'categorical'

    for column in original_df.columns:
        feature_type = identify_feature_type(original_df[column])
        original_values = original_df[column].dropna()
        imputed_values = imputed_df[column]

        try:
            if feature_type == 'continuous':
                stat, p_value = ks_2samp(original_values, imputed_values)
            elif feature_type in ['binary', 'categorical']:
                contingency_table = np.array([
                    [(original_values == 0).sum(), (original_values == 1).sum()],
                    [(imputed_values == 0).sum(), (imputed_values == 1).sum()]
                ])
                stat, p_value, _, _ = chi2_contingency(contingency_table)
            else:
                continue

            evaluation_summary.append({
                'Column': column,
                'Feature_Type': feature_type,
                'Statistic': stat,
                'P_Value': p_value,
                'Significant_Difference': 'Yes' if p_value < 0.05 else 'No'
            })

        except ValueError as e:
            print(f"Skipping {column} due to zero frequencies in the contingency table.")
            skipped_features.append(column)

    evaluation_summary_df = pd.DataFrame(evaluation_summary)
    print(evaluation_summary_df.sort_values(by='P_Value').to_string(index=False))

    return evaluation_summary_df, skipped_features


def plot_imputation_success(missing_summary_df, evaluation_summary_df):
    merged_df = pd.merge(missing_summary_df, evaluation_summary_df, on='Column', how='inner')

    fig = px.scatter(
        merged_df,
        x='Missing_Percentage',
        y='P_Value',
        color='Significant_Difference',
        hover_name='Column',
        title='Impact of Missing Percentage on Imputation Success',
        labels={'Missing_Percentage': 'Missing Percentage', 'P_Value': 'P-Value from Statistical Test'},
        size_max=10
    )

    fig.update_traces(marker=dict(size=10))
    fig.show()

    significance_counts = evaluation_summary_df['Significant_Difference'].value_counts()
    total_count = len(evaluation_summary_df)
    percentages = [(count / total_count) * 100 for count in significance_counts.values]

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(significance_counts.index, significance_counts.values, color=['blue', 'red'])

    for bar, percentage in zip(bars, percentages):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{percentage:.1f}%', ha='center', va='bottom')

    ax.set_xlabel('Significant Difference')
    ax.set_ylabel('Count')
    ax.set_title('Significant Difference in Imputation: Counts and Percentages')
    plt.tight_layout()
    plt.show()

    