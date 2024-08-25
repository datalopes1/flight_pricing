# %% Imports

# Data Manipulation
import pandas as pd
import numpy as np

# Machine Learning
from xgboost import XGBRegressor

# Pre-processing
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from category_encoders import TargetEncoder
# %% Load and Prepare Data
df = pd.read_csv("../../data/raw/Clean_Dataset.csv")
df = df.drop(columns=['Unnamed: 0', 'flight'], axis=1)

features = df.drop(columns='price', axis=1).columns.to_list()
target = 'price'

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=21)
# %% Preprocessing Pipelines

num_features = X_train.select_dtypes(include='number').columns.to_list()
cat_features = X_train.select_dtypes(exclude='number').columns.to_list()

cat_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy="most_frequent")),
    ('encoder', TargetEncoder())
])

num_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', cat_transformer, cat_features),
        ('num', num_transformer, num_features)
    ])

# %% Model Training
best_params = {
    'n_estimators': 956, 
    'learning_rate': 0.08885057738863303, 
    'max_depth': 10, 
    'subsample': 0.560743995211729, 
    'colsample_bytree': 0.9985920323190209, 
    'min_child_weight': 9
}

reg = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', XGBRegressor(**best_params, random_state=21))
])

reg.fit(X_train, y_train)
# %% Model Evaluation
y_pred = reg.predict(X_test)

model_metrics = {
    'MSE': metrics.mean_squared_error(y_test, y_pred),
    'RMSE': metrics.mean_squared_error(y_test, y_pred, squared=False),
    'MAE': metrics.mean_absolute_error(y_test, y_pred),
    'MAPE': metrics.mean_absolute_percentage_error(y_test, y_pred),
    'R2 Score': metrics.r2_score(y_test, y_pred)
}

print(model_metrics)
# %% Save Model and Metrics
model_series = pd.Series({
    'model': reg,
    'features': features,
    'metrics': model_metrics
})

model_series.to_pickle("../../models/model.pkl")