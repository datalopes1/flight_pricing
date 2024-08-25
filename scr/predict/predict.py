# %% Imports

import pandas as pd
# %% Load Data and Model

df = pd.read_csv("../../data/raw/Clean_Dataset.csv")
model_series = pd.read_pickle("../../models/model.pkl")
# %% Predictions
X = df[model_series['features']]
y_pred = model_series['model'].predict(X)

df['pred_price'] = y_pred
# %% Save Results

results = df[['flight', 'airline', 'price', 'pred_price']].copy()
results.to_excel("../../data/processed/model_predictions.xlsx", index=False)