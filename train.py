import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import joblib
import warnings

warnings.filterwarnings("ignore")

# === 1. Load Data ===
df = pd.read_csv("./crypto.csv")
df['Date'] = pd.to_datetime(df['Date'])

# === 2. Select Ticker ===
ticker = "XRP-USD"
df = df[df['ticker'] == ticker]
df = df.sort_values(by='Date')

# === 3. Feature Engineering ===
df['Price_Change'] = df['Close'] - df['Open']
df['Price_Change_pct'] = (df['Close'] - df['Open']) / df['Open'] * 100
df['MA_7'] = df['Close'].rolling(window=7).mean()
df['MA_30'] = df['Close'].rolling(window=30).mean()
df['High_Low_Diff'] = df['High'] - df['Low']

# === 4. Create Target ===
df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
df.dropna(inplace=True)

# === 5. Prepare Dataset ===
features = ['Open', 'High', 'Low', 'Close', 'Volume',
            'Price_Change', 'Price_Change_pct', 'MA_7', 'MA_30', 'High_Low_Diff']
X = df[features]
y = df['Target']

# === 6. Train/Test Split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# === 7. SMOTE for Imbalanced Data ===
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# === 8. Hyperparameter Tuning ===
param_grid = {
    "n_estimators": [100, 300],
    "max_depth": [4, 6],
    "learning_rate": [0.01, 0.1],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0]
}

xgb_model = xgb.XGBClassifier(
    eval_metric="logloss",
    use_label_encoder=False,
    tree_method="gpu_hist",
    random_state=42
)

grid_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    cv=3,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train_res, y_train_res)

best_params = grid_search.best_params_

final_model = xgb.XGBClassifier(
    **best_params,
    eval_metric="logloss",
    use_label_encoder=False,
    tree_method="gpu_hist",
    random_state=42
)
# early stop
final_model.fit(
    X_train_res, y_train_res,
    eval_set=[(X_test, y_test)],
    early_stopping_rounds=10,
    verbose=True
)
y_pred = final_model.predict(X_test)
y_pred_proba = final_model.predict_proba(X_test)[:, 1]

print("\n=== Best Hyperparameters ===")
print(best_params)
print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred))
print("ROC AUC Score:", roc_auc_score(y_test, y_pred_proba))

joblib.dump(final_model, "best_xgboost.pkl")
print("\nTraining Complete!")
