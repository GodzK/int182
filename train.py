import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

# === STEP 1: โหลดข้อมูล ===
data = pd.read_csv('stocks.csv')

# === STEP 2: สร้าง target (ราคาขึ้น = 1, ลงหรือเท่าเดิม = 0) ===
data['target'] = (data['Close'].shift(-1) > data['Close']).astype(int)
data = data[:-1]  # ลบแถวสุดท้ายที่ไม่มีข้อมูลวันถัดไป

# === STEP 3: สร้างฟีเจอร์ทางเทคนิค (Technical Indicators) ===
data['SMA_5'] = data['Close'].rolling(window=5).mean()
data['SMA_10'] = data['Close'].rolling(window=10).mean()
data['EMA_5'] = data['Close'].ewm(span=5, adjust=False).mean()
data['Momentum'] = data['Close'] - data['Close'].shift(5)
data['Volatility'] = data['Close'].rolling(window=5).std()

# === STEP 4: สร้าง Lag Features ===
data['Close_lag1'] = data['Close'].shift(1)
data['Close_lag2'] = data['Close'].shift(2)
data['Volume_lag1'] = data['Volume'].shift(1)

# === STEP 5: ลบข้อมูลที่มี NaN หลังจาก rolling/shift ===
data.dropna(inplace=True)

# === STEP 6: เตรียมข้อมูลสำหรับโมเดล ===
X = data.drop(columns=['Date', 'Company', 'target'])
y = data['target']

print("🔍 ก่อนใช้ SMOTE:")
print(y.value_counts())

# === STEP 7: แบ่งข้อมูล ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# === STEP 8: แก้ปัญหาข้อมูลไม่สมดุลด้วย SMOTE ===
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

print("\n✅ หลังใช้ SMOTE:")
print(pd.Series(y_train_res).value_counts())

# === STEP 9: Scaling ===
scaler = StandardScaler()
X_train_res = scaler.fit_transform(X_train_res)
X_test = scaler.transform(X_test)

# === STEP 10: ปรับพารามิเตอร์ด้วย GridSearchCV (XGBoost) ===
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
grid = GridSearchCV(model, param_grid, scoring='roc_auc', cv=5, n_jobs=-1)
grid.fit(X_train_res, y_train_res)

print("\n🏆 พารามิเตอร์ที่ดีที่สุด:", grid.best_params_)

best_model = grid.best_estimator_

# === STEP 11: ประเมินผลโมเดล ===
y_pred = best_model.predict(X_test)
y_pred_proba = best_model.predict_proba(X_test)[:, 1]

print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))

print("\n📌 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

roc_auc = roc_auc_score(y_test, y_pred_proba)
print("\n🔥 ROC AUC Score:", roc_auc)
