# -*- coding: utf-8 -*-
# Sentiment Classification Pipeline (Complete) - สำหรับการพรีเซนต์โปรเจกต์ INT182
# ใช้ได้บน Google Colab / Jupyter Notebook

# 0. ติดตั้งไลบรารี (รันครั้งแรกบน Colab)
# !pip install -q nltk scikit-learn matplotlib seaborn joblib

# 1. นำเข้าไลบรารี
import pandas as pd
import numpy as np
import re
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder
import joblib

# สำหรับการทำ text cleaning (ใช้ nltk)
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# 2. โหลด dataset
# สมมติไฟล์ชื่อ reviews.csv อยู่ในโฟลเดอร์เดียวกัน
# ไฟล์ต้องมีคอลัมน์: 'review' และ 'sentiment'
df = pd.read_csv('./IMDB Dataset.csv')  # เปลี่ยนชื่อไฟล์ถ้าจำเป็น
print("Dataset shape:", df.shape)
df.head()

# 3. สำรวจข้อมูลเบื้องต้น
print(df['sentiment'].value_counts())
print(df.info())

# 4. Preprocessing / Cleaning function
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    if pd.isna(text):
        return ""
    # ลบ HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    # ลบ non-letters
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    # lowercase
    text = text.lower()
    # tokenization แบบง่าย
    tokens = text.split()
    # remove stopwords + lemmatize + ปรับความยาวคำ
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words and len(t) > 1]
    return " ".join(tokens)

# Apply cleaning (อาจใช้เวลาถ้ามีข้อมูลเยอะ)
df['clean_review'] = df['review'].astype(str).apply(clean_text)

# ดูตัวอย่าง
df[['review','clean_review','sentiment']].head()

# 5. แปลง label เป็นตัวเลข
le = LabelEncoder()
df['label'] = le.fit_transform(df['sentiment'])  # positive->1, negative->0 (ขึ้นกับ dataset)
print("Label mapping:", dict(zip(le.classes_, le.transform(le.classes_))))

# 6. แบ่งข้อมูล Train / Test
X = df['clean_review']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
print("Train:", X_train.shape, "Test:", X_test.shape)

# 7. สร้าง pipelines สำหรับแต่ละโมเดล
pipelines = {
    'lr_tfidf': Pipeline([
        ('tfidf', TfidfVectorizer(max_df=0.9, min_df=2, ngram_range=(1,2))),
        ('clf', LogisticRegression(max_iter=1000))
    ]),
    'nb_tfidf': Pipeline([
        ('tfidf', TfidfVectorizer(max_df=0.9, min_df=2, ngram_range=(1,2))),
        ('clf', MultinomialNB())
    ]),
    'svc_tfidf': Pipeline([
        ('tfidf', TfidfVectorizer(max_df=0.9, min_df=2, ngram_range=(1,2))),
        ('clf', LinearSVC(max_iter=2000))
    ]),
    'rf_count': Pipeline([
        ('count', CountVectorizer(max_df=0.9, min_df=2, ngram_range=(1,2))),
        ('clf', RandomForestClassifier(n_estimators=200, random_state=42))
    ])
}

# 8. เทรนทุกโมเดลเป็น baseline
results = []
for name, pipe in pipelines.items():
    print("Training", name)
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    results.append((name, acc, report))
    print(name, "Accuracy:", acc)
    print(classification_report(y_test, y_pred))

# 9. เปรียบเทียบผลลัพธ์เบื้องต้น
summary = pd.DataFrame([{
    'model': r[0],
    'accuracy': r[1],
    'precision_pos': r[2]['1']['precision'] if '1' in r[2] else None,
    'recall_pos': r[2]['1']['recall'] if '1' in r[2] else None,
    'f1_pos': r[2]['1']['f1-score'] if '1' in r[2] else None
} for r in results])
print(summary.sort_values(by='accuracy', ascending=False))

# 10. Fine-tune (ตัวอย่างใช้ GridSearch สำหรับ LogisticRegression และ MultinomialNB)
param_grid_lr = {
    'tfidf__max_df': [0.8, 0.9],
    'tfidf__min_df': [1, 2],
    'tfidf__ngram_range': [(1,1),(1,2)],
    'clf__C': [0.1, 1, 10]
}
grid_lr = GridSearchCV(pipelines['lr_tfidf'], param_grid_lr, cv=3, n_jobs=-1, verbose=1)
grid_lr.fit(X_train, y_train)
print("Best LR params:", grid_lr.best_params_)
best_lr = grid_lr.best_estimator_

param_grid_nb = {
    'tfidf__max_df': [0.8, 0.9],
    'tfidf__min_df': [1,2],
    'tfidf__ngram_range': [(1,1),(1,2)],
    'clf__alpha': [0.5, 1.0]
}
grid_nb = GridSearchCV(pipelines['nb_tfidf'], param_grid_nb, cv=3, n_jobs=-1, verbose=1)
grid_nb.fit(X_train, y_train)
print("Best NB params:", grid_nb.best_params_)
best_nb = grid_nb.best_estimator_

# 11. ประเมินโมเดลที่ปรับแต่งแล้ว (best_lr, best_nb) และโมเดลอื่นๆ (เช่น svc)
models_to_eval = {
    'LogisticRegression (tuned)': best_lr,
    'MultinomialNB (tuned)': best_nb,
    'LinearSVC': pipelines['svc_tfidf'],
    'RandomForest': pipelines['rf_count']
}

eval_results = {}
for name, model in models_to_eval.items():
    print("Evaluating", name)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    clf_report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    # try ROC-AUC if possible (binary)
    try:
        if hasattr(model, "predict_proba"):
            y_scores = model.predict_proba(X_test)[:,1]
        elif hasattr(model, "decision_function"):
            y_scores = model.decision_function(X_test)
        else:
            y_scores = None
        roc = roc_auc_score(y_test, y_scores) if y_scores is not None else None
    except Exception as e:
        roc = None

    eval_results[name] = {
        'accuracy': acc,
        'report': clf_report,
        'confusion_matrix': cm,
        'roc_auc': roc
    }
    print(name, "Accuracy:", acc)
    print(clf_report)
    print("Confusion matrix:\n", cm)
    print("ROC AUC:", roc)

# 12. วาด Confusion Matrix ของโมเดลที่ดีที่สุด (เลือกตาม accuracy)
best_model_name = max(eval_results.items(), key=lambda x: x[1]['accuracy'])[0]
print("Best model:", best_model_name)
best_model = models_to_eval[best_model_name]

# สร้าง Confusion Matrix plot
y_pred_best = best_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred_best)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title(f"Confusion Matrix - {best_model_name}")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=200)
plt.show()

# 13. ROC curve (ถ้ามีคะแนน)
try:
    if hasattr(best_model, "predict_proba"):
        y_scores = best_model.predict_proba(X_test)[:,1]
    else:
        y_scores = best_model.decision_function(X_test)
    fpr, tpr, _ = roc_curve(y_test, y_scores)
    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, lw=2)
    plt.plot([0,1],[0,1], '--', color='gray')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {best_model_name} (AUC={roc_auc_score(y_test,y_scores):.3f})")
    plt.savefig("roc_curve.png", dpi=200)
    plt.show()
except Exception as e:
    print("ROC not available:", e)

# 14. แสดง Top features (คำที่มีผลบวก/ลบ) — สำหรับ Logistic Regression
if 'LogisticRegression' in best_model_name:
    # extract tfidf and classifier
    tfidf = best_model.named_steps['tfidf']
    clf = best_model.named_steps['clf']
    feature_names = tfidf.get_feature_names_out()
    coefs = clf.coef_[0]
    top_pos_idx = np.argsort(coefs)[-20:][::-1]
    top_neg_idx = np.argsort(coefs)[:20]
    top_pos = [(feature_names[i], coefs[i]) for i in top_pos_idx]
    top_neg = [(feature_names[i], coefs[i]) for i in top_neg_idx]

    print("Top positive features:")
    for f,c in top_pos[:20]:
        print(f, round(c,3))
    print
# Save model
joblib.dump(best_model, 'best_model.pkl')
print("✅ Model saved as best_model.pkl")

# Save label encoder (จำเป็นเวลาโหลดกลับมาใช้)
joblib.dump(le, 'label_encoder.pkl')
print("✅ Label encoder saved as label_encoder.pkl")