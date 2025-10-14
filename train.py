# train.py

# 1. Import Libraries ที่จำเป็น
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("--- [1/6] เริ่มกระบวนการโหลดข้อมูล ---")
# 2. โหลด Dataset
student = fetch_ucirepo(id=320)
X = student.data.features
y = student.data.targets
df = pd.concat([X, y], axis=1)
print("✅ โหลดข้อมูลสำเร็จ")

print("\n--- [2/6] เริ่มกระบวนการเตรียมข้อมูล ---")
# 3. เตรียมข้อมูล (Data Preprocessing)
# สร้าง Target Variable ใหม่ (สอบผ่าน/ไม่ผ่าน ที่เกรด 10)
df['passed'] = np.where(df['G3'] >= 10, 1, 0)

# ลบคอลัมน์ที่ไม่ต้องการ (G1, G2 คือข้อมูลที่ใกล้เคียงเฉลยเกินไป, G3 คือเฉลย)
df = df.drop(['G1', 'G2', 'G3'], axis=1)

# แยก Features (X) และ Target (y) ใหม่อีกครั้ง
X = df.drop(columns=['passed'])
y = df['passed']

# แปลงข้อมูล Categorical เป็นตัวเลข และจัดเก็บ Columns ไว้ใช้ตอน Predict
X_encoded = pd.get_dummies(X, drop_first=True)
model_columns = X_encoded.columns
joblib.dump(model_columns, 'model_columns.pkl') # บันทึกรายชื่อคอลัมน์

print("✅ เตรียมข้อมูลสำเร็จ")

print("\n--- [3/6] เริ่มแบ่งข้อมูลสำหรับ Train และ Test ---")
# 4. แบ่งข้อมูล
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
print(f"ขนาด Training set: {X_train.shape}")
print(f"ขนาด Testing set: {X_test.shape}")

print("\n--- [4/6] เริ่มการฝึกสอนโมเดล ---")
# 5. สร้างและ Train โมเดล
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
print("✅ ฝึกสอนโมเดลสำเร็จ")

print("\n--- [5/6] เริ่มการประเมินผลโมเดล ---")
# 6. ประเมินผลโมเดล
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Fail (0)', 'Pass (1)']))

# สร้างและบันทึก Confusion Matrix เป็นไฟล์ภาพ
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Fail (0)', 'Pass (1)'],
            yticklabels=['Fail (0)', 'Pass (1)'])
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.title("Confusion Matrix")
plt.savefig('confusion_matrix.png') # บันทึกเป็นไฟล์ .png
print("\n✅ Confusion Matrix ถูกบันทึกเป็นไฟล์ 'confusion_matrix.png'")

print("\n--- [6/6] เริ่มการบันทึกโมเดล ---")
# 7. บันทึกโมเดลที่ Train เสร็จแล้ว
joblib.dump(model, 'student_classifier.pkl')
print("✅ โมเดลถูกบันทึกเป็นไฟล์ 'student_classifier.pkl'")
print("\n🎉 กระบวนการทั้งหมดเสร็จสิ้น!")