# Step 1 & 2: โหลด dataset (เหมือนเดิม)
from ucimlrepo import fetch_ucirepo
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression # << เปลี่ยนจาก LinearRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report # << เปลี่ยนชุดประเมินผล
import joblib
# โหลด dataset
student = fetch_ucirepo(id=320)

# แยก Features (X) และ Target (y)
X = student.data.features
y = student.data.targets

print("✅ Loaded successfully!")

# Step 3: รวม features และ target เข้าด้วยกัน (เหมือนเดิม)
df = pd.concat([X, y], axis=1)

# Step 4: [เปลี่ยน!] สร้าง Target Variable ใหม่สำหรับ Classification
# เราจะเปลี่ยนโจทย์เป็นการทำนายว่า "สอบผ่าน" หรือ "สอบไม่ผ่าน"
# โดยกำหนดให้เกรด G3 ตั้งแต่ 10 ขึ้นไปคือ "ผ่าน" (แทนด้วยเลข 1)
# และน้อยกว่า 10 คือ "ไม่ผ่าน" (แทนด้วยเลข 0)
df['passed'] = np.where(df['G3'] >= 10, 1, 0)

print("\nดูตัวอย่างข้อมูลหลังเพิ่มคอลัมน์ 'passed':")
print(df[['G3', 'passed']].head())


# Step 5: ลบข้อมูลที่ไม่ต้องการ (เหมือนเดิม + เพิ่ม G3)
# นอกจาก G1, G2 เราจะลบ G3 ออกด้วย เพราะมันคือเฉลยของคอลัมน์ 'passed' ที่เราเพิ่งสร้าง
df = df.drop(['G1', 'G2', 'G3'], axis=1)


# Step 6: แยก features (X) และ target (y) ใหม่อีกครั้ง
X = df.drop(columns=['passed'])
y = df['passed'] # << target ของเราตอนนี้คือ 'passed'


# Step 7: แปลงข้อมูลประเภท Categorical เป็นตัวเลข (เหมือนเดิม)
X = pd.get_dummies(X, drop_first=True)
print("\nหลังจากแปลงข้อมูลแล้ว:", X.shape)


# Step 8: แบ่งข้อมูล Train/Test (เหมือนเดิม)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training set:", X_train.shape)
print("Testing set:", X_test.shape)


# Step 9: [เปลี่ยน!] สร้างและ Train โมเดล (Logistic Regression)
# Logistic Regression เป็นโมเดลพื้นฐานที่นิยมใช้สำหรับงาน Classification
model = LogisticRegression(max_iter=1000) # เพิ่ม max_iter เพื่อให้แน่ใจว่าโมเดลเรียนรู้จนเสร็จ
model.fit(X_train, y_train)

print("\n✅ Classification Model trained successfully!")

joblib.dump(model, 'student_pass_classifier.pkl')
print("✅ Model has been saved to student_pass_classifier.pkl")

# Step 10: [เปลี่ยน!] ประเมินผลโมเดลสำหรับ Classification
y_pred = model.predict(X_test)

# Accuracy: ความแม่นยำโดยรวม
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.2f}")

# Classification Report: ดูค่า Precision, Recall, F1-score ของแต่ละกลุ่ม (ผ่าน/ไม่ผ่าน)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Fail (0)', 'Pass (1)']))


# Step 11: [เปลี่ยน!] แสดงกราฟ Confusion Matrix
# กราฟนี้จะบอกว่าโมเดลทำนาย "ถูก" หรือ "ผิด" ไปในแต่ละกลุ่มอย่างไร
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Fail (0)', 'Pass (1)'],
            yticklabels=['Fail (0)', 'Pass (1)'])
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.title("Confusion Matrix")
plt.show()

# Step 12: [เปลี่ยน!] ดูว่าปัจจัยไหนมีผลต่อการ "ผ่าน/ไม่ผ่าน"
# ค่าสัมประสิทธิ์จะบอกว่าปัจจัยไหนเพิ่มโอกาส "ผ่าน" (ค่าบวก) หรือเพิ่มโอกาส "ไม่ผ่าน" (ค่าลบ)
coeff_df = pd.DataFrame(model.coef_[0], X.columns, columns=['Coefficient']).sort_values(by='Coefficient', ascending=False)
print("\nTop 10 Features Influencing Passing:")
print(coeff_df.head(10))