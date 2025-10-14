# Step 2: โหลด dataset
from ucimlrepo import fetch_ucirepo
import pandas as pd

# โหลด dataset
student = fetch_ucirepo(id=320)

# แยก Features (X) และ Target (y)
X = student.data.features
y = student.data.targets

print("✅ Loaded successfully!")
print("Shape of X:", X.shape)
print("Shape of y:", y.shape)



# Step 3: รวม features และ target เข้าด้วยกันเพื่อดูภาพรวม
df = pd.concat([X, y], axis=1)
print(df.head())


# Step 4: ตรวจสอบข้อมูลเบื้องต้น
print(df.info())
print(df.describe())
print(df.isnull().sum())  # ตรวจสอบค่าว่าง


# Step 5: ลบข้อมูลที่ไม่ต้องการ (optional)
# G1 และ G2 มีความสัมพันธ์สูงกับ G3 ดังนั้นหากต้องการ model ที่ทำนายยากและมีประโยชน์มากกว่า ควรลบ G1, G2 ออก
df = df.drop(['G1', 'G2'], axis=1)


# Step 6: แยก features (X) และ target (y) ใหม่อีกครั้ง
X = df.drop(columns=['G3'])
y = df['G3']


# Step 7: แปลงข้อมูลประเภท Categorical เป็นตัวเลข (One-Hot Encoding)
X = pd.get_dummies(X, drop_first=True)
print("หลังจากแปลงข้อมูลแล้ว:", X.shape)



# Step 8: แบ่งข้อมูล Train/Test
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training set:", X_train.shape)
print("Testing set:", X_test.shape)


# Step 9: สร้างและ Train โมเดล (Linear Regression)
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)

print("✅ Model trained successfully!")



# Step 10: ประเมินผลโมเดล
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"Root Mean Squared Error: {rmse:.2f}")
print(f"R² Score: {r2:.2f}")


# Step 11: แสดงกราฟความสัมพันธ์ระหว่างค่า G3 จริง vs ค่าทำนาย
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(6,6))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel("Actual G3")
plt.ylabel("Predicted G3")
plt.title("Actual vs Predicted Final Grades")
plt.show()

# Step 12: ดูว่าสมการถ่วงน้ำหนักของแต่ละตัวแปร (feature importance)
coeff_df = pd.DataFrame(model.coef_, X.columns, columns=['Coefficient']).sort_values(by='Coefficient', ascending=False)
print(coeff_df.head(10))
