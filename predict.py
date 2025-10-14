# predict.py

import pandas as pd
import joblib

def predict_student_performance(student_data):
    """
    ฟังก์ชันสำหรับโหลดโมเดลและทำนายผลจากข้อมูลนักเรียนใหม่
    """
    # โหลดโมเดลและรายชื่อคอลัมน์ที่บันทึกไว้
    try:
        model = joblib.load('student_classifier.pkl')
        model_columns = joblib.load('model_columns.pkl')
    except FileNotFoundError:
        print("ไม่พบไฟล์โมเดล! กรุณารัน train.py ก่อน")
        return

    # สร้าง DataFrame จากข้อมูลนักเรียนใหม่
    df_new = pd.DataFrame(student_data, index=[0])

    # แปลงข้อมูลใหม่ให้มีคอลัมน์เหมือนตอน train
    df_new_encoded = pd.get_dummies(df_new, drop_first=True)
    df_final = df_new_encoded.reindex(columns=model_columns, fill_value=0)

    # ทำนายผล
    prediction = model.predict(df_final)
    prediction_proba = model.predict_proba(df_final)

    # แสดงผล
    result = "ผ่าน (Pass)" if prediction[0] == 1 else "ไม่ผ่าน (Fail)"
    confidence = prediction_proba[0][prediction[0]] * 100

    print("\n--- ผลการทำนาย ---")
    print(f"ผลลัพธ์: นักเรียนมีแนวโน้มที่จะ '{result}'")
    print(f"ความเชื่อมั่น: {confidence:.2f}%")
    print("-------------------")


if __name__ == '__main__':
    # === ลองป้อนข้อมูลนักเรียนสมมติ ===

    # ตัวอย่างนักเรียนที่ 1: ดูมีแนวโน้มจะสอบผ่าน
    student_1 = {
        'school': 'GP', 'sex': 'F', 'age': 16, 'address': 'U',
        'famsize': 'GT3', 'Pstatus': 'T', 'Medu': 4, 'Fedu': 4,
        'Mjob': 'teacher', 'Fjob': 'health', 'reason': 'reputation',
        'guardian': 'mother', 'traveltime': 1, 'studytime': 3,
        'failures': 0, 'schoolsup': 'no', 'famsup': 'yes',
        'paid': 'yes', 'activities': 'yes', 'nursery': 'yes',
        'higher': 'yes', 'internet': 'yes', 'romantic': 'no',
        'famrel': 4, 'freetime': 3, 'goout': 2, 'Dalc': 1,
        'Walc': 1, 'health': 3, 'absences': 2
    }
    print("--- [Case 1] ทำนายผลนักเรียนตัวอย่างคนที่ 1 ---")
    predict_student_performance(student_1)

    # ตัวอย่างนักเรียนที่ 2: ดูมีปัจจัยเสี่ยง
    student_2 = {
        'school': 'MS', 'sex': 'M', 'age': 18, 'address': 'R',
        'famsize': 'GT3', 'Pstatus': 'T', 'Medu': 1, 'Fedu': 1,
        'Mjob': 'other', 'Fjob': 'other', 'reason': 'course',
        'guardian': 'father', 'traveltime': 3, 'studytime': 1,
        'failures': 3, 'schoolsup': 'yes', 'famsup': 'no',
        'paid': 'no', 'activities': 'no', 'nursery': 'no',
        'higher': 'no', 'internet': 'no', 'romantic': 'yes',
        'famrel': 2, 'freetime': 5, 'goout': 5, 'Dalc': 4,
        'Walc': 5, 'health': 5, 'absences': 15
    }
    print("\n--- [Case 2] ทำนายผลนักเรียนตัวอย่างคนที่ 2 ---")
    predict_student_performance(student_2)