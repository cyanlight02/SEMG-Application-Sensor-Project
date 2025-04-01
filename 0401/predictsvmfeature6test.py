import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import joblib

# CSV 파일 로드
csv_filename = "emg_features_100101111.csv"#------------------------------------------------------------------------------------CSV 파일
data = pd.read_csv(csv_filename)

# 레이블이 0인 데이터 제거
data = data[data["Label"] != 0]

# 특징 값과 레이블 분리
X = data[["S1_Std", "S1_Diff", "S1_Waveform", "S2_Std", "S2_Diff", "S2_Waveform"]]
y = data["Label"]

# 데이터 분할 (훈련 80%, 테스트 20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 데이터 정규화
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 정규화 스케일러 저장
joblib.dump(scaler, "scaler1101111040122250.pkl")#-----------------------------------------------------------------------------스케일러 파일
print("정규화 스케일러가 저장되었습니다: scaler.pkl")

# SVM 모델 학습
svm_model = SVC(kernel='rbf', C=1.0, gamma='scale')
svm_model.fit(X_train, y_train)

# 예측 수행
y_pred = svm_model.predict(X_test)

# 성능 평가
accuracy = accuracy_score(y_test, y_pred)
print(f"테스트 정확도: {accuracy:.4f}")
print(classification_report(y_test, y_pred))

# 학습된 모델 저장
joblib.dump(svm_model, "svm_emg_model_0401_0010121401255.pkl")#------------------------------------------------------------------모델 파일
print("모델이 저장되었습니다: svm_emg_model.pkl")
