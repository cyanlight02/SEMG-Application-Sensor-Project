import pandas as pd
import numpy as np
import joblib
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

# 특징 추출 함수
def extract_features(emg_signal):
    features = []
    features.append(np.max(emg_signal))                        # Max
    features.append(np.min(emg_signal))                        # Min
    features.append(np.std(emg_signal))                        # SD
    features.append(np.sqrt(np.mean(emg_signal**2)))           # RMS
    features.append(np.mean(np.abs(emg_signal)))               # MAV
    features.append(np.mean(np.abs(np.diff(emg_signal))))      # AAC
    features.append(np.argmax(emg_signal**2))                  # AFB
    features.append(np.sum(np.abs(np.diff(np.sign(emg_signal))))) # ZC
    features.append(np.sum(np.abs(np.diff(emg_signal)) > 0.01))   # SSC
    features.append(np.sum(np.abs(np.diff(emg_signal))))       # WL
    features.append(np.var(emg_signal))                        # VAR
    features.append(pd.Series(emg_signal).skew())              # Skewness
    features.append(pd.Series(emg_signal).kurtosis())          # Kurtosis
    features.append(np.median(np.abs(emg_signal - np.median(emg_signal))))  # MAD
    features.append(np.sum(emg_signal**2))                     # Energy
    features.append(np.ptp(emg_signal))                        # PTP
    return features

# 세그먼트 생성 함수
def create_segments(df, window_size=20, overlap=10): 
    segments = []
    labels = []
    for start in range(0, len(df) - window_size + 1, overlap):
        segment = df.iloc[start:start + window_size]
        if len(segment) == window_size and start != 0 and start + window_size != len(df):
            ch1_features = extract_features(segment['CH1'].values)
            ch2_features = extract_features(segment['CH2'].values)
            segments.append(ch1_features + ch2_features)
            labels.append(segment['Label'].mode()[0])
    return np.array(segments), np.array(labels)

# CSV 파일 읽기
try:
    df = pd.read_csv('EMG_test.csv')
    df['CH1'] = pd.to_numeric(df['CH1'], errors='coerce')
    df['CH2'] = pd.to_numeric(df['CH2'], errors='coerce')
    df.dropna(inplace=True)
except Exception as e:
    print(f"CSV 파일을 읽는 중 오류 발생: {e}")
    exit()

# 세그먼트 생성
X, y = create_segments(df)

# 특징 CSV 파일 저장
df_features = pd.DataFrame(X)
df_features['Label'] = y
df_features.to_csv('EMG_features.csv', index=False)

# 데이터 스케일링
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 데이터 분할 및 학습
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
model = SVC(kernel='rbf', C=10, gamma='scale')  # 최적화된 파라미터 추가
model.fit(X_train, y_train)

joblib.dump(model, 'emg_svm_model_aaa2.pkl')

# 결과 출력
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
