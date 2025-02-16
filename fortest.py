import numpy as np
import pandas as pd
import time
from scipy.stats import skew, kurtosis
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from collections import deque

# 1. EMG 신호에서 특징 추출 함수
def extract_features_from_signal(emg_signal):
    """
    주어진 EMG 신호에서 특징을 추출하는 함수
    emg_signal: EMG 신호 (리스트 혹은 배열)
    """
    features = []
    
    # 1. Root Mean Square (RMS)
    rms = np.sqrt(np.mean(np.square(emg_signal)))
    features.append(rms)

    # 2. Mean Absolute Value (MAV)
    mav = np.mean(np.abs(emg_signal))
    features.append(mav)

    # 3. Zero Crossings (ZC)
    zc = np.count_nonzero(np.diff(np.sign(emg_signal)))
    features.append(zc)

    # 4. Slope Sign Change (SSC)
    ssc = np.count_nonzero(np.diff(np.sign(np.diff(emg_signal)) ))
    features.append(ssc)

    # 5. Waveform Length (WL)
    wl = np.sum(np.abs(np.diff(emg_signal)))
    features.append(wl)

    # 6. Variance (VAR)
    var = np.var(emg_signal)
    features.append(var)

    # 7. Skewness
    skewness = skew(emg_signal)
    features.append(skewness)

    # 8. Kurtosis
    kurt = kurtosis(emg_signal)
    features.append(kurt)

    # 9. Median Absolute Deviation (MAD)
    mad = np.median(np.abs(emg_signal - np.median(emg_signal)))
    features.append(mad)

    # 10. Energy
    energy = np.sum(np.square(emg_signal))
    features.append(energy)

    # 11. Peak-to-Peak (PTP)
    ptp = np.ptp(emg_signal)
    features.append(ptp)

    return features

# 2. 실시간 EMG 데이터 수집 및 분류
class RealTimeEMGGestureClassifier:
    def __init__(self, model=None):
        """
        실시간 EMG 데이터 수집 및 분류
        model: 사전 훈련된 SVM 모델
        """
        self.model = model if model is not None else SVC(kernel='rbf')  # RBF 커널을 사용하는 SVM 모델
        self.scaler = StandardScaler()  # 데이터 정규화용 StandardScaler
        self.data_queue = deque(maxlen=200)  # 실시간 데이터를 위한 큐 (최대 200개 데이터)

    def train(self, X_train, y_train):
        """
        훈련 데이터를 사용하여 SVM 모델 학습
        X_train: 특징 데이터
        y_train: 레이블 데이터
        """
        X_train_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_train_scaled, y_train)

    def predict(self, features):
        """
        실시간 특징을 사용하여 손동작 예측
        features: 실시간으로 추출된 EMG 신호의 특징
        """
        features_scaled = self.scaler.transform([features])  # 데이터 정규화
        return self.model.predict(features_scaled)[0]  # 예측값 반환

    def generate_emg_data(self):
        """
        임의로 EMG 데이터를 생성하는 함수
        """
        # 임의로 생성한 EMG 데이터 (예시: 범위 -100에서 100 사이의 값)
        return np.random.randint(-100, 100, size=200)

    def process_emg_signal(self):
        """
        실시간 EMG 신호를 읽고 특징을 추출하여 손동작을 분류
        """
        while True:
            # 임의의 EMG 데이터를 생성
            emg_signal = self.generate_emg_data()  # 임의 데이터 생성
            print(f"Generated EMG Signal: {emg_signal}")  # 생성된 데이터 출력

            features = extract_features_from_signal(emg_signal)  # 특징 추출
            print(f"Extracted Features: {features}")  # 추출된 특징 출력

            # 손동작 예측
            predicted_label = self.predict(features)
            print(f"Predicted Label: {predicted_label}")  # 예측 결과 출력

            time.sleep(1)  # 1초마다 새 데이터를 처리

# 3. 모델 학습 및 실시간 데이터 수집
if __name__ == "__main__":
    # 훈련 데이터 (예시: CSV 파일에서 학습 데이터를 읽어와서 사용)
    csv_file = 'emg_features_test_new.csv'  # 훈련 데이터를 저장한 CSV 파일 경로
    data = pd.read_csv(csv_file)
    X_train = data.iloc[:, :-1].values
    y_train = data.iloc[:, -1].values

    # 실시간 분류기 인스턴스 생성
    classifier = RealTimeEMGGestureClassifier()  # 블루투스 포트가 필요 없으므로 삭제
    classifier.train(X_train, y_train)  # 모델 학습

    # 실시간 데이터 수집 및 예측
    classifier.process_emg_signal()
