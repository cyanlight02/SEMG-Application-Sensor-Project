import serial
import numpy as np
from collections import deque
import pandas as pd
from tensorflow.keras.models import load_model
import datetime
import math

# 전처리 객체 및 모델 로드
model = load_model('emg_classifier_04-01.h5')
scaler = pd.read_pickle('scaler_04-01.pkl')
pca = pd.read_pickle('pca_04-01.pkl')

# 버퍼 설정
WINDOW_SIZE = 50
raw_buffer1 = deque(maxlen=WINDOW_SIZE)
raw_buffer2 = deque(maxlen=WINDOW_SIZE)
feature_window = deque(maxlen=WINDOW_SIZE)

# 특징 계산 함수 (데이터 수집 코드와 동일)
def calculate_features(buffer):
    """원본 데이터 수집 코드와 동일한 특징 계산 로직"""
    if len(buffer) < 2:
        return [0.0]*4  # 4개 특징만 반환
    
    # 시간 영역 특징
    std = np.std(buffer)
    diff = buffer[-1] - buffer[-2]
    waveform = max(buffer) - min(buffer)
    
    return [std, diff, waveform]  # 4개 반환

# 실시간 처리 클래스
class RealTimePredictor:
    def __init__(self):
        self.ser = serial.Serial('COM9', 115200, timeout=1)
        
    def process_sample(self):
        line = self.ser.readline().decode().strip()
        if line.count(',') == 5:
            try:
                data = line.split(',')
                v1 = int(data[0])
                v2 = int(data[3])
                
                # 원시 데이터 버퍼 업데이트
                raw_buffer1.append(v1)
                raw_buffer2.append(v2)
                
                # 특징 계산 가능 여부 확인
                if len(raw_buffer1) == WINDOW_SIZE and len(raw_buffer2) == WINDOW_SIZE:
                    # 센서별 특징 계산
                    s1_features = calculate_features(raw_buffer1)
                    s2_features = calculate_features(raw_buffer2)
                    combined = s1_features + s2_features
                    
                    # 전처리 파이프라인 적용
                    scaled = scaler.transform([combined])
                    pca_transformed = pca.transform(scaled)[0]
                    
                    # 특징 윈도우 업데이트
                    feature_window.append(pca_transformed)
                    
                    # 예측 수행
                    if len(feature_window) == WINDOW_SIZE:
                        input_data = np.array(feature_window).reshape(1, WINDOW_SIZE, -1)
                        prediction = model.predict(input_data, verbose=0)
                        return np.argmax(prediction[0])
            except ValueError as e:
                print(f"값 변환 오류: {str(e)}")
            except Exception as e:
                print(f"처리 오류: {str(e)}")
        return None

# 메인 실행 루프
if __name__ == "__main__":
    predictor = RealTimePredictor()
    print("실시간 예측 시작 (Ctrl+C로 종료)")
    
    try:
        while True:
            predicted_label = predictor.process_sample()
            if predicted_label is not None:
                current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                print(f"[{current_time}] 예측 라벨: {predicted_label}")
    except KeyboardInterrupt:
        predictor.ser.close()
        print("시스템 종료")
