# predict_emg.py
import serial
import numpy as np
from tensorflow.keras.models import load_model
from collections import deque
import math
from joblib import load
import keyboard

# 설정 값 (학습 코드와 완전히 동일해야 함)
SERIAL_PORT = 'COM9'  # 시리얼 포트
BAUD_RATE = 115200    # 통신 속도
WINDOW_SIZE = 50      # 버퍼 크기 (학습시와 동일)

# 저장된 모델 및 Scaler 로드
model = load_model('best_model.h5')  # 학습된 모델
scaler = load('emg_scaler.bin')      # 학습시 사용한 Scaler

# 실시간 버퍼 초기화 (데이터 수집 코드와 구조 일치)
raw_buffer1 = deque(maxlen=WINDOW_SIZE)
raw_buffer2 = deque(maxlen=WINDOW_SIZE)

def calculate_features(buffer):
    """학습 코드와 동일한 특징 계산 함수"""
    if len(buffer) < 2:
        return [0.0] * 6  # 6개 특징
    
    mean = sum(buffer) / len(buffer)
    std = math.sqrt(sum((x - mean)**2 for x in buffer) / len(buffer))
    diff = buffer[-1] - buffer[-2]
    zcr = sum(1 for i in range(1, len(buffer)) if (buffer[i-1] * buffer[i] < 0))
    rms = math.sqrt(sum(x**2 for x in buffer) / len(buffer))
    waveform = max(buffer) - min(buffer)
    
    return [mean, std, diff, zcr, rms, waveform]

def realtime_prediction():
    print("실시간 EMG 예측 시작 (스페이스바로 종료)")
    
    with serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1) as ser:
        while True:
            # 시리얼 데이터 읽기
            line = ser.readline().decode().strip()
            data = line.split(',')
            
            if len(data) == 6:  # 유효한 데이터 라인만 처리
                try:
                    # 센서 값 추출 (학습 데이터와 동일 포맷)
                    sensor1_raw = int(data[0])  # S1 원시 값
                    sensor2_raw = int(data[3])  # S2 원시 값
                    
                    # 버퍼 업데이트
                    raw_buffer1.append(sensor1_raw)
                    raw_buffer2.append(sensor2_raw)
                    
                    # 윈도우 버퍼가 찼을 때 예측 수행
                    if len(raw_buffer1) == WINDOW_SIZE:
                        # 특징 추출 (학습시와 동일 순서)
                        features1 = calculate_features(raw_buffer1)  # S1 특징
                        features2 = calculate_features(raw_buffer2)  # S2 특징
                        
                        # 특징 결합 (S1 원시 값 + S1 특징 + S2 원시 값 + S2 특징)
                        combined_features = [sensor1_raw] + features1 + [sensor2_raw] + features2
                        
                        # 스케일링 (학습시와 동일 전처리)
                        scaled_data = scaler.transform([combined_features])
                        
                        # 예측 수행
                        prediction = model.predict(scaled_data, verbose=0)
                        pred_label = np.argmax(prediction)
                        confidence = np.max(prediction)
                        
                        # 예측 결과 및 수신된 데이터 출력
                        print(f"\r▶ 예측: {pred_label} (신뢰도: {confidence*100:.2f}%) | 수신 데이터: {data}", end="", flush=True)
                        
                        # 버퍼 초기화
                        raw_buffer1.clear()
                        raw_buffer2.clear()
                        
                except ValueError as ve:
                    print(f"\n데이터 파싱 오류 발생: {ve} - 입력 데이터: {data}")
                except Exception as e:
                    print(f"\n치명적 오류: {str(e)}")
                    break
            
            # 스페이스바로 종료
            if keyboard.is_pressed('space'):
                print("\n예측을 종료합니다.")
                break

if __name__ == "__main__":
    realtime_prediction()


