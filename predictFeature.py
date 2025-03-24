import serial
import joblib
import numpy as np
import pandas as pd
import keyboard

# SVM 모델 로드
model = joblib.load('emg_svm_model_aaa2.pkl')

# 시리얼 포트 설정
ser = serial.Serial('COM3', 115200)

# 예측 모드 상태
prediction_mode = False

def extract_features(emg_signal):
    features = []
    # CH1 특징 추출
    features.append(np.max(emg_signal[:, 0]))                        # Max
    features.append(np.min(emg_signal[:, 0]))                        # Min
    features.append(np.std(emg_signal[:, 0]))                        # SD
    features.append(np.sqrt(np.mean(emg_signal[:, 0]**2)))           # RMS
    features.append(np.mean(np.abs(emg_signal[:, 0])))               # MAV
    features.append(np.mean(np.abs(np.diff(emg_signal[:, 0]))))      # AAC
    features.append(np.argmax(emg_signal[:, 0]**2))                  # AFB
    features.append(np.sum(np.abs(np.diff(np.sign(emg_signal[:, 0]))))) # ZC
    features.append(np.sum(np.abs(np.diff(emg_signal[:, 0])) > 0.01))   # SSC
    features.append(np.sum(np.abs(np.diff(emg_signal[:, 0]))))       # WL
    features.append(np.var(emg_signal[:, 0]))                        # VAR
    features.append(pd.Series(emg_signal[:, 0]).skew())              # Skewness
    features.append(pd.Series(emg_signal[:, 0]).kurtosis())          # Kurtosis
    features.append(np.median(np.abs(emg_signal[:, 0] - np.median(emg_signal[:, 0]))))  # MAD
    features.append(np.sum(emg_signal[:, 0]**2))                     # Energy
    features.append(np.ptp(emg_signal[:, 0]))                        # PTP

    # CH2 특징 추출
    features.append(np.max(emg_signal[:, 1]))                        # Max
    features.append(np.min(emg_signal[:, 1]))                        # Min
    features.append(np.std(emg_signal[:, 1]))                        # SD
    features.append(np.sqrt(np.mean(emg_signal[:, 1]**2)))           # RMS
    features.append(np.mean(np.abs(emg_signal[:, 1])))               # MAV
    features.append(np.mean(np.abs(np.diff(emg_signal[:, 1]))))      # AAC
    features.append(np.argmax(emg_signal[:, 1]**2))                  # AFB
    features.append(np.sum(np.abs(np.diff(np.sign(emg_signal[:, 1]))))) # ZC
    features.append(np.sum(np.abs(np.diff(emg_signal[:, 1])) > 0.01))   # SSC
    features.append(np.sum(np.abs(np.diff(emg_signal[:, 1]))))       # WL
    features.append(np.var(emg_signal[:, 1]))                        # VAR
    features.append(pd.Series(emg_signal[:, 1]).skew())              # Skewness
    features.append(pd.Series(emg_signal[:, 1]).kurtosis())          # Kurtosis
    features.append(np.median(np.abs(emg_signal[:, 1] - np.median(emg_signal[:, 1]))))  # MAD
    features.append(np.sum(emg_signal[:, 1]**2))                     # Energy
    features.append(np.ptp(emg_signal[:, 1]))                        # PTP

    return np.array(features).reshape(1, -1)

# 데이터 수집 및 예측 루프
try:
    emg_data = []

    while True:
        if ser.in_waiting:
            line = ser.readline().decode('utf-8').strip()
            try:
                ch1, ch2 = map(int, line.split(','))
                emg_data.append([ch1, ch2])

                if len(emg_data) >= 20:  # 20개 샘플마다 특징 추출 (10개씩 겹치기)
                    segment = np.array(emg_data[-20:])
                    features = extract_features(segment)
                    #print(f"추출된 특징: {features}")  # 추가된 디버깅 코드

                    if prediction_mode:
                        prediction = model.predict(features)
                        print(f"예측 레이블: {prediction[0]}")
                    else:
                        print("[예측 모드 OFF] 데이터 수신 중...")
            except ValueError:
                print("데이터 오류: 수신 데이터 형식이 잘못되었습니다.")

        if keyboard.is_pressed('o'):  # 예측 모드 ON
            prediction_mode = True
            print("[예측 모드 ON]")

        if keyboard.is_pressed('f'):  # 예측 모드 OFF
            prediction_mode = False
            print("[예측 모드 OFF]")

        if keyboard.is_pressed('q'):  # 종료
            print("프로그램 종료")
            break

except KeyboardInterrupt:
    print("프로그램 강제 종료")

finally:
    ser.close()
