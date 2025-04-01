import serial
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from datetime import datetime  # datetime 모듈 임포트

# 모델 로드
model = load_model('Ntrained_model_4_20250326_154239.h5')  # 훈련된 모델 파일 경로

# 시리얼 포트 설정 (포트 이름은 시스템에 맞게 수정해야 합니다)
ser = serial.Serial('COM9', 115200)  # Windows
# ser = serial.Serial('/dev/ttyUSB0', 115200)  # Linux/Mac

# 데이터 정규화용 스케일러
scaler = StandardScaler()  # 필요시 로드한 스케일러를 사용해야 함

# EMG 데이터 수신 및 예측
data_buffer = []  # 최근 데이터 저장을 위한 버퍼
window_size = 50  # 시계열 길이 설정

while True:
    try:
        line = ser.readline().decode('utf-8').strip()
        emg_values = line.split(", ")

        # 데이터의 길이가 2인지 확인 (EMGData와 EMGData2)
        if len(emg_values) == 2:
            emg_data = [float(emg_values[0]), float(emg_values[1])]
            data_buffer.append(emg_data)

            # 버퍼가 window_size 개 데이터를 넘으면 예측 수행
            if len(data_buffer) == window_size:
                # NumPy 배열로 변환하고 정규화
                data_array = np.array(data_buffer)
                data_array = scaler.fit_transform(data_array)  # 스케일러를 사용하여 정규화

                # 예측 수행
                predicted_label = np.argmax(model.predict(data_array.reshape(1, window_size, 2)))

                # 현재 시각 가져오기
                current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

                # 라벨값과 시각 출력
                print(f"[{current_time}] Predicted Label: {predicted_label}")

                # 버퍼 초기화
                data_buffer.pop(0)

    except KeyboardInterrupt:
        print("프로그램을 종료합니다.")
        break
    except Exception as e:
        print(f"오류 발생: {e}")

ser.close()  # 시리얼 포트 닫기
