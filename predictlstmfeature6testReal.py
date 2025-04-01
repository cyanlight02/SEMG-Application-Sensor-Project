import serial
import tensorflow as tf
import numpy as np

# LSTM 모델 불러오기
model = tf.keras.models.load_model("lstm_emg_model.h5")

# 시리얼 포트 연결 (아두이노와 연결된 포트 설정 필요)
ser = serial.Serial('COM4', 115200)

# 윈도우 크기
window_size = 30
window_data = []

# 특징 추출 함수 정의
def calculate_features(data):
    if len(data) < 2:
        return [0, 0, 0]  # 데이터가 부족할 경우 기본값 반환
    
    rms = np.sqrt(np.mean(np.square(data)))  # RMS 계산
    diff = np.mean(np.diff(data))  # 차분(Diff) 계산
    wl = np.sum(np.abs(np.diff(data)))  # Waveform Length 계산
    
    return [rms, diff, wl]

while True:
    try:
        # 시리얼 데이터 읽기
        line = ser.readline().decode('utf-8').strip()
        if line:
            # 데이터를 쉼표로 분리
            data_split = line.split(',')

            # 데이터가 2개 이상인지 확인
            if len(data_split) >= 2:
                try:
                    # 센서1, 센서2 데이터를 float로 변환
                    ch1 = float(data_split[0])
                    ch2 = float(data_split[1])
                    print(f"센서1: {ch1}, 센서2: {ch2}")

                    # 센서 데이터를 리스트에 저장
                    window_data.append([ch1, ch2])

                    # 윈도우 크기만큼 데이터가 쌓이면 특징 계산 및 예측 수행
                    if len(window_data) == window_size:
                        # 센서1, 센서2에 대한 특징 추출
                        ch1_values = [d[0] for d in window_data]
                        ch2_values = [d[1] for d in window_data]
                        
                        features = calculate_features(ch1_values) + calculate_features(ch2_values)
                        input_data = np.array(features).reshape(1, 1, 6)  # (배치, 시간, 특성 수)

                        # 모델을 사용하여 손동작 예측
                        prediction = model.predict(input_data)
                        predicted_class = np.argmax(prediction)
                        print(f"예측된 손동작: {predicted_class}")

                        # 윈도우 데이터 최신화 (FIFO 방식 유지)
                        window_data.pop(0)

                except ValueError:
                    print("잘못된 데이터 형식입니다:", data_split)
            else:
                print("데이터가 부족합니다. 두 개 이상의 값이 필요합니다.")
    
    except KeyboardInterrupt:
        print("프로그램 종료")
        ser.close()
        break
