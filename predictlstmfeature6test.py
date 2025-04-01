import serial
import tensorflow as tf
import numpy as np

# LSTM 모델 불러오기
model = tf.keras.models.load_model('emg_classifier_03-31.h5')

# 시리얼 포트 연결 (COM4 포트, 115200 baud rate)
ser = serial.Serial('COM4', 115200)

# 윈도우 크기
window_size = 30
window_data_ch1 = []
window_data_ch2 = []

def calculate_features(data):
    """
    RMS, Diff, Waveform Length 특징을 계산하는 함수
    """
    data = np.array(data)
    
    # RMS 계산
    rms = np.sqrt(np.mean(np.square(data)))
    
    # Diff 계산 (현재 값 - 이전 값)
    diff = np.mean(np.abs(np.diff(data)))
    
    # Waveform Length 계산
    wl = np.sum(np.abs(np.diff(data)))
    
    return [rms, diff, wl]

while True:
    # 시리얼 데이터 읽기
    line = ser.readline().decode('utf-8').strip()
    if line:
        data_split = line.split(',')

        if len(data_split) >= 2:
            try:
                # 센서 데이터 변환
                ch1 = float(data_split[0])
                ch2 = float(data_split[1])
                print(f"센서1: {ch1}, 센서2: {ch2}")

                # 데이터 윈도우에 추가
                window_data_ch1.append(ch1)
                window_data_ch2.append(ch2)

                # 윈도우 크기만큼 데이터가 쌓였을 때 예측
                if len(window_data_ch1) == window_size:
                    # 특징 추출
                    features_ch1 = calculate_features(window_data_ch1)
                    features_ch2 = calculate_features(window_data_ch2)

                    # 입력 데이터 생성
                    input_data = np.array(features_ch1 + features_ch2).reshape(1, 1, 6)  # (배치, 타임스텝, 특징 개수)
                    
                    # 모델 예측
                    prediction = model.predict(input_data)
                    predicted_class = np.argmax(prediction)
                    print(f"예측된 손동작: {predicted_class}")

                    # 윈도우 데이터 갱신 (이전 데이터 제거)
                    window_data_ch1 = window_data_ch1[1:]
                    window_data_ch2 = window_data_ch2[1:]

            except ValueError:
                print("잘못된 데이터 형식입니다:", data_split)
        else:
            print("데이터가 부족합니다. 두 개 이상의 값이 필요합니다.")
