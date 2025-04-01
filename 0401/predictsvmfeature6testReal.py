import numpy as np
import joblib
import serial
import pandas as pd  # DataFrame 변환을 위해 추가
from sklearn.preprocessing import StandardScaler

# 저장된 SVM 모델 및 정규화 스케일러 로드
model_filename = "svm_emg_model_0401_0010121401255.pkl"#-------------------------------------------------------------모델 파일
scaler_filename = "scaler1101111040122250.pkl"#------------------------------------------------------------------스케일러 파일

svm_model = joblib.load(model_filename)
scaler = joblib.load(scaler_filename)

# 시리얼 포트 설정 (아두이노 연결)
ser = serial.Serial('COM8', 115200)

# 윈도우 크기 설정
window_size = 30
window_data = []

# 실시간 데이터 수집 및 예측
while True:
    try:
        # 시리얼 데이터 읽기
        line = ser.readline().decode('utf-8').strip()
        if line:
            data_split = line.split(',')
            if len(data_split) >= 2:
                try:
                    # 센서1, 센서2 값 변환
                    s1 = float(data_split[0])
                    s2 = float(data_split[1])
                    
                    # 📌 **필터 적용: -10 ~ 10 사이 값은 0으로 변환**
                    #if -10 <= s1 <= 10:
                    #    s1 = 0
                    #if -10 <= s2 <= 10:
                    #    s2 = 0

                    # 윈도우 데이터 추가
                    window_data.append([s1, s2])
                    
                    # 윈도우 크기만큼 데이터가 쌓였을 때 특징 추출 및 예측 수행
                    if len(window_data) == window_size:
                        window_array = np.array(window_data)
                        
                        # RMS 계산
                        s1_rms = np.sqrt(np.mean(np.square(window_array[:, 0])))
                        s2_rms = np.sqrt(np.mean(np.square(window_array[:, 1])))
                        
                        # Diff 계산 (차분 값의 평균)
                        s1_diff = np.mean(np.abs(np.diff(window_array[:, 0])))
                        s2_diff = np.mean(np.abs(np.diff(window_array[:, 1])))
                        
                        # Waveform 계산 (절댓값 평균)
                        s1_waveform = np.mean(np.abs(window_array[:, 0]))
                        s2_waveform = np.mean(np.abs(window_array[:, 1]))
                        
                        # 새로운 데이터 배열 → DataFrame 변환
                        feature_names = ["S1_Std", "S1_Diff", "S1_Waveform", "S2_Std", "S2_Diff", "S2_Waveform"]
                        new_data = pd.DataFrame([[s1_rms, s1_diff, s1_waveform, s2_rms, s2_diff, s2_waveform]],
                                                columns=feature_names)
                        
                        # 데이터 정규화 (스케일러 적용)
                        new_data_scaled = scaler.transform(new_data)
                        
                        # 예측 수행
                        predicted_label = svm_model.predict(new_data_scaled)
                        print(f"예측된 레이블: {predicted_label[0]} 센서(1,2): {[s1, s2]}")
                        
                        # 윈도우 데이터 갱신 (이전 데이터 유지)
                        window_data.pop(0)
                    
                except ValueError:
                    print("잘못된 데이터 형식입니다:", data_split)
            else:
                print("데이터가 부족합니다.")
    except KeyboardInterrupt:
        print("실시간 예측 종료.")
        ser.close()
        break  
