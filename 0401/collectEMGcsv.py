import serial
import csv
import numpy as np
import keyboard

# 시리얼 포트 설정 (COM4, 115200 baud rate)
ser = serial.Serial('COM8', 115200)

# 윈도우 크기 설정
window_size = 30
window_data_ch1 = []
window_data_ch2 = []
current_label = 0  # 기본 레이블 값 (0)
recording = False  # 데이터 저장 여부

# CSV 파일 초기화
csv_filename = "emg_features_100101111.csv"#------------------------------------------------------------------------파일 이름
with open(csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["S1_Std", "S1_Diff", "S1_Waveform", "S2_Std", "S2_Diff", "S2_Waveform", "Label"])

def calculate_features(data):
    """
    RMS, Diff, Waveform Length 특징을 계산하는 함수
    """
    data = np.array(data)
    
    # RMS 계산
    rms = np.sqrt(np.mean(np.square(data)))
    
    # Diff 계산 (현재 값 - 이전 값의 평균)
    diff = np.mean(np.abs(np.diff(data)))
    
    # Waveform Length 계산
    wl = np.sum(np.abs(np.diff(data)))
    
    return [rms, diff, wl]

while True:
    # 현재 입력된 키 확인
    for i in range(1, 10):  # 숫자 1~9에 대해 체크
        if keyboard.is_pressed(str(i)):
            current_label = i
            break
    else:
        current_label = 0  # 아무 키도 눌리지 않으면 0

    # 데이터 저장 시작 ('z' 키)
    if keyboard.is_pressed('z'):
        recording = True
        print("데이터 저장 시작")
        while keyboard.is_pressed('z'):  # 키를 뗄 때까지 기다림
            pass
    
    # 데이터 저장 중지 ('x' 키)
    if keyboard.is_pressed('x'):
        recording = False
        print("데이터 저장 중지")
        while keyboard.is_pressed('x'):  # 키를 뗄 때까지 기다림
            pass
    
    # 프로그램 종료 ('q' 키)
    if keyboard.is_pressed('q'):
        print("프로그램 종료")
        break

    # 시리얼 데이터 읽기
    line = ser.readline().decode('utf-8').strip()
    if line:
        data_split = line.split(',')

        if len(data_split) >= 2:
            try:
                # 센서 데이터 변환
                ch1 = float(data_split[0])
                ch2 = float(data_split[1])

                # 📌 **필터 적용: -10 ~ 10 사이 값은 0으로 변환**
                #if -10 <= ch1 <= 10:
                #    ch1 = 0
                #if -10 <= ch2 <= 10:
                #    ch2 = 0

                print(f"센서1: {ch1}, 센서2: {ch2}, 레이블: {current_label}")

                # 데이터 저장이 활성화된 경우만 처리
                if recording:
                    # 데이터 윈도우에 추가
                    window_data_ch1.append(ch1)
                    window_data_ch2.append(ch2)

                    # 윈도우 크기만큼 데이터가 쌓였을 때 특징 추출 및 저장
                    if len(window_data_ch1) == window_size:
                        # 특징 추출
                        features_ch1 = calculate_features(window_data_ch1)
                        features_ch2 = calculate_features(window_data_ch2)
                        
                        # CSV 파일에 저장
                        with open(csv_filename, mode='a', newline='') as file:
                            writer = csv.writer(file)
                            writer.writerow(features_ch1 + features_ch2 + [current_label])
                        
                        print(f"저장됨: {features_ch1 + features_ch2 + [current_label]}")

                        # 윈도우 데이터 갱신 (이전 데이터 제거)
                        window_data_ch1 = window_data_ch1[1:]
                        window_data_ch2 = window_data_ch2[1:]

            except ValueError:
                print("잘못된 데이터 형식입니다:", data_split)
        else:
            print("데이터가 부족합니다. 두 개 이상의 값이 필요합니다.")
