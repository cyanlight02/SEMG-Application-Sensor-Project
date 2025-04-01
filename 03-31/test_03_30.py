import serial
import csv
import os
from collections import deque
import math
import keyboard

# 시리얼 포트 설정
SERIAL_PORT = 'COM9'
BAUD_RATE = 115200
WINDOW_SIZE = 50
BUFFER_MAX = 1000

# 버퍼 초기화
raw_buffer1 = deque(maxlen=WINDOW_SIZE)
raw_buffer2 = deque(maxlen=WINDOW_SIZE)
history_buffer1 = deque(maxlen=BUFFER_MAX)
history_buffer2 = deque(maxlen=BUFFER_MAX)

def get_filename(base_name):
    """기본 이름에 따라 파일 이름 생성"""
    return f"{base_name}.csv"

def calculate_features(buffer):
    """버퍼에서 특징 추출"""
    if len(buffer) < 2:
        return [0.0]*6  # 6개 특징
    
    mean = sum(buffer) / len(buffer)
    std = math.sqrt(sum((x - mean)**2 for x in buffer) / len(buffer))
    diff = buffer[-1] - buffer[-2]
    zcr = sum(1 for i in range(1, len(buffer)) if buffer[i-1] * buffer[i] < 0)
    rms = math.sqrt(sum(x**2 for x in buffer) / len(buffer))
    waveform = max(buffer) - min(buffer)
    
    return [mean, std, diff, zcr, rms, waveform]  # 6개 반환

def collect_data(label, filename):
    print(f"저장 파일: {filename}")
    
    with serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1) as ser, \
         open(filename, 'a', newline='') as csvfile:  # 'a' 모드로 열기
        
        writer = csv.writer(csvfile)
        
        # 헤더가 이미 있는지 확인하고, 없으면 작성
        if csvfile.tell() == 0:
            headers = [
                'S1_Raw', 'S1_Mean', 'S1_Std', 'S1_Diff', 
                'S1_ZCR', 'S1_RMS', 'S1_Waveform',
                'S2_Raw', 'S2_Mean', 'S2_Std', 'S2_Diff',
                'S2_ZCR', 'S2_RMS', 'S2_Waveform',
                'Label'
            ]
            writer.writerow(headers)
        
        # 캘리브레이션
        print("캘리브레이션 중...")
        while len(history_buffer1) < BUFFER_MAX:
            line = ser.readline().decode().strip()
            if line.count(',') == 5:
                try:
                    v1 = int(line.split(',')[0])
                    history_buffer1.append(v1)
                    v2 = int(line.split(',')[3])
                    history_buffer2.append(v2)
                except ValueError:
                    continue
        
        print("데이터 수집 시작 (스페이스바로 종료)")
        while True:
            line = ser.readline().decode().strip()
            data = line.split(',')
            
            if len(data) == 6:
                try:
                    # 센서 값 파싱
                    v1 = int(data[0])
                    v2 = int(data[3])
                    
                    # 버퍼 업데이트 (원본 값 사용)
                    raw_buffer1.append(v1)
                    raw_buffer2.append(v2)
                    
                    # 특징 추출 (Raw + 6개 특징)
                    if len(raw_buffer1) == WINDOW_SIZE:
                        features1 = [v1] + calculate_features(raw_buffer1)
                        features2 = [v2] + calculate_features(raw_buffer2)
                        
                        # 라벨 입력
                        full_row = features1 + features2 + [label]
                        writer.writerow(full_row)
                        
                except ValueError:
                    print(f"파싱 오류: {data}")
                except Exception as e:
                    print(f"오류: {str(e)}")
            
            try:
                if keyboard.is_pressed('space'):  # 스페이스바로 수집 중지
                    print("라벨 입력 화면으로 돌아갑니다...")
                    break
            except Exception as e:
                print(f"키보드 감지 오류: {str(e)}")

if __name__ == "__main__":
    filename = None
    while True:
        print("입력하실 라벨값을 눌러주세요 (ESC 종료)")
        label = None
        while label is None:
            for i in range(10):  # 0부터 9까지의 숫자 키패드 입력 대기
                if keyboard.is_pressed(str(i)):
                    label = i
                    print(f"라벨이 {label}로 설정되었습니다.")
                    filename = get_filename('emg_data')  # 파일 이름을 새로 생성
                    break
            try:
                if keyboard.is_pressed('esc'):  # ESC로 프로그램 종료
                    print("프로그램 종료")
                    exit()
            except Exception as e:
                print(f"키보드 감지 오류: {str(e)}")
        
        collect_data(label, filename)  # 파일 이름을 인자로 전달하여 데이터 수집
