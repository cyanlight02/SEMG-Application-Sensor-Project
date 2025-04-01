import serial
import csv
import os
import keyboard

# 시리얼 포트 설정 (포트 이름은 시스템에 맞게 수정해야 합니다)
ser = serial.Serial('COM9', 115200)  # Windows
# ser = serial.Serial('/dev/ttyUSB0', 115200)  # Linux/Mac

# 저장할 파일 이름 생성 함수
def get_filename(base_name):
    i = 0
    while True:
        if i == 0:
            filename = f"{base_name}.csv"
        else:
            filename = f"{base_name}_{i}.csv"
        
        if not os.path.exists(filename):
            return filename
        i += 1

def collect_data():
    filename = get_filename('data')  # 파일 이름 결정
    print(f"데이터 수집을 시작합니다. 파일: {filename}")

    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['EMGData', 'EMGData2', 'Label'])  # 헤더 추가

        while True:
            try:
                line = ser.readline().decode('utf-8').strip()
                data = line.split(", ")

                # 데이터의 길이가 2인지 확인 (EMGData, EMGData2)
                if len(data) == 2:
                    # 키패드 입력 처리
                    label = 0  # 기본값
                    for i in range(1, 10):
                        if keyboard.is_pressed(str(i)):
                            label = i
                            break
                    
                    # 데이터 추가
                    writer.writerow([data[0], data[1], label])  # EMGData, EMGData2, Label 추가
                    print([data[0], data[1], label])  # 콘솔에 출력
                else:
                    print(f"잘못된 데이터: {data}")  # 잘못된 데이터 출력

                # ESC 키 입력 감지
                if keyboard.is_pressed('esc'):
                    print("데이터 수집을 종료합니다. Enter 키를 눌러 새로운 측정을 시작하거나 ESC 키를 한 번 더 눌러 종료합니다.")
                    break

            except KeyboardInterrupt:
                print("강제로 종료되었습니다.")
                break

while True:
    print("Enter 키를 누르면 데이터 수집을 시작합니다.")
    keyboard.wait('enter')  # Enter 키를 기다림
    collect_data()  # 데이터 수집 시작

    # 종료 여부 확인
    while True:
        if keyboard.is_pressed('esc'):
            print("프로그램을 종료합니다.")
            ser.close()  # 시리얼 포트 닫기
            exit()  # 프로그램 종료
        elif keyboard.is_pressed('enter'):
            break  # Enter 키를 누르면 새로운 측정을 시작
