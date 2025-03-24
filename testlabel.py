import serial
import csv
import keyboard  # 키 입력을 감지하는 라이브러리

# 시리얼 포트 설정
ser = serial.Serial('COM3', baudrate=115200)
k = []  # 데이터 저장 리스트
label = 1  # 초기 레이블 값 설정
# 1: 중립 2: 주먹 3: 가위 4: 총 5: 보
save_data = False  # 저장 모드 초기 상태: OFF

print("➡️ 's' 키: 저장 모드 ON/OFF")
print("⏹️ 종료하려면 'q' 키를 누르세요.")

try:
    while True:
        data = ser.readline().decode('utf-8').strip().split(',')  # 데이터 수신

        # 저장 모드가 비활성화된 경우 데이터 출력
        if not save_data:
            print(data)

        # 레이블 설정 (숫자 1~5)
        for i in range(1, 6):
            if keyboard.is_pressed(str(i)):
                label = i
                print(f"🏷️ 레이블 {label} 설정")

        # 저장 모드일 경우에만 데이터 추가
        if save_data and len(data) == 2:
            print(data)  # 저장 모드일 경우에만 데이터 출력
            k.append(data + [label])

        # 키 입력에 따른 저장 모드 전환
        if keyboard.is_pressed('s') and not save_data:
            print("✅ 저장 모드 ON")
            save_data = True
        elif keyboard.is_pressed('a') and save_data:
            print("❌ 저장 모드 OFF")
            save_data = False

        # 'q' 키로 종료
        if keyboard.is_pressed('q'):
            print("\n🔚 데이터 수집 종료")
            break

except KeyboardInterrupt:
    print("\n🔚 데이터 수집 종료")

# CSV 파일 저장
if k:
    with open('EMG_test.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['CH1', 'CH2', 'Label'])  # 헤더 추가
        for item in k:
            writer.writerow(item)
        writer.writerow(['Total', str(len(k))])  # 총 데이터 개수 추가
    print("✅ 데이터가 성공적으로 저장되었습니다.")
else:
    print("❌ 저장된 데이터가 없습니다.")

ser.close()  # 시리얼 포트 종료
print("🔌 시리얼 포트 종료 완료")
