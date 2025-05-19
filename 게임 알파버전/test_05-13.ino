#include "Arduino.h"
#include "EMGFilters.h"
#include "SoftwareSerial.h"

#define SensorInputPin1 A5
#define SensorInputPin2 A0
#define BT_RX 10  // 아두이노 10번 핀은 블루투스 모듈의 TX와 연결
#define BT_TX 11  // 아두이노 11번 핀은 블루투스 모듈의 RX와 연결

// 센서 1 - 기본 이동 평균 필터 설정
#define FILTER_SIZE1 10
int filterBuffer1[FILTER_SIZE1] = {0};
int filterIndex1 = 0;

// 센서 2 - 중앙값 필터 + 이동 평균 필터
#define FILTER_SIZE2 15
int filterBuffer2[FILTER_SIZE2] = {0};
int sortedBuffer2[FILTER_SIZE2] = {0};  // 중앙값 계산용 정렬 버퍼
int filterIndex2 = 0;

// 센서 2용 EMA 필터 계수 (0에 가까울수록 변화가 느려짐, 0~1 사이 값)
#define EMA_ALPHA 0.5
int lastEmaValue2 = 0;

// 두 센서 모두 노이즈 임계값 필터 적용
#define NOISE_THRESHOLD1 1
#define NOISE_THRESHOLD2 2  // 센서2에 매우 강한 임계값 필터 적용
int prevFilteredValue1 = 0;
int prevFilteredValue2 = 0;

// 블루투스 시리얼 인스턴스 생성
SoftwareSerial BTSerial(BT_RX, BT_TX);

EMGFilters myFilter1;
EMGFilters myFilter2;
int sampleRate = SAMPLE_FREQ_500HZ;
int humFreq = NOTCH_FREQ_50HZ;

unsigned long timeBudget;

// 중앙값 필터를 위한 정렬 함수
void bubbleSort(int arr[], int size) {
    for(int i=0; i<size-1; i++) {
        for(int j=0; j<size-i-1; j++) {
            if(arr[j] > arr[j+1]) {
                int temp = arr[j];
                arr[j] = arr[j+1];
                arr[j+1] = temp;
            }
        }
    }
}

void setup() {
    // 버퍼 초기화
    for(int i=0; i<FILTER_SIZE1; i++) {
        filterBuffer1[i] = 0;
    }
    for(int i=0; i<FILTER_SIZE2; i++) {
        filterBuffer2[i] = 0;
        sortedBuffer2[i] = 0;
    }
    
    myFilter1.init(sampleRate, humFreq, true, true, true);
    myFilter2.init(sampleRate, humFreq, true, true, true);
    
    // 블루투스 시리얼만 초기화
    BTSerial.begin(9600);
    
    timeBudget = 1e6 / sampleRate;
    
    // 블루투스로 시작 메시지 전송
    delay(1000);
    BTSerial.println("EMG_BT_START");
}

void loop() {
    unsigned long timeStamp = micros();

    // 두 센서 처리
    int rawValue1 = analogRead(SensorInputPin1);
    int filteredValue1 = myFilter1.update(rawValue1);
    
    int rawValue2 = analogRead(SensorInputPin2);
    int filteredValue2 = myFilter2.update(rawValue2);

    // 센서 1 - 이동 평균 필터 적용
    filterBuffer1[filterIndex1] = filteredValue1;
    filterIndex1 = (filterIndex1 + 1) % FILTER_SIZE1;
    
    int avgValue1 = 0;
    for(int i=0; i<FILTER_SIZE1; i++) {
        avgValue1 += filterBuffer1[i];
    }
    avgValue1 /= FILTER_SIZE1;
    
    // 센서 1에 노이즈 임계값 필터 적용
    if (abs(avgValue1 - prevFilteredValue1) < NOISE_THRESHOLD1) {
        avgValue1 = prevFilteredValue1;
    } else {
        prevFilteredValue1 = avgValue1;
    }
    
    // 센서 2 - 다단계 필터링 적용
    // 1. 버퍼에 저장
    filterBuffer2[filterIndex2] = filteredValue2;
    filterIndex2 = (filterIndex2 + 1) % FILTER_SIZE2;
    
    // 2. 중앙값 필터 적용 (극단적인 변동 제거)
    for(int i=0; i<FILTER_SIZE2; i++) {
        sortedBuffer2[i] = filterBuffer2[i];
    }
    bubbleSort(sortedBuffer2, FILTER_SIZE2);
    int medianValue2 = sortedBuffer2[FILTER_SIZE2/2];
    
    // 3. 이동 평균 필터 적용 (추가 스무딩)
    int avgValue2 = 0;
    for(int i=0; i<FILTER_SIZE2; i++) {
        avgValue2 += filterBuffer2[i];
    }
    avgValue2 /= FILTER_SIZE2;
    
    // 4. EMA 필터 적용 (부드러운 변화)
    if (lastEmaValue2 == 0) {
        lastEmaValue2 = avgValue2;  // 첫 실행시 초기화
    }
    int emaValue2 = (EMA_ALPHA * avgValue2) + ((1.0 - EMA_ALPHA) * lastEmaValue2);
    lastEmaValue2 = emaValue2;
    
    // 5. 강화된 임계값 필터 적용
    if (abs(emaValue2 - prevFilteredValue2) < NOISE_THRESHOLD2) {
        emaValue2 = prevFilteredValue2;
    } else {
        prevFilteredValue2 = emaValue2;
    }

    // 블루투스로 데이터 전송
    BTSerial.print(avgValue1);
    BTSerial.print(",");
    BTSerial.println(emaValue2);

    // 샘플링 속도 유지
    unsigned long elapsedTime = micros() - timeStamp;
    if (elapsedTime < timeBudget) {
        delayMicroseconds(timeBudget - elapsedTime);
    }
}