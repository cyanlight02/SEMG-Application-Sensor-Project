int i = 0;
int sum = 0;
int IEMG = 0; //근육의 활동 정도
int MAV = 0; //수축 강도
int RMS = 0; //(근육에) 가한 힘의 크기
int sqrtSum = 0;
bool sensorActive = false;

void setup() 
{
  Serial.begin(115200);
  while (!Serial); // optionally wait for serial terminal to open
  Serial.println("MyoWare Example_01_analogRead_SINGLE");
}

void loop() 
{  
  //s로 시작 e로 종료
  if (Serial.available()) {
    char command = Serial.read();
    if (command == 's') {
      sensorActive = true;
      Serial.println("Sensor ON");
    } else if (command == 'e') {
      sensorActive = false;
      Serial.println("Sensor OFF");
    }
  }


  if (sensorActive) {
    int line = 0;
    int sensorValue = analogRead(A0); // read the input on analog pin A0
    sum += sensorValue;
    sqrtSum += sensorValue * sensorValue;
    i++;

    //특징 추출
    if (i > 3 && i % 3 == 1) { //200ms 단위로 세그먼트를 나눠서 분석
      IEMG = sum;
      MAV = sum / 4;
      RMS = sqrt(sqrtSum);
      //마지막 50ms는 다음 세그먼트에 포함시킴
      sum = sensorValue;
      sqrtSum = sensorValue * sensorValue;
    }

    //Serial.print("Value: ");
    Serial.print(sensorValue); // print out the value you read
    Serial.print(",");
    //Serial.print("IEMG: ");
    Serial.print(IEMG); // print out the value you read
    Serial.print(",");
    //Serial.print("MAV: ");
    Serial.print(MAV); // print out the value you read
    Serial.print(",");
    //Serial.print("RMS: ");
    Serial.println(RMS); // print out the value you read

    delay(50); // to avoid overloading the serial terminal
  }
  
}
