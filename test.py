import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

# 데이터 로드
data = pd.read_csv('data_9.csv')

# EMG 데이터와 레이블 분리
X = data[['EMGData', 'EMGData2']].values
y = data['Label'].values

# 데이터 정규화
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 레이블을 원-핫 인코딩
y = to_categorical(y)

# 시계열 길이 설정
window_size = 30

# 시계열 데이터로 변형하는 함수
def create_dataset(X, y, window_size):
    Xs, ys = [], []
    for i in range(len(X) - window_size):
        Xs.append(X[i:i + window_size])
        ys.append(y[i + window_size])
    return np.array(Xs), np.array(ys)

# 데이터셋 생성
X, y = create_dataset(X, y, window_size)

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# LSTM 모델 정의
model = Sequential()
model.add(LSTM(100, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(50))  # 두 번째 LSTM 층 추가
model.add(Dropout(0.2))
model.add(Dense(50, activation='relu'))  # Dense 층 추가
model.add(BatchNormalization())
model.add(Dense(y.shape[1], activation='softmax'))  # 출력층

# 모델 컴파일
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 조기 종료 설정
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# 모델 학습
model.fit(X_train, y_train, epochs=200, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stopping])

# 모델 평가
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

model.save('trained_model_3.h5')
