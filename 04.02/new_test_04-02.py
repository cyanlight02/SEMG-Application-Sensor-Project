import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from joblib import dump  # 추가

# 데이터 불러오기
df = pd.read_csv('emg_data.csv')

# 특성과 라벨 분리
X = df.drop('Label', axis=1).values
y = df['Label'].values

# 라벨 인코딩
y = to_categorical(y)

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# 데이터 스케일링
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Scaler 저장 ▶▶▶ 추가된 부분 ◀◀◀
dump(scaler, 'emg_scaler.bin')

# 모델 구성
model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))

# 모델 컴파일
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 콜백 설정
callbacks = [
    EarlyStopping(monitor='val_loss', patience=10),
    ModelCheckpoint('best_model.h5', save_best_only=True)
]

# 모델 학습
history = model.fit(X_train, y_train,
                    epochs=100,
                    batch_size=32,
                    validation_split=0.2,
                    callbacks=callbacks)

# 평가
model.load_weights('best_model.h5')
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {accuracy:.4f}')