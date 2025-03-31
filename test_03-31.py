pip install tensorflow pyserial pandas scikit-learn
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.utils import class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, LSTM, Dense, Dropout, BatchNormalization, MaxPooling1D
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from tensorflow.keras.utils import to_categorical
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

# 데이터 로드 및 전처리
def load_and_preprocess(file_path):
    data = pd.read_csv(file_path)
    
    # 클래스 분포 시각화
    plt.figure(figsize=(10,6))
    sns.countplot(x='Label', data=data)
    plt.title('Class Distribution')
    plt.show()
    
    # 특성과 레이블 분리
    X = data.drop(columns=['Label']).values
    y = data['Label'].values
    
    # Robust Scaling
    scaler = RobustScaler(quantile_range=(5, 95))
    X = scaler.fit_transform(X)
    
    # 차원 축소 (PCA)
    pca = PCA(n_components=0.95)
    X = pca.fit_transform(X)
    
    # 데이터 증강
    X, y = augment_data(X, y)
    
    return X, y, scaler, pca

# 데이터 증강 함수
def augment_data(X, y):
    augmented_X = []
    augmented_y = []
    
    for class_id in np.unique(y):
        class_indices = np.where(y == class_id)[0]
        class_data = X[class_indices]
        
        # 노이즈 추가
        noise = np.random.normal(0, 0.05, class_data.shape)
        augmented_X.append(class_data + noise)
        augmented_y.extend([class_id]*len(class_data))
        
        # 시간축 이동
        shifted = np.roll(class_data, shift=2, axis=1)
        augmented_X.append(shifted)
        augmented_y.extend([class_id]*len(class_data))
    
    return np.vstack([X] + augmented_X), np.hstack([y] + augmented_y)

# 시계열 데이터 생성
def create_sequences(X, y, window_size=50):
    X_seq, y_seq = [], []
    for i in range(len(X) - window_size):
        X_seq.append(X[i:i+window_size])
        y_seq.append(y[i+window_size])
    return np.array(X_seq), to_categorical(np.array(y_seq))

# 학습률 스케줄러
def lr_scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * np.exp(-0.1)

# 모델 아키텍처
def build_model(input_shape, num_classes):
    model = Sequential([
        Conv1D(64, 3, activation='relu', input_shape=input_shape, kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        MaxPooling1D(2),
        Dropout(0.3),
        
        LSTM(128, return_sequences=True, kernel_regularizer=l2(0.001)),
        LSTM(64, kernel_regularizer=l2(0.001)),
        
        Dense(128, activation='selu', kernel_initializer='lecun_normal'),
        BatchNormalization(),
        Dropout(0.4),
        
        Dense(num_classes, activation='softmax')
    ])
    
    optimizer = Adam(learning_rate=0.001, clipnorm=1.0)
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy', 
                           tf.keras.metrics.Precision(),
                           tf.keras.metrics.Recall()])
    return model

# 메인 실행 블록
if __name__ == "__main__":
    # 데이터 준비
    X, y, scaler, pca = load_and_preprocess('emg_data_2.csv')
    X_seq, y_seq = create_sequences(X, y)
    
    # 클래스 가중치 계산
    class_weights = class_weight.compute_class_weight(
        'balanced',
        classes=np.unique(np.argmax(y_seq, axis=1)),
        y=np.argmax(y_seq, axis=1)
    )
    class_weights = dict(enumerate(class_weights))
    
    # 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(
        X_seq, y_seq, test_size=0.2, stratify=y_seq, random_state=42
    )
    
    # 모델 구성
    model = build_model(X_train.shape[1:], y_seq.shape[1])
    
    # 콜백 설정
    callbacks = [
        EarlyStopping(monitor='val_accuracy', patience=15, restore_best_weights=True),
        LearningRateScheduler(lr_scheduler),
        ModelCheckpoint('best_model.h5', save_best_only=True)
    ]
    
    # 모델 학습
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=100,
        batch_size=128,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )
    
    # 성능 평가
    test_results = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n최종 테스트 성능:")
    print(f"정확도: {test_results[1]:.4f}")
    print(f"정밀도: {test_results[2]:.4f}")
    print(f"재현율: {test_results[3]:.4f}")
    
    # 학습 곡선 시각화
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy Over Epochs')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss Over Epochs')
    plt.legend()
    plt.show()

    # 모델 저장
    model.save('emg_classifier_v2.h5')
    pd.to_pickle(scaler, 'scaler.pkl')
    pd.to_pickle(pca, 'pca.pkl')
