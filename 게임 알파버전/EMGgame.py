import sys
import os
import time
import csv
import json
import random
import serial
import shutil
import numpy as np
import torch
from datetime import datetime
from collections import deque
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QPushButton, QLabel, QComboBox, 
                            QGridLayout, QMessageBox, QProgressBar, QFrame,
                            QSplashScreen, QSizePolicy, QLineEdit, QDialog,
                            QStackedWidget, QTabWidget, QTextEdit, QDoubleSpinBox,
                            QGraphicsDropShadowEffect)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread, QSize, QPropertyAnimation, QEasingCurve
from PyQt5.QtGui import QPixmap, QFont, QPalette, QColor, QIcon, QImage, QPainter, QLinearGradient, QGradient


# ------------------------------------------------------------------------
# 디자인 상수 - 네오모피즘 스타일
# ------------------------------------------------------------------------
# 이 부분을 수정하여 전체 디자인 색상을 바꿀 수 있습니다
BG_COLOR = "#E6E7EE"  # 밝은 회색 배경
TEXT_COLOR = "#31344b"  # 어두운 네이비
PRIMARY_COLOR = "#5e72e4"  # 보라색 계열 (강조색)
SECONDARY_COLOR = "#2dce89"  # 녹색 계열
ACCENT_COLOR = "#11cdef"  # 청록색
DANGER_COLOR = "#F5365C"  # 빨간색/위험 색상


# ------------------------------------------------------------------------
# 상수 및 설정
# ------------------------------------------------------------------------
# 센서 및 시리얼 설정
DEFAULT_SERIAL_PORT = 'COM9'  # 기본 시리얼 포트 (필요시 수정)
DEFAULT_BAUD_RATE = 115200    # 기본 통신 속도 (필요시 수정)

# 프로그램 경로 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
USERS_DIR = os.path.join(BASE_DIR, "users")
IMAGE_DIR = os.path.join(BASE_DIR, "images")

# 모델 관련 상수
SEQUENCE_LENGTH = 200  # 트랜스포머 모델용 시퀀스 길이
BUFFER_SIZE = 500      # 감지용 롤링 버퍼 크기

# 데이터 수집 관련 상수
DETECTION_WINDOW = 20  # 동작 감지에 사용할 윈도우 크기
DETECTION_THRESHOLD = 5  # Z-동작 감지 임계값 (필요시 수정하여 감도 조절)
COOLDOWN_PERIOD = 2    # 동작 저장 후 대기 시간(초)
POST_CAPTURE = 30      # 동작 종료 후 추가 캡처 프레임 수
TREND_WINDOW = 10      # 추세 감지에 사용할 윈도우 크기
TREND_THRESHOLD = 3    # 추세 감지 임계값 (필요시 수정하여 감도 조절)
DISPERSION_THRESHOLD = 3 # 분산 감지 임계값 (필요시 수정하여 감도 조절)

# 동작 라벨 상수
LABEL_SCISSORS = 1     # 가위
LABEL_ROCK = 2         # 바위
LABEL_PAPER = 3        # 보

# 게임 결과 상수
RESULT_WIN = 0
RESULT_LOSE = 1
RESULT_DRAW = 2

# 모델 파일 이름
MODEL_FILE = "emg_model.pth"
CONFIG_FILE = "config.json"

# 학습 설정 - 고성능 모델
HIGH_PERFORMANCE_MODEL = {
    'd_model': 256,
    'nhead': 8,
    'num_layers': 4,
    'dim_feedforward': 512,
    'learning_rate': 0.0001,
    'batch_size': 8,
    'num_epochs': 150
}


# ------------------------------------------------------------------------
# 네오모피즘 UI 컴포넌트
# ------------------------------------------------------------------------
# 네오모피즘 버튼 클래스
class NeumorphicButton(QPushButton):
    def __init__(self, text, color=None, parent=None):
        super().__init__(text, parent)
        self.color = color
        self.setFixedHeight(50)
        self.setFont(QFont('Segoe UI', 12))
        self.setCursor(Qt.PointingHandCursor)
        
        # 기본 스타일 설정
        self.setFlat(True)
        if color:
            self.setStyleSheet(f"""
                QPushButton {{
                    background-color: {color};
                    color: white;
                    border-radius: 12px;
                    padding: 10px 20px;
                    text-align: center;
                }}
                QPushButton:hover {{
                    background-color: {self.lighten_color(color, 10)};
                }}
                QPushButton:pressed {{
                    background-color: {self.darken_color(color, 10)};
                }}
                QPushButton:disabled {{
                    background-color: #D1D9E6;
                    color: #8898aa;
                }}
            """)
        else:
            # 네오모피즘 스타일 - 그림자 효과
            self.setStyleSheet(f"""
                QPushButton {{
                    background-color: {BG_COLOR};
                    color: {TEXT_COLOR};
                    border-radius: 12px;
                    padding: 10px 20px;
                    text-align: center;
                    border: none;
                }}
                QPushButton:hover {{
                    background-color: #EFF0F7;
                }}
                QPushButton:pressed {{
                    background-color: #D1D9E6;
                }}
                QPushButton:disabled {{
                    color: #8898aa;
                }}
            """)
        
        # 그림자 효과 추가
        if not color:
            # 네오모피즘 그림자 효과
            shadow = QGraphicsDropShadowEffect(self)
            shadow.setBlurRadius(15)
            shadow.setColor(QColor(0, 0, 0, 50))
            shadow.setOffset(3, 3)
            self.setGraphicsEffect(shadow)
        else:
            # 컬러 버튼은 일반 그림자
            shadow = QGraphicsDropShadowEffect(self)
            shadow.setBlurRadius(15)
            shadow.setColor(QColor(0, 0, 0, 60))
            shadow.setOffset(2, 2)
            self.setGraphicsEffect(shadow)
    
    def lighten_color(self, color, amount=20):
        """색상을 밝게 만듭니다."""
        c = QColor(color)
        h, s, l, a = c.getHslF()
        l = min(1.0, l + amount / 100)
        c.setHslF(h, s, l, a)
        return c.name()
    
    def darken_color(self, color, amount=20):
        """색상을 어둡게 만듭니다."""
        c = QColor(color)
        h, s, l, a = c.getHslF()
        l = max(0.0, l - amount / 100)
        c.setHslF(h, s, l, a)
        return c.name()


# 네오모피즘 카드 위젯
class NeumorphicCard(QFrame):
    def __init__(self, parent=None, inset=False):
        super().__init__(parent)
        self.setFrameShape(QFrame.StyledPanel)
        
        if inset:
            # 음각(inset) 스타일 - 들어간 느낌
            self.setStyleSheet(f"""
                QFrame {{
                    background-color: #D1D9E6;
                    border-radius: 15px;
                    border: none;
                }}
            """)
            
            # 안쪽으로 들어간 효과의 그림자
            inner_shadow = QGraphicsDropShadowEffect(self)
            inner_shadow.setBlurRadius(15)
            inner_shadow.setColor(QColor(0, 0, 0, 60))
            inner_shadow.setOffset(3, 3)
            self.setGraphicsEffect(inner_shadow)
        else:
            # 양각(outset) 스타일 - 나온 느낌
            self.setStyleSheet(f"""
                QFrame {{
                    background-color: {BG_COLOR};
                    border-radius: 15px;
                    border: none;
                }}
            """)
            
            # 바깥쪽 그림자 효과
            outer_shadow = QGraphicsDropShadowEffect(self)
            outer_shadow.setBlurRadius(20)
            outer_shadow.setColor(QColor(0, 0, 0, 50))
            outer_shadow.setOffset(5, 5)
            self.setGraphicsEffect(outer_shadow)


# 로고 위젯 (네오모피즘 스타일 로고)
class LogoWidget(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(200, 100)
        
        # 임시 로고 이미지 생성 및 설정
        logo_file = os.path.join(IMAGE_DIR, "neumorphic_logo.png")
        if not os.path.exists(logo_file):
            self.create_logo(logo_file)
        
        self.setPixmap(QPixmap(logo_file))
        self.setScaledContents(True)
    
    def create_logo(self, file_path):
        """네오모피즘 스타일의 로고 이미지 생성"""
        pixmap = QPixmap(200, 100)
        pixmap.fill(Qt.transparent)
        
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # 배경 (부드러운 곡선)
        painter.setBrush(QColor(BG_COLOR))
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(0, 0, 200, 100, 20, 20)
        
        # 텍스트 그림자 효과 (하단 우측)
        painter.setPen(QColor(0, 0, 0, 30))
        painter.setFont(QFont("Segoe UI", 24, QFont.Bold))
        painter.drawText(pixmap.rect().adjusted(3, 3, 3, 3), Qt.AlignCenter, "EMG Game")
        
        # 텍스트 하이라이트 효과 (상단 좌측)
        painter.setPen(QColor(255, 255, 255, 180))
        painter.drawText(pixmap.rect().adjusted(-1, -1, -1, -1), Qt.AlignCenter, "EMG Game")
        
        # 메인 텍스트
        painter.setPen(QColor(TEXT_COLOR))
        painter.drawText(pixmap.rect(), Qt.AlignCenter, "EMG Game")
        
        painter.end()
        pixmap.save(file_path)


# ------------------------------------------------------------------------
# 유틸리티 함수
# ------------------------------------------------------------------------
def ensure_dir_exists(dir_path):
    """디렉토리가 존재하지 않으면 생성"""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        return True
    return False

def get_user_dir(username, password):
    """사용자 디렉토리 경로 반환"""
    # 간단한 해시 생성 (실제 애플리케이션에서는 더 안전한 방법 사용 권장)
    dir_name = f"{username}_{hash(password) % 10000:04d}"
    user_dir = os.path.join(USERS_DIR, dir_name)
    ensure_dir_exists(user_dir)
    return user_dir

def user_has_model(username, password):
    """사용자의 학습된 모델이 있는지 확인"""
    user_dir = get_user_dir(username, password)
    model_path = os.path.join(user_dir, MODEL_FILE)
    return os.path.exists(model_path)

def load_user_config(username, password):
    """사용자 설정 로드"""
    user_dir = get_user_dir(username, password)
    config_path = os.path.join(user_dir, CONFIG_FILE)
    
    # 기본 설정
    default_config = {
        "sensitivity": 1.0,
        "model_accuracy": 0.0,
        "data_count": {str(LABEL_SCISSORS): 0, str(LABEL_ROCK): 0, str(LABEL_PAPER): 0}
    }
    
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            # 누락된 설정이 있으면 기본값으로 보완
            for key, value in default_config.items():
                if key not in config:
                    config[key] = value
            return config
        except:
            pass
    
    # 설정 파일이 없거나 오류가 발생한 경우 기본 설정 사용
    save_user_config(username, password, default_config)
    return default_config

def save_user_config(username, password, config):
    """사용자 설정 저장"""
    user_dir = get_user_dir(username, password)
    config_path = os.path.join(user_dir, CONFIG_FILE)
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)

def create_placeholder_images():
    """게임에 필요한 임시 이미지 생성"""
    ensure_dir_exists(IMAGE_DIR)
    
    # 임시 이미지 파일 경로
    image_files = {
        "logo": os.path.join(IMAGE_DIR, "logo.png"),
        "background": os.path.join(IMAGE_DIR, "background.png"),
        "scissors": os.path.join(IMAGE_DIR, "scissors.png"),
        "rock": os.path.join(IMAGE_DIR, "rock.png"),
        "paper": os.path.join(IMAGE_DIR, "paper.png"),
        "ready": os.path.join(IMAGE_DIR, "ready.png"),
        "win": os.path.join(IMAGE_DIR, "win.png"),
        "lose": os.path.join(IMAGE_DIR, "lose.png"),
        "draw": os.path.join(IMAGE_DIR, "draw.png"),
        "connection_good": os.path.join(IMAGE_DIR, "connection_good.png"),
        "connection_medium": os.path.join(IMAGE_DIR, "connection_medium.png"),
        "connection_bad": os.path.join(IMAGE_DIR, "connection_bad.png"),
    }
    
    # 각 이미지 색상 및 크기 설정
    image_specs = {
        "logo": ((100, 100, 200), (400, 200)),
        "background": ((50, 50, 80), (800, 600)),
        "scissors": ((180, 180, 180), (300, 300)),
        "rock": ((200, 200, 200), (300, 300)),
        "paper": ((240, 240, 240), (300, 300)),
        "ready": ((100, 200, 100), (300, 100)),
        "win": ((100, 200, 100), (300, 100)),
        "lose": ((200, 100, 100), (300, 100)),
        "draw": ((200, 200, 100), (300, 100)),
        "connection_good": ((50, 200, 50), (100, 100)),
        "connection_medium": ((200, 200, 50), (100, 100)),
        "connection_bad": ((200, 50, 50), (100, 100)),
    }
    
    # 이미지 파일 생성
    for name, file_path in image_files.items():
        if not os.path.exists(file_path):
            color, size = image_specs[name]
            image = QImage(size[0], size[1], QImage.Format_RGB32)
            image.fill(QColor(*color))
            
            # 간단한 텍스트 추가
            painter = QPainter(image)
            painter.setPen(QColor(0, 0, 0))
            painter.setFont(QFont('Arial', 30))
            painter.drawText(image.rect(), Qt.AlignCenter, name.upper())
            painter.end()
            
            image.save(file_path)
            print(f"임시 이미지 '{file_path}' 생성 완료")
    
    return image_files

def get_computer_choice():
    """컴퓨터의 가위/바위/보 선택"""
    return random.choice([LABEL_SCISSORS, LABEL_ROCK, LABEL_PAPER])

def determine_winner(player_choice, computer_choice):
    """가위바위보 승패 결정"""
    # 무승부
    if player_choice == computer_choice:
        return RESULT_DRAW
    
    # 승리 조건: 가위>보, 바위>가위, 보>바위
    if ((player_choice == LABEL_SCISSORS and computer_choice == LABEL_PAPER) or
        (player_choice == LABEL_ROCK and computer_choice == LABEL_SCISSORS) or 
        (player_choice == LABEL_PAPER and computer_choice == LABEL_ROCK)):
        return RESULT_WIN
    
    # 나머지는 패배
    return RESULT_LOSE

def get_label_name(label):
    """라벨 번호에 맞는 이름 반환"""
    if label == LABEL_SCISSORS:
        return "가위"
    elif label == LABEL_ROCK:
        return "바위"
    elif label == LABEL_PAPER:
        return "보"
    else:
        return f"알 수 없음({label})"


# ------------------------------------------------------------------------
# EMG 모델 정의
# ------------------------------------------------------------------------
class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1), :]
        return self.dropout(x)

class TransformerEncoderLayer(torch.nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super().__init__()
        self.self_attn = torch.nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = torch.nn.Linear(d_model, dim_feedforward)
        self.dropout = torch.nn.Dropout(dropout)
        self.linear2 = torch.nn.Linear(dim_feedforward, d_model)
        self.norm1 = torch.nn.LayerNorm(d_model)
        self.norm2 = torch.nn.LayerNorm(d_model)
        self.dropout1 = torch.nn.Dropout(dropout)
        self.dropout2 = torch.nn.Dropout(dropout)
        self.activation = torch.nn.ReLU()

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src2 = self.norm1(src)
        src2, _ = self.self_attn(src2, src2, src2, attn_mask=src_mask,
                                key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

class EMGTransformer(torch.nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, dim_feedforward, num_classes, dropout=0.1):
        super().__init__()
        
        self.embedding = torch.nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        encoder_layers = []
        for _ in range(num_layers):
            encoder_layers.append(TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout))
        self.transformer_encoder = torch.nn.ModuleList(encoder_layers)
        
        self.fc = torch.nn.Linear(d_model, num_classes)
    
    def forward(self, x, mask=None):
        x = self.embedding(x)
        x = self.pos_encoder(x)
        
        for layer in self.transformer_encoder:
            x = layer(x, src_mask=mask)
        
        # 시퀀스의 평균을 취해 분류
        x = torch.mean(x, dim=1)
        
        output = self.fc(x)
        return output


# ------------------------------------------------------------------------
# 시리얼 통신 스레드
# ------------------------------------------------------------------------
class SerialThread(QThread):
    """
    SerialThread 클래스: EMG 센서와의 시리얼 통신을 담당하는 스레드
    - data_received: EMG 센서 데이터 수신 시 발생하는 시그널
    - error_occurred: 오류 발생 시 발생하는 시그널
    - connection_status: 연결 상태 변경 시 발생하는 시그널
    - connected: 연결 성공 시 발생하는 시그널
    - disconnected: 연결 종료 시 발생하는 시그널
    """
    data_received = pyqtSignal(float, float)  # EMG 데이터 (센서1, 센서2) 시그널
    error_occurred = pyqtSignal(str)          # 오류 메시지 시그널
    connection_status = pyqtSignal(bool, str) # 연결 상태 및 메시지 시그널
    connected = pyqtSignal()                  # 연결 성공 시그널
    disconnected = pyqtSignal()              # 연결 종료 시그널
    
    def __init__(self, port=DEFAULT_SERIAL_PORT, baud_rate=DEFAULT_BAUD_RATE, parent=None):
        super().__init__(parent)
        self.port = port
        self.baud_rate = baud_rate
        self.running = False
        self.ser = None
        self.auto_connect = True
        
    def set_port(self, port):
        """통신 포트 설정"""
        self.port = port
        
    def set_baud_rate(self, baud_rate):
        """통신 속도 설정"""
        self.baud_rate = baud_rate
        
    def connect_serial(self):
        """시리얼 연결 시작"""
        if self.ser:
            self.disconnect_serial()
            
        try:
            self.ser = serial.Serial(self.port, self.baud_rate, timeout=1)
            time.sleep(1)  # 연결 안정화 대기
            self.ser.reset_input_buffer()
            self.connection_status.emit(True, f"연결 성공: {self.port}")
            self.connected.emit()
            return True
        except Exception as e:
            self.connection_status.emit(False, f"연결 실패: {str(e)}")
            self.ser = None
            return False
    
    def disconnect_serial(self):
        """시리얼 연결 종료"""
        if self.ser:
            self.running = False
            try:
                self.ser.close()
            except:
                pass
            self.ser = None
            self.disconnected.emit()
            self.connection_status.emit(False, "연결 종료")
            
    def run(self):
        """스레드 실행"""
        # 자동 연결 모드인 경우 연결 시도
        if self.auto_connect and not self.ser:
            if not self.connect_serial():
                self.error_occurred.emit("자동 연결 실패")
                return
                
        if not self.ser:
            self.error_occurred.emit("연결되지 않음")
            return
            
        self.running = True
        while self.running:
            try:
                if self.ser and self.ser.in_waiting > 0:
                    line = self.ser.readline().decode().strip()
                    if ',' in line:
                        try:
                            values = line.split(',')
                            if len(values) >= 2:
                                s1 = float(values[0])
                                s2 = float(values[1])
                                self.data_received.emit(s1, s2)
                        except ValueError:
                            continue
                time.sleep(0.001)  # CPU 부하 감소
            except Exception as e:
                self.error_occurred.emit(f"데이터 읽기 오류: {str(e)}")
                self.running = False
                break
                
        self.disconnect_serial()

    def get_available_ports(self):
        """사용 가능한 시리얼 포트 목록 가져오기"""
        ports = []
        try:
            import serial.tools.list_ports
            for port in serial.tools.list_ports.comports():
                ports.append(port.device)
        except:
            pass
            
        # 기본 포트가 없으면 추가
        if DEFAULT_SERIAL_PORT not in ports:
            ports.append(DEFAULT_SERIAL_PORT)
            
        return ports


# ------------------------------------------------------------------------
# EMG 데이터 처리 클래스
# ------------------------------------------------------------------------
class EMGProcessor:
    """
    EMGProcessor 클래스: EMG 센서 데이터를 처리하고 동작을 감지하는 클래스
    - 신호 처리와 동작 감지 로직을 담당
    - 감도 설정을 통해 동작 감지 민감도를 조절할 수 있음
    """
    def __init__(self, sensitivity=1.0):
        # 데이터 버퍼
        self.data_buffer = deque(maxlen=BUFFER_SIZE)
        self.sequence_buffer = []
        
        # 상태 변수
        self.is_recording = False
        self.cooldown = False
        self.cooldown_start = 0
        self.post_capture_count = 0
        self.baseline_mean = (0, 0)
        self.baseline_std = (1, 1)
        self.detection_blocked = False
        self.detection_block_time = 0
        self.detection_block_duration = 0
        
        # 설정
        self.sensitivity = sensitivity
        
    def reset(self):
        """상태 초기화"""
        self.data_buffer.clear()
        self.sequence_buffer.clear()
        self.is_recording = False
        self.cooldown = False
        self.post_capture_count = 0
        self.detection_blocked = False
        
    def set_sensitivity(self, sensitivity):
        """감도 설정 - 높을수록 작은 동작도 감지됨"""
        self.sensitivity = sensitivity
        
    def calculate_baseline(self):
        """버퍼 데이터에서 기준선 통계 계산"""
        if len(self.data_buffer) < 100:
            return (0, 0), (1, 1)
            
        data = np.array(self.data_buffer)
        sensor1_data = data[:, 0]
        sensor2_data = data[:, 1]
        
        mean1 = np.mean(sensor1_data)
        mean2 = np.mean(sensor2_data)
        std1 = np.std(sensor1_data) + 1
        std2 = np.std(sensor2_data) + 1
        
        return (mean1, mean2), (std1, std2)
        
    def process_data(self, s1, s2):
        """EMG 데이터 처리"""
        # 데이터 버퍼에 추가
        self.data_buffer.append((s1, s2))
        
        # 주기적으로 기준선 업데이트
        if not self.is_recording and len(self.data_buffer) % 50 == 0:
            self.baseline_mean, self.baseline_std = self.calculate_baseline()
            
        # 대기 시간 확인
        if self.cooldown:
            if time.time() - self.cooldown_start >= COOLDOWN_PERIOD:
                self.cooldown = False
                # 데이터 버퍼 초기화
                self.data_buffer.clear()
                
                # 일시적인 감지 차단
                self.detection_blocked = True
                self.detection_block_time = time.time()
                self.detection_block_duration = 1.0
                
                return "cooldown_ended"
        
        # 동작 감지
        elif not self.is_recording and self.detect_movement():
            self.is_recording = True
            self.sequence_buffer = list(self.data_buffer)[-30:]  # 감지 전 30프레임 포함
            self.post_capture_count = 0
            return "movement_detected"
            
        # 동작 기록 중
        elif self.is_recording:
            # 값 추가
            self.sequence_buffer.append((s1, s2))
            
            # 동작 종료 확인
            if not self.is_movement_continuing():
                self.post_capture_count += 1
                if self.post_capture_count >= POST_CAPTURE:
                    # 동작 기록 종료
                    result = None
                    if len(self.sequence_buffer) >= 20:
                        result = "movement_completed"
                    else:
                        result = "movement_too_short"
                        
                    # 상태 업데이트
                    self.is_recording = False
                    self.cooldown = True
                    self.cooldown_start = time.time()
                    sequence = self.sequence_buffer.copy()
                    self.sequence_buffer = []
                    
                    if result == "movement_completed":
                        return result, sequence
                    else:
                        return result, None
            else:
                self.post_capture_count = 0
                
            # 최대 길이 확인
            if len(self.sequence_buffer) >= SEQUENCE_LENGTH * 1.5:
                # 동작 기록 종료
                sequence = self.sequence_buffer.copy()
                self.is_recording = False
                self.cooldown = True
                self.cooldown_start = time.time()
                self.sequence_buffer = []
                return "movement_completed", sequence
                
        return None
        
    def detect_movement(self):
        """여러 방법을 통합한 동작 감지"""
        # 감지 차단 확인
        if self.detection_blocked:
            if time.time() - self.detection_block_time >= self.detection_block_duration:
                self.detection_blocked = False
            else:
                return False
                
        if len(self.data_buffer) < DETECTION_WINDOW:
            return False
            
        # 감지에 사용할 최근 데이터 가져오기
        recent_data = list(self.data_buffer)[-DETECTION_WINDOW:]
        
        # 1. Z-점수 기반 감지
        z_scores_s1 = [(x[0] - self.baseline_mean[0]) / self.baseline_std[0] for x in recent_data]
        z_scores_s2 = [(x[1] - self.baseline_mean[1]) / self.baseline_std[1] for x in recent_data]
        
        # 유의미한 편차 개수 계산 (감도 조절 적용)
        adjusted_threshold = DETECTION_THRESHOLD / self.sensitivity
        count_s1 = sum(1 for z in z_scores_s1 if abs(z) > adjusted_threshold)
        count_s2 = sum(1 for z in z_scores_s2 if abs(z) > adjusted_threshold)
        
        # 2. 추세 기반 감지
        if len(recent_data) >= TREND_WINDOW:
            trend_data = recent_data[-TREND_WINDOW:]
            s1_values = [x[0] for x in trend_data]
            s2_values = [x[1] for x in trend_data]
            
            # 기울기 측정
            s1_slope = np.polyfit(range(TREND_WINDOW), s1_values, 1)[0]
            s2_slope = np.polyfit(range(TREND_WINDOW), s2_values, 1)[0]
            
            # 추세가 유의미하게 증가하거나 감소하는지 확인
            trend_detected = (abs(s1_slope) > TREND_THRESHOLD * self.baseline_std[0] or 
                             abs(s2_slope) > TREND_THRESHOLD * self.baseline_std[1])
        else:
            trend_detected = False
        
        # 3. 분산 기반 감지
        recent_window = recent_data[-5:]  # 가장 최근 5개 프레임
        s1_recent = [x[0] for x in recent_window]
        s2_recent = [x[1] for x in recent_window]
        
        s1_var = np.var(s1_recent)
        s2_var = np.var(s2_recent)
        
        variance_detected = (s1_var > (self.baseline_std[0] * DISPERSION_THRESHOLD * self.sensitivity) or 
                            s2_var > (self.baseline_std[1] * DISPERSION_THRESHOLD * self.sensitivity))
        
        # 종합 판단: 세 가지 방법 중 하나라도 해당되면 동작 감지
        movement_detected = ((count_s1 >= 3 or count_s2 >= 3) or trend_detected or variance_detected) and not self.is_recording and not self.cooldown

        return movement_detected
        
    def is_movement_continuing(self):
        """동작이 계속되고 있는지 확인"""
        if len(self.data_buffer) < 5:
            return True
            
        # 최근 데이터 가져오기
        recent_data = list(self.data_buffer)[-5:]
        
        # Z-점수 계산 (감도 조절 적용)
        adjusted_threshold = (DETECTION_THRESHOLD/2) / self.sensitivity
        z_scores_s1 = [(x[0] - self.baseline_mean[0]) / self.baseline_std[0] for x in recent_data]
        z_scores_s2 = [(x[1] - self.baseline_mean[1]) / self.baseline_std[1] for x in recent_data]
        
        # 두 가지 조건 중 하나라도 만족하면 동작 계속으로 판단
        # 1. Z-점수 기반
        for z1, z2 in zip(z_scores_s1, z_scores_s2):
            if abs(z1) > adjusted_threshold or abs(z2) > adjusted_threshold:
                return True
                
        # 2. 추세 또는 분산 기반
        if len(recent_data) >= 5:
            s1_values = [x[0] for x in recent_data]
            s2_values = [x[1] for x in recent_data]
            
            # 분산 확인
            s1_var = np.var(s1_values)
            s2_var = np.var(s2_values)
            
            if (s1_var > (self.baseline_std[0] * DISPERSION_THRESHOLD * self.sensitivity) or 
                s2_var > (self.baseline_std[1] * DISPERSION_THRESHOLD * self.sensitivity)):
                return True
        
        return False
        
    def get_connection_quality(self):
        """연결 품질 평가 (0: 나쁨, 1: 보통, 2: 좋음)"""
        if len(self.data_buffer) < 100:
            return 0
            
        data = np.array(self.data_buffer)
        sensor1_data = data[-100:, 0]
        sensor2_data = data[-100:, 1]
        
        # 노이즈 수준 평가
        std1 = np.std(sensor1_data)
        std2 = np.std(sensor2_data)
        
        # 신호 안정성 평가
        stability = (std1 + std2) / 2
        
        if stability < 0.5:
            return 2  # 좋음
        elif stability < 2.0:
            return 1  # 보통
        else:
            return 0  # 나쁨


# ------------------------------------------------------------------------
# EMG 모델 예측 클래스
# ------------------------------------------------------------------------
class EMGPredictor:
    """
    EMGPredictor 클래스: 학습된 모델을 사용하여 EMG 데이터를 통해 동작 예측 
    - 가위/바위/보 동작을 인식하는 모델 로드
    - 예측 기능 제공
    """
    def __init__(self, model_path=None):
        self.model = None
        self.scaler = None
        self.config = None
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
            
    def load_model(self, model_path):
        """학습된 모델 로드"""
        try:
            checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
            self.config = checkpoint['config']
            
            # 모델 생성
            self.model = EMGTransformer(
                input_dim=2,  # EMG 센서 2개
                d_model=self.config['d_model'],
                nhead=self.config['nhead'],
                num_layers=self.config['num_layers'],
                dim_feedforward=self.config['dim_feedforward'],
                num_classes=4,  # 최소 4개 클래스 (배경 + 가위,바위,보)
                dropout=0
            )
            
            # 모델 가중치 로드
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            
            # 스케일러 로드
            self.scaler = checkpoint['scaler']
            
            return True
        except Exception as e:
            print(f"모델 로드 오류: {str(e)}")
            return False
            
    def predict(self, sequence):
        """EMG 시퀀스 예측"""
        if self.model is None:
            return None, 0
            
        # 데이터 준비
        sequence_copy = sequence.copy()
        
        # 패딩 처리
        sequence_length = SEQUENCE_LENGTH
        if len(sequence_copy) < sequence_length:
            padding = [(0, 0)] * (sequence_length - len(sequence_copy))
            sequence_copy += padding
        elif len(sequence_copy) > sequence_length:
            sequence_copy = sequence_copy[:sequence_length]
            
        # 예측용 배열 생성
        X = np.array(sequence_copy).reshape(1, sequence_length, 2)
        
        # 정규화
        X_reshaped = X.reshape(-1, X.shape[-1])
        X_normalized = self.scaler.transform(X_reshaped)
        X = X_normalized.reshape(X.shape)
        
        # 예측
        try:
            with torch.no_grad():
                X_tensor = torch.tensor(X, dtype=torch.float32)
                outputs = self.model(X_tensor)
                probabilities = torch.softmax(outputs, dim=1)[0]
                prediction = torch.argmax(probabilities).item()
                confidence = probabilities[prediction].item()
                
            return prediction, confidence
        except Exception as e:
            print(f"예측 오류: {str(e)}")
            return None, 0


# ------------------------------------------------------------------------
# 로그인 위젯
# ------------------------------------------------------------------------
class LoginWidget(QWidget):
    """
    LoginWidget 클래스: 로그인 화면을 제공하는 위젯
    - 사용자 인증과 메인 화면으로의 전환을 처리
    - 네오모피즘 디자인 적용
    """
    login_successful = pyqtSignal(str, str)  # username, password
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        
    def init_ui(self):
        """UI 초기화 - 네오모피즘 디자인 적용"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(40, 40, 40, 40)
        layout.setSpacing(30)
        
        # 배경 설정
        self.setStyleSheet(f"background-color: {BG_COLOR};")
        
        # 로고 및 제목 카드
        title_card = NeumorphicCard()
        title_layout = QVBoxLayout(title_card)
        title_layout.setContentsMargins(20, 20, 20, 20)
        
        # 로고 위젯
        self.logo = LogoWidget()
        
        # 타이틀 레이블
        self.title_label = QLabel("EMG 게임 시스템")
        self.title_label.setStyleSheet(f"color: {TEXT_COLOR}; font-size: 32px; font-weight: bold;")
        self.title_label.setAlignment(Qt.AlignCenter)
        
        title_layout.addWidget(self.logo, 0, Qt.AlignCenter)
        title_layout.addWidget(self.title_label)
        
        # 로그인 카드
        login_card = NeumorphicCard()
        login_layout = QVBoxLayout(login_card)
        login_layout.setContentsMargins(40, 40, 40, 40)
        login_layout.setSpacing(15)
        
        # 로그인 폼
        self.username_label = QLabel("아이디:")
        self.username_label.setStyleSheet(f"color: {TEXT_COLOR}; font-size: 16px; font-weight: bold;")
        
        # 사용자명 입력 필드
        username_frame = NeumorphicCard(inset=True)
        username_layout = QVBoxLayout(username_frame)
        username_layout.setContentsMargins(10, 5, 10, 5)
        
        self.username_edit = QLineEdit()
        self.username_edit.setPlaceholderText("아이디 입력")
        self.username_edit.setStyleSheet(f"""
            QLineEdit {{
                background-color: transparent;
                border: none;
                color: {TEXT_COLOR};
                font-size: 16px;
            }}
        """)
        self.username_edit.setFixedHeight(40)
        
        username_layout.addWidget(self.username_edit)
        
        # 비밀번호 레이블
        self.password_label = QLabel("비밀번호:")
        self.password_label.setStyleSheet(f"color: {TEXT_COLOR}; font-size: 16px; font-weight: bold;")
        
        # 비밀번호 입력 필드
        password_frame = NeumorphicCard(inset=True)
        password_layout = QVBoxLayout(password_frame)
        password_layout.setContentsMargins(10, 5, 10, 5)
        
        self.password_edit = QLineEdit()
        self.password_edit.setPlaceholderText("비밀번호 입력")
        self.password_edit.setEchoMode(QLineEdit.Password)
        self.password_edit.setStyleSheet(f"""
            QLineEdit {{
                background-color: transparent;
                border: none;
                color: {TEXT_COLOR};
                font-size: 16px;
            }}
        """)
        self.password_edit.setFixedHeight(40)
        
        password_layout.addWidget(self.password_edit)
        
        # 로그인 버튼 (강조 색상)
        self.login_btn = NeumorphicButton("로그인", PRIMARY_COLOR)
        self.login_btn.setFixedHeight(55)
        self.login_btn.setFont(QFont("Segoe UI", 16))
        self.login_btn.clicked.connect(self.login)
        
        # 안내 메시지
        self.message_label = QLabel("아이디와 비밀번호는 데이터 저장 위치로 사용됩니다.\n아무 값이나 입력하셔도 됩니다.")
        self.message_label.setAlignment(Qt.AlignCenter)
        self.message_label.setStyleSheet(f"color: #8898aa; font-size: 14px;")
        
        # 로그인 카드에 위젯 추가
        login_layout.addWidget(self.username_label)
        login_layout.addWidget(username_frame)
        login_layout.addSpacing(5)
        login_layout.addWidget(self.password_label)
        login_layout.addWidget(password_frame)
        login_layout.addSpacing(20)
        login_layout.addWidget(self.login_btn)
        login_layout.addWidget(self.message_label)
        
        # 메인 레이아웃에 카드 추가
        layout.addWidget(title_card)
        layout.addWidget(login_card)
        
    def login(self):
        """로그인 처리"""
        username = self.username_edit.text().strip()
        password = self.password_edit.text().strip()
        
        if not username:
            QMessageBox.warning(self, "로그인 오류", "아이디를 입력해주세요.")
            return
            
        if not password:
            QMessageBox.warning(self, "로그인 오류", "비밀번호를 입력해주세요.")
            return
            
        # 로그인 성공 신호 발생
        self.login_successful.emit(username, password)
        
    def clear_fields(self):
        """입력 필드 초기화"""
        self.username_edit.clear()
        self.password_edit.clear()
        self.username_edit.setFocus()


# ------------------------------------------------------------------------
# 메인 화면 위젯
# ------------------------------------------------------------------------
class MainMenuWidget(QWidget):
    """
    MainMenuWidget 클래스: 메인 메뉴 화면을 제공하는 위젯
    - 각 기능(가이드, 데이터 수집, 게임)으로의 접근 제공
    - 네오모피즘 디자인이 적용됨
    """
    show_guide = pyqtSignal()
    show_data_collection = pyqtSignal()
    show_rps_game = pyqtSignal()
    show_muk_game = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.username = ""
        self.init_ui()
        
    def init_ui(self):
        """UI 초기화 - 네오모피즘 디자인 적용"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(40, 40, 40, 40)
        layout.setSpacing(30)
        
        # 배경 설정
        self.setStyleSheet(f"background-color: {BG_COLOR};")
        
        # 환영 카드
        welcome_card = NeumorphicCard()
        welcome_layout = QVBoxLayout(welcome_card)
        welcome_layout.setContentsMargins(20, 20, 20, 20)
        
        # 환영 메시지
        self.welcome_label = QLabel("어서오세요, 사용자님! 저랑 게임하실래요?")
        self.welcome_label.setStyleSheet(f"color: {TEXT_COLOR}; font-size: 28px; font-weight: bold;")
        self.welcome_label.setAlignment(Qt.AlignCenter)
        
        welcome_layout.addWidget(self.welcome_label)
        
        # 메뉴 그리드 생성
        menu_grid = QGridLayout()
        menu_grid.setSpacing(30)
        
        # 각 메뉴 항목을 카드로 생성
        # 1. 사용 가이드
        guide_card = NeumorphicCard()
        guide_layout = QVBoxLayout(guide_card)
        guide_layout.setContentsMargins(20, 20, 20, 20)
        
        # 아이콘 표시 (이모지)
        guide_icon = QLabel("📋")
        guide_icon.setStyleSheet("font-size: 40px;")
        guide_icon.setAlignment(Qt.AlignCenter)
        
        # 타이틀
        guide_title = QLabel("사용 가이드")
        guide_title.setStyleSheet(f"color: {TEXT_COLOR}; font-size: 20px; font-weight: bold;")
        guide_title.setAlignment(Qt.AlignCenter)
        
        # 설명
        guide_desc = QLabel("EMG 센서 설정 및\n기본 사용법 안내")
        guide_desc.setStyleSheet("color: #8898aa; font-size: 14px;")
        guide_desc.setAlignment(Qt.AlignCenter)
        
        # 버튼
        self.guide_btn = NeumorphicButton("시작하기")
        self.guide_btn.clicked.connect(self.on_guide_clicked)
        
        guide_layout.addWidget(guide_icon)
        guide_layout.addWidget(guide_title)
        guide_layout.addWidget(guide_desc)
        guide_layout.addWidget(self.guide_btn)
        
        # 2. 동작 학습
        training_card = NeumorphicCard()
        training_layout = QVBoxLayout(training_card)
        training_layout.setContentsMargins(20, 20, 20, 20)
        
        # 아이콘
        training_icon = QLabel("🔄")
        training_icon.setStyleSheet("font-size: 40px;")
        training_icon.setAlignment(Qt.AlignCenter)
        
        # 타이틀
        training_title = QLabel("동작 학습 하러가기")
        training_title.setStyleSheet(f"color: {TEXT_COLOR}; font-size: 20px; font-weight: bold;")
        training_title.setAlignment(Qt.AlignCenter)
        
        # 설명
        training_desc = QLabel("가위/바위/보 동작\n인식 데이터 수집")
        training_desc.setStyleSheet("color: #8898aa; font-size: 14px;")
        training_desc.setAlignment(Qt.AlignCenter)
        
        # 버튼
        self.training_btn = NeumorphicButton("시작하기")
        self.training_btn.clicked.connect(self.on_training_clicked)
        
        training_layout.addWidget(training_icon)
        training_layout.addWidget(training_title)
        training_layout.addWidget(training_desc)
        training_layout.addWidget(self.training_btn)
        
        # 3. 가위바위보 게임
        rps_card = NeumorphicCard()
        rps_layout = QVBoxLayout(rps_card)
        rps_layout.setContentsMargins(20, 20, 20, 20)
        
        # 아이콘
        rps_icon = QLabel("✌️")
        rps_icon.setStyleSheet("font-size: 40px;")
        rps_icon.setAlignment(Qt.AlignCenter)
        
        # 타이틀
        rps_title = QLabel("가위바위보 게임")
        rps_title.setStyleSheet(f"color: {TEXT_COLOR}; font-size: 20px; font-weight: bold;")
        rps_title.setAlignment(Qt.AlignCenter)
        
        # 설명
        rps_desc = QLabel("EMG 센서로 하는\n가위바위보 게임")
        rps_desc.setStyleSheet("color: #8898aa; font-size: 14px;")
        rps_desc.setAlignment(Qt.AlignCenter)
        
        # 버튼
        self.rps_btn = NeumorphicButton("플레이", SECONDARY_COLOR)
        self.rps_btn.clicked.connect(self.on_rps_clicked)
        
        rps_layout.addWidget(rps_icon)
        rps_layout.addWidget(rps_title)
        rps_layout.addWidget(rps_desc)
        rps_layout.addWidget(self.rps_btn)
        
        # 4. 묵찌빠 게임
        muk_card = NeumorphicCard()
        muk_layout = QVBoxLayout(muk_card)
        muk_layout.setContentsMargins(20, 20, 20, 20)
        
        # 아이콘
        muk_icon = QLabel("🎮")
        muk_icon.setStyleSheet("font-size: 40px;")
        muk_icon.setAlignment(Qt.AlignCenter)
        
        # 타이틀
        muk_title = QLabel("묵찌빠 게임")
        muk_title.setStyleSheet(f"color: {TEXT_COLOR}; font-size: 20px; font-weight: bold;")
        muk_title.setAlignment(Qt.AlignCenter)
        
        # 설명
        muk_desc = QLabel("EMG 센서로 하는\n묵찌빠 게임")
        muk_desc.setStyleSheet("color: #8898aa; font-size: 14px;")
        muk_desc.setAlignment(Qt.AlignCenter)
        
        # 버튼
        self.muk_btn = NeumorphicButton("플레이", SECONDARY_COLOR)
        self.muk_btn.clicked.connect(self.on_muk_clicked)
        
        muk_layout.addWidget(muk_icon)
        muk_layout.addWidget(muk_title)
        muk_layout.addWidget(muk_desc)
        muk_layout.addWidget(self.muk_btn)
        
        # 그리드에 카드 추가
        menu_grid.addWidget(guide_card, 0, 0)
        menu_grid.addWidget(training_card, 0, 1)
        menu_grid.addWidget(rps_card, 1, 0)
        menu_grid.addWidget(muk_card, 1, 1)
        
        # 메인 레이아웃에 추가
        layout.addWidget(welcome_card)
        layout.addLayout(menu_grid)
        
    def set_username(self, username):
        """사용자 이름 설정"""
        self.username = username
        # 환영 메시지 텍스트 수정 - 이 부분을 수정하여 환영 메시지를 변경할 수 있습니다
        self.welcome_label.setText(f"어서오세요, {username}님! 저랑 게임하실래요?")
        
    def on_guide_clicked(self):
        """사용 가이드 버튼 클릭 처리"""
        self.show_guide.emit()
        
    def on_training_clicked(self):
        """동작 학습 버튼 클릭 처리"""
        self.show_data_collection.emit()
        
    def on_rps_clicked(self):
        """가위바위보 게임 버튼 클릭 처리"""
        self.show_rps_game.emit()
        
    def on_muk_clicked(self):
        """묵찌빠 게임 버튼 클릭 처리"""
        self.show_muk_game.emit()


# ------------------------------------------------------------------------
# 사용 가이드 위젯
# ------------------------------------------------------------------------
class GuideWidget(QWidget):
    """
    GuideWidget 클래스: 사용 가이드를 제공하는 위젯
    - EMG 센서 사용법과 게임 진행 방법 소개
    - 단계별 가이드를 제공
    - 네오모피즘 디자인 적용
    """
    guide_completed = pyqtSignal(bool)  # True: 학습 필요, False: 메인 메뉴로 이동
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # 상태 변수
        self.current_page = 0
        self.total_pages = 3
        self.has_model = False
        self.username = ""
        self.password = ""
        
        # EMG 프로세서 및 시리얼 스레드
        self.emg_processor = EMGProcessor()
        self.serial_thread = None
        
        # UI 초기화
        self.init_ui()
        
    def init_ui(self):
        """UI 초기화 - 네오모피즘 디자인 적용"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(40, 40, 40, 40)
        layout.setSpacing(30)
        
        # 배경 설정
        self.setStyleSheet(f"background-color: {BG_COLOR};")
        
        # 제목 카드
        title_card = NeumorphicCard()
        title_layout = QVBoxLayout(title_card)
        title_layout.setContentsMargins(20, 20, 20, 20)
        
        # 제목 텍스트 - 이 부분을 수정하여 제목을 변경할 수 있습니다
        self.title_label = QLabel("사용 가이드")
        self.title_label.setStyleSheet(f"color: {TEXT_COLOR}; font-size: 28px; font-weight: bold;")
        self.title_label.setAlignment(Qt.AlignCenter)
        
        title_layout.addWidget(self.title_label)
        
        # 콘텐츠 카드
        content_card = NeumorphicCard()
        content_layout = QVBoxLayout(content_card)
        content_layout.setContentsMargins(30, 30, 30, 30)
        
        # 콘텐츠 스택 위젯
        self.content_stack = QStackedWidget()
        
        # 페이지 1: 소개
        page1 = QWidget()
        page1_layout = QVBoxLayout(page1)
        
        # 가이드 텍스트 1페이지 - 이 부분을 수정하여 가이드 텍스트를 변경할 수 있습니다
        self.page1_text = QTextEdit()
        self.page1_text.setReadOnly(True)
        self.page1_text.setStyleSheet(f"""
            QTextEdit {{
                background-color: {BG_COLOR};
                border: none;
                color: {TEXT_COLOR};
                font-size: 24px;
                line-height: 1.5;
            }}
        """)
        self.page1_text.setText("""
<h2 style='color: #5e72e4;'>EMG 게임 시스템 사용 안내</h2>
<p>안녕하세요! EMG 센서를 이용한 가위바위보 및 묵찌빠 게임에 오신 것을 환영합니다.</p>
<p>이 게임은 팔에 착용하는 EMG(근전도) 센서를 통해 손동작을 인식하여 게임을 즐길 수 있습니다.</p>
<p>다음 페이지에서 센서 착용 방법과 게임 진행 방법에 대해 알아보겠습니다.</p>
<p>오른쪽 하단의 '다음' 버튼을 클릭하여 계속 진행해주세요.</p>
        """)
        
        page1_layout.addWidget(self.page1_text)
        
        # 페이지 2: 암밴드 착용 상태
        page2 = QWidget()
        page2_layout = QVBoxLayout(page2)
        
        # 가이드 텍스트 2페이지 - 이 부분을 수정하여 가이드 텍스트를 변경할 수 있습니다
        self.page2_text = QTextEdit()
        self.page2_text.setReadOnly(True)
        self.page2_text.setStyleSheet(f"""
            QTextEdit {{
                background-color: {BG_COLOR};
                border: none;
                color: {TEXT_COLOR};
                font-size: 24px;
                line-height: 1.5;
            }}
        """)
        self.page2_text.setText("""
<h2 style='color: #5e72e4;'>EMG 센서 착용 방법</h2>
<p>EMG 센서는 팔의 근육 신호를 감지하는 장치입니다. 올바른 착용 방법은 다음과 같습니다:</p>
<ol>
    <li>팔뚝의 상단, 손목에서 약 5~10cm 떨어진 곳에 센서를 부착합니다.</li>
    <li>센서가 피부와 잘 접촉되었는지 확인합니다.</li>
    <li>물티슈로 피부를 닦아주면 정확도가 상승합니다.</li>
</ol>
<p>아래 착용 상태 표시등을 통해 센서의 부착 상태를 확인할 수 있습니다.</p>
        """)
        
        # 상태 표시 영역 (네오모피즘 스타일)
        status_card = NeumorphicCard(inset=True)
        status_layout = QHBoxLayout(status_card)
        
        # 상태 라벨
        self.connection_status_label = QLabel("암밴드 착용 상태:")
        self.connection_status_label.setStyleSheet(f"color: {TEXT_COLOR}; font-size: 18px; font-weight: bold;")
        
        # 상태 아이콘 프레임
        self.connection_indicator_frame = QFrame()
        self.connection_indicator_frame.setFixedSize(40, 40)
        self.connection_indicator_frame.setStyleSheet(f"""
            background-color: #2DCE89;
            border-radius: 30px;
            margin: 5px;
        """)
        
        # 그림자 효과
        indicator_shadow = QGraphicsDropShadowEffect(self.connection_indicator_frame)
        indicator_shadow.setBlurRadius(10)
        indicator_shadow.setColor(QColor(45, 206, 137, 150))
        indicator_shadow.setOffset(0, 0)
        self.connection_indicator_frame.setGraphicsEffect(indicator_shadow)
        
        # 상태 텍스트
        self.connection_text = QLabel("착용 상태: 좋음")
        self.connection_text.setStyleSheet("color: #2DCE89; font-size: 16px; font-weight: bold;")
        
        status_layout.addWidget(self.connection_status_label)
        status_layout.addWidget(self.connection_indicator_frame)
        status_layout.addWidget(self.connection_text)
        status_layout.addStretch()
        
        page2_layout.addWidget(self.page2_text)
        page2_layout.addWidget(status_card)
        
        # 페이지 3: 동작 인식 테스트
        page3 = QWidget()
        page3_layout = QVBoxLayout(page3)
        
        # 가이드 텍스트 3페이지 - 이 부분을 수정하여 가이드 텍스트를 변경할 수 있습니다
        self.page3_text = QTextEdit()
        self.page3_text.setReadOnly(True)
        self.page3_text.setStyleSheet(f"""
            QTextEdit {{
                background-color: {BG_COLOR};
                border: none;
                color: {TEXT_COLOR};
                font-size: 24px;
                line-height: 1.5;
            }}
        """)
        self.page3_text.setText("""
<h2 style='color: #5e72e4;'>동작 인식 테스트</h2>
<p>이제 간단한 동작 인식 테스트를 진행하겠습니다.</p>
<p>다음 동작을 천천히, 명확하게 수행해보세요:</p>
<p>동작감지에는 1~2초 정도의 딜레이가 있습니다.</p>
<ol>
    <li>'가위' 모양을 만들어보세요.</li>
    <li>'바위' 모양을 만들어보세요.</li>
    <li>'보' 모양을 만들어보세요.</li>
</ol>
<p>각 동작을 취하면 아래 상태가 업데이트됩니다.</p>
        """)
        
        # 감지 상태 카드
        detection_card = NeumorphicCard(inset=True)
        detection_layout = QVBoxLayout(detection_card)
        
        # 감지 상태 레이블 - 이 부분을 수정하여 감지 상태 텍스트를 변경할 수 있습니다
        self.detection_label = QLabel("동작 감지: 대기 중...")
        self.detection_label.setStyleSheet(f"""
            color: {TEXT_COLOR};
            font-size: 30px;
            font-weight: bold;
            padding: 15px;
            text-align: center;
        """)
        self.detection_label.setAlignment(Qt.AlignCenter)
        
        detection_layout.addWidget(self.detection_label)
        
        page3_layout.addWidget(self.page3_text)
        page3_layout.addWidget(detection_card)
        
        # 스택에 페이지 추가
        self.content_stack.addWidget(page1)
        self.content_stack.addWidget(page2)
        self.content_stack.addWidget(page3)
        
        content_layout.addWidget(self.content_stack)
        
        # 네비게이션 버튼
        nav_layout = QHBoxLayout()
        
        # 이전/다음 버튼
        self.prev_btn = NeumorphicButton("이전")
        self.prev_btn.setEnabled(False)
        self.prev_btn.clicked.connect(self.prev_page)
        
        self.next_btn = NeumorphicButton("다음", PRIMARY_COLOR)
        self.next_btn.clicked.connect(self.next_page)
        
        nav_layout.addStretch()
        nav_layout.addWidget(self.prev_btn)
        nav_layout.addWidget(self.next_btn)
        
        # 레이아웃에 카드 추가
        layout.addWidget(title_card)
        layout.addWidget(content_card)
        layout.addLayout(nav_layout)
        
        # 상태 업데이트 타이머
        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self.update_status)
        
    def set_user(self, username, password):
        """사용자 정보 설정 및 모델 확인"""
        self.username = username
        self.password = password
        self.has_model = user_has_model(username, password)
        
    def start(self):
        """가이드 시작 및 시리얼 연결"""
        self.current_page = 0
        self.content_stack.setCurrentIndex(0)
        self.prev_btn.setEnabled(False)
        self.next_btn.setEnabled(True)
        
        # 시리얼 스레드 시작
        if self.serial_thread:
            self.serial_thread.disconnect_serial()
            self.serial_thread = None
            
        self.serial_thread = SerialThread()
        self.serial_thread.data_received.connect(self.process_data)
        self.serial_thread.start()
        
        # 상태 업데이트 타이머 시작
        self.update_timer.start(500)
        
    def stop(self):
        """가이드 중지 및 시리얼 연결 해제"""
        if self.serial_thread:
            self.serial_thread.disconnect_serial()
            self.serial_thread = None
            
        self.update_timer.stop()
        
    def process_data(self, s1, s2):
        """EMG 데이터 처리"""
        result = self.emg_processor.process_data(s1, s2)
        
        if result == "movement_detected":
            # 동작 감지 텍스트 변경 - 이 부분을 수정하여 감지 텍스트를 변경할 수 있습니다
            self.detection_label.setText("동작 감지: 감지됨!")
            self.detection_label.setStyleSheet("font-size: 20px; color: red; font-weight: bold;")
            # 1초 후 원래 상태로 복귀하는 타이머
            QTimer.singleShot(1000, self.reset_detection_label)
            
    def reset_detection_label(self):
        """동작 감지 레이블 리셋"""
        # 동작 감지 기본 텍스트 - 이 부분을 수정하여 기본 감지 텍스트를 변경할 수 있습니다
        self.detection_label.setText("동작 감지: 대기 중...")
        self.detection_label.setStyleSheet(f"font-size: 20px; color: {TEXT_COLOR}; font-weight: bold;")
        
    def update_status(self):
        """연결 상태 및 UI 업데이트"""
        if self.current_page == 1:
            # 암밴드 착용 상태 업데이트
            quality = self.emg_processor.get_connection_quality()
            
            if quality == 2:
                self.connection_indicator_frame.setStyleSheet("background-color: #2DCE89; border-radius: 20px; margin: 5px;")
                self.connection_text.setText("착용 상태: 좋음")
                self.connection_text.setStyleSheet("color: #2DCE89; font-size: 16px; font-weight: bold;")
            elif quality == 1:
                self.connection_indicator_frame.setStyleSheet("background-color: #FB8C00; border-radius: 20px; margin: 5px;")
                self.connection_text.setText("착용 상태: 보통")
                self.connection_text.setStyleSheet("color: #FB8C00; font-size: 16px; font-weight: bold;")
            else:
                self.connection_indicator_frame.setStyleSheet("background-color: #F5365C; border-radius: 20px; margin: 5px;")
                self.connection_text.setText("착용 상태: 나쁨")
                self.connection_text.setStyleSheet("color: #F5365C; font-size: 16px; font-weight: bold;")
                
    def prev_page(self):
        """이전 페이지 이동"""
        if self.current_page > 0:
            self.current_page -= 1
            self.content_stack.setCurrentIndex(self.current_page)
            
            # 버튼 상태 업데이트
            self.prev_btn.setEnabled(self.current_page > 0)
            self.next_btn.setText("다음")
            
    def next_page(self):
        """다음 페이지 이동"""
        if self.current_page < self.total_pages - 1:
            self.current_page += 1
            self.content_stack.setCurrentIndex(self.current_page)
            
            # 버튼 상태 업데이트
            self.prev_btn.setEnabled(True)
            
            # 마지막 페이지일 경우 다음 버튼 텍스트 변경
            if self.current_page == self.total_pages - 1:
                self.next_btn.setText("완료")
        else:
            # 가이드 완료
            self.complete_guide()
            
    def complete_guide(self):
        """가이드 완료 처리"""
        self.stop()
        
        # 모델이 없으면 데이터 수집으로, 있으면 메인 메뉴로
        # 요청에 따라 메인 메뉴로 변경됨 - False로 항상 메인메뉴로 이동
        self.guide_completed.emit(False)


# ------------------------------------------------------------------------
# 동작 데이터 수집 위젯
# ------------------------------------------------------------------------
class DataCollectionWidget(QWidget):
    """
    DataCollectionWidget 클래스: 동작 데이터 수집 화면을 제공하는 위젯
    - 가위/바위/보 동작 데이터를 수집하고 저장
    - 감도 설정 및 데이터 수집 기능 제공
    - 네오모피즘 디자인 적용
    """
    collection_completed = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # 상태 변수
        self.username = ""
        self.password = ""
        self.user_dir = ""
        self.data_file = ""
        self.current_label = None
        self.label_counts = {LABEL_SCISSORS: 0, LABEL_ROCK: 0, LABEL_PAPER: 0}
        self.target_count = 10  # 각 동작 당 목표 수집 횟수
        self.sensitivity = 1.0
        
        # EMG 프로세서 및 시리얼 스레드
        self.emg_processor = EMGProcessor(self.sensitivity)
        self.serial_thread = None
        
        # UI 초기화
        self.init_ui()
        
    def init_ui(self):
        """UI 초기화 - 네오모피즘 디자인 적용"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(40, 40, 40, 40)
        layout.setSpacing(30)
        
        # 배경 설정
        self.setStyleSheet(f"background-color: {BG_COLOR};")
        
        # 제목 카드
        title_card = NeumorphicCard()
        title_layout = QVBoxLayout(title_card)
        title_layout.setContentsMargins(20, 20, 20, 20)
        
        # 제목 텍스트 - 이 부분을 수정하여 제목을 변경할 수 있습니다
        self.title_label = QLabel("동작 데이터 수집")
        self.title_label.setStyleSheet(f"color: {TEXT_COLOR}; font-size: 28px; font-weight: bold;")
        self.title_label.setAlignment(Qt.AlignCenter)
        
        title_layout.addWidget(self.title_label)
        
        # 탭 영역 카드
        tab_card = NeumorphicCard()
        tab_layout = QVBoxLayout(tab_card)
        tab_layout.setContentsMargins(20, 20, 20, 20)
        
        # 탭 위젯 생성
        self.tab_widget = QTabWidget()
        self.tab_widget.setStyleSheet(f"""
            QTabWidget::pane {{
                border: none;
                background-color: {BG_COLOR};
                border-radius: 15px;
            }}
            QTabBar::tab {{
                background-color: #D1D9E6;
                color: {TEXT_COLOR};
                padding: 12px 20px;
                margin-right: 5px;
                border-radius: 8px 8px 0 0;
                font-size: 16px;
            }}
            QTabBar::tab:selected {{
                background-color: {PRIMARY_COLOR};
                color: white;
                font-weight: bold;
            }}
        """)
        
        # 탭 1: 감도 설정
        sensitivity_tab = QWidget()
        sensitivity_layout = QVBoxLayout(sensitivity_tab)
        sensitivity_layout.setContentsMargins(20, 20, 20, 20)
        
        # 감도 설정 설명 텍스트 - 이 부분을 수정하여 설명 텍스트를 변경할 수 있습니다
        self.sensitivity_text = QTextEdit()
        self.sensitivity_text.setReadOnly(True)
        self.sensitivity_text.setStyleSheet(f"""
            QTextEdit {{
                background-color: {BG_COLOR};
                border: none;
                color: {TEXT_COLOR};
                font-size: 24px;
                line-height: 1.5;
            }}
        """)
        self.sensitivity_text.setText("""
<h2 style='color: #5e72e4;'>감도 설정</h2>
<p>먼저 감도를 조절하여 동작을 정확하게 감지할 수 있도록 설정하세요.</p>
<p>감도가 높을수록 작은 움직임도 감지되지만, 오감지가 발생할 수 있습니다.</p>
<p>몇 가지 동작을 취해보며 올바른 감도를 찾아보세요.</p>
        """)
        
        # 감도 조절 영역
        sensitivity_control_card = NeumorphicCard()
        sensitivity_control_layout = QHBoxLayout(sensitivity_control_card)
        
        # 감도 설정 레이블 - 이 부분을 수정하여 레이블 텍스트를 변경할 수 있습니다
        self.sensitivity_label = QLabel("감도 설정:")
        self.sensitivity_label.setStyleSheet(f"color: {TEXT_COLOR}; font-size: 24px; font-weight: bold;")
        
        # 감도 슬라이더 프레임
        slider_frame = NeumorphicCard(inset=True)
        slider_layout = QHBoxLayout(slider_frame)
        slider_layout.setContentsMargins(10, 5, 10, 5)
        
        # 감도 조절 슬라이더
        self.sensitivity_slider = QDoubleSpinBox()
        self.sensitivity_slider.setRange(0.5, 2.0)
        self.sensitivity_slider.setSingleStep(0.1)
        self.sensitivity_slider.setValue(1.0)
        self.sensitivity_slider.setStyleSheet(f"""
            QDoubleSpinBox {{
                background-color: transparent;
                border: none;
                color: {TEXT_COLOR};
                font-size: 16px;
            }}
            QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{
                background-color: #D1D9E6;
                border-radius: 4px;
            }}
        """)
        self.sensitivity_slider.valueChanged.connect(self.update_sensitivity)
        
        slider_layout.addWidget(self.sensitivity_slider)
        
        sensitivity_control_layout.addWidget(self.sensitivity_label)
        sensitivity_control_layout.addWidget(slider_frame)
        sensitivity_control_layout.addStretch()
        
        # 감지 상태 카드
        detection_card = NeumorphicCard(inset=True)
        detection_layout = QVBoxLayout(detection_card)
        
        # 감지 상태 레이블 - 이 부분을 수정하여 감지 상태 텍스트를 변경할 수 있습니다
        self.detection_label = QLabel("동작 감지: 대기 중...")
        self.detection_label.setStyleSheet(f"""
            color: {TEXT_COLOR};
            font-size: 24px;
            font-weight: bold;
            padding: 15px;
            text-align: center;
        """)
        self.detection_label.setAlignment(Qt.AlignCenter)
        
        detection_layout.addWidget(self.detection_label)
        
        sensitivity_layout.addWidget(self.sensitivity_text)
        sensitivity_layout.addWidget(sensitivity_control_card)
        sensitivity_layout.addWidget(detection_card)
        
        # 탭 2: 데이터 수집
        collection_tab = QWidget()
        collection_layout = QVBoxLayout(collection_tab)
        collection_layout.setContentsMargins(20, 20, 20, 20)
        
        # 데이터 수집 설명 텍스트 - 이 부분을 수정하여 설명 텍스트를 변경할 수 있습니다
        self.collection_text = QTextEdit()
        self.collection_text.setReadOnly(True)
        self.collection_text.setStyleSheet(f"""
            QTextEdit {{
                background-color: {BG_COLOR};
                border: none;
                color: {TEXT_COLOR};
                font-size: 24px;
                line-height: 1.5;
            }}
        """)
        self.collection_text.setText("""
<h2 style='color: #5e72e4;'>데이터 수집</h2>
<p>각 동작을 10번씩 수집합니다. 지시에 따라 동작을 취해주세요.</p>
<p>잘못 입력한 경우 '마지막 데이터 삭제' 버튼을 눌러 삭제할 수 있습니다.</p>
<p>각 동작별로 5회 이상 수집했다면 '다음 동작' 버튼으로 넘어갈 수 있습니다.</p>
        """)
        
        # 현재 동작 및 진행 상황 카드
        status_card = NeumorphicCard()
        status_layout = QVBoxLayout(status_card)
        status_layout.setContentsMargins(25, 25, 25, 25)
        
        # 현재 동작 표시 - 이 부분을 수정하여 현재 동작 텍스트를 변경할 수 있습니다
        self.current_action_label = QLabel("현재 동작: 없음")
        self.current_action_label.setStyleSheet(f"""
            color: {PRIMARY_COLOR};
            font-size: 24px;
            font-weight: bold;
            margin-bottom: 15px;
        """)
        self.current_action_label.setAlignment(Qt.AlignCenter)
        
        # 동작별 진행 상태 그리드
        progress_grid = QGridLayout()
        progress_grid.setSpacing(20)
        
        # 가위 진행 상태
        scissors_card = NeumorphicCard()
        scissors_card.setStyleSheet(f"""
            background-color: #EEF2FF;
            border-radius: 15px;
        """)
        scissors_layout = QVBoxLayout(scissors_card)
        
        # 가위 상태 레이블 - 이 부분을 수정하여 레이블 텍스트를 변경할 수 있습니다
        scissors_label = QLabel("가위 ✌️")
        scissors_label.setStyleSheet(f"color: {PRIMARY_COLOR}; font-size: 24px; font-weight: bold;")
        scissors_label.setAlignment(Qt.AlignCenter)
        
        # 프로그레스 바 컨테이너
        scissors_progress_frame = NeumorphicCard(inset=True)
        scissors_progress_layout = QVBoxLayout(scissors_progress_frame)
        scissors_progress_layout.setContentsMargins(10, 5, 10, 5)
        
        # 가위 진행 상태 표시
        self.scissors_count_label = QLabel("0/10")
        self.scissors_count_label.setStyleSheet(f"color: {TEXT_COLOR}; font-size: 18px; font-weight: bold;")
        self.scissors_count_label.setAlignment(Qt.AlignCenter)
        
        scissors_progress_layout.addWidget(self.scissors_count_label)
        
        scissors_layout.addWidget(scissors_label)
        scissors_layout.addWidget(scissors_progress_frame)
        
        # 바위 진행 상태
        rock_card = NeumorphicCard()
        rock_card.setStyleSheet(f"""
            background-color: #FFFBEB;
            border-radius: 15px;
        """)
        rock_layout = QVBoxLayout(rock_card)
        
        # 바위 상태 레이블 - 이 부분을 수정하여 레이블 텍스트를 변경할 수 있습니다
        rock_label = QLabel("바위 ✊")
        rock_label.setStyleSheet("color: #FB8C00; font-size: 24px; font-weight: bold;")
        rock_label.setAlignment(Qt.AlignCenter)
        
        # 프로그레스 바 컨테이너
        rock_progress_frame = NeumorphicCard(inset=True)
        rock_progress_layout = QVBoxLayout(rock_progress_frame)
        rock_progress_layout.setContentsMargins(10, 5, 10, 5)
        
        # 바위 진행 상태 표시
        self.rock_count_label = QLabel("0/10")
        self.rock_count_label.setStyleSheet(f"color: {TEXT_COLOR}; font-size: 18px; font-weight: bold;")
        self.rock_count_label.setAlignment(Qt.AlignCenter)
        
        rock_progress_layout.addWidget(self.rock_count_label)
        
        rock_layout.addWidget(rock_label)
        rock_layout.addWidget(rock_progress_frame)
        
        # 보 진행 상태
        paper_card = NeumorphicCard()
        paper_card.setStyleSheet(f"""
            background-color: #E8F5E9;
            border-radius: 15px;
        """)
        paper_layout = QVBoxLayout(paper_card)
        
        # 보 상태 레이블 - 이 부분을 수정하여 레이블 텍스트를 변경할 수 있습니다
        paper_label = QLabel("보 ✋")
        paper_label.setStyleSheet("color: #2DCE89; font-size: 24px; font-weight: bold;")
        paper_label.setAlignment(Qt.AlignCenter)
        
        # 프로그레스 바 컨테이너
        paper_progress_frame = NeumorphicCard(inset=True)
        paper_progress_layout = QVBoxLayout(paper_progress_frame)
        paper_progress_layout.setContentsMargins(10, 5, 10, 5)
        
        # 보 진행 상태 표시
        self.paper_count_label = QLabel("0/10")
        self.paper_count_label.setStyleSheet(f"color: {TEXT_COLOR}; font-size: 18px; font-weight: bold;")
        self.paper_count_label.setAlignment(Qt.AlignCenter)
        
        paper_progress_layout.addWidget(self.paper_count_label)
        
        paper_layout.addWidget(paper_label)
        paper_layout.addWidget(paper_progress_frame)
        
        # 그리드에 각 동작 카드 추가
        progress_grid.addWidget(scissors_card, 0, 0)
        progress_grid.addWidget(rock_card, 0, 1)
        progress_grid.addWidget(paper_card, 0, 2)
        
        # 버튼 레이아웃
        buttons_layout = QHBoxLayout()
        
        # 컨트롤 버튼 - 이 부분을 수정하여 버튼 텍스트를 변경할 수 있습니다
        self.delete_btn = NeumorphicButton("마지막 데이터 삭제", DANGER_COLOR)
        self.delete_btn.clicked.connect(self.delete_last_data)
        self.delete_btn.setEnabled(False)
        
        self.next_action_btn = NeumorphicButton("다음 동작", PRIMARY_COLOR)
        self.next_action_btn.clicked.connect(self.next_action)
        self.next_action_btn.setEnabled(False)
        
        buttons_layout.addWidget(self.delete_btn)
        buttons_layout.addStretch()
        buttons_layout.addWidget(self.next_action_btn)
        
        # 상태 카드에 위젯 추가
        status_layout.addWidget(self.current_action_label)
        status_layout.addLayout(progress_grid)
        status_layout.addSpacing(10)
        status_layout.addLayout(buttons_layout)
        
        # 데이터 수집 탭에 위젯 추가
        collection_layout.addWidget(self.collection_text)
        collection_layout.addWidget(status_card)
        
        # 탭 추가
        self.tab_widget.addTab(sensitivity_tab, "1. 감도 설정")
        self.tab_widget.addTab(collection_tab, "2. 데이터 수집")
        self.tab_widget.currentChanged.connect(self.on_tab_changed)
        
        tab_layout.addWidget(self.tab_widget)
        
        # 로그 카드
        log_card = NeumorphicCard(inset=True)
        log_layout = QVBoxLayout(log_card)
        
        # 로그 영역 레이블
        log_header = QLabel("로그")
        log_header.setStyleSheet(f"font-size: 16px; font-weight: bold; color: {TEXT_COLOR};")
        
        # 로그 텍스트 영역
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(150)
        self.log_text.setStyleSheet(f"""
            QTextEdit {{
                background-color: transparent;
                border: none;
                color: {TEXT_COLOR};
                font-family: 'Consolas', monospace;
                font-size: 14px;
                line-height: 1.3;
            }}
        """)
        
        log_layout.addWidget(log_header)
        log_layout.addWidget(self.log_text)
        
        # 네비게이션 버튼
        nav_layout = QHBoxLayout()
        
        # 뒤로/완료 버튼 - 이 부분을 수정하여 버튼 텍스트를 변경할 수 있습니다
        self.back_btn = NeumorphicButton("뒤로")
        self.back_btn.clicked.connect(self.go_back)
        
        self.complete_btn = NeumorphicButton("수집 완료", PRIMARY_COLOR)
        self.complete_btn.clicked.connect(self.complete_collection)
        self.complete_btn.setEnabled(False)
        
        nav_layout.addWidget(self.back_btn)
        nav_layout.addStretch()
        nav_layout.addWidget(self.complete_btn)
        
        # 레이아웃에 위젯 추가
        layout.addWidget(title_card)
        layout.addWidget(tab_card)
        layout.addWidget(log_card)
        layout.addLayout(nav_layout)
        
    def set_user(self, username, password):
        """사용자 정보 설정"""
        self.username = username
        self.password = password
        self.user_dir = get_user_dir(username, password)
        
        # 사용자 설정 로드
        config = load_user_config(username, password)
        self.sensitivity = config["sensitivity"]
        self.sensitivity_slider.setValue(self.sensitivity)
        self.emg_processor.set_sensitivity(self.sensitivity)
        
        # 데이터 카운트 업데이트
        for label_str, count in config["data_count"].items():
            label = int(label_str)
            self.label_counts[label] = count
            
        self.update_count_labels()
        
    def start(self):
        """데이터 수집 시작"""
        # 탭 초기화
        self.tab_widget.setCurrentIndex(0)
        
        # 시리얼 스레드 시작
        if self.serial_thread:
            self.serial_thread.disconnect_serial()
            self.serial_thread = None
            
        self.serial_thread = SerialThread()
        self.serial_thread.data_received.connect(self.process_data)
        self.serial_thread.start()
        
        # 데이터 파일 생성
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.data_file = os.path.join(self.user_dir, f"emg_data_{timestamp}.csv")
        
        # CSV 파일 생성 및 헤더 작성
        with open(self.data_file, 'w', newline='') as f:
            writer = csv.writer(f)
            headers = ['Label', 'Actual_Length']
            for i in range(SEQUENCE_LENGTH):
                headers.extend([f'S1_T{i}', f'S2_T{i}'])
            writer.writerow(headers)
            
        self.add_log(f"데이터 파일 생성: {self.data_file}")
        
    def update_sensitivity(self, value):
        """감도 설정 업데이트"""
        self.sensitivity = value
        self.emg_processor.set_sensitivity(value)
        self.add_log(f"감도 설정: {value:.1f}")
        
        # 사용자 설정 저장
        config = load_user_config(self.username, self.password)
        config["sensitivity"] = value
        save_user_config(self.username, self.password, config)
        
    def process_data(self, s1, s2):
        """EMG 데이터 처리"""
        result = self.emg_processor.process_data(s1, s2)
        
        if isinstance(result, tuple) and result[0] == "movement_completed":
            # 동작 감지 완료 - 현재 라벨이 설정된 경우만 저장
            if self.current_label is not None:
                self.save_sequence(result[1], self.current_label)
                
            self.detection_label.setText("동작 감지: 완료!")
            self.detection_label.setStyleSheet(f"font-size: 24px; color: {SECONDARY_COLOR}; font-weight: bold;")
            QTimer.singleShot(1000, self.reset_detection_label)
            
        elif result == "movement_detected":
            self.detection_label.setText("동작 감지: 감지됨!")
            self.detection_label.setStyleSheet(f"font-size: 24px; color: {PRIMARY_COLOR}; font-weight: bold;")
            
    def reset_detection_label(self):
        """동작 감지 레이블 리셋"""
        self.detection_label.setText("동작 감지: 대기 중...")
        self.detection_label.setStyleSheet(f"font-size: 24px; color: {TEXT_COLOR}; font-weight: bold;")
        
    def save_sequence(self, sequence, label):
        """EMG 시퀀스 저장"""
        if len(sequence) < 20:
            self.add_log("시퀀스가 너무 짧습니다. 무시합니다.")
            return
            
        # 시퀀스 길이 조정
        actual_length = len(sequence)
        if actual_length < SEQUENCE_LENGTH:
            padding = [(0, 0)] * (SEQUENCE_LENGTH - actual_length)
            sequence = sequence + padding
        elif actual_length > SEQUENCE_LENGTH:
            sequence = sequence[:SEQUENCE_LENGTH]
            actual_length = SEQUENCE_LENGTH
            
        # CSV에 저장
        with open(self.data_file, 'a', newline='') as f:
            writer = csv.writer(f)
            row = [label, actual_length]
            # 모든 센서 값 추가
            for s1, s2 in sequence:
                row.extend([s1, s2])
            writer.writerow(row)
            
        # 라벨 카운트 업데이트
        self.label_counts[label] += 1
        self.update_count_labels()
        
        # 설정 파일에 저장
        config = load_user_config(self.username, self.password)
        config["data_count"] = {str(k): v for k, v in self.label_counts.items()}
        save_user_config(self.username, self.password, config)
        
        # 버튼 상태 업데이트
        self.delete_btn.setEnabled(True)
        if self.label_counts[self.current_label] >= 5:
            self.next_action_btn.setEnabled(True)
            
        # 목표 카운트 달성 확인
        if self.label_counts[self.current_label] >= self.target_count:
            self.next_action()
            
        self.add_log(f"'{get_label_name(label)}' 동작 데이터 저장 완료 ({self.label_counts[label]}/{self.target_count})")
        
    def delete_last_data(self):
        """마지막 데이터 삭제"""
        try:
            # 파일에서 마지막 줄 제거
            with open(self.data_file, 'r', newline='') as f:
                lines = f.readlines()
                
            if len(lines) <= 1:
                self.add_log("삭제할 데이터가 없습니다.")
                return
                
            # 마지막 줄에서 라벨 추출
            last_line = lines[-1].strip()
            if last_line:
                last_label = int(last_line.split(',')[0])
                if last_label in self.label_counts:
                    self.label_counts[last_label] = max(0, self.label_counts[last_label] - 1)
                    
            # 파일 다시 쓰기
            with open(self.data_file, 'w', newline='') as f:
                f.writelines(lines[:-1])
                
            # 카운트 업데이트
            self.update_count_labels()
            
            # 설정 파일에 저장
            config = load_user_config(self.username, self.password)
            config["data_count"] = {str(k): v for k, v in self.label_counts.items()}
            save_user_config(self.username, self.password, config)
            
            # 버튼 상태 업데이트
            if self.current_label and self.label_counts[self.current_label] < 5:
                self.next_action_btn.setEnabled(False)
                
            self.add_log("마지막 데이터가 삭제되었습니다.")
            
        except Exception as e:
            self.add_log(f"데이터 삭제 오류: {str(e)}")
            
    def next_action(self):
        """다음 동작으로 이동"""
        # 현재 라벨에 따라 다음 라벨 설정
        if self.current_label is None:
            self.current_label = LABEL_SCISSORS
        elif self.current_label == LABEL_SCISSORS and self.label_counts[LABEL_SCISSORS] >= 5:
            self.current_label = LABEL_ROCK
        elif self.current_label == LABEL_ROCK and self.label_counts[LABEL_ROCK] >= 5:
            self.current_label = LABEL_PAPER
        elif self.current_label == LABEL_PAPER and self.label_counts[LABEL_PAPER] >= 5:
            # 모든 동작 수집 완료
            self.current_label = None
            self.current_action_label.setText("모든 동작 수집 완료!")
            self.next_action_btn.setEnabled(False)
            self.complete_btn.setEnabled(True)
            self.add_log("모든 동작 데이터 수집이 완료되었습니다. '수집 완료' 버튼을 클릭하세요.")
            return
            
        # 라벨 텍스트 업데이트
        if self.current_label:
            # 현재 동작 레이블 텍스트 변경 - 이 부분을 수정하여 텍스트를 변경할 수 있습니다
            self.current_action_label.setText(f"현재 동작: {get_label_name(self.current_label)}")
            self.next_action_btn.setEnabled(self.label_counts[self.current_label] >= 5)
            self.add_log(f"다음 동작: {get_label_name(self.current_label)}")
            
    def update_count_labels(self):
        """카운트 레이블 업데이트"""
        # 각 동작 카운트 텍스트 - 이 부분을 수정하여 카운트 텍스트를 변경할 수 있습니다
        self.scissors_count_label.setText(f"가위: {self.label_counts[LABEL_SCISSORS]}/{self.target_count}")
        self.rock_count_label.setText(f"바위: {self.label_counts[LABEL_ROCK]}/{self.target_count}")
        self.paper_count_label.setText(f"보: {self.label_counts[LABEL_PAPER]}/{self.target_count}")
        
        # 모든 동작이 최소 요구사항을 충족하는지 확인
        all_min_collected = all(count >= 5 for count in self.label_counts.values())
        self.complete_btn.setEnabled(all_min_collected)
        
    def on_tab_changed(self, index):
        """탭 변경 이벤트 처리"""
        if index == 1:  # 데이터 수집 탭으로 이동
            if self.current_label is None:
                self.next_action()
                
    def add_log(self, message):
        """로그 메시지 추가"""
        timestamp = datetime.now().strftime('%H:%M:%S')
        self.log_text.append(f"[{timestamp}] {message}")
        # 스크롤을 항상 최하단으로
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
        
    def go_back(self):
        """뒤로 가기"""
        # 사용자 설정 저장
        config = load_user_config(self.username, self.password)
        config["sensitivity"] = self.sensitivity
        config["data_count"] = {str(k): v for k, v in self.label_counts.items()}
        save_user_config(self.username, self.password, config)
        
        if self.serial_thread:
            self.serial_thread.disconnect_serial()
            self.serial_thread = None
            
        self.collection_completed.emit()
        
    def complete_collection(self):
        """데이터 수집 완료"""
        self.add_log("데이터 수집 완료. 모델 학습으로 이동합니다.")
        
        if self.serial_thread:
            self.serial_thread.disconnect_serial()
            self.serial_thread = None
            
        self.collection_completed.emit()


# ------------------------------------------------------------------------
# 모델 학습 위젯
# ------------------------------------------------------------------------
class ModelTrainingWidget(QWidget):
    """
    ModelTrainingWidget 클래스: 모델 학습 화면을 제공하는 위젯
    - 수집된 데이터를 기반으로 EMG 동작 인식 모델 학습
    - 학습 진행 상황 및 결과를 표시
    - 네오모피즘 디자인 적용
    """
    training_completed = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # 상태 변수
        self.username = ""
        self.password = ""
        self.user_dir = ""
        self.is_training = False
        self.best_accuracy = 0.0
        self.best_model_params = HIGH_PERFORMANCE_MODEL.copy()
        
        # UI 초기화
        self.init_ui()
        
    def init_ui(self):
        """UI 초기화 - 네오모피즘 디자인 적용"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(40, 40, 40, 40)
        layout.setSpacing(30)
        
        # 배경 설정
        self.setStyleSheet(f"background-color: {BG_COLOR};")
        
        # 제목 카드
        title_card = NeumorphicCard()
        title_layout = QVBoxLayout(title_card)
        title_layout.setContentsMargins(20, 20, 20, 20)
        
        # 제목 텍스트 - 이 부분을 수정하여 제목을, 변경할 수 있습니다
        self.title_label = QLabel("모델 학습")
        self.title_label.setStyleSheet(f"color: {TEXT_COLOR}; font-size: 32px; font-weight: bold;")
        self.title_label.setAlignment(Qt.AlignCenter)
        
        title_layout.addWidget(self.title_label)
        
        # 정보 카드
        info_card = NeumorphicCard()
        info_layout = QVBoxLayout(info_card)
        info_layout.setContentsMargins(25, 25, 25, 25)
        
        # 정보 텍스트 - 이 부분을 수정하여 정보 텍스트를 변경할 수 있습니다
        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        self.info_text.setStyleSheet(f"""
            QTextEdit {{
                background-color: {BG_COLOR};
                border: none;
                color: {TEXT_COLOR};
                font-size: 24px;
                line-height: 1.5;
            }}
        """)
        self.info_text.setText("""
<h2 style='color: #5e72e4;'>모델 학습</h2>
<p>수집된 데이터를 기반으로 EMG 동작 인식 모델을 학습합니다.</p>
<p>학습은 자동으로 진행되며, 최적의 모델을 찾기 위해 여러 번 시도할 수 있습니다.</p>
<p>학습이 완료되면 자동으로 다음 단계로 넘어갑니다.</p>
        """)
        
        info_layout.addWidget(self.info_text)
        
        # 진행 상황 카드
        progress_card = NeumorphicCard()
        progress_layout = QVBoxLayout(progress_card)
        progress_layout.setContentsMargins(25, 25, 25, 25)
        
        # 진행 상태 레이블 - 이 부분을 수정하여 진행 상태 텍스트를 변경할 수 있습니다
        self.progress_label = QLabel("준비 중...")
        self.progress_label.setStyleSheet(f"color: {PRIMARY_COLOR}; font-size: 24px; font-weight: bold;")
        self.progress_label.setAlignment(Qt.AlignCenter)
        
        # 프로그레스 바 컨테이너
        progress_bar_frame = NeumorphicCard(inset=True)
        progress_bar_layout = QVBoxLayout(progress_bar_frame)
        progress_bar_layout.setContentsMargins(10, 5, 10, 5)
        
        # 진행 상태 프로그레스 바
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFixedHeight(20)
        self.progress_bar.setStyleSheet(f"""
            QProgressBar {{
                background-color: transparent;
                border: none;
                text-align: center;
                font-size: 18px;
                font-weight: bold;
            }}
            QProgressBar::chunk {{
                background-color: {PRIMARY_COLOR};
                border-radius: 5px;
            }}
        """)
        
        progress_bar_layout.addWidget(self.progress_bar)
        
        progress_layout.addWidget(self.progress_label)
        progress_layout.addWidget(progress_bar_frame)
        
        # 로그 카드
        log_card = NeumorphicCard(inset=True)
        log_layout = QVBoxLayout(log_card)
        
        # 로그 헤더 - 이 부분을 수정하여 로그 헤더 텍스트를 변경할 수 있습니다
        log_header = QLabel("학습 로그")
        log_header.setStyleSheet(f"font-size: 16px; font-weight: bold; color: {TEXT_COLOR};")
        
        # 로그 텍스트 영역
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet(f"""
            QTextEdit {{
                background-color: transparent;
                border: none;
                color: {TEXT_COLOR};
                font-family: 'Consolas', monospace;
                font-size: 14px;
                line-height: 1.3;
            }}
        """)
        
        log_layout.addWidget(log_header)
        log_layout.addWidget(self.log_text)
        
        # 버튼 레이아웃
        button_layout = QHBoxLayout()
        
        # 버튼 - 이 부분을 수정하여 버튼 텍스트를 변경할 수 있습니다
        self.back_btn = NeumorphicButton("뒤로")
        self.back_btn.clicked.connect(self.go_back)
        
        self.train_btn = NeumorphicButton("학습 시작", PRIMARY_COLOR)
        self.train_btn.clicked.connect(self.start_training)
        
        button_layout.addWidget(self.back_btn)
        button_layout.addStretch()
        button_layout.addWidget(self.train_btn)
        
        # 레이아웃에 카드 추가
        layout.addWidget(title_card)
        layout.addWidget(info_card)
        layout.addWidget(progress_card)
        layout.addWidget(log_card)
        layout.addLayout(button_layout)
        
    def set_user(self, username, password):
        """사용자 정보 설정"""
        self.username = username
        self.password = password
        self.user_dir = get_user_dir(username, password)
        
    def start(self):
        """위젯 시작"""
        self.reset_ui()
        self.check_data()
        
    def reset_ui(self):
        """UI 초기화"""
        self.progress_label.setText("준비 중...")
        self.progress_bar.setValue(0)
        self.log_text.clear()
        self.train_btn.setEnabled(True)
        self.back_btn.setEnabled(True)
        
    def check_data(self):
        """데이터 파일 확인"""
        data_files = [f for f in os.listdir(self.user_dir) if f.endswith('.csv')]
        
        if not data_files:
            self.add_log("데이터 파일이 없습니다. 먼저 데이터를 수집해주세요.")
            self.train_btn.setEnabled(False)
            return False
            
        self.add_log(f"{len(data_files)}개의 데이터 파일을 찾았습니다.")
        
        # 데이터 분포 확인
        config = load_user_config(self.username, self.password)
        data_count = config.get("data_count", {})
        
        self.add_log("데이터 분포:")
        for label, count in data_count.items():
            self.add_log(f"  - {get_label_name(int(label))}: {count}개")
            
        return True
        
    def start_training(self):
        """모델 학습 시작"""
        if self.is_training:
            return
            
        # UI 업데이트
        self.is_training = True
        self.train_btn.setEnabled(False)
        self.back_btn.setEnabled(False)
        self.progress_label.setText("데이터 준비 중...")
        self.progress_bar.setValue(5)
        
        # 학습 스레드 시작
        self.training_thread = TrainingThread(self.username, self.password)
        self.training_thread.progress_update.connect(self.update_progress)
        self.training_thread.training_complete.connect(self.training_complete)
        self.training_thread.start()
        
    def update_progress(self, progress, message):
        """학습 진행 상황 업데이트"""
        self.progress_bar.setValue(progress)
        if message:
            self.add_log(message)
            self.progress_label.setText(message)
            
    def training_complete(self, success, model_path, accuracy):
        """학습 완료 처리"""
        self.is_training = False
        
        if success:
            self.add_log(f"학습 완료! 모델 저장 경로: {model_path}")
            self.add_log(f"모델 정확도: {accuracy:.2f}")
            
            # 사용자 설정 업데이트
            config = load_user_config(self.username, self.password)
            config["model_accuracy"] = accuracy
            save_user_config(self.username, self.password, config)
            
            # 학습 결과에 따른 다음 단계
            if accuracy >= 0.9:
                self.add_log("학습 성공! 모델 정확도가 90% 이상입니다.")
                self.progress_label.setText("학습 성공!")
                
                # 잠시 후 메인 화면으로 자동 이동
                QTimer.singleShot(3000, self.training_completed.emit)
            else:
                self.add_log("모델 정확도가 90% 미만입니다. 다른 설정으로 다시 시도합니다.")
                self.progress_label.setText("다른 설정으로 재시도 중...")
                self.progress_bar.setValue(0)
                
                # 다른 설정으로 재시도
                self.try_different_settings()
        else:
            self.add_log("학습 실패! 뒤로 버튼을 눌러 데이터 수집으로 돌아가세요.")
            self.progress_label.setText("학습 실패!")
            self.back_btn.setEnabled(True)
            
    def try_different_settings(self):
        """다른 설정으로 학습 시도"""
        # 모델 설정 변경
        params = self.generate_new_params()
        
        self.add_log("새로운 설정으로 학습 시도")
        
        # 학습 재시작
        self.training_thread = TrainingThread(self.username, self.password, params)
        self.training_thread.progress_update.connect(self.update_progress)
        self.training_thread.training_complete.connect(self.training_complete)
        self.training_thread.start()
        
    def generate_new_params(self):
        """새로운 모델 파라미터 생성"""
        # 간단한 변형 - 실제 애플리케이션에서는 더 체계적인 방법 사용 권장
        params = HIGH_PERFORMANCE_MODEL.copy()
        
        # 랜덤한 변화 추가
        params['d_model'] = random.choice([128, 192, 256])
        params['num_layers'] = random.choice([2, 3, 4])
        params['learning_rate'] = random.choice([0.0001, 0.0003, 0.0005])
        
        return params
        
    def add_log(self, message):
        """로그 메시지 추가"""
        timestamp = datetime.now().strftime('%H:%M:%S')
        self.log_text.append(f"[{timestamp}] {message}")
        # 스크롤을 항상 최하단으로
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
        
    def go_back(self):
        """뒤로 가기"""
        if self.is_training:
            # 학습 중단 확인
            reply = QMessageBox.question(
                self, 
                '학습 중단', 
                "학습을 중단하시겠습니까?",
                QMessageBox.Yes | QMessageBox.No, 
                QMessageBox.No
            )
            
            if reply == QMessageBox.No:
                return
                
            # 학습 스레드 중지
            if hasattr(self, 'training_thread') and self.training_thread.isRunning():
                self.training_thread.terminate()
                self.training_thread.wait()
                
        self.training_completed.emit()


# ------------------------------------------------------------------------
# 모델 학습 스레드
# ------------------------------------------------------------------------
class TrainingThread(QThread):
    """
    TrainingThread 클래스: 모델 학습을 담당하는 스레드
    - 백그라운드에서 모델 학습 수행
    - 학습 진행 상황과 결과를 위젯에 전달
    """
    progress_update = pyqtSignal(int, str)
    training_complete = pyqtSignal(bool, str, float)  # success, model_path, accuracy
    
    def __init__(self, username, password, model_params=None):
        super().__init__()
        self.username = username
        self.password = password
        self.user_dir = get_user_dir(username, password)
        self.model_params = model_params or HIGH_PERFORMANCE_MODEL.copy()
        
    def run(self):
        try:
            # 데이터 로드
            self.progress_update.emit(5, "데이터 로드 중...")
            X, y = self.load_data()
            
            if X is None or len(X) == 0:
                self.progress_update.emit(0, "데이터를 찾을 수 없습니다.")
                self.training_complete.emit(False, "", 0.0)
                return
                
            # 데이터 분포 확인
            label_counts = {}
            for label in np.unique(y):
                count = np.sum(y == label)
                label_counts[int(label)] = int(count)
                
            self.progress_update.emit(10, f"데이터 분포: {label_counts}")
            
            # 데이터 전처리
            self.progress_update.emit(20, "데이터 전처리 중...")
            X_train, X_val, y_train, y_val, scaler = self.preprocess_data(X, y)
            
            # 모델 학습
            self.progress_update.emit(30, "모델 학습 시작...")
            model, accuracy = self.train_model(X_train, y_train, X_val, y_val)
            
            # 모델 저장
            self.progress_update.emit(90, "모델 저장 중...")
            model_path = self.save_model(model, scaler, accuracy)
            
            # 완료
            self.progress_update.emit(100, f"학습 완료! 정확도: {accuracy:.2f}")
            self.training_complete.emit(True, model_path, accuracy)
            
        except Exception as e:
            self.progress_update.emit(0, f"학습 오류: {str(e)}")
            self.training_complete.emit(False, "", 0.0)

    def load_data(self):
        """CSV 데이터 로드"""
        data_files = [f for f in os.listdir(self.user_dir) if f.endswith('.csv')]
        
        if not data_files:
            return None, None
            
        all_X = []
        all_y = []
        
        for file in data_files:
            file_path = os.path.join(self.user_dir, file)
            try:
                # CSV 파일 읽기
                with open(file_path, 'r') as f:
                    lines = f.readlines()[1:]  # 헤더 제외
                    
                for line in lines:
                    values = line.strip().split(',')
                    label = int(values[0])
                    actual_length = int(values[1])
                    
                    # EMG 데이터 추출
                    emg_data = []
                    for i in range(SEQUENCE_LENGTH):
                        s1_idx = 2 + i * 2
                        s2_idx = 3 + i * 2
                        if s1_idx < len(values) and s2_idx < len(values):
                            emg_data.append([float(values[s1_idx]), float(values[s2_idx])])
                        else:
                            emg_data.append([0.0, 0.0])  # 패딩
                    
                    all_X.append(emg_data)
                    all_y.append(label)
                    
            except Exception as e:
                self.progress_update.emit(0, f"파일 '{file}' 로드 오류: {str(e)}")
                
        if all_X and all_y:
            return np.array(all_X), np.array(all_y)
        else:
            return None, None
            
    def preprocess_data(self, X, y):
        """데이터 전처리"""
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import train_test_split
        
        # 데이터 증강
        X_aug = []
        y_aug = []
        
        # 원본 데이터
        X_aug.append(X)
        y_aug.append(y)
        
        # 노이즈 추가
        noise_level = 0.03
        X_noisy = X + np.random.normal(0, noise_level, X.shape)
        X_aug.append(X_noisy)
        y_aug.append(y)
        
        # 증강 데이터 결합
        X = np.vstack(X_aug)
        y = np.concatenate(y_aug)
        
        # 훈련/검증 데이터 분할
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # 정규화
        scaler = StandardScaler()
        X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
        scaler.fit(X_train_reshaped)
        
        # 정규화 적용
        X_train_shape = X_train.shape
        X_val_shape = X_val.shape
        
        X_train_normalized = scaler.transform(X_train.reshape(-1, X_train.shape[-1]))
        X_val_normalized = scaler.transform(X_val.reshape(-1, X_val.shape[-1]))
        
        X_train = X_train_normalized.reshape(X_train_shape)
        X_val = X_val_normalized.reshape(X_val_shape)
        
        return X_train, X_val, y_train, y_val, scaler
        
    def train_model(self, X_train, y_train, X_val, y_val):
        """모델 학습"""
        # 학습 파라미터
        d_model = self.model_params['d_model']
        nhead = self.model_params['nhead']
        num_layers = self.model_params['num_layers']
        dim_feedforward = self.model_params['dim_feedforward']
        learning_rate = self.model_params['learning_rate']
        batch_size = self.model_params['batch_size']
        num_epochs = self.model_params['num_epochs']
        
        # 모델 생성
        model = EMGTransformer(
            input_dim=2,  # EMG 센서 2개
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            num_classes=4,  # 배경(0) + 가위(1), 바위(2), 보(3)
            dropout=0.1
        )
        
        # 손실 함수 및 옵티마이저
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        # 학습률 스케줄러
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        # 데이터셋 생성
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long)
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val, dtype=torch.long)
        
        # 학습 루프
        best_accuracy = 0.0
        patience_counter = 0
        patience = 10
        
        for epoch in range(num_epochs):
            # 배치 처리
            num_batches = len(X_train) // batch_size + (1 if len(X_train) % batch_size != 0 else 0)
            total_loss = 0.0
            correct = 0
            total = 0
            
            model.train()
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min(start_idx + batch_size, len(X_train))
                
                batch_X = X_train_tensor[start_idx:end_idx]
                batch_y = y_train_tensor[start_idx:end_idx]
                
                # 순전파
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                
                # 역전파
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item() * len(batch_X)
                
                # 정확도 계산
                _, predicted = torch.max(outputs, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
                
            # 평균 손실 및 정확도
            avg_loss = total_loss / total
            train_accuracy = correct / total
            
            # 검증
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                outputs = model(X_val_tensor)
                val_loss = criterion(outputs, y_val_tensor).item()
                _, predicted = torch.max(outputs, 1)
                val_total = y_val_tensor.size(0)
                val_correct = (predicted == y_val_tensor).sum().item()
                
            val_accuracy = val_correct / val_total
            
            # 학습률 스케줄러 업데이트
            scheduler.step(val_loss)
            
            # 진행 상황 업데이트
            progress = 30 + (60 * (epoch + 1) // num_epochs)
            message = f"에폭 {epoch+1}/{num_epochs} - 검증 정확도: {val_accuracy:.4f}"
            self.progress_update.emit(progress, message)
            
            # 최고 성능 모델 저장
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                patience_counter = 0
            else:
                patience_counter += 1
                
            # 조기 종료
            if patience_counter >= patience:
                self.progress_update.emit(progress, f"{message} - 에폭 {epoch+1}에서 조기 종료")
                break
                
        return model, best_accuracy
        
    def save_model(self, model, scaler, accuracy):
        """학습된 모델 저장"""
        model_path = os.path.join(self.user_dir, MODEL_FILE)
        
        # 모델 저장
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': self.model_params,
            'scaler': scaler,
            'accuracy': accuracy
        }, model_path)
        
        return model_path


# ------------------------------------------------------------------------
# 가위바위보 게임 위젯
# ------------------------------------------------------------------------
class RockPaperScissorsGameWidget(QWidget):
    """
    RockPaperScissorsGameWidget 클래스: 가위바위보 게임 화면을 제공하는 위젯
    - EMG 센서 기반 가위바위보 게임 제공
    - 사용자와 컴퓨터 간의 대결
    - 네오모피즘 디자인 적용
    """
    back_to_main = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # 상태 변수
        self.username = ""
        self.password = ""
        self.user_dir = ""
        self.game_state = 0  # 0: 준비, 1: 카운트다운, 2: 게임 중, 3: 결과
        self.player_score = 0
        self.computer_score = 0
        self.round = 0
        self.max_rounds = 10
        self.countdown_value = 3
        self.player_choice = None
        self.computer_choice = None
        self.game_result = None
        
        # EMG 프로세서 및 예측기
        self.emg_processor = EMGProcessor()
        self.predictor = None
        self.serial_thread = None
        
        # UI 초기화
        self.init_ui()
        
        # 타이머 설정
        self.countdown_timer = QTimer(self)
        self.countdown_timer.timeout.connect(self.update_countdown)
        
        self.game_timer = QTimer(self)
        self.game_timer.timeout.connect(self.update_game)
        
        self.detection_timer = QTimer(self)
        self.detection_timer.timeout.connect(self.update_detection)
        
    def init_ui(self):
        """UI 초기화 - 네오모피즘 디자인 적용"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(40, 40, 40, 40)
        layout.setSpacing(30)
        
        # 배경 설정
        self.setStyleSheet(f"background-color: {BG_COLOR};")
        
        # 제목 카드
        title_card = NeumorphicCard()
        title_layout = QVBoxLayout(title_card)
        title_layout.setContentsMargins(20, 20, 20, 20)
        
        # 제목 텍스트 - 이 부분을 수정하여 제목을 변경할 수 있습니다
        self.title_label = QLabel("가위바위보 게임")
        self.title_label.setStyleSheet(f"color: {TEXT_COLOR}; font-size: 28px; font-weight: bold;")
        self.title_label.setAlignment(Qt.AlignCenter)
        
        title_layout.addWidget(self.title_label)
        
        # 점수 카드
        score_card = NeumorphicCard()
        score_layout = QHBoxLayout(score_card)
        score_layout.setContentsMargins(20, 15, 20, 15)
        
        # 플레이어 점수
        player_score_card = NeumorphicCard()
        player_score_card.setStyleSheet("""
            background-color: #EEF2FF;
            border-radius: 15px;
        """)
        player_score_layout = QVBoxLayout(player_score_card)
        
        # 플레이어 점수 레이블 - 이 부분을 수정하여 플레이어 점수 텍스트를 변경할 수 있습니다
        self.player_score_label = QLabel("플레이어: 0")
        self.player_score_label.setStyleSheet(f"color: {PRIMARY_COLOR}; font-size: 18px; font-weight: bold;")
        self.player_score_label.setAlignment(Qt.AlignCenter)
        
        player_score_layout.addWidget(self.player_score_label)
        
        # 라운드 표시
        round_card = NeumorphicCard()
        round_layout = QVBoxLayout(round_card)
        
        # 라운드 레이블 - 이 부분을 수정하여 라운드 텍스트를 변경할 수 있습니다
        self.round_label = QLabel("라운드: 0/10")
        self.round_label.setStyleSheet(f"color: {TEXT_COLOR}; font-size: 18px; font-weight: bold;")
        self.round_label.setAlignment(Qt.AlignCenter)
        
        round_layout.addWidget(self.round_label)
        
        # 컴퓨터 점수
        computer_score_card = NeumorphicCard()
        computer_score_card.setStyleSheet("""
            background-color: #FFF5F5;
            border-radius: 15px;
        """)
        computer_score_layout = QVBoxLayout(computer_score_card)
        
        # 컴퓨터 점수 레이블 - 이 부분을 수정하여 컴퓨터 점수 텍스트를 변경할 수 있습니다
        self.computer_score_label = QLabel("컴퓨터: 0")
        self.computer_score_label.setStyleSheet("color: #F5365C; font-size: 18px; font-weight: bold;")
        self.computer_score_label.setAlignment(Qt.AlignCenter)
        
        computer_score_layout.addWidget(self.computer_score_label)
        
        # 점수 레이아웃에 추가
        score_layout.addWidget(player_score_card)
        score_layout.addStretch()
        score_layout.addWidget(round_card)
        score_layout.addStretch()
        score_layout.addWidget(computer_score_card)
        
        # 게임 영역 카드
        game_card = NeumorphicCard()
        game_layout = QHBoxLayout(game_card)
        game_layout.setContentsMargins(30, 30, 30, 30)
        game_layout.setSpacing(20)
        
        # 플레이어 영역
        player_layout = QVBoxLayout()
        
        # 플레이어 레이블 - 이 부분을 수정하여 플레이어 레이블 텍스트를 변경할 수 있습니다
        self.player_label = QLabel("플레이어")
        self.player_label.setStyleSheet(f"color: {PRIMARY_COLOR}; font-size: 24px; font-weight: bold;")
        self.player_label.setAlignment(Qt.AlignCenter)
        
        # 플레이어 상태 표시 - 이 부분 추가
        player_status_card = NeumorphicCard(inset=True)
        player_status_layout = QVBoxLayout(player_status_card)
        
        # 플레이어 상태 레이블 - 이 부분 추가
        self.player_status = QLabel("대기중")
        self.player_status.setStyleSheet(f"color: {PRIMARY_COLOR}; font-size: 18px;")
        self.player_status.setAlignment(Qt.AlignCenter)
        
        player_status_layout.addWidget(self.player_status)
        
        # 플레이어 선택 표시
        player_choice_card = NeumorphicCard()
        player_choice_card.setFixedSize(200, 200)
        player_choice_card.setStyleSheet("""
            background-color: #EEF2FF;
            border-radius: 100px;
        """)
        
        player_choice_layout = QVBoxLayout(player_choice_card)
        
        # 플레이어 선택 레이블 - 이 부분을 수정하여 플레이어 선택 텍스트를 변경할 수 있습니다
        self.player_choice_label = QLabel("?")
        self.player_choice_label.setStyleSheet("font-size: 72px;")
        self.player_choice_label.setAlignment(Qt.AlignCenter)
        
        player_choice_layout.addWidget(self.player_choice_label)
        
        player_layout.addWidget(self.player_label)
        player_layout.addWidget(player_status_card)  # 이 부분 추가
        player_layout.addWidget(player_choice_card, 0, Qt.AlignCenter)
        
        # 중앙 영역
        center_layout = QVBoxLayout()
        
        # VS 레이블 - 이 부분을 수정하여 VS 텍스트를 변경할 수 있습니다
        self.vs_label = QLabel("VS")
        self.vs_label.setStyleSheet(f"color: {TEXT_COLOR}; font-size: 28px; font-weight: bold;")
        self.vs_label.setAlignment(Qt.AlignCenter)
        
        # 카운트다운 표시
        countdown_card = NeumorphicCard()
        countdown_card.setFixedSize(80, 80)
        countdown_card.setStyleSheet(f"""
            background-color: {SECONDARY_COLOR};
            border-radius: 40px;
        """)
        
        countdown_layout = QVBoxLayout(countdown_card)
        
        # 카운트다운 레이블 - 이 부분을 수정하여 카운트다운 텍스트를 변경할 수 있습니다
        self.countdown_label = QLabel("3")
        self.countdown_label.setStyleSheet("font-size: 36px; font-weight: bold; color: white;")
        self.countdown_label.setAlignment(Qt.AlignCenter)
        
        countdown_layout.addWidget(self.countdown_label)
        
        # 상태 표시
        status_card = NeumorphicCard(inset=True)
        status_layout = QVBoxLayout(status_card)
        
        # 상태 레이블 - 이 부분을 수정하여 상태 텍스트를 변경할 수 있습니다
        self.status_label = QLabel("게임을 시작하려면 '시작' 버튼을 누르세요")
        self.status_label.setStyleSheet(f"color: {TEXT_COLOR}; font-size: 16px;")
        self.status_label.setAlignment(Qt.AlignCenter)
        
        status_layout.addWidget(self.status_label)
        
        center_layout.addWidget(self.vs_label)
        center_layout.addWidget(countdown_card, 0, Qt.AlignCenter)
        center_layout.addWidget(status_card)
        center_layout.addStretch()
        
        # 컴퓨터 영역
        computer_layout = QVBoxLayout()
        
        # 컴퓨터 레이블 - 이 부분을 수정하여 컴퓨터 레이블 텍스트를 변경할 수 있습니다
        self.computer_label = QLabel("컴퓨터")
        self.computer_label.setStyleSheet("color: #F5365C; font-size: 24px; font-weight: bold;")
        self.computer_label.setAlignment(Qt.AlignCenter)
        
        # 컴퓨터 상태 표시 - 이 부분 추가
        computer_status_card = NeumorphicCard(inset=True)
        computer_status_layout = QVBoxLayout(computer_status_card)
        
        # 컴퓨터 상태 레이블 - 이 부분 추가
        self.computer_status = QLabel("대기중")
        self.computer_status.setStyleSheet("color: #F5365C; font-size: 18px;")
        self.computer_status.setAlignment(Qt.AlignCenter)
        
        computer_status_layout.addWidget(self.computer_status)
        
        # 컴퓨터 선택 표시
        computer_choice_card = NeumorphicCard()
        computer_choice_card.setFixedSize(200, 200)
        computer_choice_card.setStyleSheet("""
            background-color: #FFF5F5;
            border-radius: 100px;
        """)
        
        computer_choice_layout = QVBoxLayout(computer_choice_card)
        
        # 컴퓨터 선택 레이블 - 이 부분을 수정하여 컴퓨터 선택 텍스트를 변경할 수 있습니다
        self.computer_choice_label = QLabel("?")
        self.computer_choice_label.setStyleSheet("font-size: 72px;")
        self.computer_choice_label.setAlignment(Qt.AlignCenter)
        
        computer_choice_layout.addWidget(self.computer_choice_label)
        
        computer_layout.addWidget(self.computer_label)
        computer_layout.addWidget(computer_status_card)  # 이 부분 추가
        computer_layout.addWidget(computer_choice_card, 0, Qt.AlignCenter)
        
        # 게임 레이아웃에 추가
        game_layout.addLayout(player_layout)
        game_layout.addLayout(center_layout)
        game_layout.addLayout(computer_layout)
        
        # 결과 표시 카드
        result_card = NeumorphicCard()
        result_layout = QVBoxLayout(result_card)
        
        # 결과 레이블 - 이 부분을 수정하여 결과 텍스트를 변경할 수 있습니다
        self.result_label = QLabel("")
        self.result_label.setStyleSheet("""
            font-size: 32px;
            font-weight: bold;
            color: #2DCE89;
        """)
        self.result_label.setAlignment(Qt.AlignCenter)
        
        result_layout.addWidget(self.result_label)
        
        # 버튼 레이아웃
        button_layout = QHBoxLayout()
        
        # 버튼 - 이 부분을 수정하여 버튼 텍스트를 변경할 수 있습니다
        self.back_btn = NeumorphicButton("뒤로")
        self.back_btn.clicked.connect(self.go_back)
        
        self.start_btn = NeumorphicButton("시작", PRIMARY_COLOR)
        self.start_btn.clicked.connect(self.start_game)
        
        button_layout.addWidget(self.back_btn)
        button_layout.addStretch()
        button_layout.addWidget(self.start_btn)
        
        # 레이아웃에 카드 추가
        layout.addWidget(title_card)
        layout.addWidget(score_card)
        layout.addWidget(game_card)
        layout.addWidget(result_card)
        layout.addLayout(button_layout)
        
        # 초기에 카운트다운 레이블 숨기기
        countdown_card.setVisible(False)
        
        # 카운트다운 카드 저장
        self.countdown_card = countdown_card
        
    def set_user(self, username, password):
        """사용자 정보 설정"""
        self.username = username
        self.password = password
        self.user_dir = get_user_dir(username, password)
        
        # 모델 로드
        model_path = os.path.join(self.user_dir, MODEL_FILE)
        self.predictor = EMGPredictor(model_path)
        
        # 감도 설정 로드
        config = load_user_config(username, password)
        self.emg_processor.set_sensitivity(config["sensitivity"])
        
    def start(self):
        """게임 시작 준비"""
        self.reset_game()
        
        # 시리얼 스레드 시작
        if self.serial_thread:
            self.serial_thread.disconnect_serial()
            self.serial_thread = None
            
        self.serial_thread = SerialThread()
        self.serial_thread.data_received.connect(self.process_data)
        self.serial_thread.start()
        
        # 상태 업데이트 타이머 시작
        self.detection_timer.start(100)
        
    def stop(self):
        """게임 중지"""
        self.countdown_timer.stop()
        self.game_timer.stop()
        self.detection_timer.stop()
        
        if self.serial_thread:
            self.serial_thread.disconnect_serial()
            self.serial_thread = None
            
    def reset_game(self):
        """게임 상태 초기화"""
        self.game_state = 0
        self.player_score = 0
        self.computer_score = 0
        self.round = 0
        self.player_choice = None
        self.computer_choice = None
        self.game_result = None
        self.current_attacker = None
        
        # UI 업데이트
        self.player_score_label.setText(f"플레이어: {self.player_score}")
        self.computer_score_label.setText(f"컴퓨터: {self.computer_score}")
        self.round_label.setText(f"라운드: {self.round}/{self.max_rounds}")
        
        self.player_choice_label.setText("?")
        self.computer_choice_label.setText("?")
        
        self.player_status.setText("대기중")
        self.computer_status.setText("대기중")
        
        self.result_label.setText("")
        self.status_label.setText("게임을 시작하려면 '시작' 버튼을 누르세요")
        
        self.countdown_card.setVisible(False)
        self.start_btn.setEnabled(True)
        self.start_btn.setText("시작")
        
    def process_data(self, s1, s2):
        """EMG 데이터 처리"""
        result = self.emg_processor.process_data(s1, s2)
        
        # 동작 감지 중일 때만 처리
        if (self.game_state == 1 or self.game_state == 2) and isinstance(result, tuple) and result[0] == "movement_completed":
            sequence = result[1]
            # 동작 예측
            prediction, confidence = self.predictor.predict(sequence)
            
            # 충분한 신뢰도를 가진 가위/바위/보 동작만 처리
            if prediction in [LABEL_SCISSORS, LABEL_ROCK, LABEL_PAPER] and confidence >= 0.5:
                self.player_choice = prediction
                self.update_player_choice()
                
    def update_player_choice(self):
        """플레이어 선택 업데이트"""
        if self.player_choice == LABEL_SCISSORS:
            self.player_choice_label.setText("✌️")
        elif self.player_choice == LABEL_ROCK:
            self.player_choice_label.setText("✊")
        elif self.player_choice == LABEL_PAPER:
            self.player_choice_label.setText("✋")
            
    def update_computer_choice(self):
        """컴퓨터 선택 업데이트"""
        if self.computer_choice == LABEL_SCISSORS:
            self.computer_choice_label.setText("✌️")
        elif self.computer_choice == LABEL_ROCK:
            self.computer_choice_label.setText("✊")
        elif self.computer_choice == LABEL_PAPER:
            self.computer_choice_label.setText("✋")
            
    def start_game(self):
        """게임 시작"""
        if self.game_state == 0:
            # 새 라운드 시작
            self.round += 1
            self.round_label.setText(f"라운드: {self.round}/{self.max_rounds}")
            
            # 첫 판은 가위바위보
            self.game_state = 1
            self.player_choice = None
            self.computer_choice = None
            self.current_attacker = None
            
            self.player_status.setText("")
            self.computer_status.setText("")
            
            self.start_countdown()
            self.start_btn.setEnabled(False)
            
    def start_countdown(self):
        """카운트다운 시작"""
        self.countdown_value = 3
        self.countdown_label.setText(str(self.countdown_value))
        self.countdown_card.setVisible(True)
        self.status_label.setText("준비...")
        self.countdown_timer.start(1000)
        
    def update_countdown(self):
        """카운트다운 업데이트"""
        self.countdown_value -= 1
        self.countdown_label.setText(str(self.countdown_value))
        
        if self.countdown_value <= 0:
            self.countdown_timer.stop()
            
            # 카운트다운 끝, 게임 시작
            self.countdown_card.setVisible(False)
            
            if self.game_state == 1:
                self.status_label.setText("가위... 바위... 보!")
            else:
                self.status_label.setText("묵... 찌... 빠!")
                
            # 플레이어 선택 초기화
            self.player_choice = None
            self.player_choice_label.setText("?")
            
            # 컴퓨터 선택 초기화
            self.computer_choice = None
            self.computer_choice_label.setText("?")
            
            # 게임 타이머 시작 (5초 후 결과 확인)
            self.game_timer.start(5000)
            
    def update_game(self):
        """게임 상태 업데이트"""
        self.game_timer.stop()
        
        # 컴퓨터 선택
        self.computer_choice = get_computer_choice()
        self.update_computer_choice()
        
        # 플레이어가 선택하지 않았으면 랜덤 선택
        if self.player_choice is None:
            self.status_label.setText("시간 초과! 랜덤으로 선택됩니다.")
            self.player_choice = get_computer_choice()  # 랜덤 선택
            self.update_player_choice()
            
        # 가위바위보 결과 확인
        if self.game_state == 1:
            # 첫 판 가위바위보
            result = determine_winner(self.player_choice, self.computer_choice)
            
            if result == RESULT_WIN:
                # 플레이어 승리, 공격자가 됨
                self.current_attacker = True
                self.player_status.setText("공격자")
                self.computer_status.setText("수비자")
                self.result_label.setText("선공권 획득!")
                self.result_label.setStyleSheet("font-size: 32px; font-weight: bold; color: #2DCE89;")
                
            elif result == RESULT_LOSE:
                # 컴퓨터 승리, 공격자가 됨
                self.current_attacker = False
                self.player_status.setText("수비자")
                self.computer_status.setText("공격자")
                self.result_label.setText("후공!")
                self.result_label.setStyleSheet("font-size: 32px; font-weight: bold; color: #F5365C;")
                
            else:
                # 무승부, 재경기
                self.result_label.setText("무승부! 다시 시작")
                self.result_label.setStyleSheet("font-size: 32px; font-weight: bold; color: #FB8C00;")
                QTimer.singleShot(2000, self.start_countdown)
                return
                
            # 묵찌빠 시작
            self.game_state = 2
            QTimer.singleShot(2000, self.start_countdown)
            
        elif self.game_state == 2:
            # 묵찌빠 진행
            
            # 선택 표시
            player_choice_text = get_label_name(self.player_choice)
            computer_choice_text = get_label_name(self.computer_choice)
            self.status_label.setText(f"{player_choice_text} vs {computer_choice_text}")
            
            # 같은 손 모양인 경우 공격자 승리
            if self.player_choice == self.computer_choice:
                if self.current_attacker:
                    # 플레이어 승리
                    self.player_score += 1
                    self.player_score_label.setText(f"플레이어: {self.player_score}")
                    self.result_label.setText("플레이어 승리!")
                    self.result_label.setStyleSheet("font-size: 32px; font-weight: bold; color: #2DCE89;")
                else:
                    # 컴퓨터 승리
                    self.computer_score += 1
                    self.computer_score_label.setText(f"컴퓨터: {self.computer_score}")
                    self.result_label.setText("컴퓨터 승리!")
                    self.result_label.setStyleSheet("font-size: 32px; font-weight: bold; color: #F5365C;")
                    
                # 라운드 종료, 다음 라운드 확인
                self.check_game_end()
                
            else:
                # 다른 손 모양, 가위바위보 규칙으로 공격자 결정
                result = determine_winner(self.player_choice, self.computer_choice)
                
                if result == RESULT_WIN:
                    # 플레이어 공격자로 전환
                    self.current_attacker = True
                    self.player_status.setText("공격자")
                    self.computer_status.setText("수비자")
                    self.result_label.setText("공격권 획득!")
                    self.result_label.setStyleSheet("font-size: 32px; font-weight: bold; color: #2DCE89;")
                    
                elif result == RESULT_LOSE:
                    # 컴퓨터 공격자로 전환
                    self.current_attacker = False
                    self.player_status.setText("수비자")
                    self.computer_status.setText("공격자")
                    self.result_label.setText("수비!")
                    self.result_label.setStyleSheet("font-size: 32px; font-weight: bold; color: #F5365C;")
                    
                else:
                    # 묵찌빠에서 무승부는 공격자 유지
                    if self.current_attacker:
                        self.result_label.setText("무승부! 공격 유지")
                    else:
                        self.result_label.setText("무승부! 수비 유지")
                        
                self.result_label.setStyleSheet("font-size: 32px; font-weight: bold; color: #FB8C00;")
                
                # 묵찌빠 계속 진행
                QTimer.singleShot(2000, self.start_countdown)
                
    def check_game_end(self):
        """게임 종료 확인"""
        # 최대 라운드 도달 확인
        if self.round >= self.max_rounds or self.player_score > self.max_rounds // 2 or self.computer_score > self.max_rounds // 2:
            # 게임 종료
            self.game_state = 3
            
            if self.player_score > self.computer_score:
                self.status_label.setText("게임 종료 - 플레이어 승리!")
            elif self.player_score < self.computer_score:
                self.status_label.setText("게임 종료 - 컴퓨터 승리!")
            else:
                self.status_label.setText("게임 종료 - 무승부!")
                
            self.start_btn.setText("다시 시작")
            self.start_btn.setEnabled(True)
            self.game_state = 0
        else:
            # 다음 라운드 준비
            QTimer.singleShot(2000, self.next_round)
            
    def next_round(self):
        """다음 라운드 시작"""
        self.start_game()
        
    def update_detection(self):
        """동작 감지 상태 업데이트"""
        # 게임 상태에 따른 처리
        if self.game_state == 1 or self.game_state == 2:
            # 게임 중일 때만 플레이어 선택 표시
            if self.player_choice:
                self.update_player_choice()
                
    def go_back(self):
        """뒤로 가기"""
        self.stop()
        self.back_to_main.emit()


# ------------------------------------------------------------------------
# 메인 애플리케이션
# ------------------------------------------------------------------------
class EMGGameApplication(QMainWindow):
    """
    EMGGameApplication 클래스: 메인 애플리케이션 윈도우
    - 각 위젯(화면)을 관리하고 전환
    - 프로그램 초기화 및 종료 처리
    - 네오모피즘 디자인 적용
    """
    def __init__(self):
        super().__init__()
        
        # 상태 변수
        self.username = ""
        self.password = ""
        
        # UI 초기화
        self.init_ui()
        
        # 이미지 폴더 및 파일 확인
        self.check_resources()
        
    def init_ui(self):
        """UI 초기화"""
        # 메인 윈도우 설정
        self.setWindowTitle("EMG 가위바위보 & 묵찌빠 게임")
        self.setMinimumSize(800, 600)
        
        # 배경 설정
        self.setStyleSheet(f"background-color: {BG_COLOR};")
        
        # 중앙 위젯
        self.central_widget = QStackedWidget()
        self.setCentralWidget(self.central_widget)
        
        # 로그인 위젯
        self.login_widget = LoginWidget()
        self.login_widget.login_successful.connect(self.on_login)
        
        # 메인 메뉴 위젯
        self.main_menu = MainMenuWidget()
        self.main_menu.show_guide.connect(self.show_guide)
        self.main_menu.show_data_collection.connect(self.show_data_collection)
        self.main_menu.show_rps_game.connect(self.show_rps_game)
        self.main_menu.show_muk_game.connect(self.show_muk_game)
        
        # 사용 가이드 위젯
        self.guide_widget = GuideWidget()
        self.guide_widget.guide_completed.connect(self.on_guide_completed)
        
        # 데이터 수집 위젯
        self.data_collection = DataCollectionWidget()
        self.data_collection.collection_completed.connect(self.show_model_training)
        
        # 모델 학습 위젯
        self.model_training = ModelTrainingWidget()
        self.model_training.training_completed.connect(self.show_main_menu)
        
        # 가위바위보 게임 위젯
        self.rps_game = RockPaperScissorsGameWidget()
        self.rps_game.back_to_main.connect(self.show_main_menu)
        
        # 묵찌빠 게임 위젯
        self.muk_game = MukJjiPpaGameWidget()
        self.muk_game.back_to_main.connect(self.show_main_menu)
        
        # 스택에 위젯 추가
        self.central_widget.addWidget(self.login_widget)
        self.central_widget.addWidget(self.main_menu)
        self.central_widget.addWidget(self.guide_widget)
        self.central_widget.addWidget(self.data_collection)
        self.central_widget.addWidget(self.model_training)
        self.central_widget.addWidget(self.rps_game)
        self.central_widget.addWidget(self.muk_game)
        
        # 로그인 화면으로 시작
        self.central_widget.setCurrentWidget(self.login_widget)
        
    def check_resources(self):
        """필요한 리소스 확인"""
        # 사용자 디렉토리 확인
        ensure_dir_exists(USERS_DIR)
        
        # 이미지 생성
        create_placeholder_images()
        
    def on_login(self, username, password):
        """로그인 처리"""
        self.username = username
        self.password = password
        
        # 사용자 디렉토리 생성
        user_dir = get_user_dir(username, password)
        ensure_dir_exists(user_dir)
        
        # 모델 파일 확인
        has_model = user_has_model(username, password)
        
        if has_model:
            # 모델이 있으면 메인 메뉴로
            self.show_main_menu()
        else:
            # 모델이 없으면 가이드로
            self.show_guide()
            
    def show_main_menu(self):
        """메인 메뉴 표시"""
        self.main_menu.set_username(self.username)
        self.central_widget.setCurrentWidget(self.main_menu)
        
    def show_guide(self):
        """가이드 표시"""
        self.guide_widget.set_user(self.username, self.password)
        self.guide_widget.start()
        self.central_widget.setCurrentWidget(self.guide_widget)
        
    def on_guide_completed(self, need_training):
        """가이드 완료 처리"""
        if need_training:
            # 모델 학습 필요
            self.show_data_collection()
        else:
            # 메인 메뉴로
            self.show_main_menu()
            
    def show_data_collection(self):
        """데이터 수집 화면 표시"""
        self.data_collection.set_user(self.username, self.password)
        self.data_collection.start()
        self.central_widget.setCurrentWidget(self.data_collection)
        
    def show_model_training(self):
        """모델 학습 화면 표시"""
        self.model_training.set_user(self.username, self.password)
        self.model_training.start()
        self.central_widget.setCurrentWidget(self.model_training)
        
    def show_rps_game(self):
        """가위바위보 게임 화면 표시"""
        self.rps_game.set_user(self.username, self.password)
        self.rps_game.start()
        self.central_widget.setCurrentWidget(self.rps_game)
        
    def show_muk_game(self):
        """묵찌빠 게임 화면 표시"""
        self.muk_game.set_user(self.username, self.password)
        self.muk_game.start()
        self.central_widget.setCurrentWidget(self.muk_game)
        
    def closeEvent(self, event):
        """프로그램 종료 처리"""
        # 모든 스레드 정리
        for widget in [self.guide_widget, self.data_collection, self.rps_game, self.muk_game]:
            if hasattr(widget, 'stop'):
                widget.stop()
                
        event.accept()


# ------------------------------------------------------------------------
# 묵찌빠 게임 위젯
# ------------------------------------------------------------------------
class MukJjiPpaGameWidget(QWidget):
    """
    MukJjiPpaGameWidget 클래스: 묵찌빠 게임 화면을 제공하는 위젯
    - EMG 센서 기반 묵찌빠 게임 제공
    - 가위바위보로 선공을 결정한 뒤 묵찌빠 게임 진행
    - 네오모피즘 디자인 적용
    """
    back_to_main = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # 상태 변수
        self.username = ""
        self.password = ""
        self.user_dir = ""
        self.game_state = 0  # 0: 준비, 1: 첫 판 가위바위보, 2: 묵찌빠, 3: 결과
        self.player_score = 0
        self.computer_score = 0
        self.round = 0
        self.max_rounds = 5
        self.countdown_value = 3
        self.player_choice = None
        self.computer_choice = None
        self.game_result = None
        
        # 묵찌빠 전용 변수
        self.current_attacker = None  # 현재 공격자 (True: 플레이어, False: 컴퓨터)
        
        # EMG 프로세서 및 예측기
        self.emg_processor = EMGProcessor()
        self.predictor = None
        self.serial_thread = None
        
        # UI 초기화
        self.init_ui()
        
        # 타이머 설정
        self.countdown_timer = QTimer(self)
        self.countdown_timer.timeout.connect(self.update_countdown)
        
        self.game_timer = QTimer(self)
        self.game_timer.timeout.connect(self.update_game)
        
        self.detection_timer = QTimer(self)
        self.detection_timer.timeout.connect(self.update_detection)
        
    def init_ui(self):
        """UI 초기화 - 네오모피즘 디자인 적용"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(40, 40, 40, 40)
        layout.setSpacing(30)
        
        # 배경 설정
        self.setStyleSheet(f"background-color: {BG_COLOR};")
        
        # 제목 카드
        title_card = NeumorphicCard()
        title_layout = QVBoxLayout(title_card)
        title_layout.setContentsMargins(20, 20, 20, 20)
        
        # 제목 텍스트 - 이 부분을 수정하여 제목을 변경할 수 있습니다
        self.title_label = QLabel("묵찌빠 게임")
        self.title_label.setStyleSheet(f"color: {TEXT_COLOR}; font-size: 28px; font-weight: bold;")
        self.title_label.setAlignment(Qt.AlignCenter)
        
        title_layout.addWidget(self.title_label)
        
        # 상단 정보 레이아웃
        top_layout = QHBoxLayout()
        
        # 점수 카드
        score_card = NeumorphicCard()
        score_layout = QVBoxLayout(score_card)
        score_layout.setContentsMargins(20, 15, 20, 15)
        
        # 점수 및 라운드 그리드
        score_grid = QGridLayout()
        
        # 플레이어 점수
        player_score_card = NeumorphicCard(inset=True)
        player_score_layout = QVBoxLayout(player_score_card)
        
        # 플레이어 점수 레이블 - 이 부분을 수정하여 플레이어 점수 텍스트를 변경할 수 있습니다
        self.player_score_label = QLabel("플레이어: 0")
        self.player_score_label.setStyleSheet(f"color: {PRIMARY_COLOR}; font-size: 16px; font-weight: bold;")
        self.player_score_label.setAlignment(Qt.AlignCenter)
        
        player_score_layout.addWidget(self.player_score_label)
        
        # 라운드 표시
        round_card = NeumorphicCard(inset=True)
        round_layout = QVBoxLayout(round_card)
        
        # 라운드 레이블 - 이 부분을 수정하여 라운드 텍스트를 변경할 수 있습니다
        self.round_label = QLabel("라운드: 0/5")
        self.round_label.setStyleSheet(f"color: {TEXT_COLOR}; font-size: 16px; font-weight: bold;")
        self.round_label.setAlignment(Qt.AlignCenter)
        
        round_layout.addWidget(self.round_label)
        
        # 컴퓨터 점수
        computer_score_card = NeumorphicCard(inset=True)
        computer_score_layout = QVBoxLayout(computer_score_card)
        
        # 컴퓨터 점수 레이블 - 이 부분을 수정하여 컴퓨터 점수 텍스트를 변경할 수 있습니다
        self.computer_score_label = QLabel("컴퓨터: 0")
        self.computer_score_label.setStyleSheet("color: #F5365C; font-size: 16px; font-weight: bold;")
        self.computer_score_label.setAlignment(Qt.AlignCenter)
        
        computer_score_layout.addWidget(self.computer_score_label)
        
        # 그리드에 점수 위젯 추가
        score_grid.addWidget(player_score_card, 0, 0)
        score_grid.addWidget(round_card, 0, 1)
        score_grid.addWidget(computer_score_card, 0, 2)
        
        score_layout.addLayout(score_grid)
        
        # 규칙 카드
        rules_card = NeumorphicCard()
        rules_layout = QVBoxLayout(rules_card)
        
        # 규칙 제목 - 이 부분을 수정하여 규칙 제목 텍스트를 변경할 수 있습니다
        rules_title = QLabel("게임 규칙")
        rules_title.setStyleSheet(f"color: {TEXT_COLOR}; font-size: 16px; font-weight: bold;")
        rules_title.setAlignment(Qt.AlignCenter)
        
        # 규칙 텍스트 영역
        rules_text_card = NeumorphicCard(inset=True)
        rules_text_layout = QVBoxLayout(rules_text_card)
        
        # 규칙 텍스트 - 이 부분을 수정하여 규칙 텍스트를 변경할 수 있습니다
        self.rules_text = QLabel(
            "1. 첫 판 가위바위보에서 이긴 사람이 공격자가 됩니다.\n"
            "2. 공격자와 수비자가 같은 손 모양을 내면 공격자가 승리합니다.\n"
            "3. 다른 손 모양을 내면 가위바위보 규칙에 따라 이긴 사람이 공격자가 됩니다."
        )
        self.rules_text.setStyleSheet(f"color: {TEXT_COLOR}; font-size: 13px; line-height: 1.4;")
        self.rules_text.setWordWrap(True)
        
        rules_text_layout.addWidget(self.rules_text)
        
        rules_layout.addWidget(rules_title)
        rules_layout.addWidget(rules_text_card)
        
        # 상단 레이아웃에 추가
        top_layout.addWidget(score_card, 3)
        top_layout.addWidget(rules_card, 4)
        
        # 게임 영역 카드
        game_card = NeumorphicCard()
        game_layout = QHBoxLayout(game_card)
        game_layout.setContentsMargins(30, 30, 30, 30)
        game_layout.setSpacing(20)
        
        # 플레이어 영역
        player_layout = QVBoxLayout()
        
        # 플레이어 레이블 - 이 부분을 수정하여 플레이어 레이블 텍스트를 변경할 수 있습니다
        self.player_label = QLabel("플레이어")
        self.player_label.setStyleSheet(f"color: {PRIMARY_COLOR}; font-size: 22px; font-weight: bold;")
        self.player_label.setAlignment(Qt.AlignCenter)
        
        # 플레이어 상태 표시
        player_status_card = NeumorphicCard(inset=True)
        player_status_layout = QVBoxLayout(player_status_card)
        
        # 플레이어 상태 레이블 - 이 부분을 수정하여 플레이어 상태 텍스트를 변경할 수 있습니다
        self.player_status = QLabel("대기중")
        self.player_status.setStyleSheet(f"color: {PRIMARY_COLOR}; font-size: 14px;")
        self.player_status.setAlignment(Qt.AlignCenter)
        
        player_status_layout.addWidget(self.player_status)
        
        # 플레이어 선택 표시
        player_choice_card = NeumorphicCard()
        player_choice_card.setFixedSize(180, 180)
        player_choice_card.setStyleSheet("""
            background-color: #EEF2FF;
            border-radius: 90px;
        """)
        
        player_choice_layout = QVBoxLayout(player_choice_card)
        
        # 플레이어 선택 레이블 - 이 부분을 수정하여 플레이어 선택 텍스트를 변경할 수 있습니다
        self.player_choice_label = QLabel("?")
        self.player_choice_label.setStyleSheet("font-size: 72px;")
        self.player_choice_label.setAlignment(Qt.AlignCenter)
        
        player_choice_layout.addWidget(self.player_choice_label)
        
        player_layout.addWidget(self.player_label)
        player_layout.addWidget(player_status_card)
        player_layout.addWidget(player_choice_card, 0, Qt.AlignCenter)
        
        # 중앙 영역
        center_layout = QVBoxLayout()
        
        # VS 레이블 - 이 부분을 수정하여 VS 텍스트를 변경할 수 있습니다
        self.vs_label = QLabel("VS")
        self.vs_label.setStyleSheet(f"color: {TEXT_COLOR}; font-size: 28px; font-weight: bold;")
        self.vs_label.setAlignment(Qt.AlignCenter)
        
        # 카운트다운 표시
        countdown_card = NeumorphicCard()
        countdown_card.setFixedSize(80, 80)
        countdown_card.setStyleSheet(f"""
            background-color: {SECONDARY_COLOR};
            border-radius: 40px;
        """)
        
        countdown_layout = QVBoxLayout(countdown_card)
        
        # 카운트다운 레이블 - 이 부분을 수정하여 카운트다운 텍스트를 변경할 수 있습니다
        self.countdown_label = QLabel("3")
        self.countdown_label.setStyleSheet("font-size: 36px; font-weight: bold; color: white;")
        self.countdown_label.setAlignment(Qt.AlignCenter)
        
        countdown_layout.addWidget(self.countdown_label)
        
        # 상태 표시
        status_card = NeumorphicCard(inset=True)
        status_layout = QVBoxLayout(status_card)
        
        # 상태 레이블 - 이 부분을 수정하여 상태 텍스트를 변경할 수 있습니다
        self.status_label = QLabel("게임을 시작하려면 '시작' 버튼을 누르세요")
        self.status_label.setStyleSheet(f"color: {TEXT_COLOR}; font-size: 16px;")
        self.status_label.setAlignment(Qt.AlignCenter)
        
        status_layout.addWidget(self.status_label)
        
        center_layout.addWidget(self.vs_label)
        center_layout.addWidget(countdown_card, 0, Qt.AlignCenter)
        center_layout.addWidget(status_card)
        center_layout.addStretch()
        
        # 컴퓨터 영역
        computer_layout = QVBoxLayout()
        
        # 컴퓨터 레이블 - 이 부분을 수정하여 컴퓨터 레이블 텍스트를 변경할 수 있습니다
        self.computer_label = QLabel("컴퓨터")
        self.computer_label.setStyleSheet("color: #F5365C; font-size: 22px; font-weight: bold;")
        self.computer_label.setAlignment(Qt.AlignCenter)
        
        # 컴퓨터 상태 표시
        computer_status_card = NeumorphicCard(inset=True)
        computer_status_layout = QVBoxLayout(computer_status_card)
        
        # 컴퓨터 상태 레이블 - 이 부분을 수정하여 컴퓨터 상태 텍스트를 변경할 수 있습니다
        self.computer_status = QLabel("대기중")
        self.computer_status.setStyleSheet("color: #F5365C; font-size: 14px;")
        self.computer_status.setAlignment(Qt.AlignCenter)
        
        computer_status_layout.addWidget(self.computer_status)
        
        # 컴퓨터 선택 표시
        computer_choice_card = NeumorphicCard()
        computer_choice_card.setFixedSize(180, 180)
        computer_choice_card.setStyleSheet("""
            background-color: #FFF5F5;
            border-radius: 90px;
        """)
        
        computer_choice_layout = QVBoxLayout(computer_choice_card)
        
        # 컴퓨터 선택 레이블 - 이 부분을 수정하여 컴퓨터 선택 텍스트를 변경할 수 있습니다
        self.computer_choice_label = QLabel("?")
        self.computer_choice_label.setStyleSheet("font-size: 72px;")
        self.computer_choice_label.setAlignment(Qt.AlignCenter)
        
        computer_choice_layout.addWidget(self.computer_choice_label)
        
        computer_layout.addWidget(self.computer_label)
        computer_layout.addWidget(computer_status_card)
        computer_layout.addWidget(computer_choice_card, 0, Qt.AlignCenter)
        
        # 게임 레이아웃에 추가
        game_layout.addLayout(player_layout)
        game_layout.addLayout(center_layout)
        game_layout.addLayout(computer_layout)
        
        # 결과 표시 카드
        result_card = NeumorphicCard()
        result_layout = QVBoxLayout(result_card)
        
        # 결과 레이블 - 이 부분을 수정하여 결과 텍스트를 변경할 수 있습니다
        self.result_label = QLabel("")
        self.result_label.setStyleSheet("""
            font-size: 32px;
            font-weight: bold;
            color: #2DCE89;
        """)
        self.result_label.setAlignment(Qt.AlignCenter)
        
        result_layout.addWidget(self.result_label)
        
        # 버튼 레이아웃
        button_layout = QHBoxLayout()
        
        # 버튼 - 이 부분을 수정하여 버튼 텍스트를 변경할 수 있습니다
        self.back_btn = NeumorphicButton("뒤로")
        self.back_btn.clicked.connect(self.go_back)
        
        self.start_btn = NeumorphicButton("시작", PRIMARY_COLOR)
        self.start_btn.clicked.connect(self.start_game)
        
        button_layout.addWidget(self.back_btn)
        button_layout.addStretch()
        button_layout.addWidget(self.start_btn)
        
        # 레이아웃에 카드 추가
        layout.addWidget(title_card)
        layout.addLayout(top_layout)
        layout.addWidget(game_card)
        layout.addWidget(result_card)
        layout.addLayout(button_layout)
        
        # 초기에 카운트다운 레이블 숨기기
        countdown_card.setVisible(False)
        
        # 카운트다운 카드 저장
        self.countdown_card = countdown_card
        
    def set_user(self, username, password):
        """사용자 정보 설정"""
        self.username = username
        self.password = password
        self.user_dir = get_user_dir(username, password)
        
        # 모델 로드
        model_path = os.path.join(self.user_dir, MODEL_FILE)
        self.predictor = EMGPredictor(model_path)
        
        # 감도 설정 로드
        config = load_user_config(username, password)
        self.emg_processor.set_sensitivity(config["sensitivity"])
        
    def start(self):
        """게임 시작 준비"""
        self.reset_game()
        
        # 시리얼 스레드 시작
        if self.serial_thread:
            self.serial_thread.disconnect_serial()
            self.serial_thread = None
            
        self.serial_thread = SerialThread()
        self.serial_thread.data_received.connect(self.process_data)
        self.serial_thread.start()
        
        # 상태 업데이트 타이머 시작
        self.detection_timer.start(100)
        
    def stop(self):
        """게임 중지"""
        self.countdown_timer.stop()
        self.game_timer.stop()
        self.detection_timer.stop()
        
        if self.serial_thread:
            self.serial_thread.disconnect_serial()
            self.serial_thread = None
            
    def reset_game(self):
        """게임 상태 초기화"""
        self.game_state = 0
        self.player_score = 0
        self.computer_score = 0
        self.round = 0
        self.player_choice = None
        self.computer_choice = None
        self.game_result = None
        self.current_attacker = None
        
        # UI 업데이트
        self.player_score_label.setText(f"플레이어: {self.player_score}")
        self.computer_score_label.setText(f"컴퓨터: {self.computer_score}")
        self.round_label.setText(f"라운드: {self.round}/{self.max_rounds}")
        
        self.player_choice_label.setText("?")
        self.computer_choice_label.setText("?")
        
        self.player_status.setText("대기중")
        self.computer_status.setText("대기중")
        
        self.result_label.setText("")
        self.status_label.setText("게임을 시작하려면 '시작' 버튼을 누르세요")
        
        self.countdown_card.setVisible(False)
        self.start_btn.setEnabled(True)
        self.start_btn.setText("시작")
        
    def process_data(self, s1, s2):
        """EMG 데이터 처리"""
        result = self.emg_processor.process_data(s1, s2)
        
        # 동작 감지 중일 때만 처리
        if (self.game_state == 1 or self.game_state == 2) and isinstance(result, tuple) and result[0] == "movement_completed":
            sequence = result[1]
            # 동작 예측
            prediction, confidence = self.predictor.predict(sequence)
            
            # 충분한 신뢰도를 가진 가위/바위/보 동작만 처리
            if prediction in [LABEL_SCISSORS, LABEL_ROCK, LABEL_PAPER] and confidence >= 0.5:
                self.player_choice = prediction
                self.update_player_choice()
                
    def update_player_choice(self):
        """플레이어 선택 업데이트"""
        if self.player_choice == LABEL_SCISSORS:
            self.player_choice_label.setText("✌️")
        elif self.player_choice == LABEL_ROCK:
            self.player_choice_label.setText("✊")
        elif self.player_choice == LABEL_PAPER:
            self.player_choice_label.setText("✋")
            
    def update_computer_choice(self):
        """컴퓨터 선택 업데이트"""
        if self.computer_choice == LABEL_SCISSORS:
            self.computer_choice_label.setText("✌️")
        elif self.computer_choice == LABEL_ROCK:
            self.computer_choice_label.setText("✊")
        elif self.computer_choice == LABEL_PAPER:
            self.computer_choice_label.setText("✋")
            
    def start_game(self):
        """게임 시작"""
        if self.game_state == 0:
            # 새 라운드 시작
            self.round += 1
            self.round_label.setText(f"라운드: {self.round}/{self.max_rounds}")
            
            # 첫 판은 가위바위보
            self.game_state = 1
            self.player_choice = None
            self.computer_choice = None
            self.current_attacker = None
            
            self.player_status.setText("")
            self.computer_status.setText("")
            
            self.start_countdown()
            self.start_btn.setEnabled(False)
            
    def start_countdown(self):
        """카운트다운 시작"""
        self.countdown_value = 3
        self.countdown_label.setText(str(self.countdown_value))
        self.countdown_card.setVisible(True)
        self.status_label.setText("준비...")
        self.countdown_timer.start(1000)
        
    def update_countdown(self):
        """카운트다운 업데이트"""
        self.countdown_value -= 1
        self.countdown_label.setText(str(self.countdown_value))
        
        if self.countdown_value <= 0:
            self.countdown_timer.stop()
            
            # 카운트다운 끝, 게임 시작
            self.countdown_card.setVisible(False)
            
            if self.game_state == 1:
                self.status_label.setText("가위... 바위... 보!")
            else:
                self.status_label.setText("묵... 찌... 빠!")
                
            # 플레이어 선택 초기화
            self.player_choice = None
            self.player_choice_label.setText("?")
            
            # 컴퓨터 선택 초기화
            self.computer_choice = None
            self.computer_choice_label.setText("?")
            
            # 게임 타이머 시작 (5초 후 결과 확인)
            self.game_timer.start(5000)
            
    def update_game(self):
        """게임 상태 업데이트"""
        self.game_timer.stop()
        
        # 컴퓨터 선택
        self.computer_choice = get_computer_choice()
        self.update_computer_choice()
        
        # 플레이어가 선택하지 않았으면 랜덤 선택
        if self.player_choice is None:
            self.status_label.setText("시간 초과! 랜덤으로 선택됩니다.")
            self.player_choice = get_computer_choice()  # 랜덤 선택
            self.update_player_choice()
            
        # 가위바위보 결과 확인
        if self.game_state == 1:
            # 첫 판 가위바위보
            result = determine_winner(self.player_choice, self.computer_choice)
            
            if result == RESULT_WIN:
                # 플레이어 승리, 공격자가 됨
                self.current_attacker = True
                self.player_status.setText("공격자")
                self.computer_status.setText("수비자")
                self.result_label.setText("선공권 획득!")
                self.result_label.setStyleSheet("font-size: 32px; font-weight: bold; color: #2DCE89;")
                
            elif result == RESULT_LOSE:
                # 컴퓨터 승리, 공격자가 됨
                self.current_attacker = False
                self.player_status.setText("수비자")
                self.computer_status.setText("공격자")
                self.result_label.setText("후공!")
                self.result_label.setStyleSheet("font-size: 32px; font-weight: bold; color: #F5365C;")
                
            else:
                # 무승부, 재경기
                self.result_label.setText("무승부! 다시 시작")
                self.result_label.setStyleSheet("font-size: 32px; font-weight: bold; color: #FB8C00;")
                QTimer.singleShot(2000, self.start_countdown)
                return
                
            # 묵찌빠 시작
            self.game_state = 2
            QTimer.singleShot(2000, self.start_countdown)
            
        elif self.game_state == 2:
            # 묵찌빠 진행
            
            # 선택 표시
            player_choice_text = get_label_name(self.player_choice)
            computer_choice_text = get_label_name(self.computer_choice)
            self.status_label.setText(f"{player_choice_text} vs {computer_choice_text}")
            
            # 같은 손 모양인 경우 공격자 승리
            if self.player_choice == self.computer_choice:
                if self.current_attacker:
                    # 플레이어 승리
                    self.player_score += 1
                    self.player_score_label.setText(f"플레이어: {self.player_score}")
                    self.result_label.setText("플레이어 승리!")
                    self.result_label.setStyleSheet("font-size: 32px; font-weight: bold; color: #2DCE89;")
                else:
                    # 컴퓨터 승리
                    self.computer_score += 1
                    self.computer_score_label.setText(f"컴퓨터: {self.computer_score}")
                    self.result_label.setText("컴퓨터 승리!")
                    self.result_label.setStyleSheet("font-size: 32px; font-weight: bold; color: #F5365C;")
                    
                # 라운드 종료, 다음 라운드 확인
                self.check_game_end()
                
            else:
                # 다른 손 모양, 가위바위보 규칙으로 공격자 결정
                result = determine_winner(self.player_choice, self.computer_choice)
                
                if result == RESULT_WIN:
                    # 플레이어 공격자로 전환
                    self.current_attacker = True
                    self.player_status.setText("공격자")
                    self.computer_status.setText("수비자")
                    self.result_label.setText("공격권 획득!")
                    self.result_label.setStyleSheet("font-size: 32px; font-weight: bold; color: #2DCE89;")
                    
                elif result == RESULT_LOSE:
                    # 컴퓨터 공격자로 전환
                    self.current_attacker = False
                    self.player_status.setText("수비자")
                    self.computer_status.setText("공격자")
                    self.result_label.setText("수비!")
                    self.result_label.setStyleSheet("font-size: 32px; font-weight: bold; color: #F5365C;")
                    
                else:
                    # 묵찌빠에서 무승부는 공격자 유지
                    if self.current_attacker:
                        self.result_label.setText("무승부! 공격 유지")
                    else:
                        self.result_label.setText("무승부! 수비 유지")
                        
                self.result_label.setStyleSheet("font-size: 32px; font-weight: bold; color: #FB8C00;")
                
                # 묵찌빠 계속 진행
                QTimer.singleShot(2000, self.start_countdown)
                
    def check_game_end(self):
        """게임 종료 확인"""
        # 최대 라운드 도달 확인
        if self.round >= self.max_rounds or self.player_score > self.max_rounds // 2 or self.computer_score > self.max_rounds // 2:
            # 게임 종료
            self.game_state = 3
            
            if self.player_score > self.computer_score:
                self.status_label.setText("게임 종료 - 플레이어 승리!")
            elif self.player_score < self.computer_score:
                self.status_label.setText("게임 종료 - 컴퓨터 승리!")
            else:
                self.status_label.setText("게임 종료 - 무승부!")
                
            self.start_btn.setText("다시 시작")
            self.start_btn.setEnabled(True)
            self.game_state = 0
        else:
            # 다음 라운드 준비
            QTimer.singleShot(2000, self.next_round)
            
    def next_round(self):
        """다음 라운드 시작"""
        self.start_game()
        
    def update_detection(self):
        """동작 감지 상태 업데이트"""
        # 게임 상태에 따른 처리
        if self.game_state == 1 or self.game_state == 2:
            # 게임 중일 때만 플레이어 선택 표시
            if self.player_choice:
                self.update_player_choice()
                
    def go_back(self):
        """뒤로 가기"""
        self.stop()
        self.back_to_main.emit()

# ------------------------------------------------------------------------
# 메인 실행
# ------------------------------------------------------------------------
def main():
    app = QApplication(sys.argv)
    
    # 스플래시 화면 표시
    splash_pix = QPixmap(os.path.join(IMAGE_DIR, "logo.png"))
    splash = QSplashScreen(splash_pix)
    splash.show()
    app.processEvents()
    
    # 메인 윈도우 생성
    window = EMGGameApplication()
    
    # 창 표시
    window.show()
    splash.finish(window)
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
