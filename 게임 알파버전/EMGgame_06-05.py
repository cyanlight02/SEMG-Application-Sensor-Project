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
import pygame
import socket
import platform
import threading
import subprocess
try:
    import win32gui
    import win32con
    import win32process # EMGgame_05-21.py 에 있었던 import
    PYWIN32_AVAILABLE = True
except ImportError:
    PYWIN32_AVAILABLE = False
    print("경고: pywin32 라이브러리를 찾을 수 없습니다. Unity 창 숨김/표시 기능이 비활성화됩니다.")
from datetime import datetime
from collections import deque
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QPushButton, QLabel, QComboBox, 
                            QGridLayout, QMessageBox, QProgressBar, QFrame,
                            QSplashScreen, QSizePolicy, QLineEdit, QDialog,
                            QStackedWidget, QTabWidget, QTextEdit, QDoubleSpinBox,
                            QGraphicsDropShadowEffect, QLayout, QRadioButton, QTableWidgetItem, QTableWidget,QSlider, QStyle, QGroupBox)
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
COMMON_DATA_DIR = os.path.join(BASE_DIR, "common_data")  # 공통 데이터 폴더

# 모델 관련 상수
SEQUENCE_LENGTH = 200  # 트랜스포머 모델용 시퀀스 길이
BUFFER_SIZE = 500      # 감지용 롤링 버퍼 크기

# 데이터 수집 관련 상수
DETECTION_WINDOW = 20  # 동작 감지에 사용할 윈도우 크기
DETECTION_THRESHOLD = 6  # Z-동작 감지 임계값 (필요시 수정하여 감도 조절)
COOLDOWN_PERIOD = 2    # 동작 저장 후 대기 시간(초)
POST_CAPTURE = 30      # 동작 종료 후 추가 캡처 프레임 수
TREND_WINDOW = 10      # 추세 감지에 사용할 윈도우 크기
TREND_THRESHOLD = 4    # 추세 감지 임계값 (필요시 수정하여 감도 조절)
DISPERSION_THRESHOLD = 4 # 분산 감지 임계값 (필요시 수정하여 감도 조절)

# 동작 라벨 상수
LABEL_IDLE = 0       # 대기상태
LABEL_SCISSORS = 11   # 가위
LABEL_ROCK = 12      # 바위
LABEL_PAPER = 13      # 보

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
        self.setFixedSize(250, 80)
        
        # 임시 로고 이미지 생성 및 설정
        logo_file = os.path.join(IMAGE_DIR, "neumorphic_logo.png")
        if not os.path.exists(logo_file):
            self.create_logo(logo_file)
        
        self.setPixmap(QPixmap(logo_file))
        self.setScaledContents(True)
    
    def create_logo(self, file_path):
        """네오모피즘 스타일의 로고 이미지 생성"""
        pixmap = QPixmap(250, 80)
        pixmap.fill(Qt.transparent)
        
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # 배경 (부드러운 곡선)
        painter.setBrush(QColor(BG_COLOR))
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(0, 0, 250, 80, 20, 20)
        
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
    dir_name = f"{username}_{password}"
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
        "data_count": {
            str(LABEL_IDLE): 0,
            str(LABEL_SCISSORS): 0, 
            str(LABEL_ROCK): 0, 
            str(LABEL_PAPER): 0
        }
    }
    
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            # 누락된 설정이 있으면 기본값으로 보완
            for key, value in default_config.items():
                if key not in config:
                    config[key] = value
            
            # 대기상태 라벨이 없으면 추가
            if "data_count" in config and str(LABEL_IDLE) not in config["data_count"]:
                config["data_count"][str(LABEL_IDLE)] = 0
                
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
        "idle": os.path.join(IMAGE_DIR, "idle.png"),
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
        "idle": ((100, 100, 100), (300, 300)),
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
# 필요한 파일들을 생성하는 함수 추가
def create_audio_directory():
    """오디오 디렉토리 생성 및 안내"""
    audio_dir = os.path.join(BASE_DIR, "audio")
    ensure_dir_exists(audio_dir)
    
    # 필요한 음성 파일 목록
    required_files = [
        "ready.wav",           # 준비 음성
        "rockpaperscissors.wav", # 가위바위보 통합 음성
        "mukjjippa.wav",       # 묵찌빠 통합 음성
        "win.wav",             # 승리
        "lose.wav",            # 패배
        "draw.wav",            # 무승부
        "mukjjippa_ready.wav"  # 묵찌빠 준비
    ]
    
    # 없는 파일들을 확인하고 안내 파일 생성
    missing_files = []
    for file_name in required_files:
        file_path = os.path.join(audio_dir, file_name)
        if not os.path.exists(file_path):
            missing_files.append(file_name)
    
    if missing_files:
        # 안내 파일 생성
        info_file = os.path.join(audio_dir, "음성파일_안내.txt")
        with open(info_file, 'w', encoding='utf-8') as f:
            f.write("EMG 게임 시스템 음성 파일 안내\n")
            f.write("=" * 40 + "\n\n")
            f.write("다음 음성 파일들을 이 폴더에 추가하세요:\n\n")
            f.write("필수 음성 파일:\n")
            f.write("- ready.wav : 게임 준비 음성\n")
            f.write("- rockpaperscissors.wav : '가위바위보' 통합 음성\n")
            f.write("- mukjjippa.wav : '묵찌빠' 통합 음성\n")
            f.write("- win.wav : 승리 음성\n")
            f.write("- lose.wav : 패배 음성\n")
            f.write("- draw.wav : 무승부 음성\n")
            f.write("- mukjjippa_ready.wav : 묵찌빠 준비 음성\n\n")
            f.write("누락된 파일들:\n")
            for file_name in missing_files:
                f.write(f"- {file_name}\n")
            f.write("\n주의사항:\n")
            f.write("- 음성 파일은 WAV 형식이어야 합니다.\n")
            f.write("- 파일이 없으면 해당 음성이 재생되지 않습니다.\n")
            f.write("- 가위바위보 음성은 약 1-2초 길이로 준비하세요.")
    
    return audio_dir
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
    if label == LABEL_IDLE:
        return "대기"
    elif label == LABEL_SCISSORS:
        return "가위"
    elif label == LABEL_ROCK:
        return "바위"
    elif label == LABEL_PAPER:
        return "보"
    else:
        return f"알 수 없음({label})"
# ------------------------------------------------------------------------
# 음성 기능 모델 정의
# ------------------------------------------------------------------------
class GameAudioManager:
    """게임 음성 관리 클래스"""
    def __init__(self):
        try:
            pygame.mixer.init()
            self.audio_enabled = True
            self.audio_dir = os.path.join(BASE_DIR, "audio")
            ensure_dir_exists(self.audio_dir)
        except:
            self.audio_enabled = False
            print("pygame 초기화 실패 - 음성 기능이 비활성화됩니다.")
    
    def play_sound(self, sound_name):
        """음성 파일 재생"""
        if not self.audio_enabled:
            return
            
        try:
            sound_path = os.path.join(self.audio_dir, f"{sound_name}.wav")
            if os.path.exists(sound_path):
                sound = pygame.mixer.Sound(sound_path)
                sound.play()
            else:
                print(f"음성 파일 없음: {sound_path}")
        except Exception as e:
            print(f"음성 재생 오류: {e}")
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

# ------------------------------------------------------------------------!

class UnityWindowHandlerThread(QThread):
       finished_handling = pyqtSignal()
       error_occurred = pyqtSignal(str)
   
       def __init__(self, unity_exe_name, window_title_keyword, logo_duration_seconds):
           super().__init__()
           self.unity_exe_name = unity_exe_name
           self.window_title_keyword = window_title_keyword
           # EMGgame_05-21.py에서는 hide_duration_seconds 였으나, 05-22의 스타일을 따라 logo_duration_seconds 로 유지
           self.logo_duration_seconds = logo_duration_seconds 
           self.process = None
           self.unity_hwnd = None # 찾은 Unity 창의 핸들 저장
   
       def _find_unity_window(self):
           if not PYWIN32_AVAILABLE or not self.process: 
               return False
   
           target_pid = self.process.pid if hasattr(self.process, 'pid') else None
           attempts = 0
           max_attempts = 20 # 10초간 시도
           while attempts < max_attempts:
               temp_hwnd = None
               def callback(hwnd, extra_args):
                   nonlocal temp_hwnd
                   # EMGgame_05-21.py의 로직 사용
                   if not win32gui.IsWindowVisible(hwnd) or self.window_title_keyword.lower() not in win32gui.GetWindowText(hwnd).lower():
                       return True
                   
                   if target_pid:
                       _, found_pid = win32process.GetWindowThreadProcessId(hwnd)
                       if found_pid == target_pid:
                           temp_hwnd = hwnd
                           return False 
                   else: 
                       temp_hwnd = hwnd
                       return False
                   return True
               
               win32gui.EnumWindows(callback, None)
   
               if temp_hwnd:
                   self.unity_hwnd = temp_hwnd
                   print(f"Unity 창 찾음: HWND={self.unity_hwnd}, Title='{win32gui.GetWindowText(self.unity_hwnd)}'")
                   return True
               
               time.sleep(0.5)
               attempts += 1
           print(f"'{self.window_title_keyword}' 키워드에 해당하는 Unity 창을 시간 내에 찾지 못했습니다.")
           return False
   
       def _send_window_to_back(self):
           if not PYWIN32_AVAILABLE or not self.unity_hwnd:
               return False
           try:
               win32gui.SetWindowPos(self.unity_hwnd,
                                     win32con.HWND_BOTTOM, 
                                     0, 0, 0, 0, 
                                     win32con.SWP_NOMOVE | win32con.SWP_NOSIZE | win32con.SWP_NOACTIVATE)
               print(f"Unity 창(HWND: {self.unity_hwnd})을 뒤로 보냈습니다.")
               return True
           except Exception as e:
               print(f"Unity 창(HWND: {self.unity_hwnd}) 뒤로 보내기 오류: {e}")
               return False
   
       def _bring_window_to_front_and_activate(self):
           if not PYWIN32_AVAILABLE or not self.unity_hwnd:
               return False
           try:
               win32gui.ShowWindow(self.unity_hwnd, win32con.SW_RESTORE)
               win32gui.SetWindowPos(self.unity_hwnd,
                                     win32con.HWND_TOP, 
                                     0, 0, 0, 0,
                                     win32con.SWP_NOMOVE | win32con.SWP_NOSIZE)
               win32gui.SetForegroundWindow(self.unity_hwnd)
               print(f"Unity 창(HWND: {self.unity_hwnd})을 앞으로 가져와 활성화했습니다.")
               return True
           except Exception as e:
               print(f"Unity 창(HWND: {self.unity_hwnd}) 앞으로 가져오기/활성화 오류: {e}")
               try:
                   win32gui.ShowWindow(self.unity_hwnd, win32con.SW_SHOW)
               except:
                   pass
               return False
   
       def run(self):
           if not PYWIN32_AVAILABLE:
               self.error_occurred.emit("pywin32 라이브러리가 없어 창 제어를 수행할 수 없습니다.")
               try:
                   self.process = subprocess.Popen([self.unity_exe_name])
                   time.sleep(self.logo_duration_seconds) 
                   self.finished_handling.emit()
               except Exception as e:
                   self.error_occurred.emit(f"{self.unity_exe_name} 실행 실패: {e}")
               return
   
           try:
               self.process = subprocess.Popen([self.unity_exe_name])
               time.sleep(0.5) 
   
               if self._find_unity_window():
                   if self._send_window_to_back():
                       time.sleep(self.logo_duration_seconds)
                       self._bring_window_to_front_and_activate()
                   else:
                       self.error_occurred.emit("Unity 창을 뒤로 보내는 데 실패했습니다.")
                       time.sleep(self.logo_duration_seconds)
                       if self.unity_hwnd: 
                            self._bring_window_to_front_and_activate()
               else:
                   self.error_occurred.emit(f"'{self.window_title_keyword}' 창을 찾지 못했습니다. Unity는 실행되었을 수 있습니다.")
                   time.sleep(self.logo_duration_seconds)
   
               self.finished_handling.emit()
   
           except FileNotFoundError:
               self.error_occurred.emit(f"오류: {self.unity_exe_name}을(를) 찾을 수 없습니다.")
           except Exception as e:
               self.error_occurred.emit(f"Unity 창 처리 중 오류: {e}")
               if self.unity_hwnd:
                   self._bring_window_to_front_and_activate()
   
       def get_process(self):
           return self.process
   
       def terminate_process(self):
           # 이 스레드는 시작 시에만 관여하므로, 종료는 is_process_running과 taskkill 조합으로 처리
           pass
# ------------------------------------------------------------------------!

# ------------------------------------------------------------------------
# 시리얼 통신 스레드 - 싱글톤 패턴 적용
# ------------------------------------------------------------------------
class SerialThread(QThread):
    """
    개선된 SerialThread 클래스 - 싱글톤 제거, 재사용 가능하도록 수정
    """
    data_received = pyqtSignal(float, float)
    error_occurred = pyqtSignal(str)
    connection_status = pyqtSignal(bool, str)
    connected = pyqtSignal()
    disconnected = pyqtSignal()
    
    def __init__(self, port=DEFAULT_SERIAL_PORT, baud_rate=DEFAULT_BAUD_RATE, parent=None):
        super().__init__(parent)
        self.port = port
        self.baud_rate = baud_rate
        self.running = False
        self.ser = None
        self.auto_connect = True
        self.connection_type = "usb"
        self._stop_flag = False  # 명시적인 중지 플래그 추가
    
    def set_port(self, port):
        """통신 포트 설정"""
        self.port = port
        
    def set_baud_rate(self, baud_rate):
        """통신 속도 설정"""
        self.baud_rate = baud_rate

    def set_connection_type(self, connection_type):
        """연결 방식 설정 (usb 또는 bluetooth)"""
        self.connection_type = connection_type
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
    def connect_serial(self):
        """시리얼 연결 시작"""
        if self.ser:
            self.disconnect_serial()
            
        try:
            print(f"Connecting: {self.connection_type} mode on {self.port} at {self.baud_rate}bps")
            
            self.ser = serial.Serial(self.port, self.baud_rate, timeout=1)
            
            # 연결 후 안정화 대기
            wait_time = 2.0 if self.connection_type == "bluetooth" else 1.0
            time.sleep(wait_time)
            
            self.ser.reset_input_buffer()
            self.connection_status.emit(True, f"연결 성공: {self.port} ({self.connection_type})")
            self.connected.emit()
            
            return True
        except Exception as e:
            self.connection_status.emit(False, f"연결 실패: {str(e)}")
            self.ser = None
            return False
    
    def disconnect_serial(self):
        """시리얼 연결 종료"""
        self._stop_flag = True  # 중지 플래그 설정
        self.running = False
        
        if self.ser:
            try:
                self.ser.close()
            except:
                pass
            self.ser = None
            self.disconnected.emit()
            self.connection_status.emit(False, "연결 종료")
    
    def stop(self):
        """스레드 중지"""
        self._stop_flag = True
        self.running = False
        self.wait()  # 스레드가 완전히 종료될 때까지 대기
            
    def run(self):
        """스레드 실행"""
        self._stop_flag = False  # 중지 플래그 초기화
        
        # 자동 연결 모드인 경우 연결 시도
        if self.auto_connect and not self.ser:
            if not self.connect_serial():
                self.error_occurred.emit("자동 연결 실패")
                return
                
        if not self.ser:
            self.error_occurred.emit("연결되지 않음")
            return
            
        self.running = True
        
        while self.running and not self._stop_flag:
            try:
                if self.ser and self.ser.in_waiting > 0:
                    line = self.ser.readline().decode('utf-8', errors='replace').strip()
                    
                    if line == "USB_CONNECTED" or line == "BT_CONNECTED" or line == "EMG_BT_START":
                        continue
                    
                    if ',' in line:
                        try:
                            values = line.split(',')
                            if len(values) >= 2:
                                s1 = float(values[0])
                                s2 = float(values[1])
                                self.data_received.emit(s1, s2)
                        except ValueError:
                            continue
                        
                wait_time = 0.005 if self.connection_type == "bluetooth" else 0.001
                time.sleep(wait_time)
                
            except Exception as e:
                if not self._stop_flag:  # 정상 종료가 아닌 경우에만 에러 표시
                    self.error_occurred.emit(f"데이터 읽기 오류: {str(e)}")
                self.running = False
                break
        
        # 스레드 종료 시 연결 정리
        if not self._stop_flag:  # 비정상 종료인 경우
            self.disconnect_serial()
# ------------------------------------------------------------------------
# EMG 데이터 처리 클래스
# ------------------------------------------------------------------------
class EMGProcessor:
    """
    EMGProcessor 클래스: EMG 센서 데이터를 처리하고 동작을 감지하는 클래스
    - 신호 처리와 동작 감지 로직을 담당
    - 감도 설정을 통해 동작 감지 민감도를 조절할 수 있음
    - 사용자 설정에 따른 임계값 적용
    """
    def __init__(self, sensitivity=1.0, config=None):
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
        
        # 기본 설정
        self.sensitivity = sensitivity
        
        # 동작 감지 설정
        self.detection_threshold = DETECTION_THRESHOLD
        self.trend_threshold = TREND_THRESHOLD
        self.dispersion_threshold = DISPERSION_THRESHOLD
        self.cooldown_period = COOLDOWN_PERIOD
        # 설정 저장 (이 줄 추가)
        self.config = config or {}

        # 설정이 있으면 적용
        if config:
            self.apply_config(config)
        
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
    
    def apply_config(self, config):
        """사용자 설정 적용"""
        # 설정에서 가져온 임계값 적용
        self.sensitivity = config.get("sensitivity", self.sensitivity)
        self.detection_threshold = config.get("detection_threshold", self.detection_threshold)
        self.trend_threshold = config.get("trend_threshold", self.trend_threshold)
        self.dispersion_threshold = config.get("dispersion_threshold", self.dispersion_threshold)
        self.cooldown_period = config.get("cooldown_period", self.cooldown_period)
        
    
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
            if time.time() - self.cooldown_start >= self.cooldown_period:
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
        adjusted_threshold = self.detection_threshold / self.sensitivity
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
            trend_detected = (abs(s1_slope) > self.trend_threshold * self.baseline_std[0] or 
                             abs(s2_slope) > self.trend_threshold * self.baseline_std[1])
        else:
            trend_detected = False
        
        # 3. 분산 기반 감지
        recent_window = recent_data[-5:]  # 가장 최근 5개 프레임
        s1_recent = [x[0] for x in recent_window]
        s2_recent = [x[1] for x in recent_window]
        
        s1_var = np.var(s1_recent)
        s2_var = np.var(s2_recent)
        
        variance_detected = (s1_var > (self.baseline_std[0] * self.dispersion_threshold * self.sensitivity) or 
                            s2_var > (self.baseline_std[1] * self.dispersion_threshold * self.sensitivity))
        
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
        adjusted_threshold = (self.detection_threshold/2) / self.sensitivity
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
            
            if (s1_var > (self.baseline_std[0] * self.dispersion_threshold * self.sensitivity) or 
                s2_var > (self.baseline_std[1] * self.dispersion_threshold * self.sensitivity)):
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
    - 가위/바위/보/대기 동작을 인식하는 모델 로드
    - 예측 기능 제공
    """
    def __init__(self, model_path=None):
        self.model = None
        self.scaler = None
        self.config = None
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
            
    def load_model(self, model_path):
        """학습된 모델 로드 - 통합 모델 전용"""
        try:
            checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
            self.config = checkpoint['config']
            
            # 통합 모델: 0, 1-8, 11-13 라벨 지원
            num_classes = 14  # 최소 14개 클래스 필요 (0-13)

            # 모델 생성
            self.model = EMGTransformer(
                input_dim=2,  # EMG 센서 2개
                d_model=self.config['d_model'],
                nhead=self.config['nhead'],
                num_layers=self.config['num_layers'],
                dim_feedforward=self.config['dim_feedforward'],
                num_classes=num_classes,
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
# 위젯 기본 클래스 - 공통 기능 통합
# ------------------------------------------------------------------------
class BaseEMGWidget(QWidget):
    """
    BaseEMGWidget 클래스: 모든 EMG 관련 위젯의 기본 클래스
    - EMG 데이터 처리, 시리얼 연결, 로깅 등 공통 기능 제공
    """
    back_to_main = pyqtSignal()  # 뒤로가기 시그널
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # 상태 변수
        self.username = ""
        self.password = ""
        self.user_dir = ""
        
        # 신뢰도 임계값
        self.confidence_threshold = 0.5
        
        # EMG 프로세서 및 예측기
        self.emg_processor = EMGProcessor()
        self.predictor = None
        
        # 위젯 활성화 상태
        self.is_active = False
        
        # 로그 영역 (여러 위젯에서 공통으로 사용)
        self.log_text = None
        
    def set_user(self, username, password):
        """사용자 정보 설정"""
        self.username = username
        self.password = password
        self.user_dir = get_user_dir(username, password)
        
        # 사용자 설정 로드
        config = load_user_config(self.username, self.password)
        
        # EMG 프로세서 설정 적용
        self.emg_processor.apply_config(config)
        
        # 신뢰도 임계값 설정
        self.confidence_threshold = config.get("confidence_threshold", 0.5)
        
    def create_log_area(self, parent_layout):
        """로그 영역 생성 - 공통 UI 요소"""
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
        parent_layout.addWidget(log_card)
        
        return self.log_text
        
    def add_log(self, message):
        """로그 메시지 추가"""
        if not self.log_text:
            print(f"로그: {message}")  # 로그 위젯이 없으면 콘솔에 출력
            return
            
        timestamp = datetime.now().strftime('%H:%M:%S')
        self.log_text.append(f"[{timestamp}] {message}")
        # 스크롤을 항상 최하단으로
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
        
    def subscribe_to_serial(self):
        """전역 시리얼 스레드에 구독"""
        main_app = self.window()
        if hasattr(main_app, "subscribe_widget"):
            # 이미 구독되어 있는지 확인
            if self in main_app.connected_widgets:
                return True
            return main_app.subscribe_widget(self)
        return False

    def unsubscribe_from_serial(self):
        """전역 시리얼 스레드 구독 해제 - 개선된 버전"""
        main_app = self.window()
        if hasattr(main_app, "unsubscribe_widget"):
            # 구독되어 있는 경우에만 해제
            if self in main_app.connected_widgets:
                return main_app.unsubscribe_widget(self)
        return False
    
    def start(self):
        """위젯 시작 - 자식 클래스에서 오버라이드"""
        pass
        
    def stop(self):
        """위젯 중지 - 자식 클래스에서 오버라이드"""
        pass
        
    def process_data(self, s1, s2):
        """EMG 데이터 처리 - 자식 클래스에서 오버라이드"""
        pass
        
    def go_back(self):
        """뒤로 가기"""
        self.stop()
        self.back_to_main.emit()


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
        title_layout.setContentsMargins(20, 10, 20, 10)
        title_layout.setSpacing(5)
        # 로고 위젯
        self.logo = LogoWidget()
        
        # 타이틀 레이블
        self.title_label = QLabel("근전도 신호에 기반한\n입력 프로그램")
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
        self.message_label = QLabel("아이디와 비밀번호는 데이터 저장 위치로 사용됩니다.")
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
# IP 주소 입력 다이얼로그
# ------------------------------------------------------------------------
class IPAddressDialog(QDialog):
    """IP 주소 입력 다이얼로그"""
    
    def __init__(self, default_ip="172.20.10.8", parent=None):
        super().__init__(parent)
        self.setWindowTitle("안드로이드 연결 설정")
        self.setModal(True)
        self.ip_address = default_ip
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(30, 30, 30, 30)
        layout.setSpacing(20)
        
        # 다이얼로그 스타일
        self.setStyleSheet(f"background-color: {BG_COLOR};")
        
        # 타이틀
        title = QLabel("안드로이드 연결 설정")
        title.setStyleSheet(f"color: {TEXT_COLOR}; font-size: 24px; font-weight: bold;")
        title.setAlignment(Qt.AlignCenter)
        
        # 설명 텍스트
        description = QLabel("안드로이드 디바이스의 IP 주소를 입력하세요.\n포트는 5000번을 사용합니다.")
        description.setStyleSheet(f"color: {TEXT_COLOR}; font-size: 16px;")
        description.setAlignment(Qt.AlignCenter)
        
        # IP 주소 입력 필드
        ip_label = QLabel("IP 주소:")
        ip_label.setStyleSheet(f"color: {TEXT_COLOR}; font-size: 16px; font-weight: bold;")
        
        # IP 입력 프레임
        ip_frame = NeumorphicCard(inset=True)
        ip_layout = QVBoxLayout(ip_frame)
        ip_layout.setContentsMargins(10, 5, 10, 5)
        
        self.ip_edit = QLineEdit()
        self.ip_edit.setText(self.ip_address)
        self.ip_edit.setPlaceholderText("예: 192.168.1.100")
        self.ip_edit.setStyleSheet(f"""
            QLineEdit {{
                background-color: transparent;
                border: none;
                color: {TEXT_COLOR};
                font-size: 16px;
                padding: 5px;
            }}
        """)
        self.ip_edit.setFixedHeight(40)
        
        ip_layout.addWidget(self.ip_edit)
        
        # 버튼들
        button_layout = QHBoxLayout()
        
        cancel_btn = NeumorphicButton("취소")
        cancel_btn.clicked.connect(self.reject)
        
        connect_btn = NeumorphicButton("연결", PRIMARY_COLOR)
        connect_btn.clicked.connect(self.accept_connection)
        
        button_layout.addWidget(cancel_btn)
        button_layout.addWidget(connect_btn)
        
        # 레이아웃에 추가
        layout.addWidget(title)
        layout.addWidget(description)
        layout.addSpacing(10)
        layout.addWidget(ip_label)
        layout.addWidget(ip_frame)
        layout.addSpacing(20)
        layout.addLayout(button_layout)
        
        # 다이얼로그 크기 설정
        self.setFixedSize(400, 300)
        
    def accept_connection(self):
        """연결 승인 처리"""
        ip_text = self.ip_edit.text().strip()
        if not ip_text:
            QMessageBox.warning(self, "입력 오류", "IP 주소를 입력해주세요.")
            return
            
        # 간단한 IP 형식 검증
        parts = ip_text.split('.')
        if len(parts) != 4:
            QMessageBox.warning(self, "입력 오류", "올바른 IP 주소 형식을 입력해주세요.\n예: 192.168.1.100")
            return
            
        try:
            for part in parts:
                num = int(part)
                if not (0 <= num <= 255):
                    raise ValueError
        except ValueError:
            QMessageBox.warning(self, "입력 오류", "올바른 IP 주소 형식을 입력해주세요.\n예: 192.168.1.100")
            return
            
        self.ip_address = ip_text
        self.accept()
        
    def get_ip_address(self):
        """입력된 IP 주소 반환"""
        return self.ip_address


# ------------------------------------------------------------------------
# 안드로이드 연결 위젯
# ------------------------------------------------------------------------
class AndroidConnectionWidget(BaseEMGWidget):
    """
    AndroidConnectionWidget 클래스: EMG 센서로 감지된 동작을 안드로이드로 전송하는 위젯
    - 안드로이드 디바이스와 소켓 통신으로 연결
    - EMG 동작 인식을 통해 라벨값을 전송
    - 네오모피즘 디자인 적용
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # 연결 상태
        self.is_connected = False
        self.is_running = False
        self.android_ip = "192.168.0.66"  # 기본 IP
        self.android_port = 5000
        self.socket_connection = None
        
        # 감지된 동작 타이머
        self.detection_timer = QTimer(self)
        self.detection_timer.timeout.connect(self.update_detection)
        
        # 동작 실행 쿨다운 타이머
        self.action_cooldown = False
        self.cooldown_timer = QTimer(self)
        self.cooldown_timer.timeout.connect(self.reset_action_cooldown)
        
        # 연결 상태 확인 타이머
        self.connection_check_timer = QTimer(self)
        self.connection_check_timer.timeout.connect(self.check_connection_status)
        
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
        
        # 제목 텍스트
        self.title_label = QLabel("안드로이드 연결")
        self.title_label.setStyleSheet(f"color: {TEXT_COLOR}; font-size: 28px; font-weight: bold;")
        self.title_label.setAlignment(Qt.AlignCenter)
        
        title_layout.addWidget(self.title_label)
        
        # 연결 정보 카드
        connection_card = NeumorphicCard()
        connection_layout = QVBoxLayout(connection_card)
        connection_layout.setContentsMargins(20, 15, 20, 15)
        connection_layout.setSpacing(15)
        
        # 연결 상태 표시
        status_layout = QHBoxLayout()
        
        status_label = QLabel("연결 상태:")
        status_label.setStyleSheet(f"color: {TEXT_COLOR}; font-size: 16px; font-weight: bold;")
        
        self.status_indicator = QLabel("●")
        self.status_indicator.setStyleSheet("color: red; font-size: 20px;")
        
        self.status_text = QLabel("연결되지 않음")
        self.status_text.setStyleSheet(f"color: {DANGER_COLOR}; font-size: 16px; font-weight: bold;")
        
        status_layout.addWidget(status_label)
        status_layout.addWidget(self.status_indicator)
        status_layout.addWidget(self.status_text)
        status_layout.addStretch()
        
        # IP 주소 표시
        ip_layout = QHBoxLayout()
        
        ip_label = QLabel("대상 IP:")
        ip_label.setStyleSheet(f"color: {TEXT_COLOR}; font-size: 16px; font-weight: bold;")
        
        self.ip_display = QLabel(self.android_ip)
        self.ip_display.setStyleSheet(f"color: {PRIMARY_COLOR}; font-size: 16px; font-weight: bold;")
        
        self.change_ip_btn = NeumorphicButton("IP 변경")
        self.change_ip_btn.setFixedHeight(35)
        self.change_ip_btn.clicked.connect(self.change_ip_address)
        
        ip_layout.addWidget(ip_label)
        ip_layout.addWidget(self.ip_display)
        ip_layout.addStretch()
        ip_layout.addWidget(self.change_ip_btn)
        
        connection_layout.addLayout(status_layout)
        connection_layout.addLayout(ip_layout)
        
        # 상태 정보 카드
        info_card = NeumorphicCard()
        info_layout = QHBoxLayout(info_card)
        info_layout.setContentsMargins(20, 15, 20, 15)
        
        # 설명 텍스트
        info_text = QLabel("EMG 센서를 통해 감지된 동작의 라벨값을 안드로이드로 전송합니다.\n"
                          "연결 후 '시작' 버튼을 눌러 동작 감지를 활성화하세요.")
        info_text.setStyleSheet(f"color: {TEXT_COLOR}; font-size: 14px;")
        info_text.setAlignment(Qt.AlignLeft)
        info_text.setWordWrap(True)
        
        # 현재 감지 상태
        detection_layout = QVBoxLayout()
        
        detection_title = QLabel("감지된 동작")
        detection_title.setStyleSheet(f"color: {TEXT_COLOR}; font-size: 14px; font-weight: bold;")
        detection_title.setAlignment(Qt.AlignCenter)
        
        self.detected_label = QLabel("없음")
        self.detected_label.setStyleSheet(f"color: {PRIMARY_COLOR}; font-size: 18px; font-weight: bold;")
        self.detected_label.setAlignment(Qt.AlignCenter)
        
        detection_layout.addWidget(detection_title)
        detection_layout.addWidget(self.detected_label)
        
        info_layout.addWidget(info_text, 2)
        info_layout.addLayout(detection_layout, 1)
        
        # 동작 가이드 카드
        guide_card = NeumorphicCard()
        guide_layout = QVBoxLayout(guide_card)
        guide_layout.setContentsMargins(20, 15, 20, 15)
        
        # 가이드 제목
        guide_title = QLabel("동작 라벨 가이드")
        guide_title.setStyleSheet(f"color: {TEXT_COLOR}; font-size: 16px; font-weight: bold;")
        guide_title.setAlignment(Qt.AlignCenter)
        
        # 가이드 텍스트
        guide_text = QLabel("• 동작 1~8: 각각의 커스텀 동작\n"
                           "• 가위 (라벨 11): 가위 동작\n"
                           "• 바위 (라벨 12): 바위 동작\n"
                           "• 보 (라벨 13): 보 동작\n"
                           "• 대기 상태 (라벨 0)는 전송하지 않습니다.")
        guide_text.setStyleSheet(f"color: {TEXT_COLOR}; font-size: 14px; line-height: 1.5;")
        guide_text.setAlignment(Qt.AlignLeft)
        
        guide_layout.addWidget(guide_title)
        guide_layout.addWidget(guide_text)
        
        # 버튼 레이아웃
        button_layout = QHBoxLayout()
        
        # 뒤로 버튼
        self.back_btn = NeumorphicButton("뒤로")
        self.back_btn.clicked.connect(self.go_back)
        
        # 연결 버튼
        self.connect_btn = NeumorphicButton("연결", SECONDARY_COLOR)
        self.connect_btn.clicked.connect(self.toggle_connection)
        
        # 시작 버튼
        self.start_btn = NeumorphicButton("시작", PRIMARY_COLOR)
        self.start_btn.clicked.connect(self.toggle_detection)
        self.start_btn.setEnabled(False)
        
        button_layout.addWidget(self.back_btn)
        button_layout.addStretch()
        button_layout.addWidget(self.connect_btn)
        button_layout.addWidget(self.start_btn)
        
        # 로그 영역
        self.create_log_area(layout)
        
        # 레이아웃에 위젯 추가
        layout.addWidget(title_card)
        layout.addWidget(connection_card)
        layout.addWidget(info_card)
        layout.addWidget(guide_card)
        layout.addLayout(button_layout)
        
    def set_user(self, username, password):
        """사용자 정보 설정"""
        super().set_user(username, password)
        
        # 통합 모델 로드
        model_path = os.path.join(self.user_dir, "emg_model.pth")
        
        if os.path.exists(model_path):
            self.predictor = EMGPredictor(model_path)
            self.add_log("통합 모델 로드 완료.")
            print(f"모델 로드 성공: {model_path}")
        else:
            self.add_log("통합 동작 인식 모델이 없습니다. 먼저 데이터를 수집하고 학습해주세요.")
            self.connect_btn.setEnabled(False)
            
    def change_ip_address(self):
        """IP 주소 변경"""
        dialog = IPAddressDialog(self.android_ip, self)
        if dialog.exec_() == QDialog.Accepted:
            self.android_ip = dialog.get_ip_address()
            self.ip_display.setText(self.android_ip)
            self.add_log(f"IP 주소가 {self.android_ip}로 변경되었습니다.")
            
            # 연결되어 있으면 재연결
            if self.is_connected:
                self.disconnect_android()
                
    def toggle_connection(self):
        """연결/해제 토글"""
        if not self.is_connected:
            self.connect_android()
        else:
            self.disconnect_android()
            
    def connect_android(self):
        """안드로이드 연결"""
        try:
            import socket
            
            self.add_log(f"안드로이드 연결 시도: {self.android_ip}:{self.android_port}")
            
            # 소켓 생성 및 연결
            self.socket_connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket_connection.settimeout(5)  # 5초 타임아웃
            self.socket_connection.connect((self.android_ip, self.android_port))
            
            # 연결 성공
            self.is_connected = True
            self.update_connection_status()
            self.add_log("안드로이드 연결 성공!")
            
            # 연결 상태 확인 타이머 시작
            self.connection_check_timer.start(3000)  # 3초마다 확인
            
        except Exception as e:
            self.add_log(f"안드로이드 연결 실패: {str(e)}")
            if self.socket_connection:
                try:
                    self.socket_connection.close()
                except:
                    pass
                self.socket_connection = None
                
    def disconnect_android(self):
        """안드로이드 연결 해제"""
        # 감지 중지
        if self.is_running:
            self.stop_detection()
            
        self.is_connected = False
        self.connection_check_timer.stop()
        
        if self.socket_connection:
            try:
                self.socket_connection.close()
            except:
                pass
            self.socket_connection = None
            
        self.update_connection_status()
        self.add_log("안드로이드 연결 해제됨.")
        
    def update_connection_status(self):
        """연결 상태 UI 업데이트"""
        if self.is_connected:
            self.status_indicator.setStyleSheet("color: green; font-size: 20px;")
            self.status_text.setText("연결됨")
            self.status_text.setStyleSheet(f"color: {SECONDARY_COLOR}; font-size: 16px; font-weight: bold;")
            self.connect_btn.setText("연결 해제")
            self.connect_btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: {DANGER_COLOR};
                    color: white;
                    border-radius: 12px;
                    padding: 10px 20px;
                    text-align: center;
                }}
                QPushButton:hover {{
                    background-color: {self.darken_color(DANGER_COLOR, 10)};
                }}
            """)
            self.start_btn.setEnabled(True)
        else:
            self.status_indicator.setStyleSheet("color: red; font-size: 20px;")
            self.status_text.setText("연결되지 않음")
            self.status_text.setStyleSheet(f"color: {DANGER_COLOR}; font-size: 16px; font-weight: bold;")
            self.connect_btn.setText("연결")
            self.connect_btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: {SECONDARY_COLOR};
                    color: white;
                    border-radius: 12px;
                    padding: 10px 20px;
                    text-align: center;
                }}
                QPushButton:hover {{
                    background-color: {self.lighten_color(SECONDARY_COLOR, 10)};
                }}
            """)
            self.start_btn.setEnabled(False)
            
    def darken_color(self, color, amount=20):
        """색상을 어둡게 만듭니다."""
        c = QColor(color)
        h, s, l, a = c.getHslF()
        l = max(0.0, l - amount / 100)
        c.setHslF(h, s, l, a)
        return c.name()
        
    def lighten_color(self, color, amount=20):
        """색상을 밝게 만듭니다."""
        c = QColor(color)
        h, s, l, a = c.getHslF()
        l = min(1.0, l + amount / 100)
        c.setHslF(h, s, l, a)
        return c.name()
        
    def toggle_detection(self):
        """감지 시작/중지 토글"""
        if not self.is_running:
            self.start_detection()
        else:
            self.stop_detection()
            
    def start_detection(self):
        """동작 감지 시작"""
        if not self.is_connected or not self.predictor:
            return
            
        # EMG 프로세서 설정 적용 (이 부분이 누락되어 있었음!)
        config = load_user_config(self.username, self.password)
        self.emg_processor.apply_config(config)
        
        # 전역 시리얼 스레드에 구독
        if self.subscribe_to_serial():
            self.is_running = True
            self.start_btn.setText("중지")
            self.start_btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: {DANGER_COLOR};
                    color: white;
                    border-radius: 12px;
                    padding: 10px 20px;
                    text-align: center;
                }}
                QPushButton:hover {{
                    background-color: {self.darken_color(DANGER_COLOR, 10)};
                }}
            """)
            
            # 타이머 시작
            self.detection_timer.start(100)
            
            self.add_log("동작 감지 시작. EMG 센서로 동작을 취해보세요.")
        else:
            self.add_log("EMG 센서 연결 실패. 연결을 확인하세요.")
            
    def stop_detection(self):
        """동작 감지 중지"""
        self.is_running = False
        self.unsubscribe_from_serial()
        self.detection_timer.stop()
        
        self.start_btn.setText("시작")
        self.start_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {PRIMARY_COLOR};
                color: white;
                border-radius: 12px;
                padding: 10px 20px;
                text-align: center;
            }}
            QPushButton:hover {{
                background-color: {self.lighten_color(PRIMARY_COLOR, 10)};
            }}
        """)
        
        self.add_log("동작 감지 중지됨.")
        
    def process_data(self, s1, s2):
        """EMG 데이터 처리"""
        if not self.is_running or not self.predictor or not self.is_connected:
            return
            
        result = self.emg_processor.process_data(s1, s2)
        
        if isinstance(result, tuple) and result[0] == "movement_completed":
            sequence = result[1]
            
            # 동작 예측
            prediction, confidence = self.predictor.predict(sequence)
            
            # 디버깅 로그
            print(f"안드로이드 연결: 예측 결과 - 라벨 {prediction}, 신뢰도 {confidence:.2f}")
            
            # 충분한 신뢰도를 가진 동작만 처리
            if confidence >= self.confidence_threshold:
                # 대기 상태(라벨 0)는 무시
                if prediction == LABEL_IDLE:
                    self.detected_label.setText("감지된 동작: 대기 상태")
                    print(f"대기 상태 감지됨 (신뢰도: {confidence:.2f})")
                    return
                
                # 유효한 라벨 처리 (1-8, 11-13)
                if (1 <= prediction <= 8) or prediction in (LABEL_SCISSORS, LABEL_ROCK, LABEL_PAPER):
                    # 라벨 표시
                    if prediction == LABEL_SCISSORS:
                        self.detected_label.setText("감지된 동작: 가위")
                    elif prediction == LABEL_ROCK:
                        self.detected_label.setText("감지된 동작: 바위")
                    elif prediction == LABEL_PAPER:
                        self.detected_label.setText("감지된 동작: 보")
                    else:
                        self.detected_label.setText(f"감지된 동작: 동작 {prediction}")
                    
                    # 안드로이드로 라벨값 전송 (쿨다운 체크)
                    if not self.action_cooldown:
                        self.send_label_to_android(prediction)
                        self.action_cooldown = True
                        self.cooldown_timer.start(1000)  # 1초 쿨다운
                    
    def send_label_to_android(self, label):
        """안드로이드로 라벨값 전송"""
        if not self.is_connected or not self.socket_connection:
            return
            
        try:
            # 테스트 코드와 동일한 형식 사용
            message = f"{label}\n"
            self.socket_connection.sendall(message.encode('utf-8'))
            
            self.add_log(f"라벨 {label} 전송 완료")
            
        except Exception as e:
            self.add_log(f"라벨 전송 실패: {str(e)}")
            # 연결 오류 시 재연결 시도
            self.disconnect_android()
            
    def reset_action_cooldown(self):
        """동작 실행 쿨다운 초기화"""
        self.action_cooldown = False
        self.cooldown_timer.stop()
        
    def check_connection_status(self):
        """연결 상태 확인"""
        if not self.is_connected or not self.socket_connection:
            return
            
        try:
            # 더미 데이터 전송으로 연결 상태 확인
            self.socket_connection.settimeout(1)
            # 실제로는 전송하지 않고 소켓 상태만 확인
        except:
            self.add_log("연결이 끊어진 것 같습니다. 재연결을 시도하세요.")
            self.disconnect_android()
            
    def update_detection(self):
        """감지 상태 업데이트"""
        # 현재는 특별한 작업이 필요 없음
        pass
        
    def stop(self):
        """위젯 중지"""
        if self.is_running:
            self.stop_detection()
        if self.is_connected:
            self.disconnect_android()
            
    def go_back(self):
        """뒤로 가기"""
        self.stop()
        super().go_back()

# ------------------------------------------------------------------------
# 게임 선택 대화상자
# ------------------------------------------------------------------------
class GameSelectionDialog(QDialog):
    """게임 선택 대화상자"""
    game_selected = pyqtSignal(int)  # 0: 가위바위보, 1: 묵찌빠
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("게임 선택")
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(30, 30, 30, 30)
        layout.setSpacing(20)
        
        # 다이얼로그 스타일
        self.setStyleSheet(f"background-color: {BG_COLOR};")
        
        # 타이틀
        title = QLabel("게임을 선택하세요")
        title.setStyleSheet(f"color: {TEXT_COLOR}; font-size: 24px; font-weight: bold;")
        title.setAlignment(Qt.AlignCenter)
        
        # 게임 선택 버튼
        rps_btn = NeumorphicButton("가위바위보 게임", PRIMARY_COLOR)
        rps_btn.clicked.connect(lambda: self.select_game(0))
        
        muk_btn = NeumorphicButton("묵찌빠 게임", SECONDARY_COLOR)
        muk_btn.clicked.connect(lambda: self.select_game(1))
        
        cancel_btn = NeumorphicButton("취소")
        cancel_btn.clicked.connect(self.reject)
        
        # 레이아웃에 추가
        layout.addWidget(title)
        layout.addSpacing(10)
        layout.addWidget(rps_btn)
        layout.addWidget(muk_btn)
        layout.addSpacing(10)
        layout.addWidget(cancel_btn)
        
        # 다이얼로그 크기 설정
        self.setFixedSize(400, 350)
        
    def select_game(self, game_index):
        """게임 선택 처리"""
        self.game_selected.emit(game_index)
        self.accept()


# ------------------------------------------------------------------------
# 메인 메뉴 위젯
# ------------------------------------------------------------------------
class MainMenuWidget(QWidget):
    """
    MainMenuWidget 클래스: 메인 메뉴 화면을 제공하는 위젯
    - 각 기능으로의 접근 제공
    - 네오모피즘 디자인이 적용됨
    """
    # 신호 정의
    show_guide = pyqtSignal()
    show_data_collection = pyqtSignal()
    show_game_selection = pyqtSignal()  # 게임 선택 대화상자 표시
    show_screen_control = pyqtSignal()  # 화면 제어 모드 표시
    show_recording = pyqtSignal()       # 촬영 모드 표시
    show_settings = pyqtSignal()        # 설정 표시
    show_android_connection = pyqtSignal()
    show_hand_model = pyqtSignal()
    def __init__(self, parent=None):
        super().__init__(parent)
        self.username = ""
        self.init_ui()
        self.unity_server_socket = None
        self.unity_client_socket = None
        self.unity_server_thread = None
        self.unity_process = None # Unity 프로세스 저장
        self.unity_window_handler_thread = None
        
    def init_ui(self):
        """UI 초기화 - 안드로이드 연결 버튼 추가"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(40, 40, 40, 40)
        layout.setSpacing(30)
        
        # 배경 설정
        self.setStyleSheet(f"background-color: {BG_COLOR};")
        
        # 환영 카드
        welcome_card = NeumorphicCard()
        # QGridLayout 사용하여 더 정확한 위치 제어
        welcome_layout = QGridLayout(welcome_card)
        welcome_layout.setContentsMargins(15, 10, 15, 10)
        
        # 환영 메시지
        self.welcome_label = QLabel("어서오세요, 사용자님! 저랑 게임하실래요?")
        self.welcome_label.setStyleSheet(f"color: {TEXT_COLOR}; font-size: 28px; font-weight: bold;")
        self.welcome_label.setAlignment(Qt.AlignCenter)
        
        # 버튼들을 수직으로 배치
        button_widget = QWidget()
        buttons_layout = QVBoxLayout(button_widget)
        buttons_layout.setContentsMargins(0, 0, 0, 0)  # 마진 제거
        buttons_layout.setSpacing(10)  # 버튼 간 간격
        
        # 안드로이드 연결 버튼
        self.android_btn = NeumorphicButton("모바일", ACCENT_COLOR)
        self.android_btn.setFixedSize(150, 50)
        self.android_btn.clicked.connect(self.on_android_connection_clicked)
        
        # 손모델 보기 버튼
        self.hand_model_btn = NeumorphicButton("손모델", SECONDARY_COLOR)
        self.hand_model_btn.setFixedSize(150, 50)
        self.hand_model_btn.clicked.connect(self.on_hand_model_clicked)
        
        # 버튼 레이아웃에 버튼 추가
        buttons_layout.addWidget(self.android_btn)
        buttons_layout.addWidget(self.hand_model_btn)
        
        # Grid에 환영 메시지와 버튼 위젯 추가
        # (행, 열, 행 범위, 열 범위)
        welcome_layout.addWidget(self.welcome_label, 0, 0, 1, 3)  # 0행 0열, 1행 3열 차지
        welcome_layout.addWidget(button_widget, 0, 2, 1, 1, Qt.AlignBottom | Qt.AlignRight)  # 0행 2열, 우측 하단 정렬
        
        # 열 비율 설정 (0열:1, 1열:3, 2열:1)
        welcome_layout.setColumnStretch(0, 1)
        welcome_layout.setColumnStretch(1, 3)
        welcome_layout.setColumnStretch(2, 1)
        
        # 메인 레이아웃에 추가
        layout.addWidget(welcome_card)
        
        # 메뉴 그리드 생성 (기존과 동일)
        menu_grid = QGridLayout()
        menu_grid.setSpacing(25)
        
        # 1. 사용 가이드
        guide_card = self._create_menu_card(
            "📋", "사용 가이드", "EMG 센서 설정 및\n기본 사용법 안내", 
            self.on_guide_clicked
        )
        
        # 2. 동작 학습
        training_card = self._create_menu_card(
            "🔄", "동작 학습 하러가기", "가위/바위/보/대기 동작\n인식 데이터 수집", 
            self.on_training_clicked
        )
        
        # 3. 게임 모드
        game_card = self._create_menu_card(
            "🎮", "게임 모드", "EMG 센서로 하는\n다양한 게임", 
            self.on_game_clicked, SECONDARY_COLOR
        )
        
        # 4. 화면 제어 모드
        screen_control_card = self._create_menu_card(
            "🖥️", "화면 제어 모드", "EMG 센서로\n화면 제어하기", 
            self.on_screen_control_clicked
        )
        
        # 5. 촬영 모드
        recording_card = self._create_menu_card(
            "📷", "촬영 모드", "EMG 센서로\n사진/동영상 촬영 제어", 
            self.on_recording_clicked
        )
        
        # 6. 설정
        settings_card = self._create_menu_card(
            "⚙️", "설정", "프로그램 설정 및\n사용자 관리", 
            self.on_settings_clicked
        )
        
        # 그리드에 카드 추가
        menu_grid.addWidget(guide_card, 0, 0)
        menu_grid.addWidget(training_card, 0, 1)
        menu_grid.addWidget(game_card, 0, 2)
        menu_grid.addWidget(screen_control_card, 1, 0)
        menu_grid.addWidget(recording_card, 1, 1)
        menu_grid.addWidget(settings_card, 1, 2)
        
        # 메인 레이아웃에 추가
        layout.addWidget(welcome_card)
        layout.addLayout(menu_grid)

#----------------------------------------------------------------------!        
    def start_unity_server(self):
        if self.unity_server_socket is None:
            try:
                self.unity_server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                # 포트가 이미 사용 중일 경우를 대비해 SO_REUSEADDR 옵션 설정
                self.unity_server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                self.unity_server_socket.bind(("127.0.0.1", 65432))
                self.unity_server_socket.listen(1)
                print("Unity 통신 서버 시작: 127.0.0.1:65432")
                self.add_log_to_main_window("손 모델 제어: Unity 통신 서버 시작 (127.0.0.1:65432)")
                
                self.unity_server_thread = threading.Thread(target=self.unity_client_accept_thread, daemon=True)
                self.unity_server_thread.start()
                return True
            except Exception as e:
                print(f"Unity 통신 서버 시작 실패: {e}")
                self.add_log_to_main_window(f"손 모델 제어: Unity 통신 서버 시작 실패 - {e}")
                self.unity_server_socket = None
                return False
        return True # 이미 실행 중인 경우도 성공으로 간주

    def unity_client_accept_thread(self):
        while self.unity_server_socket:
            try:
                client, addr = self.unity_server_socket.accept()
                if self.unity_client_socket: # 이미 연결된 클라이언트가 있다면 이전 연결 종료
                    try:
                        self.unity_client_socket.close()
                    except:
                        pass
                self.unity_client_socket = client
                print(f"Unity 클라이언트 연결됨: {addr}")
                self.add_log_to_main_window(f"손 모델 제어: Unity 클라이언트 연결됨 - {addr}")

                # 클라이언트로부터 데이터 수신 스레드 (필요하다면 구현, 현재는 Unity에서 데이터 받는 로직 없음)
                # threading.Thread(target=self.unity_client_recv_thread, args=(client,), daemon=True).start()

            except socket.error as e:
                # 서버 소켓이 닫혔을 때 발생하는 오류 (예: 프로그램 종료 시)는 무시
                if self.unity_server_socket and self.unity_server_socket.fileno() == -1 : # 소켓이 닫혔는지 확인
                    print("Unity 통신 서버 소켓이 닫혔습니다 (accept 중단).")
                    break
                print(f"Unity 클라이언트 accept 오류: {e}")
                # self.add_log_to_main_window(f"손 모델 제어: Unity 클라이언트 accept 오류 - {e}") # 너무 많은 로그 방지
                time.sleep(1) # 잠시 대기 후 다시 시도
            except Exception as e:
                print(f"Unity 클라이언트 accept 중 예외 발생: {e}")
                break # 루프 종료

    def handle_unity_client_disconnection(self):
        if self.unity_client_socket:
            try:
                self.unity_client_socket.close()
            except Exception as e:
                print(f"Unity 클라이언트 소켓 닫기 오류: {e}")
            self.unity_client_socket = None
            print("Unity 클라이언트 연결이 끊어졌습니다.")
            self.add_log_to_main_window("손 모델 제어: Unity 클라이언트 연결 끊김")
    
    def on_unity_task_completed(self, button_text_to_restore):
        button_to_change = self.hand_model_btn
        button_to_change.setText(button_text_to_restore)
        button_to_change.setEnabled(True)
        print(f"손모델(Unity) 버튼 텍스트 '{button_text_to_restore}'로 복원, 활성화됨 - 실제 버튼 참조 후 활성화 필요")

        log_msg = "Unity 창 처리 완료."
        if hasattr(self.window(), 'add_log'):
            self.window().add_log(log_msg)
        else:
            print(log_msg)
        if self.unity_window_handler_thread: 
                self.unity_window_handler_thread.quit()
                self.unity_window_handler_thread = None
    
    def on_unity_task_error(self, button_text_to_restore, error_message):
        button_to_change = self.hand_model_btn
        button_to_change.setText(button_text_to_restore)
        button_to_change.setEnabled(True)
        print(f"손모델(Unity) 버튼 텍스트 '{button_text_to_restore}'로 복원 (오류), 활성화됨 - 실제 버튼 참조 후 활성화 필요")

        log_msg = f"Unity 창 처리 오류: {error_message}"
        if hasattr(self.window(), 'add_log'):
            self.window().add_log(log_msg)
        else:
            print(log_msg)
        if self.unity_window_handler_thread: 
                self.unity_window_handler_thread.quit()
                self.unity_window_handler_thread = None
    
    def is_process_running(self, process_name):
        try:
            process = subprocess.Popen(['tasklist'], stdout=subprocess.PIPE, text=True, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            output, _ = process.communicate()
            if process_name.lower() in output.lower():
                return True
        except FileNotFoundError:
            log_msg = f"'{process_name}' 실행 상태 확인 불가: tasklist 명령을 찾을 수 없습니다."
            if hasattr(self.window(), 'add_log'):
                self.window().add_log(log_msg)
            else:
                print(log_msg)
            return False
        except Exception as e:
            log_msg = f"'{process_name}' 실행 상태 확인 중 오류: {e}"
            if hasattr(self.window(), 'add_log'):
                self.window().add_log(log_msg)
            else:
                print(log_msg)
            return False
        return False        
    def process_data(self, s1, s2):
        """EMG 데이터 수신 처리 (위젯 구독용)"""
        # 손모델 모드일 때만 처리
        if hasattr(self, '_is_subscribed_for_hand_model') and self._is_subscribed_for_hand_model:
            self.process_emg_for_hand_model(s1, s2)
#----------------------------------------------------------------------!
        
    def _create_menu_card(self, icon_text, title, desc, click_handler, button_color=None):
        """메뉴 카드 생성 헬퍼 함수"""
        card = NeumorphicCard()
        layout = QVBoxLayout(card)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # 아이콘
        icon = QLabel(icon_text)
        icon.setStyleSheet("font-size: 60px;")
        icon.setAlignment(Qt.AlignCenter)
        
        # 타이틀
        title_label = QLabel(title)
        title_label.setStyleSheet(f"color: {TEXT_COLOR}; font-size: 24px; font-weight: bold;")
        title_label.setAlignment(Qt.AlignCenter)
        
        # 설명
        desc_label = QLabel(desc)
        desc_label.setStyleSheet("color: #8898aa; font-size: 18px;")
        desc_label.setAlignment(Qt.AlignCenter)
        
        # 버튼
        btn = NeumorphicButton("시작하기" if button_color is None else "플레이", button_color)
        btn.clicked.connect(click_handler)
        
        layout.addWidget(icon)
        layout.addWidget(title_label)
        layout.addWidget(desc_label)
        layout.addWidget(btn)
        
        return card
        
    def set_username(self, username):
        """사용자 이름 설정"""
        self.username = username
        # 환영 메시지 텍스트 수정 - 이 부분을 수정하여 환영 메시지를 변경할 수 있습니다
        self.welcome_label.setText(f"어서오세요, {username}님! 저랑 게임하실래요?")
    
    def on_guide_clicked(self):
        """사용 가이드 버튼 클릭 처리"""
        self.show_guide.emit()
    
    def on_training_clicked(self):
        """동작 학습 버튼 클릭 처리 - 화면 크기 변경 기능 제거"""
        self.show_data_collection.emit()
    
    def on_game_clicked(self):
        """게임 버튼 클릭 처리"""
        self.show_game_selection.emit()
    
    def on_screen_control_clicked(self):
        """화면 제어 버튼 클릭 처리 - 화면 크기 변경 기능 제거"""
        self.show_screen_control.emit()
    
    def on_recording_clicked(self):
        """촬영 모드 버튼 클릭 처리"""
        self.show_recording.emit()
    
    def on_settings_clicked(self):
        """설정 버튼 클릭 처리"""
        self.show_settings.emit()
    def on_android_connection_clicked(self):
        """안드로이드 연결 버튼 클릭 처리"""
        self.show_android_connection.emit()
    def on_hand_model_clicked(self):
        """손모델 보기 버튼 클릭 처리"""
        self.show_hand_model.emit()
# ------------------------------------------------------------------------!        
    def on_hand_model_clicked(self):
        sender_button = self.sender() # 클릭된 버튼 가져오기

        # 1. EMGHand.exe 프로세스가 실행 중인지 확인
        if self.is_process_running("EMGHand.exe"):
            try:
                if platform.system() == "Windows":
                    # 2. 실행 중이면 프로세스 종료
                    # CREATE_NO_WINDOW 플래그는 Windows에서 cmd 창이 뜨는 것을 방지합니다.
                    subprocess.run(
                        ["taskkill", "/F", "/IM", "EMGHand.exe"], 
                        check=True, 
                        creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0) # Windows 전용 플래그
                    )
                    self.add_log_to_main_window("'EMGHand.exe'가 실행 중이어서 종료했습니다.")
                    
                    # 3. Unity 핸들러 스레드 정리 (존재하고 실행 중일 경우)
                    if hasattr(self, 'unity_handler_thread') and self.unity_handler_thread and self.unity_handler_thread.isRunning():
                        self.unity_handler_thread.terminate_process() # 스레드가 관리하는 프로세스 종료 시도
                        self.unity_handler_thread.quit()
                        self.unity_handler_thread.wait(2000) # 최대 2초 대기
                        if self.unity_handler_thread.isRunning(): # 여전히 실행 중이면 강제 종료 고려 (덜 안전)
                            self.unity_handler_thread.terminate()
                            self.unity_handler_thread.wait()
                        self.unity_handler_thread = None
                        self.add_log_to_main_window("Unity 핸들러 스레드를 종료했습니다.")

                    # 4. Unity 서버 스레드 정리 (존재하고 실행 중일 경우)
                    if hasattr(self, 'unity_server_thread') and self.unity_server_thread:
                        if self.unity_server_thread.is_alive(): # is_running 대신 is_alive() 사용
                            self.add_log_to_main_window("Unity 서버 스레드가 활성 상태이므로 stop_unity_server()를 호출합니다.")
                            self.stop_unity_server() # 기존 서버 중지 함수 호출
                            self.add_log_to_main_window("Unity 서버 스레드 종료 시도 완료.") # stop_unity_server 내부 로그에 따라 중복될 수 있음
                        else:
                            # 스레드 객체는 있지만 실행 중이 아닐 때 stop_unity_server()를 호출하여
                            # 관련 리소스(예: 소켓, 참조)가 정리되도록 할 수 있습니다.
                            # stop_unity_server()는 내부적으로 is_alive()를 체크합니다.
                            self.add_log_to_main_window("Unity 서버 스레드가 활성 상태는 아니지만, 정리를 위해 stop_unity_server()를 호출합니다.")
                            self.stop_unity_server()
                            self.add_log_to_main_window("Unity 서버 스레드(비활성) 정리 시도 완료.")
                    else:
                        self.add_log_to_main_window("Unity 서버 스레드 객체가 존재하지 않습니다.")
                    
                    # 5. EMG 데이터 구독 해제
                    main_app = self.window()
                    if isinstance(main_app, EMGGameApplication):
                        if hasattr(self, '_is_subscribed_for_hand_model') and self._is_subscribed_for_hand_model:
                            try:
                                main_app.unsubscribe_widget(self)  # 변경: self.process_emg_for_hand_model → self
                                self.add_log_to_main_window("손 모델 EMG 데이터 구독을 해제했습니다.")
                                self._is_subscribed_for_hand_model = False
                            except Exception as e:
                                self.add_log_to_main_window(f"손 모델 EMG 구독 해제 중 오류 발생 (무시 가능): {e}")
                    
                    # 6. 버튼 상태 업데이트 (예: 텍스트 변경, 다시 활성화 등)
                    if sender_button and hasattr(sender_button, 'original_text'):
                        sender_button.setText(sender_button.original_text)
                        sender_button.setEnabled(True)
                    # 또는 특정 상태를 나타내는 텍스트로 변경
                    # if sender_button:
                    #    sender_button.setText("손 모델 시작")

                    #QMessageBox.information(self, "알림", "'EMGHand.exe'가 종료되었습니다.")
                    return # 함수 실행을 여기서 마침 (다시 켜는 로직을 실행하지 않음)

                else:
                    # Windows가 아닌 경우에 대한 처리 (예: 로그만 남기거나, 사용자에게 알림)
                    self.add_log_to_main_window("Windows가 아닌 시스템에서는 'EMGHand.exe'를 자동으로 종료할 수 없습니다.")
                    QMessageBox.warning(self, "지원되지 않는 기능", "현재 운영체제에서는 실행 중인 프로그램을 자동으로 종료할 수 없습니다.")
                    # 이 경우, 기존 로직을 계속 진행할지 아니면 여기서 멈출지 결정해야 합니다.
                    # 사용자가 직접 끄도록 안내하고 return 할 수도 있습니다.
                    return

            except subprocess.CalledProcessError as e:
                # taskkill 명령어 실행 실패
                self.add_log_to_main_window(f"'EMGHand.exe' 종료 중 오류 발생: {e}")
                QMessageBox.warning(self, "오류", f"'EMGHand.exe'를 종료하는 데 실패했습니다: {e}\n수동으로 종료해주세요.")
                return # 오류 발생 시에도 일단 함수 종료하여 다시 켜는 로직 방지
            except Exception as e:
                # 기타 예외 처리
                self.add_log_to_main_window(f"손 모델 종료 처리 중 예외 발생: {e}")
                QMessageBox.critical(self, "예외 발생", f"손 모델 종료 중 예상치 못한 오류가 발생했습니다: {e}")
                return # 예외 발생 시에도 일단 함수 종료
        self.add_log_to_main_window("손 모델 버튼 클릭됨")
        # 1. Unity 통신 서버 시작
        if not self.unity_server_thread:
            self.add_log_to_main_window("손 모델 제어: Unity 서버 시작 시도")
            self.start_unity_server()
        else:
            self.add_log_to_main_window("손 모델 제어: Unity 서버 이미 시작됨")

        # 2. Unity 앱 실행 (또는 포커스)
        unity_exe_name = "EMGHand.exe"
        script_dir = os.path.dirname(os.path.abspath(__file__))
        unity_exe_path = os.path.join(script_dir, unity_exe_name)
        unity_app_dir = script_dir

        if not self.is_process_running(unity_exe_name):
            self.add_log_to_main_window(f"손 모델 제어: Unity 프로세스 시작 시도 - {unity_exe_name}")
            try:
                # Unity 앱을 시작할 때 작업 디렉토리를 Unity 앱의 폴더로 설정
                subprocess.Popen(unity_exe_path, cwd=unity_app_dir)
                self.add_log_to_main_window("Unity 손 모델 앱을 시작합니다. 연결 대기 중...")
            except Exception as e:
                self.add_log_to_main_window(f"Unity 손 모델 앱 시작 실패: {e}")
                QMessageBox.critical(self, "오류", f"Unity 손 모델 앱({unity_exe_path})을 실행할 수 없습니다.\n실행 파일이 스크립트와 동일한 폴더에 있는지 확인하세요.\n오류: {e}")
                if self.unity_server_thread:
                    self.stop_unity_server()
                return
        else:
            self.add_log_to_main_window("손 모델 제어: Unity 프로세스 이미 실행 중")
            # Unity 창을 앞으로 가져오는 로직 추가 가능 (선택 사항)
            # self.focus_unity_window() # 별도 구현 필요


        # 3. EMG 프로세서 및 예측기 초기화 (MainMenuWidget 전용)
        main_app = self.window()
        current_username = None
        current_password = None

        if hasattr(main_app, 'username') and hasattr(main_app, 'password'):
            current_username = main_app.username
            current_password = main_app.password
        else:
            self.add_log_to_main_window("오류: 사용자 이름/비밀번호를 메인 앱에서 찾을 수 없습니다.")
            QMessageBox.critical(self, "오류", "사용자 정보를 찾을 수 없어 EMG 처리를 시작할 수 없습니다.")
            return

        if not hasattr(self, 'emg_processor_hand_model') or self.emg_processor_hand_model is None:
            try:
                config = load_user_config(current_username, current_password)
                self.emg_processor_hand_model = EMGProcessor(config=config)
                self.add_log_to_main_window("MainMenuWidget: 손 모델용 EMGProcessor 초기화 완료.")
            except Exception as e:
                self.add_log_to_main_window(f"MainMenuWidget: 손 모델용 EMGProcessor 초기화 실패: {e}")
                self.emg_processor_hand_model = None # 실패 시 None으로 설정
        
        if not hasattr(self, 'predictor_hand_model') or self.predictor_hand_model is None:
            user_dir = get_user_dir(current_username, current_password)
            # 통합 모델 "emg_model.pth"을 사용한다고 가정합니다.
            # 만약 다른 모델 (예: "action_model.pth")을 사용해야 한다면 경로를 수정하세요.
            model_path = os.path.join(user_dir, "emg_model.pth")
            if os.path.exists(model_path):
                self.predictor_hand_model = EMGPredictor(model_path=model_path)
                if self.predictor_hand_model.model is None: # 모델 로드 실패 내부 확인
                    self.predictor_hand_model = None
                    self.add_log_to_main_window(f"경고: 손 모델 제어용 모델 로드 실패 ({model_path}).")
                    QMessageBox.warning(self, "모델 로드 실패", f"손 모델 제어를 위한 사용자 모델({os.path.basename(model_path)})을 로드하지 못했습니다.")
                else:
                    self.add_log_to_main_window(f"MainMenuWidget: 손 모델용 EMGPredictor 초기화 완료 ({model_path}).")
            else:
                self.predictor_hand_model = None
                self.add_log_to_main_window(f"경고: 손 모델 제어용 사용자 모델 없음 ({model_path}).")
                QMessageBox.warning(self, "모델 없음", f"손 모델 제어를 위한 사용자 모델({os.path.basename(model_path)})이 없습니다. 데이터 수집 및 학습을 먼저 진행해주세요.")

        # 4. EMG 데이터 처리를 위한 시리얼 연결 구독
        if isinstance(main_app, EMGGameApplication):
            # 기존 구독 해제 로직 (선택적이지만 중복 구독 방지에 좋음)
            if hasattr(self, '_is_subscribed_for_hand_model') and self._is_subscribed_for_hand_model:
                try:
                    main_app.unsubscribe_widget(self)  # 변경: self.process_emg_for_hand_model → self
                    self.add_log_to_main_window("기존 손 모델 EMG 구독 해제됨.")
                except Exception as e:
                    self.add_log_to_main_window(f"손 모델 EMG 구독 해제 중 오류(무시 가능): {e}")

            main_app.subscribe_widget(self)  # 변경: self.process_emg_for_hand_model → self
            self._is_subscribed_for_hand_model = True
            self.add_log_to_main_window("손 모델 제어를 위해 EMG 데이터 수신을 시작합니다.")


    def process_emg_for_hand_model(self, s1, s2):
        """손 모델 제어를 위한 EMG 데이터 처리 및 Unity 전송"""
        if not self.unity_server_thread:
            # print("Debug Hand Model: Unity server not started") # 너무 잦은 로그 방지
            return
        if not hasattr(self, 'predictor_hand_model') or self.predictor_hand_model is None:
            # print("Debug Hand Model: Predictor not available or not loaded")
            return
        if not hasattr(self, 'emg_processor_hand_model') or self.emg_processor_hand_model is None:
            # print("Debug Hand Model: EMG Processor not available")
            return

        result = self.emg_processor_hand_model.process_data(s1, s2)

        if isinstance(result, tuple) and result[0] == "movement_completed":
            sequence = result[1]
            try:
                prediction, confidence = self.predictor_hand_model.predict(sequence)
            except Exception as e:
                self.add_log_to_main_window(f"손 모델 예측 중 오류: {e}")
                return

            # self.add_log_to_main_window(f"손 모델 제어: 예측: {prediction}, 신뢰도: {confidence:.2f}") # 상세 로그

            label_to_send_val = "2,2,2,2,2"  # 기본값 (IDLE 또는 0,0,0,0,0 효과)
            
            current_config = self.emg_processor_hand_model.config if self.emg_processor_hand_model.config else {}
            confidence_threshold = float(current_config.get('recognition_confidence_threshold', 0.5)) # 기본값 0.5
            
            # 신뢰도 확인       
            if confidence >= confidence_threshold:
                if prediction == 11: # 사용자가 요청한 첫 번째 경우
                    label_to_send_val = "1,2,2,0,0" # Unity에서 (0,0,0,0,0)으로 해석할 값
                    # self.add_log_to_main_window(f"손 모델: Prediction 1 (->0) 감지 (신뢰도: {confidence:.2f})")
                elif prediction == 12: # 사용자가 요청한 두 번째 경우
                    label_to_send_val = "1,0,0,0,0" # Unity에서 (2,2,2,2,2)으로 해석할 값
                    # self.add_log_to_main_window(f"손 모델: Prediction 2 (->2) 감지 (신뢰도: {confidence:.2f})")
                elif prediction == 13: # 
                    label_to_send_val = "2,2,2,2,2" # 또는 다른 IDLE에 해당하는 값
                elif prediction == 1: # 사용자가 요청한 두 번째 경우
                    label_to_send_val = "2,2,2,2,2" # Unity에서 (2,2,2,2,2)으로 해석할 값
                    label_to_send_val = "2,0,2,2,2"
                    label_to_send_val = "2,2,2,2,2"
                #     self.add_log_to_main_window(f"손 모델: IDLE 감지 (신뢰도: {confidence:.2f})")
                # 여기에 다른 prediction 값에 대한 처리를 추가할 수 있습니다.
                # 예: elif prediction == 4: label_to_send_val = "2,2,2,2,2"
                # 지금은 예시로 아무 값이나 넣음 ==> 동작에 따라 넣어줘
                # 0: 굽힌 상태 1: 중간 2: 완전히 편 상태
                # 엄지, 검지, 중지, 약지, 소지 순
                #-------------------------------------------------------------------------------------------------------------------------------------------------------------------------<<<<<<<여기 중요!
                else:
                    # prediction이 1 또는 2가 아니지만 신뢰도가 높은 경우, 기본값(0) 또는 특정 값을 보낼 수 있습니다.
                    # 현재는 prediction 1, 2 외에는 label_to_send_val이 0으로 유지됩니다.
                    # self.add_log_to_main_window(f"손 모델: Prediction {prediction} (미지정 동작) 감지, 기본값 전송 (신뢰도: {confidence:.2f})")
                    pass


            if self.unity_client_socket:
                try:
                    data_to_send = f"{label_to_send_val}\n" # 문자열로 변환하여 전송
                    self.unity_client_socket.sendall(data_to_send.encode('utf-8'))
                    # 로그가 너무 많이 쌓이는 것을 방지하기 위해 조건부 로깅 또는 제거
                    self.add_log_to_main_window(f"Unity로 손 모델 데이터 전송: {label_to_send_val} (Raw pred: {prediction}, Conf: {confidence:.2f})")
                except socket.error as e:
                    self.add_log_to_main_window(f"Unity로 데이터 전송 실패: {e}")
                    self.handle_unity_client_disconnection() # 연결 끊김 처리
                except Exception as e: # 다른 예외 처리 (예: 소켓이 갑자기 닫힌 경우)
                    self.add_log_to_main_window(f"Unity 데이터 전송 중 예외 발생: {e}")
                    self.handle_unity_client_disconnection()
            # else:
                # self.add_log_to_main_window("Unity 클라이언트가 연결되지 않았습니다. 데이터 전송 실패.") # 매우 자주 발생 가능
                pass
        elif isinstance(result, str) and result == "no_movement":
            # 움직임이 없을 때 Unity에 특정 값을 보내고 싶다면 여기에 추가 (예: 기본 자세 유지)
            # if self.unity_client_socket:
            #     try:
            #         data_to_send = "0\n" # 예시: 기본값 0 전송
            #         self.unity_client_socket.sendall(data_to_send.encode('utf-8'))
            #     except socket.error:
            #         self.handle_unity_client_disconnection()
            pass

    def stop_unity_server(self):
        if self.unity_client_socket:
            try:
                self.unity_client_socket.close()
            except Exception as e:
                print(f"Unity 클라이언트 소켓 닫기 오류 (서버 종료 중): {e}")
            self.unity_client_socket = None
        
        if self.unity_server_socket:
            try:
                self.unity_server_socket.close()
                print("Unity 통신 서버 중지됨.")
                self.add_log_to_main_window("손 모델 제어: Unity 통신 서버 중지됨")
            except Exception as e:
                print(f"Unity 통신 서버 중지 실패: {e}")
                self.add_log_to_main_window(f"손 모델 제어: Unity 통신 서버 중지 실패 - {e}")
            self.unity_server_socket = None

        if self.unity_server_thread and self.unity_server_thread.is_alive():
            # 서버 스레드가 정상적으로 종료되도록 약간의 시간을 줄 수 있습니다.
            # socket.accept()에서 블록되어 있을 수 있으므로, 소켓을 닫는 것이 중요합니다.
            pass

        if self.unity_process and self.unity_process.poll() is None:
            try:
                self.unity_process.terminate() # Unity 프로세스 종료
                self.unity_process.wait(timeout=5) # 종료 대기
                print("Unity 프로세스 종료됨.")
                self.add_log_to_main_window("손 모델 제어: Unity 프로세스 종료됨")
            except subprocess.TimeoutExpired:
                print("Unity 프로세스 종료 시간 초과, 강제 종료 시도.")
                self.unity_process.kill()
                self.add_log_to_main_window("손 모델 제어: Unity 프로세스 강제 종료됨")

            except Exception as e:
                print(f"Unity 프로세스 종료 실패: {e}")
                self.add_log_to_main_window(f"손 모델 제어: Unity 프로세스 종료 실패 - {e}")
            self.unity_process = None

    # MainMenuWidget의 다른 메서드에서 로그 추가 시 (예시)
    # self.parent().parent().add_log("로그 메시지") 대신
    # self.add_log_to_main_window("로그 메시지") 사용 가능하도록 메서드 추가

    def add_log_to_main_window(self, message):
        # MainMenuWidget이 EMGGameApplication의 직접적인 자식이 아닐 수 있음
        # 따라서 window()를 사용하여 최상위 창을 찾고, 해당 창에 로그를 추가하는 메서드가 있는지 확인
        main_win = self.window()
        if hasattr(main_win, 'add_log'):
            main_win.add_log(f"[MainMenu] {message}")
        else:
            print(f"[MainMenu Log] {message}") # 대체 로그 출력
# ------------------------------------------------------------------------!

# ------------------------------------------------------------------------
# 사용 가이드 위젯
# ------------------------------------------------------------------------
class GuideWidget(BaseEMGWidget):
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
<p>안녕하세요! EMG 센서를 이용한 동작 인식 프로그램에 오신 것을 환영합니다.</p>
<p>이 프로그램은 팔에 착용하는 EMG(근전도) 센서를 통해 손동작을 인식하여 동작합니다.</p>
<p>PC 프로그램 기능으로는 게임모드 화면제어모드 촬영 모드가 있습니다.</p>
<p>다음 페이지에서 센서 착용 방법에 대해 알아보겠습니다.</p>
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
    <li>'대기' 상태로 손을 편안하게 유지해보세요.</li>
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
        super().set_user(username, password)
        self.has_model = user_has_model(username, password)
        
    def start(self):
        """가이드 시작 및 시리얼 연결"""
        self.current_page = 0
        self.content_stack.setCurrentIndex(0)
        self.prev_btn.setEnabled(False)
        self.next_btn.setEnabled(True)
        
        # 전역 시리얼 스레드에 구독
        if self.subscribe_to_serial():
            self.is_active = True
            self.add_log("EMG 센서에 연결되었습니다.")
        else:
            self.add_log("EMG 센서 연결에 실패했습니다.")
            
        # 상태 업데이트 타이머 시작
        self.update_timer.start(500)
        
    def stop(self):
        """가이드 중지 및 시리얼 연결 해제"""
        self.is_active = False
        self.unsubscribe_from_serial()
        self.update_timer.stop()
        
    def process_data(self, s1, s2):
        """EMG 데이터 처리"""
        if not self.is_active:
            return
            
        result = self.emg_processor.process_data(s1, s2)
        
        if result == "movement_detected":
            # 동작 감지 텍스트 변경
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
        self.guide_completed.emit(False)


# ------------------------------------------------------------------------
# 동작 데이터 수집 위젯
# ------------------------------------------------------------------------
class DataCollectionWidget(BaseEMGWidget):
    """
    DataCollectionWidget 클래스: 동작 데이터 수집 화면을 제공하는 위젯
    - 가위/바위/보 동작 데이터 및 동작 제어용 데이터 수집
    - 감도 설정 및 데이터 수집 기능 제공
    - 네오모피즘 디자인 적용 (1920x1080 최적화)
    """
    collection_completed = pyqtSignal(str)  # 모델 타입 전달 (game 또는 action)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # 라벨 카운트 및 데이터 수집 상태
        self.label_counts = {
            LABEL_IDLE: 0,
            LABEL_SCISSORS: 0,
            LABEL_ROCK: 0,
            LABEL_PAPER: 0,
        }
        # 1-8 라벨 추가 (동작 제어용)
        for i in range(1, 9):
            self.label_counts[i] = 0
            
        self.current_label = None
        self.target_count = 100
        self.data_file = ""
        self.sensitivity = 1.0
        # UI 초기화
        self.init_ui()
        
    # DataCollectionWidget의 init_ui 메소드에서 버튼 생성 부분 수정

    def init_ui(self):
        """UI 초기화 - 1920x1080 해상도 최적화 (버튼 오류 수정)"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)  # 여백 줄임
        layout.setSpacing(15)  # 간격 줄임
        
        # 배경 설정
        self.setStyleSheet(f"background-color: {BG_COLOR};")
        
        # 제목 카드 - 높이 줄임
        title_card = NeumorphicCard()
        title_layout = QVBoxLayout(title_card)
        title_layout.setContentsMargins(15, 10, 15, 10)  # 패딩 줄임
        
        # 제목 텍스트
        self.title_label = QLabel("동작 데이터 수집")
        self.title_label.setStyleSheet(f"color: {TEXT_COLOR}; font-size: 24px; font-weight: bold;")  # 폰트 크기 줄임
        self.title_label.setAlignment(Qt.AlignCenter)
        
        title_layout.addWidget(self.title_label)
        
        # 탭 영역 카드
        tab_card = NeumorphicCard()
        tab_layout = QVBoxLayout(tab_card)
        tab_layout.setContentsMargins(15, 15, 15, 15)  # 패딩 줄임
        
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
                padding: 8px 15px;  /* 패딩 줄임 */
                margin-right: 5px;
                border-radius: 8px 8px 0 0;
                font-size: 14px;  /* 폰트 크기 줄임 */
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
        
        # 감도 설정 설명 텍스트
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
        
        # 감도 설정 레이블
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
        
        # 감지 상태 레이블
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
        
        # 탭 2: 통합 데이터 수집 탭 (기존의 게임+동작 통합)
        collection_tab = QWidget()
        collection_layout = QVBoxLayout(collection_tab)
        collection_layout.setContentsMargins(15, 15, 15, 15)  # 패딩 줄임

        # 데이터 수집 설명 텍스트 - 높이 줄임
        self.collection_text = QTextEdit()
        self.collection_text.setReadOnly(True)
        self.collection_text.setMaximumHeight(100)  # 높이 제한
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
        <h3 style='color: #5e72e4;'>동작 데이터 수집</h3>
            """)
        
        # 현재 동작 및 진행 상황 카드
        status_card = NeumorphicCard()
        status_layout = QVBoxLayout(status_card)
        status_layout.setContentsMargins(15, 15, 15, 15)  # 패딩 줄임
        
        # 현재 동작 표시
        self.current_action_label = QLabel("현재 선택 동작: 없음")
        self.current_action_label.setStyleSheet(f"""
            color: {PRIMARY_COLOR};
            font-size: 18px;  /* 폰트 크기 줄임 */
            font-weight: bold;
            margin-bottom: 10px;
        """)
        self.current_action_label.setAlignment(Qt.AlignCenter)
        
        # 동작 선택 영역 제목
        action_selection_title = QLabel("동작을 클릭하여 선택하세요 최소 5번의 수집이 필요합니다")
        action_selection_title.setStyleSheet(f"color: {TEXT_COLOR}; font-size: 16px; font-weight: bold;")
        action_selection_title.setAlignment(Qt.AlignCenter)
        
        # 진행 상태를 표시할 그리드 레이아웃 (가로로 배열) - 간격 줄임
        progress_grid = QGridLayout()
        progress_grid.setSpacing(10)  # 간격 줄임
        
        # 동작 카드 및 카운트 라벨 저장 딕셔너리
        self.count_labels = {}
        self.progress_cards = {}  # 카드를 저장할 딕셔너리 추가
        
        # 첫 번째 행: 가위/바위/보 - 가위 색상 변경
        actions_row1 = [
            ("가위", LABEL_SCISSORS, "#F0F8E8", "#4CAF50"),  # 연한 녹색 배경, 진한 녹색 텍스트
            ("바위", LABEL_ROCK, "#FFFBEB", "#FB8C00"),
            ("보", LABEL_PAPER, "#E8F5E9", "#2DCE89")
        ]
        
        # 두 번째 행: 동작 1-4
        actions_row2 = [(f"동작 {i}", i, "#E6F4FF", ACCENT_COLOR) for i in range(1, 5)]
        
        # 세 번째 행: 동작 5-8
        actions_row3 = [(f"동작 {i}", i, "#E6F4FF", ACCENT_COLOR) for i in range(5, 9)]
        
        # 각 행에 대한 진행 상태 카드 생성 - 크기 줄임
        for row, actions in enumerate([actions_row1, actions_row2, actions_row3]):
            for col, (action_name, label, bg_color, progress_color) in enumerate(actions):
                # 진행 상태 카드 생성 (클릭 가능하게) - 크기 줄임
                progress_card = NeumorphicCard()
                progress_card.setFixedHeight(120)  # 높이 줄임
                progress_card.setCursor(Qt.PointingHandCursor)
                
                # 기본 스타일 설정 (라벨 투명하게)
                progress_card.setStyleSheet(f"""
                    NeumorphicCard {{
                        background-color: {bg_color};
                        border-radius: 12px;
                        border: none;
                    }}
                    NeumorphicCard:hover {{
                        background-color: {self.lighten_color(bg_color, 10)};
                    }}
                    NeumorphicCard QLabel {{
                        background-color: transparent !important;
                        border: none !important;
                    }}
                """)
                
                # 카드 클릭 이벤트 연결
                progress_card.mousePressEvent = lambda event, l=label: self.on_progress_card_clicked(l)
                
                progress_card_layout = QVBoxLayout(progress_card)
                progress_card_layout.setContentsMargins(8, 8, 8, 8)  # 패딩 줄임
                
                # 아이콘 또는 이모지 추가 (선택사항) - 크기 줄임
                icon_label = QLabel()
                if label == LABEL_SCISSORS:
                    icon_label.setText("✌️")
                elif label == LABEL_ROCK:
                    icon_label.setText("✊")
                elif label == LABEL_PAPER:
                    icon_label.setText("✋")
                else:
                    icon_label.setText(f"{label}")
                icon_label.setStyleSheet("font-size: 24px; background-color: transparent !important; border: none !important;")  # 투명 배경 명시
                icon_label.setAlignment(Qt.AlignCenter)
                
                # 동작 이름 레이블
                action_label = QLabel(action_name)
                action_label.setStyleSheet(f"color: {TEXT_COLOR}; font-size: 24px; font-weight: bold; background-color: transparent !important; border: none !important;")  # 투명 배경 명시
                action_label.setAlignment(Qt.AlignCenter)
                
                # 카운트 레이블
                count_label = QLabel(f"0/{self.target_count}")
                count_label.setAlignment(Qt.AlignCenter)
                count_label.setStyleSheet(f"color: {progress_color}; font-size: 14px; font-weight: bold; background-color: transparent !important; border: none !important;")  # 투명 배경 명시
                
                # 카드에 위젯 추가
                progress_card_layout.addWidget(icon_label)
                progress_card_layout.addWidget(action_label)
                progress_card_layout.addWidget(count_label)
                
                # 그리드에 카드 추가
                progress_grid.addWidget(progress_card, row, col)
                
                # 카운트 라벨과 카드 저장
                self.count_labels[label] = count_label
                self.progress_cards[label] = progress_card
        
        # 버튼 레이아웃
        buttons_layout = QHBoxLayout()
        
        # 컨트롤 버튼 - 크기 줄임 (여기가 중요!)
        self.delete_btn = NeumorphicButton("마지막 데이터 삭제", DANGER_COLOR)
        self.delete_btn.setFixedHeight(40)  # 높이 줄임
        self.delete_btn.clicked.connect(self.delete_last_data)
        self.delete_btn.setEnabled(False)
        
        self.complete_btn = NeumorphicButton("수집 완료", PRIMARY_COLOR)
        self.complete_btn.setFixedHeight(40)  # 높이 줄임
        self.complete_btn.clicked.connect(self.complete_collection)
        self.complete_btn.setEnabled(False)
        
        buttons_layout.addWidget(self.delete_btn)
        buttons_layout.addStretch()
        buttons_layout.addWidget(self.complete_btn)
        
        # 카드에 위젯 추가
        status_layout.addWidget(self.current_action_label)
        status_layout.addWidget(action_selection_title)
        status_layout.addLayout(progress_grid)
        status_layout.addLayout(buttons_layout)  # 버튼 레이아웃 추가!
        
        # 탭에 카드 추가
        collection_layout.addWidget(self.collection_text)
        collection_layout.addWidget(status_card)
        
        # 탭 추가
        self.tab_widget.addTab(sensitivity_tab, "1. 감도 설정")
        self.tab_widget.addTab(collection_tab, "2. 동작 데이터 수집")
        self.tab_widget.currentChanged.connect(self.on_tab_changed)
        
        tab_layout.addWidget(self.tab_widget)
        
        # 로그 카드 - 높이 줄임
        log_card = NeumorphicCard(inset=True)
        log_layout = QVBoxLayout(log_card)
        log_layout.setContentsMargins(10, 8, 10, 8)  # 패딩 줄임
        
        # 로그 영역 레이블
        log_header = QLabel("로그")
        log_header.setStyleSheet(f"font-size: 14px; font-weight: bold; color: {TEXT_COLOR};")  # 폰트 크기 줄임
        
        # 로그 텍스트 영역
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(100)  # 높이 제한
        self.log_text.setStyleSheet(f"""
            QTextEdit {{
                background-color: transparent;
                border: none;
                color: {TEXT_COLOR};
                font-family: 'Consolas', monospace;
                font-size: 12px;  /* 폰트 크기 줄임 */
                line-height: 1.2;
            }}
        """)
        
        log_layout.addWidget(log_header)
        log_layout.addWidget(self.log_text)
        
        # 네비게이션 버튼
        nav_layout = QHBoxLayout()
        
        # 뒤로/완료 버튼 - 크기 줄임
        self.back_btn = NeumorphicButton("뒤로")
        self.back_btn.setFixedHeight(40)  # 높이 줄임
        self.back_btn.clicked.connect(self.go_back)
        
        nav_layout.addWidget(self.back_btn)
        nav_layout.addStretch()
        
        # 레이아웃에 위젯 추가
        layout.addWidget(title_card)
        layout.addWidget(tab_card)
        layout.addWidget(log_card)
        layout.addLayout(nav_layout)
    
    def lighten_color(self, color, amount=20):
        """색상을 밝게 만듭니다."""
        c = QColor(color)
        h, s, l, a = c.getHslF()
        l = min(1.0, l + amount / 100)
        c.setHslF(h, s, l, a)
        return c.name()

    def on_progress_card_clicked(self, label):
        """진행 상황 카드 클릭 이벤트 - 라벨 투명성 확실히 보장"""
        # 이전 선택된 카드 스타일 복원
        if hasattr(self, 'current_label') and self.current_label in self.progress_cards:
            prev_card = self.progress_cards[self.current_label]
            
            # 기본 스타일로 복원 (라벨이 확실히 투명하도록)
            if self.current_label == LABEL_SCISSORS:
                prev_card.setStyleSheet("""
                    NeumorphicCard {
                        background-color: #F0F8E8;
                        border-radius: 12px;
                        border: none;
                    }
                    NeumorphicCard:hover {
                        background-color: #F2FAE8;
                    }
                    NeumorphicCard QLabel {
                        background-color: transparent !important;
                        border: none !important;
                    }
                """)
            elif self.current_label == LABEL_ROCK:
                prev_card.setStyleSheet("""
                    NeumorphicCard {
                        background-color: #FFFBEB;
                        border-radius: 12px;
                        border: none;
                    }
                    NeumorphicCard:hover {
                        background-color: #FFFCE8;
                    }
                    NeumorphicCard QLabel {
                        background-color: transparent !important;
                        border: none !important;
                    }
                """)
            elif self.current_label == LABEL_PAPER:
                prev_card.setStyleSheet("""
                    NeumorphicCard {
                        background-color: #E8F5E9;
                        border-radius: 12px;
                        border: none;
                    }
                    NeumorphicCard:hover {
                        background-color: #EAF6EB;
                    }
                    NeumorphicCard QLabel {
                        background-color: transparent !important;
                        border: none !important;
                    }
                """)
            else:
                prev_card.setStyleSheet("""
                    NeumorphicCard {
                        background-color: #E6F4FF;
                        border-radius: 12px;
                        border: none;
                    }
                    NeumorphicCard:hover {
                        background-color: #E8F5FF;
                    }
                    NeumorphicCard QLabel {
                        background-color: transparent !important;
                        border: none !important;
                    }
                """)
        
        # 새 동작 설정
        self.current_label = label
        
        # 선택된 카드 강조 표시 (라벨이 확실히 투명하도록)
        selected_card = self.progress_cards[label]
        selected_card.setStyleSheet(f"""
            NeumorphicCard {{
                background-color: {PRIMARY_COLOR};
                border-radius: 12px;
                border: 3px solid {self.lighten_color(PRIMARY_COLOR, 30)};
            }}
            NeumorphicCard QLabel {{
                background-color: transparent !important;
                border: none !important;
                color: white !important;
            }}
        """)
        
        # 라벨에 따른 텍스트 설정
        if label == LABEL_SCISSORS:
            self.current_action_label.setText("현재 선택 동작: 가위")
        elif label == LABEL_ROCK:
            self.current_action_label.setText("현재 선택 동작: 바위")
        elif label == LABEL_PAPER:
            self.current_action_label.setText("현재 선택 동작: 보")
        else:
            self.current_action_label.setText(f"현재 선택 동작: 동작 {label}")
        
        self.add_log(f"동작 선택: {self.current_action_label.text()}")
    def set_user(self, username, password):
        """사용자 정보 설정"""
        super().set_user(username, password)
        
        # 라벨 카운트 로드
        config = load_user_config(username, password)
        
        # 통합된 라벨 카운트 로드
        if "label_counts" in config:
            for label_str, count in config["label_counts"].items():
                label = int(label_str)
                if label in self.label_counts:
                    self.label_counts[label] = count   
                    
        self.update_count_labels()
        
    def start(self):
        """데이터 수집 시작"""
        # 탭 초기화
        self.tab_widget.setCurrentIndex(0)
        
        # 대기 상태 데이터 확인
        self.check_idle_data()
        
        # 전역 시리얼 스레드에 구독
        if self.subscribe_to_serial():
            self.is_active = True
            self.add_log("EMG 센서에 연결되었습니다.")
        else:
            self.add_log("EMG 센서 연결에 실패했습니다.")
        
    def stop(self):
        """데이터 수집 중지"""
        self.is_active = False
        self.unsubscribe_from_serial()
        
    def process_data(self, s1, s2):
        """EMG 데이터 처리"""
        if not self.is_active:
            return
            
        result = self.emg_processor.process_data(s1, s2)
        
        # 동작 감지 중일 때만 처리
        if self.tab_widget.currentIndex() == 0:
            # 감도 설정 탭
            if isinstance(result, tuple) and result[0] == "movement_completed":
                self.detection_label.setText("동작 감지: 완료!")
                self.detection_label.setStyleSheet(f"font-size: 16px; color: {SECONDARY_COLOR}; font-weight: bold;")
                QTimer.singleShot(1000, self.reset_detection_label)
                
            elif result == "movement_detected":
                self.detection_label.setText("동작 감지: 감지됨!")
                self.detection_label.setStyleSheet(f"font-size: 16px; color: {PRIMARY_COLOR}; font-weight: bold;")
                
        elif self.tab_widget.currentIndex() == 1 and self.current_label is not None:
            # 데이터 수집 탭
            if isinstance(result, tuple) and result[0] == "movement_completed":
                sequence = result[1]
                self.save_sequence(sequence, self.current_label)
    
    def check_idle_data(self):
        """대기 상태 데이터 확인"""
        ensure_dir_exists(COMMON_DATA_DIR)
        idle_files = [f for f in os.listdir(COMMON_DATA_DIR) if f.startswith('idle_') and f.endswith('.csv')]
        
        if not idle_files:
            self.add_log("대기 상태 데이터가 없습니다. 시스템이 대기 상태를 인식하지 못할 수 있습니다.")
        else:
            # 대기 상태 데이터 개수 확인
            idle_count = 0
            for file in idle_files:
                try:
                    with open(os.path.join(COMMON_DATA_DIR, file), 'r') as f:
                        # 헤더 제외 행 수 계산
                        rows = sum(1 for _ in f) - 1
                        idle_count += rows
                except:
                    pass
            
            # 대기 상태 데이터 개수 설정
            self.label_counts[LABEL_IDLE] = idle_count
            
            # UI 업데이트는 하지 않음 (사용자에게 보여주지 않기 위해)
            self.add_log(f"대기 상태 데이터 {idle_count}개를 공통 데이터 폴더에서 로드했습니다.")
    
    def create_data_file(self):
        """통합 데이터 파일 생성"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 통합된 데이터 파일명 사용
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
    
    def reset_detection_label(self):
        """동작 감지 레이블 리셋"""
        self.detection_label.setText("동작 감지: 대기 중...")
        self.detection_label.setStyleSheet(f"font-size: 16px; color: {TEXT_COLOR}; font-weight: bold;")
        
    def save_sequence(self, sequence, label):
        """EMG 시퀀스 저장 - 통합 버전"""
        if len(sequence) < 50:
            self.add_log("시퀀스가 너무 짧습니다. 무시합니다.")
            return
            
        # 데이터 파일이 없으면 생성
        if not hasattr(self, 'data_file') or not self.data_file:
            self.create_data_file()
            
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
        
        # 버튼 상태 업데이트 - 통합 버전
        self.delete_btn.setEnabled(True)
        
        # 설정 파일에 저장
        config = load_user_config(self.username, self.password)
        
        # 통합된 카운트 정보 저장
        config["label_counts"] = {str(k): v for k, v in self.label_counts.items()}
        save_user_config(self.username, self.password, config)
        
        # 라벨에 따른 로그 메시지
        if label == LABEL_SCISSORS:
            label_name = "가위"
        elif label == LABEL_ROCK:
            label_name = "바위"
        elif label == LABEL_PAPER:
            label_name = "보"
        else:
            label_name = f"라벨 {label}"
        
        self.add_log(f"'{label_name}' 동작 데이터 저장 완료 ({self.label_counts[label]}/{self.target_count})")
        
        # 목표 카운트 달성 시 자동으로 다음 동작 제안
        if self.label_counts[label] >= self.target_count:
            # 목표 달성 로그 출력
            self.add_log(f"'{label_name}' 동작의 목표 데이터 수 수집 완료!")
            
        # 완료 가능 여부 확인
        self.check_completion()
        
    def delete_last_data(self):
        """마지막 데이터 삭제 - 통합 버전"""
        if not hasattr(self, 'data_file') or not self.data_file:
            self.add_log("삭제할 데이터가 없습니다.")
            return
            
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
                
                # 라벨 카운트 감소
                if last_label in self.label_counts:
                    self.label_counts[last_label] = max(0, self.label_counts[last_label] - 1)
                        
            # 파일 다시 쓰기
            with open(self.data_file, 'w', newline='') as f:
                f.writelines(lines[:-1])
                
            # 카운트 업데이트
            self.update_count_labels()
            
            # 설정 파일에 저장
            config = load_user_config(self.username, self.password)
            config["label_counts"] = {str(k): v for k, v in self.label_counts.items()}
            save_user_config(self.username, self.password, config)
            
            self.add_log("마지막 데이터가 삭제되었습니다.")
            
            # 완료 가능 여부 확인
            self.check_completion()
            
        except Exception as e:
            self.add_log(f"데이터 삭제 오류: {str(e)}")
    
    def update_count_labels(self):
        """카운트 레이블 업데이트 - 통합 버전"""
        # 진행 상태의 모든 동작 카운트 업데이트
        for label, count_label in self.count_labels.items():  # 튜플이 아닌 라벨만
            current_count = self.label_counts.get(label, 0)
            count_label.setText(f"{current_count}/{self.target_count}")
                    
        # 완료 가능 여부 확인
        self.check_completion()
        
    def check_completion(self):
        """완료 가능 여부 확인 - 통합 버전"""
        # 게임 모드 완료 확인 (가위/바위/보)
        game_min_collected = (
            self.label_counts[LABEL_SCISSORS] >= 5 and 
            self.label_counts[LABEL_ROCK] >= 5 and 
            self.label_counts[LABEL_PAPER] >= 5
        )
        
        # 삭제 버튼 활성화
        has_data = sum(self.label_counts.values()) > 0
        self.delete_btn.setEnabled(has_data)
        
        # 완료 버튼 활성화 (가위바위보 + 최소 3개 동작)
        self.complete_btn.setEnabled(game_min_collected)
    
    def on_tab_changed(self, index):
        """탭 변경 이벤트 처리"""
        # 현재 수집 모드 설정
        if index == 1:  # 동작 데이터 수집 탭
            # 동작 선택을 초기화 - 통합 버전
            if self.current_label is None:
                # 초기 동작으로 가위(LABEL_SCISSORS) 선택
                self.on_progress_card_clicked(LABEL_SCISSORS)
                    
            # 완료 버튼 상태 업데이트
            self.check_completion()
        else:
            # 감도 설정 탭이거나 다른 탭인 경우
            self.complete_btn.setEnabled(False)

    def complete_collection(self):
        """데이터 수집 완료"""
        # 완료 버튼 상태 확인
        if not self.complete_btn.isEnabled():
            self.add_log("아직 데이터가 충분하지 않습니다. 각 동작당 최소 5회 이상 수집해주세요.")
            return
        
        self.add_log(f"데이터 수집 완료. 모델 학습으로 이동합니다.")
        
        # 시리얼 스레드 구독 해제
        self.is_active = False
        self.unsubscribe_from_serial()
        
        # 통합 모델 학습 신호 발생
        self.collection_completed.emit("unified")
    
    def go_back(self):
        """뒤로 가기"""
        # 사용자 설정 저장
        config = load_user_config(self.username, self.password)
        config["sensitivity"] = self.sensitivity
        config["label_counts"] = {str(k): v for k, v in self.label_counts.items()}
        save_user_config(self.username, self.password, config)
        
        # 부모 클래스의 go_back 호출
        super().go_back()
        self.collection_completed.emit("none")  # 모델 학습 안 함
# ------------------------------------------------------------------------
# 모델 학습 위젯
# ------------------------------------------------------------------------
class ModelTrainingWidget(BaseEMGWidget):
    """
    ModelTrainingWidget 클래스: 모델 학습 화면을 제공하는 위젯
    - 수집된 데이터를 기반으로 EMG 동작 인식 모델 학습
    - 학습 진행 상황 및 결과를 표시
    - 네오모피즘 디자인 적용
    """
    training_completed = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # 학습 관련 변수
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
        
        # 제목 텍스트
        self.title_label = QLabel("모델 학습")
        self.title_label.setStyleSheet(f"color: {TEXT_COLOR}; font-size: 32px; font-weight: bold;")
        self.title_label.setAlignment(Qt.AlignCenter)
        
        title_layout.addWidget(self.title_label)
        
        # 정보 카드
        info_card = NeumorphicCard()
        info_layout = QVBoxLayout(info_card)
        info_layout.setContentsMargins(25, 25, 25, 25)
        
        # 정보 텍스트
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
<p>수집된 데이터와 공통 데이터를 기반으로 EMG 동작 인식 모델을 학습합니다.</p>
<p>학습은 자동으로 진행되며, 최적의 모델을 찾기 위해 여러 번 시도할 수 있습니다.</p>
<p>학습이 완료되면 자동으로 다음 단계로 넘어갑니다.</p>
        """)
        info_layout.addWidget(self.info_text)
        
        # 진행 상황 카드
        progress_card = NeumorphicCard()
        progress_card_layout = QVBoxLayout(progress_card)
        progress_card_layout.setContentsMargins(25, 25, 25, 25)
        
        # 진행 상태 레이블
        self.progress_label = QLabel("준비 중...")
        self.progress_label.setStyleSheet(f"color: {PRIMARY_COLOR}; font-size: 24px; font-weight: bold;")
        self.progress_label.setAlignment(Qt.AlignCenter)
        
        # 프로그레스 바 컨테이너
        progress_bar_frame = NeumorphicCard(inset=True)
        progress_bar_frame_layout = QVBoxLayout(progress_bar_frame)
        progress_bar_frame_layout.setContentsMargins(10, 5, 10, 5)
        
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
        
        progress_bar_frame_layout.addWidget(self.progress_bar)
        
        progress_card_layout.addWidget(self.progress_label)
        progress_card_layout.addWidget(progress_bar_frame)
        
        # 로그 카드
        log_card = NeumorphicCard(inset=True)
        log_layout = QVBoxLayout(log_card)
        
        # 로그 헤더
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
        
        # 버튼
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
        
        
    def set_user(self, username, password, model_type="unified"):
        """사용자 정보 설정 - 통합 모델 사용"""
        super().set_user(username, password)
        
        # 정보 업데이트
        self.info_text.setText("""
    <h2 style='color: #5e72e4;'>통합 모델 학습</h2>
    <p>수집된 모든 동작 데이터를 기반으로 통합 EMG 동작 인식 모델을 학습합니다.</p>
    <p>이 모델은 가위바위보와 묵찌빠 게임, 화면 제어, 촬영 제어 등 모든 기능에 사용됩니다.</p>
    <p>학습은 자동으로 진행되며, 최적의 모델을 찾기 위해 여러 번 시도할 수 있습니다.</p>
    <p>학습이 완료되면 자동으로 다음 단계로 넘어갑니다.</p>
        """)
        
    def start(self):
        """위젯 시작"""
        self.reset_ui()
        self.check_data()
        
    def stop(self):
        """위젯 중지"""
        if self.is_training and hasattr(self, 'training_thread') and self.training_thread.isRunning():
            self.training_thread.terminate()
            self.training_thread.wait()
        
    # 나머지 메서드들은 기존 코드와 동일
    def reset_ui(self):
        """UI 초기화"""
        self.progress_label.setText("준비 중...")
        self.progress_bar.setValue(0)
        self.log_text.clear()
        self.train_btn.setEnabled(True)
        self.back_btn.setEnabled(True)
        
    def check_data(self):
        """데이터 파일 확인 - 통합 버전"""
        # 모든 CSV 파일 검색
        data_files = [f for f in os.listdir(self.user_dir) if f.endswith('.csv')]
        
        if not data_files:
            self.add_log("데이터 파일이 없습니다. 먼저 데이터를 수집해주세요.")
            self.train_btn.setEnabled(False)
            return False
            
        self.add_log(f"{len(data_files)}개의 데이터 파일을 찾았습니다.")
        
        # 공통 데이터 확인
        common_files = []
        if os.path.exists(COMMON_DATA_DIR):
            common_files = [f for f in os.listdir(COMMON_DATA_DIR) if f.endswith('.csv')]
            if common_files:
                self.add_log(f"{len(common_files)}개의 공통 데이터 파일을 찾았습니다.")
        
        # 데이터 분포 확인
        config = load_user_config(self.username, self.password)
        
        # 통합 데이터 카운트
        label_counts = config.get("label_counts", {})
        
        # 데이터 분포 출력
        self.add_log("데이터 분포:")
        
        # 게임 동작 (가위/바위/보)
        scissors_count = int(label_counts.get(str(LABEL_SCISSORS), 0))
        rock_count = int(label_counts.get(str(LABEL_ROCK), 0))
        paper_count = int(label_counts.get(str(LABEL_PAPER), 0))
        
        self.add_log(f"  - 가위(라벨 {LABEL_SCISSORS}): {scissors_count}개")
        self.add_log(f"  - 바위(라벨 {LABEL_ROCK}): {rock_count}개")
        self.add_log(f"  - 보(라벨 {LABEL_PAPER}): {paper_count}개")
        
        # 추가 동작 (라벨 1-8)
        for i in range(1, 9):
            count = int(label_counts.get(str(i), 0))
            if count > 0:
                self.add_log(f"  - 동작 {i}: {count}개")
                
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
        
        # 통합된 모델 파일명 설정
        model_file = "emg_model.pth"
        
        # 학습 스레드 시작
        self.training_thread = TrainingThread(self.username, self.password, model_file=model_file)
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
        """학습 완료 처리 - 통합 모델 버전"""
        self.is_training = False
        
        if success:
            self.add_log(f"학습 완료! 모델 저장 경로: {model_path}")
            self.add_log(f"모델 정확도: {accuracy:.2f}")
            
            # 사용자 설정 업데이트
            config = load_user_config(self.username, self.password)
            
            # 통합 모델 정확도 저장
            config["model_accuracy"] = accuracy
            
            save_user_config(self.username, self.password, config)
            
            # 학습 결과에 따른 다음 단계
            if accuracy >= 0.9:  # 정확도 요구치
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
        """다른 설정으로 학습 시도 - 통합 모델 버전"""
        # 모델 설정 변경
        params = self.generate_new_params()
        
        self.add_log("새로운 설정으로 학습 시도")
        
        # 통합된 모델 파일명 사용
        model_file = "emg_model.pth"
        
        # 학습 재시작
        self.training_thread = TrainingThread(
            self.username, self.password, 
            params, 
            model_file=model_file  # model_type 인자 제거
        )
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
        
        # 부모 클래스의 go_back 호출
        super().go_back()


class TrainingThread(QThread):
    """
    TrainingThread 클래스: 모델 학습을 담당하는 스레드
    - 백그라운드에서 모델 학습 수행
    - 학습 진행 상황과 결과를 위젯에 전달
    """
    progress_update = pyqtSignal(int, str)
    training_complete = pyqtSignal(bool, str, float)  # success, model_path, accuracy
    
    def __init__(self, username, password, model_params=None, model_type=None, model_file="emg_model.pth"):
        super().__init__()
        self.username = username
        self.password = password
        self.user_dir = get_user_dir(username, password)
        self.model_params = model_params or HIGH_PERFORMANCE_MODEL.copy()
        self.model_file = model_file
        
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
        """모든 사용자 데이터와 공통 데이터 로드 - 통합 버전"""
        # 모든 데이터 파일 가져오기
        user_data_files = [f for f in os.listdir(self.user_dir) if f.endswith('.csv')]
    
        # 공통 데이터 로드
        common_data_files = []
        if os.path.exists(COMMON_DATA_DIR):
            common_data_files = [f for f in os.listdir(COMMON_DATA_DIR) if f.endswith('.csv')]
    
        if not user_data_files and not common_data_files:
            return None, None
        
        all_X = []
        all_y = []
    
        # 사용자 데이터 로드
        for file in user_data_files:
            file_path = os.path.join(self.user_dir, file)
            self._load_file_data(file_path, all_X, all_y)
    
        # 공통 데이터 로드
        for file in common_data_files:
            file_path = os.path.join(COMMON_DATA_DIR, file)
            self._load_file_data(file_path, all_X, all_y)
    
        if all_X and all_y:
            return np.array(all_X), np.array(all_y)
        else:
            return None, None

    def _load_file_data(self, file_path, all_X, all_y):
        """CSV 파일에서 데이터 로드"""
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
            self.progress_update.emit(0, f"파일 '{os.path.basename(file_path)}' 로드 오류: {str(e)}")
            
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
        
        # 클래스 수 결정 - 통합 모델
        # 0(idle), 1-8(화면제어), 11-13(가위바위보)
        max_label = max(np.max(y_train), np.max(y_val), 13)
        num_classes = max_label + 1
        
        # 모델 생성
        model = EMGTransformer(
            input_dim=2,  # EMG 센서 2개
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            num_classes=num_classes,
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
        model_path = os.path.join(self.user_dir, self.model_file)
        
        # 모델 저장
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': self.model_params,
            'scaler': scaler,
            'accuracy': accuracy,
            'model_type': 'unified'
        }, model_path)
        
        return model_path




# ------------------------------------------------------------------------
# 게임 기본 위젯 - 가위바위보, 묵찌빠 게임의 공통 기능
# ------------------------------------------------------------------------
class BaseGameWidget(BaseEMGWidget):
    """
    BaseGameWidget 클래스: 게임 공통 기능을 제공하는 기본 위젯
    - 가위바위보와 묵찌빠 게임의 공통 기능을 포함
    - EMG 센서 처리 및 게임 상태 관리 기능 제공
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # 게임 상태 변수
        self.game_state = 0  # 0: 준비, 1: 카운트다운, 2: 게임 중, 3: 결과
        self.player_score = 0
        self.computer_score = 0
        self.round = 0
        self.countdown_value = 3
        self.player_choice = None
        self.computer_choice = None
        self.game_result = None
        
        # 타이머 설정
        self.countdown_timer = QTimer(self)
        self.countdown_timer.timeout.connect(self.update_countdown)
        
        self.game_timer = QTimer(self)
        self.game_timer.timeout.connect(self.update_game)
        
        self.detection_timer = QTimer(self)
        self.detection_timer.timeout.connect(self.update_detection)
        
    def set_user(self, username, password):
        """사용자 정보 설정"""
        super().set_user(username, password)
        
        # 통합 모델 로드
        model_path = os.path.join(self.user_dir, "emg_model.pth")
        self.predictor = EMGPredictor(model_path)
        
    def start(self):
        """게임 시작 준비"""
        self.reset_game()
        
        # 전역 시리얼 스레드에 구독
        if self.subscribe_to_serial():
            self.is_active = True
            self.detection_timer.start(100)
        else:
            # 시리얼 연결 실패 처리
            QMessageBox.warning(self, "연결 오류", "EMG 센서 연결에 실패했습니다.")
        
    def stop(self):
        """게임 중지"""
        self.is_active = False
        self.unsubscribe_from_serial()
        self.countdown_timer.stop()
        self.game_timer.stop()
        self.detection_timer.stop()
        
    def reset_game(self):
        """게임 상태 초기화"""
        self.game_state = 0
        self.player_score = 0
        self.computer_score = 0
        self.round = 0
        self.player_choice = None
        self.computer_choice = None
        self.game_result = None
        
    def process_data(self, s1, s2):
        """EMG 데이터 처리"""
        if not self.is_active:
            return
        # 게임 결과 표시 중(game_state=3)일 때는 동작 감지 처리 중지
        if self.game_state == 3:
            return    
        result = self.emg_processor.process_data(s1, s2)
        
        # 동작 감지 중일 때만 처리
        if self.game_state == 2 and isinstance(result, tuple) and result[0] == "movement_completed":
            sequence = result[1]
            # 동작 예측
            prediction, confidence = self.predictor.predict(sequence)
            
            # 예측 결과 디버깅 출력
            print(f"예측: 라벨 {prediction}, 신뢰도 {confidence:.2f}")
            
            # 다음과 같이 수정:
            if confidence >= self.confidence_threshold:
                # 🎯 가위바위보 라벨(11,12,13)만 허용
                if prediction in (LABEL_SCISSORS, LABEL_ROCK, LABEL_PAPER):
                    self.player_choice = prediction
                    self.update_player_choice()
                    print(f"플레이어 선택: {get_label_name(prediction)}")
                elif prediction == LABEL_IDLE:
                    print(f"대기 상태 감지됨 (신뢰도: {confidence:.2f}) - 무시")
                else:
                    print(f"게임에서 사용하지 않는 동작: 라벨 {prediction} - 무시")
                
                # 대기 상태도 로그에는 기록
                if prediction == LABEL_IDLE:
                    print(f"대기 상태 감지됨 (신뢰도: {confidence:.2f})")
    
    # 자식 클래스에서 구현할 메서드들
    def update_player_choice(self):
        """플레이어 선택 업데이트 - 자식 클래스에서 구현"""
        pass
            
    def update_computer_choice(self):
        """컴퓨터 선택 업데이트 - 자식 클래스에서 구현"""
        pass
            
    def start_game(self):
        """게임 시작 - 자식 클래스에서 구현"""
        pass
            
    def start_countdown(self):
        """카운트다운 시작 - 자식 클래스에서 구현"""
        pass
            
    def update_countdown(self):
        """카운트다운 업데이트 - 자식 클래스에서 구현"""
        pass
            
    def update_game(self):
        """게임 상태 업데이트 - 자식 클래스에서 구현"""
        pass
            
    def check_game_end(self):
        """게임 종료 확인 - 자식 클래스에서 구현"""
        pass
            
    def update_detection(self):
        """동작 감지 상태 업데이트"""
        # 게임 상태에 따른 처리
        if self.game_state == 1 or self.game_state == 2:
            # 게임 중일 때만 플레이어 선택 표시
            if self.player_choice:
                self.update_player_choice()


# ------------------------------------------------------------------------
# 가위바위보 게임 위젯
# ------------------------------------------------------------------------
class RockPaperScissorsGameWidget(BaseGameWidget):
    """
    RockPaperScissorsGameWidget 클래스: 가위바위보 게임 화면을 제공하는 위젯
    - EMG 센서 기반 가위바위보 게임 제공
    - 사용자와 컴퓨터 간의 대결
    - 네오모피즘 디자인 적용
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.max_rounds = 10
        self.init_ui()
        self.audio_manager = GameAudioManager()
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
        self.player_label.setStyleSheet(f"color: {PRIMARY_COLOR}; font-size: 32px; font-weight: bold;")
        self.player_label.setAlignment(Qt.AlignCenter)
        
        
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
        self.player_choice_label.setStyleSheet("font-size: 90px;")
        self.player_choice_label.setAlignment(Qt.AlignCenter)
        
        player_choice_layout.addWidget(self.player_choice_label)
        
        player_layout.addWidget(self.player_label)
        #player_layout.addWidget(player_status_card)  # 이 부분 추가
        player_layout.addWidget(player_choice_card, 0, Qt.AlignCenter)
        
        # 중앙 영역
        center_layout = QVBoxLayout()
        
        # VS 레이블 - 이 부분을 수정하여 VS 텍스트를 변경할 수 있습니다
        self.vs_label = QLabel("VS")
        self.vs_label.setStyleSheet(f"color: {TEXT_COLOR}; font-size: 40px; font-weight: bold;")
        self.vs_label.setAlignment(Qt.AlignCenter)
        
        # 카운트다운 표시
        countdown_card = NeumorphicCard()
        countdown_card.setFixedSize(80, 80)
        countdown_card.setStyleSheet(f"""
            background-color: {SECONDARY_COLOR};
            border-radius: 50px;
        """)
        
        countdown_layout = QVBoxLayout(countdown_card)
        
        # 카운트다운 레이블 - 이 부분을 수정하여 카운트다운 텍스트를 변경할 수 있습니다
        self.countdown_label = QLabel("3")
        self.countdown_label.setStyleSheet("font-size: 44px; font-weight: bold; color: white;")
        self.countdown_label.setAlignment(Qt.AlignCenter)
        
        countdown_layout.addWidget(self.countdown_label)
        
        # 상태 표시
        status_card = NeumorphicCard(inset=True)
        status_layout = QVBoxLayout(status_card)
        
        # 상태 레이블 - 이 부분을 수정하여 상태 텍스트를 변경할 수 있습니다
        self.status_label = QLabel("게임을 시작하려면 '시작' 버튼을 누르세요")
        self.status_label.setStyleSheet(f"color: {TEXT_COLOR}; font-size: 32px;")
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
        self.computer_label.setStyleSheet("color: #F5365C; font-size: 32px; font-weight: bold;")
        self.computer_label.setAlignment(Qt.AlignCenter)
        
        
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
        self.computer_choice_label.setStyleSheet("font-size: 90px;")
        self.computer_choice_label.setAlignment(Qt.AlignCenter)
        
        computer_choice_layout.addWidget(self.computer_choice_label)
        
        computer_layout.addWidget(self.computer_label)
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
      
        
    # 부모 클래스에서 오버라이드한 메서드들
    def reset_game(self):
        """게임 상태 초기화"""
        super().reset_game()
        
        # UI 업데이트
        self.player_score_label.setText(f"플레이어: {self.player_score}")
        self.computer_score_label.setText(f"컴퓨터: {self.computer_score}")
        self.round_label.setText(f"라운드: {self.round}/{self.max_rounds}")
        
        self.player_choice_label.setText("?")
        self.computer_choice_label.setText("?")
        
        self.result_label.setText("")
        self.status_label.setText("게임을 시작하려면 '시작' 버튼을 누르세요")
        
        self.countdown_card.setVisible(False)
        self.start_btn.setEnabled(True)
        self.start_btn.setText("시작")
        
    def update_player_choice(self):
        """플레이어 선택 업데이트"""
        if self.player_choice == LABEL_IDLE:
            self.player_choice_label.setText("🖐️")  # 대기 상태
        elif self.player_choice == LABEL_SCISSORS:
            self.player_choice_label.setText("✌️")  # 가위
        elif self.player_choice == LABEL_ROCK:
            self.player_choice_label.setText("✊")  # 바위
        elif self.player_choice == LABEL_PAPER:
            self.player_choice_label.setText("✋")  # 보
            
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
            
            # 게임 상태 업데이트
            self.game_state = 1
            self.player_choice = None
            self.computer_choice = None
            
            self.start_countdown()
            self.start_btn.setEnabled(False)
            
    def start_countdown(self):
        """카운트다운 시작"""
        self.countdown_value = 3
        self.countdown_label.setText(str(self.countdown_value))
        self.countdown_card.setVisible(True)
        self.status_label.setText("준비...")
        # 가위바위보 시작 음성 재생
        self.audio_manager.play_sound("ready")
        self.countdown_timer.start(1000)
        
    def update_countdown(self):
        """카운트다운 업데이트 - 가위바위보 통합 음성"""
        self.countdown_value -= 1
        self.countdown_label.setText(str(self.countdown_value))
        
        if self.countdown_value <= 0:
            self.countdown_timer.stop()
            
            # 카운트다운 끝, "가위바위보" 음성 재생
            self.countdown_card.setVisible(False)
            self.status_label.setText("가위바위보!")
            
            # "가위바위보" 음성 재생
            self.audio_manager.play_sound("rockpaperscissors")  # 또는 "가위바위보"
                
            # 플레이어 선택 초기화
            self.player_choice = None
            self.player_choice_label.setText("?")
            
            # 컴퓨터 선택 초기화
            self.computer_choice = None
            self.computer_choice_label.setText("?")
            
            # 음성 재생 후 1초 뒤에 결과 확인하도록 타이머 설정
            self.game_timer.start(3000)  # 1초로 변경
            
    def update_game(self):
        """게임 상태 업데이트 - 음성 추가"""
        self.game_timer.stop()
        
        # 컴퓨터 선택
        self.computer_choice = get_computer_choice()
        self.update_computer_choice()
        
        # 플레이어가 선택하지 않았거나 대기 상태면 랜덤 선택
        if self.player_choice is None or self.player_choice == LABEL_IDLE:
            self.status_label.setText("시간 초과! 랜덤으로 선택됩니다.")
            self.player_choice = get_computer_choice()  # 랜덤 선택
            self.update_player_choice()
            
        # 선택 표시
        player_choice_text = get_label_name(self.player_choice)
        computer_choice_text = get_label_name(self.computer_choice)
        self.status_label.setText(f"{player_choice_text} vs {computer_choice_text}")
    
        # 가위바위보 결과 확인
        result = determine_winner(self.player_choice, self.computer_choice)
        # 게임 상태를 결과 표시로 변경
        self.game_state = 3
        if result == RESULT_WIN:
            # 플레이어 승리
            self.player_score += 1
            self.player_score_label.setText(f"플레이어: {self.player_score}")
            self.result_label.setText("플레이어 승리!")
            self.result_label.setStyleSheet("font-size: 32px; font-weight: bold; color: #2DCE89;")
            # 승리 음성 재생
            self.audio_manager.play_sound("win")
            
        elif result == RESULT_LOSE:
            # 컴퓨터 승리
            self.computer_score += 1
            self.computer_score_label.setText(f"컴퓨터: {self.computer_score}")
            self.result_label.setText("컴퓨터 승리!")
            self.result_label.setStyleSheet("font-size: 32px; font-weight: bold; color: #F5365C;")
            # 패배 음성 재생
            self.audio_manager.play_sound("lose")
            
        else:
            # 무승부
            self.result_label.setText("무승부!")
            self.result_label.setStyleSheet("font-size: 32px; font-weight: bold; color: #FB8C00;")
            # 무승부 음성 재생
            self.audio_manager.play_sound("draw")
    
        # 라운드 종료, 다음 라운드 확인
        QTimer.singleShot(2000, self.check_game_end)
                
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
            QTimer.singleShot(1000, self.next_round)
            
    def next_round(self):
        """다음 라운드 시작"""
        # 게임 상태 초기화
        self.game_state = 0
        self.player_choice = None
        self.computer_choice = None
        self.result_label.setText("")
        self.player_choice_label.setText("?")
        self.computer_choice_label.setText("?")
    
        # 다음 라운드 시작
        self.start_game()


# ------------------------------------------------------------------------
# 묵찌빠 게임 위젯
# ------------------------------------------------------------------------
class MukJjiPpaGameWidget(BaseGameWidget):
    """
    MukJjiPpaGameWidget 클래스: 묵찌빠 게임 화면을 제공하는 위젯
    - EMG 센서 기반 묵찌빠 게임 제공
    - 가위바위보로 선공을 결정한 뒤 묵찌빠 게임 진행
    - 네오모피즘 디자인 적용
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.max_rounds = 5
        self.current_attacker = None  # 현재 공격자 (True: 플레이어, False: 컴퓨터)
        self.init_ui()
        self.audio_manager = GameAudioManager()
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
        
        # 상태 레이블 -이 부분을 수정하여 상태 텍스트를 변경할 수 있습니다
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
      
        
    # 부모 클래스에서 오버라이드한 메서드들
    def reset_game(self):
        """게임 상태 초기화"""
        super().reset_game()
        
        # 묵찌빠 게임 전용 변수 초기화
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
        
    def update_player_choice(self):
        """플레이어 선택 업데이트"""
        if self.player_choice == LABEL_IDLE:
            self.player_choice_label.setText("🖐️")  # 대기 상태
        elif self.player_choice == LABEL_SCISSORS:
            self.player_choice_label.setText("✌️")  # 가위
        elif self.player_choice == LABEL_ROCK:
            self.player_choice_label.setText("✊")  # 바위
        elif self.player_choice == LABEL_PAPER:
            self.player_choice_label.setText("✋")  # 보
            
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
    
    # 나머지 메서드들은 기존 코드와 동일
    def start_countdown(self):
        """카운트다운 시작"""
        self.countdown_value = 3
        self.countdown_label.setText(str(self.countdown_value))
        self.countdown_card.setVisible(True)
        self.status_label.setText("준비...")
        # 게임 상태에 따른 음성 재생
        if self.game_state == 1:
            self.audio_manager.play_sound("ready")  # 가위바위보 준비
        else:
            self.audio_manager.play_sound("mukjjippa_ready")  # 묵찌빠 준비
        self.countdown_timer.start(1000)
        
    def update_countdown(self):
        """카운트다운 업데이트 - 통합 음성"""
        self.countdown_value -= 1
        
        if self.countdown_value > 0:
            self.countdown_label.setText(str(self.countdown_value))
        else:
            self.countdown_timer.stop()
            
            # 카운트다운 끝, 게임 시작
            self.countdown_card.setVisible(False)
            
            if self.game_state == 1:
                # 가위바위보 - 통합 음성 재생
                self.status_label.setText("가위바위보!")
                self.audio_manager.play_sound("rockpaperscissors")
            else:
                # 묵찌빠 - 통합 음성 재생
                self.status_label.setText("묵찌빠!")
                self.audio_manager.play_sound("mukjjippa")  # 묵찌빠 통합 음성
                
            # 플레이어 선택 초기화
            self.player_choice = None
            self.player_choice_label.setText("?")
            
            # 컴퓨터 선택 초기화
            self.computer_choice = None
            self.computer_choice_label.setText("?")
            
            # 음성 재생 후 1초 뒤에 결과 확인하도록 타이머 설정
            self.game_timer.start(3000)  # 1초로 변경
            
    def update_game(self):
        """게임 상태 업데이트"""
        self.game_timer.stop()
        
        # 컴퓨터 선택
        self.computer_choice = get_computer_choice()
        self.update_computer_choice()
        
        # 플레이어가 선택하지 않았거나 대기 상태면 랜덤 선택
        if self.player_choice is None or self.player_choice == LABEL_IDLE:
            self.status_label.setText("시간 초과! 랜덤으로 선택됩니다.")
            self.player_choice = get_computer_choice()  # 랜덤 선택
            self.update_player_choice()
            
        # 가위바위보 결과 확인
        if self.game_state == 1:
            # 첫 판 가위바위보
            result = determine_winner(self.player_choice, self.computer_choice)
            self.game_state = 3
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
            self.game_state = 3
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
        # 게임 상태 초기화
        self.game_state = 0
        self.player_choice = None
        self.computer_choice = None
        self.result_label.setText("")
        self.player_choice_label.setText("?")
        self.computer_choice_label.setText("?")
    
        # 다음 라운드 시작
        self.start_game()



# ------------------------------------------------------------------------
# 설정 위젯
# ------------------------------------------------------------------------
class SettingsWidget(BaseEMGWidget):
    """
    SettingsWidget 클래스: 애플리케이션 설정 화면을 제공하는 위젯
    - EMG 센서 감도 및 포트 설정
    - 게임 설정 제공
    - 동작 인식 세부 설정 제공
    - 네오모피즘 디자인 적용
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
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
        
        # 제목 텍스트
        self.title_label = QLabel("설정")
        self.title_label.setStyleSheet(f"color: {TEXT_COLOR}; font-size: 28px; font-weight: bold;")
        self.title_label.setAlignment(Qt.AlignCenter)
        
        title_layout.addWidget(self.title_label)
        
        # 설정 탭 위젯
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
        
        # EMG 센서 설정 탭
        sensor_tab = QWidget()
        sensor_layout = QVBoxLayout(sensor_tab)
        
        # 센서 설정 카드
        sensor_card = NeumorphicCard()
        sensor_card_layout = QVBoxLayout(sensor_card)
        
        # 연결 방식 설정
        connection_type_frame = QFrame()
        connection_type_layout = QHBoxLayout(connection_type_frame)

        # 연결 방식 레이블
        connection_type_label = QLabel("연결 방식:")
        connection_type_label.setStyleSheet(f"color: {TEXT_COLOR}; font-size: 16px; font-weight: bold;")

        # 연결 방식 라디오 버튼
        self.usb_radio = QRadioButton("USB 케이블")
        self.usb_radio.setStyleSheet(f"color: {TEXT_COLOR}; font-size: 16px;")
        self.usb_radio.setChecked(True)
        self.usb_radio.toggled.connect(self.on_connection_type_changed)

        self.bluetooth_radio = QRadioButton("블루투스")
        self.bluetooth_radio.setStyleSheet(f"color: {TEXT_COLOR}; font-size: 16px;")
        self.bluetooth_radio.toggled.connect(self.on_connection_type_changed)

        # 레이아웃에 추가
        connection_type_layout.addWidget(connection_type_label)
        connection_type_layout.addWidget(self.usb_radio)
        connection_type_layout.addWidget(self.bluetooth_radio)

        # 센서 카드 레이아웃에 추가 (포트 설정 전에)
        sensor_card_layout.addWidget(connection_type_frame)
        
        # 포트 설정
        port_frame = QFrame()
        port_layout = QHBoxLayout(port_frame)
        
        # 포트 레이블
        port_label = QLabel("시리얼 포트:")
        port_label.setStyleSheet(f"color: {TEXT_COLOR}; font-size: 16px; font-weight: bold;")
        
        # 포트 선택 콤보박스
        self.port_combo = QComboBox()
        self.port_combo.setStyleSheet(f"""
            QComboBox {{
                background-color: {BG_COLOR};
                border: 1px solid #D1D9E6;
                border-radius: 8px;
                padding: 8px;
                font-size: 16px;
                color: {TEXT_COLOR};
            }}
            QComboBox::drop-down {{
                border: none;
                padding-right: 15px;
            }}
        """)
        
        # 포트 새로고침 버튼
        self.refresh_port_btn = NeumorphicButton("새로고침")
        self.refresh_port_btn.clicked.connect(self.refresh_ports)
        
        port_layout.addWidget(port_label)
        port_layout.addWidget(self.port_combo, 1)
        port_layout.addWidget(self.refresh_port_btn)
        
        # 통신 속도 설정
        baud_frame = QFrame()
        baud_layout = QHBoxLayout(baud_frame)
        
        # 통신 속도 레이블
        baud_label = QLabel("통신 속도:")
        baud_label.setStyleSheet(f"color: {TEXT_COLOR}; font-size: 16px; font-weight: bold;")
        
        # 통신 속도 표시 레이블 (콤보박스 대신)
        self.baud_value_label = QLabel("115200") # 기본값은 USB 통신 속도
        self.baud_value_label.setStyleSheet(f"""
            color: {TEXT_COLOR};
            font-size: 16px;
            background-color: {BG_COLOR};
            border: 1px solid #D1D9E6;
            border-radius: 8px;
            padding: 8px;
        """)

        baud_layout.addWidget(baud_label)
        baud_layout.addWidget(self.baud_value_label, 1)
        
        # 감도 설정
        sensitivity_frame = QFrame()
        sensitivity_layout = QHBoxLayout(sensitivity_frame)
        
        # 감도 레이블
        sensitivity_label = QLabel("감도 설정:")
        sensitivity_label.setStyleSheet(f"color: {TEXT_COLOR}; font-size: 16px; font-weight: bold;")
        
        # 감도 슬라이더
        self.sensitivity_slider = QDoubleSpinBox()
        self.sensitivity_slider.setRange(0.5, 2.0)
        self.sensitivity_slider.setSingleStep(0.1)
        self.sensitivity_slider.setValue(1.0)
        self.sensitivity_slider.setStyleSheet(f"""
            QDoubleSpinBox {{
                background-color: {BG_COLOR};
                border: 1px solid #D1D9E6;
                border-radius: 8px;
                padding: 8px;
                font-size: 16px;
                color: {TEXT_COLOR};
            }}
        """)
        
        sensitivity_layout.addWidget(sensitivity_label)
        sensitivity_layout.addWidget(self.sensitivity_slider, 1)
        
        # 연결 테스트 버튼
        test_button = NeumorphicButton("연결 테스트", PRIMARY_COLOR)
        test_button.clicked.connect(self.test_connection)
        
        # 센서 카드에 위젯 추가
        sensor_card_layout.addWidget(port_frame)
        sensor_card_layout.addWidget(baud_frame)
        sensor_card_layout.addWidget(sensitivity_frame)
        sensor_card_layout.addWidget(test_button)
        
        # 센서 탭에 카드 추가
        sensor_layout.addWidget(sensor_card)
        sensor_layout.addStretch()
        
        # 동작 인식 설정 탭
        recognition_tab = QWidget()
        recognition_layout = QVBoxLayout(recognition_tab)
        
        # 동작 인식 설정 카드
        recognition_card = NeumorphicCard()
        recognition_card_layout = QVBoxLayout(recognition_card)
        
        # 설정 설명
        recognition_desc = QLabel("동작 인식 세부 설정을 조정하여 인식 정확도를 향상시킬 수 있습니다.")
        recognition_desc.setStyleSheet(f"color: {TEXT_COLOR}; font-size: 16px;")
        recognition_desc.setWordWrap(True)
        
        # 동작 감지 임계값 설정
        detection_threshold_frame = QFrame()
        detection_threshold_layout = QHBoxLayout(detection_threshold_frame)
        
        # 임계값 레이블
        detection_threshold_label = QLabel("동작 감지 임계값:")
        detection_threshold_label.setStyleSheet(f"color: {TEXT_COLOR}; font-size: 16px; font-weight: bold;")
        detection_threshold_label.setToolTip("값이 클수록 더 강한 동작만 감지합니다.")
        
        # 임계값 슬라이더
        self.detection_threshold_slider = QDoubleSpinBox()
        self.detection_threshold_slider.setRange(3.0, 10.0)
        self.detection_threshold_slider.setSingleStep(0.5)
        self.detection_threshold_slider.setValue(DETECTION_THRESHOLD)
        self.detection_threshold_slider.setStyleSheet(f"""
            QDoubleSpinBox {{
                background-color: {BG_COLOR};
                border: 1px solid #D1D9E6;
                border-radius: 8px;
                padding: 8px;
                font-size: 16px;
                color: {TEXT_COLOR};
            }}
        """)
        
        detection_threshold_layout.addWidget(detection_threshold_label)
        detection_threshold_layout.addWidget(self.detection_threshold_slider, 1)
        
        # 추세 감지 임계값 설정
        trend_threshold_frame = QFrame()
        trend_threshold_layout = QHBoxLayout(trend_threshold_frame)
        
        # 추세 임계값 레이블
        trend_threshold_label = QLabel("추세 감지 임계값:")
        trend_threshold_label.setStyleSheet(f"color: {TEXT_COLOR}; font-size: 16px; font-weight: bold;")
        trend_threshold_label.setToolTip("값이 클수록 더 빠른 변화만 감지합니다.")
        
        # 추세 임계값 슬라이더
        self.trend_threshold_slider = QDoubleSpinBox()
        self.trend_threshold_slider.setRange(2.0, 8.0)
        self.trend_threshold_slider.setSingleStep(0.5)
        self.trend_threshold_slider.setValue(TREND_THRESHOLD)
        self.trend_threshold_slider.setStyleSheet(f"""
            QDoubleSpinBox {{
                background-color: {BG_COLOR};
                border: 1px solid #D1D9E6;
                border-radius: 8px;
                padding: 8px;
                font-size: 16px;
                color: {TEXT_COLOR};
            }}
        """)
        
        trend_threshold_layout.addWidget(trend_threshold_label)
        trend_threshold_layout.addWidget(self.trend_threshold_slider, 1)
        
        # 분산 감지 임계값 설정
        dispersion_threshold_frame = QFrame()
        dispersion_threshold_layout = QHBoxLayout(dispersion_threshold_frame)
        
        # 분산 임계값 레이블
        dispersion_threshold_label = QLabel("분산 감지 임계값:")
        dispersion_threshold_label.setStyleSheet(f"color: {TEXT_COLOR}; font-size: 16px; font-weight: bold;")
        dispersion_threshold_label.setToolTip("값이 클수록 더 불규칙한 변화만 감지합니다.")
        
        # 분산 임계값 슬라이더
        self.dispersion_threshold_slider = QDoubleSpinBox()
        self.dispersion_threshold_slider.setRange(2.0, 8.0)
        self.dispersion_threshold_slider.setSingleStep(0.5)
        self.dispersion_threshold_slider.setValue(DISPERSION_THRESHOLD)
        self.dispersion_threshold_slider.setStyleSheet(f"""
            QDoubleSpinBox {{
                background-color: {BG_COLOR};
                border: 1px solid #D1D9E6;
                border-radius: 8px;
                padding: 8px;
                font-size: 16px;
                color: {TEXT_COLOR};
            }}
        """)
        
        dispersion_threshold_layout.addWidget(dispersion_threshold_label)
        dispersion_threshold_layout.addWidget(self.dispersion_threshold_slider, 1)
        
        # 쿨다운 설정
        cooldown_frame = QFrame()
        cooldown_layout = QHBoxLayout(cooldown_frame)
        
        # 쿨다운 레이블
        cooldown_label = QLabel("동작 쿨다운 시간 (초):")
        cooldown_label.setStyleSheet(f"color: {TEXT_COLOR}; font-size: 16px; font-weight: bold;")
        cooldown_label.setToolTip("동작 감지 후 다음 동작을 감지할 때까지의 대기 시간")
        
        # 쿨다운 슬라이더
        self.cooldown_slider = QDoubleSpinBox()
        self.cooldown_slider.setRange(1.0, 5.0)
        self.cooldown_slider.setSingleStep(0.5)
        self.cooldown_slider.setValue(COOLDOWN_PERIOD)
        self.cooldown_slider.setStyleSheet(f"""
            QDoubleSpinBox {{
                background-color: {BG_COLOR};
                border: 1px solid #D1D9E6;
                border-radius: 8px;
                padding: 8px;
                font-size: 16px;
                color: {TEXT_COLOR};
            }}
        """)
        
        cooldown_layout.addWidget(cooldown_label)
        cooldown_layout.addWidget(self.cooldown_slider, 1)
        
        # 신뢰도 임계값 설정
        confidence_frame = QFrame()
        confidence_layout = QHBoxLayout(confidence_frame)
        
        # 신뢰도 레이블
        confidence_label = QLabel("모델 신뢰도 임계값:")
        confidence_label.setStyleSheet(f"color: {TEXT_COLOR}; font-size: 16px; font-weight: bold;")
        confidence_label.setToolTip("값이 클수록 더 높은 신뢰도의 예측만 수용합니다.")
        
        # 신뢰도 슬라이더
        self.confidence_slider = QDoubleSpinBox()
        self.confidence_slider.setRange(0.3, 0.9)
        self.confidence_slider.setSingleStep(0.05)
        self.confidence_slider.setValue(0.5)  # 기본값
        self.confidence_slider.setStyleSheet(f"""
            QDoubleSpinBox {{
                background-color: {BG_COLOR};
                border: 1px solid #D1D9E6;
                border-radius: 8px;
                padding: 8px;
                font-size: 16px;
                color: {TEXT_COLOR};
            }}
        """)
        
        confidence_layout.addWidget(confidence_label)
        confidence_layout.addWidget(self.confidence_slider, 1)
        
        # 기본값 복원 버튼
        reset_defaults_button = NeumorphicButton("기본값으로 복원", SECONDARY_COLOR)
        reset_defaults_button.clicked.connect(self.reset_recognition_defaults)
        
        # 설정 저장 버튼
        save_recognition_button = NeumorphicButton("설정 저장", PRIMARY_COLOR)
        save_recognition_button.clicked.connect(self.save_recognition_settings)
        
        # 버튼 레이아웃
        recognition_buttons_layout = QHBoxLayout()
        recognition_buttons_layout.addWidget(reset_defaults_button)
        recognition_buttons_layout.addStretch()
        recognition_buttons_layout.addWidget(save_recognition_button)
        
        # 동작 인식 카드에 위젯 추가
        recognition_card_layout.addWidget(recognition_desc)
        recognition_card_layout.addWidget(detection_threshold_frame)
        recognition_card_layout.addWidget(trend_threshold_frame)
        recognition_card_layout.addWidget(dispersion_threshold_frame)
        recognition_card_layout.addWidget(cooldown_frame)
        recognition_card_layout.addWidget(confidence_frame)
        recognition_card_layout.addLayout(recognition_buttons_layout)
        
        # 동작 인식 탭에 카드 추가
        recognition_layout.addWidget(recognition_card)
        recognition_layout.addStretch()
        
        # 게임 설정 탭
        game_tab = QWidget()
        game_layout = QVBoxLayout(game_tab)
        
        # 게임 설정 카드
        game_card = NeumorphicCard()
        game_card_layout = QVBoxLayout(game_card)
        
        # 가위바위보 라운드 설정
        rps_round_frame = QFrame()
        rps_round_layout = QHBoxLayout(rps_round_frame)
        
        # 라운드 레이블
        rps_round_label = QLabel("가위바위보 라운드:")
        rps_round_label.setStyleSheet(f"color: {TEXT_COLOR}; font-size: 16px; font-weight: bold;")
        
        # 라운드 선택 콤보박스
        self.rps_round_combo = QComboBox()
        self.rps_round_combo.setStyleSheet(f"""
            QComboBox {{
                background-color: {BG_COLOR};
                border: 1px solid #D1D9E6;
                border-radius: 8px;
                padding: 8px;
                font-size: 16px;
                color: {TEXT_COLOR};
            }}
            QComboBox::drop-down {{
                border: none;
                padding-right: 15px;
            }}
        """)
        
        self.rps_round_combo.addItems(["3","5", "10", "15"])
        self.rps_round_combo.setCurrentText("10")
        
        rps_round_layout.addWidget(rps_round_label)
        rps_round_layout.addWidget(self.rps_round_combo, 1)
        
        # 묵찌빠 라운드 설정
        muk_round_frame = QFrame()
        muk_round_layout = QHBoxLayout(muk_round_frame)
        
        # 라운드 레이블
        muk_round_label = QLabel("묵찌빠 라운드:")
        muk_round_label.setStyleSheet(f"color: {TEXT_COLOR}; font-size: 16px; font-weight: bold;")
        
        # 라운드 선택 콤보박스
        self.muk_round_combo = QComboBox()
        self.muk_round_combo.setStyleSheet(f"""
            QComboBox {{
                background-color: {BG_COLOR};
                border: 1px solid #D1D9E6;
                border-radius: 8px;
                padding: 8px;
                font-size: 16px;
                color: {TEXT_COLOR};
            }}
            QComboBox::drop-down {{
                border: none;
                padding-right: 15px;
            }}
        """)
        
        self.muk_round_combo.addItems(["3", "5", "7", "9"])
        self.muk_round_combo.setCurrentText("5")
        
        muk_round_layout.addWidget(muk_round_label)
        muk_round_layout.addWidget(self.muk_round_combo, 1)
        
        # 설정 저장 버튼
        save_button = NeumorphicButton("설정 저장", PRIMARY_COLOR)
        save_button.clicked.connect(self.save_settings)
        
        # 게임 카드에 위젯 추가
        game_card_layout.addWidget(rps_round_frame)
        game_card_layout.addWidget(muk_round_frame)
        game_card_layout.addWidget(save_button)
        
        # 게임 탭에 카드 추가
        game_layout.addWidget(game_card)
        game_layout.addStretch()
        
        # 외관 설정 탭 (새로 추가)
        appearance_tab = QWidget()
        appearance_layout = QVBoxLayout(appearance_tab)
        
        # 화면 크기 설정 카드
        screen_card = NeumorphicCard()
        screen_layout = QVBoxLayout(screen_card)
        screen_layout.setContentsMargins(20, 20, 20, 20)
        
        # 화면 크기 설정 제목
        screen_title = QLabel("화면 크기 설정")
        screen_title.setStyleSheet(f"color: {TEXT_COLOR}; font-size: 18px; font-weight: bold;")
        screen_title.setAlignment(Qt.AlignCenter)
        
        # 현재 화면 크기 표시
        current_size_frame = QFrame()
        current_size_layout = QHBoxLayout(current_size_frame)
        
        current_size_label = QLabel("현재 화면 크기:")
        current_size_label.setStyleSheet(f"color: {TEXT_COLOR}; font-size: 16px; font-weight: bold;")
        
        self.current_size_value = QLabel()
        self.current_size_value.setStyleSheet(f"color: {PRIMARY_COLOR}; font-size: 16px;")
        
        # 현재 크기 업데이트
        window_size = self.window().size()
        self.current_size_value.setText(f"{window_size.width()} x {window_size.height()}")
        
        current_size_layout.addWidget(current_size_label)
        current_size_layout.addWidget(self.current_size_value)
        current_size_layout.addStretch()
        
        # 미리 정의된 화면 크기 버튼
        preset_frame = QFrame()
        preset_layout = QGridLayout(preset_frame)
        preset_layout.setHorizontalSpacing(20)
        preset_layout.setVerticalSpacing(15)
        
        preset_sizes = [
            ("소형", 800, 600),
            ("중형", 1024, 768),
            ("대형", 1280, 800),
            ("와이드", 1440, 900),
            ("풀사이즈", 1920, 1080)
        ]
        
        # 현재 행 설정
        row, col = 0, 0
        max_cols = 3  # 한 행에 표시할 최대 열 수
        
        for name, width, height in preset_sizes:
            # 버튼 생성
            size_btn = NeumorphicButton(f"{name} ({width}x{height})")
            size_btn.clicked.connect(lambda checked, w=width, h=height: self.set_window_size(w, h))
            
            # 그리드에 추가
            preset_layout.addWidget(size_btn, row, col)
            
            # 다음 위치 계산
            col += 1
            if col >= max_cols:
                col = 0
                row += 1
        
        # 폰트 크기 설정
        font_frame = QFrame()
        font_layout = QHBoxLayout(font_frame)
        
        font_label = QLabel("폰트 크기 조절:")
        font_label.setStyleSheet(f"color: {TEXT_COLOR}; font-size: 16px; font-weight: bold;")
        
        # 폰트 크기 슬라이더
        self.font_scale_slider = QSlider(Qt.Horizontal)
        self.font_scale_slider.setRange(80, 120)
        self.font_scale_slider.setValue(100)
        self.font_scale_slider.setTickPosition(QSlider.TicksBelow)
        self.font_scale_slider.setTickInterval(10)
        self.font_scale_slider.setStyleSheet(f"""
            QSlider::groove:horizontal {{
                height: 8px;
                background: #D1D9E6;
                border-radius: 4px;
            }}
            QSlider::handle:horizontal {{
                background: {PRIMARY_COLOR};
                border: none;
                width: 16px;
                margin-top: -4px;
                margin-bottom: -4px;
                border-radius: 8px;
            }}
            QSlider::sub-page:horizontal {{
                background: {PRIMARY_COLOR};
                border-radius: 4px;
            }}
        """)
        
        # 현재 폰트 크기 표시
        self.font_scale_value = QLabel("100%")
        self.font_scale_value.setStyleSheet(f"color: {PRIMARY_COLOR}; font-size: 16px; min-width: 50px;")
        
        # 슬라이더 값 변경 시 이벤트 연결
        self.font_scale_slider.valueChanged.connect(self.update_font_scale)
        
        font_layout.addWidget(font_label)
        font_layout.addWidget(self.font_scale_slider)
        font_layout.addWidget(self.font_scale_value)
        
        # 카드에 위젯 추가
        screen_layout.addWidget(screen_title)
        screen_layout.addWidget(current_size_frame)
        screen_layout.addWidget(preset_frame)
        screen_layout.addWidget(font_frame)
        
        # 외관 탭에 카드 추가
        appearance_layout.addWidget(screen_card)
        appearance_layout.addStretch()
        
        # 사용자 정보 탭
        user_tab = QWidget()
        user_layout = QVBoxLayout(user_tab)
        
        # 사용자 정보 카드
        user_card = NeumorphicCard()
        user_card_layout = QVBoxLayout(user_card)
        
        # 현재 사용자 정보 표시
        user_info_frame = NeumorphicCard(inset=True)
        user_info_layout = QVBoxLayout(user_info_frame)
        
        # 사용자 레이블
        self.user_info_label = QLabel("현재 사용자: ")
        self.user_info_label.setStyleSheet(f"color: {TEXT_COLOR}; font-size: 16px; font-weight: bold;")
        
        # 모델 정보 레이블
        self.model_info_label = QLabel("모델 정확도: ")
        self.model_info_label.setStyleSheet(f"color: {TEXT_COLOR}; font-size: 16px;")
        
        user_info_layout.addWidget(self.user_info_label)
        user_info_layout.addWidget(self.model_info_label)
        
        # 사용자 데이터 관리 섹션
        user_data_frame = QFrame()
        user_data_layout = QVBoxLayout(user_data_frame)
        
        # 데이터 관리 레이블
        user_data_label = QLabel("데이터 관리")
        user_data_label.setStyleSheet(f"color: {TEXT_COLOR}; font-size: 18px; font-weight: bold; margin-top: 20px;")
        
        # 데이터 초기화 버튼
        reset_data_button = NeumorphicButton("데이터 초기화", DANGER_COLOR)
        reset_data_button.clicked.connect(self.reset_user_data)
        
        user_data_layout.addWidget(user_data_label)
        user_data_layout.addWidget(reset_data_button)
        
        # 사용자 카드에 위젯 추가
        user_card_layout.addWidget(user_info_frame)
        user_card_layout.addLayout(user_data_layout)
        
        # 사용자 탭에 카드 추가
        user_layout.addWidget(user_card)
        user_layout.addStretch()
        
        # 탭 위젯에 탭 추가
        self.tab_widget.addTab(sensor_tab, "EMG 센서 설정")
        self.tab_widget.addTab(recognition_tab, "동작 인식 설정")
        self.tab_widget.addTab(game_tab, "게임 설정")
        self.tab_widget.addTab(appearance_tab, "화면 설정")  # 새 탭 추가
        self.tab_widget.addTab(user_tab, "사용자 정보")
        
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
        
        # 뒤로 버튼
        self.back_btn = NeumorphicButton("뒤로")
        self.back_btn.clicked.connect(self.go_back)
        
        # 레이아웃에 위젯 추가
        layout.addWidget(title_card)
        layout.addWidget(self.tab_widget)
        layout.addWidget(log_card)
        layout.addWidget(self.back_btn)
        
        
    def set_user(self, username, password):
        """사용자 정보 설정"""
        super().set_user(username, password)
        
        # UI 업데이트
        self.user_info_label.setText(f"현재 사용자: {username}")
        
        # 사용자 설정 로드
        config = load_user_config(username, password)
        
        # 모델 정확도 표시
        game_model_accuracy = config.get('game_model_accuracy', config.get('model_accuracy', 0.0))
        action_model_accuracy = config.get('action_model_accuracy', 0.0)
        
        model_info_text = f"게임 모델 정확도: {game_model_accuracy:.2f}"
        if action_model_accuracy > 0:
            model_info_text += f" | 동작 제어 모델 정확도: {action_model_accuracy:.2f}"
            
        self.model_info_label.setText(model_info_text)
        
        # 감도 설정 업데이트
        self.sensitivity_slider.setValue(config.get("sensitivity", 1.0))
        
        # 동작 인식 설정 업데이트
        self.detection_threshold_slider.setValue(config.get("detection_threshold", DETECTION_THRESHOLD))
        self.trend_threshold_slider.setValue(config.get("trend_threshold", TREND_THRESHOLD))
        self.dispersion_threshold_slider.setValue(config.get("dispersion_threshold", DISPERSION_THRESHOLD))
        self.cooldown_slider.setValue(config.get("cooldown_period", COOLDOWN_PERIOD))
        self.confidence_slider.setValue(config.get("confidence_threshold", 0.5))
        
        # 게임 설정 업데이트
        if "rps_rounds" in config:
            self.rps_round_combo.setCurrentText(str(config["rps_rounds"]))
        if "muk_rounds" in config:
            self.muk_round_combo.setCurrentText(str(config["muk_rounds"]))
        
        # 연결 방식 설정 업데이트
        if "connection_type" in config:
            connection_type = config["connection_type"]
            if connection_type == "usb":
                self.usb_radio.setChecked(True)
            else:
                self.bluetooth_radio.setChecked(True)
                
        # 포트와 통신 속도 설정
        if "port" in config:
            index = self.port_combo.findText(config["port"])
            if index >= 0:
                self.port_combo.setCurrentIndex(index)
                
        # 통신 속도 표시 업데이트
        if "baud_rate" in config:
            self.baud_value_label.setText(str(config["baud_rate"]))
        else:
            # 기본값 설정
            connection_type = "usb" if self.usb_radio.isChecked() else "bluetooth"
            baud_rate = 115200 if connection_type == "usb" else 9600
            self.baud_value_label.setText(str(baud_rate))
            
        self.add_log(f"'{username}' 계정 설정 로드 완료")
        
        # 포트 목록 업데이트
        self.refresh_ports()
    # 화면 크기 설정 메서드 추가
    def set_window_size(self, width, height):
        """화면 크기 설정"""
        main_window = self.window()
        if isinstance(main_window, QMainWindow):
            # 화면 중앙에 위치하도록 조정
            main_window.setGeometry(
                QStyle.alignedRect(
                    Qt.LeftToRight,
                    Qt.AlignCenter,
                    QSize(width, height),
                    QApplication.desktop().availableGeometry()
                )
            )
            
            # 현재 크기 업데이트
            window_size = main_window.size()
            self.current_size_value.setText(f"{window_size.width()} x {window_size.height()}")

    # 폰트 크기 조절 메서드 추가
    def update_font_scale(self, value):
        """폰트 크기 조절"""
        # 폰트 크기 퍼센트 업데이트
        self.font_scale_value.setText(f"{value}%")
        
        # 메인 애플리케이션에 폰트 스케일 적용
        main_window = self.window()
        if hasattr(main_window, 'apply_font_scale'):
            scale_factor = value / 100.0
            main_window.apply_font_scale(scale_factor)
    # 나머지 메서드들은 기존 코드와 동일
    def on_connection_type_changed(self):
        """연결 방식 변경 시 호출"""
        if self.usb_radio.isChecked():
            # USB 케이블 선택
            ports = self.refresh_ports()
            # COM9를 기본값으로 제안하지만 다른 포트도 선택 가능
            if "COM9" in ports:
                self.port_combo.setCurrentText("COM9")
            self.baud_value_label.setText("115200")  # USB 통신 속도 표시
            self.add_log("USB 케이블 연결 방식 선택됨 (115200 baud)")
        else:
            # 블루투스 선택
            ports = self.refresh_ports()
            # COM10을 기본값으로 제안하지만 다른 포트도 선택 가능
            if "COM10" in ports:
                self.port_combo.setCurrentText("COM10")
            self.baud_value_label.setText("9600")  # 블루투스 통신 속도 표시
            self.add_log("블루투스 연결 방식 선택됨 (9600 baud)")
        
    def refresh_ports(self):
        """사용 가능한 포트 목록 업데이트"""
        # 전역 시리얼 스레드 참조 얻기
        main_app = self.window()
        if hasattr(main_app, "global_serial_thread"):
            serial_thread = main_app.global_serial_thread
        else:
            serial_thread = SerialThread()
            
        # 포트 목록 가져오기
        ports = serial_thread.get_available_ports()
        
        # 콤보박스 업데이트
        self.port_combo.clear()
        self.port_combo.addItems(ports)
        
        # 기본 포트 선택
        if DEFAULT_SERIAL_PORT in ports:
            self.port_combo.setCurrentText(DEFAULT_SERIAL_PORT)
            
        self.add_log(f"포트 목록 업데이트: {', '.join(ports)}")
        return ports
    
    def test_connection(self):
        """연결 테스트 - 전역 SerialThread 설정 업데이트"""
        port = self.port_combo.currentText()
        connection_type = "usb" if self.usb_radio.isChecked() else "bluetooth"
        baud_rate = 115200 if connection_type == "usb" else 9600

        self.add_log(f"연결 설정 업데이트: {port}, {baud_rate} bps, {connection_type} 모드")
        
        # 메인 애플리케이션 참조 얻기
        main_app = self.window()
        if hasattr(main_app, "update_serial_config"):
            # 전역 SerialThread 설정 업데이트
            if main_app.update_serial_config(port, baud_rate, connection_type):
                # 설정 저장
                self.save_connection_settings(port, baud_rate, connection_type)
                self.add_log("연결 설정이 업데이트되었습니다.")
                
                # 연결 상태 확인
                if main_app.global_serial_thread and main_app.global_serial_thread.ser:
                    self.add_log(f"시리얼 포트 열림: {main_app.global_serial_thread.ser.is_open}")
                else:
                    self.add_log("시리얼 포트가 열리지 않았습니다.")
            else:
                self.add_log("연결 업데이트 실패")
        else:
            self.add_log("전역 SerialThread를 찾을 수 없습니다.")
    
    def save_connection_settings(self, port, baud_rate, connection_type):
        """연결 설정만 저장"""
        config = load_user_config(self.username, self.password)
        config["port"] = port
        config["baud_rate"] = baud_rate
        config["connection_type"] = connection_type
        save_user_config(self.username, self.password, config)
        self.add_log("연결 설정이 저장되었습니다.")
    
    def reset_recognition_defaults(self):
        """동작 인식 설정 기본값으로 복원"""
        self.detection_threshold_slider.setValue(DETECTION_THRESHOLD)
        self.trend_threshold_slider.setValue(TREND_THRESHOLD)
        self.dispersion_threshold_slider.setValue(DISPERSION_THRESHOLD)
        self.cooldown_slider.setValue(COOLDOWN_PERIOD)
        self.confidence_slider.setValue(0.5)
        
        self.add_log("동작 인식 설정을 기본값으로 복원했습니다.")
    
    def save_recognition_settings(self):
        """동작 인식 설정 저장"""
        # 설정값 추출
        detection_threshold = self.detection_threshold_slider.value()
        trend_threshold = self.trend_threshold_slider.value()
        dispersion_threshold = self.dispersion_threshold_slider.value()
        cooldown_period = self.cooldown_slider.value()
        confidence_threshold = self.confidence_slider.value()
        
        # 사용자 설정 로드 및 업데이트
        config = load_user_config(self.username, self.password)
        config["detection_threshold"] = detection_threshold
        config["trend_threshold"] = trend_threshold
        config["dispersion_threshold"] = dispersion_threshold
        config["cooldown_period"] = cooldown_period
        config["confidence_threshold"] = confidence_threshold
        
        # 설정 저장
        save_user_config(self.username, self.password, config)
        
        self.add_log("동작 인식 설정이 저장되었습니다.")
        
        # 알림 표시
        QMessageBox.information(self, "설정 저장", "동작 인식 설정이 성공적으로 저장되었습니다.")
    
    def save_settings(self):
        """설정 저장"""
        # 센서 설정
        port = self.port_combo.currentText()
        sensitivity = self.sensitivity_slider.value()
        # 연결 방식 설정
        connection_type = "usb" if self.usb_radio.isChecked() else "bluetooth"
        baud_rate = 115200 if connection_type == "usb" else 9600
        # 게임 설정
        rps_rounds = int(self.rps_round_combo.currentText())
        muk_rounds = int(self.muk_round_combo.currentText())
        
        # 사용자 설정 로드 및 업데이트
        config = load_user_config(self.username, self.password)
        config["sensitivity"] = sensitivity
        config["rps_rounds"] = rps_rounds
        config["muk_rounds"] = muk_rounds
        config["port"] = port
        config["baud_rate"] = baud_rate
        config["connection_type"] = connection_type
        
        # 설정 저장
        save_user_config(self.username, self.password, config)
        
        self.add_log("설정 저장 완료")
        
        # 알림 표시
        QMessageBox.information(self, "설정 저장", "설정이 성공적으로 저장되었습니다.")
        
    def reset_user_data(self):
        """사용자 데이터 초기화"""
        # 확인 대화상자
        reply = QMessageBox.question(
            self, 
            '데이터 초기화', 
            "정말로 모든 데이터를 초기화하시겠습니까? 이 작업은 취소할 수 없습니다.",
            QMessageBox.Yes | QMessageBox.No, 
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            # 설정 초기화
            config = load_user_config(self.username, self.password)
            config["game_data_count"] = {
                str(LABEL_IDLE): 0,
                str(LABEL_SCISSORS): 0, 
                str(LABEL_ROCK): 0, 
                str(LABEL_PAPER): 0
            }
            config["action_data_count"] = {str(i): 0 for i in range(1, 11)}
            config["game_model_accuracy"] = 0.0
            config["action_model_accuracy"] = 0.0
            config["model_accuracy"] = 0.0  # 이전 버전과의 호환성 유지
            save_user_config(self.username, self.password, config)
            
            # 데이터 파일 삭제
            data_files = [f for f in os.listdir(self.user_dir) if f.endswith('.csv')]
            for file in data_files:
                os.remove(os.path.join(self.user_dir, file))
                
            # 모델 파일 삭제
            model_files = ["emg_model.pth", MODEL_FILE]
            for model_file in model_files:
                model_path = os.path.join(self.user_dir, model_file)
                if os.path.exists(model_path):
                    os.remove(model_path)
                
            self.add_log("사용자 데이터 초기화 완료")
            
            # UI 업데이트
            self.model_info_label.setText("모델 정확도: 0.00")
            
            # 알림 표시
            QMessageBox.information(self, "데이터 초기화", "사용자 데이터가 성공적으로 초기화되었습니다.")
     


# ------------------------------------------------------------------------
# 화면 제어 모드 위젯
# ------------------------------------------------------------------------
class ScreenControlWidget(BaseEMGWidget):
    """
    ScreenControlWidget 클래스: EMG 센서로 화면을 제어하는 위젯
    - 라벨 1~10에 따른 PC 화면 제어 기능 제공
    - 사용자 커스텀 동작 설정 지원
    - 네오모피즘 디자인 적용 (1920x1080 최적화)
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # 상태 변수
        self.is_running = False
        self.current_window = None
        
        # 동작 설정 딕셔너리 (기본값)
        self.action_settings = {
            "browser": {str(i): "없음" for i in range(1, 14)},
            "youtube": {str(i): "없음" for i in range(1, 14)},
            "ppt": {str(i): "없음" for i in range(1, 14)}
        }
        
        # 기본값 설정
        self.set_default_actions()
        
        # 사용 가능한 동작 목록
        self.available_actions = {
            "browser": ["없음", "왼쪽 탭으로 이동", "오른쪽 탭으로 이동", "새 탭", "탭 닫기", "새로고침", "홈으로 이동", "Alt+Tab"],
            "youtube": ["없음", "왼쪽 탭으로 이동", "오른쪽 탭으로 이동", "재생/일시정지", "볼륨 증가", "볼륨 감소", "다음 동영상", "이전 동영상", "전체화면", "Alt+Tab"],
            "ppt": ["없음", "다음 슬라이드", "이전 슬라이드", "슬라이드쇼 시작", "슬라이드쇼 종료", "Alt+Tab"]
        }
        
        # 타이머들
        self.detection_timer = QTimer(self)
        self.detection_timer.timeout.connect(self.update_detection)
        
        self.action_cooldown = False
        self.cooldown_timer = QTimer(self)
        self.cooldown_timer.timeout.connect(self.reset_action_cooldown)
        
        # UI 초기화
        self.init_ui()
        
    def init_ui(self):
        """UI 초기화 - 저장 버튼 추가"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)
        
        # 배경 설정
        self.setStyleSheet(f"background-color: {BG_COLOR};")
        
        # 제목 카드
        title_card = NeumorphicCard()
        title_layout = QVBoxLayout(title_card)
        title_layout.setContentsMargins(15, 10, 15, 10)
        
        # 제목 텍스트
        self.title_label = QLabel("화면 제어 모드")
        self.title_label.setStyleSheet(f"color: {TEXT_COLOR}; font-size: 24px; font-weight: bold;")
        self.title_label.setAlignment(Qt.AlignCenter)
        
        title_layout.addWidget(self.title_label)
        
        # 상단 정보 영역 (한 줄로 배치)
        info_layout = QHBoxLayout()
        
        # 설명 카드
        info_card = NeumorphicCard()
        info_card_layout = QVBoxLayout(info_card)
        info_card_layout.setContentsMargins(15, 10, 15, 10)
        
        # 설명 텍스트
        info_text = QLabel("EMG 센서를 통해 감지된 동작으로 PC를 제어할 수 있습니다.\n"
                        "애플리케이션을 선택하고 각 동작에 대한 기능을 설정한 후 '시작' 버튼을 누르세요.")
        info_text.setStyleSheet(f"color: {TEXT_COLOR}; font-size: 14px;")
        info_text.setAlignment(Qt.AlignLeft)
        info_text.setWordWrap(True)
        
        # 현재 상태 표시
        self.status_label = QLabel("대기 중")
        self.status_label.setStyleSheet(f"color: {PRIMARY_COLOR}; font-size: 16px; font-weight: bold;")
        self.status_label.setAlignment(Qt.AlignCenter)
        
        info_card_layout.addWidget(info_text)
        info_card_layout.addWidget(self.status_label)
        
        # 현재 감지 상태 카드
        detection_card = NeumorphicCard()
        detection_layout = QVBoxLayout(detection_card)
        detection_layout.setContentsMargins(15, 10, 15, 10)
        
        # 현재 인식된 창 타입
        window_title = QLabel("인식된 창")
        window_title.setStyleSheet(f"color: {TEXT_COLOR}; font-size: 14px; font-weight: bold;")
        window_title.setAlignment(Qt.AlignCenter)
        
        self.window_type_label = QLabel("알 수 없음")
        self.window_type_label.setStyleSheet(f"color: {SECONDARY_COLOR}; font-size: 16px; font-weight: bold;")
        self.window_type_label.setAlignment(Qt.AlignCenter)
        
        # 현재 감지된 동작
        detection_title = QLabel("감지된 동작")
        detection_title.setStyleSheet(f"color: {TEXT_COLOR}; font-size: 14px; font-weight: bold;")
        detection_title.setAlignment(Qt.AlignCenter)
        
        self.detected_label = QLabel("없음")
        self.detected_label.setStyleSheet(f"color: {PRIMARY_COLOR}; font-size: 16px; font-weight: bold;")
        self.detected_label.setAlignment(Qt.AlignCenter)
        
        detection_layout.addWidget(window_title)
        detection_layout.addWidget(self.window_type_label)
        detection_layout.addWidget(detection_title)
        detection_layout.addWidget(self.detected_label)
        
        # 상단 레이아웃에 추가
        info_layout.addWidget(info_card, 2)
        info_layout.addWidget(detection_card, 1)
        
        # 앱 선택 카드
        app_selection_card = NeumorphicCard()
        app_selection_layout = QVBoxLayout(app_selection_card)
        app_selection_layout.setContentsMargins(15, 10, 15, 10)
        
        app_selection_title = QLabel("애플리케이션 선택")
        app_selection_title.setStyleSheet(f"color: {TEXT_COLOR}; font-size: 16px; font-weight: bold;")
        app_selection_title.setAlignment(Qt.AlignCenter)
        
        # 앱 선택 버튼 그리드
        app_button_grid = QGridLayout()
        app_button_grid.setSpacing(10)
        
        # 브라우저 버튼
        browser_card = self._create_app_card("인터넷 브라우저", "#2196F3", "browser")
        # 유튜브 버튼
        youtube_card = self._create_app_card("유튜브", "#FF0000", "youtube")
        # 파워포인트 버튼
        ppt_card = self._create_app_card("PowerPoint", "#FF5722", "ppt")
        
        app_button_grid.addWidget(browser_card, 0, 0)
        app_button_grid.addWidget(youtube_card, 0, 1)
        app_button_grid.addWidget(ppt_card, 0, 2)
        
        app_selection_layout.addWidget(app_selection_title)
        app_selection_layout.addLayout(app_button_grid)
        
        # 현재 선택된 앱 표시
        self.current_app_label = QLabel("현재 선택된 앱: 인터넷 브라우저")
        self.current_app_label.setStyleSheet(f"""
            color: {PRIMARY_COLOR};
            font-size: 18px;
            font-weight: bold;
            margin: 10px;
        """)
        self.current_app_label.setAlignment(Qt.AlignCenter)
        
        # 동작 설정 카드
        action_card = NeumorphicCard()
        action_layout = QVBoxLayout(action_card)
        action_layout.setContentsMargins(15, 15, 15, 15)
        
        # 동작 설정 그리드 레이아웃
        action_grid = QGridLayout()
        action_grid.setSpacing(10)
        
        # 동작 카드 저장 딕셔너리
        self.action_combo_boxes = {}
        self.current_app = "browser"  # 기본 앱
        
        # 첫 번째 행: 동작 1-4
        actions_row1 = [(f"동작 {i}", i, "#E6F4FF") for i in range(1, 5)]
        
        # 두 번째 행: 동작 5-8  
        actions_row2 = [(f"동작 {i}", i, "#E6F4FF") for i in range(5, 9)]
        
        # 세 번째 행: 가위/바위/보
        actions_row3 = [
            ("가위", LABEL_SCISSORS, "#F0F8E8"),
            ("바위", LABEL_ROCK, "#FFFBEB"), 
            ("보", LABEL_PAPER, "#E8F5E9")
        ]
        
        # 각 행에 대한 동작 카드 생성
        for row, actions in enumerate([actions_row1, actions_row2]):
            for col, (action_name, label, bg_color) in enumerate(actions):
                action_card_item = self._create_action_card(action_name, label, bg_color)
                action_grid.addWidget(action_card_item, row, col)
        
        # 가위바위보는 별도 레이아웃으로 가운데 정렬
        rps_layout = QHBoxLayout()
        rps_layout.addStretch()
        
        for col, (action_name, label, bg_color) in enumerate(actions_row3):
            rps_card_item = self._create_action_card(action_name, label, bg_color, fixed_width=200)
            rps_layout.addWidget(rps_card_item)
            if col < 2:
                rps_layout.addSpacing(10)
        
        rps_layout.addStretch()
        
        # 액션 레이아웃에 추가
        action_layout.addWidget(self.current_app_label)
        action_layout.addLayout(action_grid)
        action_layout.addLayout(rps_layout)
        
        # 버튼 레이아웃 - 저장 버튼 추가
        button_layout = QHBoxLayout()
        
        # 저장 버튼 추가
        self.save_btn = NeumorphicButton("설정 저장", SECONDARY_COLOR)
        self.save_btn.setFixedHeight(40)
        self.save_btn.clicked.connect(self.save_all_settings)
        
        # 뒤로 버튼
        self.back_btn = NeumorphicButton("뒤로")
        self.back_btn.setFixedHeight(40)
        self.back_btn.clicked.connect(self.go_back)
        
        # 시작 버튼
        self.start_btn = NeumorphicButton("시작", PRIMARY_COLOR)
        self.start_btn.setFixedHeight(40)
        self.start_btn.clicked.connect(self.toggle_control)
        
        button_layout.addWidget(self.save_btn)
        button_layout.addWidget(self.back_btn)
        button_layout.addStretch()
        button_layout.addWidget(self.start_btn)
        
        # 레이아웃에 위젯 추가
        layout.addWidget(title_card)
        layout.addLayout(info_layout)
        layout.addWidget(app_selection_card)
        layout.addWidget(action_card)
        layout.addLayout(button_layout)
    def _create_app_card(self, name, color, app_key):
        """앱 선택 카드 생성"""
        card = NeumorphicCard()
        card.setFixedHeight(60)
        card.setCursor(Qt.PointingHandCursor)
        
        layout = QVBoxLayout(card)
        layout.setContentsMargins(8, 8, 8, 8)
        
        # 앱 이름 레이블
        label = QLabel(name)
        label.setStyleSheet(f"color: {color}; font-size: 14px; font-weight: bold;")
        label.setAlignment(Qt.AlignCenter)
        
        layout.addWidget(label)
        
        # 클릭 이벤트 처리
        card.mousePressEvent = lambda event, k=app_key: self.select_app(k)
        
        return card
    
    def _create_action_card(self, action_name, label, bg_color, fixed_width=None):
        """동작 카드 생성 헬퍼 함수"""
        action_card_item = NeumorphicCard()
        action_card_item.setFixedHeight(120)
        if fixed_width:
            action_card_item.setFixedWidth(fixed_width)
        
        action_card_item.setStyleSheet(f"""
            NeumorphicCard {{
                background-color: {bg_color};
                border-radius: 12px;
                border: none;
            }}
            NeumorphicCard QLabel {{
                background-color: transparent !important;
                border: none !important;
            }}
            NeumorphicCard QComboBox {{
                background-color: {BG_COLOR};
                border: 1px solid #D1D9E6;
                border-radius: 6px;
                padding: 4px;
                font-size: 11px;
                color: {TEXT_COLOR};
            }}
        """)
        
        action_card_layout = QVBoxLayout(action_card_item)
        action_card_layout.setContentsMargins(8, 8, 8, 8)
        
        # 동작 이름 레이블
        action_label = QLabel(action_name)
        action_label.setStyleSheet(f"color: {TEXT_COLOR}; font-size: 24px; font-weight: bold; background-color: transparent !important; border: none !important;")
        action_label.setAlignment(Qt.AlignCenter)
        
        # 기능 선택 콤보박스
        combo = QComboBox()
        combo.addItems(self.available_actions[self.current_app])
        combo.setCurrentText(self.action_settings[self.current_app].get(str(label), "없음"))
        combo.currentTextChanged.connect(lambda text, l=label: self.update_action_setting(self.current_app, l, text))
        combo.setStyleSheet(f"""
            QComboBox {{
                background-color: {BG_COLOR};
                border: 1px solid #D1D9E6;
                border-radius: 6px;
                padding: 4px;
                font-size: 18px;
                color: {TEXT_COLOR};
            }}
            QComboBox::drop-down {{
                border: none;
                padding-right: 10px;
            }}
        """)
        
        # 카드에 위젯 추가
        action_card_layout.addWidget(action_label)
        action_card_layout.addWidget(combo)
        
        # 콤보박스 저장
        self.action_combo_boxes[label] = combo
        
        return action_card_item
    def save_all_settings(self):
        """모든 설정 저장 - 개선된 버전"""
        # 현재 콤보박스 설정 먼저 저장
        self.save_current_settings()
        
        print("=== 설정 저장 시작 ===")
        print("현재 저장될 설정:")
        for app, settings in self.action_settings.items():
            print(f"  {app}:")
            for label, action in settings.items():
                if action != "없음":
                    print(f"    동작 {label}: {action}")
        
        # 파일에 저장
        success = self.save_settings()
        
        if success:
            # 사용자에게 알림
            self.status_label.setText("설정이 저장되었습니다.")
            QTimer.singleShot(2000, lambda: self.status_label.setText("대기 중"))
            
            # 성공 메시지 표시
            QMessageBox.information(self, "설정 저장", "화면 제어 설정이 성공적으로 저장되었습니다.")
            print("=== 설정 저장 완료 ===")


    def select_app(self, app_key):
        """앱 타입 선택 - 설정 보존하도록 수정"""
        # 현재 설정을 먼저 저장
        if hasattr(self, 'current_app') and hasattr(self, 'action_combo_boxes'):
            self.save_current_settings()
            print(f"'{self.current_app}' 앱 설정 저장 완료")
        
        # 앱 변경
        old_app = getattr(self, 'current_app', None)
        self.current_app = app_key
        
        # 앱 이름 업데이트
        app_names = {
            "browser": "인터넷 브라우저",
            "youtube": "유튜브", 
            "ppt": "PowerPoint"
        }
        self.current_app_label.setText(f"현재 선택된 앱: {app_names[app_key]}")
        
        # 저장된 설정으로 콤보박스 업데이트 (한 번만!)
        self.load_settings_to_comboboxes()
        print(f"'{app_key}' 앱으로 전환 완료")
    # 나머지 메소드들은 기존과 동일...
    def load_settings(self):
        """사용자 설정 로드 - 개선된 버전"""
        config_path = os.path.join(self.user_dir, "screen_control_config.json")
        
        print(f"설정 파일 경로: {config_path}")
        print(f"설정 파일 존재: {os.path.exists(config_path)}")
        
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    settings = json.load(f)
                    loaded_settings = settings.get("action_settings", {})
                    
                print("파일에서 로드된 설정:")
                for app, app_settings in loaded_settings.items():
                    print(f"  {app}:")
                    for label, action in app_settings.items():
                        if action != "없음":
                            print(f"    동작 {label}: {action}")
                
                # 기본 설정과 로드된 설정 병합
                for app in self.action_settings.keys():
                    if app in loaded_settings:
                        for label in self.action_settings[app].keys():
                            if label in loaded_settings[app]:
                                self.action_settings[app][label] = loaded_settings[app][label]
                
                print("화면 제어 설정 로드 완료")
                
            except Exception as e:
                print(f"설정 로드 오류: {e}")
                # 기본 설정으로 초기화
                self.set_default_actions()
        else:
            print("설정 파일이 없으므로 기본 설정 사용")
            # 설정 파일이 없으면 기본 설정 사용
            self.set_default_actions()
            self.save_settings()  # 기본 설정 저장
    
    def save_settings(self):
        """사용자 설정 저장 - 개선된 버전"""
        config_path = os.path.join(self.user_dir, "screen_control_config.json")
        settings = {
            "action_settings": self.action_settings
        }
        
        try:
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(settings, f, indent=4, ensure_ascii=False)
            print(f"화면 제어 설정 저장 완료: {config_path}")
            return True
        except Exception as e:
            print(f"설정 저장 오류: {e}")
            return False
    
    def set_user(self, username, password):
        """사용자 정보 설정 - 설정 로드 개선"""
        super().set_user(username, password)
        
        print(f"=== 사용자 설정 로드 시작: {username} ===")
        
        # 설정 로드
        self.load_settings()
        
        # UI에 설정 적용 (초기화)
        if hasattr(self, 'action_combo_boxes') and self.action_combo_boxes:
            self.load_settings_to_comboboxes()
        
        # 통합 모델 로드
        model_path = os.path.join(self.user_dir, "emg_model.pth")
        
        if os.path.exists(model_path):
            self.predictor = EMGPredictor(model_path)
            self.status_label.setText("통합 모델 로드 완료. 시작 버튼을 눌러 제어를 시작하세요.")
        else:
            self.status_label.setText("통합 동작 인식 모델이 없습니다. 먼저 데이터를 수집하고 학습해주세요.")
            self.start_btn.setEnabled(False)
            
        print(f"=== 사용자 설정 로드 완료: {username} ===")           
    def update_table_from_settings(self):
        """설정에서 테이블 업데이트"""
        # 설정 스택 업데이트 로직이 필요하면 여기에 추가
        pass
                    
    def update_action_setting(self, app_key, label, text):
        """동작 설정 업데이트 - 빈 문자열 처리 개선"""
        label_key = str(label)
        
        # 빈 문자열이면 "없음"으로 처리
        if not text or text.strip() == "":
            text = "없음"
        
        self.action_settings[app_key][label_key] = text


        
    def toggle_control(self):
        """제어 시작/중지 토글"""
        if not self.is_running:
            self.start_control()
        else:
            self.stop_control()
            
    def start_control(self):
        """제어 시작"""
        # EMG 프로세서 설정 적용 (중요!)
        config = load_user_config(self.username, self.password)
        self.emg_processor.apply_config(config)
        
        # 전역 시리얼 스레드에 구독
        if self.subscribe_to_serial():
            self.is_running = True
            self.start_btn.setText("중지")
            self.status_label.setText("화면 제어 실행 중... 동작을 취해보세요.")
            
            # 타이머 시작
            self.detection_timer.start(100)
            
            # 초기 창 타입 감지
            self.detect_current_window()

        else:
            self.status_label.setText("EMG 센서 연결 실패. 연결을 확인하세요.")
            
    def stop_control(self):
        """제어 중지"""
        self.is_running = False
        self.unsubscribe_from_serial()
        self.start_btn.setText("시작")
        self.status_label.setText("화면 제어 중지됨.")
        self.detection_timer.stop()
        
    def process_data(self, s1, s2):
        """EMG 데이터 처리 - 통합 라벨 지원"""
        if not self.is_running or not self.predictor:
            return
        
            
        result = self.emg_processor.process_data(s1, s2)
        
        if isinstance(result, tuple) and result[0] == "movement_completed":
            sequence = result[1]
            
            # 동작 예측
            prediction, confidence = self.predictor.predict(sequence)
            
            # 디버깅 로그
            print(f"화면제어: 예측 결과 - 동작 {prediction}, 신뢰도 {confidence:.2f}")
            
            # 충분한 신뢰도를 가진 동작만 처리
            if confidence >= self.confidence_threshold:
                # 대기 상태(라벨 0)는 무시
                if prediction == LABEL_IDLE:
                    self.detected_label.setText("감지된 동작: 대기 상태")
                    print(f"대기 상태 감지됨 (신뢰도: {confidence:.2f})")
                    return
                
                # 라벨 1-8, 11-13 처리
                if (1 <= prediction <= 8) or prediction in (LABEL_SCISSORS, LABEL_ROCK, LABEL_PAPER):
                    # 라벨 설명 추가
                    if prediction == LABEL_SCISSORS:
                        self.detected_label.setText(f"감지된 동작: 가위")
                    elif prediction == LABEL_ROCK:
                        self.detected_label.setText(f"감지된 동작: 바위")
                    elif prediction == LABEL_PAPER:
                        self.detected_label.setText(f"감지된 동작: 보")
                    else:
                        self.detected_label.setText(f"감지된 동작: 동작 {prediction}")
                    
                    # 동작 실행 (쿨다운 체크)
                    if not self.action_cooldown:
                        self.execute_action(prediction)
                        self.action_cooldown = True
                        self.cooldown_timer.start(1500)  # 1.5초 쿨다운
                    
    def reset_action_cooldown(self):
        """동작 실행 쿨다운 초기화"""
        self.action_cooldown = False
        self.cooldown_timer.stop()
        
    def update_detection(self):
        """감지 상태 업데이트"""
        # 현재 창 타입 감지
        self.detect_current_window()
        
    def detect_current_window(self):
        """현재 활성화된 창 타입 감지"""
        try:
            # 현재 활성화된 창 정보 가져오기 (pywin32 필요)
            import win32gui
            
            window_title = win32gui.GetWindowText(win32gui.GetForegroundWindow())
            
            # 창 타입 판별
            if "PowerPoint" in window_title or ".ppt" in window_title.lower() or ".pptx" in window_title.lower():
                self.current_window = "ppt"
                self.window_type_label.setText("PPT")
                self.window_type_label.setStyleSheet(f"color: #FF5722; font-size: 16px; font-weight: bold;")
            elif "YouTube" in window_title:
                self.current_window = "youtube"
                self.window_type_label.setText("유튜브")
                self.window_type_label.setStyleSheet(f"color: #FF0000; font-size: 16px; font-weight: bold;")
            elif "Chrome" in window_title or "Firefox" in window_title or "Edge" in window_title or "Safari" in window_title:
                self.current_window = "browser"
                self.window_type_label.setText("인터넷 브라우저")
                self.window_type_label.setStyleSheet(f"color: #2196F3; font-size: 16px; font-weight: bold;")
            else:
                self.current_window = None
                self.window_type_label.setText("알 수 없음")
                self.window_type_label.setStyleSheet(f"color: {SECONDARY_COLOR}; font-size: 16px; font-weight: bold;")
        except:
            self.current_window = None
            self.window_type_label.setText("감지 실패")
            self.window_type_label.setStyleSheet(f"color: {DANGER_COLOR}; font-size: 16px; font-weight: bold;")
            
    def execute_action(self, label_value):
        """동작 실행"""
        if not self.current_window:
            return
            
        action = self.action_settings[self.current_window][str(label_value)]
        if action == "없음":
            return
            
        # 동작 실행
        try:
            # pyautogui 사용 (설치 필요)
            import pyautogui
            pyautogui.FAILSAFE = False
            
            # 브라우저 동작
            if self.current_window == "browser" or self.current_window == "youtube":
                if action == "왼쪽 탭으로 이동":
                    pyautogui.hotkey('ctrl', 'shift', 'tab')
                elif action == "오른쪽 탭으로 이동":
                    pyautogui.hotkey('ctrl', 'tab')
                elif action == "새 탭":
                    pyautogui.hotkey('ctrl', 't')
                elif action == "탭 닫기":
                    pyautogui.hotkey('ctrl', 'w')
                elif action == "새로고침":
                    pyautogui.hotkey('f5')
                elif action == "홈으로 이동":
                    pyautogui.hotkey('alt', 'home')
                    
            # 유튜브 전용 동작
            if self.current_window == "youtube":
                if action == "재생/일시정지":
                    pyautogui.press('k')
                elif action == "볼륨 증가":
                    pyautogui.press('up')
                elif action == "볼륨 감소":
                    pyautogui.press('down')
                elif action == "다음 동영상":
                    pyautogui.press('shift+n')
                elif action == "이전 동영상":
                    pyautogui.press('shift+p')
                elif action == "전체화면":
                    pyautogui.press('f')
                    
            # PPT 동작
            if self.current_window == "ppt":
                if action == "다음 슬라이드":
                    pyautogui.press('right')
                elif action == "이전 슬라이드":
                    pyautogui.press('left')
                elif action == "슬라이드쇼 시작":
                    pyautogui.press('f5')
                elif action == "슬라이드쇼 종료":
                    pyautogui.press('esc')
                    
            # 공통 동작
            if action == "Alt+Tab":
                pyautogui.keyDown('alt')
                pyautogui.press('tab')
                pyautogui.keyUp('alt')
                
            # 동작 실행 표시
            self.status_label.setText(f"동작 실행: {action}")
            QTimer.singleShot(800, lambda: self.status_label.setText("화면 제어 실행 중... 동작을 취해보세요."))
                
        except Exception as e:
            self.status_label.setText(f"동작 실행 오류: {str(e)}")
    def set_default_actions(self):
        """기본 동작 설정"""
        # 브라우저 기본 설정
        self.action_settings["browser"]["1"] = "왼쪽 탭으로 이동"
        self.action_settings["browser"]["2"] = "오른쪽 탭으로 이동"
        self.action_settings["browser"]["4"] = "Alt+Tab"
        self.action_settings["browser"]["11"] = "새 탭"
        self.action_settings["browser"]["12"] = "탭 닫기"
        self.action_settings["browser"]["13"] = "새로고침"
        
        # 유튜브 기본 설정
        self.action_settings["youtube"]["1"] = "왼쪽 탭으로 이동"
        self.action_settings["youtube"]["2"] = "오른쪽 탭으로 이동"
        self.action_settings["youtube"]["3"] = "재생/일시정지"
        self.action_settings["youtube"]["4"] = "Alt+Tab"
        self.action_settings["youtube"]["5"] = "볼륨 증가"
        self.action_settings["youtube"]["6"] = "볼륨 감소"
        self.action_settings["youtube"]["7"] = "다음 동영상"
        self.action_settings["youtube"]["8"] = "이전 동영상"
        
        # PPT 기본 설정
        self.action_settings["ppt"]["1"] = "다음 슬라이드"
        self.action_settings["ppt"]["2"] = "이전 슬라이드"
        self.action_settings["ppt"]["3"] = "슬라이드쇼 시작"
        self.action_settings["ppt"]["4"] = "Alt+Tab"
    def save_current_settings(self):
        """현재 콤보박스 설정을 저장 - 개선된 버전"""
        if not hasattr(self, 'current_app') or not hasattr(self, 'action_combo_boxes'):
            return
            
        print(f"'{self.current_app}' 앱의 현재 설정 저장 중...")
        
        for label, combo in self.action_combo_boxes.items():
            if combo and combo.currentText():
                current_text = combo.currentText().strip()
                if not current_text:
                    current_text = "없음"
                self.action_settings[self.current_app][str(label)] = current_text
                print(f"  동작 {label}: {current_text}")
    
    def load_settings_to_comboboxes(self):
        """저장된 설정을 콤보박스에 로드 - 개선된 버전"""
        if not hasattr(self, 'current_app') or not hasattr(self, 'action_combo_boxes'):
            return
            
        print(f"'{self.current_app}' 앱 설정 로드 중...")
        
        # 모든 시그널 먼저 해제
        for label, combo in self.action_combo_boxes.items():
            if combo:
                try:
                    combo.currentTextChanged.disconnect()
                except:
                    pass
        
        # 콤보박스 업데이트
        for label, combo in self.action_combo_boxes.items():
            if combo:
                # 콤보박스 아이템 업데이트
                combo.clear()
                combo.addItems(self.available_actions[self.current_app])
                
                # 저장된 설정 적용
                saved_action = self.action_settings[self.current_app].get(str(label), "없음")
                if not saved_action or saved_action.strip() == "":
                    saved_action = "없음"
                
                # 콤보박스에 해당 아이템이 있는지 확인
                index = combo.findText(saved_action)
                if index >= 0:
                    combo.setCurrentIndex(index)
                else:
                    combo.setCurrentText("없음")
                
                print(f"  동작 {label}: {saved_action}")
        
        # 시그널 재연결 (클로저 문제 해결)
        for label, combo in self.action_combo_boxes.items():
            if combo:
                # lambda 대신 functools.partial 사용 또는 별도 메서드 생성
                def create_handler(l):
                    return lambda text: self.update_action_setting(self.current_app, l, text)
                
                combo.currentTextChanged.connect(create_handler(label))
    def go_back(self):
        """뒤로 가기 - 설정 저장 후 이동"""
        # 현재 설정 저장
        self.save_current_settings()
        self.save_settings()
        print("화면 제어 모드 종료 - 설정 저장됨")
        
        # 부모 클래스의 go_back 호출
        super().go_back()

# ------------------------------------------------------------------------
# 촬영 모드 위젯
# ------------------------------------------------------------------------
class RecordingWidget(BaseEMGWidget):
    """
    RecordingWidget 클래스: EMG 센서로 사진/동영상 촬영을 제어하는 위젯
    - 웹캠을 통한 사진 및 동영상 촬영 제어 기능 제공
    - EMG 동작 인식을 통한 제어
    - 네오모피즘 디자인 적용
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # 카메라 상태
        self.camera = None
        self.recording = False
        self.paused = False
        self.video_writer = None
        self.current_video_path = None
        self.output_dir = None
        self.is_running = False
        
        # 감지된 동작 타이머
        self.detection_timer = QTimer(self)
        self.detection_timer.timeout.connect(self.update_detection)
        
        # 동작 실행 쿨다운 타이머
        self.action_cooldown = False
        self.cooldown_timer = QTimer(self)
        self.cooldown_timer.timeout.connect(self.reset_action_cooldown)
        
        # 프레임 업데이트 타이머
        self.frame_timer = QTimer(self)
        self.frame_timer.timeout.connect(self.update_frame)
        
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
        
        # 제목 텍스트
        self.title_label = QLabel("EMG 촬영 모드")
        self.title_label.setStyleSheet(f"color: {TEXT_COLOR}; font-size: 28px; font-weight: bold;")
        self.title_label.setAlignment(Qt.AlignCenter)
        
        title_layout.addWidget(self.title_label)
        
        # 카메라 뷰 카드
        camera_card = NeumorphicCard()
        camera_layout = QVBoxLayout(camera_card)
        camera_layout.setContentsMargins(10, 10, 10, 10)
        camera_layout.setSpacing(10)
        
        # 카메라 뷰 라벨
        self.camera_view = QLabel("카메라가 연결되지 않았습니다.")
        self.camera_view.setAlignment(Qt.AlignCenter)
        self.camera_view.setStyleSheet(f"""
            background-color: #000000;
            color: white;
            border-radius: 10px;
            padding: 10px;
            font-size: 16px;
        """)
        self.camera_view.setMinimumSize(640, 480)
        
        camera_layout.addWidget(self.camera_view)
        
        # 컨트롤 카드
        control_card = NeumorphicCard()
        control_layout = QHBoxLayout(control_card)
        control_layout.setContentsMargins(20, 15, 20, 15)
        control_layout.setSpacing(20)
        
        # 상태 레이블
        self.status_label = QLabel("준비 중...")
        self.status_label.setStyleSheet(f"color: {TEXT_COLOR}; font-size: 16px; font-weight: bold;")
        
        # 감지된 동작 레이블
        self.detected_label = QLabel("감지된 동작: 없음")
        self.detected_label.setStyleSheet(f"color: {PRIMARY_COLOR}; font-size: 16px; font-weight: bold;")
        
        # 녹화 상태 표시
        self.recording_indicator = QLabel("")
        self.recording_indicator.setFixedSize(20, 20)
        self.recording_indicator.setStyleSheet("background-color: grey; border-radius: 10px;")
        
        # 버튼 레이아웃
        button_layout = QHBoxLayout()
        
        # 시작/중지 버튼
        self.start_btn = NeumorphicButton("시작", PRIMARY_COLOR)
        self.start_btn.clicked.connect(self.toggle_camera)
        
        # 저장 경로 설정 버튼
        self.path_btn = NeumorphicButton("저장 경로 설정")
        self.path_btn.clicked.connect(self.set_output_dir)
        
        button_layout.addWidget(self.start_btn)
        button_layout.addWidget(self.path_btn)
        
        # 컨트롤 레이아웃에 위젯 추가
        control_layout.addWidget(self.status_label, stretch=3)
        control_layout.addWidget(self.detected_label, stretch=3)
        control_layout.addWidget(self.recording_indicator)
        control_layout.addLayout(button_layout, stretch=4)
        
        # 동작 가이드 카드
        guide_card = NeumorphicCard()
        guide_layout = QVBoxLayout(guide_card)
        guide_layout.setContentsMargins(20, 15, 20, 15)
        
        # 가이드 제목
        guide_title = QLabel("동작 가이드")
        guide_title.setStyleSheet(f"color: {TEXT_COLOR}; font-size: 18px; font-weight: bold;")
        guide_title.setAlignment(Qt.AlignCenter)
        
        # 가이드 텍스트
        guide_text = QLabel("• 동작 1: 사진 촬영\n• 동작 2: 동영상 촬영 시작/중지\n• 동작 3: 동영상 일시정지/재개\n\n* 대기 상태를 감지하면 동작을 실행하지 않습니다.")
        guide_text.setStyleSheet(f"color: {TEXT_COLOR}; font-size: 16px; line-height: 1.5;")
        guide_text.setAlignment(Qt.AlignLeft)
        
        guide_layout.addWidget(guide_title)
        guide_layout.addWidget(guide_text)
        
        # 뒤로 버튼
        self.back_btn = NeumorphicButton("뒤로")
        self.back_btn.clicked.connect(self.go_back)
        
        # 레이아웃에 위젯 추가
        layout.addWidget(title_card)
        layout.addWidget(camera_card, stretch=4)
        layout.addWidget(control_card)
        layout.addWidget(guide_card)
        layout.addWidget(self.back_btn)
        
    def set_user(self, username, password):
        """사용자 정보 설정 - 통합 모델 전용"""
        super().set_user(username, password)
        
        # 출력 디렉토리 설정
        self.output_dir = os.path.join(self.user_dir, "recordings")
        ensure_dir_exists(self.output_dir)
        
        # 통합 모델 로드
        model_path = os.path.join(self.user_dir, "emg_model.pth")
        
        if os.path.exists(model_path):
            self.predictor = EMGPredictor(model_path)
            self.status_label.setText("통합 모델 로드 완료. 시작 버튼을 눌러 카메라를 활성화하세요.")

        else:
            self.status_label.setText("통합 동작 인식 모델이 없습니다. 먼저 데이터를 수집하고 학습해주세요.")
            self.start_btn.setEnabled(False)
        
    def toggle_camera(self):
        """카메라 시작/중지 토글"""
        if not self.is_running:
            self.start_camera()
        else:
            self.stop_camera()
            
    def start_camera(self):
        """카메라 시작"""
        try:
            # OpenCV 카메라 초기화 시도
            import cv2
            
            self.camera = cv2.VideoCapture(0)
            if not self.camera.isOpened():
                self.status_label.setText("카메라를 열 수 없습니다. 연결을 확인하세요.")
                return
                
            # EMG 프로세서 설정 적용 (중요!)
            config = load_user_config(self.username, self.password)
            self.emg_processor.apply_config(config)
            
            # 전역 시리얼 스레드에 구독
            if self.subscribe_to_serial():
                self.is_running = True
                self.start_btn.setText("중지")
                self.status_label.setText("카메라 활성화됨. EMG 센서로 동작을 제어하세요.")
                
                # 타이머 시작
                self.detection_timer.start(100)
                self.frame_timer.start(30)  # ~30 FPS

            else:
                self.status_label.setText("EMG 센서 연결 실패. 연결을 확인하세요.")
                self.camera.release()
                self.camera = None
                
        except ImportError:
            self.status_label.setText("OpenCV 모듈이 설치되지 않았습니다. 'pip install opencv-python'을 실행하세요.")
        except Exception as e:
            self.status_label.setText(f"카메라 초기화 오류: {str(e)}")
            
    def stop_camera(self):
        """카메라 중지"""
        # 녹화 중이면 중지
        if self.recording:
            self.stop_recording()
            
        # 카메라 해제
        if self.camera:
            self.camera.release()
            self.camera = None
            
        # 구독 해제
        self.is_running = False
        self.unsubscribe_from_serial()
        
        # 타이머 중지
        self.detection_timer.stop()
        self.frame_timer.stop()
        
        # UI 업데이트
        self.start_btn.setText("시작")
        self.status_label.setText("카메라 비활성화됨.")
        self.camera_view.setText("카메라가 연결되지 않았습니다.")
        self.camera_view.setStyleSheet(f"""
            background-color: #000000;
            color: white;
            border-radius: 10px;
            padding: 10px;
            font-size: 16px;
        """)
        
    def update_frame(self):
        """카메라 프레임 업데이트"""
        if not self.camera:
            return
            
        try:
            # 프레임 읽기
            ret, frame = self.camera.read()
            if not ret:
                self.status_label.setText("카메라에서 프레임을 읽을 수 없습니다.")
                return
                
            # 동영상 녹화 중이면 프레임 저장
            if self.recording and self.video_writer and not self.paused:
                self.video_writer.write(frame)
                
                # 녹화 표시 깜빡임
                if int(time.time() * 2) % 2 == 0:
                    self.recording_indicator.setStyleSheet("background-color: red; border-radius: 10px;")
                else:
                    self.recording_indicator.setStyleSheet("background-color: darkred; border-radius: 10px;")
            elif self.recording and self.paused:
                # 일시정지 중인 경우
                self.recording_indicator.setStyleSheet("background-color: orange; border-radius: 10px;")
            else:
                self.recording_indicator.setStyleSheet("background-color: grey; border-radius: 10px;")
                
            # OpenCV 프레임을 Qt 이미지로 변환
            import cv2
            
            # 프레임 크기 조정
            height, width, channel = frame.shape
            bytesPerLine = 3 * width
            
            # BGR -> RGB 변환
            cv_rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # QImage 생성
            qt_image = QImage(cv_rgb_image.data, width, height, bytesPerLine, QImage.Format_RGB888)
            
            # QPixmap 생성 및 설정
            pixmap = QPixmap.fromImage(qt_image)
            self.camera_view.setPixmap(pixmap)
            self.camera_view.setScaledContents(True)
            
        except Exception as e:
            self.status_label.setText(f"프레임 업데이트 오류: {str(e)}")
         
            
    def process_data(self, s1, s2):
        """EMG 데이터 처리"""
        if not self.is_running or not self.predictor:
            return
            
        result = self.emg_processor.process_data(s1, s2)
        
        if isinstance(result, tuple) and result[0] == "movement_completed":
            sequence = result[1]
            
            # 동작 예측
            prediction, confidence = self.predictor.predict(sequence)
            
            # 디버깅 로그
            print(f"촬영모드: 예측 결과 - 라벨 {prediction}, 신뢰도 {confidence:.2f}")
            
            # 충분한 신뢰도를 가진 동작만 처리
            if confidence >= self.confidence_threshold:
                # 대기 상태(라벨 0)는 무시
                if prediction == LABEL_IDLE:
                    self.detected_label.setText("감지된 동작: 대기 상태")
                    print(f"대기 상태 감지됨 (신뢰도: {confidence:.2f})")
                    return
                
                # 라벨 1-3 또는 가위/바위/보에 따른 기능 실행
                if 1 <= prediction <= 3 or prediction in (LABEL_SCISSORS, LABEL_ROCK, LABEL_PAPER):
                    # 라벨 표시
                    if prediction == LABEL_SCISSORS:
                        self.detected_label.setText(f"감지된 동작: 가위")
                    elif prediction == LABEL_ROCK:
                        self.detected_label.setText(f"감지된 동작: 바위")
                    elif prediction == LABEL_PAPER:
                        self.detected_label.setText(f"감지된 동작: 보")
                    else:
                        self.detected_label.setText(f"감지된 동작: 동작 {prediction}")
                    
                    # 동작 실행 (쿨다운 체크)
                    if not self.action_cooldown:
                        self.execute_action(prediction)
                        self.action_cooldown = True
                        self.cooldown_timer.start(1500)  # 1.5초 쿨다운
                    
    def reset_action_cooldown(self):
        """동작 실행 쿨다운 초기화"""
        self.action_cooldown = False
        self.cooldown_timer.stop()
        
    def update_detection(self):
        """감지 상태 업데이트"""
        # 현재는 특별한 작업이 필요 없음
        pass
        
    def execute_action(self, label_value):
        """동작 실행 - 통합 라벨 지원"""
        if not self.camera:
            return
            
        try:
            # 라벨 1-3 또는 가위/바위/보에 따른 기능 실행
            if label_value == 1 or label_value == LABEL_SCISSORS:
                # 사진 촬영
                self.take_photo()
                
            elif label_value == 2 or label_value == LABEL_ROCK:
                # 동영상 촬영 시작/중지
                if not self.recording:
                    self.start_recording()
                else:
                    self.stop_recording()
                    
            elif label_value == 3 or label_value == LABEL_PAPER:
                # 동영상 일시정지/재개
                self.toggle_pause_recording()
                
        except Exception as e:
            self.status_label.setText(f"동작 실행 오류: {str(e)}")
    
    # 나머지 메서드들은 기존 코드와 동일
    def take_photo(self):
        """사진 촬영"""
        if not self.camera or not self.output_dir:
            return
            
        try:
            # 현재 시간으로 파일명 생성
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"photo_{timestamp}.jpg"
            filepath = os.path.join(self.output_dir, filename)
            
            # 현재 프레임 저장
            ret, frame = self.camera.read()
            if not ret:
                self.status_label.setText("사진을 찍을 수 없습니다.")
                return
                
            # 사진 저장
            import cv2
            cv2.imwrite(filepath, frame)
            
            # 상태 업데이트
            self.status_label.setText(f"사진 저장 완료: {filename}")
            
            # 카메라 뷰에 사진 효과 적용
            self.camera_view.setStyleSheet("border: 5px solid white;")
            QTimer.singleShot(200, lambda: self.camera_view.setStyleSheet(""))
            
        except Exception as e:
            self.status_label.setText(f"사진 촬영 오류: {str(e)}")
            
    def start_recording(self):
        """동영상 녹화 시작"""
        if not self.camera or not self.output_dir or self.recording:
            return
            
        try:
            import cv2
            
            # 현재 시간으로 파일명 생성
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"video_{timestamp}.avi"
            filepath = os.path.join(self.output_dir, filename)
            
            # 동영상 코덱 및 해상도 설정
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            fps = 20.0
            
            # 카메라 해상도 가져오기
            width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # 비디오 라이터 생성
            self.video_writer = cv2.VideoWriter(filepath, fourcc, fps, (width, height))
            self.current_video_path = filepath
            
            # 상태 업데이트
            self.recording = True
            self.paused = False
            self.status_label.setText("동영상 녹화 중...")
            
        except Exception as e:
            self.status_label.setText(f"녹화 시작 오류: {str(e)}")
            
    def stop_recording(self):
        """동영상 녹화 중지"""
        if not self.recording or not self.video_writer:
            return
            
        try:
            # 비디오 라이터 해제
            if self.video_writer:
                self.video_writer.release()
                self.video_writer = None
                
            # 상태 업데이트
            self.recording = False
            self.paused = False
            self.status_label.setText(f"녹화 완료: {os.path.basename(self.current_video_path)}")
            self.recording_indicator.setStyleSheet("background-color: grey; border-radius: 10px;")
            
        except Exception as e:
            self.status_label.setText(f"녹화 중지 오류: {str(e)}")
            
    def toggle_pause_recording(self):
        """동영상 녹화 일시정지/재개"""
        if not self.recording:
            return
            
        try:
            # 일시정지/재개 상태 토글
            self.paused = not self.paused
            
            if self.paused:
                self.status_label.setText("녹화 일시정지됨")
                self.recording_indicator.setStyleSheet("background-color: orange; border-radius: 10px;")
            else:
                self.status_label.setText("녹화 재개됨")
                
        except Exception as e:
            self.status_label.setText(f"녹화 일시정지/재개 오류: {str(e)}")
            
    def set_output_dir(self):
        """출력 디렉토리 설정"""
        from PyQt5.QtWidgets import QFileDialog
        
        output_dir = QFileDialog.getExistingDirectory(
            self, "저장 경로 선택", self.output_dir, 
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks
        )
        
        if output_dir:
            self.output_dir = output_dir
            self.status_label.setText(f"저장 경로 설정됨: {output_dir}")
     


# ------------------------------------------------------------------------
# 메인 애플리케이션
# ------------------------------------------------------------------------
class EMGGameApplication(QMainWindow):
    """
    EMGGameApplication 클래스: 메인 애플리케이션 윈도우
    - 각 위젯(화면)을 관리하고 전환
    - 전역 SerialThread 관리
    - 프로그램 초기화 및 종료 처리
    - 네오모피즘 디자인 적용
    """
    def __init__(self):
        """초기화 메소드 - 화면 크기 조정 관련 코드 제거"""
        super().__init__()
        
        # 폰트 스케일링 관련 변수 추가
        self.base_font_sizes = {}  # 원본 폰트 크기 저장용
        self.last_scale_factor = 1.0  # 마지막 스케일 팩터
        
        # 상태 변수
        self.username = ""
        self.password = ""
        
        # 전역 SerialThread 관리
        self.global_serial_thread = None
        self.connected_widgets = []  # 현재 데이터를 구독 중인 위젯 목록
        
        # UI 초기화
        self.init_ui()
        
        # 이미지 폴더 및 파일 확인
        self.check_resources()
        
    def resizeEvent(self, event):
        """창 크기 변경 이벤트 처리"""
        super().resizeEvent(event)
        
    def apply_font_scale(self, scale_factor):
        """폰트 스케일 적용 - 중앙 위젯만 처리"""
        # 현재 표시된 위젯에만 스케일 적용
        current_widget = self.central_widget.currentWidget()
        if current_widget:
            self.apply_font_scale_to_widget(current_widget, scale_factor)
        
    def apply_font_scale_to_widget(self, widget, scale_factor):
        """위젯에 폰트 스케일 적용"""
        try:
            # 기본 폰트 크기 저장 또는 가져오기
            widget_id = id(widget)
        
            # 버튼, 레이블, 콤보박스 등 텍스트를 표시하는 위젯만 처리
            if isinstance(widget, (QPushButton, QLabel, QComboBox, QLineEdit)):
                font = widget.font()
            
                # 원본 크기 저장
                if widget_id not in self.base_font_sizes:
                    if font.pointSizeF() > 0:
                        self.base_font_sizes[widget_id] = font.pointSizeF()
                    elif font.pixelSize() > 0:
                        self.base_font_sizes[widget_id] = font.pixelSize() / 100.0  # 포인트 크기로 변환
                    else:
                        self.base_font_sizes[widget_id] = 10.0  # 기본값
            
                # 스케일 적용
                base_size = self.base_font_sizes[widget_id]
                new_size = base_size * scale_factor
            
                # 범위 제한
                new_size = max(8.0, min(new_size, 24.0))
            
                # 포인트 크기로 설정
                font.setPointSizeF(new_size)
                widget.setFont(font)
        
            # 자식 위젯 처리 (QLayout은 제외)
            for child in widget.children():
                if isinstance(child, QWidget) and not isinstance(child, QLayout):
                    self.apply_font_scale_to_widget(child, scale_factor)

        except Exception as e:
            print(f"Font scaling error: {e}")
            
    def init_ui(self):
        """UI 초기화 - 1920x1080 전체화면 기준"""
        # 메인 윈도우 설정
        self.setWindowTitle("EMG 동작 인식 프로그램")
        self.setMinimumSize(1200, 800)  # 최소 크기 증가
        
        # 배경 설정
        self.setStyleSheet(f"background-color: {BG_COLOR};")
        
        # 중앙 위젯
        self.central_widget = QStackedWidget()
        self.setCentralWidget(self.central_widget)
        
        # 게임 선택 대화상자
        self.game_selection_dialog = GameSelectionDialog(self)
        self.game_selection_dialog.game_selected.connect(self.on_game_selected)
        
        # 로그인 위젯
        self.login_widget = LoginWidget()
        self.login_widget.login_successful.connect(self.on_login)
        
        # 메인 메뉴 위젯
        self.main_menu = MainMenuWidget()
        self.main_menu.show_guide.connect(self.show_guide)
        self.main_menu.show_data_collection.connect(self.show_data_collection)
        self.main_menu.show_game_selection.connect(self.show_game_selection)
        self.main_menu.show_screen_control.connect(self.show_screen_control)
        self.main_menu.show_recording.connect(self.show_recording)
        self.main_menu.show_settings.connect(self.show_settings)
        
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
        
        # 화면 제어 모드 위젯
        self.screen_control = ScreenControlWidget()
        self.screen_control.back_to_main.connect(self.show_main_menu)
        
        # 촬영 모드 위젯
        self.recording = RecordingWidget()
        self.recording.back_to_main.connect(self.show_main_menu)
        
        # 설정 위젯
        self.settings = SettingsWidget()
        self.settings.back_to_main.connect(self.show_main_menu)
        
        # 스택에 위젯 추가
        self.central_widget.addWidget(self.login_widget)
        self.central_widget.addWidget(self.main_menu)
        self.central_widget.addWidget(self.guide_widget)
        self.central_widget.addWidget(self.data_collection)
        self.central_widget.addWidget(self.model_training)
        self.central_widget.addWidget(self.rps_game)
        self.central_widget.addWidget(self.muk_game)
        self.central_widget.addWidget(self.screen_control)
        self.central_widget.addWidget(self.recording)
        self.central_widget.addWidget(self.settings)
        
        # 로그인 화면으로 시작
        self.central_widget.setCurrentWidget(self.login_widget)
        # 안드로이드 연결 위젯 추가
        self.android_connection = AndroidConnectionWidget()
        self.android_connection.back_to_main.connect(self.show_main_menu)

        # 메인 메뉴에 시그널 연결 추가
        self.main_menu.show_android_connection.connect(self.show_android_connection)
        self.main_menu.show_hand_model.connect(self.show_hand_model)
        # 스택에 위젯 추가
        self.central_widget.addWidget(self.android_connection)
        
    def check_resources(self):
        """필요한 리소스 확인"""
        # 사용자 디렉토리 확인
        ensure_dir_exists(USERS_DIR)
        
        # 공통 데이터 디렉토리 확인
        ensure_dir_exists(COMMON_DATA_DIR)
        
        # 이미지 생성
        create_placeholder_images()
    def show_android_connection(self):
        """안드로이드 연결 화면 표시"""
        # 통합 모델 확인
        user_dir = get_user_dir(self.username, self.password)
        has_model = os.path.exists(os.path.join(user_dir, "emg_model.pth"))
        
        if not has_model:
            QMessageBox.warning(
                self, 
                '모델 없음', 
                "통합 동작 인식 모델이 없습니다. 먼저 데이터 수집에서 필요한 동작 데이터를 모으고 모델을 학습해주세요.",
                QMessageBox.Ok
            )
            return
            
        self.android_connection.set_user(self.username, self.password)
        self.central_widget.setCurrentWidget(self.android_connection)    
    def init_global_serial_thread(self, config):
        """전역 SerialThread 초기화 - 개선된 버전"""
        # 기존 스레드가 있으면 완전히 정리
        if self.global_serial_thread:
            print("기존 SerialThread 정리 중...")
            
            # 모든 연결된 위젯의 시그널 해제
            for widget in self.connected_widgets[:]:  # 복사본으로 순회
                self.unsubscribe_widget(widget)
            
            # 스레드 정리
            if self.global_serial_thread.isRunning():
                self.global_serial_thread.stop()
            
            # 객체 삭제
            self.global_serial_thread.deleteLater()
            self.global_serial_thread = None
            
            # 이벤트 처리 및 잠시 대기
            QApplication.processEvents()
            time.sleep(0.1)
        
        print("새 SerialThread 생성 중...")
        
        # 새 SerialThread 생성
        self.global_serial_thread = SerialThread(parent=self)
        
        # 설정 적용
        connection_type = config.get("connection_type", "usb")
        port = config.get("port", DEFAULT_SERIAL_PORT)
        baud_rate = config.get("baud_rate", 115200 if connection_type == "usb" else 9600)
        
        self.global_serial_thread.set_connection_type(connection_type)
        self.global_serial_thread.set_port(port)
        self.global_serial_thread.set_baud_rate(baud_rate)
        
        print(f"SerialThread 설정: {port}, {baud_rate}bps, {connection_type}")
        
        # 스레드 시작
        self.global_serial_thread.start()
        
        # 연결 상태 확인 (약간의 대기 시간 후)
        QApplication.processEvents()
        time.sleep(0.1)
        
        success = (self.global_serial_thread.ser is not None and 
                  self.global_serial_thread.ser.is_open)
        
        print(f"전역 SerialThread 초기화: {'성공' if success else '실패'}")
        
        if success:
            print(f"시리얼 포트 상태: {self.global_serial_thread.ser.is_open}")
        
        return success
    
    def update_serial_config(self, port, baud_rate, connection_type):
        """전역 SerialThread 설정 업데이트 - 개선된 버전"""
        print(f"SerialThread 설정 업데이트 요청: {port}, {baud_rate}bps, {connection_type}")
        
        # 설정만 변경하는 경우 확인
        if self.global_serial_thread:
            # 현재 설정과 동일한지 확인
            current_port = self.global_serial_thread.port
            current_baud = self.global_serial_thread.baud_rate
            current_type = self.global_serial_thread.connection_type
            
            if (current_port == port and 
                current_baud == baud_rate and 
                current_type == connection_type):
                print("설정이 동일함, 변경 없음")
                return True  # 변경 없음
        
        # 설정이 다르면 재초기화
        config = {
            "port": port, 
            "baud_rate": baud_rate, 
            "connection_type": connection_type
        }
        return self.init_global_serial_thread(config)
    
    def subscribe_widget(self, widget):
        """위젯을 SerialThread에 구독 추가 - 개선된 버전"""
        if not self.global_serial_thread:
            print(f"구독 실패: SerialThread가 없음 ({widget.__class__.__name__})")
            return False
            
        if widget in self.connected_widgets:
            print(f"이미 구독 중: {widget.__class__.__name__}")
            return True
            
        try:
            # 중복 연결 방지를 위해 먼저 disconnect 시도
            try:
                self.global_serial_thread.data_received.disconnect(widget.process_data)
            except:
                pass
            
            # 새로 연결
            self.global_serial_thread.data_received.connect(widget.process_data)
            self.connected_widgets.append(widget)
            print(f"위젯 구독 성공: {widget.__class__.__name__}")
            return True
            
        except Exception as e:
            print(f"구독 중 오류: {widget.__class__.__name__} - {e}")
            return False
    
    def unsubscribe_widget(self, widget):
        """위젯을 SerialThread 구독에서 제거 - 개선된 버전"""
        if widget not in self.connected_widgets:
            return False
            
        if not self.global_serial_thread:
            # SerialThread가 없어도 리스트에서는 제거
            self.connected_widgets.remove(widget)
            return True
            
        try:
            self.global_serial_thread.data_received.disconnect(widget.process_data)
            self.connected_widgets.remove(widget)
            print(f"위젯 구독 해제 성공: {widget.__class__.__name__}")
            return True
        except Exception as e:
            print(f"구독 해제 중 오류: {widget.__class__.__name__} - {e}")
            # 오류가 발생해도 리스트에서는 제거
            if widget in self.connected_widgets:
                self.connected_widgets.remove(widget)
            return False
        
    def on_login(self, username, password):
        """로그인 처리"""
        self.username = username
        self.password = password
        
        # 사용자 디렉토리 생성
        user_dir = get_user_dir(username, password)
        ensure_dir_exists(user_dir)
        
        # 전역 SerialThread 초기화
        config = load_user_config(username, password)
        self.init_global_serial_thread(config)
        
        # 게임 모델 파일 확인
        has_model = os.path.exists(os.path.join(user_dir, "emg_model.pth"))
        
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
        
    def show_model_training(self, model_type="unified"):
        """모델 학습 화면 표시 - 통합 모델 사용"""
        if model_type == "none":
            # 학습 없이 메인 메뉴로
            self.show_main_menu()
            return
            
        self.model_training.set_user(self.username, self.password)
        self.model_training.start()
        self.central_widget.setCurrentWidget(self.model_training)
        
    def show_game_selection(self):
        """게임 선택 대화상자 표시"""
        # 게임 모델 확인
        user_dir = get_user_dir(self.username, self.password)
        has_model = os.path.exists(os.path.join(user_dir, "emg_model.pth"))
            
        if not has_model:
            QMessageBox.warning(
                self, 
                '모델 없음', 
                "게임 모델이 없습니다. 먼저 데이터 수집에서 게임 데이터를 모으고 모델을 학습해주세요.",
                QMessageBox.Ok
            )
            return
            
        self.game_selection_dialog.exec_()
        
    def on_game_selected(self, game_index):
        """게임 선택 처리"""
        if game_index == 0:
            self.show_rps_game()
        else:
            self.show_muk_game()
            
    def show_rps_game(self):
        """가위바위보 게임 화면 표시"""
        self.rps_game.set_user(self.username, self.password)
        
        # 게임 설정 로드
        config = load_user_config(self.username, self.password)
        if "rps_rounds" in config:
            self.rps_game.max_rounds = config["rps_rounds"]
            
        self.rps_game.start()
        self.central_widget.setCurrentWidget(self.rps_game)
        
    def show_muk_game(self):
        """묵찌빠 게임 화면 표시"""
        self.muk_game.set_user(self.username, self.password)
        
        # 게임 설정 로드
        config = load_user_config(self.username, self.password)
        if "muk_rounds" in config:
            self.muk_game.max_rounds = config["muk_rounds"]
            
        self.muk_game.start()
        self.central_widget.setCurrentWidget(self.muk_game)
        
    def show_screen_control(self):
        """화면 제어 모드 표시 - 통합 모델 전용"""
        # 통합 모델 확인
        user_dir = get_user_dir(self.username, self.password)
        has_model = os.path.exists(os.path.join(user_dir, "emg_model.pth"))
                
        if not has_model:
            QMessageBox.warning(
                self, 
                '모델 없음', 
                "통합 동작 인식 모델이 없습니다. 먼저 데이터 수집에서 필요한 동작 데이터를 모으고 모델을 학습해주세요.",
                QMessageBox.Ok
            )
            return
                
        self.screen_control.set_user(self.username, self.password)
        self.central_widget.setCurrentWidget(self.screen_control)
        
    def show_recording(self):
        """촬영 모드 표시"""
        # 모델 확인
        user_dir = get_user_dir(self.username, self.password)
        has_model = os.path.exists(os.path.join(user_dir, "emg_model.pth"))
            
        if not has_model:
            QMessageBox.warning(
                self, 
                '모델 없음', 
                "동작 제어 모델이 없습니다. 먼저 데이터 수집에서 동작 제어 데이터를 모으고 모델을 학습해주세요.",
                QMessageBox.Ok
            )
            return
            
        # OpenCV 설치 확인
        try:
            import cv2
        except ImportError:
            QMessageBox.warning(
                self,
                'OpenCV 누락',
                "OpenCV 모듈이 설치되지 않았습니다. 촬영 모드를 사용하려면 'pip install opencv-python'을 실행하세요.",
                QMessageBox.Ok
            )
            return
            
        self.recording.set_user(self.username, self.password)
        self.central_widget.setCurrentWidget(self.recording)
        
    def show_settings(self):
        """설정 화면 표시"""
        self.settings.set_user(self.username, self.password)
        self.central_widget.setCurrentWidget(self.settings)
        
    def closeEvent(self, event):
        """프로그램 종료 처리"""
        print("프로그램 종료 중...")
        
        # 모든 활성 위젯 정리
        for widget in self.connected_widgets[:]:  # 복사본으로 순회
            if hasattr(widget, 'stop'):
                widget.stop()
            self.unsubscribe_widget(widget)
        
        # 전역 SerialThread 정리
        if self.global_serial_thread:
            print("전역 SerialThread 정리 중...")
            if self.global_serial_thread.isRunning():
                self.global_serial_thread.stop()
            self.global_serial_thread.disconnect_serial()
            self.global_serial_thread = None
            
        self.connected_widgets.clear()
        
        print("종료 완료")
        event.accept()
#--------------------------------------------------------!
    def closeEvent(self, event):
        print("애플리케이션 종료 요청...")
        # Unity 통신 서버 중지
        if self.main_menu and hasattr(self.main_menu, 'stop_unity_server'):
            self.main_menu.stop_unity_server()

        # 기존 종료 로직
        if self.global_serial_thread and self.global_serial_thread.isRunning():
            print("시리얼 스레드 종료 중...")
            self.global_serial_thread.stop()
            self.global_serial_thread.wait()  # 스레드가 완전히 종료될 때까지 대기
            print("시리얼 스레드 종료됨.")

        # 다른 위젯들의 stop 메서드 호출 (필요한 경우)
        # 예를 들어, self.active_widget 이 현재 활성화된 위젯을 가리킨다면
        if hasattr(self, 'active_widget') and self.active_widget and hasattr(self.active_widget, 'stop'):
            print(f"{self.active_widget.__class__.__name__} 종료 중...")
            self.active_widget.stop()
            print(f"{self.active_widget.__class__.__name__} 종료됨.")

        # Unity 핸들러 스레드 종료
        if hasattr(self, 'unity_handler_thread') and self.unity_handler_thread:
            if self.unity_handler_thread.isRunning():
                print("Unity 핸들러 스레드 종료 시도...")
                self.unity_handler_thread.quit() # quit()을 먼저 시도
                if not self.unity_handler_thread.wait(3000): # 3초 대기
                    print("Unity 핸들러 스레드가 정상적으로 종료되지 않아 강제 종료합니다.")
                    self.unity_handler_thread.terminate() # 강제 종료
                    self.unity_handler_thread.wait() # 강제 종료 후 대기
                print("Unity 핸들러 스레드 종료됨.")
            
            # Unity 핸들러가 관리하는 프로세스가 있다면 그것도 종료
            if hasattr(self.unity_handler_thread, 'terminate_process'):
                self.unity_handler_thread.terminate_process()


        print("모든 정리 작업 완료. 애플리케이션 종료.")
        event.accept()
#--------------------------------------------------------!        
    def show_hand_model(self):
        """손모델 보기 화면 표시 (임시 - 빈 창)"""
        QMessageBox.information(
            self, 
            '손모델 보기', 
            "손모델 보기 기능이 곧 추가될 예정입니다.",
            QMessageBox.Ok
        )
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