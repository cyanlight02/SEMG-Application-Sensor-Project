import sys
import os
import csv
import json
import time
import serial
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pyqtgraph as pg
from tqdm import tqdm
from datetime import datetime
from collections import deque
from threading import Thread, Lock
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QPushButton, QLabel, QComboBox, 
                            QTabWidget, QLineEdit, QTextEdit, QTableWidget, 
                            QTableWidgetItem, QGroupBox, QGridLayout, QMessageBox,
                            QFileDialog, QListWidget, QProgressBar, QDialog,
                            QDialogButtonBox, QFormLayout, QSpinBox, QDoubleSpinBox,
                            QListWidgetItem, QFrame, QCheckBox)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread, QObject
from PyQt5.QtGui import QColor, QFont, QPalette

# ------------------------------------------------------------------------
# 공통 상수 및 설정
# ------------------------------------------------------------------------
# 센서 및 시리얼 설정
DEFAULT_SERIAL_PORT = 'COM9'
DEFAULT_BAUD_RATE = 115200

# 모델 관련 상수
SEQUENCE_LENGTH = 200  # 트랜스포머 모델용 시퀀스 길이 (500Hz 기준 0.4초)
BUFFER_SIZE = 500      # 감지용 롤링 버퍼 크기

# 데이터 수집 관련 상수
DETECTION_WINDOW = 20  # 동작 감지에 사용할 윈도우 크기
DETECTION_THRESHOLD = 5  # 동작 감지 임계값
COOLDOWN_PERIOD = 2    # 동작 저장 후 대기 시간(초)
POST_CAPTURE = 30      # 동작 종료 후 추가 캡처 프레임 수
DATA_FOLDER = "emg_dataset"  # 데이터를 저장할 폴더 이름
TREND_WINDOW = 10      # 추세 감지에 사용할 윈도우 크기
TREND_THRESHOLD = 3  # 추세 감지 임계값
DISPERSION_THRESHOLD = 3 #분산 감지 임계값값
# 모델 저장 위치
MODEL_FILE = "emg_transformer_model.pth"
LABEL_FILE = "emg_labels.json"

# ------------------------------------------------------------------------
# 공통 모델 정의 (모든 모듈에서 사용하는 트랜스포머 모델)
# ------------------------------------------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1), :]
        return self.dropout(x)

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src2 = self.norm1(src)
        src2, _ = self.self_attn(src2, src2, src2, attn_mask=src_mask,
                                key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

class EMGTransformer(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, dim_feedforward, num_classes, dropout=0.1):
        super().__init__()
        
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        encoder_layers = []
        for _ in range(num_layers):
            encoder_layers.append(TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout))
        self.transformer_encoder = nn.ModuleList(encoder_layers)
        
        self.fc = nn.Linear(d_model, num_classes)
    
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
# 유틸리티 함수
# ------------------------------------------------------------------------
def ensure_data_folder_exists():
    """데이터 저장 폴더가 존재하지 않으면 생성"""
    if not os.path.exists(DATA_FOLDER):
        os.makedirs(DATA_FOLDER)
        return f"데이터 저장 폴더 '{DATA_FOLDER}' 생성 완료"
    return f"데이터 폴더 '{DATA_FOLDER}' 확인 완료"

def load_label_names():
    """라벨 이름 로드 - 외부 파일이 있으면 로드, 없으면 기본값 사용"""
    # 기본 라벨 설정
    label_names = {
        1: "검지",
        2: "가위",
        3: "바위",
        4: "보",
        5: "1",
        6: "2",
        7: "3",
        8: "핑거스냅"
    }
    
    # 외부 라벨 파일이 있는지 확인
    if os.path.exists(LABEL_FILE):
        try:
            with open(LABEL_FILE, 'r', encoding='utf-8') as f:
                loaded_labels = json.load(f)
                # 정수 키로 변환 (JSON은 키를 문자열로 저장)
                label_names = {int(k): v for k, v in loaded_labels.items()}
            print(f"라벨 파일 '{LABEL_FILE}'을 성공적으로 로드했습니다.")
        except Exception as e:
            print(f"라벨 파일 로드 중 오류 발생: {e}")
            print("기본 라벨 설정을 사용합니다.")
    else:
        print("라벨 파일이 없습니다. 기본 라벨 설정을 사용합니다.")
    
    return label_names

def save_label_names(label_names):
    """라벨 이름을 파일로 저장"""
    try:
        with open(LABEL_FILE, 'w', encoding='utf-8') as f:
            json.dump(label_names, f, ensure_ascii=False, indent=2)
        return True, f"라벨 설정이 '{LABEL_FILE}' 파일에 저장되었습니다."
    except Exception as e:
        return False, f"라벨 설정 저장 중 오류 발생: {e}"

# ------------------------------------------------------------------------
# 멀티스레딩 시리얼 데이터 리더
# ------------------------------------------------------------------------
class SerialReaderThread(QThread):
    data_received = pyqtSignal(float, float)
    error_occurred = pyqtSignal(str)
    connection_status = pyqtSignal(bool, str)
    
    def __init__(self, port=DEFAULT_SERIAL_PORT, baud_rate=DEFAULT_BAUD_RATE, parent=None):
        super().__init__(parent)
        self.port = port
        self.baud_rate = baud_rate
        self.running = False
        self.ser = None
    
    def set_port(self, port):
        self.port = port
    
    def set_baud_rate(self, baud_rate):
        self.baud_rate = baud_rate
    
    def connect_serial(self):
        try:
            self.ser = serial.Serial(self.port, self.baud_rate, timeout=1)
            time.sleep(2)  # 연결 안정화 대기
            self.ser.reset_input_buffer()
            self.connection_status.emit(True, f"{self.port}에 성공적으로 연결되었습니다.")
            return True
        except Exception as e:
            self.connection_status.emit(False, f"연결 실패: {str(e)}")
            return False
    
    def run(self):
        if not self.connect_serial():
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
                self.error_occurred.emit(f"시리얼 읽기 오류: {str(e)}")
                self.running = False
                break
    
    def stop(self):
        self.running = False
        if self.ser:
            self.ser.close()
            self.ser = None

# ------------------------------------------------------------------------
# 데이터 수집 위젯
# ------------------------------------------------------------------------
class EMGDataCollectorWidget(QWidget):
    status_update = pyqtSignal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.serial_thread = None
        
        # 상태 변수 초기화
        self.data_buffer = deque(maxlen=BUFFER_SIZE)
        self.sequence_buffer = []
        self.is_recording = False
        self.cooldown = False
        self.cooldown_start = 0
        self.current_label = None
        self.label_counts = {i: 0 for i in range(10)}
        self.current_values = (0, 0)
        self.baseline_mean = (0, 0)
        self.baseline_std = (0, 0)
        self.post_capture_count = 0
        self.filename = None
        self.base_filename = "emg_data"
        self.file_index = 0
        self.sensitivity_level = 1.0
        self.detection_blocked = False
        self.detection_block_time = 0
        self.detection_block_duration = 0
        
        # UI 초기화
        self.init_ui()
        
        # 타이머 설정 (상태 업데이트용)
        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self.update_status_display)
        self.update_timer.start(300)  # 300ms마다 업데이트
        
        # 데이터 폴더 확인
        ensure_data_folder_exists()
    
    def init_ui(self):
        layout = QVBoxLayout()
        
        # 연결 설정 그룹
        connection_group = QGroupBox("연결 설정")
        connection_layout = QHBoxLayout()
        
        self.port_label = QLabel("시리얼 포트:")
        self.port_combo = QComboBox()
        self.refresh_ports()
        self.port_combo.setEditable(True)  # 직접 입력 가능
        
        self.connect_btn = QPushButton("연결")
        self.connect_btn.clicked.connect(self.connect_serial)
        
        self.disconnect_btn = QPushButton("연결 해제")
        self.disconnect_btn.clicked.connect(self.disconnect_serial)
        self.disconnect_btn.setEnabled(False)
        
        connection_layout.addWidget(self.port_label)
        connection_layout.addWidget(self.port_combo)
        connection_layout.addWidget(self.connect_btn)
        connection_layout.addWidget(self.disconnect_btn)
        connection_group.setLayout(connection_layout)
        
        # 데이터 수집 설정 그룹
        collection_group = QGroupBox("데이터 수집 설정")
        collection_layout = QGridLayout()
        
        self.filename_label = QLabel("기본 파일명:")
        self.filename_edit = QLineEdit(self.base_filename)
        
        self.sensitivity_label = QLabel("감도 수준:")
        self.sensitivity_spin = QDoubleSpinBox()
        self.sensitivity_spin.setRange(0.5, 2.0)
        self.sensitivity_spin.setSingleStep(0.1)
        self.sensitivity_spin.setValue(self.sensitivity_level)
        self.sensitivity_spin.valueChanged.connect(self.update_sensitivity)
        
        collection_layout.addWidget(self.filename_label, 0, 0)
        collection_layout.addWidget(self.filename_edit, 0, 1)
        collection_layout.addWidget(self.sensitivity_label, 1, 0)
        collection_layout.addWidget(self.sensitivity_spin, 1, 1)
        collection_group.setLayout(collection_layout)
        
        # 라벨 선택 그룹
        label_group = QGroupBox("라벨 선택")
        label_layout = QGridLayout()
        
        self.label_buttons = []
        for i in range(10):
            btn = QPushButton(f"라벨 {i}")
            btn.setCheckable(True)
            btn.clicked.connect(lambda checked, i=i: self.select_label(i))
            self.label_buttons.append(btn)
            label_layout.addWidget(btn, i // 5, i % 5)
        # 휴식 상태 버튼 추가
        self.rest_btn = QPushButton("휴식 상태(라벨 취소)")
        self.rest_btn.setCheckable(True)
        self.rest_btn.clicked.connect(self.select_rest_state)
        self.rest_btn.setStyleSheet("background-color: #5c8a8a;")  # 구분을 위한 색상 지정

        # 버튼을 레이아웃 마지막 줄에 추가 (2개의 열을 차지하도록)
        label_layout.addWidget(self.rest_btn, 2, 0, 1, 5)  # 2행, 0열에서 시작해 1행 5열 차지
        label_group.setLayout(label_layout)
        
        # 데이터 시각화 그룹
        visual_group = QGroupBox("실시간 EMG 데이터")
        visual_layout = QVBoxLayout()

        # 그래프 표시 여부 체크박스 추가
        self.show_graph_check = QCheckBox("실시간 그래프 표시")
        self.show_graph_check.setChecked(True)
        visual_layout.addWidget(self.show_graph_check)

        # matplotlib 대신 pyqtgraph 사용
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('k')  # 검은색 배경
        self.plot_widget.setTitle("실시간 EMG 신호", color='w')
        self.plot_widget.setLabel('left', 'EMG 값', color='w')
        self.plot_widget.setLabel('bottom', '시간', color='w')
        self.plot_widget.setYRange(-20, 20)  # y축 범위 설정 (이 부분을 원하는 값으로 수정 가능)
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)

        # 두 센서를 위한 그래프 선 생성
        self.curve1 = self.plot_widget.plot(pen=pg.mkPen('r', width=2), name="센서 1")
        self.curve2 = self.plot_widget.plot(pen=pg.mkPen('b', width=2), name="센서 2")

        # 플롯 데이터
        self.plot_data1 = []
        self.plot_data2 = []
        self.max_plot_points = 200  # 그래프에 표시할 최대 포인트 수

        visual_layout.addWidget(self.plot_widget)
        visual_group.setLayout(visual_layout)
        
        # 상태 및 제어 그룹
        status_group = QGroupBox("상태 및 제어")
        status_layout = QVBoxLayout()
        
        self.status_text = QTextEdit()
        self.status_text.setReadOnly(True)
        self.status_text.setMaximumHeight(150)
        
        self.delete_last_btn = QPushButton("마지막 항목 삭제")
        self.delete_last_btn.clicked.connect(self.delete_last_sequence)
        self.delete_last_btn.setEnabled(False)
        
        status_layout.addWidget(self.status_text)
        status_layout.addWidget(self.delete_last_btn)
        status_group.setLayout(status_layout)
        
        # 메인 레이아웃에 추가
        layout.addWidget(connection_group)
        layout.addWidget(collection_group)
        layout.addWidget(label_group)
        layout.addWidget(visual_group)
        layout.addWidget(status_group)
        
        self.setLayout(layout)
        
        # 초기 상태 업데이트
        self.add_status_message("EMG 데이터 수집을 시작하려면 연결 버튼을 누르세요.")
    def select_rest_state(self):
        """휴식 상태(라벨 없음) 선택"""
        # 모든 라벨 버튼 선택 해제
        for btn in self.label_buttons:
            btn.setChecked(False)
    
        # 휴식 버튼만 선택
        self.rest_btn.setChecked(True)
    
        # 라벨 취소
        self.current_label = None
        self.add_status_message("휴식 상태로 전환됨. 동작 데이터가 수집되지 않습니다.")
    def refresh_ports(self):
        """사용 가능한 시리얼 포트 목록 갱신"""
        self.port_combo.clear()
        try:
            import serial.tools.list_ports
            ports = serial.tools.list_ports.comports()
            for port in ports:
                self.port_combo.addItem(port.device)
            
            # 기본 포트 추가
            if self.port_combo.findText(DEFAULT_SERIAL_PORT) == -1:
                self.port_combo.addItem(DEFAULT_SERIAL_PORT)
            
            self.port_combo.setCurrentText(DEFAULT_SERIAL_PORT)
        except:
            self.port_combo.addItem(DEFAULT_SERIAL_PORT)
    
    def connect_serial(self):
        """시리얼 포트 연결 및 데이터 수집 시작"""
        port = self.port_combo.currentText()
        
        # 파일명 설정
        self.base_filename = self.filename_edit.text()
        if not self.base_filename:
            self.base_filename = "emg_data"
        
        # 파일 생성
        self.filename = self.get_unique_filename()
        self.add_status_message(f"데이터 파일: {self.filename}")
        
        # CSV 파일 생성 및 헤더 작성
        with open(self.filename, 'w', newline='') as f:
            writer = csv.writer(f)
            headers = ['Label', 'Actual_Length']
            for i in range(SEQUENCE_LENGTH):
                headers.extend([f'S1_T{i}', f'S2_T{i}'])
            writer.writerow(headers)
        
        # 라벨 카운트 초기화
        self.label_counts = {i: 0 for i in range(10)}
        self.update_label_buttons()
        
        # 시리얼 스레드 생성 및 시작
        self.serial_thread = SerialReaderThread(port=port)
        self.serial_thread.data_received.connect(self.process_data)
        self.serial_thread.error_occurred.connect(self.handle_error)
        self.serial_thread.connection_status.connect(self.handle_connection_status)
        self.serial_thread.start()
        
        # UI 상태 업데이트
        self.connect_btn.setEnabled(False)
        self.port_combo.setEnabled(False)
        self.delete_last_btn.setEnabled(True)
    
    def disconnect_serial(self):
        """시리얼 포트 연결 해제 및 데이터 수집 중단"""
        if self.serial_thread:
            self.serial_thread.stop()
            self.serial_thread.wait()  # 스레드가 종료될 때까지 대기
            self.serial_thread = None
        
        # UI 상태 업데이트
        self.connect_btn.setEnabled(True)
        self.port_combo.setEnabled(True)
        self.disconnect_btn.setEnabled(False)
        
        self.add_status_message("연결이 해제되었습니다.")
    
    def handle_connection_status(self, success, message):
        """연결 상태 처리"""
        self.add_status_message(message)
        if success:
            self.disconnect_btn.setEnabled(True)
        else:
            self.connect_btn.setEnabled(True)
            self.port_combo.setEnabled(True)
    
    def handle_error(self, error_message):
        """오류 처리"""
        self.add_status_message(f"오류: {error_message}")
        self.disconnect_serial()
    
    def process_data(self, s1, s2):
        """수신된 EMG 데이터 처리"""
        # 현재 값 저장
        self.current_values = (s1, s2)
        
        # 그래프 업데이트를 위한 데이터 추가
        self.plot_data1.append(s1)
        self.plot_data2.append(s2)
        
        # 최대 포인트 수 제한
        if len(self.plot_data1) > self.max_plot_points:
            self.plot_data1.pop(0)
            self.plot_data2.pop(0)
        
        # 버퍼에 추가
        self.data_buffer.append((s1, s2))
        
        # 현재 라벨이 선택되어 있는지 확인
        if self.current_label is None:
            return
        
        # 주기적으로 기준선 업데이트 (적응형)
        if not self.is_recording and len(self.data_buffer) % 50 == 0:
            self.baseline_mean, self.baseline_std = self.calculate_baseline()
        
        # 대기 시간 확인
        if self.cooldown:
            if time.time() - self.cooldown_start >= COOLDOWN_PERIOD:
                self.cooldown = False
                self.add_status_message("대기 시간 종료. 동작 감지 준비 완료.")
                # 데이터 버퍼의 최근 부분을 중립값으로 설정하여 이전 동작의 영향 감소
                # 1. 데이터 버퍼 완전히 비우기
                self.data_buffer.clear()
        
                # 2. 잠시 동작 감지 블로킹 설정
                self.detection_blocked = True
                self.detection_block_time = time.time()
                self.detection_block_duration = 1.0  # 1초 동안 감지 차단
        
                # 로그 추가
                self.add_log("버퍼 초기화 및 감지 일시 중지")
        
        # 동작 감지 및 기록
        elif not self.is_recording:
            if self.detect_movement():
                self.start_recording()
        
        # 동작 기록 중
        else:
            # 현재 값을 시퀀스에 추가
            self.sequence_buffer.append((s1, s2))
            
            # 기록 중지 여부 확인
            if not self.is_movement_continuing():
                self.post_capture_count += 1
                if self.post_capture_count >= POST_CAPTURE:
                    self.save_sequence()
            else:
                self.post_capture_count = 0
            
            # 최대 길이 확인
            if len(self.sequence_buffer) >= SEQUENCE_LENGTH * 1.5:
                self.save_sequence()
    
    def update_plot(self):
        """그래프 업데이트"""
        # 그래프 비활성화 체크 추가
        if not self.show_graph_check.isChecked():
            return
        
        if self.plot_data1 and self.plot_data2:
            # pyqtgraph 방식으로 데이터 업데이트
            self.curve1.setData(self.plot_data1)
            self.curve2.setData(self.plot_data2)
    
    def calculate_baseline(self):
        """버퍼 데이터에서 기준선 통계 계산"""
        if len(self.data_buffer) < 100:
            return (0, 0), (1, 1)  # 데이터가 충분하지 않은 경우 기본값
            
        data = np.array(self.data_buffer)
        sensor1_data = data[:, 0]
        sensor2_data = data[:, 1]
        
        mean1 = np.mean(sensor1_data)
        mean2 = np.mean(sensor2_data)
        std1 = np.std(sensor1_data) + 1  # 0으로 나누는 것을 방지하기 위해 1 추가
        std2 = np.std(sensor2_data) + 1
        
        return (mean1, mean2), (std1, std2)
    
    def detect_movement(self):
        """여러 방법을 통합한 동작 감지"""
        # 감지 차단 확인
        if self.detection_blocked:
            if time.time() - self.detection_block_time >= self.detection_block_duration:
                self.detection_blocked = False
                self.add_log("동작 감지 재개")
            else:
                return False
            if len(self.data_buffer) < DETECTION_WINDOW:
                return False
            
        # 감지에 사용할 최근 데이터 가져오기
        recent_data = list(self.data_buffer)[-DETECTION_WINDOW:]
        
        # 1. Z-점수 기반 감지 (기존 방식)
        z_scores_s1 = [(x[0] - self.baseline_mean[0]) / self.baseline_std[0] for x in recent_data]
        z_scores_s2 = [(x[1] - self.baseline_mean[1]) / self.baseline_std[1] for x in recent_data]
        
        # 유의미한 편차 개수 계산 (감도 조절 적용)
        adjusted_threshold = DETECTION_THRESHOLD / self.sensitivity_level
        count_s1 = sum(1 for z in z_scores_s1 if abs(z) > adjusted_threshold)
        count_s2 = sum(1 for z in z_scores_s2 if abs(z) > adjusted_threshold)
        
        # 2. 추세 기반 감지 (새로운 방식)
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
        
        variance_detected = (s1_var > (self.baseline_std[0] * DISPERSION_THRESHOLD * self.sensitivity_level) or 
                            s2_var > (self.baseline_std[1] * DISPERSION_THRESHOLD * self.sensitivity_level))
        
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
        adjusted_threshold = (DETECTION_THRESHOLD/2) / self.sensitivity_level
        z_scores_s1 = [(x[0] - self.baseline_mean[0]) / self.baseline_std[0] for x in recent_data]
        z_scores_s2 = [(x[1] - self.baseline_mean[1]) / self.baseline_std[1] for x in recent_data]
        
        # 두 가지 조건 중 하나라도 만족하면 동작 계속으로 판단
        # 1. Z-점수 기반 (기존 방식)
        for z1, z2 in zip(z_scores_s1, z_scores_s2):
            if abs(z1) > adjusted_threshold or abs(z2) > adjusted_threshold:
                return True
                
        # 2. 추세 또는 분산 기반 (새로운 방식)
        if len(recent_data) >= 5:
            s1_values = [x[0] for x in recent_data]
            s2_values = [x[1] for x in recent_data]
            
            # 분산 확인
            s1_var = np.var(s1_values)
            s2_var = np.var(s2_values)
            
            if (s1_var > (self.baseline_std[0] * DISPERSION_THRESHOLD * self.sensitivity_level) or 
                s2_var > (self.baseline_std[1] * DISPERSION_THRESHOLD * self.sensitivity_level)):
                return True
        
        return False
    
    def start_recording(self):
        """새로운 동작 시퀀스 기록 시작"""
        self.is_recording = True
        # 감지 이전의 데이터 일부 포함
        self.sequence_buffer = list(self.data_buffer)[-30:]  # 감지 전 30프레임 포함
        self.post_capture_count = 0
        self.add_status_message("동작 감지! 기록 중...")
    
    def save_sequence(self):
        """기록된 시퀀스를 CSV에 저장"""
        if len(self.sequence_buffer) < 20:  # 의미 있는 기록으로 보기에 너무 짧음
            self.add_status_message("시퀀스가 너무 짧아 무시합니다...")
            self.is_recording = False
            self.sequence_buffer = []
            return
            
        # 트랜스포머 모델용 시퀀스 길이 조정
        actual_length = len(self.sequence_buffer)
        sequence = self.sequence_buffer.copy()
        
        # 필요시 패딩 (트랜스포머의 고정 길이 입력을 위해)
        if actual_length < SEQUENCE_LENGTH:
            padding = [(0, 0)] * (SEQUENCE_LENGTH - actual_length)
            sequence = sequence + padding
        elif actual_length > SEQUENCE_LENGTH:
            # 긴 시퀀스에서 가장 관련성 높은 부분 선택
            sequence = sequence[:SEQUENCE_LENGTH]
            actual_length = SEQUENCE_LENGTH
        
        # CSV에 기록
        with open(self.filename, 'a', newline='') as f:
            writer = csv.writer(f)
            row = [self.current_label, actual_length]
            # 모든 센서 값 추가
            for s1, s2 in sequence:
                row.extend([s1, s2])
            writer.writerow(row)
        
        # 상태 업데이트
        self.label_counts[self.current_label] += 1
        self.update_label_buttons()
        self.is_recording = False
        self.cooldown = True
        self.cooldown_start = time.time()
        self.sequence_buffer = []
        self.add_status_message(f"라벨 {self.current_label}의 시퀀스가 저장되었습니다 (총: {self.label_counts[self.current_label]}개)")
    
    def delete_last_sequence(self):
        """가장 최근에 저장된 데이터 한 줄을 삭제"""
        try:
            # 파일이 존재하는지 확인
            if not os.path.exists(self.filename):
                self.add_status_message("삭제할 파일이 존재하지 않습니다.")
                return
                
            # 파일 내용 모두 읽기
            with open(self.filename, 'r', newline='') as f:
                lines = f.readlines()
                
            # 파일에 헤더만 있는 경우
            if len(lines) <= 1:
                self.add_status_message("삭제할 데이터가 없습니다.")
                return
                
            # 마지막 줄 삭제 전에 라벨 확인
            last_line = lines[-1].strip()
            if last_line:
                try:
                    last_label = int(last_line.split(',')[0])
                    self.label_counts[last_label] = max(0, self.label_counts[last_label] - 1)
                    self.update_label_buttons()
                except:
                    pass  # 라벨 확인 실패 시 무시
            
            # 마지막 줄 삭제
            lines = lines[:-1]
            
            # 파일 다시 쓰기
            with open(self.filename, 'w', newline='') as f:
                f.writelines(lines)
                
            self.add_status_message("마지막 데이터 항목이 삭제되었습니다.")
            
        except Exception as e:
            self.add_status_message(f"데이터 삭제 중 오류 발생: {e}")
    
    def get_unique_filename(self):
        """고유한 CSV 출력 파일명 생성 (폴더 경로 포함)"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        while True:
            filename = os.path.join(DATA_FOLDER, f"emg_{self.base_filename}_{timestamp}_{self.file_index}.csv")
            if not os.path.exists(filename):
                return filename
            self.file_index += 1
    
    def select_label(self, label):
        """라벨 선택"""
        for i, btn in enumerate(self.label_buttons):
            btn.setChecked(i == label)
    
        # 휴식 버튼 선택 해제
        self.rest_btn.setChecked(False)
    
        self.current_label = label
        self.add_status_message(f"라벨 {label} 선택됨 (현재 샘플 수: {self.label_counts[label]})")
    
    def update_label_buttons(self):
        """라벨 버튼 텍스트 업데이트"""
        for i, btn in enumerate(self.label_buttons):
            btn.setText(f"라벨 {i} ({self.label_counts[i]})")
    
    def update_sensitivity(self, value):
        """감도 수준 업데이트"""
        self.sensitivity_level = value
        self.add_status_message(f"감도 수준이 {value:.1f}로 설정되었습니다.")
    
    def add_status_message(self, message):
        """상태 메시지 추가"""
        timestamp = datetime.now().strftime('%H:%M:%S')
        self.status_text.append(f"[{timestamp}] {message}")
        # 스크롤을 항상 최하단으로
        scrollbar = self.status_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
        
        # 상위 클래스에도 상태 업데이트 전달
        self.status_update.emit(message)
    
    def update_status_display(self):
        """상태 표시 업데이트 (타이머에 의해 주기적으로 호출)"""
        # 그래프 업데이트
        self.update_plot()
        
        # 현재 상태에 따른 메시지 표시
        status_msg = ""
        if not self.serial_thread or not self.serial_thread.running:
            return
        
        if self.current_label is None:
            status_msg = "라벨을 선택하세요."
        elif self.cooldown:
            elapsed = time.time() - self.cooldown_start
            status_msg = f"대기 시간: {COOLDOWN_PERIOD - elapsed:.1f}초 남음"
        elif self.is_recording:
            status_msg = f"동작 기록 중... ({len(self.sequence_buffer)} 프레임)"
        else:
            status_msg = "동작 감지 대기 중..."
        
        # 윈도우 타이틀 업데이트 (상태 정보 표시)
        if self.parent:
            self.parent.setWindowTitle(f"EMG 동작 인식 시스템 - 데이터 수집 - {status_msg}")
    
    def closeEvent(self, event):
        """위젯 종료 처리"""
        if self.serial_thread:
            self.serial_thread.stop()
            self.serial_thread.wait()
        event.accept()

# ------------------------------------------------------------------------
# 모델 학습 위젯
# ------------------------------------------------------------------------
class TrainingWorker(QObject):
    update_progress = pyqtSignal(int, str)
    finished = pyqtSignal(bool, str)
    update_plot = pyqtSignal(list, list, list, list)
    
    def __init__(self, data_dir, model_params=None):
        super().__init__()
        self.data_dir = data_dir
        
        # 기본 모델 매개변수
        default_params = {
            'input_dim': 2,
            'd_model': 128,
            'nhead': 8,
            'num_layers': 3,
            'dim_feedforward': 256,
            'num_classes': 10,
            'dropout': 0.1,
            'batch_size': 8,
            'num_epochs': 100,
            'learning_rate': 0.0003,
            'sequence_length': SEQUENCE_LENGTH
        }
        
        # 사용자 지정 매개변수로 업데이트
        self.params = default_params
        if model_params:
            self.params.update(model_params)
        
        # 학습 데이터 변수
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
    
    def load_data(self):
        """CSV 데이터를 불러와 모델 학습 형식으로 변환"""
        all_files = [f for f in os.listdir(self.data_dir) if f.endswith('.csv')]
        
        if not all_files:
            return None, None, None
        
        dataframes = []
        for file in all_files:
            file_path = os.path.join(self.data_dir, file)
            df = pd.read_csv(file_path)
            dataframes.append(df)
        
        # 모든 데이터 병합
        data = pd.concat(dataframes, ignore_index=True)
        
        # 라벨과 시퀀스 데이터 분리
        labels = data['Label'].values
        
        # 실제 길이 확인
        actual_lengths = data['Actual_Length'].values
        
        # 시퀀스 데이터 추출 (S1_T0, S2_T0, ... 열)
        sensor_columns = [col for col in data.columns if col.startswith('S1_') or col.startswith('S2_')]
        features = data[sensor_columns].values
        
        # 특성 재구성: [배치, 시퀀스 길이, 채널 수]
        X = np.zeros((len(labels), self.params['sequence_length'], 2))
        
        for i in range(len(features)):
            for t in range(self.params['sequence_length']):
                X[i, t, 0] = features[i][t*2]     # S1 값
                X[i, t, 1] = features[i][t*2+1]   # S2 값
        
        return X, labels, actual_lengths
    
    def train_epoch(self, model, train_loader, optimizer, criterion):
        """한 에폭 학습"""
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # 그래디언트 클리핑
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        # 평균 손실 및 정확도 계산
        train_loss = train_loss / len(train_loader.dataset)
        train_acc = correct / total
        
        return train_loss, train_acc
    
    def validate(self, model, val_loader, criterion):
        """검증"""
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        # 평균 손실 및 정확도 계산
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = correct / total
        
        return val_loss, val_acc
    
    def run(self):
        """모델 학습 실행"""
        try:
            # 시드 설정
            torch.manual_seed(42)
            np.random.seed(42)
            
            self.update_progress.emit(0, f"사용 장치: {self.device}")
            
            # 데이터 로드 및 전처리
            self.update_progress.emit(5, "데이터 로드 중...")
            X, y, actual_lengths = self.load_data()
            
            if X is None or len(X) == 0:
                self.finished.emit(False, "데이터를 찾을 수 없습니다. 데이터 수집을 먼저 실행해주세요.")
                return
            
            # 라벨 분포 분석
            label_stats = {}
            for label in sorted(np.unique(y)):
                count = np.sum(y == label)
                label_stats[int(label)] = int(count)
            
            self.update_progress.emit(10, f"라벨 분포: {label_stats}")
            
            # 데이터 증강 적용
            self.update_progress.emit(15, "데이터 증강 적용 중...")
            X_aug = []
            y_aug = []
            
            # 원본 데이터 포함
            X_aug.append(X)
            y_aug.append(y)
            
            # 노이즈 추가 (데이터 증강)
            noise_level = 0.03
            X_noisy = X + np.random.normal(0, noise_level, X.shape)
            X_aug.append(X_noisy)
            y_aug.append(y)
            
            # 시간 축 왜곡 (데이터 증강)
            X_warped = np.zeros_like(X)
            for i in range(len(X)):
                for c in range(X.shape[2]):  # 각 채널에 대해
                    x = X[i, :, c]
                    # 시간 왜곡 효과 (랜덤하게 늘이거나 줄임)
                    f = np.linspace(0, 1, len(x))
                    f_warped = f**np.random.uniform(0.7, 1.3)
                    f_warped = f_warped / f_warped.max()
                    x_warped = np.interp(f, f_warped, x)
                    X_warped[i, :, c] = x_warped
            X_aug.append(X_warped)
            y_aug.append(y)
            
            # 진폭 변화 추가
            X_scaled = np.zeros_like(X)
            for i in range(len(X)):
                scale_factor = np.random.uniform(0.8, 1.2)
                X_scaled[i] = X[i] * scale_factor
            X_aug.append(X_scaled)
            y_aug.append(y)
            
            # 증강된 데이터 결합
            X = np.vstack(X_aug)
            y = np.concatenate(y_aug)
            
            self.update_progress.emit(20, f"증강 후 데이터 크기: {X.shape}")
            
            # 훈련/검증/테스트 데이터 분할
            self.update_progress.emit(25, "데이터 분할 중...")
            X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42, stratify=y_train_val)
            
            self.update_progress.emit(30, f"훈련 데이터: {X_train.shape}, 검증 데이터: {X_val.shape}, 테스트 데이터: {X_test.shape}")
            
            # 데이터 정규화
            self.update_progress.emit(35, "데이터 정규화 중...")
            scaler = StandardScaler()
            # 배치와 시간 차원을 결합하여 각 특성을 독립적으로 정규화
            X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
            scaler.fit(X_train_reshaped)
            
            # 정규화 적용
            for dataset in [X_train, X_val, X_test]:
                shape = dataset.shape
                dataset_reshaped = dataset.reshape(-1, shape[-1])
                dataset_normalized = scaler.transform(dataset_reshaped)
                dataset[:] = dataset_normalized.reshape(shape)
            
            # 데이터셋 및 데이터 로더 생성
            class EMGDataset(Dataset):
                def __init__(self, X, y):
                    self.X = torch.tensor(X, dtype=torch.float32)
                    self.y = torch.tensor(y, dtype=torch.long)
                
                def __len__(self):
                    return len(self.y)
                
                def __getitem__(self, idx):
                    return self.X[idx], self.y[idx]
            
            train_dataset = EMGDataset(X_train, y_train)
            val_dataset = EMGDataset(X_val, y_val)
            test_dataset = EMGDataset(X_test, y_test)
            
            train_loader = DataLoader(train_dataset, batch_size=self.params['batch_size'], shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=self.params['batch_size'])
            test_loader = DataLoader(test_dataset, batch_size=self.params['batch_size'])
            
            # 모델 생성
            self.update_progress.emit(40, "모델 초기화 중...")
            model = EMGTransformer(
                input_dim=self.params['input_dim'],
                d_model=self.params['d_model'],
                nhead=self.params['nhead'],
                num_layers=self.params['num_layers'],
                dim_feedforward=self.params['dim_feedforward'],
                num_classes=self.params['num_classes'],
                dropout=self.params['dropout']
            ).to(self.device)
            
            # 손실 함수 및 옵티마이저 설정
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(
                model.parameters(), 
                lr=self.params['learning_rate'], 
                weight_decay=0.0001
            )
            
            # 학습률 스케줄러 추가
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=5, verbose=True
            )
            
            # 모델 훈련
            self.update_progress.emit(45, "모델 훈련 시작...")
            
            best_val_loss = float('inf')
            patience_counter = 0
            patience = 10
            
            self.train_losses = []
            self.val_losses = []
            self.train_accs = []
            self.val_accs = []
            
            for epoch in range(self.params['num_epochs']):
                # 훈련
                train_loss, train_acc = self.train_epoch(model, train_loader, optimizer, criterion)
                self.train_losses.append(train_loss)
                self.train_accs.append(train_acc)
                
                # 검증
                val_loss, val_acc = self.validate(model, val_loader, criterion)
                self.val_losses.append(val_loss)
                self.val_accs.append(val_acc)
                
                # 학습률 스케줄러 업데이트
                scheduler.step(val_loss)
                
                # 진행률 업데이트
                progress = 45 + (45 * (epoch + 1) / self.params['num_epochs'])
                progress_msg = f"에폭 {epoch+1}/{self.params['num_epochs']} - 훈련 손실: {train_loss:.4f}, 훈련 정확도: {train_acc:.4f}, 검증 손실: {val_loss:.4f}, 검증 정확도: {val_acc:.4f}"
                self.update_progress.emit(int(progress), progress_msg)
                
                # 그래프 업데이트
                self.update_plot.emit(
                    self.train_losses, 
                    self.val_losses, 
                    self.train_accs, 
                    self.val_accs
                )
                
                # 조기 종료 검사
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # 최고 성능 모델 저장
                    torch.save(model.state_dict(), 'best_emg_transformer.pth')
                    self.update_progress.emit(int(progress), f"{progress_msg}\n최고 성능 모델 저장됨: best_emg_transformer.pth")
                else:
                    patience_counter += 1
                    
                if patience_counter >= patience:
                    self.update_progress.emit(int(progress), f"{progress_msg}\n에폭 {epoch+1}에서 조기 종료")
                    break
            
            # 최고 성능 모델 로드
            model.load_state_dict(torch.load('best_emg_transformer.pth'))
            
            # 모델 평가
            self.update_progress.emit(90, "테스트 데이터에서 모델 평가 중...")
            model.eval()
            correct = 0
            total = 0
            
            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs, 1)
                    
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            
            test_accuracy = correct / total
            self.update_progress.emit(95, f"테스트 정확도: {test_accuracy:.4f}")
            
            # 모델 저장
            self.update_progress.emit(98, "모델 저장 중...")
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler': scaler,
                'config': self.params
            }, MODEL_FILE)
            
            self.update_progress.emit(100, f"완료! 모델이 {MODEL_FILE}에 저장되었습니다.")
            self.finished.emit(True, f"모델 학습 완료! 테스트 정확도: {test_accuracy:.4f}")
            
        except Exception as e:
            self.finished.emit(False, f"학습 중 오류 발생: {str(e)}")

class ModelLearningWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        
        # 학습 스레드
        self.worker = None
        self.worker_thread = None
        
        # UI 초기화
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout()
        
        # 모델 매개변수 그룹
        params_group = QGroupBox("모델 매개변수")
        params_layout = QGridLayout()
        
        # 드롭다운으로 매개변수 템플릿 선택 (기본, 작은 모델, 큰 모델 등)
        self.template_label = QLabel("모델 템플릿:")
        self.template_combo = QComboBox()
        self.template_combo.addItems(["기본 모델", "경량 모델", "고성능 모델"])
        self.template_combo.currentIndexChanged.connect(self.load_template)
        
        params_layout.addWidget(self.template_label, 0, 0)
        params_layout.addWidget(self.template_combo, 0, 1)
        
        # 주요 매개변수 설정
        self.d_model_label = QLabel("d_model:")
        self.d_model_spin = QSpinBox()
        self.d_model_spin.setRange(16, 512)
        self.d_model_spin.setSingleStep(16)
        self.d_model_spin.setValue(128)
        
        self.nhead_label = QLabel("nhead:")
        self.nhead_spin = QSpinBox()
        self.nhead_spin.setRange(1, 16)
        self.nhead_spin.setValue(8)
        
        self.num_layers_label = QLabel("num_layers:")
        self.num_layers_spin = QSpinBox()
        self.num_layers_spin.setRange(1, 12)
        self.num_layers_spin.setValue(3)
        
        self.dim_feedforward_label = QLabel("dim_feedforward:")
        self.dim_feedforward_spin = QSpinBox()
        self.dim_feedforward_spin.setRange(32, 1024)
        self.dim_feedforward_spin.setSingleStep(32)
        self.dim_feedforward_spin.setValue(256)
        
        self.learning_rate_label = QLabel("learning_rate:")
        self.learning_rate_spin = QDoubleSpinBox()
        self.learning_rate_spin.setRange(0.00001, 0.01)
        self.learning_rate_spin.setSingleStep(0.0001)
        self.learning_rate_spin.setDecimals(6)
        self.learning_rate_spin.setValue(0.0003)
        
        self.batch_size_label = QLabel("batch_size:")
        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(1, 128)
        self.batch_size_spin.setValue(8)
        
        self.epochs_label = QLabel("num_epochs:")
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(10, 500)
        self.epochs_spin.setValue(100)
        
        params_layout.addWidget(self.d_model_label, 1, 0)
        params_layout.addWidget(self.d_model_spin, 1, 1)
        params_layout.addWidget(self.nhead_label, 2, 0)
        params_layout.addWidget(self.nhead_spin, 2, 1)
        params_layout.addWidget(self.num_layers_label, 3, 0)
        params_layout.addWidget(self.num_layers_spin, 3, 1)
        params_layout.addWidget(self.dim_feedforward_label, 4, 0)
        params_layout.addWidget(self.dim_feedforward_spin, 4, 1)
        params_layout.addWidget(self.learning_rate_label, 5, 0)
        params_layout.addWidget(self.learning_rate_spin, 5, 1)
        params_layout.addWidget(self.batch_size_label, 6, 0)
        params_layout.addWidget(self.batch_size_spin, 6, 1)
        params_layout.addWidget(self.epochs_label, 7, 0)
        params_layout.addWidget(self.epochs_spin, 7, 1)
        
        params_group.setLayout(params_layout)
        
        # 데이터 증강 옵션 그룹
        augment_group = QGroupBox("데이터 증강 옵션")
        augment_layout = QVBoxLayout()
        
        self.noise_check = QCheckBox("노이즈 추가")
        self.noise_check.setChecked(True)
        self.warp_check = QCheckBox("시간 축 왜곡")
        self.warp_check.setChecked(True)
        self.scale_check = QCheckBox("진폭 변화")
        self.scale_check.setChecked(True)
        
        augment_layout.addWidget(self.noise_check)
        augment_layout.addWidget(self.warp_check)
        augment_layout.addWidget(self.scale_check)
        
        augment_group.setLayout(augment_layout)
        
        # 학습 진행 상황
        progress_group = QGroupBox("학습 진행 상황")
        progress_layout = QVBoxLayout()
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        
        progress_layout.addWidget(self.progress_bar)
        progress_layout.addWidget(self.log_text)
        
        progress_group.setLayout(progress_layout)
        
        # 학습 곡선 그래프
        plot_group = QGroupBox("학습 곡선")
        plot_layout = QVBoxLayout()

        # PyQtGraph 사용하여 두 개의 그래프 생성
        self.plot_widget = pg.GraphicsLayoutWidget()

        # 첫 번째 하위 플롯 (손실)
        self.loss_plot = self.plot_widget.addPlot(row=0, col=0)
        self.loss_plot.setTitle("손실", color='w')
        self.loss_plot.setLabel('left', '손실', color='w')
        self.loss_plot.setLabel('bottom', '에폭', color='w')
        self.loss_plot.addLegend()
        self.loss_plot.showGrid(x=True, y=True, alpha=0.3)
        self.train_loss_curve = self.loss_plot.plot(pen=pg.mkPen('b', width=2), name="훈련 손실")
        self.val_loss_curve = self.loss_plot.plot(pen=pg.mkPen('r', width=2), name="검증 손실")

        # 두 번째 하위 플롯 (정확도)
        self.acc_plot = self.plot_widget.addPlot(row=0, col=1)
        self.acc_plot.setTitle("정확도", color='w')
        self.acc_plot.setLabel('left', '정확도', color='w')
        self.acc_plot.setLabel('bottom', '에폭', color='w')
        self.acc_plot.addLegend()
        self.acc_plot.showGrid(x=True, y=True, alpha=0.3)
        self.train_acc_curve = self.acc_plot.plot(pen=pg.mkPen('b', width=2), name="훈련 정확도")
        self.val_acc_curve = self.acc_plot.plot(pen=pg.mkPen('r', width=2), name="검증 정확도")

        plot_layout.addWidget(self.plot_widget)
        plot_group.setLayout(plot_layout)
        
        # 학습 제어 버튼
        control_layout = QHBoxLayout()
        
        self.start_btn = QPushButton("학습 시작")
        self.start_btn.clicked.connect(self.start_training)
        
        self.stop_btn = QPushButton("학습 중단")
        self.stop_btn.clicked.connect(self.stop_training)
        self.stop_btn.setEnabled(False)
        
        control_layout.addWidget(self.start_btn)
        control_layout.addWidget(self.stop_btn)
        
        # 레이아웃에 추가
        layout.addWidget(params_group)
        layout.addWidget(augment_group)
        layout.addWidget(progress_group)
        layout.addWidget(plot_group)
        layout.addLayout(control_layout)
        
        self.setLayout(layout)
        
        # 로그에 초기 메시지 추가
        self.add_log("모델 학습을 시작하려면 '학습 시작' 버튼을 누르세요.")
        self.add_log(f"데이터 폴더: {DATA_FOLDER}")
    
    def load_template(self, index):
        """모델 템플릿 변경"""
        if index == 0:  # 기본 모델
            self.d_model_spin.setValue(128)
            self.nhead_spin.setValue(8)
            self.num_layers_spin.setValue(3)
            self.dim_feedforward_spin.setValue(256)
            self.learning_rate_spin.setValue(0.0003)
            self.batch_size_spin.setValue(8)
            self.epochs_spin.setValue(100)
        elif index == 1:  # 경량 모델
            self.d_model_spin.setValue(64)
            self.nhead_spin.setValue(4)
            self.num_layers_spin.setValue(2)
            self.dim_feedforward_spin.setValue(128)
            self.learning_rate_spin.setValue(0.0005)
            self.batch_size_spin.setValue(16)
            self.epochs_spin.setValue(50)
        elif index == 2:  # 고성능 모델
            self.d_model_spin.setValue(256)
            self.nhead_spin.setValue(8)
            self.num_layers_spin.setValue(4)
            self.dim_feedforward_spin.setValue(512)
            self.learning_rate_spin.setValue(0.0001)
            self.batch_size_spin.setValue(8)
            self.epochs_spin.setValue(150)
    
    def get_model_params(self):
        """UI에서 모델 매개변수 가져오기"""
        return {
            'd_model': self.d_model_spin.value(),
            'nhead': self.nhead_spin.value(),
            'num_layers': self.num_layers_spin.value(),
            'dim_feedforward': self.dim_feedforward_spin.value(),
            'learning_rate': self.learning_rate_spin.value(),
            'batch_size': self.batch_size_spin.value(),
            'num_epochs': self.epochs_spin.value()
        }
    
    def add_log(self, message):
        """로그 메시지 추가"""
        timestamp = datetime.now().strftime('%H:%M:%S')
        self.log_text.append(f"[{timestamp}] {message}")
        # 스크롤을 항상 최하단으로
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    def update_progress(self, value, message):
        """진행률 업데이트"""
        self.progress_bar.setValue(value)
        if message:
            self.add_log(message)
        
        # 상위 클래스에 타이틀 업데이트
        if self.parent:
            progress_str = f"{value}%" if value < 100 else "완료"
            self.parent.setWindowTitle(f"EMG 동작 인식 시스템 - 모델 학습 - {progress_str}")
    
    def update_plot(self, train_losses, val_losses, train_accs, val_accs):
        """학습 곡선 그래프 업데이트"""
        # 데이터 준비
        epochs = list(range(1, len(train_losses) + 1))
    
        # 손실 그래프 업데이트
        self.train_loss_curve.setData(epochs, train_losses)
        self.val_loss_curve.setData(epochs, val_losses)
    
        # 정확도 그래프 업데이트
        self.train_acc_curve.setData(epochs, train_accs)
        self.val_acc_curve.setData(epochs, val_accs)
    
    def start_training(self):
        """학습 시작"""
        # 데이터 폴더 확인
        if not os.path.exists(DATA_FOLDER):
            self.add_log(f"오류: 데이터 폴더 {DATA_FOLDER}가 존재하지 않습니다.")
            return
        
        files = [f for f in os.listdir(DATA_FOLDER) if f.endswith('.csv')]
        if not files:
            self.add_log(f"오류: {DATA_FOLDER}에 CSV 파일이 없습니다. 먼저 데이터를 수집하세요.")
            return
        
        # 모델 매개변수 가져오기
        model_params = self.get_model_params()
        
        # 데이터 증강 옵션 추가
        model_params['use_noise'] = self.noise_check.isChecked()
        model_params['use_warp'] = self.warp_check.isChecked()
        model_params['use_scale'] = self.scale_check.isChecked()
        
        # 프로그레스 바 및 UI 상태 업데이트
        self.progress_bar.setValue(0)
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        
        # 워커 스레드 생성 및 시작
        self.worker = TrainingWorker(DATA_FOLDER, model_params)
        self.worker_thread = QThread()
        self.worker.moveToThread(self.worker_thread)
        
        # 시그널 연결
        self.worker_thread.started.connect(self.worker.run)
        self.worker.update_progress.connect(self.update_progress)
        self.worker.update_plot.connect(self.update_plot)
        self.worker.finished.connect(self.training_finished)
        self.worker.finished.connect(self.worker_thread.quit)
        
        # 스레드 시작
        self.worker_thread.start()
        
        self.add_log("학습을 시작합니다...")
    
    def stop_training(self):
        """학습 중단"""
        if self.worker_thread and self.worker_thread.isRunning():
            # Worker 스레드 종료 요청
            self.worker_thread.quit()
            self.worker_thread.wait(3000)  # 최대 3초 대기
            
            if self.worker_thread.isRunning():
                self.worker_thread.terminate()  # 강제 종료 (추천하지 않음)
                self.add_log("학습이 강제 종료되었습니다.")
            else:
                self.add_log("학습이 중단되었습니다.")
            
            # UI 상태 업데이트
            self.start_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
    
    def training_finished(self, success, message):
        """학습 완료 처리"""
        if success:
            # 학습 성공
            QMessageBox.information(self, "학습 완료", message)
            self.update_progress(100, message)
        else:
            # 학습 실패
            QMessageBox.warning(self, "학습 오류", message)
            self.update_progress(0, message)
        
        # UI 상태 업데이트
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
    
    def closeEvent(self, event):
        """위젯 종료 처리"""
        self.stop_training()
        event.accept()

# ------------------------------------------------------------------------
# 모델 테스트 위젯
# ------------------------------------------------------------------------
class EMGTestWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.serial_thread = None
        
        # 모델 및 상태 변수
        self.model = None
        self.scaler = None
        self.config = None
        self.data_buffer = deque(maxlen=BUFFER_SIZE)
        self.sequence_buffer = []
        self.is_recording = False
        self.cooldown = False
        self.cooldown_start = 0
        self.baseline_mean = (0, 0)
        self.baseline_std = (1, 1)
        self.post_capture_count = 0
        self.current_values = (0, 0)
        self.sensitivity_level = 1.0
        self.label_names = load_label_names()
        self.last_prediction = None
        self.last_confidence = 0
        self.detection_blocked = False
        self.detection_block_time = 0
        self.detection_block_duration = 0
        
        # UI 초기화
        self.init_ui()
        
        # 타이머 설정 (상태 업데이트용)
        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self.update_status_display)
        self.update_timer.start(300)  # 300ms마다 업데이트
    
    def init_ui(self):
        layout = QVBoxLayout()
        
        # 모델 로드 그룹
        model_group = QGroupBox("모델 로드")
        model_layout = QHBoxLayout()
        
        self.model_label = QLabel("모델 파일:")
        self.model_path = QLineEdit(MODEL_FILE)
        self.model_path.setReadOnly(True)
        
        self.load_model_btn = QPushButton("모델 로드")
        self.load_model_btn.clicked.connect(self.load_model)
        
        model_layout.addWidget(self.model_label)
        model_layout.addWidget(self.model_path)
        model_layout.addWidget(self.load_model_btn)
        
        model_group.setLayout(model_layout)
        
        # 연결 설정 그룹
        connection_group = QGroupBox("연결 설정")
        connection_layout = QHBoxLayout()
        
        self.port_label = QLabel("시리얼 포트:")
        self.port_combo = QComboBox()
        self.refresh_ports()
        self.port_combo.setEditable(True)  # 직접 입력 가능
        
        self.connect_btn = QPushButton("연결")
        self.connect_btn.clicked.connect(self.connect_serial)
        self.connect_btn.setEnabled(False)  # 모델 로드 후 활성화
        
        self.disconnect_btn = QPushButton("연결 해제")
        self.disconnect_btn.clicked.connect(self.disconnect_serial)
        self.disconnect_btn.setEnabled(False)
        
        connection_layout.addWidget(self.port_label)
        connection_layout.addWidget(self.port_combo)
        connection_layout.addWidget(self.connect_btn)
        connection_layout.addWidget(self.disconnect_btn)
        
        connection_group.setLayout(connection_layout)
        
        # 감도 설정
        sensitivity_group = QGroupBox("감도 설정")
        sensitivity_layout = QHBoxLayout()
        
        self.sensitivity_label = QLabel("감도 수준:")
        self.sensitivity_spin = QDoubleSpinBox()
        self.sensitivity_spin.setRange(0.5, 2.0)
        self.sensitivity_spin.setSingleStep(0.1)
        self.sensitivity_spin.setValue(self.sensitivity_level)
        self.sensitivity_spin.valueChanged.connect(self.update_sensitivity)
        
        sensitivity_layout.addWidget(self.sensitivity_label)
        sensitivity_layout.addWidget(self.sensitivity_spin)
        
        sensitivity_group.setLayout(sensitivity_layout)
        
        # 감지 결과 표시 그룹
        result_group = QGroupBox("감지 결과")
        result_layout = QVBoxLayout()
        
        self.result_label = QLabel("인식 대기 중...")
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setStyleSheet("font-size: 24pt; font-weight: bold;")
        
        self.confidence_label = QLabel("신뢰도: -")
        self.confidence_label.setAlignment(Qt.AlignCenter)
        self.confidence_label.setStyleSheet("font-size: 14pt;")
        
        result_layout.addWidget(self.result_label)
        result_layout.addWidget(self.confidence_label)
        
        result_group.setLayout(result_layout)
        
        # 데이터 시각화 그룹
        visual_group = QGroupBox("실시간 EMG 데이터")
        visual_layout = QVBoxLayout()

        # 그래프 표시 여부 체크박스 추가
        self.show_graph_check = QCheckBox("실시간 그래프 표시")
        self.show_graph_check.setChecked(True)
        visual_layout.addWidget(self.show_graph_check)

        # matplotlib 대신 pyqtgraph 사용
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('k')  # 검은색 배경
        self.plot_widget.setTitle("실시간 EMG 신호", color='w')
        self.plot_widget.setLabel('left', 'EMG 값', color='w')
        self.plot_widget.setLabel('bottom', '시간', color='w')
        self.plot_widget.setYRange(-20, 20)  # y축 범위 설정 (이 부분을 원하는 값으로 수정 가능)
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)

        # 두 센서를 위한 그래프 선 생성
        self.curve1 = self.plot_widget.plot(pen=pg.mkPen('r', width=2), name="센서 1")
        self.curve2 = self.plot_widget.plot(pen=pg.mkPen('b', width=2), name="센서 2")

        # 플롯 데이터
        self.plot_data1 = []
        self.plot_data2 = []
        self.max_plot_points = 200  # 그래프에 표시할 최대 포인트 수

        visual_layout.addWidget(self.plot_widget)
        visual_group.setLayout(visual_layout)
        
        # 로그 및 상태 그룹
        log_group = QGroupBox("로그 및 상태")
        log_layout = QVBoxLayout()
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(100)
        
        log_layout.addWidget(self.log_text)
        
        log_group.setLayout(log_layout)
        
        # 메인 레이아웃에 추가
        layout.addWidget(model_group)
        layout.addWidget(connection_group)
        layout.addWidget(sensitivity_group)
        layout.addWidget(result_group)
        layout.addWidget(visual_group)
        layout.addWidget(log_group)
        
        self.setLayout(layout)
        
        # 초기 로그 메시지
        self.add_log("모델을 로드한 후 연결하세요.")
    
    def refresh_ports(self):
        """사용 가능한 시리얼 포트 목록 갱신"""
        self.port_combo.clear()
        try:
            import serial.tools.list_ports
            ports = serial.tools.list_ports.comports()
            for port in ports:
                self.port_combo.addItem(port.device)
            
            # 기본 포트 추가
            if self.port_combo.findText(DEFAULT_SERIAL_PORT) == -1:
                self.port_combo.addItem(DEFAULT_SERIAL_PORT)
            
            self.port_combo.setCurrentText(DEFAULT_SERIAL_PORT)
        except:
            self.port_combo.addItem(DEFAULT_SERIAL_PORT)
    
    def load_model(self):
        """모델 로드"""
        try:
            model_file = self.model_path.text()
            if not os.path.exists(model_file):
                self.add_log(f"오류: {model_file} 파일이 존재하지 않습니다.")
                return
            
            self.add_log(f"모델 로드 중: {model_file}...")
            
            checkpoint = torch.load(model_file, map_location=torch.device('cpu'))
            self.config = checkpoint['config']
            
            # 모델 생성
            self.model = EMGTransformer(
                input_dim=self.config['input_dim'],
                d_model=self.config['d_model'],
                nhead=self.config['nhead'],
                num_layers=self.config['num_layers'],
                dim_feedforward=self.config['dim_feedforward'],
                num_classes=self.config['num_classes'],
                dropout=0
            )
            
            # 모델 가중치 로드
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            
            # 스케일러 로드
            self.scaler = checkpoint['scaler']
            
            self.add_log("모델 로드 완료!")
            self.add_log(f"모델 구성: d_model={self.config['d_model']}, nhead={self.config['nhead']}, layers={self.config['num_layers']}")
            
            # 연결 버튼 활성화
            self.connect_btn.setEnabled(True)
            
            # 사용 가능한 라벨 표시
            self.add_log("인식 가능한 동작 목록:")
            for label_id, label_name in sorted(self.label_names.items()):
                self.add_log(f"  {label_id}: {label_name}")
            
            return True
        except Exception as e:
            self.add_log(f"모델 로드 오류: {str(e)}")
            return False
    
    def connect_serial(self):
        """시리얼 포트 연결 및 데이터 수집 시작"""
        port = self.port_combo.currentText()
        
        # 시리얼 스레드 생성 및 시작
        self.serial_thread = SerialReaderThread(port=port)
        self.serial_thread.data_received.connect(self.process_data)
        self.serial_thread.error_occurred.connect(self.handle_error)
        self.serial_thread.connection_status.connect(self.handle_connection_status)
        self.serial_thread.start()
        
        # UI 상태 업데이트
        self.connect_btn.setEnabled(False)
        self.port_combo.setEnabled(False)
    
    def disconnect_serial(self):
        """시리얼 포트 연결 해제 및 데이터 수집 중단"""
        if self.serial_thread:
            self.serial_thread.stop()
            self.serial_thread.wait()  # 스레드가 종료될 때까지 대기
            self.serial_thread = None
        
        # UI 상태 업데이트
        self.connect_btn.setEnabled(True)
        self.port_combo.setEnabled(True)
        self.disconnect_btn.setEnabled(False)
        
        # 결과 초기화
        self.result_label.setText("인식 대기 중...")
        self.confidence_label.setText("신뢰도: -")
        
        self.add_log("연결이 해제되었습니다.")
    
    def handle_connection_status(self, success, message):
        """연결 상태 처리"""
        self.add_log(message)
        if success:
            self.disconnect_btn.setEnabled(True)
            self.add_log("센서 보정 중...")
        else:
            self.connect_btn.setEnabled(True)
            self.port_combo.setEnabled(True)
    
    def handle_error(self, error_message):
        """오류 처리"""
        self.add_log(f"오류: {error_message}")
        self.disconnect_serial()
    
    def update_sensitivity(self, value):
        """감도 수준 업데이트"""
        self.sensitivity_level = value
        self.add_log(f"감도 수준이 {value:.1f}로 설정되었습니다.")
    
    def process_data(self, s1, s2):
        """수신된 EMG 데이터 처리"""
        # 현재 값 저장
        self.current_values = (s1, s2)
        
        # 그래프 업데이트를 위한 데이터 추가
        self.plot_data1.append(s1)
        self.plot_data2.append(s2)
        
        # 최대 포인트 수 제한
        if len(self.plot_data1) > self.max_plot_points:
            self.plot_data1.pop(0)
            self.plot_data2.pop(0)
        
        # 버퍼에 추가
        self.data_buffer.append((s1, s2))
        
        # 주기적으로 기준선 업데이트 (적응형)
        if not self.is_recording and len(self.data_buffer) % 50 == 0:
            self.baseline_mean, self.baseline_std = self.calculate_baseline()
        
        # 대기 시간 확인
        if self.cooldown:
            if time.time() - self.cooldown_start >= COOLDOWN_PERIOD:
                self.cooldown = False
                self.add_log("대기 시간 종료. 동작 감지 준비 완료.")
                # 데이터 버퍼의 최근 부분을 중립값으로 설정하여 이전 동작의 영향 감소
                # 1. 데이터 버퍼 완전히 비우기
                self.data_buffer.clear()
        
                # 2. 잠시 동작 감지 블로킹 설정
                self.detection_blocked = True
                self.detection_block_time = time.time()
                self.detection_block_duration = 1.0  # 1초 동안 감지 차단
        
                # 로그 추가
                self.add_log("버퍼 초기화 및 감지 일시 중지")

        # 동작 감지
        elif not self.is_recording and self.detect_movement():
            self.is_recording = True
            self.sequence_buffer = list(self.data_buffer)[-30:]  # 감지 전 30프레임 포함
            self.post_capture_count = 0
            self.add_log("동작 감지됨! 기록 중...")
        
        # 동작 기록 중
        elif self.is_recording:
            # 값 추가
            self.sequence_buffer.append((s1, s2))
            
            # 동작 종료 확인
            if not self.is_movement_continuing():
                self.post_capture_count += 1
                if self.post_capture_count >= POST_CAPTURE:
                    # 동작 기록 종료 및 예측
                    self.add_log(f"동작 기록 완료 (길이: {len(self.sequence_buffer)})")
                    
                    if len(self.sequence_buffer) >= 20:  # 최소 길이 확인
                        prediction, confidence = self.predict_movement()
                        if prediction is not None:
                            self.show_result(prediction, confidence)
                    else:
                        self.add_log("동작이 너무 짧아서 무시합니다.")
                    
                    # 상태 업데이트
                    self.is_recording = False
                    self.cooldown = True
                    self.cooldown_start = time.time()
                    self.sequence_buffer = []
                    self.add_log(f"대기 시간 {COOLDOWN_PERIOD}초 시작...")
            else:
                self.post_capture_count = 0
            
            # 최대 길이 확인
            if len(self.sequence_buffer) >= self.config['sequence_length'] * 1.5:
                self.add_log("최대 길이 도달. 동작 기록 종료.")
                prediction, confidence = self.predict_movement()
                if prediction is not None:
                    self.show_result(prediction, confidence)
                
                # 상태 업데이트
                self.is_recording = False
                self.cooldown = True
                self.cooldown_start = time.time()
                self.sequence_buffer = []
                self.add_log(f"대기 시간 {COOLDOWN_PERIOD}초 시작...")
    
    def show_result(self, prediction, confidence):
        """예측 결과 표시"""
        self.last_prediction = prediction
        self.last_confidence = confidence
        
        # 라벨 이름 가져오기
        label_name = self.label_names.get(prediction, f"알 수 없음({prediction})")
        
        # 결과 표시 업데이트
        self.result_label.setText(f"{label_name}")
        self.confidence_label.setText(f"신뢰도: {confidence:.2f}")
        
        # 신뢰도에 따른 스타일 변경
        if confidence >= 0.8:
            self.result_label.setStyleSheet("font-size: 24pt; font-weight: bold; color: green;")
        elif confidence >= 0.5:
            self.result_label.setStyleSheet("font-size: 24pt; font-weight: bold; color: orange;")
        else:
            self.result_label.setStyleSheet("font-size: 24pt; font-weight: bold; color: red;")
        
        self.add_log(f"감지된 동작: {label_name} (신뢰도: {confidence:.2f})")
    
    def update_plot(self):
        """그래프 업데이트"""
        # 그래프 비활성화 체크 추가
        if not self.show_graph_check.isChecked():
            return
        
        if self.plot_data1 and self.plot_data2:
            # pyqtgraph 방식으로 데이터 업데이트
            self.curve1.setData(self.plot_data1)
            self.curve2.setData(self.plot_data2)
    
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
    
    def detect_movement(self):
        """여러 방법을 통합한 동작 감지"""
        # 감지 차단 확인
        if self.detection_blocked:
            if time.time() - self.detection_block_time >= self.detection_block_duration:
                self.detection_blocked = False
                self.add_log("동작 감지 재개")
            else:
                return False
            if len(self.data_buffer) < DETECTION_WINDOW:
                return False
            
        # 감지에 사용할 최근 데이터 가져오기
        recent_data = list(self.data_buffer)[-DETECTION_WINDOW:]
        
        # 1. Z-점수 기반 감지 (기존 방식)
        z_scores_s1 = [(x[0] - self.baseline_mean[0]) / self.baseline_std[0] for x in recent_data]
        z_scores_s2 = [(x[1] - self.baseline_mean[1]) / self.baseline_std[1] for x in recent_data]
        
        # 유의미한 편차 개수 계산 (감도 조절 적용)
        adjusted_threshold = DETECTION_THRESHOLD / self.sensitivity_level
        count_s1 = sum(1 for z in z_scores_s1 if abs(z) > adjusted_threshold)
        count_s2 = sum(1 for z in z_scores_s2 if abs(z) > adjusted_threshold)
        
        # 2. 추세 기반 감지 (새로운 방식)
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
        
        variance_detected = (s1_var > (self.baseline_std[0] * DISPERSION_THRESHOLD * self.sensitivity_level) or 
                            s2_var > (self.baseline_std[1] * DISPERSION_THRESHOLD * self.sensitivity_level))
        
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
        adjusted_threshold = (DETECTION_THRESHOLD/2) / self.sensitivity_level
        z_scores_s1 = [(x[0] - self.baseline_mean[0]) / self.baseline_std[0] for x in recent_data]
        z_scores_s2 = [(x[1] - self.baseline_mean[1]) / self.baseline_std[1] for x in recent_data]
        
        # 두 가지 조건 중 하나라도 만족하면 동작 계속으로 판단
        # 1. Z-점수 기반 (기존 방식)
        for z1, z2 in zip(z_scores_s1, z_scores_s2):
            if abs(z1) > adjusted_threshold or abs(z2) > adjusted_threshold:
                return True
                
        # 2. 추세 또는 분산 기반 (새로운 방식)
        if len(recent_data) >= 5:
            s1_values = [x[0] for x in recent_data]
            s2_values = [x[1] for x in recent_data]
            
            # 분산 확인
            s1_var = np.var(s1_values)
            s2_var = np.var(s2_values)
            
            if (s1_var > (self.baseline_std[0] * DISPERSION_THRESHOLD * self.sensitivity_level) or 
                s2_var > (self.baseline_std[1] * DISPERSION_THRESHOLD * self.sensitivity_level)):
                return True
        
        return False
    
    def predict_movement(self):
        """동작 감지 후 라벨 예측"""
        if self.model is None:
            self.add_log("오류: 모델이 로드되지 않았습니다.")
            return None, 0
        
        # 데이터 준비
        sequence = self.sequence_buffer.copy()
        
        # 패딩 처리
        sequence_length = self.config['sequence_length']
        if len(sequence) < sequence_length:
            padding = [(0, 0)] * (sequence_length - len(sequence))
            sequence += padding
        elif len(sequence) > sequence_length:
            sequence = sequence[:sequence_length]
        
        # 예측용 배열 생성
        X = np.array(sequence).reshape(1, sequence_length, 2)
        
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
            self.add_log(f"예측 중 오류 발생: {str(e)}")
            return None, 0
    
    def add_log(self, message):
        """로그 메시지 추가"""
        timestamp = datetime.now().strftime('%H:%M:%S')
        self.log_text.append(f"[{timestamp}] {message}")
        # 스크롤을 항상 최하단으로
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    def update_status_display(self):
        """상태 표시 업데이트 (타이머에 의해 주기적으로 호출)"""
        # 그래프 업데이트
        self.update_plot()
        
        # 현재 상태에 따른 메시지 표시
        status_msg = ""
        if not self.serial_thread or not self.serial_thread.running:
            return
        
        if self.cooldown:
            elapsed = time.time() - self.cooldown_start
            status_msg = f"대기 시간: {COOLDOWN_PERIOD - elapsed:.1f}초 남음"
        elif self.is_recording:
            status_msg = f"동작 기록 중... ({len(self.sequence_buffer)} 프레임)"
        else:
            status_msg = "동작 감지 대기 중..."
        
        # 윈도우 타이틀 업데이트 (상태 정보 표시)
        if self.parent:
            self.parent.setWindowTitle(f"EMG 동작 인식 시스템 - 테스트 중 - {status_msg}")
    
    def closeEvent(self, event):
        """위젯 종료 처리"""
        if self.serial_thread:
            self.serial_thread.stop()
            self.serial_thread.wait()
        event.accept()

# ------------------------------------------------------------------------
# 라벨 관리 위젯
# ------------------------------------------------------------------------
class LabelManagerWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        
        # 라벨 데이터
        self.label_names = load_label_names()
        
        # UI 초기화
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout()
        
        # 라벨 목록 그룹
        list_group = QGroupBox("라벨 목록")
        list_layout = QVBoxLayout()
        
        self.label_table = QTableWidget()
        self.label_table.setColumnCount(2)
        self.label_table.setHorizontalHeaderLabels(["ID", "이름"])
        self.label_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.label_table.setSelectionMode(QTableWidget.SingleSelection)
        self.label_table.setEditTriggers(QTableWidget.NoEditTriggers)
        
        # 테이블 열 너비 설정
        self.label_table.setColumnWidth(0, 50)
        self.label_table.setColumnWidth(1, 200)
        
        list_layout.addWidget(self.label_table)
        list_group.setLayout(list_layout)
        
        # 라벨 편집 그룹
        edit_group = QGroupBox("라벨 편집")
        edit_layout = QGridLayout()
        
        self.id_label = QLabel("ID:")
        self.id_spin = QSpinBox()
        self.id_spin.setRange(0, 99)
        
        self.name_label = QLabel("이름:")
        self.name_edit = QLineEdit()
        
        edit_layout.addWidget(self.id_label, 0, 0)
        edit_layout.addWidget(self.id_spin, 0, 1)
        edit_layout.addWidget(self.name_label, 1, 0)
        edit_layout.addWidget(self.name_edit, 1, 1)
        
        edit_group.setLayout(edit_layout)
        
        # 작업 버튼 그룹
        button_layout = QHBoxLayout()
        
        self.add_btn = QPushButton("추가/수정")
        self.add_btn.clicked.connect(self.add_or_update_label)
        
        self.delete_btn = QPushButton("삭제")
        self.delete_btn.clicked.connect(self.delete_label)
        self.delete_btn.setEnabled(False)
        
        self.save_btn = QPushButton("저장")
        self.save_btn.clicked.connect(self.save_labels)
        
        button_layout.addWidget(self.add_btn)
        button_layout.addWidget(self.delete_btn)
        button_layout.addWidget(self.save_btn)
        
        # 상태 메시지
        self.status_label = QLabel("라벨을 추가하거나 수정하세요.")
        
        # 메인 레이아웃에 추가
        layout.addWidget(list_group)
        layout.addWidget(edit_group)
        layout.addLayout(button_layout)
        layout.addWidget(self.status_label)
        
        self.setLayout(layout)
        
        # 테이블 선택 이벤트 연결
        self.label_table.itemSelectionChanged.connect(self.on_selection_changed)
        
        # 라벨 목록 로드
        self.refresh_table()
    
    def refresh_table(self):
        """라벨 테이블 갱신"""
        self.label_table.setRowCount(0)
        
        for row, (label_id, label_name) in enumerate(sorted(self.label_names.items())):
            self.label_table.insertRow(row)
            self.label_table.setItem(row, 0, QTableWidgetItem(str(label_id)))
            self.label_table.setItem(row, 1, QTableWidgetItem(label_name))
    
    def on_selection_changed(self):
        """테이블 선택 변경 시 처리"""
        selected_items = self.label_table.selectedItems()
        
        if selected_items:
            row = selected_items[0].row()
            label_id = int(self.label_table.item(row, 0).text())
            label_name = self.label_table.item(row, 1).text()
            
            # 편집 필드 업데이트
            self.id_spin.setValue(label_id)
            self.name_edit.setText(label_name)
            
            # 삭제 버튼 활성화
            self.delete_btn.setEnabled(True)
        else:
            # 선택 없음 - 삭제 버튼 비활성화
            self.delete_btn.setEnabled(False)
    
    def add_or_update_label(self):
        """라벨 추가 또는 수정"""
        label_id = self.id_spin.value()
        label_name = self.name_edit.text().strip()
        
        if not label_name:
            self.status_label.setText("오류: 라벨 이름을 입력하세요.")
            return
        
        # 라벨 추가 또는 수정
        self.label_names[label_id] = label_name
        
        # 테이블 갱신
        self.refresh_table()
        
        # 상태 메시지 업데이트
        self.status_label.setText(f"라벨 {label_id}: '{label_name}' 추가/수정됨")
    
    def delete_label(self):
        """선택된 라벨 삭제"""
        selected_items = self.label_table.selectedItems()
        
        if selected_items:
            row = selected_items[0].row()
            label_id = int(self.label_table.item(row, 0).text())
            
            # 확인 대화상자
            reply = QMessageBox.question(
                self, 
                '라벨 삭제', 
                f"라벨 {label_id}을(를) 삭제하시겠습니까?",
                QMessageBox.Yes | QMessageBox.No, 
                QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                # 라벨 삭제
                if label_id in self.label_names:
                    del self.label_names[label_id]
                    
                    # 테이블 갱신
                    self.refresh_table()
                    
                    # 상태 메시지 업데이트
                    self.status_label.setText(f"라벨 {label_id} 삭제됨")
    
    def save_labels(self):
        """라벨 설정 저장"""
        success, message = save_label_names(self.label_names)
        
        if success:
            QMessageBox.information(self, "저장 완료", message)
        else:
            QMessageBox.warning(self, "저장 오류", message)
        
        self.status_label.setText(message)

# ------------------------------------------------------------------------
# 메인 애플리케이션 윈도우
# ------------------------------------------------------------------------
class EMGSystemApp(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # UI 초기화
        self.init_ui()
    
    def init_ui(self):
        # 메인 윈도우 설정
        self.setWindowTitle("EMG 동작 인식 시스템")
        self.setGeometry(100, 100, 1000, 800)
        
        # 중앙 위젯 및 탭 구성
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        main_layout = QVBoxLayout()
        self.central_widget.setLayout(main_layout)
        
        # 탭 위젯 생성
        self.tabs = QTabWidget()
        
        # 데이터 수집 탭
        self.collector_widget = EMGDataCollectorWidget(self)
        self.tabs.addTab(self.collector_widget, "데이터 수집")
        
        # 모델 학습 탭
        self.learner_widget = ModelLearningWidget(self)
        self.tabs.addTab(self.learner_widget, "모델 학습")
        
        # 모델 테스트 탭
        self.tester_widget = EMGTestWidget(self)
        self.tabs.addTab(self.tester_widget, "동작 테스트")
        
        # 라벨 관리 탭
        self.label_manager = LabelManagerWidget(self)
        self.tabs.addTab(self.label_manager, "라벨 관리")
        
        # 탭 변경 시 이벤트 처리
        self.tabs.currentChanged.connect(self.on_tab_changed)
        
        # 상태 바
        self.statusBar().showMessage("준비 완료")
        
        # 레이아웃에 추가
        main_layout.addWidget(self.tabs)
        
        # 최초 실행 시 데이터 폴더 확인
        self.show_data_folder_info()
    
    def on_tab_changed(self, index):
        """탭 변경 이벤트 처리"""
        tab_name = self.tabs.tabText(index)
        self.setWindowTitle(f"EMG 동작 인식 시스템 - {tab_name}")
    
    def show_data_folder_info(self):
        """데이터 폴더 정보 표시"""
        message = ensure_data_folder_exists()
        
        # 이미 수집된 데이터 파일 확인
        if os.path.exists(DATA_FOLDER):
            files = [f for f in os.listdir(DATA_FOLDER) if f.endswith('.csv')]
            if files:
                message += f"\n{len(files)}개의 데이터 파일이 있습니다."
            else:
                message += "\n데이터 파일이 없습니다. 데이터 수집을 먼저 실행하세요."
        
        # 모델 파일 확인
        if os.path.exists(MODEL_FILE):
            message += f"\n모델 파일이 있습니다: {MODEL_FILE}"
        else:
            message += f"\n모델 파일이 없습니다. 모델 학습을 실행하세요."
        
        # 메시지 표시
        self.statusBar().showMessage(message)

# ------------------------------------------------------------------------
# 메인 실행 코드
# ------------------------------------------------------------------------
def main():
    # 어플리케이션 생성
    app = QApplication(sys.argv)
    
    # 스타일 설정 (선택 사항)
    app.setStyle("Fusion")
    
    # 어두운 테마 설정 (선택 사항)
    dark_palette = QPalette()
    dark_palette.setColor(QPalette.Window, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.WindowText, Qt.white)
    dark_palette.setColor(QPalette.Base, QColor(25, 25, 25))
    dark_palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.ToolTipBase, Qt.white)
    dark_palette.setColor(QPalette.ToolTipText, Qt.white)
    dark_palette.setColor(QPalette.Text, Qt.white)
    dark_palette.setColor(QPalette.Button, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.ButtonText, Qt.white)
    dark_palette.setColor(QPalette.BrightText, Qt.red)
    dark_palette.setColor(QPalette.Link, QColor(42, 130, 218))
    dark_palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
    dark_palette.setColor(QPalette.HighlightedText, Qt.black)
    
    # 어두운 테마 적용
    app.setPalette(dark_palette)
    
    # 메인 윈도우 생성 및 표시
    window = EMGSystemApp()
    window.show()
    
    # 어플리케이션 실행
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()