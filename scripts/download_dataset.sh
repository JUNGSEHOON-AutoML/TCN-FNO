#!/bin/bash
# ==============================================================================
# download_dataset.sh
# ==============================================================================
# Zenodo SignalTrain LA2A 데이터셋을 다운로드하고 프로젝트 구조에 맞게 연결하는 스크립트.

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
DATA_DIR="$ROOT_DIR/data"

# 0. 이미 데이터가 존재하면 완료된 것으로 간주하고 즉시 반환
if [ -d "$DATA_DIR/x_t" ] && [ -d "$DATA_DIR/y_t" ] && [ "$(ls -A "$DATA_DIR/x_t" 2>/dev/null)" ] && [ "$(ls -A "$DATA_DIR/y_t" 2>/dev/null)" ]; then
    echo "[INFO] 이미 데이터셋이 다운로드되고 data/x_t 및 data/y_t에 파일들이 존재하므로 다운로드를 건너뜁니다."
    exit 0
fi

echo "=============================================================================="
echo " SignalTrain LA2A Dataset Downloader & Restructurer"
echo "=============================================================================="

# 1. signaltrain 디렉토리 생성 및 이동
mkdir -p "$DATA_DIR/signaltrain"
cd "$DATA_DIR/signaltrain"

# 2. Zenodo 데이터셋 다운로드 및 압축 해제 (온더플라이 파이프를 사용하여 디스크 공간 절약)
URL="https://zenodo.org/records/3824876/files/SignalTrain_LA2A_Dataset_1.1.tgz?download=1"
echo "[INFO] 데이터셋 다운로드 및 압축 해제 중 (온더플라이 파이프 사용): $URL"
if command -v curl &> /dev/null; then
    curl -L "$URL" | tar -xzf -
elif command -v wget &> /dev/null; then
    wget -qO- "$URL" | tar -xzf -
else
    echo "[ERROR] wget 또는 curl이 설치되어 있지 않습니다. 아래 URL에서 직접 다운로드해 주세요:"
    echo "$URL"
    exit 1
fi

# 4. 프로젝트 구조에 맞게 디렉토리 준비
echo "[INFO] 프로젝트 구조(data/x_t, data/y_t) 준비 중..."
mkdir -p "$DATA_DIR/x_t"
mkdir -p "$DATA_DIR/y_t"
rm -f "$DATA_DIR/x_t"/*
rm -f "$DATA_DIR/y_t"/*

# 압축 해제된 폴더 찾기
EXTRACTED_DIR=$(find . -maxdepth 2 -type d -name "SignalTrain_LA2A_Dataset*" | head -n 1)

if [ -z "$EXTRACTED_DIR" ]; then
    # 만약 직접 Train 폴더가 풀린 경우
    EXTRACTED_DIR=$(find . -maxdepth 2 -type d -name "Train" | xargs dirname | head -n 1)
fi

if [ -n "$EXTRACTED_DIR" ]; then
    echo "[INFO] 데이터셋 폴더 감지 완료: $EXTRACTED_DIR"
    
    # Train/Input -> data/x_t 연결
    if [ -d "$EXTRACTED_DIR/Train/Input" ]; then
        echo "[INFO] Train/Input 파일을 data/x_t 로 연결 중..."
        ln -sf "$PWD/$EXTRACTED_DIR/Train/Input"/*.wav "$DATA_DIR/x_t/" 2>/dev/null || \
        cp -f "$PWD/$EXTRACTED_DIR/Train/Input"/*.wav "$DATA_DIR/x_t/"
    elif [ -d "$EXTRACTED_DIR/Train" ]; then
        # fallback: input_*.wav 파일을 data/x_t 로 연결
        echo "[INFO] Train/ 내 input_*.wav 파일을 data/x_t 로 연결 중..."
        ln -sf "$PWD/$EXTRACTED_DIR/Train"/input_*.wav "$DATA_DIR/x_t/" 2>/dev/null || \
        cp -f "$PWD/$EXTRACTED_DIR/Train"/input_*.wav "$DATA_DIR/x_t/"
    fi
    
    # Train/Target -> data/y_t 연결
    if [ -d "$EXTRACTED_DIR/Train/Target" ]; then
        echo "[INFO] Train/Target 파일을 data/y_t 로 연결 중..."
        ln -sf "$PWD/$EXTRACTED_DIR/Train/Target"/*.wav "$DATA_DIR/y_t/" 2>/dev/null || \
        cp -f "$PWD/$EXTRACTED_DIR/Train/Target"/*.wav "$DATA_DIR/y_t/"
    elif [ -d "$EXTRACTED_DIR/Train" ]; then
        # fallback: target_*.wav 파일을 data/y_t 로 연결
        echo "[INFO] Train/ 내 target_*.wav 파일을 data/y_t 로 연결 중..."
        ln -sf "$PWD/$EXTRACTED_DIR/Train"/target_*.wav "$DATA_DIR/y_t/" 2>/dev/null || \
        cp -f "$PWD/$EXTRACTED_DIR/Train"/target_*.wav "$DATA_DIR/y_t/"
    fi
    
    echo "[SUCCESS] 데이터셋 다운로드 및 연결 작업이 성공적으로 완료되었습니다!"
    echo "  - 입력(x_t) 경로: $DATA_DIR/x_t"
    echo "  - 정답(y_t) 경로: $DATA_DIR/y_t"
else
    echo "[ERROR] 압축 해제된 데이터셋 폴더 구조를 분석하지 못했습니다."
    exit 1
fi
