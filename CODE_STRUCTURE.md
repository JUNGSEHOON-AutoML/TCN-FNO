# TCN 프로젝트 코드 구조

## 📁 폴더 구조

```
code_only/
├── microtcn/              # 핵심 모델 모듈
│   ├── __init__.py
│   ├── base.py           # 기본 모델 클래스
│   ├── data.py           # 데이터 로더
│   ├── tcn.py            # TCN 모델 (Lightning)
│   ├── tcn_bare.py       # TCN 모델 (Bare PyTorch)
│   ├── lstm.py           # LSTM 비교 모델
│   └── utils.py          # 유틸리티 함수
│
├── train.py              # 모델 학습 스크립트
├── test.py               # 모델 테스트 스크립트
├── export.py             # 모델 내보내기
├── speed.py              # 속도 벤치마크
├── plot.py               # 결과 플롯
│
├── evaluate_all_models.py           # 전체 모델 평가
├── evaluate_models.py               # 모델 평가
├── compare_models_paper_style.py    # 모델 비교 (논문 스타일)
├── find_best_tcn.py                 # 최고 모델 찾기
│
├── visualize_all_tcn_final_style.py # 전체 시각화 (최종)
├── visualize_tcn_paper_style.py     # 시각화 (논문 스타일)
├── visualize_final_tcn.py           # 최종 시각화
├── rename_visualizations.py         # 시각화 파일명 변경
│
├── setup.py              # 패키지 설정
├── requirements.txt      # 의존성 패키지
│
└── 문서 파일들
    ├── README.md
    ├── USAGE_GUIDE.md
    ├── SETUP_GUIDE.md
    ├── FFT_LOSS_GUIDE.md
    └── FINAL_EVALUATION_GUIDE.md
```

## 🚀 사용 방법

### 1. 환경 설정
```bash
pip install -r requirements.txt
pip install -e .
```

### 2. 모델 학습
```bash
python train.py --model tcn --config config.yaml
```

### 3. 모델 평가
```bash
python evaluate_all_models.py
```

### 4. 시각화
```bash
python visualize_all_tcn_final_style.py
```

## 📊 핵심 모델

### TCN (Temporal Convolutional Network)
- **파일**: `microtcn/tcn.py`, `microtcn/tcn_bare.py`
- **특징**: Causal/Non-causal convolution, Residual connections
- **변형**: μ-TCN (경량화 버전)

### LSTM (비교 모델)
- **파일**: `microtcn/lstm.py`
- **용도**: 성능 비교 baseline

## 📝 주요 기능

1. **학습**: `train.py` - 다양한 설정으로 모델 학습
2. **평가**: `evaluate_*.py` - 성능 메트릭 계산
3. **비교**: `compare_*.py` - 모델 간 성능 비교
4. **시각화**: `visualize_*.py` - 결과 시각화
5. **내보내기**: `export.py` - 학습된 모델 내보내기

## 🔧 설정

- **데이터**: WAV 파일 (x_t/, y_t/)
- **샘플링**: 192kHz → 48kHz
- **시퀀스 길이**: 57600 샘플 (0.3초)
- **배치 크기**: 16-32

## 📈 성능 메트릭

- MSE (Mean Squared Error)
- Correlation
- RMS Ratio
- FFT-based metrics

## 🎯 최고 성능 모델

- **모델**: μ-TCN-100 (Causal)
- **설정**: 4-10-5 (channels-kernel-dilation)
- **성능**: MSE < 0.01, Correlation > 0.95
