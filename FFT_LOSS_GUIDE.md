# FFT Loss 구현 가이드

## 개요

`microtcn/base.py`에 **Global FFT Loss**를 구현했습니다. STFT의 고정된 해상도 제한을 극복하고, 전역 주파수 크기(magnitude)와 위상(phase)을 더 정확하게 학습할 수 있습니다.

## 구현 내용

### 1. FFTLoss 클래스

```python
class FFTLoss(torch.nn.Module):
    """Global FFT Loss for Audio Reconstruction.
    
    Args:
        use_phase (bool): 위상 차이를 손실에 포함할지 여부 (기본값: False)
        phase_weight (float): 위상 손실 항의 가중치 (기본값: 0.1)
    """
```

**동작 방식:**
- 입력: `pred`와 `target` 파형 (Batch, Channels, Time)
- 연산: `torch.fft.rfft`를 사용하여 Real-to-Complex FFT 적용
- 메트릭: 
  - **Magnitude Loss**: FFT 크기(|FFT|) 간의 L1 거리
  - **Phase Loss** (선택적): 위상 차이 ([-π, π]로 래핑)

### 2. 지원되는 Loss 조합

`train_loss` 파라미터로 다음 조합을 사용할 수 있습니다:

- `"l1"`: L1 Loss만
- `"stft"`: STFT Loss만
- `"fft"`: FFT Loss만 (새로 추가)
- `"l1+stft"`: L1 + STFT (기본값)
- `"l1+fft"`: L1 + FFT (새로 추가)
- `"fft+stft"`: FFT + STFT (새로 추가)
- `"l1+fft+stft"`: L1 + FFT + STFT (새로 추가)

### 3. 새로운 커맨드 라인 인자

```bash
--train_loss fft              # FFT Loss 사용
--use_fft_loss                # FFT Loss 활성화 플래그
--fft_use_phase               # 위상 항 포함
--fft_phase_weight 0.1        # 위상 가중치 (기본값: 0.1)
```

## 사용 예시

### 기본 FFT Loss 사용

```bash
python train.py --train_loss fft
```

### L1 + FFT Loss 조합

```bash
python train.py --train_loss l1+fft
```

### FFT Loss + 위상 항 포함

```bash
python train.py --train_loss fft --fft_use_phase --fft_phase_weight 0.2
```

### 모든 Loss 조합 (L1 + FFT + STFT)

```bash
python train.py --train_loss l1+fft+stft
```

### 192kHz + FFT Loss

```bash
python train.py \
    --sample_rate 192000 \
    --train_loss l1+fft \
    --fft_use_phase
```

## Validation 로깅

Validation 단계에서 다음 메트릭이 자동으로 로깅됩니다:

- `val_loss`: 종합 손실
- `val_loss/L1`: L1 손실
- `val_loss/STFT`: STFT 손실
- `val_loss/FFT`: FFT 손실 (새로 추가)

TensorBoard에서 확인할 수 있습니다:

```bash
tensorboard --logdir lightning_logs
```

## FFT Loss vs STFT Loss

### STFT Loss
- **장점**: 시간-주파수 해상도 트레이드오프, 지역적 특성 포착
- **단점**: 윈도우 크기와 홉 크기에 따른 고정된 해상도 제한

### FFT Loss (새로 추가)
- **장점**: 
  - 전역 주파수 스펙트럼 학습
  - 해상도 제한 없음
  - 전체 주파수 대역의 크기와 위상 정보 포착
- **단점**: 시간적 지역성 정보 부족

### 권장 사항
- **FFT + STFT 조합**: 전역 주파수 특성과 지역적 시간-주파수 특성을 모두 학습
- **L1 + FFT**: 시간 도메인과 주파수 도메인 모두 고려

## 기술적 세부사항

### FFT 적용 차원
- FFT는 마지막 차원(Time)에 적용됩니다: `torch.fft.rfft(x, dim=-1)`
- Real-to-Complex FFT를 사용하여 효율성 향상

### 위상 처리
- 위상 차이는 [-π, π] 범위로 래핑됩니다
- `phase_diff = min(phase_diff, 2π - phase_diff)`

### Validation Aggregate Loss
- FFT Loss가 활성화된 경우: `L1 + STFT + FFT`
- 그렇지 않은 경우: `L1 + STFT` (기존 동작 유지)

## 실험 권장사항

1. **Baseline 비교**: 먼저 `l1+stft`로 학습 후 `l1+fft`와 비교
2. **조합 실험**: `fft+stft` 또는 `l1+fft+stft` 시도
3. **위상 항 실험**: `--fft_use_phase`로 위상 항의 효과 확인
4. **가중치 조정**: `--fft_phase_weight`로 위상 가중치 튜닝

## 주의사항

- `export.py`에서 JIT 컴파일 시 FFTLoss 관련 라인을 주석 처리해야 할 수 있습니다
- FFT Loss는 전체 시퀀스에 대해 계산되므로, 매우 긴 시퀀스의 경우 메모리 사용량이 증가할 수 있습니다

