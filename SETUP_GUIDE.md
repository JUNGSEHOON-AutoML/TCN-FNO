**[Context]**
I am currently training a TCN model for Audio Reconstruction.
- Current Loss: `auraloss.freq.STFTLoss` (Time-Frequency domain)
- Goal: I want to experiment with **Global FFT Loss** instead of (or in addition to) STFT Loss.
- Reason: STFT has fixed resolution limits. I want the model to learn the global frequency magnitude and phase more accurately.

**[Request]**
Please modify `@microtcn/base.py` to implement a new custom loss class named `FFTLoss`.

1. **Implement `FFTLoss`**:
   - Input: `pred` and `target` waveforms (Batch, Channels, Time).
   - Operation: Apply `torch.fft.rfft` (Real-to-Complex FFT) to both inputs.
   - Metric: Calculate L1 distance between the **Magnitudes** (|FFT|) of pred and target.
   - (Optional) Add a term for **Phase** difference if helpful.

2. **Update `training_step`**:
   - Replace or combine the existing `self.stft` loss with this new `FFTLoss`.
   - Allow me to toggle this via a config flag if possible (e.g., `use_fft_loss=True`).

3. **Check Dimensions**: Ensure the FFT is applied along the last dimension (Time).