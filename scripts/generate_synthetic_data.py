import os
import numpy as np
from scipy.io import wavfile

def generate_synthetic_data():
    # Define paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(script_dir)
    data_dir = os.path.join(root_dir, "data")
    x_t_dir = os.path.join(data_dir, "x_t")
    y_t_dir = os.path.join(data_dir, "y_t")

    os.makedirs(x_t_dir, exist_ok=True)
    os.makedirs(y_t_dir, exist_ok=True)

    print(f"Generating synthetic dataset in {data_dir}...")

    sr = 192000
    duration = 0.5  # 0.5 seconds
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)

    # Generate 5 segment pairs
    for i in range(1, 6):
        # Clean signal (x_t): combinations of sine sweeps and noise
        if i == 1:
            # Simple 1kHz sine wave
            x = np.sin(2 * np.pi * 1000 * t)
        elif i == 2:
            # Linear sine sweep from 100Hz to 5000Hz
            x = np.sin(2 * np.pi * (100 + 4900 * t / (2 * duration)) * t)
        elif i == 3:
            # Dual tone (500Hz + 2000Hz)
            x = 0.6 * np.sin(2 * np.pi * 500 * t) + 0.4 * np.sin(2 * np.pi * 2000 * t)
        elif i == 4:
            # White noise with low-pass filter
            noise = np.random.normal(0, 0.3, len(t))
            x = np.convolve(noise, np.ones(50)/50, mode='same')
        else:
            # Multi-frequency signal
            x = 0.5 * np.sin(2 * np.pi * 400 * t) + 0.3 * np.sin(2 * np.pi * 800 * t) + 0.2 * np.sin(2 * np.pi * 1600 * t)

        # Normalize to [-0.9, 0.9] to prevent clipping in raw audio
        x = x / np.max(np.abs(x)) * 0.9

        # Distorted signal (y_t): non-linear distortion (tanh saturation) + delay
        # y = tanh(x * 1.5) / tanh(1.5) to keep amplitude reasonable
        y = np.tanh(x * 1.8) / np.tanh(1.8)

        # Add a small phase shift / delay (e.g. 5 samples delay) to simulate physical distance
        delay_samples = 5
        y_delayed = np.zeros_like(y)
        y_delayed[delay_samples:] = y[:-delay_samples]
        # Add small noise to target
        y_delayed += np.random.normal(0, 0.005, len(t))

        # Convert to 16-bit PCM
        x_int16 = (x * 32767).astype(np.int16)
        y_int16 = (y_delayed * 32767).astype(np.int16)

        # Save WAV files
        x_file = os.path.join(x_t_dir, f"seg{i:04d}.wav")
        y_file = os.path.join(y_t_dir, f"seg{i:04d}.wav")

        wavfile.write(x_file, sr, x_int16)
        wavfile.write(y_file, sr, y_int16)
        print(f"  Saved seg{i:04d}.wav (Length: {len(x_int16)} samples)")

    print("Synthetic dataset generation complete!")

if __name__ == "__main__":
    generate_synthetic_data()
