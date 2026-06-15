#!/usr/bin/env python3
"""
Rename visualization files to more readable names
"""
import os
import glob
import re

# Mapping of model IDs to readable names
MODEL_NAMES = {
    "1-uTCN-300__causal__4-10-13__fraction-0.01-bs32": "01_uTCN-300_causal_f001",
    "2-uTCN-100__causal__4-10-5__fraction-1.0-bs32": "02_uTCN-100_causal",
    "3-uTCN-300__causal__4-10-13__fraction-1.0-bs32": "03_uTCN-300_causal",
    "4-uTCN-1000__causal__5-10-5__fraction-1.0-bs32": "04_uTCN-1000_causal",
    "5-uTCN-100__noncausal__4-10-5__fraction-1.0-bs32": "05_uTCN-100_noncausal",
    "6-uTCN-300__noncausal__4-10-13__fraction-1.0-bs32": "06_uTCN-300_noncausal",
    "7-uTCN-1000__noncausal__5-10-5__fraction-1.0-bs32": "07_uTCN-1000_noncausal",
    "8-TCN-300__noncausal__10-2-15__fraction-1.0-bs16": "08_TCN-300_noncausal_bs16",
    "8-TCN-300__noncausal__10-2-15__fraction-1.0-bs32": "09_TCN-300_noncausal_bs32",
    "9-uTCN-300__causal__4-10-13__fraction-0.1-bs32": "10_uTCN-300_causal_f01",
    "11-uTCN-300__causal__3-60-5__fraction-1.0-bs32": "11_uTCN-300_causal_d60",
    "12-uTCN-300__causal__4-10-13__fraction-1.0-bs32__loss-l1": "12_uTCN-300_causal_l1",
    "13-uTCN-300__noncausal__30-2-15__fraction-1.0-bs32": "13_uTCN-300_noncausal_30b",
    "14-uTCN-324-16__noncausal__10-2-15__fraction-1.0-bs32": "14_uTCN-324-16_noncausal",
}

def rename_files():
    """Rename visualization files"""
    base_dir = "tcn_visualizations"
    
    if not os.path.exists(base_dir):
        print(f"❌ Directory {base_dir} not found!")
        return
    
    # Rename waveform files
    waveform_files = glob.glob(os.path.join(base_dir, "waveform_*.png"))
    
    renamed_count = 0
    for old_path in sorted(waveform_files):
        filename = os.path.basename(old_path)
        
        # Extract model ID from filename
        # Format: waveform_1-uTCN-300__causal__4-10-13__fraction-0.01-bs32.png
        match = re.match(r"waveform_(.+)\.png", filename)
        if not match:
            continue
        
        model_id = match.group(1)
        
        if model_id in MODEL_NAMES:
            new_name = f"waveform_{MODEL_NAMES[model_id]}.png"
            new_path = os.path.join(base_dir, new_name)
            
            if os.path.exists(new_path):
                print(f"⚠️  Skipping {filename}: {new_name} already exists")
                continue
            
            os.rename(old_path, new_path)
            print(f"✓ Renamed: {filename} → {new_name}")
            renamed_count += 1
        else:
            print(f"⚠️  No mapping found for: {model_id}")
    
    # Rename metrics comparison file (optional - keep original name or rename)
    metrics_file = os.path.join(base_dir, "tcn_metrics_comparison.png")
    if os.path.exists(metrics_file):
        new_metrics = os.path.join(base_dir, "metrics_comparison.png")
        if not os.path.exists(new_metrics):
            os.rename(metrics_file, new_metrics)
            print(f"✓ Renamed: tcn_metrics_comparison.png → metrics_comparison.png")
            renamed_count += 1
    
    print(f"\n✅ Renamed {renamed_count} files")

if __name__ == "__main__":
    rename_files()


