import os
import sys
import glob
import subprocess

def main():
    # Find TCN checkpoint under bulk/ matching uTCN
    tcn_pattern = "lightning_logs/bulk/*uTCN*/lightning_logs/*/checkpoints/*.ckpt"
    tcn_ckpts = glob.glob(tcn_pattern)
    if not tcn_ckpts:
        tcn_ckpts = glob.glob("lightning_logs/bulk/**/*TCN*.ckpt", recursive=True)
    
    # Find FNO checkpoint under bulk/ matching uFNO or FNO
    fno_pattern = "lightning_logs/bulk/*uFNO*/lightning_logs/*/checkpoints/*.ckpt"
    fno_ckpts = glob.glob(fno_pattern)
    if not fno_ckpts:
        fno_ckpts = glob.glob("lightning_logs/bulk/**/*FNO*.ckpt", recursive=True)
        
    if not tcn_ckpts:
        print("[ERROR] Could not find any TCN checkpoint under lightning_logs/bulk/")
        return
    if not fno_ckpts:
        print("[ERROR] Could not find any FNO checkpoint under lightning_logs/bulk/")
        return
        
    # Sort by modification time to get the latest trained checkpoint
    tcn_ckpts.sort(key=os.path.getmtime, reverse=True)
    fno_ckpts.sort(key=os.path.getmtime, reverse=True)
    
    tcn_ckpt = tcn_ckpts[0]
    fno_ckpt = fno_ckpts[0]
    
    print(f"[EVAL] Found TCN checkpoint: {tcn_ckpt}")
    print(f"[EVAL] Found FNO checkpoint: {fno_ckpt}")
    
    cmd = [
        sys.executable, "scripts/compare_models_paper_style.py",
        "--tcn_checkpoint", tcn_ckpt,
        "--fno_checkpoint", fno_ckpt
    ]
    print(f"[EVAL] Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()
