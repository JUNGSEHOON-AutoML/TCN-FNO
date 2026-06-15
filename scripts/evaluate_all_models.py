#!/usr/bin/env python3
"""
Evaluate all trained models and generate a summary report.
"""
import os
import sys
import glob
import subprocess
from pathlib import Path

# Resolve project root directory
ROOT_DIR = Path(__file__).resolve().parent.parent
os.chdir(ROOT_DIR)
sys.path.insert(0, str(ROOT_DIR))

# NumPy compatibility patch
import numpy as np
try:
    np.object = object
    np.bool = bool
    np.int = int
    np.float = float
    np.long = int
    np.complex = complex
except AttributeError:
    pass

def find_checkpoints(base_dir="lightning_logs/bulk"):
    """Find all epoch=59 checkpoints"""
    checkpoints = []
    base_dir_path = Path(ROOT_DIR) / base_dir
    pattern = str(base_dir_path / "*" / "lightning_logs" / "version_*" / "checkpoints" / "epoch=59*.ckpt")
    files = glob.glob(pattern)
    
    # Group by config
    configs = {}
    for f in sorted(files):
        # Extract config name from path using pathlib for cross-platform support
        p = Path(f)
        parts = p.parts
        try:
            bulk_idx = parts.index("bulk")
            config_name = parts[bulk_idx + 1]
        except (ValueError, IndexError):
            config_name = p.parent.parent.parent.parent.name
        
        # Get version number
        version = "0"
        for part in parts:
            if part.startswith('version_'):
                version = part.split('_')[1]
                break
        
        if config_name not in configs:
            configs[config_name] = []
        configs[config_name].append((f, version))
    
    # For each config, use the highest version number
    for config_name, versions in configs.items():
        # Sort by version number (as integer)
        versions.sort(key=lambda x: int(x[1]), reverse=True)
        checkpoints.append((config_name, versions[0][0]))
    
    return sorted(checkpoints, key=lambda x: x[0])

def evaluate_all(root_dir="."):
    """Evaluate all models and generate summary"""
    checkpoints = find_checkpoints()
    
    print("=" * 80)
    print("Found {} trained models".format(len(checkpoints)))
    print("=" * 80)
    
    results = []
    
    for config_name, checkpoint_path in checkpoints:
        print(f"\n{'='*80}")
        print(f"Evaluating: {config_name}")
        print(f"Checkpoint: {checkpoint_path}")
        print(f"{'='*80}\n")
        
        try:
            # Run evaluation using sys.executable to ensure we use the same Python env
            cmd = [
                sys.executable, str(ROOT_DIR / "scripts" / "evaluate_models.py"),
                "--checkpoint", checkpoint_path,
                "--root_dir", root_dir,
                "--batch_size", "1"
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=str(ROOT_DIR)
            )
            
            if result.returncode == 0:
                # Parse output
                output = result.stdout
                print(output)
                
                # Extract metrics
                mse = None
                pesq = None
                for line in output.split('\n'):
                    if 'Mean Squared Error' in line:
                        try:
                            mse = float(line.split(':')[1].strip().split()[0])
                        except:
                            pass
                    if 'PESQ Score:' in line and 'N/A' not in line:
                        try:
                            pesq = float(line.split(':')[1].strip().split()[0])
                        except:
                            pass
                
                results.append({
                    'config': config_name,
                    'checkpoint': checkpoint_path,
                    'mse': mse,
                    'pesq': pesq,
                    'status': 'success'
                })
            else:
                print(f"ERROR: {result.stderr}")
                results.append({
                    'config': config_name,
                    'checkpoint': checkpoint_path,
                    'mse': None,
                    'pesq': None,
                    'status': 'failed',
                    'error': result.stderr
                })
        except Exception as e:
            print(f"ERROR: {str(e)}")
            results.append({
                'config': config_name,
                'checkpoint': checkpoint_path,
                'mse': None,
                'pesq': None,
                'status': 'error',
                'error': str(e)
            })
    
    # Print summary
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    print(f"{'Config':<50} {'MSE':<15} {'PESQ':<10} {'Status':<10}")
    print("-" * 80)
    
    for r in results:
        mse_str = f"{r['mse']:.6e}" if r['mse'] is not None else "N/A"
        pesq_str = f"{r['pesq']:.4f}" if r['pesq'] is not None else "N/A"
        print(f"{r['config']:<50} {mse_str:<15} {pesq_str:<10} {r['status']:<10}")
    
    print("=" * 80)
    
    # Save results to file
    summary_file = "evaluation_summary.txt"
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("EVALUATION SUMMARY\n")
        f.write("=" * 80 + "\n")
        f.write(f"{'Config':<50} {'MSE':<15} {'PESQ':<10} {'Status':<10}\n")
        f.write("-" * 80 + "\n")
        for r in results:
            mse_str = f"{r['mse']:.6e}" if r['mse'] is not None else "N/A"
            pesq_str = f"{r['pesq']:.4f}" if r['pesq'] is not None else "N/A"
            f.write(f"{r['config']:<50} {mse_str:<15} {pesq_str:<10} {r['status']:<10}\n")
            if r['status'] != 'success' and 'error' in r:
                f.write(f"  Error: {r['error']}\n")
        f.write("=" * 80 + "\n")
    
    print(f"\nSummary saved to: {summary_file}")
    
    return results

if __name__ == "__main__":
    root_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    evaluate_all(root_dir)


