#!/usr/bin/env python3
"""
Launcher script for torchrun training
"""

import subprocess
import sys
import os

def main():
    # Get the number of GPUs available
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    print(f"Detected {num_gpus} GPUs")
    
    # Command to run with torchrun
    cmd = [
        "torchrun",
        f"--nproc_per_node={num_gpus}",
        "fine_tune_instruct_ddp.py"
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    
    # Run the command
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Training failed with error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("Training interrupted by user")
        sys.exit(0)

if __name__ == "__main__":
    # Import torch to check availability
    try:
        import torch
    except ImportError:
        print("PyTorch not found. Please install it first.")
        sys.exit(1)
        
    main()