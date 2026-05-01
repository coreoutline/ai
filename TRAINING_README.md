# Distributed Training with CoreOutline Qwen Model

This document explains how to run the optimized distributed training for the CoreOutline Qwen model.

## Prerequisites

1. Ensure you have PyTorch with CUDA support installed
2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Running Training

### Single GPU Training
```bash
python fine_tune_instruct_ddp.py
```

### Multi-GPU Training with torchrun
```bash
# For single node, multiple GPUs
torchrun --nproc_per_node=NUM_GPUS fine_tune_instruct_ddp.py

# Example with 2 GPUs
torchrun --nproc_per_node=2 fine_tune_instruct_ddp.py

# Using the launcher script (automatically detects GPUs)
python launch_training.py
```

### Multi-Node Training
```bash
# On the main node
torchrun \
    --nproc_per_node=NUM_GPUS_PER_NODE \
    --nnodes=NUM_NODES \
    --node_rank=0 \
    --master_addr="MASTER_IP_ADDRESS" \
    --master_port=12345 \
    fine_tune_instruct_ddp.py

# On worker nodes
torchrun \
    --nproc_per_node=NUM_GPUS_PER_NODE \
    --nnodes=NUM_NODES \
    --node_rank=NODE_RANK \
    --master_addr="MASTER_IP_ADDRESS" \
    --master_port=12345 \
    fine_tune_instruct_ddp.py
```

## Key Features of the Optimized Implementation

1. **Full torchrun compatibility**: The script works seamlessly with torchrun for distributed training
2. **Proper DDP setup**: Correctly initializes distributed training with NCCL backend
3. **Memory optimization**: Uses gradient accumulation and gradient clipping
4. **Mixed precision training**: Implements automatic mixed precision for faster training
5. **Checkpointing**: Saves model checkpoints during training
6. **Proper data loading**: Uses DistributedSampler for efficient data distribution
7. **Robust error handling**: Gracefully handles errors and cleans up resources

## Configuration

You can modify training parameters in the `fine_tune_instruct_ddp.py` file:

- `BATCH_SIZE`: Batch size per GPU
- `ACCUMULATION_STEPS`: Gradient accumulation steps
- `MAX_SEQ_LENGTH`: Maximum sequence length
- `NUM_EPOCHS`: Number of training epochs
- `EVAL_FREQ`: Evaluation frequency (in epochs)
- `LEARNING_RATE`: Initial learning rate
- `WEIGHT_DECAY`: Weight decay for optimizer
- `GRADIENT_CLIP`: Gradient clipping threshold

## Model Checkpoints

The training script automatically saves:
- Best model based on validation loss: `./models/nyx_ddp_best.pth`
- Model every 10 epochs: `./models/nyx_ddp_epoch_{N}.pth`
- Final model: `./models/nyx_ddp_final.pth`

## Monitoring

The training progress is displayed in the console with:
- Current epoch and progress
- Training loss
- Validation loss (every EVAL_FREQ epochs)