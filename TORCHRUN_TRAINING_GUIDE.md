# CoreOutline Qwen Model Training with torchrun

This guide explains how to use the optimized distributed training script for the CoreOutline Qwen model.

## File Structure

- `fine_tune_instruct_ddp_optimized.py` - The main optimized training script
- `launch_training.py` - Helper script to launch training
- `TRAINING_README.md` - General training documentation
- `requirements.txt` - Python dependencies

## Prerequisites

1. Ensure you have PyTorch with CUDA support installed
2. Install required packages:
   ```
   pip install -r requirements.txt
   ```

## Running Training

### Single GPU Training
```bash
python fine_tune_instruct_ddp_optimized.py
```

### Multi-GPU Training with torchrun (Recommended)
```bash
# For single node, multiple GPUs
torchrun --nproc_per_node=NUM_GPUS fine_tune_instruct_ddp_optimized.py

# Example with 2 GPUs
torchrun --nproc_per_node=2 fine_tune_instruct_ddp_optimized.py

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
    fine_tune_instruct_ddp_optimized.py

# On worker nodes
torchrun \
    --nproc_per_node=NUM_GPUS_PER_NODE \
    --nnodes=NUM_NODES \
    --node_rank=NODE_RANK \
    --master_addr="MASTER_IP_ADDRESS" \
    --master_port=12345 \
    fine_tune_instruct_ddp_optimized.py
```

## Key Improvements in the Optimized Script

1. **Full torchrun compatibility**: Works seamlessly with torchrun for distributed training
2. **Proper DDP setup**: Correctly initializes distributed training with NCCL backend
3. **Memory optimization**: Uses gradient accumulation and gradient clipping
4. **Mixed precision training**: Implements automatic mixed precision for faster training
5. **Checkpointing**: Saves model checkpoints during training
6. **Proper data loading**: Uses DistributedSampler for efficient data distribution
7. **Robust error handling**: Gracefully handles errors and cleans up resources
8. **Progress tracking**: Shows training progress with tqdm
9. **Configurable parameters**: Easy to adjust training hyperparameters

## Configuration

You can modify training parameters in the `fine_tune_instruct_ddp_optimized.py` file:

- `BATCH_SIZE`: Batch size per GPU (default: 2)
- `ACCUMULATION_STEPS`: Gradient accumulation steps (default: 4)
- `MAX_SEQ_LENGTH`: Maximum sequence length (default: 2048)
- `NUM_EPOCHS`: Number of training epochs (default: 100)
- `EVAL_FREQ`: Evaluation frequency in epochs (default: 5)
- `LEARNING_RATE`: Initial learning rate (default: 6e-5)
- `WEIGHT_DECAY`: Weight decay for optimizer (default: 0.1)
- `GRADIENT_CLIP`: Gradient clipping threshold (default: 1.0)

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

## Troubleshooting

1. **CUDA Out of Memory**: Reduce `BATCH_SIZE` or `MAX_SEQ_LENGTH`
2. **NCCL Errors**: Ensure all nodes can communicate on the specified port
3. **Data Loading Issues**: Check internet connection for downloading the dataset
4. **Permission Errors**: Ensure write permissions for the `models` directory

## Performance Tips

1. Use mixed precision training (already enabled)
2. Adjust batch size based on your GPU memory
3. Use gradient accumulation for larger effective batch sizes
4. Enable gradient checkpointing for memory efficiency