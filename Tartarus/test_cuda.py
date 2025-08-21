import torch
import gc

def print_gpu_memory():
    """Comprehensive GPU memory monitoring"""
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available - using CPU only")
        return
    
    print(f"🔧 CUDA Status:")
    print(f"   Available: {torch.cuda.is_available()}")
    print(f"   Device count: {torch.cuda.device_count()}")
    print(f"   Current device: {torch.cuda.current_device()}")
    
    # Check each GPU
    for i in range(torch.cuda.device_count()):
        torch.cuda.set_device(i)
        allocated = torch.cuda.memory_allocated(i)
        reserved = torch.cuda.memory_reserved(i)
        total = torch.cuda.get_device_properties(i).total_memory
        
        print(f"\n📊 GPU {i} ({torch.cuda.get_device_name(i)}):")
        print(f"   Allocated: {allocated / 1024**3:.2f} GB ({allocated / 1024**2:.0f} MB)")
        print(f"   Reserved:  {reserved / 1024**3:.2f} GB ({reserved / 1024**2:.0f} MB)")
        print(f"   Total:     {total / 1024**3:.2f} GB")
        print(f"   Free:      {(total - reserved) / 1024**3:.2f} GB")
        print(f"   Usage:     {reserved / total * 100:.1f}%")

def test_gpu_allocation():
    """Test GPU allocation to verify monitoring works"""
    if not torch.cuda.is_available():
        print("Cannot test - CUDA not available")
        return
    
    print("🧪 Testing GPU allocation...")
    print("Before tensor creation:")
    print_gpu_memory()
    
    # Create a test tensor
    test_tensor = torch.randn(1000, 1000, device='cuda')
    print(f"\n✅ Created tensor on GPU: {test_tensor.shape}")
    print("After tensor creation:")
    print_gpu_memory()
    
    # Clean up
    del test_tensor
    torch.cuda.empty_cache()
    gc.collect()
    print(f"\n🧹 After cleanup:")
    print_gpu_memory()

# Run the tests
print_gpu_memory()
print("\n" + "="*50)
test_gpu_allocation()