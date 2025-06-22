import torch
import torchinfo
import copy
from Zhou_XJTLU_task1.model.backbones import TFSepNet
from neural_compressor.utils.pytorch import load

# Constants
MAX_MACS = 30_000_000
MAX_PARAMS_MEMORY = 128_000
# MAX_PARAMS_MEMORY = 128 * 1024  # 128 KB
def get_model_size_bytes(model: torch.nn.Module, exclude_quantization_params=True) -> int:
    """Calculate total model size in bytes, accounting for mixed parameter dtypes."""
    dtype_to_bytes = {
        torch.float32: 4,
        torch.float16: 2,
        torch.bfloat16: 2,
        torch.int8: 1,
        torch.uint8: 1,
        torch.int32: 4,
        torch.int64: 8,
        torch.qint8: 1,
        torch.quint8: 1,
    }

    total_bytes = 0
    
    # For quantized models, use state_dict instead of parameters()
    if hasattr(model, 'state_dict'):
        for param_name, param in model.state_dict().items():
            if isinstance(param, torch.Tensor):
                # Exclude quantization parameters
                if exclude_quantization_params:
                    # Check if parameter name contains quantization-related keywords
                    if any(keyword in param_name for keyword in ['scale', 'zero_point']):
                        continue
                
                dtype = param.dtype
                num_elements = param.numel()
                bytes_per_param = dtype_to_bytes.get(dtype, 4)  # default to 4 bytes
                total_bytes += num_elements * bytes_per_param
    else:
        for param in model.parameters():
            dtype = param.dtype
            num_elements = param.numel()
            bytes_per_param = dtype_to_bytes.get(dtype, 4)
            total_bytes += num_elements * bytes_per_param

    return total_bytes

def get_torch_macs_memory_quantized(quantized_model, input_size):
    """Calculate MACs and memory for quantized model using original model."""
    
    # Method 1: Try direct calculation on quantized model
    try:
        model_profile = torchinfo.summary(quantized_model, input_size=input_size, verbose=0)
        macs = model_profile.total_mult_adds
        memory = get_model_size_bytes(quantized_model)
        return macs, memory
    except:
        pass
    
    # Method 2: Use original model for MACs calculation
    try:
        original_model = TFSepNet(
            in_channels=1,
            num_classes=10,
            base_channels=64,
            depth=17
        ).float().cpu()
        
        model_profile = torchinfo.summary(original_model, input_size=input_size, verbose=0)
        macs = model_profile.total_mult_adds
        memory = get_model_size_bytes(quantized_model)
        return macs, memory
    except Exception as e:
        print(f"Error: {e}")
        return None, None

def analyze_quantized_model():
    """Analyze quantized model complexity"""

    pt_path = "Zhou_XJTLU_task1/log_ckpt/best_model_quantized_bias.pt"
    try:
        # Load quantized state dictionary directly
        checkpoint = torch.load(pt_path, map_location='cpu')
        print("Checkpoint type:", type(checkpoint))
        print("Checkpoint keys:", list(checkpoint.keys()) if isinstance(checkpoint, dict) else "Not a dict")
        
        # Extract the actual model state dictionary
        if 'model_state_dict' in checkpoint:
            model_state_dict = checkpoint['model_state_dict']
            print("Found model_state_dict, keys:", len(model_state_dict))
        else:
            model_state_dict = checkpoint
            print("Using checkpoint directly as state_dict")
        
        # Method 2: Analyze state dictionary size directly
        print("Calculating model size from state dict...")
        
        total_bytes = 0
        filtered_count = 0
        processed_count = 0
        
        dtype_to_bytes = {
            torch.float32: 4, torch.float16: 2, torch.bfloat16: 2,
            torch.int8: 1, torch.uint8: 1, torch.int32: 4, torch.int64: 8,
            torch.qint8: 1, torch.quint8: 1,
        }
        
        print("\nAnalyzing model state dict contents:")
        for param_name, param in model_state_dict.items():
            if isinstance(param, torch.Tensor):
                print(f"  Found tensor: {param_name}, shape: {param.shape}, dtype: {param.dtype}")
                
                # Check if it should be filtered
                filter_keywords = ['scale', 'zero_point', 'best_configure']
                is_filtered = any(keyword in param_name for keyword in filter_keywords)
                
                if is_filtered:
                    print(f"    -> FILTERED (contains: {[kw for kw in filter_keywords if kw in param_name]})")
                    filtered_count += 1
                    continue
                
                processed_count += 1
                dtype = param.dtype
                num_elements = param.numel()
                bytes_per_param = dtype_to_bytes.get(dtype, 4)
                param_bytes = num_elements * bytes_per_param
                total_bytes += param_bytes
                print(f"    -> INCLUDED: {param_bytes} bytes")
            else:
                print(f"  Non-tensor item: {param_name}, type: {type(param)}")
        
        print(f"\nSummary:")
        print(f"  Total items in model_state_dict: {len(model_state_dict)}")
        print(f"  Tensor parameters found: {filtered_count + processed_count}")
        print(f"  Parameters filtered: {filtered_count}")
        print(f"  Parameters processed: {processed_count}")
        
        memory_bytes = total_bytes
        
        # Display other information
        if 'original_memory' in checkpoint and 'new_memory' in checkpoint:
            print(f"  Original memory (from checkpoint): {checkpoint['original_memory']} bytes")
            print(f"  New memory (from checkpoint): {checkpoint['new_memory']} bytes")
        
        # For MACs, use original model calculation
        print("\nCalculating MACs using original model...")
        original_model = TFSepNet(in_channels=1, num_classes=10, base_channels=64, depth=17).cpu()
        input_size = (1, 1, 512, 64)
        model_profile = torchinfo.summary(original_model, input_size=input_size, verbose=0)
        macs = model_profile.total_mult_adds
        
        print(f"\nModel Analysis Results:")
        print(f"MACs: {macs:,}")
        print(f"Memory: {memory_bytes:,} bytes ({memory_bytes/1024:.1f} KB)")
        
        # Check constraints
        mac_ok = macs <= MAX_MACS
        memory_ok = memory_bytes <= MAX_PARAMS_MEMORY
        
        print(f"MACs: {macs:,} ({'PASS' if mac_ok else 'FAIL'}, {macs/MAX_MACS*100:.1f}% of limit)")
        print(f"Memory: {memory_bytes:,} bytes ({memory_bytes/1024:.1f} KB) ({'PASS' if memory_ok else 'FAIL'}, {memory_bytes/MAX_PARAMS_MEMORY*100:.1f}% of limit)")
        print(f"Status: {'ALL CONSTRAINTS MET' if mac_ok and memory_ok else 'OPTIMIZATION NEEDED'}")
        
        if not memory_ok:
            excess_kb = (memory_bytes - MAX_PARAMS_MEMORY) / 1024
            print(f"Memory exceeds by {excess_kb:.1f} KB")
        
        if not mac_ok:
            excess_pct = (macs - MAX_MACS) / MAX_MACS * 100
            print(f"MACs exceed by {excess_pct:.1f}%")
            
    except Exception as e:
        print(f"Error in analysis: {e}")
        import traceback
        traceback.print_exc()

def debug_model_structure():
    """Debug model structure and saved checkpoint"""
    pt_path = "Zhou_XJTLU_task1/log_ckpt/best_model_quantized_bias.pt"
    
    # Check saved content
    checkpoint = torch.load(pt_path, map_location='cpu')
    print("Checkpoint type:", type(checkpoint))
    
    if isinstance(checkpoint, dict):
        print("Checkpoint keys:", list(checkpoint.keys()))
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
    else:
        # If it's directly a model
        if hasattr(checkpoint, 'state_dict'):
            state_dict = checkpoint.state_dict()
        else:
            print("Cannot extract state_dict")
            return
    
    print("\nSaved model parameter keys:")
    for i, key in enumerate(state_dict.keys()):
        print(f"  {key}")
        if i > 20:  # Only show first 20
            print("  ...")
            break
    
    # Compare with expected model structure
    model_fp = TFSepNet(in_channels=1, num_classes=10, base_channels=64, depth=17).cpu()
    print("\nExpected model parameter keys:")
    for i, key in enumerate(model_fp.state_dict().keys()):
        print(f"  {key}")
        if i > 20:
            print("  ...")
            break

# Call debug function in main
if __name__ == "__main__":
    # debug_model_structure()
    analyze_quantized_model()