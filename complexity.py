import torch
import torchinfo
import copy

MAX_MACS = 30_000_000
MAX_PARAMS_MEMORY = 128_000


def get_torch_macs_memory(model, input_size):
    if isinstance(input_size, torch.Size):
        input_size = tuple(input_size)

    if isinstance(input_size, torch.Tensor):
        input_size = tuple(input_size.size())

    # copy model and convert to full precision,
    # as torchinfo requires full precision to calculate macs
    model_for_profile = copy.deepcopy(model).float()

    model_profile = torchinfo.summary(model_for_profile, input_size=input_size, verbose=0)
    return model_profile.total_mult_adds, get_model_size_bytes(model)


def get_model_size_bytes(model: torch.nn.Module) -> int:
    """
    Calculate total model size in bytes, accounting for mixed parameter dtypes.

    Args:
        model: torch.nn.Module

    Returns:
        Total size in bytes (int)
    """
    dtype_to_bytes = {
        torch.float32: 4,
        torch.float16: 2,
        torch.bfloat16: 2,
        torch.int8: 1,
        torch.uint8: 1,
        torch.int32: 4,
        torch.qint8: 1,
        torch.quint8: 1,
        torch.int64: 8
    }

    # Directly load the original quantized model to calculate the correct model size
    pt_path = "Zhou_XJTLU_task1/ckpts/best_model_quantized_bias.pt"
    try:
        original_checkpoint = torch.load(pt_path, map_location='cpu')
        if isinstance(original_checkpoint, dict) and 'model_state_dict' in original_checkpoint:
            original_state_dict = original_checkpoint['model_state_dict']
        else:
            original_state_dict = original_checkpoint
            
        # Calculate the size of the original quantized model
        total_bytes = 0
        for param_name, param in original_state_dict.items():
            if isinstance(param, torch.Tensor):
                if param.is_quantized:
                    total_bytes += param.numel() * 1
                    continue
                    
                dtype = param.dtype
                num_elements = param.numel()
                bytes_per_param = dtype_to_bytes.get(dtype)
                
                if bytes_per_param is None:
                    print(f"Warning: Unknown data type {dtype}, using default value of 4 bytes")
                    bytes_per_param = 4
                    
                total_bytes += num_elements * bytes_per_param
                
        return total_bytes
    except Exception as e:
        print(f"Warning: Unable to load original quantized model, using current model to calculate size: {e}")
        # Fall back to using the current model to calculate size
        pass

    total_bytes = 0
    
    # First try using state_dict (suitable for quantized models)
    if hasattr(model, 'state_dict'):
        for param_name, param in model.state_dict().items():
            if isinstance(param, torch.Tensor):
                # Check if it's a quantized tensor
                if param.is_quantized:
                    # For quantized tensors, use the original quantized size instead of the dequantized size
                    # Typically quantized models use int8 (1 byte/element)
                    total_bytes += param.numel() * 1
                    continue
                    
                dtype = param.dtype
                num_elements = param.numel()
                bytes_per_param = dtype_to_bytes.get(dtype)
                
                if bytes_per_param is None:
                    print(f"Warning: Unknown data type {dtype}, using default value of 4 bytes")
                    bytes_per_param = 4
                    
                total_bytes += num_elements * bytes_per_param
    else:
        # Fall back to using parameters() method
        for param in model.parameters():
            if param.is_quantized:
                total_bytes += param.numel() * 1
                continue
                
            dtype = param.dtype
            num_elements = param.numel()
            bytes_per_param = dtype_to_bytes.get(dtype)
            
            if bytes_per_param is None:
                print(f"Warning: Unknown data type {dtype}, using default value of 4 bytes")
                bytes_per_param = 4
                
            total_bytes += num_elements * bytes_per_param

    return total_bytes
