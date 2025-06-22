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
    }

    total_bytes = 0
    
    # 首先尝试使用state_dict（适用于量化模型）
    if hasattr(model, 'state_dict'):
        for param_name, param in model.state_dict().items():
            if isinstance(param, torch.Tensor):
                dtype = param.dtype
                num_elements = param.numel()
                bytes_per_param = dtype_to_bytes.get(dtype)
                
                if bytes_per_param is None:
                    print(f"警告：未知的数据类型 {dtype}，使用默认值4字节")
                    bytes_per_param = 4
                    
                total_bytes += num_elements * bytes_per_param
                # 打印详细信息（可选）
                # print(f"参数: {param_name}, 形状: {param.shape}, 类型: {dtype}, 大小: {num_elements * bytes_per_param} 字节")
    else:
        # 回退到使用parameters()方法
        for param in model.parameters():
            dtype = param.dtype
            num_elements = param.numel()
            bytes_per_param = dtype_to_bytes.get(dtype)
            
            if bytes_per_param is None:
                print(f"警告：未知的数据类型 {dtype}，使用默认值4字节")
                bytes_per_param = 4
                
            total_bytes += num_elements * bytes_per_param

    return total_bytes
