import importlib
import argparse
import json
import os
import sys
import torch
import numpy as np
from typing import List
from complexity import get_torch_macs_memory, MAX_MACS, MAX_PARAMS_MEMORY

# Add the Zhou_XJTLU_task1 directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Zhou_XJTLU_task1'))


def load_inputs(file_paths: List[str], device_ids: List[str], model) -> List[torch.Tensor]:
    """Load and preprocess inputs for each device"""
    inputs = []
    
    for file_path, device_id in zip(file_paths, device_ids):
        # Create dummy input tensor with the expected shape for audio processing
        # Based on your model's expected input: (batch_size, channels, freq_bins, time_frames)
        # Typical audio spectrogram shape: (1, 1, 512, 64)
        dummy_input = torch.randn(1, 1, 512, 64)
        inputs.append(dummy_input)
    
    return inputs


def get_model_for_device(model, device_id: str):
    """Get model variant for specific device"""
    # Return the actual PyTorch model, not the wrapper
    if hasattr(model, 'model') and hasattr(model.model, 'backbone'):
        return model.model.backbone
    elif hasattr(model, 'backbone'):
        return model.backbone
    elif hasattr(model, 'model'):
        return model.model
    else:
        return model


def load_model():
    """Load the model using your existing API"""
    # Import your model loading function with corrected path
    from Zhou_XJTLU_task1_1 import load_model as load_model_api
    
    # Load model using your existing API
    model_wrapper = load_model_api()
    return model_wrapper


def check_complexity(
    dummy_file: str,
    device_ids: List[str],
    submission_name: str,
    submission_index: int
):
    """Check model complexity for all devices"""
    
    # Load model
    model = load_model()

    # Create dummy input for each device
    file_paths = [dummy_file for _ in device_ids]
    inputs = load_inputs(file_paths, device_ids, model)

    # Track per-device MACs and Params
    per_device = {}
    max_macs = 0
    max_params = 0

    print("\n📊 Model Complexity Check (per device)")
    for input_tensor, device_id in zip(inputs, device_ids):
        submodel = get_model_for_device(model, device_id)
        input_shape = input_tensor.shape

        macs, params_bytes = get_torch_macs_memory(submodel, input_size=input_shape)
        max_macs = max(max_macs, macs)
        max_params = max(max_params, params_bytes)

        per_device[device_id] = {
            "MACs": macs,
            "Params": params_bytes
        }

        macs_ok = macs <= MAX_MACS
        params_ok = params_bytes <= MAX_PARAMS_MEMORY
        status = "✅" if macs_ok and params_ok else "❌"

        print(f"{device_id:>3} | MACs: {macs:>10,} | Params Bytes: {params_bytes:>8,} bytes | {status}")

    # Save JSON summary
    output_dir = os.path.join("predictions", f"{submission_name}_{submission_index}")
    os.makedirs(output_dir, exist_ok=True)

    complexity_path = os.path.join(output_dir, "complexity.json")
    complexity_data = {
        "per_device": per_device,
        "max_MACs": max_macs,
        "max_Params": max_params
    }

    with open(complexity_path, "w") as f:
        json.dump(complexity_data, f, indent=2)

    print(f"\nSaved complexity info to: {complexity_path}")
    
    # Print overall summary
    print(f"\n📋 Overall Summary:")
    print(f"Max MACs: {max_macs:,} ({'PASS' if max_macs <= MAX_MACS else 'FAIL'})")
    print(f"Max Params: {max_params:,} bytes ({max_params/1024:.1f} KB) ({'PASS' if max_params <= MAX_PARAMS_MEMORY else 'FAIL'})")
    
    if max_macs <= MAX_MACS and max_params <= MAX_PARAMS_MEMORY:
        print("🎉 All constraints satisfied!")
    else:
        print("⚠️  Some constraints not met - optimization needed")


def create_dummy_wav(file_path: str, duration: float = 10.0, sample_rate: int = 32000):
    """Create a dummy WAV file for testing"""
    try:
        import soundfile as sf
        # Generate dummy audio data
        samples = int(duration * sample_rate)
        dummy_audio = np.random.randn(samples).astype(np.float32) * 0.1
        # Save as WAV file
        sf.write(file_path, dummy_audio, sample_rate)
        print(f"Created dummy audio file: {file_path}")
    except ImportError:
        print("Warning: soundfile not available. Creating a placeholder file.")
        # Create a minimal placeholder file
        with open(file_path, 'w') as f:
            f.write("dummy audio file placeholder")


def main():
    parser = argparse.ArgumentParser(description="Check model complexity for all devices using a dummy file.")
    parser.add_argument("--submission_name", type=str, default="Zhou_XJTLU_task1",
                        help="Name of the submission package, e.g., Zhou_XJTLU_task1")
    parser.add_argument("--submission_index", type=int, default=1,
                        help="Index of the submission variant, e.g., 1 for Zhou_XJTLU_task1_1")
    parser.add_argument("--dummy_file", type=str, default="dummy_test.wav",
                        help="Path to dummy audio file (will be created if not exists)")

    args = parser.parse_args()

    # Device IDs to evaluate
    device_ids = ['a', 'b', 'c'] + [f's{i}' for i in range(1, 11)]

    # Create dummy file if it doesn't exist
    if not os.path.exists(args.dummy_file):
        create_dummy_wav(args.dummy_file)

    try:
        check_complexity(
            dummy_file=args.dummy_file,
            device_ids=device_ids,
            submission_name=args.submission_name,
            submission_index=args.submission_index
        )
    except Exception as e:
        print(f"Error during complexity check: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()