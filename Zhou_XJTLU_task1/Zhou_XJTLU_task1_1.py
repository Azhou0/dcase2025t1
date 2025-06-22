import torch
import torch.nn.functional as F
import librosa
import numpy as np
from typing import List, Tuple
import os

# Import your model components
from model.backbones import TFSepNet
from model.lit_asc import LitAcousticSceneClassificationSystem
from util import CpMel
from util.static_variable import unique_labels

class ModelWrapper:
    """Model wrapper for inference"""
    
    def __init__(self, model_path: str = None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.spec_extractor = CpMel(n_mels=512)
        self.model = None
        self.class_order = unique_labels['scene']  # Scene class order
        
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path):
        backbone = TFSepNet()
        
        if 'quantized' in model_path:
            # Load quantized model and dequantize for compatibility
            checkpoint = torch.load(model_path, map_location='cpu')
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            
            # Filter and dequantize parameters
            filtered_state_dict = {}
            for key, value in state_dict.items():
                if not any(keyword in key for keyword in ['scale', 'zero_point', 'best_configure']):
                    if isinstance(value, torch.Tensor):
                        # Dequantize qint8 weights to float32
                        if value.dtype == torch.qint8:
                            filtered_state_dict[key] = value.dequantize().float()
                        # Convert int8 bias to float32
                        elif value.dtype == torch.int8:
                            filtered_state_dict[key] = value.float()
                        else:
                            filtered_state_dict[key] = value
            
            backbone.load_state_dict(filtered_state_dict, strict=False)
        else:
            # Regular model loading
            if model_path.endswith('.ckpt'):
                checkpoint = torch.load(model_path, map_location='cpu')
                state_dict = checkpoint['state_dict']
                # Remove 'backbone.' prefix
                backbone_state_dict = {k.replace('backbone.', ''): v for k, v in state_dict.items() 
                                     if k.startswith('backbone.')}
                backbone.load_state_dict(backbone_state_dict, strict=False)
            else:
                backbone.load_state_dict(torch.load(model_path, map_location='cpu'), strict=False)
        
        # Create Lightning model
        self.model = LitAcousticSceneClassificationSystem(
            backbone=backbone,
            spec_extractor=self.spec_extractor,
            class_label="scene",
            domain_label="device",
            data_augmentation={
                "mix_up": None,
                "mix_style": None,
                "spec_aug": None,
                "dir_aug": None
            }
        )
        
        self.model.to(self.device)
        self.model.eval()
        print(f"Model loaded successfully on {self.device}")
        return self.model
    
    def preprocess_audio(self, file_path: str, sampling_rate: int = 32000) -> torch.Tensor:
        """Preprocess audio file"""
        # Load audio
        wav, _ = librosa.load(file_path, sr=sampling_rate)
        wav = torch.from_numpy(wav).float()
        
        # Add batch dimension
        wav = wav.unsqueeze(0)  # (1, T)
        
        return wav
    
    def predict_single(self, file_path: str, device_id: str = None) -> torch.Tensor:
        """Predict on a single audio file"""
        # Preprocess audio
        audio = self.preprocess_audio(file_path)
        audio = audio.to(self.device)
        
        with torch.no_grad():
            # Extract spectrogram
            if self.model.spec_extractor is not None:
                spec = self.model.spec_extractor(audio).unsqueeze(1)  # (1, 1, F, T)
            else:
                spec = audio
            
            # Model inference
            logits = self.model.backbone(spec)  # (1, num_classes)
            
        return logits.squeeze(0)  # (num_classes,)

# Global model instance
_model_wrapper = None

def load_model(model_file_path: str = None):
    global _model_wrapper
    if _model_wrapper is None:
        _model_wrapper = ModelWrapper()
    
    if model_file_path:
        _model_wrapper.load_model(model_file_path)
    elif _model_wrapper.model is None:
        # Use default quantized model path
        default_model_path = "Zhou_XJTLU_task1/ckpts/best_model_quantized_bias.pt"
        _model_wrapper.load_model(default_model_path)
    
    return _model_wrapper

def predict(file_paths: List[str], 
           device_ids: List[str], 
           model_file_path: str = None, 
           use_cuda: bool = True) -> Tuple[List[torch.Tensor], List[str]]:
    """Batch prediction API function
    
    Args:
        file_paths: List of audio file paths
        device_ids: List of device IDs (for multi-device adaptation, not used in current implementation)
        model_file_path: Model file path (optional)
        use_cuda: Whether to use CUDA
    
    Returns:
        predictions: List of prediction logits
        class_order: List of class order
    """
    # Load model
    model_wrapper = load_model(model_file_path)
    
    # Set device
    if not use_cuda:
        model_wrapper.device = torch.device('cpu')
        model_wrapper.model.to(model_wrapper.device)
    
    predictions = []
    
    print(f"Processing {len(file_paths)} audio files...")
    
    for i, file_path in enumerate(file_paths):
        if i % 100 == 0:
            print(f"Processed {i}/{len(file_paths)} files")
        
        try:
            # Get corresponding device_id (if device-specific processing is needed)
            device_id = device_ids[i] if i < len(device_ids) else None
            
            # Predict single file
            logits = model_wrapper.predict_single(file_path, device_id)
            predictions.append(logits)
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            # Return zero vector as fallback
            predictions.append(torch.zeros(len(model_wrapper.class_order)))
    
    print(f"Completed processing {len(file_paths)} files")
    
    return predictions, model_wrapper.class_order