import torch
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from model.backbones import TFSepNet
from model.lit_asc import LitAcousticSceneClassificationSystem
from data.data_module import DCASEDataModule
from util import CpMel
import os
import copy

def dequantize_int8_tensor(quantized_tensor, scale, zero_point):
    """
    Dequantize int8 tensor
    """
    return (quantized_tensor.float() - zero_point) * scale

def dequantize_qint8_tensor(quantized_tensor):
    """
    Dequantize qint8 tensor (PyTorch quantized tensor)
    """
    return quantized_tensor.dequantize()

class QuantizedTFSepNet(torch.nn.Module):
    """
    Wrapper class for loading quantized models
    """
    def __init__(self, original_model, quantized_state_dict, quantization_info):
        super().__init__()
        self.original_model = original_model
        self.quantization_info = quantization_info
        
        # Load quantized weights
        self.load_quantized_weights(quantized_state_dict)
    
    def load_quantized_weights(self, quantized_state_dict):
        """Load quantized weights and dequantize all parameters"""
        print("🔄 Loading quantized weights and dequantizing all parameters...")
        
        # Get original model's state_dict
        original_state_dict = self.original_model.state_dict()
        new_state_dict = copy.deepcopy(original_state_dict)
        
        loaded_count = 0
        dequantized_weight_count = 0
        dequantized_bias_count = 0
        skipped_count = 0
        
        for param_name, param in quantized_state_dict.items():
            if isinstance(param, torch.Tensor):
                # Skip quantization metadata
                if any(keyword in param_name for keyword in ['scale', 'zero_point', 'best_configure']):
                    continue
                
                # Check if parameter exists in original model
                if param_name in original_state_dict:
                    try:
                        if param.dtype == torch.qint8:
                            # Dequantize qint8 weights
                            dequantized_param = dequantize_qint8_tensor(param)
                            new_state_dict[param_name] = dequantized_param
                            dequantized_weight_count += 1
                            print(f"  🔄 Dequantized weight {param_name}: {param.shape} -> {dequantized_param.shape}")
                            
                        elif 'bias' in param_name and param.dtype == torch.int8:
                            # Dequantize int8 bias
                            if param_name in self.quantization_info:
                                quant_info = self.quantization_info[param_name]
                                scale = quant_info['scale']
                                zero_point = quant_info['zero_point']
                                
                                dequantized_bias = dequantize_int8_tensor(param, scale, zero_point)
                                new_state_dict[param_name] = dequantized_bias
                                dequantized_bias_count += 1
                                print(f"  🔄 Dequantized bias {param_name}: {param.shape} -> {dequantized_bias.shape}")
                            else:
                                print(f"  ⚠️ Quantization info not found: {param_name}")
                                # Simple conversion to float
                                new_state_dict[param_name] = param.float()
                                
                        elif param.dtype in [torch.float32, torch.float16]:
                            # Directly load float parameters
                            if original_state_dict[param_name].shape == param.shape:
                                new_state_dict[param_name] = param
                                loaded_count += 1
                            else:
                                print(f"  ⚠️ Shape mismatch: {param_name}")
                                
                        else:
                            print(f"  ⚠️ Unknown parameter type: {param_name}, dtype: {param.dtype}")
                            
                    except Exception as e:
                        print(f"  ❌ Failed to process parameter {param_name}: {e}")
                        
                else:
                    # Parameter doesn't exist in original model, might be bias parameter
                    if 'bias' in param_name:
                        print(f"  ⚠️ Bias parameter not in original model: {param_name}")
                        # Try to add bias parameter to model
                        if self.try_add_bias_to_model(param_name, param):
                            if param.dtype == torch.int8 and param_name in self.quantization_info:
                                quant_info = self.quantization_info[param_name]
                                scale = quant_info['scale']
                                zero_point = quant_info['zero_point']
                                dequantized_bias = dequantize_int8_tensor(param, scale, zero_point)
                                new_state_dict[param_name] = dequantized_bias
                                dequantized_bias_count += 1
                            else:
                                new_state_dict[param_name] = param.float()
                    else:
                        skipped_count += 1
                        print(f"  ⚠️ Skipped parameter: {param_name}")
        
        # Load to original model
        try:
            missing_keys, unexpected_keys = self.original_model.load_state_dict(new_state_dict, strict=False)
            
            print(f"✅ Successfully loaded {loaded_count} direct parameters")
            print(f"✅ Successfully dequantized {dequantized_weight_count} weight parameters")
            print(f"✅ Successfully dequantized {dequantized_bias_count} bias parameters")
            print(f"⚠️ Skipped {skipped_count} parameters")
            
            if missing_keys:
                print(f"⚠️ Missing parameters: {len(missing_keys)} items")
                for key in missing_keys[:5]:  # Show only first 5
                    print(f"    {key}")
                    
            if unexpected_keys:
                print(f"⚠️ Unexpected parameters: {len(unexpected_keys)} items")
                for key in unexpected_keys[:5]:  # Show only first 5
                    print(f"    {key}")
                    
        except Exception as e:
            print(f"❌ Failed to load state_dict: {e}")
            raise
    
    def try_add_bias_to_model(self, param_name, param):
        """
        Try to add bias parameter to corresponding layer in model
        """
        try:
            # Parse parameter path, e.g., "conv_layers.0._conv.bias"
            parts = param_name.split('.')
            if parts[-1] == 'bias' and parts[-2] == '_conv':
                # Locate corresponding convolution layer
                module = self.original_model
                for part in parts[:-2]:  # Exclude '_conv.bias'
                    if hasattr(module, part):
                        module = getattr(module, part)
                    else:
                        return False
                
                # Get convolution layer
                if hasattr(module, '_conv'):
                    conv_layer = module._conv
                    if hasattr(conv_layer, 'bias') and conv_layer.bias is None:
                        # Add bias parameter
                        conv_layer.bias = torch.nn.Parameter(torch.zeros(param.shape))
                        print(f"    ✅ Added bias parameter: {param_name}")
                        return True
            return False
        except Exception as e:
            print(f"    ❌ Failed to add bias: {e}")
            return False
    
    def forward(self, x):
        return self.original_model(x)

def load_quantized_model(quantized_model_path):
    """
    Load quantized model and create a model for inference
    """
    print(f"📦 Loading quantized model: {quantized_model_path}")
    
    # Load saved data
    saved_data = torch.load(quantized_model_path, map_location='cpu')
    
    if isinstance(saved_data, dict) and 'model_state_dict' in saved_data:
        # New format: dictionary containing quantization info
        quantized_state_dict = saved_data['model_state_dict']
        quantization_info = saved_data.get('quantization_info', {})
        print(f"✅ Loaded new format quantized model with {len(quantization_info)} quantized biases")
    else:
        # Old format: direct OrderedDict
        quantized_state_dict = saved_data
        quantization_info = {}
        print("✅ Loaded old format quantized model")
    
    print(f"Model contains {len(quantized_state_dict)} parameters")
    
    # Create original TFSepNet model with bias allowed
    original_model = TFSepNet(
        in_channels=1,
        num_classes=10,
        base_channels=64,
        depth=17,
        load_classifier=True,
        freeze_backbone=False
    )
    
    # Create quantized model wrapper
    quantized_model = QuantizedTFSepNet(
        original_model=original_model,
        quantized_state_dict=quantized_state_dict,
        quantization_info=quantization_info
    )
    
    return quantized_model

def create_lightning_model_from_quantized(quantized_model):
    """
    Create Lightning model from quantized model
    """
    print("🔧 Creating Lightning model...")
    
    # Create spectrum extractor
    spec_extractor = CpMel(n_mels=512)
    
    # Create Lightning model
    lightning_model = LitAcousticSceneClassificationSystem(
        backbone=quantized_model,
        spec_extractor=spec_extractor,
        class_label="scene",
        domain_label="device",
        data_augmentation={
            "mix_up": None,
            "mix_style": None,
            "spec_aug": None,
            "dir_aug": None
        }
    )
    
    return lightning_model

def test_quantized_model():
    """
    Main function to test quantized model
    """
    print("🚀 Starting quantized model test")
    print("=" * 80)
    
    # Set random seed
    torch.manual_seed(42)
    L.seed_everything(42)
    
    # Quantized model path - test with bias version first
    quantized_model_path = "ckpts/best_model_quantized_bias.pt"
    
    # Check if file exists
    if not os.path.exists(quantized_model_path):
        print(f"❌ Error: Quantized model file not found {quantized_model_path}")
        return
    
    try:
        # 1. Load quantized model
        print("📋 Step 1: Loading quantized model")
        quantized_backbone = load_quantized_model(quantized_model_path)
        
        # 2. Create Lightning model
        print("\n📋 Step 2: Creating Lightning model")
        model = create_lightning_model_from_quantized(quantized_backbone)
        
        # 3. Create data module
        print("\n📋 Step 3: Creating data module")
        data_module = DCASEDataModule(
            meta_dir="data/meta_dcase_2024",
            audio_dir="../TAU-urban-acoustic-scenes-2022-mobile-development/TAU-urban-acoustic-scenes-2022-mobile-development",
            batch_size=256,
            num_workers=8,
            pin_memory=True,
            sampling_rate=32000,
            test_subset="test"
        )
        
        # 4. Create TensorBoard logger
        logger = TensorBoardLogger(
            save_dir="log",
            name="tfsepnet_quantized_test"
        )
        
        # 5. Create trainer
        print("\n📋 Step 4: Creating trainer")
        trainer = L.Trainer(
            devices=1,
            logger=logger,
            accelerator="auto",
            enable_checkpointing=False,
            enable_progress_bar=True
        )
        
        # 6. Print model info
        print("\n" + "=" * 50)
        print("📊 Quantized model test info:")
        print(f"Quantized model path: {quantized_model_path}")
        print(f"Model parameter count: {sum(p.numel() for p in model.parameters()):,}")
        print(f"Test dataset: test")
        
        # Count data types in model
        dtype_count = {}
        for name, param in model.named_parameters():
            dtype = str(param.dtype)
            dtype_count[dtype] = dtype_count.get(dtype, 0) + 1
        
        print("Parameter data type distribution:")
        for dtype, count in dtype_count.items():
            print(f"  {dtype}: {count} parameters")
        print("=" * 50)
        
        # 7. Run test
        print("\n📋 Step 5: Starting test...")
        test_results = trainer.test(model, data_module)
        
        print("\n🎉 Test completed!")
        
        # 8. Print test results
        if test_results:
            print("\n📈 Test results:")
            for key, value in test_results[0].items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value}")
        
        # # 9. Perform simple inference test
        # print("\n📋 Step 6: Performing inference validation...")
        # test_inference(model, data_module)
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

def test_inference(model, data_module):
    """
    Simple inference test
    """
    try:
        print("🧪 Performing inference validation...")
        
        # Prepare data
        data_module.setup("test")
        test_dataloader = data_module.test_dataloader()
        
        # Get one batch of data
        batch = next(iter(test_dataloader))
        
        # Check batch type and structure
        print(f"Batch type: {type(batch)}")
        if isinstance(batch, dict):
            print(f"Batch keys: {list(batch.keys())}")
            audio_data = batch['audio']
            labels = batch.get('scene_label', None)
            # Create list format for Lightning model
            batch_for_model = [audio_data, labels]
        elif isinstance(batch, (list, tuple)):
            print(f"Batch is list/tuple with {len(batch)} elements")
            if len(batch) >= 2:
                audio_data = batch[0]
                labels = batch[1]
                print(f"Element types: {[type(item) for item in batch]}")
                # Use original batch format
                batch_for_model = batch
            else:
                print("❌ Unexpected batch structure")
                return
        else:
            print(f"❌ Unknown batch type: {type(batch)}")
            return
        
        print(f"Test batch shape: {audio_data.shape}")
        print(f"Labels shape: {labels.shape if hasattr(labels, 'shape') else type(labels)}")
        
        # Set model to evaluation mode
        model.eval()
        
        with torch.no_grad():
            print("🔄 Running model inference...")
            
            # Call the Lightning model's validation_step with proper batch format
            outputs = model.validation_step(batch_for_model, 0)
            
            print(f"Output type: {type(outputs)}")
            if isinstance(outputs, dict):
                print(f"Output keys: {list(outputs.keys())}")
                if 'logits' in outputs:
                    logits = outputs['logits']
                    print(f"Output logits shape: {logits.shape}")
                    predicted_classes = torch.argmax(logits, dim=1)
                    print(f"Predicted classes (first 5): {predicted_classes[:5]}")
                    print(f"True classes (first 5): {labels[:5]}")
                elif 'scene_logits' in outputs:
                    logits = outputs['scene_logits']
                    print(f"Scene logits shape: {logits.shape}")
                    predicted_classes = torch.argmax(logits, dim=1)
                    print(f"Predicted classes (first 5): {predicted_classes[:5]}")
                    print(f"True classes (first 5): {labels[:5]}")
                else:
                    # Print all output keys to debug
                    for key, value in outputs.items():
                        if isinstance(value, torch.Tensor):
                            print(f"  {key}: shape {value.shape}")
                        else:
                            print(f"  {key}: {type(value)}")
            else:
                print(f"Direct output shape: {outputs.shape if hasattr(outputs, 'shape') else 'No shape attribute'}")
        
        print("✅ Inference validation successful!")
        
    except Exception as e:
        print(f"⚠️ Inference validation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_quantized_model()