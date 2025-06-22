import importlib
import torch
import torch.nn.functional as F
import lightning as L
import numpy as np
import pandas as pd
from typing import Dict, Optional

from lightning.pytorch.cli import OptimizerCallable, LRSchedulerCallable

import torchinfo
from util.static_variable import unique_labels
from util import ClassificationSummary
from model.backbones import _BaseBackbone
from util.lr_scheduler import exp_warmup_linear_down
from util import _SpecExtractor, ClassificationSummary, _DataAugmentation
import numpy as np
class LitAcousticSceneClassificationSystem(L.LightningModule):
    """
    Acoustic Scene Classification system based on LightningModule.
    Backbone model, data augmentation techniques and spectrogram extractor are designed to be plug-and-played.
    Backbone architecture, system complexity, classification report and confusion matrix are shown at test stage.

    Args:
        backbone (_BaseBackbone): Deep neural network backbone, e.g. cnn, transformer...
        data_augmentation (dict): A dictionary containing instances of data augmentation techniques in util/. Options: MixUp, FreqMixStyle, DeviceImpulseResponseAugmentation, SpecAugmentation. Set each to ``None`` if not use one of them.
        class_label (str): Class label. e.g. scene, device, city.
        domain_label (str): Domain label. e.g. scene, device, city.
        spec_extractor (_SpecExtractor): Spectrogram extractor used to transform 1D waveforms to 2D spectrogram. If ``None``, the input features should be 2D spectrogram.
    """

    def __init__(self,
                 backbone: _BaseBackbone,
                 data_augmentation: Dict[str, Optional[_DataAugmentation]],
                 class_label: str = "scene",
                 domain_label: str = "device",
                 spec_extractor: _SpecExtractor = None):
        super(LitAcousticSceneClassificationSystem, self).__init__()
        # Save the hyperparameters for Tensorboard visualization, 'backbone' and 'spec_extractor' are excluded.
        self.save_hyperparameters(ignore=['backbone', 'spec_extractor'])
        self.backbone = backbone
        self.data_aug = data_augmentation
        self.class_label = class_label
        self.domain_label = domain_label
        self.cla_summary = ClassificationSummary(class_label, domain_label)
        self.spec_extractor = spec_extractor

        # Save data during testing for statistical analysis
        self._test_step_outputs = {'emb': [], 'y': [], 'pred': [], 'd': []}
        # Input size of a 4D sample (1, 1, F, T), used for generating model profile.
        self._test_input_size = None

    @staticmethod
    def accuracy(logits, labels):
        pred = torch.argmax(logits, dim=1)
        acc = torch.sum(pred == labels).item() / len(labels)
        return acc, pred

    def forward(self, x):
        return self.backbone(x)

    def training_step(self, batch, batch_idx):
        # Load a batch of waveforms with size (N, X)

        x = batch[0]
        # print(f"Initial input shape: {x.shape}")
        # print(f"Input dtype: {x.dtype}")
        # Store label dices in a dict
        labels = {'scene': batch[1], 'device': batch[2], 'city': batch[3]}
        # Choose class label
        y = labels[self.class_label]
        # Instantiate data augmentations
        dir_aug = self.data_aug['dir_aug']
        mix_style = self.data_aug['mix_style']
        spec_aug = self.data_aug['spec_aug']
        mix_up = self.data_aug['mix_up']
        # Apply dir augmentation on waveform
        x = dir_aug(x, labels['device']) if dir_aug is not None else x
        # Extract spectrogram from waveform
        x = self.spec_extractor(x).unsqueeze(1) if self.spec_extractor is not None else x
        # Apply other augmentations on spectrogram
        x = mix_style(x) if mix_style is not None else x
        x = spec_aug(x) if spec_aug is not None else x
        if mix_up is not None:
            x, y = mix_up(x, y)
        # Get the predicted labels
        y_hat = self(x)
        # Calculate the loss and accuracy
        if mix_up is not None:
            pred = torch.argmax(y_hat, dim=1)
            train_loss = mix_up.lam * F.cross_entropy(y_hat, y[0]) + (1 - mix_up.lam) * F.cross_entropy(
                y_hat, y[1])
            corrects = (mix_up.lam * torch.sum(pred == y[0]) + (1 - mix_up.lam) * torch.sum(
                pred == y[1]))
            train_acc = corrects.item() / len(x)
        else:
            train_loss = F.cross_entropy(y_hat, y)
            train_acc, _ = self.accuracy(y_hat, y)
        # Log for each epoch
        self.log_dict({'train_loss': train_loss, 'train_acc': train_acc}, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return train_loss

    def validation_step(self, batch, batch_idx):
        x = batch[0]
        labels = {'scene': batch[1], 'device': batch[2], 'city': batch[3]}
        y = labels[self.class_label]
        x = self.spec_extractor(x).unsqueeze(1) if self.spec_extractor is not None else x.unsqueeze(1)
        y_hat = self(x)
        val_loss = F.cross_entropy(y_hat, y)
        val_acc, _ = self.accuracy(y_hat, y)
        self.log_dict({'val_loss': val_loss, 'val_acc': val_acc}, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return val_acc

    def test_step(self, batch, batch_idx):
        x = batch[0]
        labels = {'scene': batch[1], 'device': batch[2], 'city': batch[3]}
        y = labels[self.class_label]
        d = labels[self.domain_label]
        x = self.spec_extractor(x).unsqueeze(1) if self.spec_extractor is not None else x.unsqueeze(1)
        # Get the input size of feature for measuring model profile
        self._test_input_size = (1, 1, x.size(-2), x.size(-1))
        y_hat = self(x)
        test_loss = F.cross_entropy(y_hat, y)
        test_acc, pred = self.accuracy(y_hat, y)
        self.log_dict({'test_loss': test_loss, 'test_acc': test_acc})

        self._test_step_outputs['y'] += y.cpu().numpy().tolist()
        self._test_step_outputs['pred'] += pred.cpu().numpy().tolist()
        self._test_step_outputs['d'] += d.cpu().numpy().tolist()
        return test_acc

    def on_test_epoch_end(self):
        tensorboard = self.logger.experiment
        # Summary the model profile
        print("\n Model Profile:")
        model_profile = torchinfo.summary(self.backbone, input_size=self._test_input_size)
        macc = model_profile.total_mult_adds
        params = model_profile.total_params
        print('MACC:\t \t %.6f' % (macc / 1e6), 'M')
        print('Params:\t \t %.3f' % (params / 1e3), 'K\n')
        # Convert the summary to string
        model_summary = str(model_profile)
        model_summary += f'\n MACC:\t \t {macc / 1e6:.3f}M'
        model_summary += f'\n Params:\t \t {params / 1e3:.3f}K\n'
        model_summary = model_summary.replace('\n', '<br/>').replace(' ', '&nbsp;').replace('\t', '&emsp;')
        tensorboard.add_text('model_summary', model_summary)
        # Generate a classification report table
        tab_report = self.cla_summary.get_table_report(self._test_step_outputs)
        tensorboard.add_text('classification_report', tab_report)
        # Generate an confusion matrix figure
        cm = self.cla_summary.get_confusion_matrix(self._test_step_outputs)
        tensorboard.add_figure('confusion_matrix', cm)
        import numpy as np
        y_all = np.array(self._test_step_outputs['y'])       # Real scene labels
        pred_all = np.array(self._test_step_outputs['pred'])   # Predicted scene labels
        d_all = np.array(self._test_step_outputs['d'])         # Device labels

        # Calculate and record accuracy for each device
        unique_devices = np.unique(d_all)
        for dev in unique_devices:
            idx = np.where(d_all == dev)[0]
            if len(idx) > 0:
                acc_dev = np.mean(y_all[idx] == pred_all[idx])
                tensorboard.add_scalar(f'test/acc_device_{dev}', acc_dev)
                print(f'Device {dev}: acc = {acc_dev:.3f}')

        # Calculate and record accuracy for each scene
        unique_scenes = np.unique(y_all)
        for scene in unique_scenes:
            idx = np.where(y_all == scene)[0]
            if len(idx) > 0:
                acc_scene = np.mean(y_all[idx] == pred_all[idx])
                tensorboard.add_scalar(f'test/acc_scene_{scene}', acc_scene)
                print(f'Scene {scene}: acc = {acc_scene:.3f}')
        # Clear accumulated data (if needed for subsequent testing)
        self._test_step_outputs = {'y': [], 'pred': [], 'd': []}

    def predict_step(self, batch):
        x = batch[0]
        x = self.spec_extractor(x).unsqueeze(1) if self.spec_extractor is not None else x.unsqueeze(1)
        return self(x)


class LitAscWithKnowledgeDistillation(LitAcousticSceneClassificationSystem):
    """
    ASC system with knowledge distillation.

    Args:
        temperature (float): A higher temperature indicates a softer distribution of pseudo-probabilities.
        kd_lambda (float): Weight to control the balance between kl loss and label loss.
        logits_index (int): Index of the logits in Dataset, as multiple logits may be used during training.
    """
    def __init__(self, temperature: float, kd_lambda: float, logits_index: int = -1, **kwargs):
        super(LitAscWithKnowledgeDistillation, self).__init__(**kwargs)
        self.temperature = temperature
        self.kd_lambda = kd_lambda
        self.logits_index = logits_index
        # KL Divergence loss for soft targets
        self.kl_div_loss = torch.nn.KLDivLoss(log_target=True)

    def training_step(self, batch, batch_idx):
        x = batch[0]
        # Store label dices in a dict
        labels = {'scene': batch[1], 'device': batch[2], 'city': batch[3]}
        # Load soft labels
        teacher_logits = batch[self.logits_index]
        y_soft = F.log_softmax(teacher_logits / self.temperature, dim=-1)
        # Load hard labels
        y = labels[self.class_label]
        # Instantiate data augmentations
        dir_aug = self.data_aug['dir_aug']
        mix_style = self.data_aug['mix_style']
        spec_aug = self.data_aug['spec_aug']
        mix_up = self.data_aug['mix_up']
        # Apply dir augmentation on waveform
        x = dir_aug(x, labels['device']) if dir_aug is not None else x
        # Extract spectrogram from waveform
        x = self.spec_extractor(x).unsqueeze(1) if self.spec_extractor is not None else x.unsqueeze(1)
        # Apply other augmentations on spectrogram
        x = mix_style(x) if mix_style is not None else x
        x = spec_aug(x) if spec_aug is not None else x
        if mix_up is not None:
            x, y, y_soft = mix_up(x, y, y_soft)
        # Get the predicted labels
        y_hat = self(x)
        # Temperature adjusted probabilities of teacher and student
        with torch.cuda.amp.autocast():
            y_hat_soft = F.log_softmax(y_hat / self.temperature, dim=-1)
        # Calculate the loss and accuracy
        if mix_up is not None:
            label_loss = mix_up.lam * F.cross_entropy(y_hat, y[0]) + (1 - mix_up.lam) * F.cross_entropy(y_hat, y[1])
            kd_loss = mix_up.lam * self.kl_div_loss(y_hat_soft, y_soft[0]) + (1 - mix_up.lam) * self.kl_div_loss(y_hat_soft, y_soft[1])
        else:
            label_loss = F.cross_entropy(y_hat, y)
            kd_loss = self.kl_div_loss(y_hat_soft, y_soft)
        kd_loss = kd_loss * (self.temperature ** 2)
        loss = self.kd_lambda * label_loss + (1 - self.kd_lambda) * kd_loss
        self.log_dict({'loss': loss, 'label_loss': label_loss, 'kd_loss': kd_loss}, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss


class LitAscWithWarmupLinearDownScheduler(LitAcousticSceneClassificationSystem):
    """
    ASC system with warmup-linear-down scheduler.
    """
    def __init__(self, optimizer: OptimizerCallable, warmup_len=4, down_len=26, min_lr=0.005, **kwargs):
        super(LitAscWithWarmupLinearDownScheduler, self).__init__(**kwargs)
        self.optimizer = optimizer
        self.warmup_len = warmup_len
        self.down_len = down_len
        self.min_lr = min_lr

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters())
        schedule_lambda = exp_warmup_linear_down(self.warmup_len, self.down_len, self.warmup_len, self.min_lr)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule_lambda)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


class LitAscWithTwoSchedulers(LitAcousticSceneClassificationSystem):
    """
    ASC system with two customized schedulers.

    Directly instantiate multiple schedulers from the yaml config file.
    For more details: https://lightning.ai/docs/pytorch/stable/cli/lightning_cli_advanced_3.html
    """
    def __init__(self, optimizer: OptimizerCallable, scheduler1: LRSchedulerCallable, scheduler2: LRSchedulerCallable, milestones, **kwargs):
        super(LitAscWithTwoSchedulers, self).__init__(**kwargs)
        self.optimizer = optimizer
        self.scheduler1 = scheduler1
        self.scheduler2 = scheduler2
        self.milestones = milestones

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters())
        scheduler1 = self.scheduler1(optimizer)
        scheduler2 = self.scheduler2(optimizer)
        scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [scheduler1, scheduler2], self.milestones)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


class LitAscWithThreeSchedulers(LitAcousticSceneClassificationSystem):
    """
    ASC system with three customized schedulers.

    Directly instantiate multiple schedulers from the yaml config file.
    For more details: https://lightning.ai/docs/pytorch/stable/cli/lightning_cli_advanced_3.html
    """
    def __init__(self, optimizer: OptimizerCallable, scheduler1: LRSchedulerCallable, scheduler2: LRSchedulerCallable, scheduler3: LRSchedulerCallable, milestones, **kwargs):
        super(LitAscWithThreeSchedulers, self).__init__(**kwargs)
        self.optimizer = optimizer
        self.scheduler1 = scheduler1
        self.scheduler2 = scheduler2
        self.scheduler3 = scheduler3
        self.milestones = milestones

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters())
        scheduler1 = self.scheduler1(optimizer)
        scheduler2 = self.scheduler2(optimizer)
        scheduler3 = self.scheduler3(optimizer)
        scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [scheduler1, scheduler2, scheduler3], self.milestones)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

class LitMultiDeviceInference(L.LightningModule):
    """
    Multi-device model inference system, supporting both testing and inference modes
    """
    
    def __init__(self, 
                 device_model_paths: Dict[str, str],
                 backbone_config: dict,
                 spec_extractor_config: dict = None,
                 class_label: str = "scene",
                 domain_label: str = "device"):
        super().__init__()
        
        self.device_model_paths = device_model_paths
        self.class_label = class_label
        self.domain_label = domain_label
        self.cla_summary = ClassificationSummary(class_label, domain_label)
        
        # Initialize spectrogram extractor
        if spec_extractor_config:
            target_class = spec_extractor_config['_target_']
            module_name, class_name = target_class.rsplit('.', 1)
            module = importlib.import_module(module_name)
            spec_class = getattr(module, class_name)
            config_copy = spec_extractor_config.copy()
            config_copy.pop('_target_', None)
            self.spec_extractor = spec_class(**config_copy)
        else:
            self.spec_extractor = None
            
        # Store configuration, load models later
        self.backbone_config = backbone_config
        self.device_models = {}
        
        # Save test and inference results
        self._test_step_outputs = {'emb': [], 'y': [], 'pred': [], 'd': []}
        self._predict_step_outputs = {'filename': [], 'predictions': [], 'device_ids': [], 'device_names': []}
        
    def on_test_start(self):
        """Load models at the start of testing"""
        if not self.device_models:
            self._load_device_models()
    
    def on_predict_start(self):
        """Load models at the start of inference"""
        if not self.device_models:
            self._load_device_models()
            
    def _load_device_models(self):
        """Manually create backbone and load backbone.* weights (including classifier) from checkpoint"""
        print("🔄 Loading device-specific models (manual state_dict)...")
        current_device = self.device

        for device, ckpt_path in self.device_model_paths.items():
            print(f"📦 Loading checkpoint for device '{device}': {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location=current_device)
            state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt

            # 1️⃣ Construct backbone
            cfg = self.backbone_config.copy()
            cls_path = cfg.pop("_target_")
            mod_name, cls_name = cls_path.rsplit(".", 1)
            backbone_cls = getattr(importlib.import_module(mod_name), cls_name)
            backbone = backbone_cls(**cfg)

            # 2️⃣ Filter keys
            backbone_state = {k.split("backbone.", 1)[1]: v
                            for k, v in state_dict.items()
                            if k.startswith(("backbone.", "module.backbone."))}

            missing = backbone.load_state_dict(backbone_state, strict=False)
            has_cls = any("classifier" in k for k in backbone_state)
            print(f"   ➜  loaded {len(backbone_state)} params "
                f"(classifier loaded: {has_cls})  missing: {len(missing.missing_keys)}")

            backbone.to(current_device).eval()
            self.device_models[device] = backbone
            print(f"✅ Device '{device}' model ready.\n")

    
    def forward(self, x, device_ids):
        """Forward pass based on device IDs to select the corresponding model"""
        batch_size = x.size(0)
        device = x.device
        
        if not self.device_models:
            self._load_device_models()
        
        unique_devices = torch.unique(device_ids)
        outputs = []
        device_masks = []
        
        for device_id in unique_devices:
            device_mask = (device_ids == device_id)
            device_x = x[device_mask]
            
            if device_x.size(0) == 0:
                continue
                
            device_name = unique_labels['device'][device_id.item()]
            
            if device_name in self.device_models:
                model = self.device_models[device_name]
            elif 'unknown' in self.device_models:
                model = self.device_models['unknown']
            else:
                model = list(self.device_models.values())[0]
            
            if next(model.parameters()).device != device:
                model = model.to(device)
            
            with torch.no_grad():
                device_output = model(device_x)
            
            outputs.append(device_output)
            device_masks.append(device_mask)
        
        if outputs:
            output_dim = outputs[0].size(-1)
            final_output = torch.zeros(batch_size, output_dim, device=device)
            for mask, output in zip(device_masks, outputs):
                final_output[mask] = output
        else:
            final_output = torch.zeros(batch_size, 10, device=device)
            
        return final_output
    
    def predict_step(self, batch, batch_idx):
        """
        Inference step - process inference data, automatically recognize data format
        """
        # Check batch format to determine data type
        if len(batch) == 4:  # InferenceDataset format: (wav, filename, device_label, device_id)
            x, filenames, device_labels, device_ids_str = batch
            device_ids = device_labels  # Use encoded device IDs
            print(f"Processing inference batch with {len(filenames)} samples")
        elif len(batch) == 2:  # AudioDataset format: (wav, filename)
            x, filenames = batch
            device_ids = torch.zeros(x.size(0), dtype=torch.long)  # Default device 0
            device_ids_str = ['unknown'] * x.size(0)
            print(f"Processing regular prediction batch with {len(filenames)} samples")
        else:
            raise ValueError(f"Unexpected batch format with {len(batch)} elements")
        
        # Extract spectrogram features
        if self.spec_extractor is not None:
            x = self.spec_extractor(x).unsqueeze(1)
        else:
            x = x.unsqueeze(1)
        
        # Ensure models are loaded
        if not self.device_models:
            self._load_device_models()
        
        # Inference
        with torch.no_grad():
            logits = self.forward(x, device_ids)
            probabilities = F.softmax(logits, dim=1)
        
        # Save results
        if len(batch) == 4:  # Inference format
            self._predict_step_outputs['filename'].extend(filenames)
            self._predict_step_outputs['predictions'].extend(probabilities.cpu().numpy())
            self._predict_step_outputs['device_ids'].extend(device_ids.cpu().numpy())
            self._predict_step_outputs['device_names'].extend(device_ids_str)
        
        return probabilities
    
    def on_predict_epoch_end(self):
        """
        Save results to CSV file after inference is complete
        """
        if not self._predict_step_outputs['filename']:
            print("No prediction results to save.")
            return
        
        # Prepare data
        filenames = self._predict_step_outputs['filename']
        predictions = np.array(self._predict_step_outputs['predictions'])
        device_names = self._predict_step_outputs['device_names']
        
        # Scene label names
        scene_labels = unique_labels['scene']
        
        # Create official submission format results
        submission_results = []
        
        for i, filename in enumerate(filenames):
            pred_idx = np.argmax(predictions[i])
            pred_scene = scene_labels[pred_idx]
            
            # Official submission format (only includes filename and scene_label)
            submission_row = [filename, pred_scene]
            
            # Add probability for each scene
            for j, scene in enumerate(scene_labels):
                submission_row.append(f"{predictions[i][j]:.4f}")
            
            submission_results.append(submission_row)
        
        # Save official submission format
        submission_header = ['filename', 'scene_label'] + scene_labels
        submission_df = pd.DataFrame(submission_results, columns=submission_header)
        submission_path = "submission.csv"
        submission_df.to_csv(submission_path, sep='\t', index=False)
        print(f"Official submission file saved to: {submission_path}")
        
        # Statistics of predictions by device
        print("\nPrediction statistics by device:")
        device_stats = {}
        for i, device_name in enumerate(device_names):
            if device_name not in device_stats:
                device_stats[device_name] = {'count': 0, 'scenes': {}}
            
            device_stats[device_name]['count'] += 1
            pred_scene = scene_labels[np.argmax(predictions[i])]
            if pred_scene not in device_stats[device_name]['scenes']:
                device_stats[device_name]['scenes'][pred_scene] = 0
            device_stats[device_name]['scenes'][pred_scene] += 1
        
        for device, stats in device_stats.items():
            print(f"{device}: {stats['count']} samples")
            for scene, count in stats['scenes'].items():
                percentage = count / stats['count'] * 100
                print(f"  {scene}: {count} ({percentage:.1f}%)")
        
        # Clear results
        self._predict_step_outputs = {'filename': [], 'predictions': [], 'device_ids': [], 'device_names': []}

    @staticmethod
    def accuracy(logits, labels):
        """Static method to calculate accuracy"""
        with torch.no_grad():
            predicted = torch.argmax(logits, dim=1)
            correct = (predicted == labels).sum().item()
            accuracy = correct / labels.size(0)
        return accuracy, predicted

    def test_step(self, batch, batch_idx):
        """Test step - process data with labels"""
        x = batch[0]
        labels = {'scene': batch[1], 'device': batch[2], 'city': batch[3]}
        y = labels[self.class_label]
        d = labels[self.domain_label]
        
        if self.spec_extractor is not None:
            x = self.spec_extractor(x).unsqueeze(1)
        else:
            x = x.unsqueeze(1)
        
        y_hat = self.forward(x, d)
        test_loss = F.cross_entropy(y_hat, y)
        test_acc, pred = self.accuracy(y_hat, y)
        
        self.log_dict({'test_loss': test_loss, 'test_acc': test_acc})
        
        self._test_step_outputs['y'] += y.cpu().numpy().tolist()
        self._test_step_outputs['pred'] += pred.cpu().numpy().tolist()
        self._test_step_outputs['d'] += d.cpu().numpy().tolist()
        
        return test_acc

    def on_test_epoch_end(self):
        """Generate report at the end of testing"""
        if not self._test_step_outputs['y']:
            return
            
        tensorboard = self.logger.experiment
        
        try:
            tab_report = self.cla_summary.get_table_report(self._test_step_outputs)
            tensorboard.add_text('classification_report', tab_report)
        except Exception as e:
            print(f"Error generating classification report: {e}")
        
        try:
            cm = self.cla_summary.get_confusion_matrix(self._test_step_outputs)
            tensorboard.add_figure('confusion_matrix', cm)
        except Exception as e:
            print(f"Error generating confusion matrix: {e}")
        
        y_all = np.array(self._test_step_outputs['y'])
        pred_all = np.array(self._test_step_outputs['pred'])
        d_all = np.array(self._test_step_outputs['d'])
        
        print("\nPer-device accuracy:")
        device_accuracies = {}
        
        for device_id in np.unique(d_all):
            device_mask = (d_all == device_id)
            device_y = y_all[device_mask]
            device_pred = pred_all[device_mask]
            device_acc = (device_y == device_pred).mean()
            device_name = unique_labels['device'][device_id]
            device_accuracies[device_name] = device_acc
            print(f"{device_name}: {device_acc:.4f} ({device_mask.sum()} samples)")
            
        overall_acc = (y_all == pred_all).mean()
        print(f"Overall accuracy: {overall_acc:.4f}")
            
        for device, acc in device_accuracies.items():
            tensorboard.add_scalar(f'test_acc_per_device/{device}', acc)
        tensorboard.add_scalar('test_acc_overall', overall_acc)
