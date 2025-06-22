import copy
import os
import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import BasePredictionWriter
from neural_compressor.config import PostTrainingQuantConfig, TuningCriterion, AccuracyCriterion
from neural_compressor.quantization import fit
import torchinfo
from model.shared import ResNorm
import torch.ao.quantization.observer as obs
from torch.ao.quantization.fake_quantize import FakeQuantize
from torch.ao.quantization.observer import ObserverBase
from neural_compressor.torch.quantization import load

class OverrideEpochStepCallback(pl.callbacks.Callback):
    """
    Override the step axis in Tensorboard with epoch. Just ignore the warning message popped out.
    """
    def __init__(self) -> None:
        super().__init__()

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        self._log_step_as_current_epoch(trainer, pl_module)

    def on_test_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        self._log_step_as_current_epoch(trainer, pl_module)

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        self._log_step_as_current_epoch(trainer, pl_module)

    def _log_step_as_current_epoch(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        pl_module.log("step", trainer.current_epoch)


class FreezeEncoderFinetuneClassifier(pl.callbacks.Callback):
    """
    Freeze the encoder of a model while fine-tuning the classifier.
    """
    def __init__(self) -> None:
        super().__init__()

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        for layer in pl_module.backbone.classifier.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        # Freeze all parameters
        for param in pl_module.backbone.parameters():
            param.requires_grad = False
        pl_module.backbone.eval()
        # Unfreeze the parameters of the classifier
        for param in pl_module.backbone.classifier.parameters():
            param.requires_grad = True
        pl_module.backbone.classifier.train()


class PredictionWriter(BasePredictionWriter):
    """
    Write the predictions of a pretrained model into a pt file.
    """
    def __init__(self, output_dir, predict_subset, write_interval="epoch"):
        super().__init__(write_interval)
        self.output_dir = output_dir
        self.predict_subset = predict_subset

    def write_on_batch_end(
        self, trainer, pl_module, prediction, batch_indices, batch, batch_idx, dataloader_idx
    ):
        pass

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        predictions = torch.cat(predictions, dim=0)
        torch.save(predictions, os.path.join(self.output_dir, f"predictions_{self.predict_subset}.pt"))