#data/data_module.py
import librosa
import numpy as np
import torch
import lightning as L
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from util import unique_labels
from typing import Optional, List

class AudioDataset(Dataset):
    """
    Dataset containing pairs of audio waveform and filename.

    Args:
        meta_dir (str): Directory of meta files, which should include meta files in csv formate.
        audio_dir (str): Directory of audios.
        subset (str): Name of required meta file. e.g. ``train``, ``valid``, ``test``...
        sampling_rate (int): Sampling rate of waveforms.
    """
    def __init__(self, meta_dir: str, audio_dir: str, subset: str, sampling_rate: int = 16000):
        self.meta_dir = meta_dir
        self.audio_dir = audio_dir
        self.subset = subset
        self.sr = sampling_rate
        self.meta_subset = pd.read_csv(f"{self.meta_dir}/{self.subset}.csv", sep='\t')

    def __len__(self):
        return len(self.meta_subset)

    def __getitem__(self, i):
        # Get the ith row from the meta csv file
        row_i = self.meta_subset.iloc[i]
        # Get the filename of audio
        filename = row_i["filename"]
        # Load audio waveform with a resample rate
        wav, _ = librosa.load(f"{self.audio_dir}/{filename}", sr=self.sr)
        wav = torch.from_numpy(wav)
        return wav, filename


class AudioLabelsDataset(AudioDataset):
    """
    Dataset containing tuples of audio waveform, scene label, device label and city label.

    Args:
        meta_dir (str): Directory of meta files, which should include meta files in csv formate.
        audio_dir (str): Directory of audios.
        subset (str): Name of required meta file. e.g. ``train``, ``valid``, ``test``...
        sampling_rate (int): Sampling rate of waveforms.
    """
    def __init__(self, meta_dir: str, audio_dir: str, subset: str, sampling_rate: int = 16000):
        super().__init__(meta_dir, audio_dir, subset, sampling_rate)

    def __getitem__(self, i):
        # Get the filename
        wav, filename = super().__getitem__(i)
        scene_label = filename.split('/')[-1].split('-')[0]
        device_label = filename.split('-')[-1].split('.')[0]
        city_label = filename.split('-')[1]
        # Encode the scene labels from string to integers
        scene_label = unique_labels['scene'].index(scene_label)
        scene_label = torch.from_numpy(np.array(scene_label, dtype=np.int64))
        # Encode the device labels from string to integers
        device_label = unique_labels['device'].index(device_label)
        device_label = torch.from_numpy(np.array(device_label, dtype=np.int64))
        # Encode the city labels from string to integers
        city_label = unique_labels['city'].index(city_label)
        city_label = torch.from_numpy(np.array(city_label, dtype=np.int64))
        return wav, scene_label, device_label, city_label


class AudioLabelsDatasetWithLogits(AudioLabelsDataset):
    """
    AudioLabelsDataset with additional logits of teacher ensemble for knowledge distillation.

    Args:
        logits_files (list): List of directories of teacher logits. e.g. ["path/to/logit/predictions.pt", ...]
    """
    def __init__(self, logits_files: list, **kwargs):
        super().__init__(**kwargs)
        logits_all = []
        for file in logits_files:
            # Load teacher logit and append to a list
            logit = torch.load(file).float()
            logits_all.append(logit)
        # Average the logits from multiple teachers
        logit_all = sum(logits_all)
        self.teacher_logit = logit_all / len(logits_files)

    def __getitem__(self, i):
        wav, scene_label, device_label, city_label = super().__getitem__(i)
        return wav, scene_label, device_label, city_label, self.teacher_logit[i]


class InferenceDataset(Dataset):
    """
    专门用于推理的数据集，处理新的CSV格式
    """
    def __init__(self, meta_dir: str, audio_dir: str, subset: str, sampling_rate: int = 16000):
        self.meta_dir = meta_dir
        self.audio_dir = audio_dir
        self.subset = subset
        self.sr = sampling_rate
        
    
        csv_path = f"{self.meta_dir}/{self.subset}.csv"
        try:
            self.meta_subset = pd.read_csv(csv_path, sep='\t')
        except:
            try:
                self.meta_subset = pd.read_csv(csv_path, sep=',')
            except:
                self.meta_subset = pd.read_csv(csv_path)
        
        print(f"Loaded {len(self.meta_subset)} samples from {csv_path}")
        print(f"CSV columns: {list(self.meta_subset.columns)}")
        
    def __len__(self):
        return len(self.meta_subset)
    
    def __getitem__(self, i):

        row_i = self.meta_subset.iloc[i]
        filename = row_i["filename"]
        device_id = row_i["device_id"]
        
        wav, _ = librosa.load(f"{self.audio_dir}/{filename}", sr=self.sr)
        wav = torch.from_numpy(wav)
        
        try:
            if device_id in unique_labels['device']:
                device_idx = unique_labels['device'].index(device_id)
            else:
                # unknown which not in the list
                device_idx = unique_labels['device'].index('unknown') if 'unknown' in unique_labels['device'] else 0
            device_label = torch.from_numpy(np.array(device_idx, dtype=np.int64))
        except Exception as e:
            print(f"Error encoding device label '{device_id}': {e}")
            device_label = torch.from_numpy(np.array(0, dtype=np.int64))
        
        return wav, filename, device_label, device_id


class AudioEvaluationDataset(AudioDataset):
    """
    For evaluate the evaluationset
    """
    def __getitem__(self, i):
        wav, filename = super().__getitem__(i)
        row = self.meta_subset.iloc[i]
        device_label = row['device_id'] if 'device_id' in row else 'unknown'
        
        try:
            if device_label in unique_labels['device']:
                device_idx = unique_labels['device'].index(device_label)
            else:
                device_idx = unique_labels['device'].index('unknown') if 'unknown' in unique_labels['device'] else 0
            device_label_tensor = torch.from_numpy(np.array(device_idx, dtype=np.int64))
        except Exception as e:
            print(f"Error encoding device label '{device_label}': {e}")
            device_label_tensor = torch.from_numpy(np.array(0, dtype=np.int64))
        
        return wav, filename, device_label_tensor


class DCASEDataModule(L.LightningDataModule):
    """
    DCASE DataModule wrapping train, validation, test and predict DataLoaders.

    Args:
        meta_dir (str): Directory of meta files, which should include meta files in csv formate.
        audio_dir (str): Directory of audios.
        batch_size (int): Batch size.
        num_workers (int): Number of workers to use for DataLoaders. Will save time for loading data to GPU but increase CPU usage.
        pin_memory (bool): If True, the data loader will copy Tensors into device/CUDA pinned memory before returning them. Will save time for data loading.
        logits_files (list): List of directories of teacher logits, e.g. ["path/to/logit/predictions.pt", ...]. If not ``None``, knowledge distillation will be applied.
        train_subset (str): Name of train meta file. e.g. train, split5, split10...
        test_subset (str): Name of test meta file.
        predict_subset (str): Name of predict meta file.
        eval_subset (str): Name of evaluation meta file.
    """
    def __init__(self, 
                 meta_dir: str, 
                 audio_dir: str, 
                 batch_size: int = 16, 
                 num_workers: int = 0, 
                 pin_memory: bool = False,
                 logits_files: Optional[List[str]] = None, 
                 train_subset: str = "train", 
                 test_subset: str = "test", 
                 predict_subset: str = "test", 
                 eval_subset: str = "test", 
                 sampling_rate: int = 16000,
                 target_devices= None,
                 **kwargs):
        super().__init__()
        self.meta_dir = meta_dir
        self.audio_dir = audio_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.train_subset = train_subset
        self.test_subset = test_subset
        self.predict_subset = predict_subset
        self.eval_subset = eval_subset
        self.logits_files = logits_files
        self.sampling_rate = sampling_rate
        self.target_devices = target_devices
        self.kwargs = kwargs

    def _filter_by_device(self, df):
        if self.target_devices:
            return df[df['filename'].str.split('-').str[-1].str.split('.').str[0].isin(self.target_devices)]
        return df
    
    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            # Add teacher logits to the dataset if using knowledge distillation
            if self.logits_files is not None:
                self.train_set = AudioLabelsDatasetWithLogits(
                    logits_files=self.logits_files, 
                    meta_dir=self.meta_dir, 
                    audio_dir=self.audio_dir, 
                    subset=self.train_subset, 
                    sampling_rate=self.sampling_rate,
                    **self.kwargs
                )
            else:
                self.train_set = AudioLabelsDataset(
                    self.meta_dir, 
                    self.audio_dir, 
                    subset=self.train_subset, 
                    sampling_rate=self.sampling_rate,
                    **self.kwargs
                )
            self.valid_set = AudioLabelsDataset(
                self.meta_dir, 
                self.audio_dir, 
                subset="valid", 
                sampling_rate=self.sampling_rate,
                **self.kwargs
            )
        if stage == "validate":
            self.valid_set = AudioLabelsDataset(
                self.meta_dir, 
                self.audio_dir, 
                subset="valid", 
                sampling_rate=self.sampling_rate,
                **self.kwargs
            )
            self.valid_set.meta_subset = self._filter_by_device(
                self.valid_set.meta_subset
            ).reset_index(drop=True)
        if stage == "test":
            self.test_set = AudioLabelsDataset(
                self.meta_dir, 
                self.audio_dir, 
                subset=self.test_subset, 
                sampling_rate=self.sampling_rate,
                **self.kwargs
            )
            self.test_set.meta_subset = (
                self._filter_by_device(self.test_set.meta_subset)
                .reset_index(drop=True)
            )
        if stage == "predict":
            csv_path = f"{self.meta_dir}/{self.predict_subset}.csv"
            try:
                try:
                    test_df = pd.read_csv(csv_path, sep='\t')
                except:
                    try:
                        test_df = pd.read_csv(csv_path, sep=',')
                    except:
                        test_df = pd.read_csv(csv_path)
                
                print(f"CSV columns: {list(test_df.columns)}")
                
                if 'device_id' in test_df.columns:
                    print("Detected inference format with device_id column")
                    self.predict_set = InferenceDataset(
                        self.meta_dir, 
                        self.audio_dir, 
                        subset=self.predict_subset, 
                        sampling_rate=self.sampling_rate
                    )
                else:
                    print("Using regular AudioDataset for prediction")
                    self.predict_set = AudioDataset(
                        self.meta_dir, 
                        self.audio_dir, 
                        subset=self.predict_subset, 
                        sampling_rate=self.sampling_rate
                    )
            except Exception as e:
                print(f"Error reading CSV: {e}")
                print("Using default AudioDataset")
                self.predict_set = AudioDataset(
                    self.meta_dir, 
                    self.audio_dir, 
                    subset=self.predict_subset, 
                    sampling_rate=self.sampling_rate
                )
        if stage == "evaluation":
            self.eval_set = AudioEvaluationDataset(
                self.meta_dir, 
                self.audio_dir, 
                subset=self.eval_subset, 
                sampling_rate=self.sampling_rate,
                **self.kwargs
            )

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True,
                          pin_memory=self.pin_memory)

    def val_dataloader(self):
        return DataLoader(self.valid_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False,
                          pin_memory=self.pin_memory)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False,
                          pin_memory=self.pin_memory)

    def predict_dataloader(self):
        return DataLoader(self.predict_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False,
                          pin_memory=self.pin_memory)
    
    def evaluation_dataloader(self):
        return DataLoader(self.eval_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False,
                          pin_memory=self.pin_memory)


class DCASEDataModuleByDevice(DCASEDataModule):
    """
    DeviceSpecific
    """
    def __init__(self, target_devices=None, exclude_devices=None, **kwargs):
        super().__init__(**kwargs)
        self.target_devices = target_devices
        self.exclude_devices = exclude_devices
        
    def _get_device_from_filename(self, filename):
        return filename.split('-')[-1].split('.')[0]
        
    def _should_include_device(self, device):
        if self.target_devices is not None:
            return device in self.target_devices
        if self.exclude_devices is not None:
            return device not in self.exclude_devices
        return True
    
    def _filter_meta_by_device(self, meta_df):
        device_mask = meta_df['filename'].apply(
            lambda x: self._should_include_device(self._get_device_from_filename(x))
        )
        filtered_meta = meta_df[device_mask].reset_index(drop=True)
        return filtered_meta
        
    def setup(self, stage: str):
        print(f"\n{'='*50}")
        print(f"Setting up data for stage: {stage}")
        print(f"Target devices: {self.target_devices}")
        print(f"{'='*50}")
        
        if stage == "fit":
            temp_train_dataset = AudioLabelsDataset(
                self.meta_dir, self.audio_dir, subset=self.train_subset, **self.kwargs
            )
            temp_valid_dataset = AudioLabelsDataset(
                self.meta_dir, self.audio_dir, subset="valid", **self.kwargs
            )
            
            original_train_meta = temp_train_dataset.meta_subset
            original_valid_meta = temp_valid_dataset.meta_subset
            
            filtered_train_meta = self._filter_meta_by_device(original_train_meta)
            filtered_valid_meta = self._filter_meta_by_device(original_valid_meta)
            
            print(f"Train data: {len(original_train_meta)} -> {len(filtered_train_meta)} samples")
            print(f"Valid data: {len(original_valid_meta)} -> {len(filtered_valid_meta)} samples")
            
            aligned_train_logits = None
            if self.logits_files:
                aligned_train_logits = self._create_filtered_logits(
                    original_train_meta, filtered_train_meta
                )
            
            self.train_set = self._create_filtered_dataset_with_logits(
                self.meta_dir, self.audio_dir, self.train_subset, aligned_train_logits
            )
            
            self.valid_set = AudioLabelsDataset(
                self.meta_dir, self.audio_dir, subset="valid", **self.kwargs
            )
            self.valid_set.meta_subset = filtered_valid_meta
            
        elif stage == "validate":
            temp_dataset = AudioLabelsDataset(
                self.meta_dir, self.audio_dir, subset="valid", **self.kwargs
            )
            filtered_meta = self._filter_meta_by_device(temp_dataset.meta_subset)
            
            self.valid_set = AudioLabelsDataset(
                self.meta_dir, self.audio_dir, subset="valid", **self.kwargs
            )
            self.valid_set.meta_subset = filtered_meta
            
        elif stage == "test":
            temp_dataset = AudioLabelsDataset(
                self.meta_dir, self.audio_dir, subset=self.test_subset, **self.kwargs
            )
            filtered_meta = self._filter_meta_by_device(temp_dataset.meta_subset)
            
            self.test_set = AudioLabelsDataset(
                self.meta_dir, self.audio_dir, subset=self.test_subset, **self.kwargs
            )
            self.test_set.meta_subset = filtered_meta
            
        elif stage == "predict":
            from data.data_module import AudioDataset
            temp_dataset = AudioDataset(
                self.meta_dir, self.audio_dir, subset=self.predict_subset, **self.kwargs
            )
            filtered_meta = self._filter_meta_by_device(temp_dataset.meta_subset)
            
            self.predict_set = AudioDataset(
                self.meta_dir, self.audio_dir, subset=self.predict_subset, **self.kwargs
            )
            self.predict_set.meta_subset = filtered_meta
            
        print(f"Setup completed for stage: {stage}")
        
    def get_device_distribution(self, stage="train"):
        if stage == "train" and hasattr(self, 'train_set'):
            meta = self.train_set.meta_subset
        elif stage == "valid" and hasattr(self, 'valid_set'):
            meta = self.valid_set.meta_subset
        elif stage == "test" and hasattr(self, 'test_set'):
            meta = self.test_set.meta_subset
        else:
            return None
            
        devices = meta['filename'].apply(self._get_device_from_filename)
        distribution = devices.value_counts().to_dict()
        
        print(f"\nDevice distribution ({stage}):")
        for device, count in distribution.items():
            print(f"  {device}: {count} samples")
        
        return distribution