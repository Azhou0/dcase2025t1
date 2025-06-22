# DCASE 2025 Task 1 - Acoustic Scene Classification

This repository contains Ziyang Zhou's submission for the DCASE 2025 Task 1 challenge on acoustic scene classification.

## Installation

First, install all required dependencies:

```bash
pip install -r requirements.txt
```

## Usage
### 1. Data Preparation
Adjust the meta and audio directories in the tfsepnet_multi_device_evaluation.yaml configuration file to point to your data locations:

data:
  class_path: data.data_module.DCASEDataModule
  init_args:
    meta_dir: ../TAU-urban-acoustic-scenes-2025-mobile-evaluation/evaluation_setup
    audio_dir: ../TAU-urban-acoustic-scenes-2025-mobile-evaluation

### 2. Calculate Complexity
To calculate the MACs and memory usage of the quantized model:

```bash
python test_complexity.py
```
### 3. Running Inference
To run inference using the TF-SepNet multi-device evaluation model(recommend):

```bash
cd Zhou_XJTLU_task1
python test.py 
```
or 
```bash
python evaluate_submission.py 
```
and follow the instructions.
