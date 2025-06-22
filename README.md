# DCASE 2025 Task 1 - Acoustic Scene Classification

This repository contains Ziyang Zhou's submission for the DCASE 2025 Task 1 challenge on acoustic scene classification.

## Installation

First, install all required dependencies:

```bash
pip install -r requirements.txt
```

## Usage
### 1. Data Preparation
cd Zhou_XJTLU_task1
Adjust the meta and audio directories in the tfsepnet_multi_device_evaluation.yaml configuration file to point to your data locations if you want to run inference using the TF-SepNet multi-device evaluation model with yaml:

data:
  class_path: data.data_module.DCASEDataModule
  init_args:
    meta_dir: ../TAU-urban-acoustic-scenes-2025-mobile-evaluation/evaluation_setup
    audio_dir: ../TAU-urban-acoustic-scenes-2025-mobile-evaluation
run python main.py test --config tfsepnet_kd_test.yaml

or just put the data in the same place for the following test.

### 2. Calculate Complexity
To calculate the MACs and memory usage of the quantized model:

```bash
python test_complexity.py --submission_name Zhou_XJTLU_task1 --submission_index 1 --dummy_file Zhou_XJTLU_task1/resources/dummy.wav 
```
or
```bash
python test_complexity_print.py
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
