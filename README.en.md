[![en](https://img.shields.io/badge/lang-English-red.svg)](README.en.md)
[![zh](https://img.shields.io/badge/语言-中文-blue.svg)](README.md)
# Qwen3-ASR LoRA Fine-tuning Project

A speech recognition LoRA fine-tuning implementation based on Qwen3-ASR.

Original Qwen3-ASR project: https://github.com/QwenLM/Qwen3-ASR
Please clone the main branch from this address and add this directory as a subdirectory

## 🚀 Features
- Fast LoRA fine-tuning with low GPU memory requirements
- Mountable directories
- Support for modifying fine-tuning parameters

## 📦 Installation Dependencies
*Please refer to the [Original Finetune README](README-Origin.md) to install all basic requirements*
- Additional requirements:
```pip install peft```

## Basic Usage
### 1. Place audio files in the augmented_data folder and downsample or upsample to 16k
```cd finetuning 
python downsample.py
python upsample.py 
Results will be automatically saved to the resampled_16k folder
Similarly, you can resample audio from the test set and save them in res_testSet

### 2. Dataset: Create (or modify) train.jsonl and eval.jsonl files
Please refer to the Original Finetune README to create the correct jsonl format
Write the audio files and corresponding labels.

#### 3. Quick LoRA Training
python lora_finetune.py --model_path Qwen/Qwen3-ASR-1.7B --train_file ./train-C.jsonl --eval_file ./eval.jsonl --output_dir ./qwen3-asr-lora-chinese-1.7B --batch_size 4 --grad_acc 1 --lr 2e-4 --epochs 20 --use_lora --lora_rank 16 --lora_alpha 32 --lora_dropout 0.05 --target_modules "all-linear" --warmup_ratio 0.05 --save_steps 19 --log_steps 3

