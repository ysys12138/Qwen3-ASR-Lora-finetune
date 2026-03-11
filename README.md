# Qwen3-ASR LoRA Fine-tuning Project

基于 Qwen3-ASR 的语音识别 LoRA 微调实现。
Qwen3-ASR项目原地址：https://github.com/QwenLM/Qwen3-ASR

## 🚀 特性
- 中文特定领域LORA微调
- 挂载目录
- 支持修改微调参数

## 📦 安装依赖
*请参照README-Origin.md安装所有基础requirements*
- 另外需要:
```pip install peft```

## 基础使用方法
### 1.把音频文件放到agumented_data中降采样或升采样到16k
```cd finetuing 
python downsample.py
python upsample.py 
```
结果会自动保存到resampled_16k文件夹
同样方法可以把test set中的音频重采样并保存在res_testSet中
### 2.创建（或修改）train.jsonl以及eval.jsonl以及eval
请参照finetuning/README-Origin.md来创建正确的jsonl格式
将音频和对应label写入。

### 3.快速Lora训练
``` python lora_finetune.py  --model_path Qwen/Qwen3-ASR-1.7B   --train_file ./train-C.jsonl   --eval_file ./eval.jsonl   --output_dir ./qwen3-asr-lora-chinese-1.7B   --batch_size 4   --grad_acc 1   --lr 2e-4 --epochs 20 --use_lora --lora_rank 16 --lora_alpha 32 --lora_dropout 0.05 --target_modules "all-linear" --warmup_ratio 0.05 --save_steps 19 --log_steps 3
 ``` 

### 4.快速尝试Lora结果
```python quick_inf.py```
