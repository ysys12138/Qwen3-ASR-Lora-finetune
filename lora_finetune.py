# coding=utf-8
# ... (保留原有的 License 和注释) ...

import argparse
import os
import re
import shutil
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from qwen3_asr_sft import patch_outer_forward, make_preprocess_fn_prefix_only, \
    copy_required_hf_files_for_qwen_asr, DataCollatorForQwen3ASRFinetuning, CastFloatInputsTrainer, \
    find_latest_checkpoint

import librosa
import torch
from datasets import load_dataset
from qwen_asr import Qwen3ASRModel
from transformers import (GenerationConfig, Trainer, TrainerCallback,
                          TrainingArguments)

# --- [修改点 1] 导入 PEFT 库 ---
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType

# ... (中间的 patch_outer_forward, find_latest_checkpoint, load_audio 等函数保持不变) ...

def parse_args():
    p = argparse.ArgumentParser("Qwen3-ASR Finetuning with LoRA")

    # Paths
    p.add_argument("--model_path", type=str, default="Qwen/Qwen3-ASR-1.7B")
    p.add_argument("--train_file", type=str, default="train.jsonl")
    p.add_argument("--eval_file", type=str, default="")
    p.add_argument("--output_dir", type=str, default="./qwen3-asr-finetuning-out")

    # Audio
    p.add_argument("--sr", type=int, default=16000)

    # Train hyper-params
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--grad_acc", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--epochs", type=float, default=1) # LoRA 可以跑更多 epoch
    p.add_argument("--log_steps", type=int, default=10)
    p.add_argument("--lr_scheduler_type", type=str, default="linear")
    p.add_argument("--warmup_ratio", type=float, default=0.02)

    # --- [修改点 2] 添加 LoRA 专用参数 ---
    p.add_argument("--use_lora", action="store_true", help="Enable LoRA fine-tuning")
    p.add_argument("--lora_rank", type=int, default=8, help="LoRA rank (default: 8)")
    p.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha (default: 16)")
    p.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
    p.add_argument("--target_modules", type=str, default="all-linear", 
                   help="Target modules for LoRA, e.g., 'q_proj,v_proj' or 'all-linear'")

    # DataLoader
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--pin_memory", type=int, default=1)
    p.add_argument("--persistent_workers", type=int, default=1)
    p.add_argument("--prefetch_factor", type=int, default=2)

    # Save
    p.add_argument("--save_strategy", type=str, default="steps")
    p.add_argument("--save_steps", type=int, default=200)
    p.add_argument("--save_total_limit", type=int, default=50)

    # Resume
    p.add_argument("--resume_from", type=str, default="")
    p.add_argument("--resume", type=int, default=0)

    return p.parse_args()

# ... (make_preprocess_fn_prefix_only, DataCollator, CastFloatInputsTrainer, copy_required_hf_files... 保持不变) ...

class MakeEveryCheckpointInferableCallback(TrainerCallback):
    def __init__(self, base_model_path: str, is_lora: bool = False):
        self.base_model_path = base_model_path
        self.is_lora = is_lora # 标记是否是 LoRA

    def on_save(self, args: TrainingArguments, state, control, **kwargs):
        # 如果不是主进程，直接返回
        if args.process_index != 0:
            return control

        # 构建 checkpoint 路径
        ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        if not os.path.isdir(ckpt_dir):
            ckpt_dir = kwargs.get("checkpoint", ckpt_dir)

        # 如果目录还不存在，直接返回
        if not os.path.exists(ckpt_dir):
            return control

        # --- 修复开始：严格对齐缩进 ---
        if self.is_lora:
            # LoRA 模式：只复制必要的配置文件，不复制大模型权重
            required = [
                "config.json", 
                "generation_config.json", 
                "preprocessor_config.json",
                "processor_config.json", 
                "tokenizer_config.json", 
                "tokenizer.json",
                "special_tokens_map.json", 
                "chat_template.json", 
                "merges.txt", 
                "vocab.json"
            ]
            
            for fn in required:
                src = os.path.join(self.base_model_path, fn)
                dst = os.path.join(ckpt_dir, fn)
                # 只有当源文件存在且目标文件不存在时才复制
                if os.path.exists(src) and not os.path.exists(dst):
                    shutil.copy2(src, dst)
        else:
            # 全量微调模式：复制所有必要文件
            copy_required_hf_files_for_qwen_asr(self.base_model_path, ckpt_dir)
            
        return control

def main():
    args_cli = parse_args()

    if not args_cli.train_file:
        raise ValueError("TRAIN_FILE is required")

    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
    dtype = torch.bfloat16 if use_bf16 else torch.float16
    
    print(f"Loading model from {args_cli.model_path} with dtype {dtype}...")
    asr_wrapper = Qwen3ASRModel.from_pretrained(
        args_cli.model_path,
        dtype=dtype,
        device_map=None, # LoRA 通常先加载到 CPU 或自动分配，这里保持原样
    )
    model = asr_wrapper.model
    processor = asr_wrapper.processor

    patch_outer_forward(model)
        # --- [修复] 手动实现 get_input_embeddings 以兼容 PEFT 保存 ---
    def _get_input_embeddings(self):
        # Qwen3-ASR 的架构通常是 thinker (LLM) + encoder (Audio)
        # 输入 embeddings 通常位于 thinker 模块中
        if hasattr(self, 'thinker') and hasattr(self.thinker, 'get_input_embeddings'):
            return self.thinker.get_input_embeddings()
        elif hasattr(self, 'thinker') and hasattr(self.thinker, 'model'):
             # 某些封装可能在 thinker.model.embed_tokens
             if hasattr(self.thinker.model, 'embed_tokens'):
                 return self.thinker.model.embed_tokens
        # 如果以上都不是，尝试直接找 embed_tokens (常见于 LLM 结构)
        if hasattr(self, 'embed_tokens'):
            return self.embed_tokens
        
        # 终极兜底：报错提示用户具体结构
        raise AttributeError("Cannot find input embeddings in Qwen3ASR model structure.")

    def _set_input_embeddings(self, value):
        if hasattr(self, 'thinker') and hasattr(self.thinker, 'set_input_embeddings'):
            return self.thinker.set_input_embeddings(value)
        if hasattr(self, 'embed_tokens'):
            self.embed_tokens = value
        else:
            raise AttributeError("Cannot set input embeddings.")

    # 将方法绑定到模型类上
    import types
    model.get_input_embeddings = types.MethodType(_get_input_embeddings, model)
    model.set_input_embeddings = types.MethodType(_set_input_embeddings, model)
    
    print("✅ Fixed `get_input_embeddings` for PEFT compatibility.")
    # ---------------------------------------------------------

    model.generation_config = GenerationConfig.from_model_config(model.config)

    # --- [修改点 3] 应用 LoRA ---
    if args_cli.use_lora:
        print("🚀 Enabling LoRA Fine-tuning...")
        print(f"   Rank: {args_cli.lora_rank}, Alpha: {args_cli.lora_alpha}, Dropout: {args_cli.lora_dropout}")
        
        # 确定目标模块
        if args_cli.target_modules == "all-linear":
            # 自动查找所有线性层 (适用于大多数 Transformer)
            target_modules = None # peft 会自动推断，或者手动指定
            # 如果自动推断失败，可以手动列出 Qwen 常见的模块名，例如：
            # target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            # 但对于 ASR 模型，可能需要查看其 thinker 结构。暂时设为 None 让 PEFT 尝试自动探测，
            # 或者根据 Qwen 架构硬编码如下（推荐）：
            target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
                "encoder_attn.q_proj", "encoder_attn.k_proj", "encoder_attn.v_proj", "encoder_attn.o_proj" # 假设有 encoder
            ]
            # 注意：如果报错说找不到模块，请打印 model 结构查看具体的层名称，然后修改上面的列表。
            # 最稳妥的方式是针对 Qwen3-ASR 的 thinker 部分。
            # 这里提供一个通用的回退策略：如果自动探测失败，用户需手动指定。
            # 为了简单，我们先尝试让 PEFT 自动找所有 linear 层：
            target_modules = "all-linear" 

        lora_config = LoraConfig(
            r=args_cli.lora_rank,
            lora_alpha=args_cli.lora_alpha,
            target_modules=target_modules,
            lora_dropout=args_cli.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM, # ASR 本质也是序列生成
            modules_to_save=None, # 如果需要微调某些特定层（如 embed_tokens），可以在这里加
        )

        # 准备模型 (如果需要 kbit 量化加载，这里需要 prepare_model_for_kbit_training，但我们是全精度加载，所以跳过)
        model = get_peft_model(model, lora_config)
        
        # 打印可训练参数比例
        model.print_trainable_parameters()
        print("✅ LoRA applied successfully.")
    else:
        print("⚠️ Running Full Fine-tuning (Not recommended for small datasets).")

    # 加载数据
    raw_ds = load_dataset(
        "json",
        data_files={
            "train": args_cli.train_file,
            **({"validation": args_cli.eval_file} if args_cli.eval_file else {}),
        },
    )
    ds = raw_ds.map(make_preprocess_fn_prefix_only(processor), num_proc=1)

    keep = {"prompt", "audio", "target", "prefix_text"}
    for split in ds.keys():
        drop = [c for c in ds[split].column_names if c not in keep]
        if drop:
            ds[split] = ds[split].remove_columns(drop)

    collator = DataCollatorForQwen3ASRFinetuning(processor=processor, sampling_rate=args_cli.sr)

    training_args = TrainingArguments(
        output_dir=args_cli.output_dir,
        per_device_train_batch_size=args_cli.batch_size,
        gradient_accumulation_steps=args_cli.grad_acc,
        learning_rate=args_cli.lr,
        num_train_epochs=args_cli.epochs,
        logging_steps=args_cli.log_steps,
        lr_scheduler_type=args_cli.lr_scheduler_type,
        warmup_ratio=args_cli.warmup_ratio,
        dataloader_num_workers=args_cli.num_workers,
        dataloader_pin_memory=(args_cli.pin_memory == 1),
        dataloader_persistent_workers=(args_cli.persistent_workers == 1),
        dataloader_prefetch_factor=args_cli.prefetch_factor if args_cli.num_workers > 0 else None,
        save_strategy=args_cli.save_strategy,
        save_steps=args_cli.save_steps,
        save_total_limit=args_cli.save_total_limit,
        save_safetensors=True,
        eval_strategy="steps",
        eval_steps=args_cli.save_steps,
        do_eval=bool(args_cli.eval_file),
        bf16=use_bf16,
        fp16=not use_bf16,
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
        report_to="none",
        # LoRA 特有的优化（可选）
        optim="adamw_torch", 
    )

    # 传入 is_lora 标志给 Callback，以便正确处理保存逻辑
    callbacks = [MakeEveryCheckpointInferableCallback(base_model_path=args_cli.model_path, is_lora=args_cli.use_lora)]

    trainer = CastFloatInputsTrainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds.get("validation", None),
        data_collator=collator,
        tokenizer=processor.tokenizer,
        callbacks=callbacks,
    )

    resume_from = (args_cli.resume_from or "").strip()
    if not resume_from and args_cli.resume == 1:
        resume_from = find_latest_checkpoint(training_args.output_dir) or ""

    if resume_from:
        if trainer.args.process_index == 0:
            print(f"[resume] resume_from_checkpoint = {resume_from}")
        trainer.train(resume_from_checkpoint=resume_from)
    else:
        trainer.train()
    
    # 训练结束后，如果是 LoRA，建议单独保存一下最终的 adapter
    if args_cli.use_lora:
        final_output = os.path.join(args_cli.output_dir, "final_lora_adapter")
        model.save_pretrained(final_output)
        print(f"💾 Final LoRA adapter saved to: {final_output}")

if __name__ == "__main__":
    main()
