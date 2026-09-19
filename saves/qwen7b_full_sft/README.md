---
library_name: transformers
license: other
base_model: /mnt/workspace/models/models/qwen--Qwen2.5-7B-Instruct/snapshots/master
tags:
- llama-factory
- full
- generated_from_trainer
model-index:
- name: qwen7b_full_sft
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# qwen7b_full_sft

This model is a fine-tuned version of [/mnt/workspace/models/models/qwen--Qwen2.5-7B-Instruct/snapshots/master](https://huggingface.co//mnt/workspace/models/models/qwen--Qwen2.5-7B-Instruct/snapshots/master) on the sft_train dataset.

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 6e-06
- train_batch_size: 4
- eval_batch_size: 8
- seed: 42
- gradient_accumulation_steps: 8
- total_train_batch_size: 32
- optimizer: Use OptimizerNames.ADAMW_TORCH_FUSED with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: cosine
- lr_scheduler_warmup_ratio: 0.05
- num_epochs: 3.0

### Training results



### Framework versions

- Transformers 4.57.1
- Pytorch 2.12.0+git6bbd260
- Datasets 4.0.0
- Tokenizers 0.22.2
