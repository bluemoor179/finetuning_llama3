import os
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'

import torch
print(torch.cuda.get_device_name())
print(torch.cuda.get_device_capability())
print(torch.cuda.is_available())


# if torch.cuda.get_device_capability()[0] >= 8:
#     !pip install -qqq flash-attn
#     attn_implementation = "flash_attention_2"
#     torch_dtype = torch.bfloat16
# else:
#     attn_implementation = "eager"
#     torch_dtype = torch.float16

### Big Model 을 작은 GPU 에 넣기
### 8Bit 로 Model 로딩해오기

#모델 불러오기
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments

llama = "./meta-llama/Meta-Llama-3.1-8B"

tokenizer = AutoTokenizer.from_pretrained(llama, local_files_only=True)

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=False,
)

model = AutoModelForCausalLM.from_pretrained(
    llama
    , quantization_config=quant_config
    , torch_dtype=torch.bfloat16
    , attn_implementation="flash_attention_2"
    , local_files_only=True)
model.config.use_cache = False
model.config.pretraining_tp = 1

print("load")

# Tokenizer load
tokenizer = AutoTokenizer.from_pretrained(
              llama)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# PEFT
from peft import LoraConfig
peft_params = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
)

# 학습 모델 설정
training_params = TrainingArguments(
    output_dir="./results",
    num_train_epochs=10,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=1,
    optim="paged_adamw_32bit",
    save_steps=25,
    logging_steps=25,
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=False,
    bf16=False,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="constant",
    report_to="tensorboard"
)

#Dataset load
from datasets import load_dataset
hkcode_dataset = "uiyong/gemini_result_kospi_0517_jsonl"
dataset = load_dataset(hkcode_dataset, split="train")

print('dataset load')

# 모델 학습
from trl import SFTTrainer, SFTConfig

sft_config = SFTConfig(
    output_dir="./results",
    num_train_epochs=10,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=1,
    optim="paged_adamw_32bit",
    save_steps=25,
    logging_steps=25,
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=False,
    bf16=False,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="constant",
    report_to="tensorboard",
    dataset_text_field="text",
    max_seq_length=1024, # None
    packing=False,
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset.select([0, 10, 20, 30, 40, 50]),
    args=sft_config,
    peft_config=peft_params,
    tokenizer=tokenizer,
)

print("train start")
trainer.train()
print("train end")