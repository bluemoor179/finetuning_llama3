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
# from datasets import load_dataset
# hkcode_dataset = "uiyong/gemini_result_kospi_0517_jsonl"
# dataset = load_dataset(hkcode_dataset, split="train")
from datasets import Dataset

def gen():
    with open('./resources/커피향 나는 열네 번째/1장/001화.txt', 'r', encoding='UTF8') as file :
        isFirst = True
        current_string = ""
        while True:
            line = file.readline()
            if not line:
                yield {"text": current_string}
                break
            if isFirst:
                isFirst = False
                yield {"text": line}
            else:
                yield {"text": current_string + line}
            line = current_string
dataset = Dataset.from_generator(gen)

print('dataset load')

# 모델 학습
from trl import SFTTrainer, SFTConfig

sft_config = SFTConfig(
    output_dir="./results",
    num_train_epochs=1, # Total number of training epochs to perform (if not an integer, will perform the decimal part percents of the last epoch before stopping training).
    per_device_train_batch_size=4, # The batch size per GPU/XPU/TPU/MPS/NPU core/CPU for training.
    gradient_accumulation_steps=1, # Number of updates steps to accumulate the gradients for, before performing a backward/update pass.
    optim="paged_adamw_32bit", #The optimizer to use: adamw_hf, adamw_torch, adamw_torch_fused, adamw_apex_fused, adamw_anyprecision or adafactor.
    save_steps=25, # Number of updates steps before two checkpoint saves if `save_strategy="steps"`. Should be an integer or a float in range `[0,1)`. If smaller than 1, will be interpreted as ratio of total training steps.
    logging_steps=25, # Number of update steps between two logs if `logging_strategy="steps"`. Should be an integer or a float in range `[0,1)`. If smaller than 1, will be interpreted as ratio of total training steps.
    learning_rate=2e-4, # The initial learning rate for [`AdamW`] optimizer.
    weight_decay=0.001, # The weight decay to apply (if not zero) to all layers except all bias and LayerNorm weights in [`AdamW`] optimizer.
    fp16=False, # Whether to use fp16 16-bit (mixed) precision training instead of 32-bit training.
    bf16=False, # Whether to use bf16 16-bit (mixed) precision training instead of 32-bit training. Requires Ampere or higher NVIDIA architecture or using CPU (use_cpu) or Ascend NPU. This is an experimental API and it may change.
    max_grad_norm=0.3, # Maximum gradient norm (for gradient clipping).
    max_steps=-1, # If set to a positive number, the total number of training steps to perform. Overrides `num_train_epochs`. For a finite dataset, training is reiterated through the dataset (if all data is exhausted) until `max_steps` is reached.
    warmup_ratio=0.03, # Ratio of total training steps used for a linear warmup from 0 to `learning_rate`.
    group_by_length=True, # Whether or not to group together samples of roughly the same length in the training dataset (to minimize padding applied and be more efficient). Only useful if applying dynamic padding.
    lr_scheduler_type="constant", # The scheduler type to use. See the documentation of [`SchedulerType`] for all possible values.
    report_to="tensorboard", # The list of integrations to report the results and logs to. Supported platforms are `"azure_ml"`, `"clearml"`, `"codecarbon"`, `"comet_ml"`, `"dagshub"`, `"dvclive"`, `"flyte"`, `"mlflow"`, `"neptune"`, `"tensorboard"`, and `"wandb"`. Use `"all"` to report to all integrations installed, `"none"` for no integrations.
    dataset_text_field="text", # The name of the text field of the dataset, in case this is passed by a user, the trainer will automatically create a `ConstantLengthDataset` based on the `dataset_text_field` argument. Defaults to None.
    max_seq_length=2048, # The maximum sequence length to use for the `ConstantLengthDataset` and for automatically creating the Dataset. Defaults to min of the smaller of the `tokenizer.model_max_length` and `1024`.
    packing=False, # Used only in case `dataset_text_field` is passed. This argument is used by the `ConstantLengthDataset` to pack the sequences of the dataset. Defaults to False.
)

trainer = SFTTrainer(
    model=model,
    # train_dataset=dataset.select([0, 10, 20, 30, 40, 50]),
    train_dataset=dataset,
    args=sft_config,
    peft_config=peft_params,
    tokenizer=tokenizer,
)

# print(dataset.select([0, 10, 20, 30, 40, 50])[0])
# print(dataset.select([0, 10, 20, 30, 40, 50])['text'])
print("train start")
trainer.train()
print("train end")