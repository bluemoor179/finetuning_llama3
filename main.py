import os
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'

# import torch

# print(torch.cuda.get_device_capability()[0])

### Big Model 을 작은 GPU 에 넣기
### 8Bit 로 Model 로딩해오기

#모델 불러오기
from transformers import AutoTokenizer, AutoModelForCausalLM

llama = "./meta-llama/Meta-Llama-3.1-8B"

tokenizer = AutoTokenizer.from_pretrained(llama, local_files_only=True)

model = AutoModelForCausalLM.from_pretrained(llama)

print("load")