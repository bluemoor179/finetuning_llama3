### setting
<!-- - pip3 install torch torchvision torchaudio -->
- pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

- pip install transformers==4.44.2 <!-- 4.44.2 -->
- pip install "numpy<2.0" 

<!-- - pip install -U accelerate==0.29.3 peft==0.10.0 bitsandbytes==0.43.1 transformers==4.40.1 trl==0.8.6 datasets==2.19.0 -->
<!-- - pip install -qqq flash-attn -->
<!-- - pip install datasets==2.16.1
- pip install transformers==4.36.2
- pip install bitsandbytes==0.42.0
- pip install peft==0.7.1
- pip install accelerate==0.26.1
- pip install einops -->

### check GPU power

- torch.cuda.get_device_capability()[0] >= 8


```
Package            Version
------------------ ------------
certifi            2024.7.4
charset-normalizer 3.3.2
colorama           0.4.6
filelock           3.15.4
fsspec             2024.6.1
huggingface-hub    0.24.6
idna               3.8
Jinja2             3.1.3
MarkupSafe         2.1.5
mpmath             1.3.0
networkx           3.2.1
numpy              1.26.4
packaging          24.1
pillow             10.2.0
pip                24.2
PyYAML             6.0.2
regex              2024.7.24
requests           2.32.3
safetensors        0.4.4
setuptools         65.5.0
sympy              1.12
tokenizers         0.19.1
torch              2.4.0+cu124
torchaudio         2.4.0+cu124
torchvision        0.19.0+cu124
tqdm               4.66.5
transformers       4.44.2
typing_extensions  4.12.2
urllib3            2.2.2
```