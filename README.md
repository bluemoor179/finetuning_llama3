```
CUDA 12.1
python -m pip install --upgrade pip
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.44.2 "numpy<2.0" accelerate==0.33.0 bitsandbytes==0.43.3
pip install peft==0.12.0 trl==0.9.6 tensorboardX==2.6.2.2
pip install packaging ninja wheel
pip install .\flash_attn-2.6.3+cu123torch2.3.1cxx11abiFALSE-cp311-cp311-win_amd64.whl
```
# https://github.com/bdashore3/flash-attention/releases
### setting
<!-- - pip3 install torch torchvision torchaudio -->
<!-- - pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 -->
<!-- - pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 -->
- python -m pip install --upgrade pip
<!-- - pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 -->
<!-- - pip3 install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124 -->
<!-- - pip3 install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121 -->

- 2.3.1
```
# ROCM 6.0 (Linux only)
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/rocm6.0
# CUDA 11.8
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu118
# CUDA 12.1
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
# CPU only
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cpu
```

- 2.4.0
```
# ROCM 6.1 (Linux only)
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/rocm6.1
# CUDA 11.8
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu118
# CUDA 12.1
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121
# CUDA 12.4
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124
# CPU only
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cpu
```

- pip install transformers==4.44.2 "numpy<2.0" accelerate==0.33.0 bitsandbytes==0.43.3
<!-- - pip install transformers -->
- pip install transformers==4.44.2
<!-- - pip install transformers==4.33.0  -->
- pip install "numpy<2.0"
- pip install accelerate==0.33.0
- pip install bitsandbytes==0.43.3
<!-- - pip install bitsandbytes==0.43.2 -->
<!-- - pip install bitsandbytes -->

<!-- - pip install certifi==2024.7.4
- pip install yarl==1.9.4 -->

- pip install packaging ninja wheel
- pip install packaging
- pip install ninja
- pip install wheel
<!-- - pip install flash-attn --no-build-isolation -->
<!-- - pip install flash-attn==2.3.2 --no-build-isolation -->
<!-- - pip install flash_attn-2.3.2+cu122-cp311-cp311-win_amd64.whl -->
- pip install .\flash_attn-2.6.3+cu123torch2.3.1cxx11abiFALSE-cp311-cp311-win_amd64.whl
<!-- - pip install .\flash_attn-2.6.3+cu123torch2.4.0cxx11abiFALSE-cp311-cp311-win_amd64.whl -->


```
Microsoft Visual Studio 2022
<!-- PS C:\Projects\finetuning_llama3> cd .\flash_attn_wheels-1.0.9\
PS C:\Projects\finetuning_llama3\flash_attn_wheels-1.0.9> python setup.py install -->
```

- pip install peft==0.12.0 trl==0.9.6 tensorboardX==2.6.2.2
<!-- - pip install peft -->
- pip install peft==0.12.0
<!-- - pip install trl -->
- pip install trl==0.9.6
- pip install tensorboardX==2.6.2.2

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


<!-- ```
Package            Version
------------------ ------------
accelerate         0.33.0
aiohappyeyeballs   2.4.0
aiohttp            3.10.5
aiosignal          1.3.1
attrs              24.2.0
bitsandbytes       0.43.3
certifi            2024.7.4
charset-normalizer 3.3.2
colorama           0.4.6
datasets           2.21.0
dill               0.3.8
docstring_parser   0.16
einops             0.8.0
filelock           3.13.1
flash_attn         2.6.3
frozenlist         1.4.1
fsspec             2024.2.0
huggingface-hub    0.24.6
idna               3.8
Jinja2             3.1.3
markdown-it-py     3.0.0
MarkupSafe         2.1.5
mdurl              0.1.2
mpmath             1.3.0
multidict          6.0.5
multiprocess       0.70.16
networkx           3.2.1
ninja              1.11.1.1
numpy              1.26.3
packaging          24.1
pandas             2.2.2
peft               0.12.0
pillow             10.2.0
pip                24.2
protobuf           5.28.0
psutil             6.0.0
pyarrow            17.0.0
Pygments           2.18.0
python-dateutil    2.9.0.post0
pytz               2024.1
PyYAML             6.0.2
regex              2024.7.24
requests           2.32.3
rich               13.8.0
safetensors        0.4.4
setuptools         65.5.0
shtab              1.7.1
six                1.16.0
sympy              1.12
tensorboardX       2.6.2.2
tokenizers         0.19.1
torch              2.4.0+cu121
torchaudio         2.4.0+cu121
torchvision        0.19.0+cu121
tqdm               4.66.5
transformers       4.44.2
trl                0.9.6
typing_extensions  4.9.0
tyro               0.8.10
tzdata             2024.1
urllib3            2.2.2
wheel              0.44.0
xxhash             3.5.0
yarl               1.9.4
``` -->

```
Package            Version
------------------ ------------
accelerate         0.33.0
aiohappyeyeballs   2.4.0
aiohttp            3.10.5
aiosignal          1.3.1
attrs              24.2.0
bitsandbytes       0.43.3
certifi            2024.8.30 #
charset-normalizer 3.3.2
colorama           0.4.6
datasets           2.21.0
dill               0.3.8
docstring_parser   0.16
einops             0.8.0
filelock           3.13.1
flash_attn         2.6.3
frozenlist         1.4.1
fsspec             2024.2.0
huggingface-hub    0.24.6
idna               3.8
Jinja2             3.1.3
markdown-it-py     3.0.0
MarkupSafe         2.1.5
mdurl              0.1.2
mpmath             1.3.0
multidict          6.0.5
multiprocess       0.70.16
networkx           3.2.1
ninja              1.11.1.1
numpy              1.26.3
packaging          24.1
pandas             2.2.2
peft               0.12.0
pillow             10.2.0
pip                24.2
protobuf           5.28.0
psutil             6.0.0
pyarrow            17.0.0
Pygments           2.18.0
python-dateutil    2.9.0.post0
pytz               2024.1
PyYAML             6.0.2
regex              2024.7.24
requests           2.32.3
rich               13.8.0
safetensors        0.4.4
setuptools         65.5.0
shtab              1.7.1
six                1.16.0
sympy              1.12
tensorboardX       2.6.2.2
tokenizers         0.19.1
torch              2.4.0+cu124
torchaudio         2.4.0+cu124
torchvision        0.19.0+cu124
tqdm               4.66.5
transformers       4.44.2
trl                0.10.1 #
typing_extensions  4.9.0
tyro               0.8.10
tzdata             2024.1
urllib3            2.2.2
wheel              0.44.0
xxhash             3.5.0
yarl               1.9.6 #
```



```
Package            Version
------------------ ------------
accelerate         0.34.0
aiohappyeyeballs   2.4.0
aiohttp            3.10.5
aiosignal          1.3.1
attrs              24.2.0
bitsandbytes       0.43.3
certifi            2024.7.4
charset-normalizer 3.3.2
colorama           0.4.6
datasets           2.21.0
dill               0.3.8
docstring_parser   0.16
einops             0.8.0
filelock           3.13.1
flash_attn         2.6.3
frozenlist         1.4.1
fsspec             2024.2.0
huggingface-hub    0.24.6
idna               3.8
Jinja2             3.1.3
markdown-it-py     3.0.0
MarkupSafe         2.1.5
mdurl              0.1.2
mpmath             1.3.0
multidict          6.0.5
multiprocess       0.70.16
networkx           3.2.1
numpy              1.26.3
packaging          24.1
pandas             2.2.2
peft               0.12.0
pillow             10.2.0
pip                24.2
protobuf           5.28.0
psutil             6.0.0
pyarrow            17.0.0
Pygments           2.18.0
python-dateutil    2.9.0.post0
pytz               2024.1
PyYAML             6.0.2
regex              2024.7.24
requests           2.32.3
rich               13.8.0
safetensors        0.4.4
setuptools         65.5.0
shtab              1.7.1
six                1.16.0
sympy              1.12
tensorboardX       2.6.2.2
tokenizers         0.19.1
torch              2.4.0+cu124
torchaudio         2.4.0+cu124
torchvision        0.19.0+cu124
tqdm               4.66.5
transformers       4.44.2
trl                0.9.6
typing_extensions  4.9.0
tyro               0.8.10
tzdata             2024.1
urllib3            2.2.2
xxhash             3.5.0
yarl               1.9.4
```

```
Package            Version
------------------ ------------
accelerate         0.33.0
aiohappyeyeballs   2.4.0
aiohttp            3.10.5
aiosignal          1.3.1
attrs              24.2.0
bitsandbytes       0.43.3
certifi            2024.8.30
charset-normalizer 3.3.2
colorama           0.4.6
datasets           2.21.0
dill               0.3.8
docstring_parser   0.16
einops             0.8.0
filelock           3.13.1
flash_attn         2.6.3
frozenlist         1.4.1
fsspec             2024.2.0
huggingface-hub    0.24.6
idna               3.8
intel-openmp       2021.4.0
Jinja2             3.1.3
markdown-it-py     3.0.0
MarkupSafe         2.1.5
mdurl              0.1.2
mkl                2021.4.0
mpmath             1.3.0
multidict          6.0.5
multiprocess       0.70.16
networkx           3.2.1
numpy              1.26.3
packaging          24.1
pandas             2.2.2
peft               0.12.0
pillow             10.2.0
pip                24.2
protobuf           5.28.0
psutil             6.0.0
pyarrow            17.0.0
Pygments           2.18.0
python-dateutil    2.9.0.post0
pytz               2024.1
PyYAML             6.0.2
regex              2024.7.24
requests           2.32.3
rich               13.8.0
safetensors        0.4.5
setuptools         65.5.0
shtab              1.7.1
six                1.16.0
sympy              1.12
tbb                2021.11.0
tensorboardX       2.6.2.2
tokenizers         0.19.1
torch              2.3.1+cu121
torchaudio         2.3.1+cu121
torchvision        0.18.1+cu121
tqdm               4.66.5
transformers       4.44.2
trl                0.9.6
typing_extensions  4.9.0
tyro               0.8.10
tzdata             2024.1
urllib3            2.2.2
xxhash             3.5.0
yarl               1.10.0
```